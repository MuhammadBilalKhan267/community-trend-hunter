import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib import error

from trend_hunter.decisions import (
    BudgetExceeded,
    DecisionError,
    JevDecisions,
    CACHE_VERSION,
    _HTTPFailure,
    _context_error_code,
    _digest,
    choice_question,
    load_api_key,
    noul_question,
)


QUESTIONS = {
    "supported": noul_question("Is the interpretation supported?"),
    "role": choice_question("What is the contribution's role?", {"report": "An independent experience", "unclear": "Insufficient context"}),
}


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.context_modes = []
        self.release = "2026-09-15"
        self.responses = []

    def __call__(self, method, path, payload, *, long_context=False):
        self.calls.append((method, path, payload))
        self.context_modes.append((method, path, long_context))
        if path == "/v1/models":
            return {"models": [{"name": "jev-latest", "release_date": self.release}]}
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        answers = {}
        for name, question in payload["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.9}
            else:
                choices = list(question["criteria"])
                answers[name] = {"type": "choice", "choice": choices[0], "confidence": 0.9,
                                 "probabilities": {choice: (1.0 if choice == choices[0] else 0.0) for choice in choices}}
        return {"model": "jev-release-1", "answers": answers, "usage": {"input_tokens": 123, "output_tokens": 17}}

    @property
    def posts(self):
        return [call for call in self.calls if call[0] == "POST"]


class JevDecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = Path(self.temp.name) / "jev.sqlite3"
        self.client = JevDecisions(self.cache, api_key="unit-test-not-real", retries=0)
        self.transport = FakeTransport()
        self.client._transport = self.transport

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_mixed_questions_share_one_request_and_report_usage(self):
        answers = self.client.ask({"examples": ["原文", "另一条"]}, QUESTIONS)
        self.assertEqual(answers["supported"]["noul"], 0.9)
        self.assertEqual(answers["role"]["choice"], "report")
        self.assertEqual(len(self.transport.posts), 1)
        self.assertEqual(self.client.resolved_model, "jev-release-1")
        self.assertEqual(self.client.usage["input_tokens"], 123)
        self.assertEqual(self.client.usage["output_tokens"], 17)
        self.assertEqual(self.client.usage["reserved_tokens"], 0)

    def test_cache_is_exact_and_dictionary_order_is_irrelevant(self):
        self.client.ask({"a": 1, "b": 2}, QUESTIONS)
        self.client.ask({"b": 2, "a": 1}, dict(reversed(list(QUESTIONS.items()))))
        self.assertEqual(len(self.transport.posts), 1)
        self.assertEqual(self.client.stats["cache_hits"], 1)
        self.client.ask({"a": 1, "b": 3}, QUESTIONS)
        altered = {"supported": noul_question("A different question")}
        self.client.ask({"a": 1, "b": 2}, altered)
        self.assertEqual(len(self.transport.posts), 3)

    def test_cache_and_actual_usage_persist_across_instances(self):
        self.client.ask("text", QUESTIONS)
        with JevDecisions(self.cache, api_key="unit-test-not-real", retries=0) as other:
            other._transport = self.transport
            other.ask("text", QUESTIONS)
            self.assertEqual(other.stats["cache_hits"], 1)
            self.assertEqual(other.usage["total_tokens"], 140)
        self.assertEqual(len(self.transport.posts), 1)

    def test_updated_model_release_invalidates_alias_cache(self):
        self.client.ask("text", QUESTIONS)
        self.transport.release = "2026-10-01"
        with JevDecisions(self.cache, api_key="unit-test-not-real", retries=0) as other:
            other._transport = self.transport
            other.ask("text", QUESTIONS)
        self.assertEqual(len(self.transport.posts), 2)

    def test_rejects_unaffordable_request_before_post(self):
        self.client.max_tokens = 1000
        with self.assertRaises(BudgetExceeded):
            self.client.ask("text", QUESTIONS)
        self.assertEqual(self.transport.posts, [])
        self.assertEqual(self.client.usage["attempts"], 0)

    def test_unknown_cost_failure_reserves_usage_across_restarts(self):
        self.transport.responses = [DecisionError("Connection failed")]
        with self.assertRaises(DecisionError):
            self.client.ask("text", QUESTIONS)
        reserved = self.client.usage["reserved_tokens"]
        self.assertGreater(reserved, 0)
        with JevDecisions(self.cache, api_key="unit-test-not-real", retries=0) as other:
            self.assertEqual(other.usage["reserved_tokens"], reserved)

    def test_unknown_cost_cannot_evade_budget_on_retry(self):
        self.client.retries = 1
        payload = {"model": "jev-latest", "state": "text", "questions": QUESTIONS}
        self.client.max_tokens = self.client._reservation(payload)
        self.transport.responses = [DecisionError("Connection failed")]
        with patch("trend_hunter.decisions.time.sleep"):
            with self.assertRaises(BudgetExceeded):
                self.client.ask("text", QUESTIONS)
        self.assertEqual(len(self.transport.posts), 1)

    def test_known_rejection_releases_reservation_and_bounded_retry_works(self):
        self.client.retries = 1
        self.transport.responses = [_HTTPFailure(429)]
        with patch("trend_hunter.decisions.time.sleep"):
            self.client.ask("text", QUESTIONS)
        self.assertEqual(len(self.transport.posts), 2)
        self.assertEqual(self.client.stats["retries"], 1)
        self.assertEqual(self.client.usage["reserved_tokens"], 0)
        self.assertEqual(self.client.usage["total_tokens"], 140)

    def test_nonstandard_529_retries_once_and_preserves_first_attempt_unknown_cost(self):
        self.client.retries = 1
        payload = {"model": "jev-latest", "state": "text", "questions": QUESTIONS}
        reservation = self.client._reservation(payload)
        self.transport.responses = [_HTTPFailure(529)]
        with patch("trend_hunter.decisions.time.sleep") as sleep:
            answers = self.client.ask("text", QUESTIONS)
        self.assertEqual(answers["supported"]["noul"], .9)
        self.assertEqual(len(self.transport.posts), 2)
        self.assertEqual(self.client.stats["model_calls"], 2)
        self.assertEqual(self.client.stats["retries"], 1)
        self.assertEqual(self.client.stats["failures"], 0)
        self.assertEqual(self.client.usage["attempts"], 2)
        self.assertEqual(self.client.usage["reserved_tokens"], reservation)
        self.assertEqual(self.client.usage["total_tokens"], 140)
        self.assertEqual(self.client.usage["accounted_tokens"], reservation + 140)
        first_attempt = self.client._db.execute(
            "SELECT status,reserved_tokens,input_tokens,output_tokens FROM jev_usage_requests ORDER BY id LIMIT 1"
        ).fetchone()
        self.assertEqual(tuple(first_attempt), ("failed", reservation, None, None))
        sleep.assert_called_once_with(1)

    def test_authentication_rejection_is_not_retried(self):
        self.client.retries = 3
        self.transport.responses = [_HTTPFailure(401)]
        with self.assertRaisesRegex(DecisionError, "HTTP 401"):
            self.client.ask("text", QUESTIONS)
        self.assertEqual(len(self.transport.posts), 1)
        self.assertEqual(self.client.usage["accounted_tokens"], 0)

    def test_context_limit_fallback_preserves_payload_and_accounts_each_post(self):
        self.transport.responses = [_HTTPFailure(400, code="max_tokens_exceeded")]
        state = {"examples": ["原文", "另一条"], "context": {"source": "unchanged"}}
        before = json.dumps({"state": state, "questions": QUESTIONS}, ensure_ascii=False, sort_keys=True)
        answers = self.client.ask(state, QUESTIONS)
        self.assertEqual(answers["supported"]["noul"], .9)
        self.assertEqual(len(self.transport.posts), 2)
        self.assertIs(self.transport.posts[0][2], self.transport.posts[1][2])
        self.assertEqual(before, json.dumps({"state": state, "questions": QUESTIONS}, ensure_ascii=False, sort_keys=True))
        self.assertEqual([mode for method, _, mode in self.transport.context_modes if method == "POST"], [False, True])
        self.assertEqual(self.client.stats["context_fallbacks"], 1)
        self.assertEqual(self.client.stats["long_context_calls"], 1)
        self.assertEqual(self.client.stats["retries"], 0)
        self.assertEqual(self.client.stats["model_calls"], 2)
        self.assertEqual(self.client.usage["total_tokens"], 140)
        self.assertEqual(self.client.usage["reserved_tokens"], 0)
        rows = self.client._db.execute(
            "SELECT fingerprint,status,input_tokens,output_tokens FROM jev_usage_requests ORDER BY id"
        ).fetchall()
        self.assertNotEqual(rows[0][0], rows[1][0])
        self.assertEqual(tuple(rows[0][1:]), ("failed", 0, 0))
        self.assertEqual(tuple(rows[1][1:]), ("success", 123, 17))

    def test_exact_long_context_cache_avoids_another_default_rejection_across_instances(self):
        self.transport.responses = [_HTTPFailure(400, code="max_tokens_exceeded")]
        self.client.ask("text", QUESTIONS)
        with JevDecisions(self.cache, api_key="unit-test-not-real", retries=0) as other:
            other._transport = self.transport
            other.ask("text", QUESTIONS)
            self.assertEqual(other.stats["cache_hits"], 1)
            self.assertEqual(other.stats["long_context_cache_hits"], 1)
            self.assertEqual(other.stats["context_fallbacks"], 0)
            self.assertEqual(other.stats["model_calls"], 0)
        self.assertEqual(len(self.transport.posts), 2)
        self.assertEqual(self.client.usage["attempts"], 2)

    def test_legacy_default_cache_key_is_unchanged_and_preferred_to_long_cache(self):
        self.client.ask("text", QUESTIONS)
        identity = self.client._resolve_model()
        payload = {"model": "jev-latest", "state": "text", "questions": QUESTIONS}
        legacy_key = _digest({"version": "jev-decisions-v1", "identity": identity, "payload": payload})
        self.assertEqual(CACHE_VERSION, "jev-decisions-v1")
        row = self.client._db.execute("SELECT * FROM jev_decision_cache WHERE cache_key=?", (legacy_key,)).fetchone()
        self.assertIsNotNone(row)
        altered = json.loads(row[4])
        altered["answers"]["supported"]["noul"] = .1
        long_key = _digest({"version": CACHE_VERSION, "identity": identity, "payload": payload, "context_mode": "long"})
        self.client._db.execute("INSERT INTO jev_decision_cache VALUES(?,?,?,?,?,?)",
                                (long_key, *row[1:4], json.dumps(altered), row[5]))
        self.client._db.commit()
        answers = self.client.ask("text", QUESTIONS)
        self.assertEqual(answers["supported"]["noul"], .9)
        self.assertEqual(len(self.transport.posts), 1)
        self.assertEqual(self.client.stats["long_context_cache_hits"], 0)

    def test_long_context_does_not_carry_over_to_another_payload(self):
        self.transport.responses = [_HTTPFailure(400, code="max_tokens_exceeded")]
        self.client.ask("large text", QUESTIONS)
        self.client.ask("different small text", QUESTIONS)
        self.assertEqual([mode for method, _, mode in self.transport.context_modes if method == "POST"], [False, True, False])

    def test_other_bad_request_or_authentication_errors_do_not_enable_long_context(self):
        for status, code in ((400, None), (400, "validation_error"), (401, "max_tokens_exceeded")):
            with self.subTest(status=status, code=code):
                self.transport.responses = [_HTTPFailure(status, code=code)]
                prior_calls = len(self.transport.posts)
                with self.assertRaises(DecisionError):
                    self.client.ask(f"case {status} {code}", QUESTIONS)
                self.assertEqual(len(self.transport.posts), prior_calls + 1)
        self.assertEqual(self.client.stats["context_fallbacks"], 0)
        self.assertEqual(self.client.stats["long_context_calls"], 0)

    def test_second_context_limit_rejection_stops_after_one_mode_transition(self):
        self.client.retries = 3
        self.transport.responses = [_HTTPFailure(400, code="max_tokens_exceeded"),
                                    _HTTPFailure(400, code="max_tokens_exceeded")]
        with self.assertRaisesRegex(DecisionError, "max_tokens_exceeded"):
            self.client.ask("text", QUESTIONS)
        self.assertEqual(len(self.transport.posts), 2)
        self.assertEqual(self.client.stats["context_fallbacks"], 1)
        self.assertEqual(self.client.stats["long_context_calls"], 1)
        self.assertEqual(self.client.stats["failures"], 1)
        self.assertEqual(self.client.usage["accounted_tokens"], 0)
        self.assertEqual(self.client._db.execute("SELECT COUNT(*) FROM jev_decision_cache").fetchone()[0], 0)

    def test_context_transition_shares_bounded_transient_retry_allowance(self):
        self.client.retries = 1
        self.transport.responses = [_HTTPFailure(529), _HTTPFailure(400, code="max_tokens_exceeded"), _HTTPFailure(529)]
        with patch("trend_hunter.decisions.time.sleep") as sleep:
            with self.assertRaisesRegex(DecisionError, "529"):
                self.client.ask("text", QUESTIONS)
        self.assertEqual(len(self.transport.posts), 3)
        self.assertEqual([mode for method, _, mode in self.transport.context_modes if method == "POST"], [False, False, True])
        self.assertEqual(self.client.stats["context_fallbacks"], 1)
        self.assertEqual(self.client.stats["retries"], 1)
        sleep.assert_called_once_with(1)

    def test_context_fallback_requires_a_new_successful_ledger_reservation(self):
        self.transport.responses = [_HTTPFailure(400, code="max_tokens_exceeded")]
        reserve = self.client._reserve
        reservations = []
        def limited(fingerprint, tokens):
            reservations.append(fingerprint)
            if len(reservations) == 2:
                raise BudgetExceeded("No room for the second request")
            return reserve(fingerprint, tokens)
        with patch.object(self.client, "_reserve", side_effect=limited):
            with self.assertRaises(BudgetExceeded):
                self.client.ask("text", QUESTIONS)
        self.assertEqual(len(reservations), 2)
        self.assertNotEqual(reservations[0], reservations[1])
        self.assertEqual(len(self.transport.posts), 1)
        self.assertEqual(self.client.stats["context_fallbacks"], 0)
        self.assertEqual(self.client.stats["long_context_calls"], 0)
        self.assertEqual(self.client.usage["accounted_tokens"], 0)

    def test_unknown_cost_long_failure_cannot_bypass_hard_budget(self):
        self.client.retries = 1
        payload = {"model": "jev-latest", "state": "text", "questions": QUESTIONS}
        reservation = self.client._reservation(payload)
        self.client.max_tokens = reservation
        self.transport.responses = [_HTTPFailure(400, code="max_tokens_exceeded"), _HTTPFailure(529)]
        with patch("trend_hunter.decisions.time.sleep"):
            with self.assertRaises(BudgetExceeded):
                self.client.ask("text", QUESTIONS)
        self.assertEqual(len(self.transport.posts), 2)
        self.assertEqual(self.client.usage["reserved_tokens"], reservation)
        self.assertEqual(self.client.usage["attempts"], 2)

    def test_bad_answers_are_not_cached_but_actual_cost_is_recorded(self):
        self.transport.responses = [{"model": "jev-release-1", "answers": {}, "usage": {"input_tokens": 500, "output_tokens": 10}}]
        with self.assertRaisesRegex(DecisionError, "names"):
            self.client.ask("text", QUESTIONS)
        self.assertEqual(self.client.usage["total_tokens"], 510)
        self.client.ask("text", QUESTIONS)
        self.assertEqual(len(self.transport.posts), 2)

    def test_missing_usage_fails_closed_and_retains_reservation(self):
        self.transport.responses = [{"model": "jev-release-1", "answers": {}}]
        with self.assertRaisesRegex(DecisionError, "usage"):
            self.client.ask("text", QUESTIONS)
        self.assertGreater(self.client.usage["reserved_tokens"], 0)

    def test_invalid_probability_is_not_accepted(self):
        self.transport.responses = [{"model": "jev-release-1", "answers": {"q": {"type": "noul", "noul": 1.1}},
                                     "usage": {"input_tokens": 1, "output_tokens": 1}}]
        with self.assertRaisesRegex(DecisionError, "probability"):
            self.client.ask("text", {"q": noul_question("Supported?")})

    def test_mismatched_choice_distribution_is_not_accepted(self):
        self.transport.responses = [{"model": "jev-release-1", "answers": {"q": {"type": "choice", "choice": "a", "confidence": 0.9,
                                                                                 "probabilities": {"a": 0.2, "b": 0.2}}},
                                     "usage": {"input_tokens": 1, "output_tokens": 1}}]
        with self.assertRaisesRegex(DecisionError, "sum to one"):
            self.client.ask("text", {"q": choice_question("Which?", {"a": "First", "b": "Second"})})

    def test_invalid_inputs_do_not_make_metadata_or_model_requests(self):
        for state, questions in [(None, QUESTIONS), ("text", {}), ("text", {"q": {"type": "text"}}), ({"bad": float("nan")}, QUESTIONS)]:
            with self.assertRaises(ValueError):
                self.client.ask(state, questions)
        self.assertEqual(self.transport.calls, [])

    def test_target_is_reported_separately_from_hard_budget(self):
        self.client.target_tokens = 100
        self.client.ask("text", QUESTIONS)
        self.assertTrue(self.client.usage["target_exceeded"])
        self.assertGreater(self.client.usage["remaining_tokens"], 0)

    def test_cache_contains_no_api_key(self):
        self.client.ask("text", QUESTIONS)
        for row in self.client._db.execute("SELECT response_json FROM jev_decision_cache"):
            self.assertNotIn("unit-test-not-real", row[0])

    def test_transport_uses_fixed_https_origin_and_does_not_echo_response(self):
        self.client._transport = JevDecisions._transport.__get__(self.client)
        opener = unittest.mock.Mock()
        opener.open.side_effect = error.HTTPError("https://api.typesafe.ai/v1/models", 401, "echoed credential", {}, io.BytesIO(b"sensitive"))
        with patch("trend_hunter.decisions.request.build_opener", return_value=opener):
            with self.assertRaises(DecisionError) as caught:
                self.client.ask("text", QUESTIONS)
        self.assertNotIn("credential", str(caught.exception))
        req = opener.open.call_args.args[0]
        self.assertEqual(req.full_url, "https://api.typesafe.ai/v1/models")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 60)

    def test_context_error_classifier_requires_an_exact_structured_allowlisted_code(self):
        for payload in ({"code": "max_tokens_exceeded"}, {"error": "max_tokens_exceeded"},
                        {"error": {"code": "max_tokens_exceeded"}},
                        {"error": {"type": "max_tokens_exceeded"}},
                        {"detail": {"error_type": "max_tokens_exceeded"}}):
            with self.subTest(payload=payload):
                self.assertEqual(_context_error_code(json.dumps(payload).encode()), "max_tokens_exceeded")
        for raw in (b'not JSON: max_tokens_exceeded', b'[]', b'{"message":"max_tokens_exceeded"}',
                    b'{"error":{"message":"max_tokens_exceeded"}}', b'{"code":"validation_error"}',
                    b'{"detail":{"message":"max_tokens_exceeded"}}',
                    b'{"code":"max_tokens_exceeded"}' + b' ' * 20_000):
            with self.subTest(raw=raw[:60]):
                self.assertIsNone(_context_error_code(raw))

    def test_real_transport_adds_long_header_only_after_context_rejection_and_keeps_body_exact(self):
        self.client._transport = JevDecisions._transport.__get__(self.client)
        fake = FakeTransport()
        metadata = fake("GET", "/v1/models", None)
        payload = {"model": "jev-latest", "state": {"originals": ["原文"]}, "questions": QUESTIONS}
        response = fake("POST", "/v1/systemone", payload)
        opener = unittest.mock.Mock()
        opener.open.side_effect = [io.BytesIO(json.dumps(metadata).encode()),
                                  error.HTTPError("https://api.typesafe.ai/v1/systemone", 400, "not logged", {},
                                                  io.BytesIO(b'{"error":{"code":"max_tokens_exceeded","message":"private echoed data"}}')),
                                  io.BytesIO(json.dumps(response).encode())]
        with patch("trend_hunter.decisions.request.build_opener", return_value=opener):
            self.client.ask(payload["state"], QUESTIONS)
        requests = [call.args[0] for call in opener.open.call_args_list]
        headers = [{name.lower(): value for name, value in req.header_items()} for req in requests]
        header = "typesafe-danger-zone-long-context"
        self.assertEqual([req.method for req in requests], ["GET", "POST", "POST"])
        self.assertNotIn(header, headers[0])
        self.assertNotIn(header, headers[1])
        self.assertEqual(headers[2][header], "true")
        self.assertEqual(requests[1].data, requests[2].data)
        self.assertEqual(json.loads(requests[1].data), payload)

    def test_transport_retains_only_safe_context_code_not_provider_error_text(self):
        self.client._transport = JevDecisions._transport.__get__(self.client)
        opener = unittest.mock.Mock()
        raw = b'{"code":"max_tokens_exceeded","message":"private echoed data"}'
        body = unittest.mock.Mock(wraps=io.BytesIO(raw))
        opener.open.side_effect = error.HTTPError("https://api.typesafe.ai/v1/systemone", 400, "private reason", {}, body)
        with patch("trend_hunter.decisions.request.build_opener", return_value=opener):
            with self.assertRaises(_HTTPFailure) as caught:
                self.client._transport("POST", "/v1/systemone", {"state": "original"})
        self.assertEqual(caught.exception.code, "max_tokens_exceeded")
        self.assertNotIn("private", str(caught.exception))
        self.assertNotIn("private", repr(caught.exception.__dict__))
        body.read.assert_called_once_with(16_385)


class CredentialTests(unittest.TestCase):
    def test_environment_wins_without_overwriting_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("JEV_API_KEY=file-only\n", encoding="utf-8")
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "env-only"}, clear=True):
                self.assertEqual(load_api_key(path), "env-only")
                self.assertNotIn("JEV_API_KEY", os.environ)

    def test_dotenv_literals_comments_and_export_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("# comment\nexport JEV_API_KEY='literal-value' # comment\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_api_key(path), "literal-value")

    def test_dotenv_does_not_expand_or_execute_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("JEV_API_KEY='${HOME}$(echo nope)'\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_api_key(path), "${HOME}$(echo nope)")

    def test_missing_credentials_produce_safe_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(DecisionError, "Set JEV_API_KEY"):
                    JevDecisions(Path(directory) / "cache.db", dotenv_path=Path(directory) / "absent")


if __name__ == "__main__":
    unittest.main()
