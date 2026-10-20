"""Hosted generation boundaries, with every HTTP request replaced locally."""
import io
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import traceback
import unittest
from unittest.mock import Mock, patch
from urllib import error

from trend_hunter.local_models import LocalModels
from trend_hunter.openai_generation import (ENDPOINT, GenerationOutputError,
                                            GenerationServiceError, OpenAIGeneration)


SCHEMA = {"type": "object", "properties": {"title": {"type": "string", "minLength": 1}},
          "required": ["title"], "additionalProperties": False}


def completion(text='{"title":"A supported topic"}'):
    return {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}}


class Response:
    def __init__(self, value):
        self.body = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return self.body


class OpenAIGenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="trend-hosted-tests-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cache = self.directory / "models.sqlite3"

    def model(self, **kwargs):
        model = OpenAIGeneration(model=kwargs.pop("model", "explicit-test-model"),
                                 api_key=kwargs.pop("api_key", "synthetic-secret"),
                                 cache_path=self.cache, **kwargs)
        self.addCleanup(model.close)
        model._opener = Mock()
        model._opener.open.return_value = Response(completion())
        return model

    def test_cache_startup_errors_are_sanitized_and_close_partial_connections(self):
        for stage in ("connect", "pragma", "schema", "commit"):
            with self.subTest(stage=stage):
                database, opener = Mock(), Mock()
                failure = sqlite3.OperationalError("private-cache-path synthetic-secret")
                if stage == "pragma":
                    database.execute.side_effect = failure
                elif stage == "schema":
                    database.execute.side_effect = [None, failure]
                elif stage == "commit":
                    database.commit.side_effect = failure
                with patch("trend_hunter.openai_generation.sqlite3.connect",
                           side_effect=failure if stage == "connect" else None,
                           return_value=database), \
                        patch("trend_hunter.openai_generation.request.build_opener", return_value=opener), \
                        self.assertRaises(GenerationServiceError) as caught:
                    OpenAIGeneration(model="explicit-test-model", api_key="synthetic-secret", cache_path=":memory:")
                self.assertIn("cache could not be opened or initialized", str(caught.exception))
                self.assertIn("no request was sent", str(caught.exception))
                display = "".join(traceback.format_exception(caught.exception))
                self.assertNotIn("private-cache-path", display)
                self.assertNotIn("synthetic-secret", display)
                opener.open.assert_not_called()
                if stage == "connect":
                    database.close.assert_not_called()
                else:
                    database.close.assert_called_once_with()

    def test_cache_cleanup_failure_does_not_mask_sanitized_startup_error(self):
        database, opener = Mock(), Mock()
        database.execute.side_effect = sqlite3.DatabaseError("private-startup-detail")
        database.close.side_effect = RuntimeError("private-cleanup-detail")
        with patch("trend_hunter.openai_generation.sqlite3.connect", return_value=database), \
                patch("trend_hunter.openai_generation.request.build_opener", return_value=opener), \
                self.assertRaises(GenerationServiceError) as caught:
            OpenAIGeneration(model="explicit-test-model", api_key="synthetic-secret", cache_path=":memory:")
        database.close.assert_called_once_with()
        opener.open.assert_not_called()
        self.assertIn("no request was sent", str(caught.exception))
        display = "".join(traceback.format_exception(caught.exception))
        self.assertNotIn("private-startup-detail", display)
        self.assertNotIn("private-cleanup-detail", display)

    def test_responses_contract_usage_and_credential_free_cache(self):
        model = self.model()
        answer = model.chat_json("Preserve evidence", {"text": "Chinese source"}, SCHEMA, 200)
        req = model._opener.open.call_args.args[0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, ENDPOINT)
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Authorization"), "Bearer synthetic-secret")
        self.assertEqual(body, {"model": "explicit-test-model", "instructions": "Preserve evidence",
                               "input": '{"text":"Chinese source"}', "store": False,
                               "max_output_tokens": 200,
                               "text": {"format": {"type": "json_schema", "name": "trend_draft",
                                                   "schema": SCHEMA, "strict": True}}})
        self.assertEqual(answer, {"title": "A supported topic"})
        self.assertEqual(model.stats["total_tokens"], 25)
        self.assertEqual(model.stats["api_calls"], 1)
        self.assertEqual(model.stats["usage_unavailable_calls"], 0)
        self.assertNotIn("synthetic-secret", "\n".join(model._db.iterdump()))
        self.assertNotIn("synthetic-secret", json.dumps(model.stats))

    def test_persistent_cache_needs_no_http_and_is_separate_from_ollama(self):
        first = self.model()
        expected = first.chat_json("Instruction", {"text": "Source"}, SCHEMA)
        second = self.model(api_key="different-synthetic-key")
        second._opener.open.side_effect = AssertionError("Cache hit must not request HTTP")
        self.assertEqual(second.chat_json("Instruction", {"text": "Source"}, SCHEMA), expected)
        self.assertEqual(second.stats["api_calls"], 0)
        self.assertEqual(second.stats["cache_hits"], 1)
        with LocalModels(cache_path=self.cache) as local:
            self.assertEqual(local._db.execute("SELECT COUNT(*) FROM chats").fetchone()[0], 0)
            self.assertEqual(local._db.execute("SELECT COUNT(*) FROM openai_chats").fetchone()[0], 1)

    def test_cache_identity_covers_model_prompt_payload_schema_and_token_limit(self):
        model = self.model()
        model.chat_json("Instruction", {"text": "Source"}, SCHEMA, 200)
        model.chat_json("Different instruction", {"text": "Source"}, SCHEMA, 200)
        model.chat_json("Instruction", {"text": "Other source"}, SCHEMA, 200)
        model.chat_json("Instruction", {"text": "Source"}, {**SCHEMA, "description": "New schema"}, 200)
        model.chat_json("Instruction", {"text": "Source"}, SCHEMA, 201)
        other = self.model(model="different-test-model")
        other.chat_json("Instruction", {"text": "Source"}, SCHEMA, 200)
        self.assertEqual(model.stats["api_calls"], 5)
        self.assertEqual(other.stats["api_calls"], 1)

    def test_incomplete_refused_and_invalid_drafts_are_not_cached_or_retried(self):
        incomplete = {**completion(), "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        refusal = completion()
        refusal["output"][1]["content"] = [{"type": "refusal", "refusal": "private source echo"}]
        for response in (incomplete, refusal, completion('{"title":false}'),
                         completion('{"title":"one","title":"two"}')):
            with self.subTest(response=response):
                model = self.model()
                model._opener.open.return_value = Response(response)
                with self.assertRaises(GenerationOutputError) as caught:
                    model.chat_json("Instruction", {"source": "private source echo"}, SCHEMA)
                self.assertNotIn("private source echo", str(caught.exception))
                if response is incomplete:
                    self.assertIn("max_output_tokens=450", str(caught.exception))
                    self.assertIn("reasoning tokens", str(caught.exception))
                self.assertEqual(model._opener.open.call_count, 1)
                self.assertEqual(model.stats["api_calls"], 1)
                self.assertEqual(model.stats["total_tokens"], 25)
                self.assertEqual(model._db.execute("SELECT COUNT(*) FROM openai_chats").fetchone()[0], 0)

    def test_http_errors_and_timeouts_are_sanitized_and_never_retried(self):
        for failure in (error.HTTPError(ENDPOINT, 429, "synthetic-secret private-source", {},
                                        io.BytesIO(b'{"error":"synthetic-secret private-source"}')),
                        TimeoutError("synthetic-secret private-source")):
            with self.subTest(failure=type(failure).__name__):
                model = self.model()
                model._opener.open.side_effect = failure
                with self.assertRaises(GenerationServiceError) as caught:
                    model.chat_json("Instruction", {"source": "private-source"}, SCHEMA)
                display = "".join(traceback.format_exception(caught.exception))
                self.assertNotIn("synthetic-secret", display)
                self.assertNotIn("private-source", display)
                self.assertEqual(model._opener.open.call_count, 1)
                self.assertEqual(model.stats["usage_unavailable_calls"], 1)
                self.assertEqual(model.stats["failures"], 1)

    def test_partial_http_read_is_sanitized_without_retry_or_cache(self):
        model = self.model()
        response = model._opener.open.return_value
        response.read = Mock(side_effect=IncompleteRead(b"synthetic-secret private-source", 100))
        with self.assertRaises(GenerationServiceError) as caught:
            model.chat_json("Instruction", {"source": "private-source"}, SCHEMA)
        display = "".join(traceback.format_exception(caught.exception))
        self.assertNotIn("synthetic-secret", display)
        self.assertNotIn("private-source", display)
        self.assertIn("billing may be unknown", display)
        self.assertEqual(model._opener.open.call_count, 1)
        self.assertEqual(model.stats["usage_unavailable_calls"], 1)
        self.assertEqual(model._db.execute("SELECT COUNT(*) FROM openai_chats").fetchone()[0], 0)

    def test_corrupt_cached_draft_is_replaced_by_one_valid_request(self):
        model = self.model()
        model.chat_json("Instruction", {}, SCHEMA)
        with model._db:
            model._db.execute("UPDATE openai_chats SET response_json = ?", ('{"title":false}',))
        self.assertEqual(model.chat_json("Instruction", {}, SCHEMA), {"title": "A supported topic"})
        self.assertEqual(model._opener.open.call_count, 2)

    def test_key_loading_is_literal_environment_first_and_never_executed(self):
        credential_file = self.directory / "credentials.env"
        credential_file.write_text("JEV_API_KEY=not-an-openai-key\nexport OPENAI_API_KEY='literal-$SECRET' # comment\n",
                                   encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            from_file = OpenAIGeneration(model="explicit", cache_path=":memory:", dotenv_path=credential_file)
            self.addCleanup(from_file.close)
            self.assertEqual(from_file._api_key, "literal-$SECRET")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-literal"}, clear=True):
            from_env = OpenAIGeneration(model="explicit", cache_path=":memory:", dotenv_path=credential_file)
            self.addCleanup(from_env.close)
            self.assertEqual(from_env._api_key, "environment-literal")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GenerationServiceError, "OPENAI_API_KEY"):
                OpenAIGeneration(model="explicit", cache_path=":memory:", dotenv_path=self.directory / "absent")

    def test_bad_configuration_and_inputs_do_not_make_a_request(self):
        for model in (None, "", " ", "model with spaces"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                OpenAIGeneration(model=model, api_key="synthetic", cache_path=":memory:")
        model = self.model()
        for limit in (0, -1, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                model.chat_json("Instruction", {}, SCHEMA, limit)
        with self.assertRaises(ValueError):
            model.chat_json("Instruction", {"bad": float("nan")}, SCHEMA)
        self.assertFalse(model._opener.open.called)


if __name__ == "__main__":
    unittest.main()
