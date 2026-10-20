import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib import error

import numpy as np

from trend_hunter.local_models import LocalGenerationError, LocalModelError, LocalModels


SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string", "minLength": 1}},
    "required": ["title"],
    "additionalProperties": False,
}


class ModelTransport:
    def __init__(self):
        self.embedding_digest = "embedding-digest-1"
        self.chat_digest = "chat-digest-1"
        self.calls = []
        self.chat_responses = []
        self.embedding_response = None

    def __call__(self, method, path, payload):
        self.calls.append((method, path, payload))
        if path == "/api/tags":
            return {
                "models": [
                    {"name": "bge-m3:latest", "digest": self.embedding_digest, "details": {"family": "bert"}},
                    {"name": "qwen3:8b", "digest": self.chat_digest},
                ]
            }
        if path == "/api/embed":
            if self.embedding_response is not None:
                return self.embedding_response
            return {"embeddings": [[len(text), 2, 3] for text in payload["input"]]}
        if path == "/api/chat":
            if self.chat_responses:
                response = self.chat_responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response
            return {"done": True, "done_reason": "stop", "message": {"content": '{"title":"帧率问题"}'}}
        if path == "/api/generate":
            return {"done": True, "done_reason": "unload"}
        raise AssertionError(path)


class LocalModelTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cache = Path(self.tempdir.name) / "models.sqlite3"
        self.transport = ModelTransport()
        self.models = LocalModels(cache_path=self.cache, batch_size=2)
        self.models._transport = self.transport

    def tearDown(self):
        self.models.close()
        self.tempdir.cleanup()

    def test_health_resolves_latest_and_returns_installed_details(self):
        result = self.models.health()
        self.assertEqual(result["embedding_model"]["name"], "bge-m3:latest")
        self.assertEqual(result["embedding_model"]["details"]["family"], "bert")
        self.assertEqual(result["chat_model"]["name"], "qwen3:8b")
        self.assertEqual(self.models.stats["health_calls"], 1)

    def test_chat_only_store_does_not_require_an_embedding_model(self):
        def transport(method, path, payload):
            response = self.transport(method, path, payload)
            if path == "/api/tags":
                response["models"] = [model for model in response["models"] if model["name"] == "qwen3:8b"]
            return response

        self.models._transport = transport
        self.assertEqual(self.models.chat_json("system", {}, SCHEMA), {"title": "帧率问题"})
        details = self.models.health(required_roles=("chat",))
        self.assertEqual(details["chat_model"]["name"], "qwen3:8b")
        self.assertIsNone(details["embedding_model"])
        with self.assertRaisesRegex(LocalModelError, "bge-m3.*not installed"):
            self.models.health()

    def test_embedding_only_store_does_not_require_a_chat_model(self):
        def transport(method, path, payload):
            response = self.transport(method, path, payload)
            if path == "/api/tags":
                response["models"] = [model for model in response["models"] if model["name"] == "bge-m3:latest"]
            return response

        self.models._transport = transport
        self.assertEqual(self.models.embed(["message"]).shape, (1, 3))
        details = self.models.health(required_roles=("embedding",))
        self.assertEqual(details["embedding_model"]["name"], "bge-m3:latest")
        self.assertIsNone(details["chat_model"])
        with self.assertRaisesRegex(LocalModelError, "qwen3:8b.*not installed"):
            self.models.health()

    def test_invalid_health_roles_fail_before_requests(self):
        for roles in ("chat", ["chat"], ("remote",)):
            with self.subTest(roles=roles), self.assertRaises(ValueError):
                self.models.health(required_roles=roles)
        self.assertEqual(self.transport.calls, [])

    def test_missing_model_and_missing_digest_are_actionable(self):
        with patch.object(self.models, "_transport", return_value={"models": []}):
            with self.assertRaisesRegex(LocalModelError, "not installed.*no download"):
                self.models.health()

    def test_malformed_installed_model_name_is_rejected(self):
        with patch.object(self.models, "_transport", return_value={"models": [{"name": ["invalid"]}]}):
            with self.assertRaisesRegex(LocalModelError, "invalid model name"):
                self.models.health()
        with patch.object(self.models, "_transport", return_value={"models": [{"name": "bge-m3:latest"}]}):
            with self.assertRaisesRegex(LocalModelError, "no digest"):
                self.models.health()

    def test_embedding_batches_normalizes_and_preserves_input_order(self):
        texts = ["短", "a longer message", "短", "第三条"]
        result = self.models.embed(texts)
        expected = np.array([[len(text), 2, 3] for text in texts], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(result, expected)
        self.assertEqual(result.dtype, np.float32)
        calls = [payload for _, path, payload in self.transport.calls if path == "/api/embed"]
        self.assertEqual([len(call["input"]) for call in calls], [2, 1])
        self.assertFalse(calls[0]["truncate"])
        self.assertEqual(calls[0]["options"], {"num_ctx": 512, "num_batch": 512})

    def test_embedding_options_are_local_copied_overrides_and_isolate_cache(self):
        self.models.embed(["same text"])
        overrides = {"num_gpu": 0, "num_batch": 64}
        with LocalModels(cache_path=self.cache, embedding_options=overrides) as cpu:
            cpu._transport = self.transport
            overrides["num_gpu"] = 99
            cpu.embed(["same text"])
            self.assertEqual(cpu.stats["embedding_calls"], 1)
            self.assertEqual(cpu.stats["embedding_cache_hits"], 0)
            request_options = [payload["options"] for _, path, payload in self.transport.calls if path == "/api/embed"][-1]
            self.assertEqual(request_options, {"num_ctx": 512, "num_batch": 64, "num_gpu": 0})
            cpu.embed(["same text"])
            self.assertEqual(cpu.stats["embedding_cache_hits"], 1)
        with LocalModels(cache_path=self.cache) as unchanged:
            unchanged._transport = self.transport
            unchanged.embed(["same text"])
            self.assertEqual(unchanged.stats["embedding_calls"], 0)
            self.assertEqual(unchanged.embedding_options, {"num_ctx": 512, "num_batch": 512})

    def test_embedding_options_validate_finite_json_and_string_keys(self):
        for options in ([], "num_gpu=0", {"num_gpu": float("nan")}, {"num_gpu": object()}, {1: 2}, {"": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LocalModels(cache_path=":memory:", embedding_options=options)

    def test_embedding_device_placement_reuses_existing_vectors(self):
        original = self.models.embed(["cached input"])
        for placement in ({"num_gpu": 0, "num_thread": 2}, {"num_gpu": 99, "num_thread": 8}):
            with self.subTest(placement=placement), LocalModels(cache_path=self.cache, embedding_options=placement) as models:
                models._transport = self.transport
                np.testing.assert_array_equal(models.embed(["cached input"]), original)
                self.assertEqual(models.stats["embedding_calls"], 0)
                self.assertEqual(models.stats["embedding_cache_hits"], 1)
        self.assertEqual(len([call for call in self.transport.calls if call[1] == "/api/embed"]), 1)

    def test_embedding_placement_is_sent_for_new_inputs_without_separating_cache(self):
        with LocalModels(cache_path=self.cache, embedding_options={"num_gpu": 0, "num_thread": 2}) as cpu:
            cpu._transport = self.transport
            original = cpu.embed(["new input"])
            options = [body["options"] for _, path, body in self.transport.calls if path == "/api/embed"][-1]
            self.assertEqual(options["num_gpu"], 0)
            self.assertEqual(options["num_thread"], 2)
        with LocalModels(cache_path=self.cache, embedding_options={"num_gpu": 99, "num_thread": 8}) as gpu:
            gpu._transport = self.transport
            np.testing.assert_array_equal(gpu.embed(["new input"]), original)
            self.assertEqual(gpu.stats["embedding_calls"], 0)
            self.assertEqual(gpu.stats["embedding_cache_hits"], 1)

    def test_embedding_cache_reuses_only_identical_input_model_and_version(self):
        original = self.models.embed(["old context: 卡顿"])
        np.testing.assert_array_equal(self.models.embed(["old context: 卡顿"]), original)
        self.assertEqual(self.models.stats["embedding_calls"], 1)
        self.assertEqual(self.models.stats["embedding_cache_hits"], 1)
        self.models.embed(["new context: 卡顿"])
        self.assertEqual(self.models.stats["embedding_calls"], 2)
        self.transport.embedding_digest = "embedding-digest-2"
        self.models.embed(["old context: 卡顿"])
        self.assertEqual(self.models.stats["embedding_calls"], 3)
        with patch("trend_hunter.embedding_identity.CACHE_VERSION", "next-version"):
            self.models.embed(["old context: 卡顿"])
        self.assertEqual(self.models.stats["embedding_calls"], 4)

    def test_embedding_cache_survives_a_new_instance(self):
        original = self.models.embed(["缓存"])
        with LocalModels(cache_path=self.cache) as another:
            another._transport = self.transport
            np.testing.assert_array_equal(another.embed(["缓存"]), original)
            self.assertEqual(another.stats["embedding_calls"], 0)
            self.assertEqual(another.stats["embedding_cache_hits"], 1)

    def test_empty_embedding_input_needs_no_model_request(self):
        self.assertEqual(self.models.embed([]).shape, (0, 0))
        self.assertEqual(self.transport.calls, [])

    def test_embedding_input_is_validated_before_requests(self):
        for texts in ("text", [""], ["  "], [None], [1]):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                self.models.embed(texts)
        self.assertEqual(self.transport.calls, [])

    def test_invalid_embedding_count_dimensions_and_values_fail(self):
        cases = (
            {"embeddings": []},
            {"embeddings": [[1, 2], [1, 2, 3]]},
            {"embeddings": [[0, 0], [1, 2]]},
            {"embeddings": [[float("nan"), 1], [1, 2]]},
            {"embeddings": [[1, float("inf")], [1, 2]]},
            {"embeddings": [["bad", 1], [1, 2]]},
        )
        for response in cases:
            with self.subTest(response=response):
                self.transport.embedding_response = response
                with self.assertRaises(LocalModelError):
                    self.models.embed(["one", "two"])
        self.assertEqual(self.models._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], 0)

    def test_corrupt_embedding_cache_is_recomputed(self):
        expected = self.models.embed(["example"])
        self.models._db.execute("UPDATE embeddings SET vector = ?", (b"bad",))
        self.models._db.commit()
        actual = self.models.embed(["example"])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.models.stats["embedding_calls"], 2)

    def test_normalized_cache_preserves_exact_float32_values(self):
        vectors = np.random.default_rng(123).normal(size=(2, 1024)).tolist()
        self.transport.embedding_response = {"embeddings": vectors}
        first = self.models.embed(["first message", "second message"])
        np.testing.assert_array_equal(self.models.embed(["first message", "second message"]), first)

    def test_failed_batch_bisects_to_eight_preserving_order_and_cache(self):
        self.models.batch_size = 32
        sizes = []

        def transport(method, path, payload):
            if path != "/api/embed":
                return self.transport(method, path, payload)
            sizes.append(len(payload["input"]))
            if len(payload["input"]) > 8:
                raise LocalModelError("unsupported value: NaN", status_code=500)
            return {"embeddings": [[int(text) + 1, 2, 3] for text in payload["input"]]}

        self.models._transport = transport
        texts = [str(index) for index in range(32)]
        result = self.models.embed(texts)
        expected = np.array([[index + 1, 2, 3] for index in range(32)], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(result, expected, atol=1e-7)
        self.assertEqual(sizes, [32, 16, 8, 8, 16, 8, 8])
        self.assertEqual(self.models.stats["embedding_retries"], 6)
        self.assertEqual(self.models.stats["embedding_batch_splits"], 3)
        np.testing.assert_array_equal(self.models.embed(texts), result)
        self.assertEqual(len(sizes), 7)
        self.assertEqual(self.models.stats["embedding_cache_hits"], 32)

    def test_invalid_vectors_also_trigger_bounded_bisection(self):
        self.models.batch_size = 8
        sizes = []

        def transport(method, path, payload):
            if path != "/api/embed":
                return self.transport(method, path, payload)
            count = len(payload["input"])
            sizes.append(count)
            vector = [float("nan"), 1, 2] if count > 2 else [1, 2, 3]
            return {"embeddings": [vector] * count}

        self.models._transport = transport
        result = self.models.embed([str(index) for index in range(8)])
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(sizes, [8, 4, 2, 2, 4, 2, 2])

    def test_single_bad_input_fails_and_valid_partial_batches_resume(self):
        self.models.batch_size = 4
        still_bad = True
        calls = []
        private_text = "private failing input"

        def transport(method, path, payload):
            if path != "/api/embed":
                return self.transport(method, path, payload)
            calls.append(list(payload["input"]))
            if still_bad and private_text in payload["input"]:
                raise LocalModelError("unsupported value: NaN", status_code=500)
            return {"embeddings": [[len(text), 2, 3] for text in payload["input"]]}

        self.models._transport = transport
        texts = ["first", "second", private_text, "fourth"]
        with self.assertRaisesRegex(LocalModelError, "single input.*no input was skipped") as raised:
            self.models.embed(texts)
        self.assertNotIn(private_text, str(raised.exception))
        self.assertEqual(self.models._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], 2)
        still_bad = False
        result = self.models.embed(texts)
        self.assertEqual(result.shape, (4, 3))
        self.assertEqual(calls[-1], [private_text, "fourth"])
        self.assertEqual(self.models.stats["embedding_cache_hits"], 2)

    def test_http_client_errors_are_not_split_or_retried(self):
        def transport(method, path, payload):
            if path == "/api/embed":
                raise LocalModelError("HTTP 400: invalid input", status_code=400)
            return self.transport(method, path, payload)

        self.models._transport = transport
        with self.assertRaisesRegex(LocalModelError, "HTTP 400"):
            self.models.embed(["one", "two"])
        self.assertEqual(self.models.stats["embedding_calls"], 1)
        self.assertEqual(self.models.stats["embedding_retries"], 0)
        self.assertEqual(self.models.stats["embedding_batch_splits"], 0)

    def test_chat_options_and_cache_are_prompt_schema_options_and_digest_specific(self):
        self.assertEqual(self.models.chat_json("system", {"ids": [1]}, SCHEMA), {"title": "帧率问题"})
        self.models.chat_json("system", {"ids": [1]}, SCHEMA)
        self.assertEqual(self.models.stats["chat_calls"], 1)
        self.assertEqual(self.models.stats["chat_cache_hits"], 1)
        self.models.chat_json("other system", {"ids": [1]}, SCHEMA)
        self.models.chat_json("system", {"ids": [2]}, SCHEMA)
        self.models.chat_json("system", {"ids": [1]}, dict(SCHEMA, description="new schema"))
        self.models.chat_json("system", {"ids": [1]}, SCHEMA, max_tokens=451)
        self.transport.chat_digest = "chat-digest-2"
        self.models.chat_json("system", {"ids": [1]}, SCHEMA)
        self.assertEqual(self.models.stats["chat_calls"], 6)
        body = [payload for _, path, payload in self.transport.calls if path == "/api/chat"][0]
        self.assertFalse(body["think"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["format"], SCHEMA)
        self.assertEqual(body["options"], {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 450})

    def test_chat_options_are_copied_and_propagated_with_explicit_token_limit(self):
        overrides = {"num_gpu": 0, "num_ctx": 8192, "temperature": 0.1, "seed": 7,
                     "num_predict": 999, "stop": ["END"]}
        with LocalModels(cache_path=self.cache, chat_options=overrides, keep_alive=0) as models:
            models._transport = self.transport
            overrides["num_gpu"] = 99
            overrides["stop"].append("ALTERED")
            models.chat_json("system", {}, SCHEMA, max_tokens=123)
            body = [body for _, path, body in self.transport.calls if path == "/api/chat"][-1]
            self.assertEqual(body["options"], {"temperature": 0.1, "seed": 7, "num_ctx": 8192,
                                               "num_gpu": 0, "num_predict": 123, "stop": ["END"]})
            self.assertEqual(body["keep_alive"], 0)
            self.assertFalse(body["think"])
            models.chat_json("system", {}, SCHEMA, max_tokens=123)
            self.assertEqual(models.stats["chat_calls"], 1)
            self.assertEqual(models.stats["chat_cache_hits"], 1)

    def test_chat_options_changes_isolate_cache_but_identical_options_reuse_it(self):
        for options, expected_calls, expected_hits in (
            ({"num_ctx": 8192, "num_gpu": 0}, 1, 0),
            ({"num_gpu": 0, "num_ctx": 8192}, 0, 1),
            ({"num_ctx": 4096, "num_gpu": 0}, 1, 0),
        ):
            with self.subTest(options=options), LocalModels(cache_path=self.cache, chat_options=options) as models:
                models._transport = self.transport
                models.chat_json("system", {}, SCHEMA)
                self.assertEqual(models.stats["chat_calls"], expected_calls)
                self.assertEqual(models.stats["chat_cache_hits"], expected_hits)

    def test_chat_options_validate_finite_json_and_string_keys(self):
        for options in ([], "num_gpu=0", {"temperature": float("nan")}, {"stop": object()}, {1: 2}, {"": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LocalModels(cache_path=":memory:", chat_options=options)

    def test_chat_retries_bad_output_once_and_does_not_cache_failure(self):
        bad_results = [
            {"done": False, "message": {"content": '{"title":"x"}'}},
            {"done": True, "done_reason": "length", "message": {"content": '{"title":"x"}'}},
            {"done": True, "message": {"content": "not JSON"}},
            {"done": True, "message": {"content": '{"title":null}'}},
            {"done": True, "message": {"content": '{"title":"x","extra":1}'}},
            {"done": True, "message": {"content": '{"title":"x","title":"y"}'}},
            {"done": True, "message": {"content": '{"title":NaN}'}},
            {"done": True, "message": {"content": "[]"}},
        ]
        for index, bad in enumerate(bad_results):
            with self.subTest(bad=bad):
                before = self.models.stats["chat_calls"]
                self.transport.chat_responses = [bad, bad]
                with self.assertRaisesRegex(LocalModelError, "after two attempts"):
                    self.models.chat_json("system", {"attempt": index}, SCHEMA)
                self.assertEqual(self.models.stats["chat_calls"] - before, 2)
        self.assertEqual(self.models._db.execute("SELECT COUNT(*) FROM chats").fetchone()[0], 0)

    def test_chat_can_recover_on_second_attempt(self):
        self.transport.chat_responses = [{"done": True, "message": {"content": "bad"}}]
        self.assertEqual(self.models.chat_json("system", {}, SCHEMA), {"title": "帧率问题"})
        self.assertEqual(self.models.stats["chat_calls"], 2)

    def test_schema_retry_supplies_failed_output_and_reason_then_caches_repair(self):
        schema = {**SCHEMA, "properties": {"title": {"type": "string", "pattern": r"[\u3400-\u9fff]"}}}
        invalid = '{"title":"English only"}'
        self.transport.chat_responses = [{"done": True, "message": {"content": invalid}}]
        payload = {"task": "write title"}
        self.assertEqual(self.models.chat_json("system", payload, schema), {"title": "帧率问题"})
        calls = [body for _, path, body in self.transport.calls if path == "/api/chat"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]["messages"]), 2)
        self.assertEqual(calls[1]["messages"][-2], {"role": "assistant", "content": invalid})
        feedback = json.loads(calls[1]["messages"][-1]["content"])
        self.assertIn("$.title", feedback["validation_error"])
        self.assertIn("pattern", feedback["validation_error"])
        self.assertEqual(payload, {"task": "write title"})
        self.models.chat_json("system", payload, schema)
        self.assertEqual(self.models.stats["chat_calls"], 2)
        self.assertEqual(self.models.stats["chat_cache_hits"], 1)

    def test_invalid_completion_is_distinct_from_service_unavailability(self):
        bad = {"done": True, "message": {"content": '{"title":null}'}}
        self.transport.chat_responses = [bad, bad]
        with self.assertRaisesRegex(LocalGenerationError, r"Validation:.*\$\.title"):
            self.models.chat_json("system", {"case": "invalid"}, SCHEMA)
        self.transport.chat_responses = [LocalModelError("offline"), LocalModelError("offline")]
        with self.assertRaises(LocalModelError) as raised:
            self.models.chat_json("system", {"case": "offline"}, SCHEMA)
        self.assertNotIsInstance(raised.exception, LocalGenerationError)

    def test_chat_nested_schema_checks(self):
        schema = {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "minItems": 1, "items": {"type": "integer"}},
                "status": {"type": "string", "enum": ["new", "growing"]},
            },
            "required": ["ids", "status"],
        }
        response = {"done": True, "message": {"content": '{"ids":[true],"status":"new"}'}}
        self.transport.chat_responses = [response, response]
        with self.assertRaises(LocalModelError):
            self.models.chat_json("system", {}, schema)

    def test_chat_input_validation_needs_no_model_request(self):
        cases = (("", {}, SCHEMA, 450), ("sys", [], SCHEMA, 450), ("sys", {}, [], 450),
                 ("sys", {}, SCHEMA, 0), ("sys", {}, SCHEMA, True), ("sys", {"x": float("nan")}, SCHEMA, 450))
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.models.chat_json(*args)
        self.assertEqual(self.transport.calls, [])

    def test_unload_uses_explicit_zero_keep_alive(self):
        self.models.health()
        self.models.unload("bge-m3")
        method, path, body = self.transport.calls[-1]
        self.assertEqual((method, path), ("POST", "/api/generate"))
        self.assertEqual(body, {"model": "bge-m3:latest", "prompt": "", "stream": False, "keep_alive": 0})
        self.assertEqual(self.models.stats["unload_calls"], 1)


class TransportBoundaryTests(unittest.TestCase):
    def test_external_urls_credentials_and_paths_are_rejected(self):
        urls = (
            "https://example.com", "http://192.168.1.2:11434", "http://127.0.0.1.evil:11434",
            "http://user:secret@127.0.0.1:11434", "http://127.0.0.1:11434/api",
            "http://127.0.0.1:11434?x=1", "http://127.0.0.1:11434#fragment",
            "file:///tmp/models", "http://127.0.0.1:0", "http://127.0.0.1:99999",
        )
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                LocalModels(base_url=url, cache_path=":memory:")

    def test_localhost_is_normalized_and_ipv6_loopback_is_allowed(self):
        with LocalModels(base_url="http://localhost:11434/", cache_path=":memory:") as models:
            self.assertEqual(models.base_url, "http://127.0.0.1:11434")
        with LocalModels(base_url="http://[::1]:11434", cache_path=":memory:") as models:
            self.assertEqual(models.base_url, "http://[::1]:11434")

    def test_transport_uses_json_no_proxy_and_rejects_redirects(self):
        with LocalModels(cache_path=":memory:") as models:
            # build_opener removes an empty ProxyHandler rather than retaining environment proxies.
            self.assertFalse(any(getattr(handler, "proxies", None) for handler in models._opener.handlers))
            redirect = next(handler for handler in models._opener.handlers if hasattr(handler, "redirect_request"))
            self.assertIsNone(redirect.redirect_request(None, None, 302, "redirect", {}, "https://external.invalid"))
            with patch.object(models._opener, "open", side_effect=error.HTTPError(
                "http://127.0.0.1:11434/api/chat", 302, "redirect", {}, io.BytesIO(b"private content")
            )):
                with self.assertRaisesRegex(LocalModelError, "HTTP 302") as raised:
                    models._request("POST", "/api/chat", {"text": "private content"})
                self.assertNotIn("private content", str(raised.exception))

    def test_transport_connection_error_is_actionable(self):
        with LocalModels(cache_path=":memory:") as models:
            with patch.object(models._opener, "open", side_effect=error.URLError("blocked")):
                with self.assertRaisesRegex(LocalModelError, "start the local service"):
                    models.health()
            self.assertEqual(models.stats["api_calls"], 1)

    def test_http_json_error_exposes_safe_bounded_operational_detail(self):
        with LocalModels(cache_path=":memory:") as models:
            body = json.dumps({"error": "failed to encode response: json: unsupported value: NaN"}).encode()
            failure = error.HTTPError(models.base_url + "/api/embed", 500, "failure", {}, io.BytesIO(body))
            with patch.object(models._opener, "open", side_effect=failure):
                with self.assertRaisesRegex(LocalModelError, "unsupported value: NaN") as raised:
                    models._request("POST", "/api/embed", {"input": ["private"]})
            self.assertEqual(raised.exception.status_code, 500)
            self.assertTrue(raised.exception.retryable_embedding)

            body = json.dumps({"error": "runtime failure api_key=secret123 input: private source content"}).encode()
            failure = error.HTTPError(models.base_url + "/api/embed", 500, "failure", {}, io.BytesIO(body))
            with patch.object(models._opener, "open", side_effect=failure):
                with self.assertRaises(LocalModelError) as raised:
                    models._request("POST", "/api/embed", {})
            self.assertNotIn("secret123", str(raised.exception))
            self.assertNotIn("private source content", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
