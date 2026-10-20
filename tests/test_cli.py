"""CLI failure reporting still writes the pipeline's partial artifacts."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run as command


class CliResultTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory(prefix="trend-cli-config-")
        self.addCleanup(workspace.cleanup)
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(workspace.name)
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_local_errors_and_cleanup_failures_write_artifacts_before_nonzero_exit(self):
        partial = {"as_of": "2026-07-14", "trends": []}
        for review, expected in (
            ({"stop_reason": "local_model_error", "local_model_error": "Unavailable local model"}, 2),
            ({"stop_reason": "embedding_cache_error", "embedding_cache_error": "Missing cached embeddings"}, 2),
            ({"stop_reason": "generation_service_error", "generation_error": "Unavailable hosted model"}, 2),
            ({"stop_reason": "eight_supported_trends", "local_cleanup_errors": [{"model": "local"}]}, 2),
            ({"stop_reason": "candidates_exhausted"}, 0),
        ):
            with self.subTest(review=review), tempfile.TemporaryDirectory(prefix="trend-cli-") as temporary:
                output = Path(temporary) / "trends.json"
                diagnostics = Path(temporary) / "review.json"
                argv = ["run.py", "--input", "unused-fixture.jsonl", "--as-of", "2026-07-14",
                        "--out", str(output), "--review-out", str(diagnostics)]
                with patch("sys.argv", argv), \
                        patch("trend_hunter.pipeline.run", return_value=(partial, review)) as pipeline_run:
                    exit_code = command.main()
                self.assertEqual(exit_code, expected)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), partial)
                self.assertEqual(json.loads(diagnostics.read_text(encoding="utf-8")), review)
                self.assertEqual(pipeline_run.call_args.kwargs["chat_context"], 16384)
                self.assertEqual(pipeline_run.call_args.kwargs["chat_provider"], "ollama")
                self.assertEqual(pipeline_run.call_args.kwargs["chat_model"], "qwen3:8b")

    def test_openai_requires_explicit_model_except_for_prepare_only(self):
        base = ["run.py", "--input", "unused.jsonl", "--as-of", "2026-07-14", "--out", "unused.json",
                "--chat-provider", "openai"]
        with patch("sys.argv", base), patch("trend_hunter.pipeline.run") as pipeline_run:
            with self.assertRaises(SystemExit) as caught:
                command.main()
            self.assertEqual(caught.exception.code, 2)
            pipeline_run.assert_not_called()
        partial = {"as_of": "2026-07-14", "trends": []}
        for extra in (["--chat-model", "explicit-hosted-model"], ["--prepare-only"]):
            with self.subTest(extra=extra), patch("sys.argv", base + extra), \
                    patch("trend_hunter.pipeline.run", return_value=(partial, {"stop_reason": "candidates_exhausted"})) as pipeline_run, \
                    patch("trend_hunter.pipeline.write_json"):
                self.assertEqual(command.main(), 0)
                self.assertEqual(pipeline_run.call_args.kwargs["chat_provider"], "openai")
                self.assertEqual(pipeline_run.call_args.kwargs["chat_model"],
                                 "explicit-hosted-model" if "--chat-model" in extra else None)

    def test_resolved_models_reach_pipeline_with_cli_precedence(self):
        Path(".env").write_text(
            "TREND_CHAT_PROVIDER=openai\nTREND_OPENAI_CHAT_MODEL=file-writer\n"
            "TREND_EMBEDDING_MODEL=file-encoder\nTREND_JEV_MODEL=file-decider\n"
            "TREND_OLLAMA_URL=http://127.0.0.1:11435\n", encoding="utf-8")
        partial = {"as_of": "2026-07-14", "trends": []}
        argv = ["run.py", "--input", "unused.jsonl", "--as-of", "2026-07-14",
                "--out", "unused.json", "--embedding-model", "cli-encoder"]
        with patch.dict(os.environ, {"TREND_OPENAI_CHAT_MODEL": "env-writer"}), \
                patch("sys.argv", argv), \
                patch("trend_hunter.pipeline.run", return_value=(partial, {"stop_reason": "candidates_exhausted"})) as pipeline_run, \
                patch("trend_hunter.pipeline.write_json"):
            self.assertEqual(command.main(), 0)
        settings = pipeline_run.call_args.kwargs
        self.assertEqual(settings["embedding_model"], "cli-encoder")
        self.assertEqual(settings["chat_provider"], "openai")
        self.assertEqual(settings["chat_model"], "env-writer")
        self.assertEqual(settings["jev_model"], "file-decider")
        self.assertEqual(settings["base_url"], "http://127.0.0.1:11435")


if __name__ == "__main__":
    unittest.main()
