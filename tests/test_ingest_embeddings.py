"""Embedding-only ingestion does not depend on writer configuration."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import ingest_embeddings as command


class IngestionConfigTests(unittest.TestCase):
    def test_embedding_settings_ignore_unrelated_provider_and_writer(self):
        with tempfile.TemporaryDirectory(prefix="trend-ingest-config-") as temporary:
            original = Path.cwd()
            try:
                os.chdir(temporary)
                Path(".env").write_text(
                    "TREND_EMBEDDING_MODEL=file-encoder\n"
                    "TREND_OLLAMA_URL=http://127.0.0.1:11435\n"
                    "TREND_CHAT_PROVIDER=unknown-provider\n"
                    "TREND_OPENAI_CHAT_MODEL=\"unterminated\n", encoding="utf-8")
                models = MagicMock()
                models.health.return_value = {"embedding_model": "env-encoder"}
                models.stats = {"embedding_calls": 0, "embedding_cache_hits": 1}
                models.embed.return_value = np.ones((1, 384), dtype=np.float32)
                factory = MagicMock()
                factory.return_value.__enter__.return_value = models
                argv = ["ingest_embeddings.py", "--as-of", "2026-07-14", "--report", "report.json"]
                with patch.dict(os.environ, {"TREND_EMBEDDING_MODEL": "env-encoder"}, clear=True), \
                        patch("sys.argv", argv), patch.object(command, "LocalModels", factory), \
                        patch.object(command, "prepare", return_value=([], [], {})), \
                        patch.object(command, "make_units", return_value=[]), \
                        patch.object(command, "unique_units", return_value=(["sample"], np.array([0]))):
                    command.main()
                self.assertEqual(factory.call_args.kwargs["embedding_model"], "env-encoder")
                self.assertEqual(factory.call_args.kwargs["base_url"], "http://127.0.0.1:11435")
                models.health.assert_called_once_with(("embedding",))
                models.chat_json.assert_not_called()
                models.unload.assert_not_called()
                report = json.loads(Path("report.json").read_text(encoding="utf-8"))
                self.assertTrue(report["complete"])
                self.assertEqual(report["dimensions"], 384)
            finally:
                os.chdir(original)

    def test_encoder_width_must_remain_consistent_between_windows(self):
        with tempfile.TemporaryDirectory(prefix="trend-ingest-width-") as temporary:
            report_path = Path(temporary) / "report.json"
            models = MagicMock()
            models.health.return_value = {"embedding_model": "test-encoder"}
            models.stats = {"embedding_calls": 0, "embedding_cache_hits": 2}
            models.embed.side_effect = [np.ones((1, 384)), np.ones((1, 768))]
            factory = MagicMock()
            factory.return_value.__enter__.return_value = models
            argv = ["ingest_embeddings.py", "--as-of", "2026-07-14", "--report", str(report_path),
                    "--residency-inputs", "1", "--embedding-model", "test-encoder",
                    "--ollama-url", "http://127.0.0.1:11434"]
            with patch("sys.argv", argv), patch.object(command, "LocalModels", factory), \
                    patch.object(command, "prepare", return_value=([], [], {})), \
                    patch.object(command, "make_units", return_value=[]), \
                    patch.object(command, "unique_units", return_value=(["first", "second"], np.array([0, 1]))):
                with self.assertRaisesRegex(RuntimeError, "dimensions changed"):
                    command.main()
            self.assertFalse(json.loads(report_path.read_text(encoding="utf-8"))["complete"])


if __name__ == "__main__":
    unittest.main()
