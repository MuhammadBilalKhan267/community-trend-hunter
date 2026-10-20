"""Settings precedence and literal parsing using synthetic temporary files only."""
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trend_hunter.config import (ConfigurationError, DEFAULT_CHAT_PROVIDER, DEFAULT_EMBEDDING_MODEL,
                                 DEFAULT_JEV_MODEL, DEFAULT_OLLAMA_CHAT_MODEL, DEFAULT_OLLAMA_URL,
                                 ModelSettings, load_setting, resolve_models)
from trend_hunter.decisions import DecisionError, load_api_key
from trend_hunter.openai_generation import GenerationServiceError, _load_api_key


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="trend-config-tests-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "settings.env"

    def write(self, source):
        self.path.write_text(source, encoding="utf-8")

    def resolve(self, **kwargs):
        return resolve_models(dotenv_path=self.path, **kwargs)

    def test_defaults_are_frozen_and_credentials_do_not_select_a_provider(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-unused-key"}, clear=True):
            settings = self.resolve()
        self.assertEqual(settings, ModelSettings(DEFAULT_EMBEDDING_MODEL, DEFAULT_CHAT_PROVIDER,
                         DEFAULT_OLLAMA_CHAT_MODEL, DEFAULT_JEV_MODEL, DEFAULT_OLLAMA_URL))
        with self.assertRaises(FrozenInstanceError):
            settings.chat_model = "changed"

    def test_cli_then_environment_then_file_precedence_for_all_roles(self):
        self.write("TREND_CHAT_PROVIDER=ollama\nTREND_EMBEDDING_MODEL=file-encoder\n"
                   "TREND_OLLAMA_CHAT_MODEL=file-local\nTREND_OPENAI_CHAT_MODEL=file-hosted\n"
                   "TREND_JEV_MODEL=file-jev\nTREND_OLLAMA_URL=http://127.0.0.1:11555\n")
        env = {"TREND_CHAT_PROVIDER": "openai", "TREND_EMBEDDING_MODEL": "env-encoder",
               "TREND_OLLAMA_CHAT_MODEL": "env-local", "TREND_OPENAI_CHAT_MODEL": "env-hosted",
               "TREND_JEV_MODEL": "env-jev", "TREND_OLLAMA_URL": "http://127.0.0.1:11666"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(self.resolve(), ModelSettings("env-encoder", "openai", "env-hosted", "env-jev",
                                                          "http://127.0.0.1:11666"))
            self.assertEqual(self.resolve(embedding_model="cli-encoder", chat_provider="ollama",
                                         chat_model="cli-local", jev_model="cli-jev", ollama_url="http://127.0.0.1:11777"),
                             ModelSettings("cli-encoder", "ollama", "cli-local", "cli-jev", "http://127.0.0.1:11777"))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.resolve(), ModelSettings("file-encoder", "ollama", "file-local", "file-jev",
                                                          "http://127.0.0.1:11555"))

    def test_environment_aliases_all_precede_file_aliases_without_file_read(self):
        with patch.dict(os.environ, {"PRIMARY": "  ", "SECONDARY": " environment-value "}, clear=True), \
                patch.object(Path, "read_text", side_effect=AssertionError("Environment must win first")):
            self.assertEqual(load_setting("PRIMARY", "SECONDARY", dotenv_path=self.path), "environment-value")
        self.write("SECONDARY=second\nPRIMARY=first\n")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(load_setting("PRIMARY", "SECONDARY", dotenv_path=self.path), "first")
            self.assertEqual(load_setting("SECONDARY", "PRIMARY", dotenv_path=self.path), "second")

    def test_literals_comments_whitelist_and_environment_immutability(self):
        self.write("IGNORED='unterminated-secret\n"
                   "export FIRST='${SECRET}$(echo not-executed) # literal' # comment\n"
                   "SECOND=unquoted-value # trailing comment\nTHIRD=\" leading spaces \" # comment\n")
        with patch.dict(os.environ, {"SECRET": "not-expanded"}, clear=True):
            before = dict(os.environ)
            self.assertEqual(load_setting("FIRST", dotenv_path=self.path), "${SECRET}$(echo not-executed) # literal")
            self.assertEqual(load_setting("SECOND", dotenv_path=self.path), "unquoted-value")
            self.assertEqual(load_setting("THIRD", dotenv_path=self.path), " leading spaces ")
            self.assertEqual(dict(os.environ), before)

    def test_empty_values_and_missing_files_use_the_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(load_setting("ABSENT", dotenv_path=self.path))
            self.assertEqual(load_setting("ABSENT", default="fallback", dotenv_path=self.path), "fallback")
            self.write("EMPTY=\nEMPTY='   '\n")
            self.assertEqual(load_setting("EMPTY", default="fallback", dotenv_path=self.path), "fallback")

    def test_only_selected_provider_model_is_read_and_openai_needs_its_own_model(self):
        self.write("TREND_OLLAMA_CHAT_MODEL=file-local\nTREND_OPENAI_CHAT_MODEL='unterminated-secret\n")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.resolve().chat_model, "file-local")
            with self.assertRaisesRegex(ConfigurationError, "Invalid quoting"):
                self.resolve(chat_provider="openai")
        self.write("TREND_OLLAMA_CHAT_MODEL=not-a-hosted-model\n")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-unused-key"}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "requires an explicit"):
                self.resolve(chat_provider="openai")
            self.assertIsNone(self.resolve(chat_provider="openai", prepare_only=True).chat_model)
            self.assertEqual(self.resolve(chat_provider="openai", chat_model="explicit-hosted").chat_model,
                             "explicit-hosted")

    def test_invalid_explicit_values_fail_without_echoing_the_input(self):
        invalid = [("chat_provider", "private-provider"), ("chat_provider", []), ("chat_provider", ""),
                   ("embedding_model", ""), ("embedding_model", "private-model\nsource"),
                   ("chat_model", ""), ("chat_model", "private-model\x00source"),
                   ("jev_model", False), ("jev_model", []), ("ollama_url", "")]
        with patch.dict(os.environ, {}, clear=True):
            for field, value in invalid:
                with self.subTest(field=field, value=value), self.assertRaises(ConfigurationError) as caught:
                    self.resolve(**{field: value})
                self.assertNotIn("private-", str(caught.exception))
                self.assertNotIn("source", str(caught.exception))
        # URL syntax and loopback policy belong to the local adapter.
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.resolve(ollama_url="https://external.invalid").ollama_url, "https://external.invalid")

    def test_parser_errors_never_echo_names_values_paths_or_file_errors(self):
        with patch.dict(os.environ, {}, clear=True):
            for name in ("private name", "private\nname", None):
                with self.subTest(name=name), self.assertRaises(ConfigurationError) as caught:
                    load_setting(name, dotenv_path=self.path)
                self.assertNotIn("private", str(caught.exception))
            for source in ("KEY='private-secret", 'KEY="private-secret" trailing'):
                self.write(source)
                with self.assertRaises(ConfigurationError) as caught:
                    load_setting("KEY", dotenv_path=self.path)
                self.assertNotIn("private-secret", str(caught.exception))
            with patch.object(Path, "read_text", side_effect=PermissionError("private-path-secret")):
                with self.assertRaises(ConfigurationError) as caught:
                    load_setting("KEY", dotenv_path=self.path)
                self.assertNotIn("private", str(caught.exception))

    def test_provider_credential_error_types_messages_and_aliases_are_preserved(self):
        self.write("JEV_API_KEY='private-secret\nOPENAI_API_KEY='private-secret\n")
        with patch.dict(os.environ, {}, clear=True):
            for loader, error_type in ((load_api_key, DecisionError), (_load_api_key, GenerationServiceError)):
                with self.subTest(loader=loader.__name__), self.assertRaises(error_type) as caught:
                    loader(self.path)
                self.assertEqual(str(caught.exception), "Invalid quoting in the local credential file")
        with patch.dict(os.environ, {"TYPESAFE_API_TOKEN": "synthetic-env-alias"}, clear=True):
            self.assertEqual(load_api_key(self.path), "synthetic-env-alias")


if __name__ == "__main__":
    unittest.main()
