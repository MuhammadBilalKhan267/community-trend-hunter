"""Literal local settings with explicit model-provider precedence."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_OLLAMA_CHAT_MODEL = "qwen3:8b"
DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_CHAT_PROVIDER = "ollama"


class ConfigurationError(ValueError):
    """A local setting cannot be read or does not satisfy its contract."""


def load_setting(*names: str, default=None, dotenv_path: str | Path = ".env"):
    """Use nonempty environment aliases, then literal file aliases, then default.

    Alias order determines precedence within each source. Only requested names
    are parsed. Quotes and comments are supported; expansion and execution are
    not. The process environment is never modified.
    """
    if not names or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                        for name in names):
        raise ConfigurationError("Setting names must be valid environment-variable identifiers")
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        source = Path(dotenv_path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return default
    except (OSError, UnicodeError, TypeError, ValueError):
        raise ConfigurationError("Unable to read the local settings file") from None
    found = {}
    for line in source.splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if not match or match.group(1) not in names:
            continue
        name, value = match.groups()
        if value.startswith(("'", '"')):
            end = value.find(value[0], 1)
            remainder = value[end + 1:].strip() if end >= 0 else ""
            if end < 0 or (remainder and not remainder.startswith("#")):
                raise ConfigurationError("Invalid quoting in the local settings file")
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if value.strip():
            found[name] = value
    return next((found[name] for name in names if name in found), default)


@dataclass(frozen=True)
class ModelSettings:
    embedding_model: str
    chat_provider: str
    chat_model: str | None
    jev_model: str
    ollama_url: str


def _model_name(value, label: str) -> str:
    if (not isinstance(value, str) or not value
            or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)):
        raise ConfigurationError(f"{label} must be a nonempty model identifier without whitespace or control characters")
    return value


def resolve_models(*, embedding_model=None, chat_provider=None, chat_model=None,
                   jev_model=None, ollama_url=None, prepare_only=False,
                   dotenv_path: str | Path = ".env") -> ModelSettings:
    """Resolve CLI > environment > literal file > defaults for each selected role."""
    def setting(explicit, name, fallback):
        return explicit if explicit is not None else load_setting(name, default=fallback, dotenv_path=dotenv_path)

    provider = setting(chat_provider, "TREND_CHAT_PROVIDER", DEFAULT_CHAT_PROVIDER)
    if not isinstance(provider, str) or provider not in {"ollama", "openai"}:
        raise ConfigurationError("chat_provider must be ollama or openai")
    if provider == "ollama":
        chat = setting(chat_model, "TREND_OLLAMA_CHAT_MODEL", DEFAULT_OLLAMA_CHAT_MODEL)
    else:
        chat = setting(chat_model, "TREND_OPENAI_CHAT_MODEL", None)
        if chat is None and not prepare_only:
            raise ConfigurationError("OpenAI generation requires an explicit chat_model or TREND_OPENAI_CHAT_MODEL")
    if chat is not None:
        chat = _model_name(chat, "chat_model")
    embedding = _model_name(setting(embedding_model, "TREND_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                            "embedding_model")
    jev = _model_name(setting(jev_model, "TREND_JEV_MODEL", DEFAULT_JEV_MODEL), "jev_model")
    url = setting(ollama_url, "TREND_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    if not isinstance(url, str) or not url.strip():
        raise ConfigurationError("ollama_url must be a nonempty string")
    # The local adapter owns URL parsing and the loopback-only trust boundary.
    return ModelSettings(embedding, provider, chat, jev, url)
