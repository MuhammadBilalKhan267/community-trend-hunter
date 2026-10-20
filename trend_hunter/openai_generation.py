"""Optional hosted drafting through the OpenAI Responses API.

Embedding and semantic decisions remain separate. Transport failures are not
retried: a timeout can leave a billable request with an unknown outcome.
"""
from __future__ import annotations

import os
from http.client import HTTPException
from pathlib import Path
import re
import sqlite3
import time
from urllib import error, request

from .local_models import _json, _key, _strict_json, _validate_schema, _NoRedirect


ENDPOINT = "https://api.openai.com/v1/responses"
CACHE_VERSION = "hosted-generation-v1"


class GenerationServiceError(RuntimeError):
    """The explicitly selected hosted generation service is unavailable."""


class GenerationOutputError(GenerationServiceError):
    """A completed request did not provide a usable structured draft."""


def _load_api_key(dotenv_path: str | Path) -> str:
    """Read one literal credential; environment wins and nothing is executed."""
    from .config import ConfigurationError, load_setting
    try:
        return load_setting("OPENAI_API_KEY", default="", dotenv_path=dotenv_path)
    except ConfigurationError as exc:
        raise GenerationServiceError(str(exc).replace("local settings file", "local credential file")) from None


class OpenAIGeneration:
    """Implement the pipeline's narrow chat_json drafting contract."""

    def __init__(self, *, model: str, cache_path: str | Path = ".cache/models.sqlite3",
                 api_key: str | None = None, dotenv_path: str | Path = ".env", timeout: float = 180):
        if not isinstance(model, str) or not model.strip() or any(c.isspace() for c in model):
            raise ValueError("An explicit OpenAI chat model is required")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("Generation timeout must be positive and finite")
        key = _load_api_key(dotenv_path) if api_key is None else api_key
        if not isinstance(key, str) or not key.strip() or any(c.isspace() for c in key.strip()):
            raise GenerationServiceError("Set OPENAI_API_KEY in the process environment or local .env file")
        self.model = model
        self._api_key = key.strip()
        self.timeout = timeout
        self._opener = request.build_opener(_NoRedirect())
        self.stats = {"requests": 0, "cache_hits": 0, "api_calls": 0, "failures": 0,
                      "api_seconds": 0.0, "input_tokens": 0, "output_tokens": 0,
                      "total_tokens": 0, "usage_unavailable_calls": 0}
        if str(cache_path) != ":memory:":
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        database = None
        try:
            database = sqlite3.connect(str(cache_path), timeout=30)
            database.execute("PRAGMA busy_timeout=30000")
            database.execute("CREATE TABLE IF NOT EXISTS openai_chats "
                             "(cache_key TEXT PRIMARY KEY, response_json TEXT NOT NULL)")
            database.commit()
        except sqlite3.Error:
            if database is not None:
                try:
                    database.close()
                except Exception:
                    # Cleanup must not replace the sanitized startup failure.
                    pass
            raise GenerationServiceError(
                "OpenAI generation cache could not be opened or initialized; no request was sent"
            ) from None
        self._db = database

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._db.close()

    def _transport(self, body: dict) -> dict:
        req = request.Request(ENDPOINT, method="POST", data=_json(body).encode("utf-8"),
                              headers={"Authorization": f"Bearer {self._api_key}",
                                       "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self._opener.open(req, timeout=self.timeout) as response:
                value = _strict_json(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            # Provider bodies can echo credentials or source text. Do not expose
            # their content, request headers, or exception chains.
            raise GenerationServiceError(f"OpenAI generation returned HTTP {exc.code}; no automatic retry was made") from None
        except (error.URLError, TimeoutError, OSError, HTTPException):
            raise GenerationServiceError(
                "OpenAI generation transport failed; delivery and billing may be unknown. "
                "No automatic retry or local fallback was made"
            ) from None
        except (ValueError, UnicodeError):
            raise GenerationServiceError("OpenAI generation returned invalid JSON; no automatic retry was made") from None
        if not isinstance(value, dict):
            raise GenerationServiceError("OpenAI generation returned a non-object response")
        return value

    def chat_json(self, system: str, payload: dict, schema: dict, max_tokens: int = 450) -> dict:
        if not isinstance(system, str) or not system.strip():
            raise ValueError("system must be a nonempty string")
        if not isinstance(payload, dict) or not isinstance(schema, dict):
            raise ValueError("payload and schema must be dictionaries")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        try:
            content = _json(payload)
            frozen_schema = _strict_json(_json(schema))
        except (ValueError, TypeError):
            raise ValueError("payload and schema must contain finite JSON-compatible values") from None
        body = {"model": self.model, "instructions": system, "input": content,
                "text": {"format": {"type": "json_schema", "name": "trend_draft",
                                    "schema": frozen_schema, "strict": True}},
                "max_output_tokens": max_tokens, "store": False}
        cache_key = _key([CACHE_VERSION, "openai", ENDPOINT, body])
        self.stats["requests"] += 1
        row = self._db.execute("SELECT response_json FROM openai_chats WHERE cache_key = ?", (cache_key,)).fetchone()
        if row:
            try:
                result = _strict_json(row[0])
                if not isinstance(result, dict):
                    raise ValueError("Draft must be an object")
                _validate_schema(result, schema)
            except (ValueError, TypeError):
                with self._db:
                    self._db.execute("DELETE FROM openai_chats WHERE cache_key = ?", (cache_key,))
            else:
                self.stats["cache_hits"] += 1
                return result
        self.stats["api_calls"] += 1
        started = time.perf_counter()
        usage_known = False
        try:
            response = self._transport(body)
            if not isinstance(response, dict):
                raise GenerationServiceError("OpenAI generation returned a non-object response")
            usage = response.get("usage")
            fields = ("input_tokens", "output_tokens", "total_tokens")
            if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in fields):
                for k in fields:
                    self.stats[k] += usage[k]
                usage_known = True
            if response.get("error"):
                raise GenerationServiceError("OpenAI reported a generation error")
            if response.get("status") != "completed":
                details = response.get("incomplete_details")
                if (response.get("status") == "incomplete" and isinstance(details, dict)
                        and details.get("reason") == "max_output_tokens"):
                    raise GenerationOutputError(
                        f"OpenAI draft reached max_output_tokens={max_tokens}; this limit includes any "
                        "reasoning tokens. No automatic retry was made"
                    )
                raise GenerationOutputError("OpenAI generation did not complete; no automatic retry was made")
            parts = []
            for item in response.get("output", []) if isinstance(response.get("output"), list) else []:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                if item.get("status", "completed") != "completed":
                    raise GenerationOutputError("OpenAI generation contained an incomplete message")
                for part in item.get("content", []) if isinstance(item.get("content"), list) else []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "refusal":
                        raise GenerationOutputError("OpenAI declined this generation request")
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        parts.append(part["text"])
            try:
                result = _strict_json("".join(parts))
                if not isinstance(result, dict):
                    raise ValueError("Draft must be an object")
                _validate_schema(result, schema)
            except (ValueError, TypeError):
                raise GenerationOutputError("OpenAI draft failed the structured output contract") from None
            with self._db:
                self._db.execute("INSERT OR REPLACE INTO openai_chats VALUES (?, ?)", (cache_key, _json(result)))
            return result
        except GenerationServiceError:
            self.stats["failures"] += 1
            raise
        finally:
            self.stats["api_seconds"] += time.perf_counter() - started
            if not usage_known:
                self.stats["usage_unavailable_calls"] += 1
