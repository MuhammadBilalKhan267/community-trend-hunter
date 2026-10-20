"""Typed Jev decisions, exact-input caching and persistent token accounting.

Reservations conservatively account for repeated state across independent
questions; actual reported usage replaces the reservation. Unknown-cost failures
retain their reservation. Provider context limits are separate from this ledger.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any
from urllib import error, request

from .config import ConfigurationError, DEFAULT_JEV_MODEL, load_setting


API_ORIGIN = "https://api.typesafe.ai"
CACHE_VERSION = "jev-decisions-v1"
_KEY_NAMES = ("JEV_API_KEY", "TYPESAFE_API_KEY", "TYPESAFE_API_TOKEN")
_MAX_BODY_BYTES = 1_048_576
_MAX_RESPONSE_BYTES = 2_097_152
_MAX_ERROR_BYTES = 16_384
_CONTEXT_LIMIT_CODE = "max_tokens_exceeded"
_LONG_CONTEXT_HEADER = "TypeSafe-Danger-Zone-Long-Context"


class DecisionError(RuntimeError):
    """A typed decision could not be obtained or validated."""


class BudgetExceeded(DecisionError):
    """A request would exceed the configured cumulative token budget."""


class _HTTPFailure(DecisionError):
    def __init__(self, status: int, *, code: str | None = None):
        # Retain only a documented allowlisted code, never provider text that
        # could echo credentials or submitted evidence.
        self.code = _CONTEXT_LIMIT_CODE if code == _CONTEXT_LIMIT_CODE else None
        detail = f" ({self.code})" if self.code else ""
        super().__init__(f"Jev returned HTTP {status}{detail}; response details omitted")
        self.status = status


def _context_error_code(raw: bytes) -> str | None:
    """Recognize an exact structured provider code, without message matching."""
    if len(raw) > _MAX_ERROR_BYTES:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        return None
    if not isinstance(value, dict):
        return None
    codes = [value.get("code"), value.get("error")]
    nested = value.get("error")
    if isinstance(nested, dict):
        codes.extend((nested.get("code"), nested.get("type")))
    detail = value.get("detail")
    if isinstance(detail, dict):
        codes.append(detail.get("error_type"))
    return _CONTEXT_LIMIT_CODE if any(code == _CONTEXT_LIMIT_CODE for code in codes) else None


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a bearer token, including to another provider endpoint.
        return None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def load_api_key(dotenv_path: str | Path = ".env") -> str:
    """Read a literal supported credential without changing the environment."""
    try:
        return load_setting(*_KEY_NAMES, default="", dotenv_path=dotenv_path)
    except ConfigurationError as exc:
        raise DecisionError(str(exc).replace("local settings file", "local credential file")) from None


def noul_question(instructions: Any, *, true: Any = None, false: Any = None) -> dict:
    question = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        question["criteria"] = {"true": true, "false": false}
    return question


def choice_question(instructions: Any, criteria: dict[str, Any]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


class JevDecisions:
    """One decision request per candidate, with several independent questions.

    Token limits apply to this cache ledger, across adapter instances. They are
    not account-wide quotas and include unknown-cost attempted requests.
    """

    def __init__(
        self,
        cache_path: str | Path,
        api_key: str | None = None,
        model: str = DEFAULT_JEV_MODEL,
        target_tokens: int = 20_000_000,
        max_tokens: int = 50_000_000,
        timeout: float = 60,
        retries: int = 1,
        dotenv_path: str | Path = ".env",
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty name")
        for name, value in (("target_tokens", target_tokens), ("max_tokens", max_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if target_tokens > max_tokens:
            raise ValueError("target_tokens must not exceed max_tokens")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 3:
            raise ValueError("retries must be an integer between zero and three")
        self._api_key = load_api_key(dotenv_path) if api_key is None else api_key.strip()
        if not self._api_key or any(char.isspace() for char in self._api_key):
            raise DecisionError("Set JEV_API_KEY or TYPESAFE_API_KEY in the environment or local .env")
        self.model = model.strip()
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.timeout = float(timeout)
        self.retries = retries
        self._model_identity = None
        self.resolved_model = None
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS jev_decision_cache (
                cache_key TEXT PRIMARY KEY,
                requested_model TEXT NOT NULL,
                model_identity TEXT NOT NULL,
                resolved_model TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jev_usage_requests (
                id INTEGER PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                started_at TEXT NOT NULL,
                status TEXT NOT NULL,
                reserved_tokens INTEGER NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                resolved_model TEXT
            );
        """)
        self._db.commit()
        self.stats = {"requests": 0, "cache_hits": 0, "http_calls": 0, "model_calls": 0, "retries": 0, "failures": 0,
                      "context_fallbacks": 0, "long_context_calls": 0, "long_context_cache_hits": 0}

    @property
    def usage(self) -> dict:
        row = self._db.execute("""
            SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                   COALESCE(SUM(CASE WHEN input_tokens IS NULL THEN reserved_tokens ELSE 0 END),0),
                   COUNT(*)
            FROM jev_usage_requests
        """).fetchone()
        input_tokens, output_tokens, reserved_tokens, attempts = map(int, row)
        total = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "reserved_tokens": reserved_tokens,
            "accounted_tokens": total + reserved_tokens,
            "attempts": attempts,
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "target_exceeded": total + reserved_tokens >= self.target_tokens,
            "remaining_tokens": max(0, self.max_tokens - total - reserved_tokens),
        }

    def _transport(self, method: str, path: str, payload: dict | None, *, long_context: bool = False) -> dict:
        if path not in {"/v1/models", "/v1/systemone"}:
            raise DecisionError("Unsupported Jev endpoint")
        body = None if payload is None else _json(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json", "Accept": "application/json"}
        if long_context and method == "POST" and path == "/v1/systemone":
            headers[_LONG_CONTEXT_HEADER] = "true"
        req = request.Request(
            API_ORIGIN + path,
            data=body,
            method=method,
            headers=headers,
        )
        # Do not send bearer credentials through ambient proxy configuration.
        opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
        self.stats["http_calls"] += 1
        try:
            with opener.open(req, timeout=self.timeout) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except error.HTTPError as exc:
            code = None
            try:
                if exc.code == 400:
                    code = _context_error_code(exc.read(_MAX_ERROR_BYTES + 1))
            except (OSError, ValueError):
                pass
            finally:
                exc.close()
            raise _HTTPFailure(exc.code, code=code) from None
        except (error.URLError, TimeoutError, OSError):
            raise DecisionError("Jev connection failed or timed out; request details omitted") from None
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise DecisionError("Jev response exceeded the permitted size")
        try:
            result = json.loads(raw.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, UnicodeError):
            raise DecisionError("Jev returned invalid JSON; response details omitted") from None
        if not isinstance(result, dict):
            raise DecisionError("Jev response must be an object")
        return result

    def _resolve_model(self) -> str:
        if self._model_identity is not None:
            return self._model_identity
        response = self._transport("GET", "/v1/models", None)
        models = response.get("models")
        if not isinstance(models, list):
            raise DecisionError("Jev model metadata is missing")
        entry = next((item for item in models if isinstance(item, dict) and item.get("name") == self.model), None)
        if entry is None:
            raise DecisionError("The requested Jev model is not available to this account")
        release = entry.get("release_date")
        if not isinstance(release, str) or not release:
            raise DecisionError("Jev model release identity is missing")
        self._model_identity = _digest({"name": self.model, "release_date": release})
        return self._model_identity

    @staticmethod
    def _validate_questions(questions: Any) -> None:
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty named dictionary")
        for name, question in questions.items():
            if not isinstance(name, str) or not name or not isinstance(question, dict):
                raise ValueError("Each question must have a name and object")
            if question.get("type") not in {"noul", "choice"}:
                raise ValueError("Only Noul and Choice questions are supported")
            if "instructions" not in question:
                raise ValueError("Each question requires instructions")
            if question["type"] == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255 or any(not isinstance(key, str) or not key for key in criteria):
                    raise ValueError("Choice criteria must contain one to 255 named choices")

    @staticmethod
    def _validate_answers(response: dict, questions: dict) -> dict:
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise DecisionError("Jev answer names do not match the requested questions")
        for name, question in questions.items():
            answer = answers[name]
            if not isinstance(answer, dict) or answer.get("type") != question["type"]:
                raise DecisionError("Jev answer type does not match its question")
            if question["type"] == "noul":
                if not _probability(answer.get("noul")):
                    raise DecisionError("Jev returned an invalid Noul probability")
            else:
                options = question["criteria"]
                probabilities = answer.get("probabilities")
                if answer.get("choice") not in options or not _probability(answer.get("confidence")):
                    raise DecisionError("Jev returned an invalid Choice answer")
                if not isinstance(probabilities, dict) or set(probabilities) != set(options) or not all(_probability(value) for value in probabilities.values()):
                    raise DecisionError("Jev returned invalid choice probabilities")
                if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.03):
                    raise DecisionError("Jev choice probabilities do not sum to one")
        return answers

    @staticmethod
    def _reservation(payload: dict) -> int:
        # State is evaluated independently for every question. UTF-8 bytes are
        # conservative versus ordinary text tokenization; allowance includes
        # provider framing and structured output, which are not documented.
        body_bytes = len(_json(payload).encode("utf-8"))
        return (body_bytes + 16_384) * len(payload["questions"])

    def _reserve(self, fingerprint: str, tokens: int) -> int:
        try:
            self._db.execute("BEGIN IMMEDIATE")
            if self.usage["accounted_tokens"] + tokens > self.max_tokens:
                raise BudgetExceeded("Jev token budget cannot accommodate this request")
            cursor = self._db.execute(
                "INSERT INTO jev_usage_requests(fingerprint,started_at,status,reserved_tokens) VALUES(?,?,?,?)",
                (fingerprint, datetime.now(timezone.utc).isoformat(), "reserved", tokens),
            )
            self._db.commit()
            return int(cursor.lastrowid)
        except Exception:
            self._db.rollback()
            raise

    def _record(self, request_id: int, status: str, response: dict | None = None, *, known_free: bool = False) -> None:
        usage = response.get("usage") if isinstance(response, dict) else None
        input_tokens = output_tokens = None
        if known_free:
            input_tokens = output_tokens = 0
        elif isinstance(usage, dict):
            values = (usage.get("input_tokens"), usage.get("output_tokens"))
            if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values):
                input_tokens, output_tokens = values
        model = response.get("model") if isinstance(response, dict) else None
        self._db.execute(
            "UPDATE jev_usage_requests SET status=?,input_tokens=?,output_tokens=?,resolved_model=? WHERE id=?",
            (status, input_tokens, output_tokens, model if isinstance(model, str) else None, request_id),
        )
        self._db.commit()

    def ask(self, state: Any, questions: dict[str, dict]) -> dict:
        self._validate_questions(questions)
        if not isinstance(state, (str, dict, list)):
            raise ValueError("state must be a string, object or array")
        payload = {"model": self.model, "state": state, "questions": questions}
        try:
            body_bytes = len(_json(payload).encode("utf-8"))
        except (TypeError, ValueError):
            raise ValueError("state and questions must contain finite JSON values") from None
        if body_bytes > _MAX_BODY_BYTES:
            raise ValueError("Jev request exceeded the permitted size")
        identity = self._resolve_model()
        # Keep the original default-mode key intact for existing cache entries.
        cache_keys = {
            False: _digest({"version": CACHE_VERSION, "identity": identity, "payload": payload}),
            True: _digest({"version": CACHE_VERSION, "identity": identity, "payload": payload,
                           "context_mode": "long"}),
        }
        self.stats["requests"] += 1
        for long_context, cache_key in cache_keys.items():
            cached = self._db.execute("SELECT response_json FROM jev_decision_cache WHERE cache_key=?", (cache_key,)).fetchone()
            if cached:
                try:
                    response = json.loads(cached[0])
                    if not isinstance(response, dict):
                        raise DecisionError("Cached Jev response must be an object")
                    answers = self._validate_answers(response, questions)
                    if not isinstance(response.get("model"), str) or not response["model"]:
                        raise DecisionError("Cached Jev response has no model identity")
                except (ValueError, DecisionError):
                    self._db.execute("DELETE FROM jev_decision_cache WHERE cache_key=?", (cache_key,))
                    self._db.commit()
                else:
                    self.resolved_model = response["model"]
                    self.stats["cache_hits"] += 1
                    self.stats["long_context_cache_hits"] += int(long_context)
                    return answers
        reservation = self._reservation(payload)
        long_context = False
        context_fallback_pending = False
        transient_retries = 0
        # One mode transition is independent of the shared transient retry
        # allowance: at most retries + 2 POSTs, with unchanged payload bytes.
        while True:
            cache_key = cache_keys[long_context]
            request_id = self._reserve(cache_key, reservation)
            self.stats["model_calls"] += 1
            if long_context:
                self.stats["long_context_calls"] += 1
                self.stats["context_fallbacks"] += int(context_fallback_pending)
                context_fallback_pending = False
            try:
                if long_context:
                    response = self._transport("POST", "/v1/systemone", payload, long_context=True)
                else:
                    response = self._transport("POST", "/v1/systemone", payload)
            except DecisionError as exc:
                status = exc.status if isinstance(exc, _HTTPFailure) else None
                self._record(request_id, "failed", known_free=status in {400, 401, 403, 404, 413, 422, 429})
                if (status == 400 and isinstance(exc, _HTTPFailure)
                        and exc.code == _CONTEXT_LIMIT_CODE and not long_context):
                    long_context = True
                    context_fallback_pending = True
                    continue
                retryable = status is None or status == 429 or 500 <= status < 600
                if retryable and transient_retries < self.retries:
                    self.stats["retries"] += 1
                    time.sleep(min(2 ** transient_retries, 4))
                    transient_retries += 1
                    continue
                self.stats["failures"] += 1
                raise
            self._record(request_id, "received", response)
            model = response.get("model")
            if not isinstance(model, str) or not model:
                raise DecisionError("Jev response has no resolved model identity")
            if not isinstance(response.get("usage"), dict) or self._db.execute(
                "SELECT input_tokens FROM jev_usage_requests WHERE id=?", (request_id,)
            ).fetchone()[0] is None:
                raise DecisionError("Jev response has no valid token usage")
            answers = self._validate_answers(response, questions)
            if self.usage["accounted_tokens"] > self.max_tokens:
                raise BudgetExceeded("Reported Jev usage exceeded the remaining token allowance; further calls stopped")
            self.resolved_model = model
            self._db.execute(
                "INSERT OR REPLACE INTO jev_decision_cache VALUES(?,?,?,?,?,?)",
                (cache_key, self.model, identity, model, _json(response), datetime.now(timezone.utc).isoformat()),
            )
            self._db.execute("UPDATE jev_usage_requests SET status='success' WHERE id=?", (request_id,))
            self._db.commit()
            return answers
    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
