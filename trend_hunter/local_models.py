"""Loopback-only Ollama inference with content-addressed local caches."""

from __future__ import annotations

import hashlib
import ipaddress
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any
from urllib import error, parse, request

import numpy as np

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_CHAT_MODEL, DEFAULT_OLLAMA_URL
from .embedding_identity import (
    CACHE_VERSION,
    cache_embedding_options,
    canonical_model_name,
    checked_profile,
    embedding_key,
    read_profiles,
    save_profile,
)


class LocalModelError(RuntimeError):
    """A local model is unavailable or returned an unusable response."""

    def __init__(self, message: str, *, status_code: int | None = None, retryable_embedding: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable_embedding = retryable_embedding


class LocalGenerationError(LocalModelError):
    """The service responded, but this particular draft could not satisfy its schema."""


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A local endpoint must never redirect a request to an external service.
        return None


def _local_url(value: str) -> str:
    if not isinstance(value, str) or any(char.isspace() for char in value):
        raise ValueError("base_url must be an HTTP(S) loopback URL without whitespace")
    try:
        url = parse.urlsplit(value)
        host = url.hostname
        port = url.port
    except ValueError as exc:
        raise ValueError("base_url is not a valid loopback URL") from exc
    if (
        url.scheme not in {"http", "https"}
        or not host
        or url.username is not None
        or url.password is not None
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("base_url must contain only an HTTP(S) loopback host and optional port")
    # Resolve this well-known alias ourselves, avoiding a DNS/proxy trust boundary.
    if host.lower() == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("Only literal loopback addresses or localhost are allowed") from exc
    if not address.is_loopback:
        raise ValueError("External model services are disabled; use a loopback address")
    authority = f"[{address}]" if address.version == 6 else str(address)
    if port is not None:
        authority += f":{port}"
    return f"{url.scheme}://{authority}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _key(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _strict_json(value: str) -> Any:
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = item
        return result

    def constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def _server_detail(body: bytes) -> str:
    """Expose a bounded JSON error diagnostic, never arbitrary echoed bodies."""
    try:
        decoded = _strict_json(body.decode("utf-8"))
    except (ValueError, UnicodeError):
        return ""
    detail = decoded.get("error") if isinstance(decoded, dict) else None
    if not isinstance(detail, str):
        return ""
    detail = re.sub(r"[\x00-\x1f\x7f]+", " ", detail)
    detail = re.sub(r"(?i)(://)[^/\s@]+@", r"\1[credentials omitted]@", detail)
    detail = re.sub(
        r"(?i)\b(authorization|api[-_ ]?key|access[-_ ]?token|password|secret)\s*[:=]\s*(?:bearer\s+)?[^\s,;]+",
        r"\1=[redacted]", detail,
    )
    detail = re.sub(r"(?i)\b(prompt|input|content|text)\s*[:=].*", r"\1=[source omitted]", detail)
    return re.sub(r"\s+", " ", detail).strip()[:240]


def _validate_schema(value: Any, schema: dict, path: str = "$") -> None:
    """Check the inline JSON Schema subset used by the candidate-audit contract."""
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: schema must be an object")
    if "$ref" in schema:
        raise ValueError("Local chat schemas must be inline; $ref is unsupported")
    for keyword in ("allOf", "anyOf", "oneOf"):
        if keyword not in schema:
            continue
        matches = 0
        alternatives = schema[keyword]
        for alternative in alternatives:
            try:
                _validate_schema(value, alternative, path)
                matches += 1
            except ValueError:
                pass
        if (keyword == "allOf" and matches != len(alternatives)) or (
            keyword == "anyOf" and matches == 0
        ) or (keyword == "oneOf" and matches != 1):
            raise ValueError(f"{path}: failed {keyword}")
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected]
    kinds = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": value is None,
    }
    if expected is not None and not any(kinds.get(kind, False) for kind in types):
        raise ValueError(f"{path}: expected {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is outside enum")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: value differs from const")
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ValueError(f"{path}: missing required properties {sorted(missing)}")
        properties = schema.get("properties", {})
        for name, item in value.items():
            if name in properties:
                _validate_schema(item, properties[name], f"{path}.{name}")
            elif schema.get("additionalProperties") is False:
                raise ValueError(f"{path}: unexpected property {name}")
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate_schema(item, schema["additionalProperties"], f"{path}.{name}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            raise ValueError(f"{path}: invalid array length")
        for index, item in enumerate(value):
            if "items" in schema:
                _validate_schema(item, schema["items"], f"{path}[{index}]")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            raise ValueError(f"{path}: invalid string length")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ValueError(f"{path}: string does not match pattern")
    if kinds["number"]:
        if value < schema.get("minimum", -float("inf")) or value > schema.get("maximum", float("inf")):
            raise ValueError(f"{path}: number outside bounds")


class LocalModels:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chat_model: str = DEFAULT_OLLAMA_CHAT_MODEL,
        cache_path: str | Path = ".cache/models.sqlite3",
        batch_size: int = 16,
        timeout: float = 180,
        embedding_options: dict | None = None,
        chat_options: dict | None = None,
        keep_alive: str | int = "5m",
        embedding_cache_only: bool = False,
    ):
        if not isinstance(embedding_cache_only, bool):
            raise ValueError("embedding_cache_only must be a boolean")
        self.embedding_cache_only = embedding_cache_only
        self.base_url = base_url if embedding_cache_only else _local_url(base_url)
        for name, value in (("embedding_model", embedding_model), ("chat_model", chat_model)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty installed model name")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if embedding_options is not None and not isinstance(embedding_options, dict):
            raise ValueError("embedding_options must be a JSON-compatible dictionary")
        if chat_options is not None and not isinstance(chat_options, dict):
            raise ValueError("chat_options must be a JSON-compatible dictionary")
        for name, value in (("embedding_options", embedding_options), ("chat_options", chat_options)):
            if any(not isinstance(key, str) or not key for key in (value or {})):
                raise ValueError(f"{name} keys must be nonempty strings")
        try:
            options = _strict_json(_json(embedding_options or {}))
            generation_options = _strict_json(_json(chat_options or {}))
        except (ValueError, TypeError) as exc:
            raise ValueError("Model options must contain finite JSON-compatible values") from exc
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.batch_size = batch_size
        self.timeout = timeout
        self.embedding_options = {"num_ctx": 512, "num_batch": 512, **options}
        self.chat_options = generation_options
        self.keep_alive = keep_alive
        self._opener = None if embedding_cache_only else request.build_opener(request.ProxyHandler({}), _NoRedirect())
        self._resolved: dict[str, dict] = {}
        self._dimensions: dict[str, int] = {}
        self.embedding_profile: dict | None = None
        self.stats = {
            "api_calls": 0,
            "health_calls": 0,
            "embedding_calls": 0,
            "embedding_retries": 0,
            "embedding_batch_splits": 0,
            "chat_calls": 0,
            "unload_calls": 0,
            "embedding_cache_hits": 0,
            "chat_cache_hits": 0,
            "api_seconds": 0.0,
            "chat_prompt_tokens": 0,
            "chat_output_tokens": 0,
        }
        if embedding_cache_only:
            try:
                self._db = sqlite3.connect(Path(cache_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
            except sqlite3.Error as exc:
                raise LocalModelError(
                    "The embedding cache cannot be opened read-only. Restore the embedding bundle with "
                    "scripts/embedding_cache.py restore, or ingest embeddings explicitly with "
                    "scripts/ingest_embeddings.py; no Ollama connection or inference was attempted."
                ) from exc
            return
        if str(cache_path) != ":memory:":
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path), timeout=30)
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(cache_key TEXT PRIMARY KEY, dimension INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS chats (cache_key TEXT PRIMARY KEY, response_json TEXT NOT NULL)"
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _transport(self, method: str, path: str, payload: dict | None) -> dict:
        data = _json(payload).encode("utf-8") if payload is not None else None
        req = request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with self._opener.open(req, timeout=self.timeout) as response:
                result = _strict_json(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            try:
                detail = _server_detail(exc.read(2048))
            except OSError:
                detail = ""
            diagnostic = f" Server detail: {detail}." if detail else ""
            raise LocalModelError(
                f"Local Ollama returned HTTP {exc.code} for {path}.{diagnostic} Check installed models, "
                "available memory, and input context length. No remote fallback was used.",
                status_code=exc.code,
                retryable_embedding=path == "/api/embed" and exc.code == 500,
            ) from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise LocalModelError(
                f"Cannot reach local Ollama at {self.base_url}; start the local service "
                "and confirm its loopback address. No model was downloaded or remote service used."
            ) from exc
        except (ValueError, UnicodeError) as exc:
            raise LocalModelError(
                f"Local Ollama returned invalid JSON for {path}", retryable_embedding=path == "/api/embed"
            ) from exc
        if not isinstance(result, dict):
            raise LocalModelError(
                f"Local Ollama returned a non-object response for {path}", retryable_embedding=path == "/api/embed"
            )
        if result.get("error"):
            raise LocalModelError(
                f"Local Ollama reported an inference error for {path}; check its local logs, "
                "available memory, and context length."
            )
        return result

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        if self.embedding_cache_only:
            raise LocalModelError("Ollama requests are disabled in embedding-cache-only mode")
        self.stats["api_calls"] += 1
        counter = {
            "/api/tags": "health_calls",
            "/api/embed": "embedding_calls",
            "/api/chat": "chat_calls",
            "/api/generate": "unload_calls",
        }.get(path)
        if counter:
            self.stats[counter] += 1
        started = time.perf_counter()
        try:
            return self._transport(method, path, payload)
        finally:
            self.stats["api_seconds"] += time.perf_counter() - started

    def health(self, required_roles: tuple[str, ...] = ("embedding", "chat")) -> dict:
        if not isinstance(required_roles, tuple) or any(role not in {"embedding", "chat"} for role in required_roles):
            raise ValueError("required_roles must be a tuple containing only 'embedding' and/or 'chat'")
        response = self._request("GET", "/api/tags")
        models = response.get("models")
        if not isinstance(models, list) or any(not isinstance(model, dict) for model in models):
            raise LocalModelError("Local Ollama /api/tags did not return an installed-model list")
        installed = {}
        for model in models:
            name = model.get("name") or model.get("model")
            if not isinstance(name, str) or not name:
                raise LocalModelError("Local Ollama /api/tags contains an invalid model name")
            installed[name] = model
        resolved, role_models = {}, {}
        for role in required_roles:
            requested = self.embedding_model if role == "embedding" else self.chat_model
            name = requested
            if name not in installed and ":" not in requested.rsplit("/", 1)[-1]:
                name = requested + ":latest"
            details = installed.get(name)
            if not details:
                names = ", ".join(sorted(str(name) for name in installed)) or "none"
                raise LocalModelError(
                    f"Required local model {requested!r} is not installed. Installed: {names}. "
                    "Select an installed model or arrange local model setup explicitly; no download was attempted."
                )
            if not isinstance(details.get("digest"), str) or not details["digest"]:
                raise LocalModelError(f"Installed model {name!r} has no digest; safe cache reuse is unavailable")
            resolved[requested] = dict(details, name=name)
            role_models[role] = resolved[requested]
        self._resolved = resolved
        return {
            "base_url": self.base_url,
            "models": models,
            "embedding_model": role_models.get("embedding"),
            "chat_model": role_models.get("chat"),
        }

    @staticmethod
    def _vector(value: Any, dimension: int | None = None) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise LocalModelError("Local embedding contains non-numeric values") from exc
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise LocalModelError("Local embedding must be a nonempty, finite vector")
        if dimension is not None and vector.size != dimension:
            raise LocalModelError("Local embedding dimensions changed within one model digest")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
            raise LocalModelError("Local embedding has a zero or invalid norm")
        return (vector / norm).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embed expects a list of nonempty strings; split or exclude empty messages first")
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.embedding_cache_only:
            return self._cached_embeddings(texts)
        self.health(required_roles=("embedding",))
        model = self._resolved[self.embedding_model]
        digest = model["digest"]
        # Snapshot overrides so the key describes precisely the request options.
        options = _strict_json(_json(self.embedding_options))
        dimension = self._dimensions.get(digest)
        entries: dict[str, dict] = {}
        for index, text in enumerate(texts):
            # Device placement does not change the encoder or input semantics.
            # Reuse existing vectors when moving an ingestion between CPU/GPU.
            cache_key = embedding_key(model["name"], digest, text, options)
            entry = entries.setdefault(cache_key, {"text": text, "indices": []})
            entry["indices"].append(index)
        output: list[np.ndarray | None] = [None] * len(texts)
        pending = []
        for cache_key, entry in entries.items():
            row = self._db.execute(
                "SELECT dimension, vector FROM embeddings WHERE cache_key = ?", (cache_key,)
            ).fetchone()
            if row is None:
                pending.append((cache_key, entry))
                continue
            try:
                raw = np.frombuffer(row[1], dtype="<f4")
                if raw.size != row[0]:
                    raise LocalModelError("Cached embedding size is invalid")
                self._vector(raw, dimension)
                if not np.isclose(np.linalg.norm(raw.astype(np.float64)), 1.0, rtol=1e-5, atol=1e-7):
                    raise LocalModelError("Cached embedding is not normalized")
                # Preserve the original float32 values exactly across cached runs.
                vector = raw.copy()
            except (ValueError, LocalModelError):
                self._db.execute("DELETE FROM embeddings WHERE cache_key = ?", (cache_key,))
                pending.append((cache_key, entry))
                continue
            dimension = len(vector)
            self.stats["embedding_cache_hits"] += len(entry["indices"])
            for index in entry["indices"]:
                output[index] = vector
        def process_batch(batch, *, retry=False):
            nonlocal dimension
            if retry:
                self.stats["embedding_retries"] += 1
            try:
                response = self._request(
                    "POST",
                    "/api/embed",
                    {
                        "model": model["name"],
                        "input": [entry["text"] for _, entry in batch],
                        "truncate": False,
                        "options": options,
                        "keep_alive": self.keep_alive,
                    },
                )
                try:
                    vectors = response.get("embeddings")
                    if not isinstance(vectors, list) or len(vectors) != len(batch):
                        raise LocalModelError("Local embedding response count does not match its input batch")
                    validated = []
                    batch_dimension = dimension
                    for value in vectors:
                        vector = self._vector(value, batch_dimension)
                        batch_dimension = len(vector)
                        validated.append(vector)
                except LocalModelError as exc:
                    exc.retryable_embedding = True
                    raise
            except LocalModelError as exc:
                if not (exc.retryable_embedding or exc.status_code == 500):
                    raise
                if len(batch) == 1:
                    identifier = batch[0][0][:12]
                    raise LocalModelError(
                        f"Local embedding failed for a single input (cache key {identifier}) after adaptive "
                        f"batch reduction. {exc} Inspect that input's length and local model logs; "
                        "no input was skipped and no vector was invented.",
                        status_code=exc.status_code,
                    ) from exc
                self.stats["embedding_batch_splits"] += 1
                midpoint = len(batch) // 2
                process_batch(batch[:midpoint], retry=True)
                process_batch(batch[midpoint:], retry=True)
                return
            dimension = batch_dimension
            with self._db:
                profile = checked_profile({"model_name": model["name"], "digest": digest,
                                           "options": options, "dimensions": dimension})
                save_profile(self._db, profile, replace=True)
                for (cache_key, entry), vector in zip(batch, validated):
                    self._db.execute(
                        "INSERT OR REPLACE INTO embeddings (cache_key, dimension, vector) VALUES (?, ?, ?)",
                        (cache_key, len(vector), vector.astype("<f4").tobytes()),
                    )
                    for index in entry["indices"]:
                        output[index] = vector
            # Successful sub-batches survive a later input failure and can resume.
            self._dimensions[digest] = dimension
            self.embedding_profile = profile

        for start in range(0, len(pending), self.batch_size):
            process_batch(pending[start : start + self.batch_size])
        with self._db:
            profile = checked_profile({"model_name": model["name"], "digest": digest,
                                       "options": options, "dimensions": dimension})
            save_profile(self._db, profile, replace=True)
        self.embedding_profile = profile
        self._dimensions[digest] = dimension
        return np.stack(output).astype(np.float32, copy=False)

    def _cached_embeddings(self, texts: list[str]) -> np.ndarray:
        remediation = (
            "Restore a matching embedding bundle with scripts/embedding_cache.py restore, or ingest "
            "embeddings explicitly with scripts/ingest_embeddings.py. "
            "No cache rows were changed and no Ollama connection or inference was attempted."
        )
        try:
            name = canonical_model_name(self.embedding_model)
            options = cache_embedding_options(self.embedding_options)
            profiles = read_profiles(self._db)
            profile = next((item for item in profiles
                            if item["model_name"] == name and _json(item["options"]) == _json(options)), None)
            if profile is None:
                raise LocalModelError("No matching saved embedding model/options profile. " + remediation)
            if self.embedding_profile is not None and profile != self.embedding_profile:
                raise LocalModelError("The saved embedding profile changed during this run. " + remediation)
            dimension = profile["dimensions"]
            entries = {}
            for index, text in enumerate(texts):
                key = embedding_key(name, profile["digest"], text, options)
                entries.setdefault(key, []).append(index)
            output = [None] * len(texts)
            missing = invalid = 0
            for key, indices in entries.items():
                row = self._db.execute(
                    "SELECT dimension, vector FROM embeddings WHERE cache_key = ?", (key,)
                ).fetchone()
                if row is None:
                    missing += 1
                    continue
                try:
                    if type(row[0]) is not int or row[0] != dimension:
                        raise ValueError("Cached vector dimensions do not match the saved profile")
                    raw = np.frombuffer(row[1], dtype="<f4")
                    if raw.size != dimension:
                        raise ValueError("Cached vector width is invalid")
                    self._vector(raw, dimension)
                    if not np.isclose(np.linalg.norm(raw.astype(np.float64)), 1.0, rtol=1e-5, atol=1e-7):
                        raise ValueError("Cached vector is not normalized")
                except (TypeError, ValueError, LocalModelError):
                    invalid += 1
                    continue
                for index in indices:
                    output[index] = raw.copy()
            if missing or invalid:
                raise LocalModelError(
                    f"Embedding cache has {missing} missing and {invalid} invalid unique input(s). " + remediation
                )
        except LocalModelError:
            raise
        except (sqlite3.Error, TypeError, ValueError, OverflowError) as exc:
            raise LocalModelError("Embedding cache metadata or storage is invalid. " + remediation) from exc
        self.stats["embedding_cache_hits"] += len(texts)
        self._dimensions[profile["digest"]] = dimension
        self.embedding_profile = profile
        return np.stack(output).astype(np.float32, copy=False)

    def chat_json(self, system: str, payload: dict, schema: dict, max_tokens: int = 450) -> dict:
        if not isinstance(system, str) or not system.strip():
            raise ValueError("system must be a nonempty string")
        if not isinstance(payload, dict) or not isinstance(schema, dict):
            raise ValueError("payload and schema must be dictionaries")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        try:
            content = _json(payload)
            _json(schema)
        except (ValueError, TypeError) as exc:
            raise ValueError("payload and schema must contain finite JSON-compatible values") from exc
        self.health(required_roles=("chat",))
        model = self._resolved[self.chat_model]
        body = {
            "model": model["name"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, **self.chat_options, "num_predict": max_tokens},
            "keep_alive": self.keep_alive,
        }
        cache_key = _key([CACHE_VERSION, "chat", model["digest"], body])
        row = self._db.execute("SELECT response_json FROM chats WHERE cache_key = ?", (cache_key,)).fetchone()
        if row:
            try:
                result = _strict_json(row[0])
                if not isinstance(result, dict):
                    raise ValueError("Chat result must be an object")
                _validate_schema(result, schema)
            except (ValueError, TypeError):
                self._db.execute("DELETE FROM chats WHERE cache_key = ?", (cache_key,))
                self._db.commit()
            else:
                self.stats["chat_cache_hits"] += 1
                return result
        failure = None
        for attempt in range(2):
            response = None
            try:
                response = self._request("POST", "/api/chat", body)
                self.stats["chat_prompt_tokens"] += int(response.get("prompt_eval_count", 0))
                self.stats["chat_output_tokens"] += int(response.get("eval_count", 0))
                if response.get("done") is not True:
                    raise ValueError("Chat did not complete")
                if response.get("done_reason") in {"length", "max_tokens"}:
                    raise ValueError("Chat was truncated by its token limit")
                message = response.get("message")
                if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                    raise ValueError("Chat has no text content")
                result = _strict_json(message["content"])
                if not isinstance(result, dict):
                    raise ValueError("Chat result must be an object")
                _validate_schema(result, schema)
            except (ValueError, TypeError) as exc:
                failure = exc
                if attempt == 0:
                    # Repeating a deterministic invalid completion unchanged
                    # gives the model no information with which to repair it.
                    message = response.get("message", {}) if isinstance(response, dict) else {}
                    invalid = message.get("content") if isinstance(message, dict) else None
                    if isinstance(invalid, str):
                        body = {**body, "messages": body["messages"] + [
                            {"role": "assistant", "content": invalid},
                            {"role": "user", "content": _json({
                                "task": "repair_json_output", "validation_error": str(exc),
                                "instruction": "Correct the preceding output to satisfy the requested schema. "
                                               "Preserve supported meaning and return only the corrected JSON.",
                            })},
                        ]}
                continue
            except LocalModelError as exc:
                failure = exc
                continue
            with self._db:
                self._db.execute(
                    "INSERT OR REPLACE INTO chats (cache_key, response_json) VALUES (?, ?)",
                    (cache_key, _json(result)),
                )
            return result
        if isinstance(failure, (ValueError, TypeError)):
            # Validation exceptions identify schema paths/types, not source
            # values. Their cause remains available without logging raw output.
            reason = _server_detail(_json({"error": str(failure)}).encode("utf-8"))
            raise LocalGenerationError(
                f"Local draft failed its output contract after two attempts. Validation: {reason}"
            ) from failure
        raise LocalModelError(
            "Local chat was unavailable after two attempts; check the local model service and logs."
        ) from failure

    def unload(self, model: str) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty installed model name")
        resolved = self._resolved.get(model, {})
        self._request(
            "POST", "/api/generate", {"model": resolved.get("name", model), "prompt": "", "stream": False, "keep_alive": 0}
        )
