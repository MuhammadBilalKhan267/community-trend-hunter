"""Validated encoder identities shared by live inference and portable caches."""

from __future__ import annotations

import hashlib
import json
import sqlite3


CACHE_VERSION = "local-models-v1"
PROFILE_TABLE = (
    "CREATE TABLE IF NOT EXISTS embedding_profiles "
    "(model_name TEXT NOT NULL, options_json TEXT NOT NULL, digest TEXT NOT NULL, "
    "dimension INTEGER NOT NULL, PRIMARY KEY (model_name, options_json))"
)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identifier(value, label: str) -> str:
    if (not isinstance(value, str) or not value
            or any(char.isspace() or ord(char) < 32 or 127 <= ord(char) < 160 for char in value)):
        raise ValueError(f"{label} must be nonempty and contain no whitespace or control characters")
    return value


def canonical_model_name(name: str) -> str:
    name = _identifier(name, "Embedding model name")
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def cache_embedding_options(options=None) -> dict:
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("Embedding options must be a finite JSON object")

    def check_keys(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) or not key for key in value):
                raise ValueError("Embedding option keys must be nonempty strings")
            for item in value.values():
                check_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_keys(item)

    check_keys(options)
    try:
        copied = json.loads(_json(options))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Embedding options must contain finite JSON-compatible values") from exc
    effective = {"num_ctx": 512, "num_batch": 512, **copied}
    return {key: value for key, value in effective.items() if key not in {"num_gpu", "num_thread"}}


def embedding_key(model_name: str, digest: str, text: str, options=None) -> str:
    """Use the original key encoding; device placement does not affect identity."""
    name = canonical_model_name(model_name)
    digest = _identifier(digest, "Embedding model digest")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Embedding input must be a nonempty string")
    value = [CACHE_VERSION, "embedding", name, digest, text, cache_embedding_options(options)]
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def checked_profile(profile) -> dict:
    if not isinstance(profile, dict):
        raise ValueError("Embedding profile must be an object")
    name = canonical_model_name(profile.get("model_name"))
    digest = _identifier(profile.get("digest"), "Embedding model digest")
    dimension = profile.get("dimensions")
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Embedding profile dimensions must be a positive integer")
    raw_options = profile.get("options")
    if not isinstance(raw_options, dict) or not {"num_ctx", "num_batch"} <= raw_options.keys():
        raise ValueError("Embedding profile options must explicitly include num_ctx and num_batch")
    options = cache_embedding_options(raw_options)
    return {"model_name": name, "digest": digest, "options": options, "dimensions": dimension}


def _parse_options(value: str) -> dict:
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate embedding option key")
            result[key] = item
        return result

    def constant(value):
        raise ValueError("Non-finite embedding option")

    if not isinstance(value, str):
        raise ValueError("Embedding profile options must be canonical JSON")
    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def read_profiles(db: sqlite3.Connection) -> list[dict]:
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='embedding_profiles'").fetchone() is None:
        return []
    profiles = []
    for name, options_json, digest, dimension in db.execute(
            "SELECT model_name, options_json, digest, dimension FROM embedding_profiles ORDER BY model_name, options_json"):
        profile = checked_profile({"model_name": name, "digest": digest,
                                   "options": _parse_options(options_json), "dimensions": dimension})
        if name != profile["model_name"] or options_json != _json(profile["options"]):
            raise ValueError("Embedding profile identity is not canonical")
        profiles.append(profile)
    return profiles


def save_profile(db: sqlite3.Connection, profile, *, replace: bool = False) -> None:
    """Save one validated name/options binding inside the caller's transaction."""
    profile = checked_profile(profile)
    options_json = _json(profile["options"])
    db.execute(PROFILE_TABLE)
    existing = db.execute(
        "SELECT digest, dimension FROM embedding_profiles WHERE model_name = ? AND options_json = ?",
        (profile["model_name"], options_json),
    ).fetchone()
    binding = (profile["digest"], profile["dimensions"])
    if existing is not None and existing != binding and not replace:
        raise ValueError("Embedding profile conflicts with the existing model/options binding")
    db.execute(
        "INSERT INTO embedding_profiles (model_name, options_json, digest, dimension) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(model_name, options_json) DO UPDATE SET digest = excluded.digest, dimension = excluded.dimension",
        (profile["model_name"], options_json, *binding),
    )
