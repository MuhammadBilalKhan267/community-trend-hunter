"""Export or restore lossless embedding shards without model inference."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trend_hunter.embedding_identity import CACHE_VERSION, checked_profile, read_profiles, save_profile


FORMAT = "embedding-cache-npz-v1"
TABLE = "CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, dimension INTEGER NOT NULL, vector BLOB NOT NULL)"


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def checked_row(key, dimension, blob):
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("Invalid embedding cache key")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("Invalid embedding dimension")
    vector = np.frombuffer(blob, dtype="<f4")
    if vector.size != dimension or not np.isfinite(vector).all():
        raise ValueError("Invalid embedding vector")
    if not np.isclose(np.linalg.norm(vector.astype(np.float64)), 1, rtol=1e-5, atol=1e-7):
        raise ValueError("Embedding is not normalized")
    return key, dimension, blob


def checked_profiles(profiles):
    """Validate explicit portable identities and reject ambiguous bindings."""
    if not isinstance(profiles, list):
        raise ValueError("Embedding profiles must be a list")
    result, seen = [], set()
    for value in profiles:
        if (not isinstance(value, dict)
                or set(value) != {"model_name", "digest", "options", "dimensions"}
                or not isinstance(value["options"], dict)):
            raise ValueError("Invalid embedding profile metadata")
        profile = checked_profile(value)
        identity = (profile["model_name"], json.dumps(profile["options"], sort_keys=True,
                                                     separators=(",", ":"), allow_nan=False))
        if identity in seen:
            raise ValueError("Duplicate embedding model/options profile in bundle")
        seen.add(identity)
        result.append(profile)
    return result


def _manifest_json(source):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate embedding manifest field")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("Non-finite embedding manifest value")

    return json.loads(source, object_pairs_hook=pairs, parse_constant=constant)


def export_embeddings(cache, destination, rows_per_shard=10_000):
    cache, destination = Path(cache).resolve(), Path(destination).resolve()
    if isinstance(rows_per_shard, bool) or not isinstance(rows_per_shard, int) or rows_per_shard < 1:
        raise ValueError("rows_per_shard must be a positive integer")
    # A fresh directory prevents accidental replacement of a versioned bundle.
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"format": FORMAT, "cache_version": CACHE_VERSION,
                "vector_encoding": "little-endian float32", "rows": 0, "shards": []}
    with closing(sqlite3.connect(cache.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        profiles = checked_profiles(read_profiles(db))
        if profiles:
            manifest["embedding_profiles"] = profiles
        cursor = db.execute("SELECT cache_key, dimension, vector FROM embeddings ORDER BY cache_key")
        while rows := cursor.fetchmany(rows_per_shard):
            for row in rows:
                checked_row(*row)
            name = f"part-{len(manifest['shards']) + 1:03d}.npz"
            target = destination / name
            np.savez_compressed(target,
                                cache_keys=np.asarray([r[0] for r in rows], dtype="S64"),
                                dimensions=np.asarray([r[1] for r in rows], dtype="<i8"),
                                vectors=np.concatenate([np.frombuffer(r[2], dtype="<f4") for r in rows]))
            manifest["shards"].append({"file": name, "rows": len(rows),
                                       "bytes": target.stat().st_size, "sha256": sha256(target)})
            manifest["rows"] += len(rows)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def restore_embeddings(bundle, cache):
    bundle, cache = Path(bundle).resolve(), Path(cache)
    manifest = _manifest_json((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or manifest.get("cache_version") != CACHE_VERSION):
        raise ValueError("Unsupported embedding bundle or cache version")
    if not isinstance(manifest.get("shards"), list) or type(manifest.get("rows")) is not int or manifest["rows"] < 0:
        raise ValueError("Invalid embedding manifest")
    profiles = checked_profiles(manifest.get("embedding_profiles", []))
    cache.parent.mkdir(parents=True, exist_ok=True)
    total, inserted, seen = 0, 0, set()
    with closing(sqlite3.connect(cache, timeout=30)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(TABLE)
        for shard in manifest["shards"]:
            name = shard.get("file", "")
            if not isinstance(name, str) or not re.fullmatch(r"part-[0-9]+\.npz", name):
                raise ValueError("Invalid embedding shard filename")
            path = (bundle / name).resolve()
            if path.parent != bundle or sha256(path) != shard.get("sha256"):
                raise ValueError("Embedding shard path or checksum mismatch")
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"cache_keys", "dimensions", "vectors"}:
                    raise ValueError("Invalid embedding shard fields")
                keys, dimensions, vectors = data["cache_keys"], data["dimensions"], data["vectors"]
                if (keys.dtype != np.dtype("S64") or keys.ndim != 1
                        or dimensions.dtype != np.dtype("<i8") or dimensions.shape != keys.shape
                        or vectors.dtype != np.dtype("<f4") or vectors.ndim != 1
                        or type(shard.get("rows")) is not int or len(keys) != shard["rows"]
                        or (dimensions < 1).any() or sum(map(int, dimensions)) != len(vectors)):
                    raise ValueError("Invalid embedding shard arrays")
                offset = 0
                for encoded, dimension in zip(keys, dimensions):
                    key = bytes(encoded).decode("ascii")
                    dimension = int(dimension)
                    blob = vectors[offset:offset + dimension].tobytes()
                    checked_row(key, dimension, blob)
                    if key in seen:
                        raise ValueError("Duplicate embedding key in bundle")
                    seen.add(key)
                    existing = db.execute("SELECT dimension, vector FROM embeddings WHERE cache_key = ?", (key,)).fetchone()
                    if existing is not None and existing != (dimension, blob):
                        raise ValueError("Bundle conflicts with an existing embedding; cache left unchanged")
                    if existing is None:
                        db.execute("INSERT INTO embeddings VALUES (?, ?, ?)", (key, dimension, blob))
                        inserted += 1
                    total += 1
                    offset += dimension
        if total != manifest["rows"]:
            raise ValueError("Embedding bundle row count mismatch")
        for profile in profiles:
            save_profile(db, profile)
    return {"rows": total, "inserted": inserted, "already_present": total - inserted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Copy only embedding entries into a new bundle directory")
    export.add_argument("--cache", default=".cache/models.sqlite3")
    export.add_argument("--out", required=True)
    export.add_argument("--rows-per-shard", type=int, default=10_000)
    restore = commands.add_parser("restore", help="Verify and merge a bundle into the runtime cache")
    restore.add_argument("--bundle", default="embeddings")
    restore.add_argument("--cache", default=".cache/models.sqlite3")
    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export_embeddings(args.cache, args.out, args.rows_per_shard)
            result = {"rows": result["rows"], "shards": len(result["shards"]), "out": args.out}
        else:
            result = restore_embeddings(args.bundle, args.cache)
        print(json.dumps(result))
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"Embedding cache: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
