from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from trend_hunter.embedding_identity import (
    CACHE_VERSION, cache_embedding_options, canonical_model_name, checked_profile,
    embedding_key, read_profiles, save_profile,
)
from trend_hunter.local_models import LocalModelError, LocalModels


class CachedEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = Path(self.temp.name) / "models.sqlite3"
        self.profile = {"model_name": "bge-m3:latest", "digest": "test-digest",
                        "options": cache_embedding_options(), "dimensions": 3}
        self.vectors = {"中文证据": np.array([0.6, 0.8, 0], dtype="<f4"),
                        "another input": np.array([0, 0, 1], dtype="<f4")}
        with closing(sqlite3.connect(self.cache)) as db, db:
            db.execute("CREATE TABLE embeddings (cache_key TEXT PRIMARY KEY, dimension INTEGER NOT NULL, vector BLOB NOT NULL)")
            save_profile(db, self.profile)
            for text, vector in self.vectors.items():
                db.execute("INSERT INTO embeddings VALUES (?, ?, ?)",
                           (embedding_key("bge-m3", "test-digest", text), 3, vector.tobytes()))

    def tearDown(self):
        self.temp.cleanup()

    def offline(self, **kwargs):
        models = LocalModels(cache_path=self.cache, embedding_cache_only=True, **kwargs)
        models.health = lambda *args, **kw: self.fail("cache-only embedding called health")
        models._transport = lambda *args, **kw: self.fail("cache-only embedding called transport")
        return models

    def rows(self):
        with closing(sqlite3.connect(self.cache)) as db, db:
            return db.execute("SELECT * FROM embeddings ORDER BY cache_key").fetchall()

    def test_cache_only_preserves_exact_vectors_duplicates_order_and_cache_bytes(self):
        before = self.cache.read_bytes()
        texts = ["another input", "中文证据", "another input"]
        with patch("trend_hunter.local_models.request.build_opener", side_effect=AssertionError("opener")):
            with self.offline(base_url="irrelevant invalid endpoint", embedding_options={"num_gpu": 0}) as models:
                vectors = models.embed(texts)
                self.assertEqual(vectors.dtype, np.dtype("float32"))
                self.assertEqual(vectors.tobytes(), np.stack([self.vectors[text] for text in texts]).tobytes())
                self.assertEqual(models.stats["embedding_cache_hits"], 3)
                self.assertEqual(models.stats["api_calls"], 0)
                self.assertEqual(models.embedding_profile, checked_profile(self.profile))
                with self.assertRaises(sqlite3.OperationalError):
                    models._db.execute("DELETE FROM embeddings")
        self.assertEqual(self.cache.read_bytes(), before)

    def test_missing_inputs_are_aggregated_without_text_or_changes(self):
        before = self.cache.read_bytes()
        secret = "private source content"
        with self.offline() as models:
            with self.assertRaises(LocalModelError) as caught:
                models.embed([secret, secret, "second missing source", "中文证据"])
            self.assertIn("2 missing and 0 invalid unique", str(caught.exception))
            self.assertIn("restore", str(caught.exception))
            self.assertIn("ingest", str(caught.exception))
            self.assertNotIn(secret, str(caught.exception))
            self.assertEqual(models.stats["embedding_cache_hits"], 0)
        self.assertEqual(self.cache.read_bytes(), before)

    def test_profile_selection_uses_exact_option_encoding_from_cache_keys(self):
        with closing(sqlite3.connect(self.cache)) as db, db:
            save_profile(db, {**self.profile, "digest": "other-revision",
                              "options": {"num_ctx": 512.0, "num_batch": 512}})
        with self.offline() as models:
            result = models.embed(["中文证据"])
            self.assertEqual(models.embedding_profile["digest"], "test-digest")
            self.assertEqual(result.tobytes(), self.vectors["中文证据"].reshape(1, -1).tobytes())

    def test_missing_profile_name_and_option_mismatches_fail_offline(self):
        for kwargs in ({"embedding_model": "other-model"}, {"embedding_options": {"num_ctx": 1024}}):
            with self.subTest(kwargs=kwargs), self.offline(**kwargs) as models:
                with self.assertRaisesRegex(LocalModelError, "No matching saved"):
                    models.embed(["中文证据"])
        with closing(sqlite3.connect(self.cache)) as db, db:
            db.execute("DROP TABLE embedding_profiles")
        before = self.cache.read_bytes()
        with self.offline() as models:
            with self.assertRaisesRegex(LocalModelError, "No matching saved"):
                models.embed(["中文证据"])
        self.assertEqual(self.cache.read_bytes(), before)

    def test_missing_cache_path_does_not_create_file(self):
        missing = Path(self.temp.name) / "absent" / "models.sqlite3"
        with self.assertRaisesRegex(LocalModelError, "read-only.*Restore"):
            LocalModels(cache_path=missing, embedding_cache_only=True)
        self.assertFalse(missing.parent.exists())

    def test_cache_only_mode_rejects_live_methods_before_transport(self):
        with LocalModels(cache_path=self.cache, embedding_cache_only=True) as models, \
                patch.object(models, "_transport", side_effect=AssertionError("No transport allowed")) as transport:
            for operation in (models.health, lambda: models.unload("bge-m3")):
                with self.subTest(operation=operation), self.assertRaisesRegex(LocalModelError, "disabled"):
                    operation()
            self.assertEqual(models.stats["api_calls"], 0)
            transport.assert_not_called()

    def test_corrupt_vectors_fail_without_deleting_or_replacing_rows(self):
        cases = [(3, b"bad"), (2, np.array([1, 0], dtype="<f4").tobytes()),
                 (3, np.array([float("nan"), 0, 1], dtype="<f4").tobytes()),
                 (3, np.array([0, 0, 0], dtype="<f4").tobytes()),
                 (3, np.array([1, 1, 1], dtype="<f4").tobytes())]
        key = embedding_key("bge-m3", "test-digest", "中文证据")
        for dimension, blob in cases:
            with self.subTest(dimension=dimension, blob=blob):
                with closing(sqlite3.connect(self.cache)) as db, db:
                    db.execute("UPDATE embeddings SET dimension = ?, vector = ? WHERE cache_key = ?", (dimension, blob, key))
                before = self.cache.read_bytes()
                with self.offline() as models:
                    with self.assertRaisesRegex(LocalModelError, "0 missing and 1 invalid unique"):
                        models.embed(["中文证据", "中文证据", "another input"])
                self.assertEqual(self.cache.read_bytes(), before)

    def test_invalid_profile_is_rejected_without_mutation(self):
        with closing(sqlite3.connect(self.cache)) as db, db:
            db.execute("UPDATE embedding_profiles SET dimension = 0")
        before = self.cache.read_bytes()
        with self.offline() as models:
            with self.assertRaisesRegex(LocalModelError, "metadata or storage is invalid"):
                models.embed(["中文证据"])
        self.assertEqual(self.cache.read_bytes(), before)

    def test_one_run_rejects_a_changed_profile_between_embedding_windows(self):
        with self.offline() as models:
            models.embed(["中文证据"])
            with closing(sqlite3.connect(self.cache)) as db, db:
                save_profile(db, {**self.profile, "digest": "new-revision"}, replace=True)
            before = self.cache.read_bytes()
            with self.assertRaisesRegex(LocalModelError, "profile changed during this run"):
                models.embed(["another input"])
            self.assertEqual(models.embedding_profile["digest"], "test-digest")
            self.assertEqual(self.cache.read_bytes(), before)

    def test_live_cache_hits_create_profile_and_digest_changes_replace_only_binding(self):
        with closing(sqlite3.connect(self.cache)) as db, db:
            db.execute("DROP TABLE embedding_profiles")
        before_rows = self.rows()
        digest = "test-digest"
        def transport(method, path, payload):
            if path == "/api/tags":
                return {"models": [{"name": "bge-m3:latest", "digest": digest}]}
            if path == "/api/embed":
                return {"embeddings": [[1, 0, 0] for _ in payload["input"]]}
            self.fail("Unexpected local endpoint")
        with LocalModels(cache_path=self.cache) as models:
            models._transport = transport
            models.embed(["中文证据"])
            self.assertEqual(models.stats["embedding_calls"], 0)
            self.assertEqual(read_profiles(models._db)[0], checked_profile(self.profile))
            self.assertEqual(self.rows(), before_rows)
            digest = "new-revision"
            models.embed(["中文证据"])
            self.assertEqual(read_profiles(models._db)[0]["digest"], digest)
        self.assertTrue(all(row in self.rows() for row in before_rows))


class EmbeddingIdentityTests(unittest.TestCase):
    def test_keys_match_original_algorithm_exactly(self):
        options = {"num_ctx": 1024, "num_gpu": 0, "num_thread": 8, "custom": "中文"}
        expected_options = {"num_ctx": 1024, "num_batch": 512, "custom": "中文"}
        original = json.dumps([CACHE_VERSION, "embedding", "org/bge-m3:latest", "digest", "中文输入", expected_options],
                              ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(embedding_key("org/bge-m3", "digest", "中文输入", options), hashlib.sha256(original.encode()).hexdigest())
        self.assertEqual(canonical_model_name("localhost:123/org/model"), "localhost:123/org/model:latest")
        self.assertEqual(canonical_model_name("org/model:revision"), "org/model:revision")

    def test_profile_validation_rejects_invalid_identity_options_and_dimensions(self):
        profile = {"model_name": "bge-m3", "digest": "digest", "options": cache_embedding_options(), "dimensions": 3}
        for field, value in (("model_name", "has whitespace"), ("digest", "bad\x00digest"),
                             ("digest", ""), ("dimensions", True), ("dimensions", 0),
                             ("options", {"num_ctx": float("nan"), "num_batch": 512}),
                             ("options", {1: 2}), ("options", {}), ("options", None),
                             ("options", {"num_gpu": 0})):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                checked_profile({**profile, field: value})
        self.assertEqual(cache_embedding_options({"num_gpu": 0}), {"num_ctx": 512, "num_batch": 512})
        without_options = {key: value for key, value in profile.items() if key != "options"}
        with self.assertRaises(ValueError):
            checked_profile(without_options)

    def test_profiles_reject_conflicts_and_do_not_commit_caller_transaction(self):
        with closing(sqlite3.connect(":memory:")) as db, db:
            self.assertEqual(read_profiles(db), [])
            profile = {"model_name": "bge-m3", "digest": "one", "options": cache_embedding_options(), "dimensions": 3}
            db.execute("BEGIN")
            save_profile(db, profile)
            self.assertTrue(db.in_transaction)
            db.rollback()
            self.assertEqual(read_profiles(db), [])
            save_profile(db, profile)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                save_profile(db, {**profile, "digest": "two"})
            self.assertEqual(read_profiles(db)[0]["digest"], "one")
            save_profile(db, {**profile, "digest": "two"}, replace=True)
            self.assertEqual(read_profiles(db)[0]["digest"], "two")

    def test_stored_options_must_be_canonical_and_finite(self):
        with closing(sqlite3.connect(":memory:")) as db, db:
            save_profile(db, {"model_name": "bge-m3", "digest": "one", "options": cache_embedding_options(), "dimensions": 3})
            for options in ('{"num_ctx":512,"num_batch":512}', '{"num_ctx":NaN}', '{"num_ctx":512,"num_ctx":512}'):
                with self.subTest(options=options):
                    db.execute("UPDATE embedding_profiles SET options_json = ?", (options,))
                    with self.assertRaises(ValueError):
                        read_profiles(db)


if __name__ == "__main__":
    unittest.main()
