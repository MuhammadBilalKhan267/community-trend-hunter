import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from scripts.embedding_cache import TABLE, export_embeddings, restore_embeddings
from trend_hunter.embedding_identity import PROFILE_TABLE, checked_profile, read_profiles, save_profile
from trend_hunter.local_models import LocalModels


class EmbeddingBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.sqlite3"
        self.bundle = self.root / "bundle"
        self.target = self.root / "target.sqlite3"
        self.rows = []
        self.profile = {"model_name": "test-encoder:latest", "digest": "test-digest-v1",
                        "options": {"num_ctx": 512, "num_batch": 512}, "dimensions": 3}
        for number, dimension in enumerate((3, 4, 3)):
            vector = np.arange(1, dimension + 1, dtype=np.float64)
            vector = (vector / np.linalg.norm(vector)).astype("<f4")
            self.rows.append((hashlib.sha256(str(number).encode()).hexdigest(), dimension, vector.tobytes()))
        with closing(sqlite3.connect(self.source)) as db, db:
            db.execute(TABLE)
            db.executemany("INSERT INTO embeddings VALUES (?, ?, ?)", self.rows)
            db.execute("CREATE TABLE chats (response_json TEXT)")
            db.execute("INSERT INTO chats VALUES ('private generated response')")

    def test_lossless_sharded_round_trip_is_repeatable_and_excludes_chats(self):
        manifest = export_embeddings(self.source, self.bundle, 2)
        self.assertEqual((manifest["rows"], len(manifest["shards"])), (3, 2))
        self.assertNotIn("embedding_profiles", manifest)
        self.assertEqual(restore_embeddings(self.bundle, self.target), {"rows": 3, "inserted": 3, "already_present": 0})
        self.assertEqual(restore_embeddings(self.bundle, self.target)["already_present"], 3)
        with closing(sqlite3.connect(self.target)) as db, db:
            self.assertEqual(db.execute("SELECT * FROM embeddings ORDER BY cache_key").fetchall(), sorted(self.rows))
            self.assertEqual(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("embeddings",)])
        # The normal model store accepts the restored schema without migration.
        with LocalModels(cache_path=self.target) as models:
            self.assertEqual(models._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], 3)
        with self.assertRaises(FileExistsError):
            export_embeddings(self.source, self.bundle)

    def test_corrupt_later_shard_rolls_back_earlier_inserts(self):
        manifest = export_embeddings(self.source, self.bundle, 2)
        path = self.bundle / manifest["shards"][-1]["file"]
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            restore_embeddings(self.bundle, self.target)
        with closing(sqlite3.connect(self.target)) as db, db:
            names = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            self.assertTrue(not names or db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0)

    def test_conflict_keeps_existing_vectors_and_unrelated_cache_tables(self):
        export_embeddings(self.source, self.bundle, 2)
        key, dimension, _ = sorted(self.rows)[-1]
        with closing(sqlite3.connect(self.target)) as db, db:
            db.execute(TABLE)
            db.execute("INSERT INTO embeddings VALUES (?, ?, ?)", (key, dimension, b"existing vector"))
            db.execute("CREATE TABLE chats (response_json TEXT)")
            db.execute("INSERT INTO chats VALUES ('preserve')")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            restore_embeddings(self.bundle, self.target)
        with closing(sqlite3.connect(self.target)) as db, db:
            self.assertEqual(db.execute("SELECT * FROM embeddings").fetchall(), [(key, dimension, b"existing vector")])
            self.assertEqual(db.execute("SELECT * FROM chats").fetchall(), [("preserve",)])

    def test_manifest_cannot_escape_bundle_directory(self):
        export_embeddings(self.source, self.bundle)
        path = self.bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["shards"][0]["file"] = "../outside.npz"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "filename"):
            restore_embeddings(self.bundle, self.target)

    def test_profile_round_trip_is_idempotent_and_preserves_unrelated_chats(self):
        with closing(sqlite3.connect(self.source)) as db, db:
            save_profile(db, self.profile)
        manifest = export_embeddings(self.source, self.bundle, 2)
        self.assertEqual(manifest["embedding_profiles"], [checked_profile(self.profile)])
        self.assertNotIn("private generated response", json.dumps(manifest))
        with closing(sqlite3.connect(self.target)) as db, db:
            db.execute("CREATE TABLE chats (response_json TEXT)")
            db.execute("INSERT INTO chats VALUES ('preserve')")
        self.assertEqual(restore_embeddings(self.bundle, self.target),
                         {"rows": 3, "inserted": 3, "already_present": 0})
        self.assertEqual(restore_embeddings(self.bundle, self.target),
                         {"rows": 3, "inserted": 0, "already_present": 3})
        with closing(sqlite3.connect(self.target)) as db:
            self.assertEqual(read_profiles(db), [checked_profile(self.profile)])
            self.assertEqual(db.execute("SELECT * FROM embeddings ORDER BY cache_key").fetchall(), sorted(self.rows))
            self.assertEqual(db.execute("SELECT * FROM chats").fetchall(), [("preserve",)])

    def test_profile_binding_conflict_rolls_back_all_new_vectors(self):
        with closing(sqlite3.connect(self.source)) as db, db:
            save_profile(db, self.profile)
        export_embeddings(self.source, self.bundle, 2)
        for number, change in enumerate(({"digest": "other-digest"}, {"dimensions": 4})):
            target = self.root / f"conflict-{number}.sqlite3"
            existing_profile = {**self.profile, **change}
            with closing(sqlite3.connect(target)) as db, db:
                db.execute(TABLE)
                db.execute("INSERT INTO embeddings VALUES (?, ?, ?)", self.rows[0])
                save_profile(db, existing_profile)
                db.execute("CREATE TABLE chats (response_json TEXT)")
                db.execute("INSERT INTO chats VALUES ('preserve')")
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "conflicts"):
                restore_embeddings(self.bundle, target)
            with closing(sqlite3.connect(target)) as db:
                self.assertEqual(db.execute("SELECT * FROM embeddings").fetchall(), [self.rows[0]])
                self.assertEqual(read_profiles(db), [checked_profile(existing_profile)])
                self.assertEqual(db.execute("SELECT * FROM chats").fetchall(), [("preserve",)])

    def test_malformed_and_duplicate_profiles_fail_before_opening_destination(self):
        manifest = export_embeddings(self.source, self.bundle)
        missing = {key: value for key, value in self.profile.items() if key != "options"}
        alias = {**self.profile, "model_name": "test-encoder"}
        invalid = [None, {}, [None], [missing], [{**self.profile, "dimensions": True}],
                   [{**self.profile, "dimensions": 0}], [{**self.profile, "options": []}],
                   [{**self.profile, "options": {}}], [{**self.profile, "digest": ""}],
                   [{**self.profile, "options": {"num_ctx": float("nan"), "num_batch": 512}}],
                   [self.profile, alias]]
        for number, profiles in enumerate(invalid):
            target = self.root / f"malformed-{number}.sqlite3"
            (self.bundle / "manifest.json").write_text(json.dumps({**manifest, "embedding_profiles": profiles}))
            with self.subTest(profiles=profiles), self.assertRaises(ValueError):
                restore_embeddings(self.bundle, target)
            self.assertFalse(target.exists())
        duplicate_option = json.dumps({**manifest, "embedding_profiles": [self.profile]}).replace(
            '"num_ctx": 512', '"num_ctx": 256, "num_ctx": 512')
        (self.bundle / "manifest.json").write_text(duplicate_option)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            restore_embeddings(self.bundle, self.target)
        self.assertFalse(self.target.exists())

    def test_bad_vector_shard_rolls_back_with_profile_metadata_present(self):
        with closing(sqlite3.connect(self.source)) as db, db:
            save_profile(db, self.profile)
        manifest = export_embeddings(self.source, self.bundle, 2)
        shard = self.bundle / manifest["shards"][-1]["file"]
        shard.write_bytes(shard.read_bytes() + b"corrupt")
        with closing(sqlite3.connect(self.target)) as db, db:
            db.execute("CREATE TABLE chats (response_json TEXT)")
            db.execute("INSERT INTO chats VALUES ('preserve')")
        with self.assertRaisesRegex(ValueError, "checksum"):
            restore_embeddings(self.bundle, self.target)
        with closing(sqlite3.connect(self.target)) as db:
            self.assertEqual(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("chats",)])
            self.assertEqual(db.execute("SELECT * FROM chats").fetchall(), [("preserve",)])

    def test_export_rejects_invalid_profile_metadata_without_writing_shards(self):
        with closing(sqlite3.connect(self.source)) as db, db:
            db.execute(PROFILE_TABLE)
            db.execute("INSERT INTO embedding_profiles VALUES (?, ?, ?, ?)",
                       ("test-encoder:latest", '{"num_ctx":NaN,"num_batch":512}', "test-digest-v1", 3))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            export_embeddings(self.source, self.bundle)
        self.assertEqual(list(self.bundle.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
