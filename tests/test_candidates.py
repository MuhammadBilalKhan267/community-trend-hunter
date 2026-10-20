import unittest
from datetime import datetime, timedelta

import numpy as np

from trend_hunter.candidates import (build_candidate_pool, build_context_index, choose_batch, choose_neighbors,
                                     extract_aliases, merge_related_bursts, sample_candidate)
from trend_hunter.data import GAME_TIMEZONE, normalize, reaction_kind
from trend_hunter.temporal import activity_key


def record(identifier, text="Substantive game feedback", minute=0, user=None, channel="chat", parent=None, day="2026-08-01"):
    stamp = datetime.fromisoformat(day + "T12:00:00+08:00") + timedelta(minutes=minute)
    return {"id": str(identifier), "text": normalize(text), "content": text, "title": "", "family": normalize(text),
            "day": stamp.date().isoformat(), "created_at": stamp.isoformat(), "_instant": stamp,
            "platform": "QQ", "channel": channel, "user_name": user or str(identifier), "parent_id": parent,
            "root_id": parent or str(identifier), "reaction": reaction_kind(normalize(text)), "trivial": False}


def inputs(records):
    units = [{"unit_id": r["id"] + ":0", "voice_id": r["id"], "text": r["text"],
              "day": r["day"], "platform": r["platform"]} for r in records if not r["reaction"]]
    vectors = np.asarray([[1., 0.] for _ in units], dtype=np.float32)
    mapping = np.arange(len(units))
    labels = np.zeros(len(units), dtype=int)
    centers = np.asarray([[1., 0.]], dtype=np.float32)
    return units, vectors, mapping, labels, centers


class CandidateTests(unittest.TestCase):
    def pool(self, records, cutoff="2026-08-01"):
        values = inputs(records)
        return build_candidate_pool(*values, records, records, cutoff), values

    def test_cutoff_filters_pool_members_and_aliases(self):
        records = [record("a", "角色甲=铁锅"), record("future", "新角色=小猫", day="2026-08-02")]
        pool, _ = self.pool(records)
        cluster = next(c for c in pool if c["kind"] == "cluster")
        self.assertEqual(cluster["member_ids"], ["a"])
        self.assertEqual([a["source_id"] for a in extract_aliases(records, "2026-08-01")], ["a"])

    def test_literal_alias_preserves_source_and_does_not_rewrite(self):
        r = record("a", "艾琳=铁锅，新人别懵")
        original = dict(r)
        aliases = extract_aliases([r], "2026-08-01")
        self.assertEqual((aliases[0]["name"], aliases[0]["alias"]), ("艾琳", "铁锅"))
        self.assertEqual(r, original)

    def test_replies_are_engagement_not_member_support(self):
        records = [record("a"), record("r", "+1", minute=1, parent="a")]
        pool, _ = self.pool(records)
        cluster = next(c for c in pool if c["kind"] == "cluster")
        self.assertEqual(cluster["member_ids"], ["a"])
        self.assertEqual(cluster["history"]["stats"]["raw_count"], 1)
        self.assertEqual(cluster["diagnostics"]["explicit_reply_count"], 1)

    def test_burst_is_nomination_not_reaction_membership(self):
        records = [record("a", "anyone else think this map looks weird")]
        records += [record(str(i), text, minute=i, user=str(i % 3))
                    for i, text in enumerate(["yes", "yea", "bad", "no", "same"], 1)]
        pool, _ = self.pool(records)
        burst = next(c for c in pool if c["kind"] == "burst")
        self.assertEqual(burst["member_ids"], ["a"])
        self.assertEqual(burst["diagnostics"]["possible_response_count"], 5)
        self.assertEqual(burst["history"]["stats"]["raw_count"], 1)
        self.assertTrue(burst["related_clusters"])

    def test_different_channels_do_not_create_burst(self):
        records = [record("a", "anyone else think this map looks weird", channel="one")]
        records += [record(str(i), "yes", minute=i, channel="two") for i in range(1, 7)]
        pool, _ = self.pool(records)
        self.assertFalse(any(c["kind"] == "burst" for c in pool))

    def test_routine_channel_response_background_does_not_become_burst(self):
        records = [record(str(i), "yes", minute=i, user=str(i % 3)) for i in range(60)]
        records.append(record("a", "anyone else think this map looks weird", minute=60))
        records += [record("after" + str(i), "yes", minute=60 + i, user=str(i % 3)) for i in range(1, 5)]
        pool, _ = self.pool(records)
        self.assertFalse(any(c["kind"] == "burst" for c in pool))

    def test_merge_burst_enriches_without_adding_support_and_is_idempotent(self):
        records = [record("a", "anyone else think this map looks weird")]
        records += [record(str(i), text, minute=i, user=str(i % 3))
                    for i, text in enumerate(["yes", "yea", "bad", "no", "same"], 1)]
        pool, _ = self.pool(records)
        original = next(c for c in pool if c["kind"] == "cluster")
        merged = merge_related_bursts(pool)
        self.assertFalse(any(c["kind"] == "burst" for c in merged))
        cluster = next(c for c in merged if c["kind"] == "cluster")
        self.assertEqual(cluster["member_ids"], original["member_ids"])
        self.assertEqual(cluster["history"], original["history"])
        self.assertEqual(cluster["diagnostics"]["discussion_burst_count"], 1)
        self.assertEqual(cluster["score"], original["score"])
        self.assertEqual(cluster["ranking_engagement"], 5)
        repeated = next(c for c in merge_related_bursts(merged) if c["kind"] == "cluster")
        self.assertEqual(cluster["score"], repeated["score"])
        self.assertEqual(repeated["diagnostics"]["discussion_burst_count"], 1)
        self.assertNotIn("discussion_bursts", original["diagnostics"])

    def test_ambiguous_burst_and_unclustered_burst_stay_separate(self):
        cluster = {"id": "cluster:1", "kind": "cluster", "score": 1, "emerging_score": 1, "diagnostics": {}}
        burst = {"id": "burst:a", "kind": "burst", "anchor_id": "a", "score": 1, "emerging_score": 1,
                 "diagnostics": {}, "related_clusters": ["cluster:1", "cluster:2"]}
        self.assertEqual(len(merge_related_bursts([cluster, burst])), 2)
        burst["related_clusters"] = []
        self.assertEqual(len(merge_related_bursts([cluster, burst])), 2)

    def test_merged_burst_context_is_capped_and_id_lists_are_internal(self):
        records = [record("a", "anyone else think this map looks weird")]
        records += [record(str(i), "yes", minute=i // 5, user=str(i % 3)) for i in range(1, 41)]
        pool, values = self.pool(records)
        cluster = next(c for c in merge_related_bursts(pool) if c["kind"] == "cluster")
        packet = sample_candidate(cluster, *values[:3], records, "2026-08-01")
        followups = [r for r in packet["context"] if r["role"] == "possible_following_response"]
        self.assertLessEqual(len(followups), 10)
        self.assertGreater(len(followups), 0)
        self.assertLessEqual(len(packet["context"]), 20)
        self.assertEqual(len(packet["context"]), len({r["id"] for r in packet["context"]}))
        self.assertNotIn("possible_response_ids", packet["diagnostics"]["discussion_bursts"][0])
        self.assertIn("possible_response_ids", cluster["diagnostics"]["discussion_bursts"][0])

    def test_context_same_channel_time_bounded_and_deduplicated(self):
        records = [record("old", minute=-10), record("other", minute=-1, channel="elsewhere"),
                   record("parent", "Desert mission discussion", minute=-1),
                   record("a", "Mission takes too long", parent="parent"),
                   record("future", minute=1, day="2026-08-02")]
        values = inputs(records)
        pool = build_candidate_pool(*values, records, records, "2026-08-01")
        candidate = next(c for c in pool if c["kind"] == "cluster")
        candidate["indices"] = [next(i for i, u in enumerate(values[0]) if u["voice_id"] == "a")]
        packet = sample_candidate(candidate, *values[:3], records, "2026-08-01")
        self.assertEqual([c["id"] for c in packet["context"]], ["parent"])
        self.assertEqual(packet["context"][0]["role"], "explicit_parent")

    def test_explicit_response_can_be_following_but_not_future(self):
        records = [record("a"), record("response", "+1", minute=1, parent="a"),
                   record("future", "+1", minute=1, parent="a", day="2026-08-02")]
        pool, values = self.pool(records)
        cluster = next(c for c in pool if c["kind"] == "cluster")
        packet = sample_candidate(cluster, *values[:3], records, "2026-08-01")
        self.assertEqual([c["id"] for c in packet["context"]], ["response"])

    def test_sampling_caps_targets_and_deduplicates_text(self):
        records = [record(str(i), "feedback variant " + str(i), minute=i) for i in range(250)]
        records += [record("duplicate", "feedback variant 1", minute=251)]
        pool, values = self.pool(records)
        candidate = next(c for c in pool if c["kind"] == "cluster")
        packet = sample_candidate(candidate, *values[:3], records, "2026-08-01", limit=100)
        self.assertEqual(len(packet["targets"]), 10)
        self.assertEqual(len({t["text"] for t in packet["targets"]}), 10)
        self.assertLessEqual(len(packet["context"]), 20)

    def test_multiple_clauses_from_one_voice_do_not_consume_sample_slots(self):
        records = [record(str(i), "feedback variant " + str(i), minute=i) for i in range(8)]
        values = list(inputs(records))
        values[0] += [{**values[0][0], "unit_id": "0:1", "text": "second concern in same source"},
                      {**values[0][0], "unit_id": "0:2", "text": "third concern in same source"}]
        values[1] = np.asarray([[1., 0.] for _ in values[0]], dtype=np.float32)
        values[2] = np.arange(len(values[0]))
        values[3] = np.zeros(len(values[0]), dtype=int)
        pool = build_candidate_pool(*values, records, records, "2026-08-01")
        packet = sample_candidate(pool[0], *values[:3], records, "2026-08-01", limit=7)
        self.assertEqual(len(packet["targets"]), 7)
        self.assertEqual(len({t["id"] for t in packet["targets"]}), 7)

    def test_balanced_batch_disjoint_and_reviewed_excluded(self):
        pool = [{"id": str(i), "score": 100 - i, "emerging_score": i} for i in range(20)]
        batch = choose_batch(pool, {"0"}, n=10)
        self.assertEqual([c["id"] for c in batch[:5]], ["1", "2", "3", "4", "5"])
        self.assertEqual([c["id"] for c in batch[5:]], ["6", "7", "8", "9", "10"])
        self.assertEqual(len({c["id"] for c in batch}), 10)

    def test_overall_prefers_supported_changes_but_unusual_queue_keeps_stable_groups(self):
        changing = [{"id": "changing" + str(i), "score": 20 - i, "emerging_score": 1,
                     "history": {"status_supported": True, "status": "growing"}} for i in range(8)]
        stable = [{"id": "stable" + str(i), "score": 1000 - i, "emerging_score": 100 - i,
                   "history": {"status_supported": False, "status": None}} for i in range(8)]
        batch = choose_batch(stable + changing, set(), n=10, ratio=.5)
        self.assertEqual([c["id"] for c in batch[:5]], ["changing" + str(i) for i in range(5)])
        self.assertEqual([c["id"] for c in batch[5:8]], ["changing" + str(i) for i in range(5, 8)])
        self.assertEqual([c["id"] for c in batch[8:]], ["stable0", "stable1"])

    def test_positive_exploration_is_not_starved_by_high_scoring_fading_groups(self):
        fading = [{"id": "fading" + str(i), "score": 100 - i, "emerging_score": 1000 - i,
                   "unusual_score": 2000 - i, "history": {"status_supported": True, "status": "fading"}}
                  for i in range(20)]
        positive = [{"id": "positive" + str(i), "score": 1, "emerging_score": 3 - i,
                     "unusual_score": 3 - i, "history": {"status_supported": True, "status": status}}
                    for i, status in enumerate(["new", "growing", "new"])]
        batch = choose_batch(fading + positive, set())
        self.assertEqual([c["id"] for c in batch[5:8]], ["positive0", "positive1", "positive2"])
        self.assertTrue(all(c["history"]["status"] == "fading" for c in batch[8:]))

    def test_positive_exploration_backfills_unusual_when_few_positive_groups(self):
        pool = [{"id": str(i), "score": 100 - i, "emerging_score": i,
                 "history": {"status_supported": True, "status": "fading"}}
                for i in range(12)]
        batch = choose_batch(pool, set())
        self.assertEqual(len(batch), 10)
        self.assertEqual(len({c["id"] for c in batch}), 10)

    @staticmethod
    def neighbor(identifier, cosine, status=None, members=()):
        return {"id": identifier, "kind": "cluster", "score": 10, "indices": [],
                "member_ids": list(members), "center": np.array([cosine, np.sqrt(1 - cosine ** 2)]),
                "history": {"status": status, "status_supported": bool(status),
                            "stats": {"recent_window": {"support": 8}, "prior_window": {"support": 5}}}}

    def test_opposite_direction_at_sixth_neighbor_precedes_nearer_stable_groups(self):
        seed = self.neighbor("seed", 1., "fading", ["a", "b", "c", "d", "e"])
        pool = [seed] + [self.neighbor("near" + str(i), .99 - i * .01) for i in range(8)]
        pool[6]["history"].update(status="growing", status_supported=True)
        concept = {"id": "concern", "candidate_ids": ["seed:refine:1"], "member_ids": seed["member_ids"],
                   "history": seed["history"], "score": 10}
        selected = choose_neighbors(pool, {"seed"}, [concept], limit=1)
        self.assertEqual(selected[0]["id"], "near5")
        context = selected[0]["neighbor_context"]
        self.assertEqual(context["geometric_neighbor_rank"], 6)
        self.assertTrue(context["opposing_direction"])
        self.assertEqual(context["reason"], "opposing_direction")

    def test_strong_source_overlap_nominates_even_outside_geometry_window(self):
        seed = self.neighbor("seed", 1., "growing", ["a", "b", "c", "d", "e"])
        near = [self.neighbor("near" + str(i), .99 - i * .01) for i in range(9)]
        shared = self.neighbor("shared", .2, "growing", ["a", "b", "c", "d", "f"])
        concept = {"id": "concern", "candidate_ids": ["seed"], "member_ids": seed["member_ids"],
                   "history": seed["history"]}
        selected = choose_neighbors([seed] + near + [shared], {"seed"}, [concept], limit=1)
        self.assertEqual(selected[0]["id"], "shared")
        self.assertGreater(selected[0]["neighbor_context"]["geometric_neighbor_rank"], 8)
        self.assertEqual(selected[0]["neighbor_context"]["source_overlap_count"], 4)
        self.assertEqual(selected[0]["neighbor_context"]["reason"], "source_overlap")

    def test_neighbor_geometry_only_nominates_and_does_not_merge_or_mutate(self):
        seed = self.neighbor("seed", 1., "new", ["a", "b", "c"])
        nearby = self.neighbor("different_concern", .999, "growing", ["x", "y", "z"])
        concept = {"id": "concern", "candidate_ids": ["seed"], "member_ids": set(seed["member_ids"]),
                   "history": seed["history"]}
        selected = choose_neighbors([seed, nearby], {"seed"}, [concept])
        self.assertEqual([c["id"] for c in selected], ["different_concern"])
        self.assertEqual(selected[0]["member_ids"], ["x", "y", "z"])
        self.assertEqual(concept["member_ids"], {"a", "b", "c"})
        self.assertNotIn("neighbor_context", nearby)
        self.assertIsNot(selected[0], nearby)
        self.assertEqual(choose_neighbors([seed, nearby], {"seed", "different_concern"}, [concept]), [])

    def test_neighbor_nominations_are_unique_and_capped_per_concept(self):
        seed = self.neighbor("seed", 1., "growing", ["a", "b", "c"])
        neighbors = [self.neighbor("near" + str(i), .99 - .01 * i) for i in range(5)]
        concept = {"id": "concern", "candidate_ids": ["seed"], "member_ids": seed["member_ids"],
                   "history": seed["history"]}
        selected = choose_neighbors([seed] + neighbors, {"seed"}, [concept, dict(concept, id="other")], limit=10)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({c["id"] for c in selected}), 2)

    def test_stable_groups_remain_available_after_changing_candidates_exhausted(self):
        changing = {"id": "changing", "score": 1, "emerging_score": 1, "history": {"status_supported": True}}
        stable = [{"id": str(i), "score": 100 - i, "emerging_score": i,
                   "history": {"status_supported": False}} for i in range(4)]
        batch = choose_batch([changing] + stable, {"changing"}, n=10, ratio=.5)
        self.assertEqual(len(batch), 4)
        self.assertEqual({c["id"] for c in batch}, {str(i) for i in range(4)})

    def test_batch_has_no_total_review_cap(self):
        pool = [{"id": str(i), "score": i, "emerging_score": i} for i in range(130)]
        batch = choose_batch(pool, {str(i) for i in range(120)}, n=10)
        self.assertEqual(len(batch), 10)

    def test_unusual_only_ratio_and_exhaustion(self):
        pool = [{"id": str(i), "score": 10 - i, "emerging_score": i} for i in range(3)]
        self.assertEqual([c["id"] for c in choose_batch(pool, [], n=10, ratio=1)], ["0", "1", "2"])

    def test_default_sample_uses_ten_distinct_sources_without_size_breakpoints(self):
        for count in (8, 20, 60):
            with self.subTest(count=count):
                records = [record(str(i), f"Varied substantive feedback {i}", minute=i) for i in range(count)]
                pool, values = self.pool(records)
                candidate = next(c for c in pool if c["kind"] == "cluster")
                packet = sample_candidate(candidate, *values[:3], records, "2026-08-01")
                self.assertEqual(len(packet["targets"]), min(count, 10))

    def test_cohesion_is_diagnostic_not_a_priority_multiplier(self):
        records = [record(str(i), f"Varied substantive feedback {i}", minute=i) for i in range(20)]
        values = list(inputs(records))
        compact = build_candidate_pool(*values, records, records, "2026-08-01")[0]
        values[1] = np.asarray([[0., 1.] for _ in values[0]], dtype=np.float32)
        dispersed = build_candidate_pool(*values, records, records, "2026-08-01")[0]
        self.assertEqual(compact["score"], dispersed["score"])
        self.assertEqual(activity_key(compact), activity_key(dispersed))
        self.assertNotEqual(compact["diagnostics"]["mean_cosine_to_center"],
                            dispersed["diagnostics"]["mean_cosine_to_center"])

    def test_main_activity_uses_ordered_ties_without_adding_breadth_or_engagement(self):
        def candidate(identifier, score, days, platforms, engagement):
            return {"id": identifier, "score": score, "ranking_engagement": engagement,
                    "history": {"stats": {"ranking_active_days": days, "ranking_platform_count": platforms}}}
        rows = [candidate("reach", 11, 1, 1, 0), candidate("days", 10, 2, 1, 0),
                candidate("platforms", 10, 1, 2, 0), candidate("engagement", 10, 1, 1, 1000)]
        self.assertEqual([c["id"] for c in sorted(rows, key=activity_key)],
                         ["reach", "days", "platforms", "engagement"])

    def test_growth_queue_uses_growth_then_recent_support_not_legacy_weights(self):
        pool = [{"id": name, "score": total, "emerging_score": legacy,
                 "history": {"status": "growing", "status_supported": True,
                             "stats": {"normalized_growth_ratio": growth, "ranking_recent_support": recent}}}
                for name, total, growth, recent, legacy in (
                    ("ordinary", 100, 1.4, 50, 999), ("early", 8, 4., 7, 0),
                    ("more-support", 12, 2., 10, 0), ("less-support", 9, 2., 6, 9999))]
        batch = choose_batch(pool, set(), n=4, ratio=1)
        self.assertEqual([c["id"] for c in batch[:2]], ["early", "more-support"])

    def test_unusual_interleaving_preserves_coverage_across_review_batches(self):
        pool = []
        for index in range(3):
            pool += [{"id": f"fade{index}", "score": 100 - index,
                      "history": {"status": "fading", "stats": {"normalized_growth_ratio": .1 + index * .1}}},
                     {"id": f"burst{index}", "kind": "burst", "score": 1,
                      "diagnostics": {"engagement_score": 30 - index}, "history": {"status": None}},
                     {"id": f"stable{index}", "score": 200 - index, "history": {"status": None}}]
        reviewed, ordered = set(), []
        for _ in range(3):
            batch = choose_batch(pool, reviewed, n=2, ratio=1)
            ordered.extend(c["id"] for c in batch)
            reviewed.update(c["id"] for c in batch)
        self.assertEqual(ordered, ["fade0", "burst0", "stable0", "fade1", "burst1", "stable1"])

    def test_context_index_cutoff_must_match(self):
        records = [record("a")]
        pool, values = self.pool(records)
        with self.assertRaises(ValueError):
            sample_candidate(pool[0], *values[:3], records, "2026-08-01",
                             context_index=build_context_index(records, "2026-08-02"))

    def test_reply_engagement_requires_same_stream_and_parent_before_child(self):
        rows = [record("a"), record("cross_channel", "+1", minute=1, parent="a", channel="other"),
                record("before_parent", "+1", minute=-1, parent="a"),
                record("valid", "+1", minute=2, parent="a")]
        index = build_context_index(rows, "2026-08-01")
        self.assertEqual([r["id"] for r in index["replies"]["a"]], ["valid"])


if __name__ == "__main__":
    unittest.main()
