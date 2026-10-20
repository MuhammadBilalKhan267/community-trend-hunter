from copy import deepcopy
import unittest

from trend_hunter.aliases import remember_review_aliases, select_packet_aliases


def source(identifier, text, stamp="2026-07-01T10:00:00+08:00", channel="feedback"):
    return {"id": identifier, "content": text, "title": None, "created_at": stamp,
            "platform": "Forum", "channel": channel}


class AliasContextTests(unittest.TestCase):
    def setUp(self):
        self.by_id = {"s1": source("s1", "星舟这个外号指的是凌月。"),
                      "s2": source("s2", "对，大家在这里这样称呼她。", "2026-07-02T10:00:00+08:00")}
        self.alias = {"alias": "星舟", "referent": "凌月", "evidence_ids": ["s1", "s2"], "confidence": .9}

    def test_joint_source_context_and_dates_are_preserved_without_rewriting_text(self):
        original = deepcopy(self.by_id)
        registry = remember_review_aliases([], [self.alias], self.by_id, "2026-07-02")
        self.assertEqual(len(registry), 1)
        assertion = registry[0]
        self.assertEqual(assertion["name"], "凌月")
        self.assertEqual(assertion["alias"], "星舟")
        self.assertEqual(assertion["source_text"], self.by_id["s1"]["content"])
        self.assertEqual(assertion["evidence_ids"], ["s1", "s2"])
        self.assertEqual([row["text"] for row in assertion["evidence"]],
                         [self.by_id["s1"]["content"], self.by_id["s2"]["content"]])
        self.assertEqual(assertion["scope"]["observed_through"], "2026-07-02")
        self.assertEqual(assertion["scope"]["kind"], "reviewed_context")
        self.assertEqual(self.by_id, original)

    def test_unknown_or_future_joint_evidence_rejects_the_assertion(self):
        self.assertEqual(remember_review_aliases([], [self.alias], self.by_id, "2026-07-01"), [])
        alias = {**self.alias, "evidence_ids": ["s1", "unknown"]}
        self.assertEqual(remember_review_aliases([], [alias], self.by_id, "2026-07-02"), [])

    def test_game_timezone_controls_cutoff_even_if_record_day_claims_otherwise(self):
        record = source("late", "星舟指凌月。", "2026-07-01T23:30:00+05:00")
        record["day"] = "2026-07-01"
        alias = {**self.alias, "evidence_ids": ["late"]}
        self.assertEqual(remember_review_aliases([], [alias], {"late": record}, "2026-07-01"), [])

    def test_repeated_reviews_deduplicate_same_source_set_and_preserve_literal_registry(self):
        literal = {"name": "凌月", "alias": "星舟", "source_id": "s1", "source_text": self.by_id["s1"]["content"],
                   "created_at": self.by_id["s1"]["created_at"], "observed_assertion_count": 3}
        original = deepcopy(literal)
        same = {**self.alias, "evidence_ids": ["s1"]}
        result = remember_review_aliases([literal], [same], self.by_id, "2026-07-02")
        self.assertEqual(result, [original])
        added = remember_review_aliases(result, [self.alias], self.by_id, "2026-07-02")
        repeated = remember_review_aliases(added, [{**self.alias, "evidence_ids": ["s2", "s1", "s1"]}],
                                           self.by_id, "2026-07-02")
        self.assertEqual(len(repeated), 2)
        self.assertEqual(literal, original)

    def test_packet_selection_uses_target_or_context_mentions_and_preserves_scope(self):
        registry = remember_review_aliases([], [self.alias], self.by_id, "2026-07-02")
        packet = {"as_of": "2026-07-02", "targets": [{"text": "星舟有点难操作。"}], "context": []}
        original = deepcopy(packet)
        selected = select_packet_aliases(registry, packet)
        self.assertEqual(selected, registry)
        self.assertEqual(packet, original)
        selected[0]["name"] = "changed"
        self.assertEqual(registry[0]["name"], "凌月")
        self.assertEqual(select_packet_aliases(registry, {"targets": [{"text": "其他角色讨论"}]}), [])
        self.assertEqual(len(select_packet_aliases(registry, {"context": [{"text": "凌月怎么样？"}]})), 1)

    def test_future_contextual_assertion_is_not_selected_for_earlier_packet(self):
        registry = remember_review_aliases([], [self.alias], self.by_id, "2026-07-02")
        packet = {"as_of": "2026-07-01", "targets": [{"text": "星舟"}]}
        self.assertEqual(select_packet_aliases(registry, packet), [])

    def test_latin_terms_do_not_match_inside_unrelated_words(self):
        registry = [{"name": "Mira", "alias": "Star", "source_id": "s1",
                     "source_text": "Mira is called Star here.", "created_at": "2026-07-01T10:00:00+08:00"}]
        self.assertEqual(select_packet_aliases(registry, {"targets": [{"text": "starting tomorrow"}]}), [])
        self.assertEqual(len(select_packet_aliases(registry, {"targets": [{"text": "STAR feels different"}]})), 1)

    def test_invalid_or_self_aliases_do_not_enter_registry(self):
        for alias in ({**self.alias, "referent": "星舟"}, {**self.alias, "evidence_ids": []},
                      {**self.alias, "alias": ""}, {**self.alias, "evidence_ids": [None]}):
            with self.subTest(alias=alias):
                self.assertEqual(remember_review_aliases([], [alias], self.by_id, "2026-07-02"), [])


if __name__ == "__main__":
    unittest.main()
