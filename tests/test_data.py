import json
import tempfile
import unittest
from pathlib import Path
from trend_hunter.data import prepare, make_units, normalize, reaction_kind


class DataTests(unittest.TestCase):
    def run_prepare(self, rows, cutoff="2026-07-01"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voices.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            return prepare(path, cutoff)

    def voice(self, id, text, stamp="2026-07-01T12:00:00+08:00", parent=None):
        return dict(id=id,content=text,title=None,parent_id=parent,platform="QQ",channel="chat",user_name=id,created_at=stamp)

    def test_future_parent_cannot_supply_context(self):
        visible, eligible, _ = self.run_prepare([self.voice("a", "确实",parent="b"),self.voice("b","每日任务负担很重","2026-07-02T00:00:00+08:00")])
        self.assertEqual(len(visible),1)
        self.assertEqual(visible[0]["context_ids"],[])
        self.assertEqual(make_units(eligible),[])

    def test_china_day_cutoff(self):
        rows=[self.voice("a","每日任务负担很重","2026-07-01T15:59:59+00:00"),self.voice("b","每日任务负担很重","2026-07-01T16:00:00+00:00")]
        self.assertEqual([r["id"] for r in self.run_prepare(rows)[0]],["a"])

    def test_substantive_reply_keeps_own_topic(self):
        rows=[self.voice("a","全新角色上线了"),self.voice("b","我的游戏一直闪退。匹配也不公平",parent="a")]
        _,eligible,_=self.run_prepare(rows)
        units=[u for u in make_units(eligible) if u["voice_id"]=="b"]
        self.assertEqual(len(units),2)
        self.assertFalse(any("角色" in u["text"] for u in units))

    def test_own_title_and_content_are_separate_contributions_without_parent_text(self):
        parent = self.voice("parent", "原帖讨论的是其他话题")
        child = self.voice("child", "等待时间实在太长", parent="parent")
        child["title"] = "关于合作模式的体验"
        _, eligible, _ = self.run_prepare([parent, child])
        units = [unit for unit in make_units(eligible) if unit["voice_id"] == "child"]
        self.assertEqual([unit["text"] for unit in units], ["等待时间实在太长", "关于合作模式的体验"])
        self.assertEqual([unit["unit_id"] for unit in units], ["child:0", "child:1"])
        self.assertFalse(any("原帖" in unit["text"] for unit in units))

    def test_duplicate_title_and_content_clauses_appear_once_per_source(self):
        row = self.voice("a", "关于合作模式的体验。等待时间实在太长。关于合作模式的体验")
        row["title"] = "关于合作模式的体验"
        _, eligible, _ = self.run_prepare([row])
        self.assertEqual([unit["text"] for unit in make_units(eligible)],
                         ["关于合作模式的体验", "等待时间实在太长"])

    def test_fifth_and_later_substantive_clauses_are_preserved(self):
        clauses = ["第一段包含完整观察", "第二段包含其他观察", "第三段描述实际体验",
                   "第四段提出相关问题", "第五段补充重要信息", "第六段给出不同意见"]
        _, eligible, _ = self.run_prepare([self.voice("a", "。".join(clauses))])
        units = make_units(eligible)
        self.assertEqual([unit["text"] for unit in units], clauses)
        self.assertEqual([unit["unit_id"] for unit in units], [f"a:{index}" for index in range(6)])

    def test_meaningful_title_only_message_is_eligible_and_embedded(self):
        row = self.voice("a", None)
        row["title"] = "等待时间实在太长"
        visible, eligible, _ = self.run_prepare([row])
        self.assertFalse(visible[0]["trivial"])
        self.assertEqual([record["id"] for record in eligible], ["a"])
        self.assertEqual([unit["text"] for unit in make_units(eligible)], ["等待时间实在太长"])

    def test_doge_preserved_and_image_not_invented(self):
        self.assertIn("doge",normalize("毫无压力<emoji:doge>"))
        self.assertEqual(normalize("<image:abc>"),"[image unavailable]")

    def test_image_only_is_not_invented_text(self):
        _,eligible,_=self.run_prepare([self.voice("a","<image:abc>")])
        self.assertEqual(make_units(eligible),[])

    def test_image_and_emoji_metadata_cannot_create_a_semantic_unit(self):
        original = "<image:opaque-id><emoji:very-long-platform-emoji-identifier>"
        visible, eligible, stats = self.run_prepare([self.voice("a", original)])
        self.assertEqual(visible[0]["content"], original)
        self.assertEqual(visible[0]["image_refs"], ["opaque-id"])
        self.assertTrue(visible[0]["media_only"])
        self.assertTrue(visible[0]["trivial"])
        self.assertEqual(eligible, [])
        self.assertEqual(make_units(eligible), [])
        self.assertEqual(stats["media_only_messages"], 1)

    def test_discord_emoji_normalization_preserves_names_without_identifiers(self):
        text = normalize("<:smile:1184769647954894858><a:wave:123456789012345678>")
        self.assertEqual(text, "[smile] [wave]")
        self.assertNotIn("1184769647954894858", text)
        self.assertNotIn("123456789012345678", text)

    def test_discord_emoji_only_messages_cannot_create_semantic_units(self):
        rows = [self.voice("static", "<:smile:1184769647954894858>"),
                self.voice("animated", "<a:wave:123456789012345678>")]
        visible, eligible, stats = self.run_prepare(rows)
        self.assertTrue(all(row["trivial"] for row in visible))
        self.assertEqual(eligible, [])
        self.assertEqual(make_units(eligible), [])
        self.assertEqual(stats["media_only_messages"], 0)

    def test_image_and_discord_emoji_are_media_only_without_readable_text(self):
        row = self.voice("a", "<image:opaque-id><:smile:1184769647954894858><a:wave:123456789012345678>")
        visible, eligible, stats = self.run_prepare([row])
        self.assertEqual(visible[0]["content"], row["content"])
        self.assertTrue(visible[0]["media_only"])
        self.assertEqual(eligible, [])
        self.assertEqual(make_units(eligible), [])
        self.assertEqual(stats["media_only_messages"], 1)

    def test_substantive_text_keeps_discord_emoji_names_without_metadata_length(self):
        row = self.voice("a", "这个界面使用很方便<:smile:1184769647954894858><a:wave:123456789012345678>")
        visible, eligible, _ = self.run_prepare([row])
        self.assertEqual(visible[0]["content"], row["content"])
        self.assertEqual([unit["text"] for unit in make_units(eligible)], ["这个界面使用很方便 [smile] [wave]"])
        _, short_eligible, _ = self.run_prepare([self.voice("short", "嗯嗯<:long_human_readable_name:1184769647954894858>")])
        self.assertEqual([unit["text"] for unit in make_units(short_eligible)], ["嗯嗯 [long_human_readable_name]"])

    def test_discord_emoji_can_supply_irony_cue_to_a_minimal_reaction(self):
        visible, eligible, _ = self.run_prepare([self.voice("a", "确实<:doge:1184769647954894858>")])
        self.assertEqual(visible[0]["reaction"], "uncertain")
        self.assertEqual(make_units(eligible), [])

    def test_mixed_image_caption_keeps_emoji_cue_without_placeholder_embedding(self):
        original = "这个界面操作很方便<image:opaque-id><emoji:doge>"
        visible, eligible, stats = self.run_prepare([self.voice("a", original)])
        self.assertEqual(visible[0]["content"], original)
        self.assertEqual(visible[0]["image_count"], 1)
        self.assertFalse(visible[0]["media_only"])
        units = make_units(eligible)
        self.assertEqual([unit["text"] for unit in units], ["这个界面操作很方便 [doge]"])
        self.assertNotIn("image unavailable", units[0]["text"])
        self.assertNotIn("image unavailable", eligible[0]["family"])
        self.assertEqual(stats["image_messages_with_text"], 1)

    def test_short_caption_survives_without_embedding_image_or_emoji_identifiers(self):
        _, eligible, _ = self.run_prepare([
            self.voice("a", "嗯嗯<image:opaque-id><emoji:very-long-identifier>")])
        self.assertEqual([unit["text"] for unit in make_units(eligible)], ["嗯嗯 [very-long-identifier]"])
        self.assertNotIn("opaque-id", make_units(eligible)[0]["text"])

    def test_meaningful_title_survives_image_and_emoji_only_body(self):
        row = self.voice("a", "<image:opaque-id><emoji:platform-identifier>")
        row["title"] = "关于界面布局的建议"
        visible, eligible, stats = self.run_prepare([row])
        self.assertFalse(visible[0]["trivial"])
        self.assertFalse(visible[0]["media_only"])
        self.assertEqual(eligible[0]["family"], row["title"])
        self.assertEqual([unit["text"] for unit in make_units(eligible)], [row["title"]])
        self.assertEqual(stats["media_only_messages"], 0)
        self.assertEqual(stats["image_messages_with_text"], 1)

    def test_short_title_clauses_survive_separately_despite_media_only_body(self):
        row = self.voice("a", "<image:opaque-id><emoji:platform-identifier>")
        row["title"] = "布局。难用"
        _, eligible, _ = self.run_prepare([row])
        self.assertEqual([unit["text"] for unit in make_units(eligible)], ["布局", "难用"])
        self.assertEqual([unit["unit_id"] for unit in make_units(eligible)], ["a:0", "a:1"])

    def test_readable_short_contributions_are_not_discarded_by_character_count(self):
        texts = ["慢", "不便", "不舒服", "ugh", "X"]
        rows = [self.voice(f"short-{index}", text) for index, text in enumerate(texts)]
        visible, eligible, _ = self.run_prepare(rows)
        units = make_units(eligible)
        self.assertEqual([unit["text"] for unit in units], texts)
        self.assertEqual([unit["voice_id"] for unit in units], [row["id"] for row in rows])
        self.assertEqual([unit["unit_id"] for unit in units], [row["id"] + ":0" for row in rows])
        self.assertEqual([row["content"] for row in visible], texts)

    def test_short_own_text_keeps_source_provenance_without_inheriting_parent_meaning(self):
        parent = self.voice("parent", "A separate surrounding discussion.")
        child = self.voice("child", "慢<image:opaque-id><emoji:uncertain>", parent="parent")
        child["title"] = "不便"
        visible, eligible, _ = self.run_prepare([parent, child])
        units = [unit for unit in make_units(eligible) if unit["voice_id"] == "child"]
        self.assertEqual([unit["text"] for unit in units], ["慢 [uncertain]", "不便"])
        self.assertEqual([unit["unit_id"] for unit in units], ["child:0", "child:1"])
        self.assertTrue(all(unit["day"] == "2026-07-01" and unit["platform"] == "QQ" for unit in units))
        self.assertEqual(next(row for row in visible if row["id"] == "child")["content"], child["content"])
        self.assertFalse(any(parent["content"] in unit["text"] for unit in units))

    def test_short_shared_body_cannot_hide_distinct_titles_in_text_family_identity(self):
        first, second = self.voice("first", "嗯嗯"), self.voice("second", "嗯嗯")
        first["title"], second["title"] = "第一份独立观察", "另一份不同观察"
        visible, eligible, _ = self.run_prepare([first, second])
        families = {row["id"]: row["family"] for row in visible}
        self.assertEqual(families["first"], "嗯嗯\n" + first["title"])
        self.assertEqual(families["second"], "嗯嗯\n" + second["title"])
        self.assertNotEqual(families["first"], families["second"])
        self.assertEqual([unit["text"] for unit in make_units(eligible)],
                         ["嗯嗯", first["title"], "嗯嗯", second["title"]])

    def test_substantive_title_survives_minimal_reaction_body_with_or_without_image(self):
        for body in ("同意", "同意<image:opaque-id>"):
            with self.subTest(body=body):
                row = self.voice("a", body)
                row["title"] = "关于界面布局的建议"
                visible, eligible, _ = self.run_prepare([row])
                self.assertIsNone(visible[0]["reaction"])
                self.assertEqual(visible[0]["family"], row["title"])
                self.assertEqual([unit["text"] for unit in make_units(eligible)], [row["title"]])

    def test_substantive_title_survives_greeting_body_with_or_without_image(self):
        for body in ("hello", "hello<image:opaque-id>", "晚安", "晚安<image:opaque-id>"):
            with self.subTest(body=body):
                row = self.voice("a", body)
                row["title"] = "关于界面布局的建议"
                visible, eligible, _ = self.run_prepare([row])
                self.assertFalse(visible[0]["trivial"])
                self.assertEqual(visible[0]["family"], row["title"])
                self.assertEqual([unit["text"] for unit in make_units(eligible)], [row["title"]])

    def test_pure_reaction_retains_its_contextual_role_and_family(self):
        row = self.voice("a", "同意<image:opaque-id>")
        visible, eligible, _ = self.run_prepare([row])
        self.assertEqual(visible[0]["reaction"], "agree")
        self.assertEqual(visible[0]["family"], "同意")
        self.assertEqual(make_units(eligible), [])

    def test_image_reference_punctuation_never_becomes_semantic_text(self):
        row = self.voice("a", "界面使用很方便<image:opaque?reference!with;punctuation>")
        _, eligible, _ = self.run_prepare([row])
        self.assertEqual([unit["text"] for unit in make_units(eligible)], ["界面使用很方便"])

    def test_empty_sources_and_media_only_sources_stay_visible_without_units(self):
        rows = [self.voice("empty", None), self.voice("blank", "  "),
                self.voice("media", "<image:opaque-id>")]
        visible, eligible, stats = self.run_prepare(rows)
        self.assertEqual(len(visible), 3)
        self.assertEqual(eligible, [])
        self.assertEqual(make_units(eligible), [])
        self.assertEqual(stats["media_only_messages"], 1)

    def test_media_cleanup_preserves_all_substantive_clauses_of_long_source(self):
        clauses = ["第一段包含完整观察", "第二段包含其他观察", "第三段描述实际体验",
                   "第四段提出相关问题", "第五段补充重要信息", "第六段给出不同意见"]
        original = "。".join(clause + f"<image:asset-{index}>" for index, clause in enumerate(clauses))
        visible, eligible, _ = self.run_prepare([self.voice("a", original)])
        self.assertEqual(visible[0]["content"], original)
        self.assertEqual(visible[0]["image_count"], 6)
        self.assertEqual([unit["text"] for unit in make_units(eligible)], clauses)

    def test_media_only_diagnostic_is_not_a_synonym_for_trivial_text(self):
        rows = [self.voice("media", "<image:asset><emoji:platform-id>"),
                self.voice("greeting", "晚安<image:asset-2>"),
                self.voice("caption", "分享界面使用感受<image:asset-3>")]
        _, _, stats = self.run_prepare(rows)
        self.assertEqual(stats["image_messages"], 3)
        self.assertEqual(stats["media_only_messages"], 1)
        self.assertEqual(stats["image_messages_with_text"], 2)
        self.assertEqual(stats["image_messages_excluded_as_trivial"], 2)

    def test_image_counts_preserve_refs_without_future_or_pixel_inference(self):
        rows = [self.voice("a", "<image:first><image:second>"),
                self.voice("b", "看看这个表现 <image:third>"),
                self.voice("future", "<image:future>", "2026-07-02T00:00:00+08:00")]
        visible, eligible, stats = self.run_prepare(rows)
        by_id = {r["id"]: r for r in visible}
        self.assertEqual(by_id["a"]["image_refs"], ["first", "second"])
        self.assertEqual(by_id["a"]["image_count"], 2)
        self.assertEqual(by_id["b"]["image_count"], 1)
        self.assertEqual(stats["image_messages"], 2)
        self.assertEqual(stats["image_placeholders"], 3)
        self.assertEqual(stats["media_only_messages"], 1)
        self.assertNotIn("a", {u["voice_id"] for u in make_units(eligible)})
        self.assertNotIn("future", by_id)

    def test_parent_order_uses_instants_not_timestamp_strings(self):
        rows=[self.voice("p","每日任务负担很重","2026-07-01T05:00:00+00:00"),self.voice("c","确实","2026-07-01T14:00:00+08:00",parent="p")]
        visible,_,_=self.run_prepare(rows)
        self.assertEqual([r["id"] for r in visible],["p","c"])
        self.assertEqual(visible[1]["context_ids"],["p"])

    def test_irony_reaction_abstains_and_same_experience_is_contextual(self):
        self.assertEqual(reaction_kind(normalize("确实<emoji:doge>")),"uncertain")
        self.assertEqual(reaction_kind("我也是"),"same_experience")
        self.assertEqual(reaction_kind("同问"),"question")


if __name__ == "__main__":
    unittest.main()
