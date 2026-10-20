"""Sample-review policy and provider boundaries, without inference or credentials."""
from copy import deepcopy
import json
import re
import unittest

from trend_hunter.review import (DRAFT_SCHEMA, PROPOSAL_SCHEMA, final_check,
                                 final_draft, review_candidate, source_proposals, compose_why, extractive_fallback,
                                 review_final_duplicates)
from trend_hunter.local_models import LocalModelError, _validate_schema


def proposal(**changes):
    item = {"title": "玩家反馈", "title_en": "Player feedback", "category": "Experience",
            "definition": "Players discuss a shared aspect of their experience.",
            "scope": "The subject identified in the supplied sources.", "uncertainty": ""}
    item.update(changes)
    return item


def packet(count=10):
    return {"id": "candidate-a", "kind": "cluster",
            "targets": [{"id": f"v{i}", "text": f"玩家体验反馈 {i}", "author": f"author-{i}"}
                        for i in range(count)],
            "context": [{"id": "parent", "text": "这里是上下文", "role": "possible_context"}],
            "metadata": {"assigned_messages": 500}}


def choice(value, confidence=.95):
    return {"type": "choice", "choice": value, "confidence": confidence,
            "probabilities": {value: confidence}}


class Local:
    def __init__(self, result=None):
        self.result = result or {"concerns": [proposal()], "aliases": []}
        self.calls = []

    def chat_json(self, system, payload, schema, max_tokens):
        self.calls.append((system, deepcopy(payload), deepcopy(schema), max_tokens))
        return deepcopy(self.result)


class Decisions:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        self.calls = []

    def ask(self, state, questions):
        self.calls.append((deepcopy(state), deepcopy(questions)))
        answers = {}
        for name, question in questions.items():
            if name in self.overrides:
                answers[name] = self.overrides[name]
            elif question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": .05 if name in {"action_unsupported_facts", "unsupported_scope"} else .95}
            else:
                criteria = question["criteria"]
                default = next((key for key in ("relevant", "supported", "match", "experience", "distinct", "neutral")
                                if key in criteria), "unclear")
                answers[name] = choice(default)
        return answers


class ReviewTests(unittest.TestCase):
    def test_release_association_uses_source_meaning_in_the_existing_request(self):
        source = packet(2)
        source["as_of"] = "2026-07-14"
        event = {"date": "2026-07-01", "version": "-", "notes": "An anniversary event with a featured activity."}
        source["past_release_calendar"] = [event, {"date": "2026-07-10", "version": "v2", "notes": "Other content."}]
        original = deepcopy(source)
        local, decisions = Local(), Decisions({"c0_release": choice("event_0", .9)})
        concern = review_candidate(source, local, decisions)["concerns"][0]
        self.assertEqual(concern["reviewed_release_associations"],
                         [{**event, "association": "content_association_only"}])
        self.assertEqual(len(decisions.calls), 1)
        question = decisions.calls[0][1]["c0_release"]
        self.assertEqual(set(question["criteria"]), {"event_0", "event_1", "none", "unclear"})
        material = question["instructions"]
        self.assertEqual(material["proposal"], proposal())
        self.assertEqual(material["focal_sources"], source["targets"])
        self.assertEqual(material["eligible_events"]["event_0"], event)
        self.assertEqual(material["interpretation_context"], source["context"])
        self.assertIn("Temporal proximity alone", material["task"])
        self.assertIn("does not establish causality", material["task"])
        self.assertEqual(source, original)

    def test_release_association_abstains_on_low_confidence_or_unrelated_events(self):
        source = packet(2)
        source["as_of"] = "2026-07-14"
        source["past_release_calendar"] = [{"date": "2026-07-01", "version": "-", "notes": "A supplied event."}]
        for answer in (choice("event_0", .74), choice("none"), choice("unclear"), None):
            with self.subTest(answer=answer):
                concern = review_candidate(source, Local(), Decisions({"c0_release": answer}))["concerns"][0]
                self.assertEqual(concern["reviewed_release_associations"], [])
        concern = review_candidate(source, Local(), Decisions({"c0_release": choice("event_0", .75)}))["concerns"][0]
        self.assertEqual(len(concern["reviewed_release_associations"]), 1)

    def test_future_or_malformed_events_cannot_enter_release_decisions(self):
        source = packet(2)
        source["as_of"] = "2026-07-14"
        source["past_release_calendar"] = [
            {"date": "2026-07-15", "version": "v2", "notes": "Future content."},
            {"date": "invalid", "version": "v3", "notes": "Malformed date."},
            {"date": "2026-07-01", "version": "", "notes": ""},
        ]
        decisions = Decisions({"c0_release": choice("event_0")})
        concern = review_candidate(source, Local(), decisions)["concerns"][0]
        self.assertNotIn("c0_release", decisions.calls[0][1])
        self.assertEqual(concern["reviewed_release_associations"], [])
        no_calendar = review_candidate(packet(2), Local(), Decisions())["concerns"][0]
        self.assertNotIn("reviewed_release_associations", no_calendar)

    def test_explicit_proposals_skip_local_generation_and_keep_one_decision_request(self):
        local, decisions = Local(), Decisions()
        supplied = {"concerns": [proposal()], "aliases": []}
        original = deepcopy(supplied)
        result = review_candidate(packet(), local, decisions, proposals=supplied)
        self.assertEqual(local.calls, [])
        self.assertEqual(len(decisions.calls), 1)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(supplied, original)

    def test_source_topics_use_original_examples_without_generalizing_quantities(self):
        source = packet(5)
        source["targets"][0]["text"] = "I spent 40 on the item and regretted it."
        source["targets"][0]["original_text"] = source["targets"][0]["text"] + " Another unrelated subject was mentioned."
        source["targets"][3]["text"] = "The item cost me 90; I think it was worth it."
        source["context"] = [{"id": "context-a", "text": "We are discussing the optional item.",
                              "role": "possible_preceding_context", "for_ids": ["v0", "v3"]}]
        original = deepcopy(source)
        proposals = source_proposals(source)
        self.assertEqual([p["source_anchor_id"] for p in proposals["concerns"]], ["v0", "v3"])
        self.assertIn(source["targets"][0]["text"], proposals["concerns"][0]["definition"])
        self.assertNotIn("unrelated subject", proposals["concerns"][0]["definition"])
        self.assertIn(source["targets"][3]["text"], proposals["concerns"][1]["definition"])
        self.assertNotIn("minimum", str(proposals))
        decisions = Decisions({"sibling_relation": choice("same"), "c0_t3_role": choice("disagreement")})
        result = review_candidate(source, None, decisions, proposals=proposals)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(result["concerns"]), 1)
        self.assertIn("v3", result["concerns"][0]["opposing_ids"])
        self.assertEqual(decisions.calls[0][0]["candidate"]["targets"], source["targets"])
        self.assertEqual(decisions.calls[0][0]["candidate"]["context"], source["context"])
        self.assertIn("do not require its literal statement to be true", decisions.calls[0][1]["c0_scope"]["instructions"])
        self.assertEqual(source, original)

    def test_source_proposals_include_only_existing_source_backed_alias_pairs(self):
        source = packet(3)
        source["aliases"] = [{"name": "Entity", "alias": "Nickname", "source_id": "alias-source",
                              "source_text": "Entity is called Nickname in this context."},
                             {"name": "Another", "alias": "Other", "source_id": "unknown"}]
        proposals = source_proposals(source)
        self.assertEqual(len(proposals["concerns"]), 1)
        self.assertEqual(proposals["aliases"], [{"alias": "Nickname", "referent": "Entity",
                                                "evidence_ids": ["alias-source"]}])

    def test_ten_examples_share_one_generator_call_and_one_decision_request(self):
        source = packet()
        original = deepcopy(source)
        local, decisions = Local(), Decisions()
        result = review_candidate(source, local, decisions)
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(decisions.calls), 1)
        self.assertEqual(len(local.calls[0][1]["candidate"]["targets"]), 10)
        self.assertEqual(len(decisions.calls[0][0]["candidate"]["targets"]), 10)
        self.assertEqual(local.calls[0][2], PROPOSAL_SCHEMA)
        self.assertEqual(source, original)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["membership_scope"], "sampled_targets_only")
        self.assertEqual(result["concerns"][0]["sample_n"], 10)
        self.assertNotIn("verified_cluster_count", result)

    def test_semantic_models_get_original_sources_and_only_role_relevant_statistics(self):
        source = packet()
        source.update(as_of="2026-07-14", aliases=[{"name": "Entity", "alias": "Nickname"}],
                      past_release_calendar=[{"date": "2026-07-01", "version": "v1"}],
                      stats={"raw_count": 300, "provisional_author_count": 200, "platform_count": 3,
                             "active_days": 15, "recent_window": {"support": 90}, "prior_window": {"support": 30},
                             "daily_activity": [["2026-07-01", 30]], "platform_activity": {"Forum": [1, 2, 3]}},
                      diagnostics={"ranking_score": 99, "temporal_flags": ["growing"],
                                   "possible_response_count": 50, "response_author_proxies": 25,
                                   "competing_anchor_ids": ["other-anchor"]})
        original = deepcopy(source)
        local, decisions = Local(), Decisions()
        review_candidate(source, local, decisions)
        proposal_packet = local.calls[0][1]["candidate"]
        self.assertNotIn("stats", proposal_packet)
        self.assertNotIn("diagnostics", proposal_packet)
        self.assertNotIn("metadata", proposal_packet)
        self.assertEqual(proposal_packet["targets"], source["targets"])
        self.assertEqual(proposal_packet["context"], source["context"])
        self.assertEqual(proposal_packet["aliases"], source["aliases"])
        self.assertEqual(proposal_packet["past_release_calendar"], source["past_release_calendar"])
        semantic_packet = decisions.calls[0][0]["candidate"]
        self.assertNotIn("stats", semantic_packet)
        self.assertNotIn("diagnostics", semantic_packet)
        self.assertNotIn("metadata", semantic_packet)
        self.assertEqual(semantic_packet["targets"], source["targets"])
        self.assertEqual(semantic_packet["context"], source["context"])
        self.assertEqual(semantic_packet["aliases"], source["aliases"])
        self.assertEqual(semantic_packet["past_release_calendar"], source["past_release_calendar"])
        self.assertEqual(semantic_packet["activity_summary"],
                         {"assigned_message_count": 300, "author_proxies": 200, "platform_count": 3,
                          "active_days": 15, "recent_support": 90, "prior_support": 30})
        self.assertEqual(semantic_packet["discussion_diagnostics"],
                         {"possible_response_count": 50, "response_author_proxies": 25,
                          "competing_anchor_source_count": 1})
        self.assertNotIn("daily_activity", str(semantic_packet))
        self.assertNotIn("platform_activity", str(semantic_packet))
        self.assertEqual(source, original)
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(decisions.calls), 1)

    def test_contamination_does_not_discard_otherwise_coherent_cluster(self):
        for unrelated in ((9,), (8, 9)):
            with self.subTest(unrelated=unrelated):
                decisions = Decisions({f"c0_t{i}_match": choice("other") for i in unrelated})
                concern = review_candidate(packet(), Local(), decisions)["concerns"][0]
                self.assertEqual(concern["status"], "accepted")
                self.assertEqual(concern["matched_ids"], [f"v{i}" for i in range(10 - len(unrelated))])
                self.assertEqual(concern["unrelated_ids"], [f"v{i}" for i in unrelated])
                self.assertEqual(concern["match_ratio"], (10 - len(unrelated)) / 10)
                self.assertFalse(set(concern["unrelated_ids"]) & set(concern["supporting_ids"]))

    def test_thresholds_classify_partial_support_and_preserve_uncertain_denominator(self):
        for matched, expected in ((10, "accepted"), (9, "accepted"), (8, "accepted"),
                                  (7, "provisional"), (6, "provisional"),
                                  (5, "provisional"), (4, "mixed"), (3, "mixed"),
                                  (2, "reject"), (1, "reject"), (0, "reject")):
            with self.subTest(matched=matched):
                decisions = Decisions({f"c0_t{i}_match": choice("unclear") for i in range(matched, 10)})
                concern = review_candidate(packet(), Local(), decisions)["concerns"][0]
                self.assertEqual(concern["status"], expected)
                self.assertEqual(concern["sample_n"], 10)
                self.assertEqual(concern["match_ratio"], matched / 10)
                self.assertEqual(len(concern["uncertain_ids"]), 10 - matched)

    def test_opposing_topic_matches_remain_evidence_but_not_endorsement(self):
        decisions = Decisions({"c0_t3_role": choice("disagreement"),
                               "c0_t4_role": choice("same_experience"),
                               "c0_t5_role": choice("agreement")})
        concern = review_candidate(packet(), Local(), decisions)["concerns"][0]
        self.assertEqual(concern["status"], "accepted")
        self.assertIn("v3", concern["matched_ids"])
        self.assertIn("v3", concern["opposing_ids"])
        self.assertNotIn("v3", concern["supporting_ids"])
        self.assertEqual(concern["same_experience_ids"], ["v4"])
        self.assertEqual(concern["contributions"]["v5"]["role"], "agreement")

    def test_eighty_percent_topic_membership_tolerates_dissent_and_two_unrelated_examples(self):
        decisions = Decisions({"c0_t8_match": choice("other"), "c0_t9_match": choice("other"),
                               "c0_t2_role": choice("disagreement"), "c0_t2_stance": choice("positive")})
        concern = review_candidate(packet(), Local(), decisions)["concerns"][0]
        self.assertEqual(concern["status"], "accepted")
        self.assertEqual(concern["match_ratio"], .8)
        self.assertIn("v2", concern["matched_ids"])
        self.assertIn("v2", concern["opposing_ids"])
        self.assertNotIn("v2", concern["supporting_ids"])
        self.assertEqual(concern["unrelated_ids"], ["v8", "v9"])

    def test_preserving_short_text_does_not_approve_relevance_or_resolve_missing_context(self):
        source = packet(1)
        source["targets"][0]["text"] = "慢"
        for overrides, expected in (({"c0_relevance": choice("irrelevant")}, "reject"),
                                    ({"c0_scope": choice("unclear")}, "mixed"),
                                    ({"c0_t0_match": choice("unclear")}, "reject")):
            with self.subTest(overrides=overrides):
                local, decisions = Local(), Decisions(overrides)
                concern = review_candidate(source, local, decisions, proposals=source_proposals(source))["concerns"][0]
                self.assertEqual(concern["status"], expected)
                self.assertNotEqual(concern["status"], "accepted")
                self.assertEqual(local.calls, [])
                self.assertEqual(len(decisions.calls), 1)

    def test_low_confidence_and_missing_decisions_are_unclear(self):
        decisions = Decisions({"c0_t0_match": choice("match", .1),
                               "c0_t1_match": {}, "c0_t2_match": None,
                               "c0_t3_match": choice("match", float("nan"))})
        concern = review_candidate(packet(), Local(), decisions)["concerns"][0]
        self.assertEqual(concern["status"], "provisional")
        self.assertEqual(concern["uncertain_ids"], ["v0", "v1", "v2", "v3"])

    def test_relevance_is_decided_by_jev_not_local_generator(self):
        for answer in ("background", "irrelevant"):
            with self.subTest(answer=answer):
                local, decisions = Local(), Decisions({"c0_relevance": choice(answer)})
                result = review_candidate(packet(), local, decisions)
                self.assertEqual(result["status"], "reject")
                self.assertEqual(len(local.calls), 1)
                self.assertEqual(len(decisions.calls), 1)

    def test_unsupported_scope_requires_refinement(self):
        concern = review_candidate(packet(), Local(),
                                   Decisions({"c0_scope": choice("overstated")}))["concerns"][0]
        self.assertEqual(concern["status"], "mixed")
        self.assertEqual(concern["supported_scope"], "")
        self.assertEqual(concern["matched_n"], 10)

    def test_uncertain_relevance_or_scope_requires_refinement_despite_many_matches(self):
        for key in ("c0_relevance", "c0_scope"):
            for answer in (choice("unclear"), choice("relevant" if key.endswith("relevance") else "supported", .1), None):
                with self.subTest(key=key, answer=answer):
                    result = review_candidate(packet(), Local(), Decisions({key: answer}))
                    concern = result["concerns"][0]
                    self.assertEqual(concern["matched_n"], 10)
                    self.assertEqual(concern["match_ratio"], 1.)
                    self.assertEqual(concern["status"], "mixed")
                    self.assertEqual(result["status"], "mixed")

    def test_provisional_requires_resolved_relevance_and_scope_too(self):
        overrides = {f"c0_t{i}_match": choice("other") for i in range(6, 10)}
        for key in ("c0_relevance", "c0_scope"):
            with self.subTest(key=key):
                decisions = Decisions({**overrides, key: choice("unclear")})
                concern = review_candidate(packet(), Local(), decisions)["concerns"][0]
                self.assertEqual(concern["match_ratio"], .6)
                self.assertEqual(concern["status"], "mixed")

    def test_single_popular_anchor_can_be_accepted_without_inventing_more_reports(self):
        source = packet(1)
        source["kind"] = "discussion_interval"
        source["metadata"]["possible_response_count"] = 200
        concern = review_candidate(source, Local(), Decisions())["concerns"][0]
        self.assertEqual(concern["status"], "accepted")
        self.assertEqual(concern["matched_ids"], ["v0"])
        self.assertEqual(concern["sample_n"], 1)

    def test_duplicate_samples_are_not_multiple_votes(self):
        source = packet(2)
        source["targets"].extend(deepcopy(source["targets"]))
        concern = review_candidate(source, Local(), Decisions())["concerns"][0]
        self.assertEqual(concern["sample_n"], 2)
        self.assertEqual(concern["matched_n"], 2)

    def test_two_concerns_are_reviewed_in_the_same_request(self):
        local = Local({"concerns": [proposal(), proposal(title_en="Another concern")], "aliases": []})
        overrides = {f"c0_t{i}_match": choice("other") for i in range(6, 10)}
        overrides.update({f"c1_t{i}_match": choice("other") for i in range(6)})
        decisions = Decisions(overrides)
        result = review_candidate(packet(), local, decisions)
        self.assertEqual([c["status"] for c in result["concerns"]], ["provisional", "mixed"])
        self.assertEqual(result["status"], "provisional")
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(decisions.calls), 1)

    def test_same_candidate_duplicate_keeps_best_supported_interpretation_and_counterevidence(self):
        local = Local({"concerns": [proposal(title_en="First wording"),
                                    proposal(title_en="Alternative wording")], "aliases": []})
        decisions = Decisions({"sibling_relation": choice("same"),
                               "c0_t8_match": choice("other"), "c0_t9_match": choice("other"),
                               "c1_t4_role": choice("disagreement"), "c1_t4_stance": choice("positive")})
        result = review_candidate(packet(), local, decisions)
        self.assertEqual(result["sibling_relation"], "same")
        self.assertEqual(len(result["concerns"]), 1)
        kept = result["concerns"][0]
        self.assertEqual(kept["proposal_index"], 1)
        self.assertEqual(kept["match_ratio"], 1.)
        self.assertEqual(kept["opposing_ids"], ["v4"])
        self.assertIn("v4", kept["matched_ids"])
        self.assertEqual(kept["stances"]["v4"], "positive")
        self.assertEqual(result["discarded_proposals"][0]["proposal_index"], 0)
        self.assertEqual(len(result["proposal"]["concerns"]), 2)
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(decisions.calls), 1)
        self.assertIn("sibling_relation", decisions.calls[0][1])

    def test_sibling_choice_prefers_qualified_scope_over_uncertain_higher_match(self):
        local = Local({"concerns": [proposal(), proposal(title_en="Alternative wording")], "aliases": []})
        decisions = Decisions({"sibling_relation": choice("same"), "c1_scope": choice("unclear"),
                               "c0_t8_match": choice("other"), "c0_t9_match": choice("other")})
        result = review_candidate(packet(), local, decisions)
        self.assertEqual(result["concerns"][0]["proposal_index"], 0)
        self.assertEqual(result["concerns"][0]["status"], "accepted")
        self.assertEqual(result["discarded_proposals"][0]["status"], "mixed")

    def test_distinct_or_uncertain_siblings_remain_separate(self):
        for relation in (choice("distinct"), choice("unclear"), choice("same", .1), None):
            with self.subTest(relation=relation):
                local = Local({"concerns": [proposal(), proposal(title_en="Another concern")], "aliases": []})
                result = review_candidate(packet(), local, Decisions({"sibling_relation": relation}))
                self.assertEqual(len(result["concerns"]), 2)
                self.assertEqual(result["discarded_proposals"], [])

    def test_source_anchor_siblings_need_confident_distinction_for_two_output_candidates(self):
        for relation in (choice("same"), choice("unclear"), choice("distinct", .1), None):
            with self.subTest(relation=relation):
                source = packet()
                proposals = source_proposals(source)
                original = deepcopy(proposals)
                local = Local()
                decisions = Decisions({"sibling_relation": relation,
                                       "c0_t8_match": choice("other"), "c0_t9_match": choice("other"),
                                       "c1_t4_role": choice("disagreement"), "c1_t4_stance": choice("positive")})
                result = review_candidate(source, local, decisions, proposals=proposals)
                self.assertEqual(local.calls, [])
                self.assertEqual(len(decisions.calls), 1)
                self.assertEqual(len(result["concerns"]), 1)
                kept = result["concerns"][0]
                self.assertEqual(kept["proposal_index"], 1)
                self.assertEqual(kept["opposing_ids"], ["v4"])
                self.assertEqual(len(result["discarded_proposals"]), 1)
                self.assertEqual(result["discarded_proposals"][0]["proposal_index"], 0)
                self.assertEqual(result["proposal"], original)
                if result["sibling_relation"] == "unclear":
                    self.assertIn("unresolved", kept["uncertainty"])

    def test_confident_distinct_source_anchors_preserve_two_separate_concerns(self):
        source = packet()
        decisions = Decisions({"sibling_relation": choice("distinct")})
        result = review_candidate(source, None, decisions, proposals=source_proposals(source))
        self.assertEqual(len(result["concerns"]), 2)
        self.assertEqual(result["sibling_relation"], "distinct")
        self.assertEqual(result["discarded_proposals"], [])
        self.assertEqual(len(decisions.calls), 1)

    def test_duplicate_concern_is_linked_to_existing_identifier(self):
        accepted = [{"id": "existing-concern", **proposal()}]
        decisions = Decisions({"c0_duplicate": choice("accepted_0")})
        concern = review_candidate(packet(), Local(), decisions, accepted)["concerns"][0]
        self.assertEqual(concern["duplicate_of"], "existing-concern")
        self.assertEqual(concern["status"], "accepted")
        self.assertEqual(decisions.calls[0][0]["accepted_concerns"][0]["id"], "existing-concern")

    def test_duplicate_question_retains_bounded_original_examples_in_the_existing_request(self):
        examples = [{"id": f"known-{index}", "content": f"Original wording with qualifier {index}.",
                     "created_at": "2026-07-01T10:00:00+08:00", "platform": "Forum",
                     "reviewed_focal_texts": [f"Reviewed concern {index}", f"Opposing view {index}"]}
                    for index in range(5)]
        accepted = [{"id": "known-concern", **proposal(definition="An illustrative seed fragment."),
                     "reviewed_examples": [examples[0], deepcopy(examples[0]), *examples[1:]]}]
        original = deepcopy(accepted)
        decisions = Decisions({"c0_duplicate": choice("accepted_0")})
        result = review_candidate(packet(), Local(), decisions, accepted)
        self.assertEqual(len(decisions.calls), 1)
        known = decisions.calls[0][0]["accepted_concerns"][0]
        self.assertEqual(known["reviewed_examples"], examples[:3])
        self.assertEqual(known["definition"], "An illustrative seed fragment.")
        question = decisions.calls[0][1]["c0_duplicate"]["instructions"]
        self.assertIn("underlying actionable player issues", question)
        self.assertIn("current reviewed matching focal examples", question)
        self.assertIn("opposing opinions", question)
        self.assertIn("Keep materially different issues separate", question)
        self.assertEqual(result["concerns"][0]["duplicate_of"], "known-concern")
        self.assertEqual(accepted, original)

    def test_aliases_require_known_source_ids_and_semantic_confirmation(self):
        aliases = [{"alias": "别称", "referent": "角色", "evidence_ids": ["parent"]},
                   {"alias": "其他", "referent": "角色", "evidence_ids": ["v0"]},
                   {"alias": "伪造", "referent": "角色", "evidence_ids": ["nonexistent"]}]
        decisions = Decisions({"alias_1": {"type": "noul", "noul": .1}})
        result = review_candidate(packet(), Local({"concerns": [proposal()], "aliases": aliases}), decisions)
        self.assertEqual([alias["alias"] for alias in result["aliases"]], ["别称"])
        self.assertNotIn("alias_2", decisions.calls[0][1])
        self.assertEqual(len(decisions.calls), 1)

    def test_source_backed_alias_assertion_can_supply_contextual_evidence(self):
        source = packet()
        source["aliases"] = [{"source_id": "alias-source", "source_text": "角色又叫别称",
                              "name": "角色", "alias": "别称"}]
        alias = {"alias": "别称", "referent": "角色", "evidence_ids": ["alias-source"]}
        result = review_candidate(source, Local({"concerns": [proposal()], "aliases": [alias]}), Decisions())
        self.assertEqual(result["aliases"][0]["evidence_ids"], ["alias-source"])

    def test_stance_is_separate_from_topic_and_report_kind(self):
        decisions = Decisions({"c0_t0_role": choice("experience"),
                               "c0_t0_stance": choice("negative"),
                               "c0_t1_role": choice("disagreement"),
                               "c0_t1_stance": choice("positive"),
                               "c0_t2_match": choice("other"),
                               "c0_t2_stance": choice("negative")})
        concern = review_candidate(packet(), Local(), decisions)["concerns"][0]
        self.assertEqual(concern["stances"]["v0"], "negative")
        self.assertEqual(concern["stances"]["v1"], "positive")
        self.assertEqual(concern["stances"]["v2"], "unclear")
        self.assertIn("v1", concern["matched_ids"])
        self.assertIn("v1", concern["opposing_ids"])

    def test_burst_attribution_checks_few_responses_without_promoting_whole_window(self):
        source = packet(1)
        source["kind"] = "burst"
        source["metadata"]["possible_response_count"] = 200
        source["context"] = [{"id": f"r{i}", "text": "赞同", "role": "possible_following_response"}
                             for i in range(20)]
        decisions = Decisions({"c0_r1_match": choice("other"),
                               "c0_r2_match": choice("unclear")})
        concern = review_candidate(source, Local(), decisions)["concerns"][0]
        self.assertTrue(concern["engagement_supported"])
        self.assertEqual(len(concern["reviewed_response_ids"]), 5)
        self.assertEqual(len(concern["matched_response_ids"]), 3)
        self.assertEqual(concern["sample_n"], 1)
        self.assertEqual(concern["matched_ids"], ["v0"])
        self.assertEqual(concern["engagement_membership_scope"], "reviewed_responses_only")
        self.assertEqual(len(decisions.calls), 1)
        self.assertIn("r19", concern["reviewed_response_ids"])

    def test_engagement_requires_both_discussion_support_and_attributed_response(self):
        source = packet(1)
        source["context"] = [{"id": "r0", "text": "同意", "role": "explicit_response"}]
        for overrides in ({"c0_engagement": {"noul": .3}},
                          {"c0_r0_match": choice("unclear")}):
            with self.subTest(overrides=overrides):
                concern = review_candidate(source, Local(), Decisions(overrides))["concerns"][0]
                self.assertFalse(concern["engagement_supported"])

    def test_invalid_policy_and_empty_targets_fail_before_model_calls(self):
        cases = [{"accept_ratio": .4}, {"provisional_ratio": float("nan")},
                 {"decision_confidence": 2}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                local, decisions = Local(), Decisions()
                with self.assertRaises(ValueError):
                    review_candidate(packet(), local, decisions, **kwargs)
                self.assertEqual(local.calls, [])
                self.assertEqual(decisions.calls, [])
        with self.assertRaises(ValueError):
            review_candidate({"targets": []}, Local(), Decisions())


class FinalDuplicateTests(unittest.TestCase):
    def snapshots(self, count=3):
        return [{"id": f"actual-concern-{index + 10}",
                 "sources": [{"id": f"original-{index}", "content": f"Original source {index} with qualification.",
                              "reviewed_focal_texts": [f"Matching focal clause {index}."]}],
                 "context": [{"id": f"context-{index}", "content": "Interpretation only."}],
                 "aliases": [], "as_of": "2026-07-24"}
                for index in range(count)]

    def test_one_request_compares_all_pairs_using_actual_ids_and_both_original_sets(self):
        snapshots = self.snapshots(8)
        original = deepcopy(snapshots)
        decisions = Decisions({"pair_0": choice("same", .9)})
        reviewed = review_final_duplicates(decisions, snapshots)
        self.assertEqual(len(decisions.calls), 1)
        self.assertEqual(len(decisions.calls[0][1]), 28)
        self.assertEqual(reviewed["confirmed_pairs"], [[snapshots[0]["id"], snapshots[1]["id"]]])
        by_id = {snapshot["id"]: snapshot for snapshot in snapshots}
        self.assertEqual(decisions.calls[0][0]["collections"], by_id)
        for name, question in decisions.calls[0][1].items():
            instruction = question["instructions"]
            left, right = reviewed["question_pairs"][name]
            self.assertEqual(instruction["left_topic_id"], left)
            self.assertEqual(instruction["right_topic_id"], right)
            self.assertEqual(instruction["left_source_ids"], [source["id"] for source in by_id[left]["sources"]])
            self.assertEqual(instruction["right_source_ids"], [source["id"] for source in by_id[right]["sources"]])
            self.assertNotIn("left_topic", instruction)
            self.assertNotIn("right_topic", instruction)
            self.assertIn("state.collections[left_topic_id]", instruction["task"])
            self.assertIn("state.collections[right_topic_id]", instruction["task"])
            self.assertIn("opposing opinions do not alone", instruction["task"])
            self.assertIn("do not alone make issues identical", instruction["task"])
        self.assertEqual(snapshots, original)
        self.assertEqual(reviewed["answers"]["pair_0"], choice("same", .9))
        serialized = json.dumps(decisions.calls[0], ensure_ascii=False)
        for snapshot in snapshots:
            self.assertEqual(serialized.count(snapshot["sources"][0]["content"]), 1)

    def test_uncertainty_low_confidence_and_missing_answers_never_confirm_merges(self):
        for answer in (choice("unclear"), choice("distinct"), choice("same", .54),
                       {"choice": "same"}, None, {"choice": "same", "confidence": float("nan")}):
            with self.subTest(answer=answer):
                reviewed = review_final_duplicates(Decisions({"pair_0": answer}), self.snapshots(2))
                self.assertEqual(reviewed["confirmed_pairs"], [])
                self.assertFalse(reviewed["pairs"][0]["duplicate"])
        boundary = review_final_duplicates(Decisions({"pair_0": choice("same", .55)}), self.snapshots(2))
        self.assertEqual(len(boundary["confirmed_pairs"]), 1)

    def test_zero_or_one_topic_needs_no_request(self):
        for count in (0, 1):
            with self.subTest(count=count):
                decisions = Decisions()
                reviewed = review_final_duplicates(decisions, self.snapshots(count))
                self.assertEqual(decisions.calls, [])
                self.assertEqual(reviewed, {"confirmed_pairs": [], "pairs": [], "answers": {}, "question_pairs": {}})

    def test_invalid_or_unbounded_snapshots_fail_before_decisions(self):
        valid = self.snapshots(2)
        invalid = [None, self.snapshots(9), [valid[0], valid[0]],
                   [{**valid[0], "sources": []}], [{**valid[0], "sources": valid[0]["sources"] * 11}],
                   [{**valid[0], "sources": valid[0]["sources"] * 2}],
                   [{**valid[0], "context": [{}] * 21}]]
        for snapshots in invalid:
            with self.subTest(snapshots=snapshots):
                decisions = Decisions()
                with self.assertRaises(ValueError):
                    review_final_duplicates(decisions, snapshots)
                self.assertEqual(decisions.calls, [])


class FinalReviewTests(unittest.TestCase):
    def writer_draft(self):
        return {"title": "玩家反馈", "title_en": "Player feedback",
                "proposed_action": "Review the supplied concern and verify the reported experience."}

    def draft(self):
        draft = self.writer_draft()
        # Legacy observation fixtures retain independent claim-grounding coverage.
        draft["category"] = "Experience"
        draft["claims"] = [{"text": "Players discuss the concern.", "evidence_ids": ["v0"], "metric_keys": []}]
        return {**draft, "why": compose_why(draft["claims"], draft["proposed_action"])}

    def test_writer_contract_has_only_titles_action_and_requires_chinese_title(self):
        raw = self.writer_draft()
        _validate_schema(raw, DRAFT_SCHEMA)
        variants = []
        for field in ("evidence_ids", "metric_keys"):
            invalid = deepcopy(raw)
            invalid[field] = ["chosen-by-writer"]
            variants.append(invalid)
        variants.extend([{**raw, "title": "English only"}, {**raw, "claims": []},
                         {**raw, "claims": [{"text": "Unchecked prose."}]}, {**raw, "why": "Unchecked prose."},
                         {**raw, "proposed_action": "x" * 301}, {**raw, "status_basis": "Invented override."}])
        for invalid in variants:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _validate_schema(invalid, DRAFT_SCHEMA)
        self.assertEqual(set(DRAFT_SCHEMA["properties"]), {"title", "title_en", "proposed_action"})
        with self.assertRaises(ValueError):
            _validate_schema({**raw, "category": "Unused classification"}, DRAFT_SCHEMA)
        pattern = DRAFT_SCHEMA["properties"]["title"]["pattern"]
        self.assertTrue(pattern.startswith("^") and pattern.endswith("$"))
        self.assertIn("㐀-鿿", pattern)
        self.assertNotIn("\\u", pattern)
        _validate_schema("玩家反馈", DRAFT_SCHEMA["properties"]["title"])
        for valid in ("甲", "甲" * 140, "玩家对SSR角色的反馈"):
            with self.subTest(valid=valid):
                self.assertIsNotNone(re.fullmatch(pattern, valid))
                _validate_schema(valid, DRAFT_SCHEMA["properties"]["title"])
        for invalid in ("", "English title only", "SSR玩家反馈", "甲" * 141,
                        '玩家"反馈', "玩家\\反馈", "玩家\n反馈", "玩家\r反馈", "玩家\t反馈", "玩家\x00反馈", "玩家\x1f反馈", "玩家\x7f反馈"):
            with self.subTest(invalid=invalid):
                self.assertIsNone(re.fullmatch(pattern, invalid))
                with self.assertRaises(ValueError):
                    _validate_schema(invalid, DRAFT_SCHEMA["properties"]["title"])

    def test_code_binds_all_selected_sources_and_never_promotes_context_to_report_support(self):
        state = {"sources": [{"id": "first", "content": "First source."},
                             {"id": "second", "content": "Second source."},
                             {"id": "first", "content": "First source."}],
                 "context": [{"id": "parent", "content": "Interpretation only."}]}
        original = deepcopy(state)
        local = Local(self.writer_draft())
        draft = final_draft(local, proposal(), state)
        self.assertEqual(draft["claims"], [{"text": "Player discussion topic: " + draft["title_en"],
                                           "evidence_ids": ["first", "second"], "metric_keys": []}])
        self.assertNotIn("parent", draft["claims"][0]["evidence_ids"])
        self.assertEqual(local.calls[0][2], DRAFT_SCHEMA)
        self.assertEqual(state, original)
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(local.calls[0][3], 1800)

    def test_generator_cannot_choose_cherry_picked_citations_or_metric_fields(self):
        state = {"sources": [{"id": "first"}, {"id": "second"}]}
        for field in ("evidence_ids", "metric_keys"):
            with self.subTest(field=field):
                raw = self.writer_draft()
                raw[field] = ["first"]
                with self.assertRaisesRegex(LocalModelError, "titles and action output contract"):
                    final_draft(Local(raw), proposal(), state)

    def test_derived_topic_caption_has_exact_identity_and_reuses_title_checks_in_one_request(self):
        state = {"sources": [{"id": "v0", "content": "An original source."},
                             {"id": "v1", "content": "Another original source."}],
                 "context": [{"id": "parent-proof", "content": "Interpretation only."}],
                 "concern": proposal(), "status_basis": "Computed activity basis."}
        original = deepcopy(state)
        draft = final_draft(Local(self.writer_draft()), proposal(), state)
        decisions = Decisions()
        checked = final_check(decisions, draft, state)
        self.assertTrue(checked["passed"])
        self.assertEqual(checked["mode"], "checked_topic_summary")
        self.assertEqual(checked["provenance"]["method"], "exact_checked_field_copy")
        self.assertEqual(checked["provenance"]["source_field"], "title_en")
        self.assertEqual(draft["claims"], [{"text": "Player discussion topic: " + draft["title_en"],
                                           "evidence_ids": ["v0", "v1"], "metric_keys": []}])
        self.assertEqual(len(decisions.calls), 1)
        questions = decisions.calls[0][1]
        self.assertEqual(set(questions), {"title", "title_en", "action_valid"})
        self.assertEqual(questions["title_en"]["instructions"]["original_sources"], state["sources"])
        self.assertEqual(questions["action_valid"]["instructions"]["interpretation_context"], state["context"])
        self.assertEqual(draft["category"], draft["title_en"])
        self.assertEqual(checked["category_mode"], "grounded_topic_label")
        self.assertIn(state["status_basis"], draft["why"])
        self.assertEqual(state, original)

        # A failed optional action is omitted; neither checked label is bypassed.
        for field in ("title", "title_en", "action_valid"):
            with self.subTest(field=field):
                failed = final_check(Decisions({field: {"noul": .2}}), draft, state)
                replacement = extractive_fallback(draft, state, failed)
                if field in {"title", "title_en"}:
                    self.assertIsNone(replacement)
                else:
                    result, audit = replacement
                    self.assertEqual(result["claims"], draft["claims"])
                    self.assertEqual(audit["mode"], "checked_topic_summary")
                    self.assertEqual(audit["rejected_generation"], {"draft": draft, "check": failed})
                    self.assertEqual(result["category"], draft["title_en"])
                    self.assertIsNone(result["proposed_action"])
                    self.assertNotIn(draft["proposed_action"], result["why"])

    def test_derived_mode_cannot_skip_claim_checks_for_arbitrary_text_or_incomplete_source_binding(self):
        state = {"sources": [{"id": "v0", "content": "Original."}, {"id": "v1", "content": "Other original."}]}
        draft = final_draft(Local(self.writer_draft()), proposal(), state)
        variants = []
        for changes in ({"text": "An arbitrary assertion."},
                        {"text": draft["claims"][0]["text"] + " Unsupported addition."},
                        {"evidence_ids": ["v0"]}, {"evidence_ids": ["v1", "v0"]},
                        {"metric_keys": ["invented_measurement"]}):
            altered = deepcopy(draft)
            altered["claims"][0].update(changes)
            altered["why"] = compose_why(altered["claims"], altered["proposed_action"])
            variants.append(altered)
        for altered in variants:
            with self.subTest(claim=altered["claims"]):
                decisions = Decisions()
                failed = final_check(decisions, altered, state)
                self.assertEqual(failed["failed_claims"], ["topic_caption_identity"])
                self.assertEqual(decisions.calls, [])
                self.assertIsNone(extractive_fallback(altered, state, failed))

    def test_production_labels_are_independent_and_share_scope_without_an_extra_veto(self):
        state = {"sources": [{"id": "original", "content": "A qualified player statement.",
                              "reviewed_focal_texts": ["A qualified player statement."]}]}
        draft = final_draft(Local(self.writer_draft()), proposal(), state)
        decisions = Decisions({"title_en": {"noul": .74}})
        checked = final_check(decisions, draft, state)
        self.assertEqual(checked["failed_claims"], ["title_en"])
        self.assertEqual(checked["required_scores"], {"title": .75, "title_en": .75, "action_valid": .75})
        self.assertIsNone(extractive_fallback(draft, state, checked))
        questions = decisions.calls[0][1]
        self.assertEqual(set(questions), {"title", "title_en", "action_valid"})
        self.assertEqual(questions["title_en"]["instructions"]["original_sources"], state["sources"])
        self.assertIn("independently against the original sources", questions["title_en"]["instructions"]["task"])
        for name in ("title", "title_en"):
            self.assertIn("Material unsupported qualifications are errors", questions[name]["instructions"]["task"])
            self.assertIn("Preserve attribution", questions[name]["instructions"]["task"])

    def test_unavailable_optional_action_check_omits_action_without_new_inference(self):
        state = {"sources": [{"id": "original", "content": "Original player concern."}],
                 "status_basis": "Activity increased over the observed comparison windows."}
        draft = final_draft(Local(self.writer_draft()), proposal(), state)
        for answer in (None, {"noul": .74}, {"noul": float("nan")}):
            with self.subTest(answer=answer):
                decisions = Decisions({"action_valid": answer})
                checked = final_check(decisions, draft, state)
                replacement, audit = extractive_fallback(draft, state, checked)
                self.assertEqual(len(decisions.calls), 1)
                self.assertIsNone(replacement["proposed_action"])
                self.assertEqual(replacement["category"], replacement["title_en"])
                self.assertNotIn("Suggested next step:", replacement["why"])
                self.assertIn(state["status_basis"], replacement["why"])
                self.assertNotIn("Player discussion topic:", replacement["why"])
                self.assertNotIn("Activity check", replacement["why"])
                self.assertEqual(audit["components"]["proposed_action"], "omitted_failed_component")
                self.assertTrue(audit["passed"])

    def test_unchecked_category_cannot_replace_the_deterministic_checked_topic_label(self):
        state = {"sources": [{"id": "original", "content": "Original player concern."}]}
        draft = final_draft(Local(self.writer_draft()), proposal(), state)
        self.assertEqual(draft["category"], draft["title_en"])
        draft["category"] = "An unverified narrow classification"
        decisions = Decisions()
        self.assertEqual(final_check(decisions, draft, state)["failed_claims"], ["topic_category_identity"])
        self.assertEqual(decisions.calls, [])

    def test_private_writer_projection_preserves_meaning_and_retry_without_numerical_inventory(self):
        first, second = "00000000-0000-0000-0000-000000000001", "11111111-1111-1111-1111-111111111111"
        previous = self.draft()
        previous["claims"][0]["evidence_ids"] = [first]
        state = {
            "concern": {"source_anchor_id": first},
            "sources": [{"id": first, "content": "First original.", "parent_id": second},
                        {"id": second, "content": "Second original."}],
            "context": [{"id": first, "content": "First original."}],
            "aliases": [{"source_id": second, "evidence_ids": [first]}],
            "as_of": "2026-07-14", "status": "growing", "first_seen": "2026-07-01",
            "uncertainty": "Sampled interpretation.",
            "source_release_associations": [{"date": "2026-07-01", "notes": "Supplied event."}],
            "measurements": {"raw_count": 30}, "engagement": {"explicit_reply_count": 2},
            "status_basis": "Recent assigned support is 12; prior support is 3.",
            "previous_draft": previous,
            "previous_draft_check": {"checks": {"claim_0": .2}, "failed_claims": ["claim_0"], "passed": False,
                                     "failed_claim_material": [{"cited_source_ids": [first]}]},
        }
        approved = {**proposal(), "source_anchor_id": second}
        original_state, original_approved = deepcopy(state), deepcopy(approved)
        local = Local(self.writer_draft())
        draft = final_draft(local, approved, state)
        payload = local.calls[0][1]
        supplied = payload["evidence"]
        self.assertEqual(supplied["sources"][0], {"id": "s0", "content": "First original.", "parent_id": "s1"})
        self.assertEqual(supplied["sources"][1]["id"], "s1")
        self.assertEqual(supplied["context"][0]["id"], "s0")
        self.assertNotIn("concern", supplied)
        self.assertEqual(supplied["aliases"], [{"source_id": "s1", "evidence_ids": ["s0"]}])
        self.assertEqual(payload["approved_concern"]["source_anchor_id"], "s1")
        self.assertNotIn("previous_draft", supplied)
        self.assertNotIn(previous["proposed_action"], json.dumps(supplied))
        self.assertEqual(supplied["previous_draft_check"], {"checks": {"claim_0": .2}, "failed_claims": ["claim_0"], "passed": False})
        for field in ("measurements", "engagement", "status_basis", "past_release_calendar", "sampled_stances"):
            self.assertNotIn(field, supplied)
        for field in ("as_of", "status", "first_seen", "uncertainty", "source_release_associations"):
            self.assertEqual(supplied[field], state[field])
        self.assertEqual(draft["claims"][0]["evidence_ids"], [first, second])
        self.assertIn(state["status_basis"], draft["why"])
        self.assertEqual(state, original_state)
        self.assertEqual(approved, original_approved)

    def test_writer_receives_full_sources_and_all_focals_without_privileged_anchor_placeholders(self):
        approved = {**proposal(definition="Underlying topic illustrated by the first source: one focal statement."),
                    "proposal_kind": "source_anchor", "source_anchor_id": "first",
                    "focal_text": "One focal statement."}
        state = {"concern": deepcopy(approved),
                 "sources": [{"id": "first", "content": "One focal statement. Another reviewed clause. Unrelated raw marker.",
                              "title": "Unrelated raw title", "created_at": "2026-07-01T10:00:00+08:00",
                              "platform": "Forum", "user_name": "Author", "day": "2026-07-01",
                              "reviewed_focal_texts": ["One focal statement.", "Another reviewed clause."]},
                             {"id": "second", "content": "A related statement with a different subject.",
                              "reviewed_focal_texts": ["A related statement with a different subject."]},
                             {"id": "legacy", "content": "Legacy original.", "title": "Legacy title"},
                             {"id": "empty", "content": "Empty focal fallback.", "reviewed_focal_texts": []},
                             {"id": "invalid", "content": "Invalid focal fallback.", "reviewed_focal_texts": [None, ""]}],
                 "context": [{"id": "parent-proof", "content": "Interpretation context retained."}],
                 "aliases": [{"name": "Entity", "alias": "Nickname", "evidence_ids": ["parent-proof"]}]}
        original_approved, original_state = deepcopy(approved), deepcopy(state)
        local = Local(self.writer_draft())
        draft = final_draft(local, approved, state)
        system, payload, _, _ = local.calls[0]
        self.assertNotIn("concern", payload["evidence"])
        for key in ("definition", "focal_text", "source_anchor_id", "source_anchor_index", "proposal_kind",
                    "title", "title_en", "category"):
            self.assertNotIn(key, payload["approved_concern"])
        self.assertEqual(payload["approved_concern"]["scope"], approved["scope"])
        self.assertEqual(payload["approved_concern"]["uncertainty"], approved["uncertainty"])
        supplied = payload["evidence"]["sources"]
        self.assertEqual([source["reviewed_focal_texts"] for source in supplied[:2]],
                         [source["reviewed_focal_texts"] for source in state["sources"][:2]])
        for source, original in zip(supplied, state["sources"], strict=True):
            self.assertEqual(source.get("title"), original.get("title"))
            self.assertEqual(source["content"], original["content"])
        self.assertIn("Unrelated raw marker.", json.dumps(payload))
        self.assertIn("Unrelated raw title", json.dumps(payload))
        for field in ("created_at", "platform", "user_name", "day"):
            self.assertEqual(supplied[0][field], state["sources"][0][field])
        for actual, original in zip(supplied[2:], state["sources"][2:]):
            self.assertEqual(actual["content"], original["content"])
            self.assertEqual(actual.get("title"), original.get("title"))
        self.assertEqual(payload["evidence"]["context"][0]["content"], state["context"][0]["content"])
        self.assertEqual(payload["evidence"]["aliases"][0]["alias"], "Nickname")
        self.assertIn("common underlying topic of all selected reviewed_focal_texts", system)
        self.assertIn("natural, recognizable labels", system)
        self.assertIn("what authors are communicating or doing", system)
        self.assertIn("subject or object involved", system)
        self.assertIn("collection's meaning and author intent", system)
        self.assertEqual(approved, original_approved)
        self.assertEqual(state, original_state)
        decisions = Decisions()
        final_check(decisions, draft, state)
        self.assertEqual(decisions.calls[0][1]["title"]["instructions"]["approved_concern"], approved)
        self.assertEqual(decisions.calls[0][1]["title"]["instructions"]["original_sources"], state["sources"])
        for name, question in decisions.calls[0][1].items():
            with self.subTest(grounding_field=name):
                interpretation = question["instructions"]["concern_interpretation"]
                self.assertIn("illustrative reference, not an exhaustive topic definition", interpretation)
                self.assertIn("matching focal texts collectively define the topic", interpretation)
                self.assertEqual(question["instructions"]["approved_concern"], approved)
        for name in ("title", "title_en"):
            task = decisions.calls[0][1][name]["instructions"]["task"]
            self.assertIn("recognizable subject and nature of concern", task)
            self.assertIn("Broad topic labels are acceptable", task)
            self.assertIn("not an exhaustive semantic scope", task)
            self.assertIn("all selected reviewed focal texts", task)
        legacy = proposal()
        final_draft(local, legacy, state)
        self.assertEqual(local.calls[1][1]["approved_concern"]["definition"], legacy["definition"])

    def test_empty_source_collection_does_not_substitute_context_as_support(self):
        draft = final_draft(Local(self.writer_draft()), proposal(), {"context": [{"id": "only-context"}]})
        self.assertEqual(draft["claims"][0]["evidence_ids"], [])
        self.assertEqual(draft["claims"][0]["metric_keys"], [])

    def test_fresh_repairs_keep_full_originals_focals_and_failed_policy_without_rejected_wording(self):
        state = {"sources": [{"id": "v0", "title": "Original qualification.",
                              "content": "Full original with a qualification and another clause.",
                              "reviewed_focal_texts": ["A focal fragment.", "Another matching fragment."],
                              "platform": "Forum", "created_at": "2026-07-01T10:00:00+08:00"},
                             {"id": "legacy", "content": "Legacy original without focal extraction."}],
                 "context": [{"id": "parent-proof", "content": "Interpretation context."}],
                 "aliases": [{"alias": "Nickname", "name": "Entity", "source_id": "parent-proof"}]}
        baseline = deepcopy(state)
        local = Local(self.writer_draft())
        previous = final_draft(local, proposal(), state)
        initial_payload = deepcopy(local.calls[0][1])
        self.assertEqual(initial_payload["task"], "draft_output")
        self.assertNotIn("repair_instruction", initial_payload)
        initial_source = initial_payload["evidence"]["sources"][0]
        self.assertEqual(initial_source["title"], state["sources"][0]["title"])
        self.assertEqual(initial_source["content"], state["sources"][0]["content"])
        for field in ("title", "title_en", "unsupported_scope", "category", "action_relevance"):
            with self.subTest(field=field):
                repair = {**deepcopy(state), "previous_draft": deepcopy(previous),
                          "previous_draft_check": {"passed": False, "failed_claims": [field],
                                                   "checks": {field: .2}}}
                before = deepcopy(repair)
                final_draft(local, proposal(), repair)
                repair_payload = local.calls[-1][1]
                self.assertEqual(repair_payload["task"], "repair_output")
                self.assertIn("failure_requirements", repair_payload["repair_instruction"])
                self.assertIn("not an approved template", repair_payload["repair_instruction"])
                supplied = repair_payload["evidence"]
                actual = supplied["sources"][0]
                self.assertEqual(actual["title"], state["sources"][0]["title"])
                self.assertEqual(actual["content"], state["sources"][0]["content"])
                self.assertNotIn("previous_draft", supplied)
                self.assertNotIn(previous["proposed_action"], json.dumps(supplied))
                self.assertEqual(supplied["previous_draft_check"], before["previous_draft_check"])
                for key in ("reviewed_focal_texts", "platform", "created_at"):
                    self.assertEqual(actual[key], state["sources"][0][key])
                self.assertEqual(supplied["sources"][1]["content"], state["sources"][1]["content"])
                self.assertEqual(supplied["context"][0]["content"], state["context"][0]["content"])
                self.assertEqual(supplied["aliases"][0]["alias"], "Nickname")
                self.assertEqual(repair, before)
        final_draft(local, proposal(), state)
        self.assertEqual(local.calls[-1][1], initial_payload)
        self.assertEqual(local.calls[0][1], initial_payload)
        self.assertEqual(state, baseline)
        for failed in (None, [], [1], [""]):
            with self.subTest(invalid_feedback=failed):
                incomplete = {**deepcopy(state), "previous_draft": deepcopy(previous),
                              "previous_draft_check": {"failed_claims": failed}}
                final_draft(local, proposal(), incomplete)
                self.assertEqual(local.calls[-1][1]["task"], "draft_output")
                self.assertNotIn("repair_instruction", local.calls[-1][1])

    def test_original_activity_basis_is_composed_and_cannot_be_overridden(self):
        basis = "Recent assigned support is 12 of 100; prior support is 3 of 100."
        state = {"status_basis": basis}
        raw = self.writer_draft()
        draft = final_draft(Local(raw), proposal(), state)
        expected = draft["title_en"] + ". " + basis
        self.assertEqual(draft["why"], expected + " Suggested next step: " + raw["proposed_action"])
        self.assertTrue(final_check(Decisions(), draft, state)["passed"])
        for changed in ({"status_basis": "Different support."}, {}, {"status_basis": {"unexpected": "object"}}):
            with self.subTest(changed=changed):
                decisions = Decisions()
                self.assertEqual(final_check(decisions, draft, changed)["failed_claims"], ["output_composition"])
                self.assertEqual(decisions.calls, [])
        plain = compose_why(draft["claims"], raw["proposed_action"])
        for absent in (None, "", "  "):
            self.assertEqual(compose_why(draft["claims"], raw["proposed_action"], absent), plain)

    def test_explicit_output_budget_and_responsibility_boundary_reach_generator(self):
        local = Local(self.writer_draft())
        final_draft(local, proposal(), {"sources": [{"id": "v0"}]}, max_tokens=900)
        system, _, schema, limit = local.calls[0]
        self.assertEqual(limit, 900)
        self.assertIn("Code attaches", system)
        self.assertIn("do not generate citations", system)
        self.assertEqual(set(schema["properties"]), {"title", "title_en", "proposed_action"})

    def test_explanation_is_exactly_checked_observation_and_labelled_action(self):
        draft = self.draft()
        self.assertEqual(draft["why"], "Players discuss the concern. Suggested next step: " + draft["proposed_action"])
        decisions = Decisions()
        draft["why"] += " Unchecked additional assertion."
        result = final_check(decisions, draft, packet())
        self.assertEqual(result["failed_claims"], ["output_composition"])
        self.assertEqual(decisions.calls, [])

    def test_malformed_or_multiple_observations_fail_before_decisions(self):
        for changes in ({"claims": []}, {"claims": [None]}, {"claims": [{"text": " "}]},
                        {"claims": self.draft()["claims"] * 2}, {"proposed_action": None},
                        {"proposed_action": ""}, {"proposed_action": "x" * 301}):
            with self.subTest(changes=changes):
                decisions = Decisions()
                result = final_check(decisions, {**self.draft(), **changes}, packet())
                self.assertEqual(result["failed_claims"], ["output_composition"])
                self.assertEqual(decisions.calls, [])

    def test_missing_or_nonfinite_grounding_answer_fails_closed(self):
        for field in ("title", "action_unsupported_facts", "unsupported_scope"):
            for replacement in ({}, None, {"noul": float("nan")}, {"noul": True}):
                with self.subTest(field=field, replacement=replacement):
                    result = final_check(Decisions({field: replacement}), self.draft(), packet())
                    self.assertFalse(result["passed"])
                    self.assertIn(field, result["failed_claims"])

    def test_valid_grounding_checks_observation_title_and_action_in_one_request(self):
        decisions = Decisions()
        result = final_check(decisions, self.draft(), packet())
        self.assertTrue(result["passed"])
        self.assertEqual(len(decisions.calls), 1)
        self.assertEqual(set(decisions.calls[0][1]),
                         {"title", "title_en", "unsupported_scope", "category", "claim_0", "action_relevance", "action_unsupported_facts"})
        self.assertAlmostEqual(result["checks"]["action_unsupported_facts"], .95)
        self.assertNotIn("failure_requirements", result)

    def test_failed_checks_return_only_their_actual_generic_tasks_and_criteria(self):
        draft = self.draft()
        state = {"sources": [{"id": "v0", "content": "Source-only marker."}],
                 "context": [{"id": "context", "content": "Context-only marker."}]}
        decisions = Decisions({"unsupported_scope": {"noul": .9}, "action_relevance": {"noul": .2}})
        result = final_check(decisions, draft, state)
        requirements = result["failure_requirements"]
        self.assertEqual(set(requirements), {"unsupported_scope", "action_relevance"})
        questions = decisions.calls[0][1]
        for name, requirement in requirements.items():
            self.assertEqual(requirement, {"task": questions[name]["instructions"]["task"],
                                           "criteria": questions[name].get("criteria", {})})
            self.assertEqual(set(requirement), {"task", "criteria"})
        self.assertNotIn("Source-only marker.", json.dumps(requirements))
        self.assertNotIn("Context-only marker.", json.dumps(requirements))
        self.assertNotIn(draft["proposed_action"], json.dumps(requirements))
        self.assertEqual(len(decisions.calls), 1)

    def test_title_gate_preserves_focal_intent_and_attribution_in_questions_and_repair_feedback(self):
        """Verify the semantic contract and rejection routing, not model accuracy."""
        cases = [
            ("How should I configure Aegis with my available equipment?", "Aegis is nicknamed Umbrella.",
             "Umbrella Community Culture"),
            ("I think rewards increasingly favor subscribers.", "Players are discussing subscription options.",
             "Subscriber Rewards Have Become Stronger"),
        ]
        for focal, background, misleading in cases:
            with self.subTest(focal=focal):
                approved = {**proposal(definition=focal), "proposal_kind": "source_anchor", "source_anchor_id": "v0"}
                state = {"concern": approved,
                         "sources": [{"id": "v0", "content": focal, "reviewed_focal_texts": [focal]}],
                         "context": [{"id": "parent-proof", "content": background, "role": "interpretation_only"}],
                         "aliases": [{"name": "Aegis", "alias": "Umbrella", "evidence_ids": ["parent-proof"]}]}
                original = deepcopy(state)
                local = Local({**self.writer_draft(), "title_en": misleading})
                draft = final_draft(local, approved, state)
                decisions = Decisions({"title": {"noul": .2}, "title_en": {"noul": .2}})
                checked = final_check(decisions, draft, state)
                self.assertEqual(checked["failed_claims"], ["title", "title_en"])
                self.assertIsNone(extractive_fallback(draft, state, checked))
                self.assertEqual(len(decisions.calls), 1)
                questions = decisions.calls[0][1]
                self.assertEqual(len(questions), 3)
                for name in ("title", "title_en"):
                    question = questions[name]
                    self.assertEqual(question["instructions"]["original_sources"], state["sources"])
                    self.assertEqual(question["instructions"]["interpretation_context"], state["context"])
                    self.assertIn("central communicative intent", question["instructions"]["task"])
                    self.assertIn("cannot replace the focal concern", question["instructions"]["task"])
                    self.assertIn("Preserve attribution", question["instructions"]["task"])
                    self.assertIn("not every source must state every part", question["criteria"]["true"])
                    self.assertIn("Topical entity overlap alone is insufficient", question["criteria"]["false"])
                repair = {**deepcopy(state), "previous_draft": deepcopy(draft), "previous_draft_check": deepcopy(checked)}
                final_draft(local, approved, repair)
                supplied = local.calls[-1][1]["evidence"]
                self.assertNotIn("previous_draft", supplied)
                self.assertEqual(supplied["previous_draft_check"]["failure_requirements"], checked["failure_requirements"])
                self.assertNotIn(focal, json.dumps(checked["failure_requirements"]))
                self.assertEqual(state, original)

    def test_unsupported_action_facts_use_opposite_polarity(self):
        result = final_check(Decisions({"action_unsupported_facts": {"noul": .9}}), self.draft(), packet())
        self.assertEqual(result["failed_claims"], ["action_unsupported_facts"])
        self.assertAlmostEqual(result["checks"]["action_unsupported_facts"], .1)
        self.assertEqual(result["answers"]["action_unsupported_facts"]["noul"], .9)

    def test_unsupported_scope_uses_inverse_polarity_and_blocks_every_fallback(self):
        """Exercise decision routing, without claiming to judge language here."""
        state = {"sources": [{"id": "v0", "content": "Original source."}]}
        draft = self.draft()
        decisions = Decisions({"unsupported_scope": {"type": "noul", "noul": .9}})
        result = final_check(decisions, draft, state)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_claims"], ["unsupported_scope"])
        self.assertAlmostEqual(result["checks"]["unsupported_scope"], .1)
        self.assertEqual(result["answers"]["unsupported_scope"]["noul"], .9)
        self.assertIsNone(extractive_fallback(draft, state, result))
        self.assertEqual(len(decisions.calls), 1)
        for other_failure in ("claim_0", "category", "action_relevance"):
            check = self.failed_review(unsupported_scope=.1, **{other_failure: .2})
            self.assertIsNone(extractive_fallback(draft, state, check))
        accepted = final_check(Decisions({"unsupported_scope": {"noul": .05}}), draft, state)
        self.assertTrue(accepted["passed"])
        self.assertAlmostEqual(accepted["checks"]["unsupported_scope"], .95)
        replacement, audit = extractive_fallback(draft, state, self.failed_review())
        self.assertEqual(audit["checks"]["unsupported_scope"], .95)

    def test_auxiliary_scope_boundary_preserves_primary_requirements_in_normal_and_fallback_paths(self):
        state = {"sources": [{"id": "v0", "content": "Original source."}]}
        # Legacy observation checks retain their previous auxiliary scope policy.
        for draft in (self.draft(),):
            for raw_scope, accepted in ((.5, True), (.51, False), (.31, True)):
                with self.subTest(mode=draft.get("generation_mode", "legacy"), raw_scope=raw_scope):
                    decisions = Decisions({"unsupported_scope": {"noul": raw_scope}})
                    checked = final_check(decisions, draft, state)
                    self.assertEqual(checked["passed"], accepted)
                    self.assertAlmostEqual(checked["checks"]["unsupported_scope"], 1 - raw_scope)
                    self.assertEqual(checked["required_scores"]["unsupported_scope"], .5)
                    self.assertEqual(checked["grounding_threshold"], .75)
                    self.assertTrue(all(score == .75 for name, score in checked["required_scores"].items()
                                        if name != "unsupported_scope"))
                    self.assertEqual(len(decisions.calls), 1)

                    category_failure = final_check(Decisions({"unsupported_scope": {"noul": raw_scope},
                                                              "category": {"noul": .74}}), draft, state)
                    self.assertIn("category", category_failure["failed_claims"])
                    fallback = extractive_fallback(draft, state, category_failure)
                    if accepted:
                        replacement, audit = fallback
                        self.assertEqual(replacement["category"], draft["title_en"])
                        self.assertEqual(audit["required_scores"], category_failure["required_scores"])
                        self.assertEqual(audit["grounding_threshold"], .75)
                        self.assertAlmostEqual(audit["checks"]["unsupported_scope"], 1 - raw_scope)
                    else:
                        self.assertIsNone(fallback)

            # A scope absence score above .5 cannot compensate for a failed
            # primary title, and the fallback uses exactly the same requirement.
            primary_failure = final_check(Decisions({"unsupported_scope": {"noul": .31},
                                                     "title_en": {"noul": .74}}), draft, state)
            self.assertEqual(primary_failure["failed_claims"], ["title_en"])
            self.assertIsNone(extractive_fallback(draft, state, primary_failure))

    def failed_review(self, **changes):
        checks = {name: .95 for name in ("title", "title_en", "unsupported_scope", "category", "action_relevance", "action_unsupported_facts", "claim_0")}
        checks["claim_0"] = .2
        checks.update(changes)
        return {"passed": False, "checks": checks,
                "failed_claims": [name for name, score in checks.items() if score < .75],
                "answers": {"claim_0": {"type": "noul", "noul": checks["claim_0"]}}}

    def test_extractively_replaced_observation_is_exact_json_quoted_originals_with_provenance(self):
        draft = self.draft()
        draft["claims"][0]["text"] = "An unsupported generated assertion."
        state = {"sources": [{"id": "v0", "content": '  原文\n"quoted" \\ emoji😋  '},
                             {"id": "v1", "content": "第二个完整原文。"},
                             {"id": "v2", "content": "A third unused source."}],
                 "context": [{"id": "context", "content": "Not independent support."}],
                 "status_basis": "Observed assignment share increased."}
        draft["why"] = compose_why(draft["claims"], draft["proposed_action"], state["status_basis"])
        check = self.failed_review()
        original = deepcopy((draft, state, check))
        replacement, audit = extractive_fallback(draft, state, check)
        quoted = replacement["claims"][0]
        self.assertEqual(json.loads(quoted["text"].removeprefix("Reviewed player excerpts: ")),
                         [source["content"] for source in state["sources"][:2]])
        self.assertEqual(quoted["evidence_ids"], ["v0", "v1"])
        self.assertEqual(quoted["metric_keys"], [])
        self.assertNotIn("unsupported generated assertion", replacement["why"])
        self.assertIn(state["status_basis"], replacement["why"])
        self.assertEqual(replacement["proposed_action"], draft["proposed_action"])
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["mode"], "extractive_provenance")
        self.assertFalse(audit["provenance"]["new_jev_approval"])
        self.assertEqual(audit["components"]["observation"], "exact_original_source_copy")
        self.assertNotIn("claim_0", audit["checks"])
        self.assertEqual(audit["rejected_generation"], {"draft": draft, "check": check})
        self.assertEqual((draft, state, check), original)

    def test_optional_failed_action_is_omitted_without_replacing_a_supported_observation(self):
        draft, state = self.draft(), {"sources": [{"id": "v0", "content": "x" * 501}]}
        check = self.failed_review(claim_0=.9, action_relevance=.4)
        replacement, audit = extractive_fallback(draft, state, check)
        self.assertEqual(replacement["claims"], draft["claims"])
        self.assertIsNone(replacement["proposed_action"])
        self.assertEqual(replacement["why"], draft["claims"][0]["text"])
        self.assertNotIn("Suggested next step", replacement["why"])
        self.assertEqual(audit["mode"], "checked_components")
        self.assertEqual(audit["components"]["observation"], "previously_jev_checked")
        self.assertEqual(audit["components"]["proposed_action"], "omitted_failed_component")
        self.assertEqual(audit["checks"]["claim_0"], .9)
        self.assertNotIn("action_relevance", audit["checks"])
        self.assertEqual(audit["rejected_generation"]["draft"]["proposed_action"], draft["proposed_action"])
        with self.assertRaises(ValueError):
            compose_why(replacement["claims"], None)
        self.assertFalse(final_check(Decisions(), replacement, state)["passed"])

    def test_failed_observation_and_action_emit_only_exact_quotes_and_activity(self):
        state = {"sources": [{"id": "v0", "content": "完整原文。"}]}
        replacement, audit = extractive_fallback(self.draft(), state, self.failed_review(action_unsupported_facts=.1))
        self.assertIsNone(replacement["proposed_action"])
        self.assertNotIn("Suggested next step", replacement["why"])
        self.assertEqual(audit["mode"], "extractive_provenance")
        self.assertEqual(audit["components"]["proposed_action"], "omitted_failed_component")

    def test_failed_or_missing_title_checks_block_fallback(self):
        state = {"sources": [{"id": "v0", "content": "Original."}]}
        for field in ("title", "title_en", "unsupported_scope"):
            with self.subTest(field=field):
                rejected_score = .49 if field == "unsupported_scope" else .74
                self.assertIsNone(extractive_fallback(self.draft(), state, self.failed_review(**{field: rejected_score})))
                check = self.failed_review()
                check["checks"].pop(field)
                self.assertIsNone(extractive_fallback(self.draft(), state, check))
        missing_category = self.failed_review()
        missing_category["checks"].pop("category")
        self.assertIsNone(extractive_fallback(self.draft(), state, missing_category))
        for altered in ("unknown_field", "output_composition", "claim_7"):
            check = self.failed_review()
            check["failed_claims"] = [altered]
            self.assertIsNone(extractive_fallback(self.draft(), state, check))

    def test_failed_category_reuses_exact_grounded_english_topic_label(self):
        draft = self.draft()
        draft["category"] = "Unsupported narrow classification"
        state = {"sources": [{"id": "v0", "content": "Original source."}]}
        check = self.failed_review(claim_0=.9, category=.56)
        original = deepcopy((draft, check))
        replacement, audit = extractive_fallback(draft, state, check)
        self.assertEqual(replacement["category"], draft["title_en"])
        self.assertEqual(replacement["claims"], draft["claims"])
        self.assertEqual(audit["category_mode"], "grounded_topic_label")
        self.assertEqual(audit["components"]["category"], "grounded_topic_label")
        self.assertNotIn("category", audit["checks"])
        self.assertEqual(audit["rejected_generation"]["draft"]["category"], draft["category"])
        self.assertEqual(audit["rejected_generation"]["check"]["checks"]["category"], .56)
        self.assertEqual((draft, check), original)
        invalid = {**draft, "title_en": "x" * 141}
        self.assertIsNone(extractive_fallback(invalid, state, check))

    def test_unknown_references_and_unavailable_or_oversized_originals_are_refused(self):
        draft = self.draft()
        unknown = deepcopy(draft)
        unknown["claims"][0]["evidence_ids"] = ["unknown"]
        state = {"sources": [{"id": "v0", "content": "Original."}]}
        self.assertIsNone(extractive_fallback(unknown, state, self.failed_review()))
        for source in ({"id": "v0", "content": "x" * 501, "title": "Must not substitute a title for long content."},
                       {"id": "v0", "content": None, "title": None, "text": "Only normalized text is available."}):
            with self.subTest(source=source):
                self.assertIsNone(extractive_fallback(draft, {"sources": [source]}, self.failed_review()))
        state = {"sources": [{"id": "v0", "content": "x" * 501},
                             {"id": "v1", "content": "", "title": 'Exact own "title".'}]}
        replacement, audit = extractive_fallback(draft, state, self.failed_review())
        self.assertEqual(replacement["claims"][0]["evidence_ids"], ["v1"])
        self.assertEqual(json.loads(replacement["claims"][0]["text"].removeprefix("Reviewed player excerpts: ")),
                         ['Exact own "title".'])
        self.assertEqual(audit["provenance"]["sources"][0]["field"], "title")

    def test_grounding_keeps_full_original_sources_metrics_and_interpretation_boundaries(self):
        evidence = {"as_of": "2026-07-14", "concern": proposal(),
                    "sources": [{"id": "source", "content": "Original source.", "created_at": "2026-07-01T10:00:00+08:00"},
                                {"id": "other", "content": "Another original.", "created_at": "2026-07-12T10:00:00+08:00"}],
                    "context": [{"id": "parent", "content": "Meaning context."}],
                    "aliases": [{"name": "Entity", "alias": "Nickname", "evidence_ids": ["parent"]}],
                    "measurements": {"raw_count": 20}, "engagement": {"explicit_reply_count": 2}}
        original = deepcopy(evidence)
        draft = self.draft()
        draft["claims"][0]["evidence_ids"] = [source["id"] for source in evidence["sources"]]
        decisions = Decisions()
        self.assertTrue(final_check(decisions, draft, evidence)["passed"])
        shared, questions = decisions.calls[0]
        self.assertEqual(set(shared), {"task", "policy"})
        claim = questions["claim_0"]["instructions"]
        self.assertEqual(claim["cited_original_sources"], evidence["sources"])
        self.assertEqual(claim["interpretation_context"], evidence["context"])
        self.assertEqual(claim["aliases"], evidence["aliases"])
        self.assertEqual(claim["approved_concern"], evidence["concern"])
        self.assertIn("not every source must state every part", claim["task"])
        self.assertIn("do not independently establish additional reports", claim["task"])
        self.assertIn("Topical overlap or co-occurrence alone is not entailment", claim["task"])
        self.assertIn("Every material factual clause", questions["claim_0"]["criteria"]["true"])
        self.assertIn("causal or temporal direction", questions["claim_0"]["criteria"]["true"])
        self.assertIn("At least one material part is unsupported", questions["claim_0"]["criteria"]["false"])
        self.assertEqual(questions["action_relevance"]["instructions"]["measurements"], evidence["measurements"])
        self.assertEqual(questions["action_relevance"]["instructions"]["engagement"], evidence["engagement"])
        self.assertEqual(questions["title"]["instructions"]["original_sources"], evidence["sources"])
        scope = questions["unsupported_scope"]["instructions"]
        self.assertEqual(scope["proposed_chinese_title"], draft["title"])
        self.assertEqual(scope["proposed_english_title"], draft["title_en"])
        self.assertEqual(scope["as_of"], evidence["as_of"])
        for question_key, state_key in (("original_sources", "sources"), ("interpretation_context", "context"),
                                        ("approved_concern", "concern"), ("aliases", "aliases")):
            self.assertEqual(scope[question_key], evidence[state_key])
        self.assertEqual(evidence, original)


if __name__ == "__main__":
    unittest.main()
