"""Reference validation detects fabricated citations without semantic claims."""
from copy import deepcopy
import unittest

from trend_hunter.grounding import claim_reference_material, reference_catalog, validate_draft_references


class GroundingReferenceTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "sources": [{"id": "source", "text": "Source material."}],
            "context": [{"id": "parent", "text": "Context material."}],
            "measurements": {"raw_count": 30, "recent_window": {"support": 12},
                             "prior_window": {"support": 3},
                             "daily_activity": [["2026-07-01", 2], ["2026-07-02", 3]],
                             "normalized_growth_ratio": None},
            "engagement": {"explicit_reply_count": 4},
            "status": "growing", "first_seen": "2026-07-01",
        }

    def check(self, identifiers=(), metrics=()):
        draft = {"claims": [{"text": "A factual claim.", "evidence_ids": list(identifiers),
                             "metric_keys": list(metrics)}]}
        return validate_draft_references(draft, self.state)

    def test_known_sources_and_context_are_valid(self):
        self.assertEqual(self.check(["source", "parent"]), [])

    def test_generation_catalog_contains_only_supplied_references_without_series_indices(self):
        original = deepcopy(self.state)
        catalog = reference_catalog(self.state)
        self.assertEqual(catalog["evidence_ids"], ["parent", "source"])
        self.assertIn("measurements.recent_window", catalog["metric_keys"])
        self.assertIn("measurements.recent_window.support", catalog["metric_keys"])
        self.assertIn("measurements.daily_activity", catalog["metric_keys"])
        self.assertFalse(any(reference.startswith("measurements.daily_activity.") for reference in catalog["metric_keys"]))
        for metric in catalog["metric_keys"]:
            with self.subTest(metric=metric):
                self.assertEqual(self.check(metrics=[metric]), [])
        self.assertEqual(self.state, original)

    def test_claim_material_resolves_actual_values_and_preserves_missing_reference_diagnostics(self):
        original = deepcopy(self.state)
        material = claim_reference_material({
            "evidence_ids": ["source", "parent", "source", "missing"],
            "metric_keys": ["raw_count", "measurements.recent_window.support", "status", "support", "unknown"],
        }, self.state)
        self.assertEqual([source["id"] for source in material["cited_original_sources"]], ["source", "parent"])
        self.assertEqual(material["missing_source_ids"], ["missing"])
        self.assertEqual(material["cited_measurements"], [
            {"reference": "raw_count", "path": "measurements.raw_count", "value": 30},
            {"reference": "measurements.recent_window.support", "path": "measurements.recent_window.support", "value": 12},
            {"reference": "status", "path": "status", "value": "growing"},
        ])
        self.assertEqual(material["unresolved_metric_references"], [
            {"reference": "support", "reason": "ambiguous"}, {"reference": "unknown", "reason": "unknown"}])
        self.assertEqual(self.state, original)

    def test_known_nested_measurements_and_engagement_are_valid(self):
        self.assertEqual(self.check(metrics=["measurements.recent_window.support",
                                             "engagement.explicit_reply_count"]), [])
        self.assertEqual(self.check(metrics=["raw_count", "explicit_reply_count", "status", "first_seen"]), [])

    def test_cutoff_date_can_be_cited_only_when_supplied(self):
        self.assertNotIn("as_of", reference_catalog(self.state)["metric_keys"])
        self.assertEqual(self.check(metrics=["as_of"]), ["claim_0:unknown_metric_reference"])
        self.state["as_of"] = "2026-07-14"
        self.assertIn("as_of", reference_catalog(self.state)["metric_keys"])
        self.assertEqual(self.check(metrics=["as_of"]), [])
        material = claim_reference_material({"evidence_ids": [], "metric_keys": ["as_of"]}, self.state)
        self.assertEqual(material["cited_measurements"],
                         [{"reference": "as_of", "path": "as_of", "value": "2026-07-14"}])

    def test_bare_leaf_must_identify_one_actual_measurement(self):
        self.assertEqual(self.check(metrics=["support"]), ["claim_0:ambiguous_metric_reference"])
        self.assertEqual(self.check(metrics=["invented_count"]), ["claim_0:unknown_metric_reference"])

    def test_unknown_source_fails_even_when_another_reference_is_valid(self):
        self.assertEqual(self.check(["source", "invented"], ["raw_count"]),
                         ["claim_0:unknown_source_reference"])

    def test_path_must_exist_and_use_allowed_namespace(self):
        for key in ("measurements.invented", "sources.0.id", "measurements..raw_count",
                    ".raw_count", "measurements.raw_count.missing", "not_measurements.raw_count"):
            with self.subTest(key=key):
                self.assertEqual(self.check(metrics=[key]), ["claim_0:unknown_metric_reference"])

    def test_series_can_be_referenced_as_a_whole_or_at_an_existing_position(self):
        self.assertEqual(self.check(metrics=["daily_activity", "measurements.daily_activity.0.1"]), [])
        self.assertEqual(self.check(metrics=["measurements.daily_activity.30"]),
                         ["claim_0:unknown_metric_reference"])
        self.assertEqual(self.check(metrics=["measurements.daily_activity.-1"]),
                         ["claim_0:unknown_metric_reference"])

    def test_missing_value_can_be_referenced_as_unavailable_without_inventing_a_number(self):
        self.assertEqual(self.check(metrics=["normalized_growth_ratio"]), [])

    def test_missing_claims_or_references_fail(self):
        self.assertEqual(validate_draft_references({"claims": []}, self.state), ["missing_factual_claims"])
        self.assertEqual(self.check(), ["claim_0:missing_reference"])
        self.assertEqual(validate_draft_references({"claims": [None]}, self.state), ["claim_0:invalid_claim"])

    def test_malformed_reference_arrays_are_not_implicitly_coerced(self):
        for refs in (None, "source", {}, [None], [""], ["  "]):
            with self.subTest(refs=refs):
                draft = {"claims": [{"text": "Claim", "evidence_ids": refs, "metric_keys": []}]}
                self.assertEqual(validate_draft_references(draft, self.state), ["claim_0:invalid_references"])

    def test_validation_does_not_mutate_sources_or_rewrite_claims(self):
        state = deepcopy(self.state)
        draft = {"claims": [{"text": "A source interpretation.", "evidence_ids": ["source"], "metric_keys": []}]}
        original = deepcopy(draft)
        self.assertEqual(validate_draft_references(draft, state), [])
        self.assertEqual(draft, original)
        self.assertEqual(state, self.state)


if __name__ == "__main__":
    unittest.main()
