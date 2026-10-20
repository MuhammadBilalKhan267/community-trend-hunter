from copy import deepcopy
import unittest

from trend_hunter.pipeline import STATUSES, validate_output


class OutputContractTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"id": "early", "day": "2026-07-01"},
                        {"id": "later", "day": "2026-07-11"},
                        {"id": "future", "day": "2026-07-15"}]
        self.valid = {"as_of": "2026-07-14", "trends": [{
            "title": "玩家反馈", "title_en": "Player concern", "category": "Experience",
            "status": "new", "first_seen": "2026-07-01", "voice_ids": ["early", "later"],
            "why": "The supplied messages support this concern."}]}

    def assert_invalid(self, value):
        with self.assertRaises(ValueError):
            validate_output(value, self.records)

    def test_valid_output_all_statuses_and_single_example_allowed(self):
        for status in STATUSES:
            with self.subTest(status=status):
                value = deepcopy(self.valid)
                value["trends"][0]["status"] = status
                value["trends"][0]["voice_ids"] = ["later"]
                validate_output(value, self.records)
        validate_output({"as_of": "2026-07-14", "trends": []}, self.records)
        validate_output({"as_of": "2024-02-29", "trends": []}, [])

    def test_top_level_must_have_exact_object_shape(self):
        for value in (None, [], True, "result", {}, {"as_of": "2026-07-14"},
                      {**self.valid, "diagnostics": {}}, {"trends": []}):
            with self.subTest(value=value):
                self.assert_invalid(value)
        for trends in (None, True, "", {}, (), 1):
            with self.subTest(trends=trends):
                self.assert_invalid({"as_of": "2026-07-14", "trends": trends})

    def test_cutoff_requires_exact_real_calendar_date(self):
        for day in (None, True, 20260714, "", "20260714", "2026-7-14", "2026-02-29",
                    "2026-13-01", "2026-07-14T00:00:00", " 2026-07-14", "2026-W29-2"):
            with self.subTest(day=day):
                self.assert_invalid({"as_of": day, "trends": []})

    def test_trend_requires_exact_object_fields(self):
        for trend in (None, [], True, "concern", {}):
            with self.subTest(trend=trend):
                self.assert_invalid({"as_of": "2026-07-14", "trends": [trend]})
        for missing in self.valid["trends"][0]:
            with self.subTest(missing=missing):
                value = deepcopy(self.valid)
                del value["trends"][0][missing]
                self.assert_invalid(value)
        value = deepcopy(self.valid)
        value["trends"][0]["confidence"] = .99
        self.assert_invalid(value)

    def test_explanatory_strings_cannot_be_empty_or_wrong_type(self):
        for field in ("title", "title_en", "category", "why"):
            for replacement in (None, True, 1, [], {}, "", " \n\t "):
                with self.subTest(field=field, replacement=replacement):
                    value = deepcopy(self.valid)
                    value["trends"][0][field] = replacement
                    self.assert_invalid(value)

    def test_status_is_a_known_string(self):
        for status in (None, True, 1, [], {}, "", "stable", "Growing"):
            with self.subTest(status=status):
                value = deepcopy(self.valid)
                value["trends"][0]["status"] = status
                self.assert_invalid(value)

    def test_first_seen_is_valid_date_not_after_cutoff_or_evidence(self):
        for day in (None, True, 20260701, "20260701", "2026-02-30", "2026-07-15",
                    "2026-07-02", "2026-07-01T00:00:00", "2026-7-1"):
            with self.subTest(day=day):
                value = deepcopy(self.valid)
                value["trends"][0]["first_seen"] = day
                self.assert_invalid(value)

    def test_evidence_is_nonempty_unique_visible_string_id_list(self):
        for identifiers in (None, True, "early", {}, (), [], [True], [1], [None],
                            [""], [" "], ["early", 1], ["early", "early"], ["missing"], ["future"]):
            with self.subTest(identifiers=identifiers):
                value = deepcopy(self.valid)
                value["trends"][0]["voice_ids"] = identifiers
                self.assert_invalid(value)

    def test_ineligible_sources_cannot_be_exported_as_evidence(self):
        for flags in ({"bot_like": True}, {"trivial": True}, {"family": ""}):
            with self.subTest(flags=flags):
                records = deepcopy(self.records)
                records[0].update(flags)
                with self.assertRaises(ValueError):
                    validate_output(self.valid, records)

    def test_eight_trends_allowed_nine_rejected(self):
        value = deepcopy(self.valid)
        value["trends"] *= 8
        validate_output(value, self.records)
        value["trends"].append(deepcopy(value["trends"][0]))
        self.assert_invalid(value)


if __name__ == "__main__":
    unittest.main()
