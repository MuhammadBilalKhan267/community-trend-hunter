import unittest
from datetime import date, timedelta

from trend_hunter.temporal import analyze_history, prepare_history


START = date(2026, 7, 1)


def voice(index, day, family=None, user=None, platform="forum", hour=23, root=None):
    day = START + timedelta(days=day)
    return {"id": str(index), "day": day.isoformat(), "platform": platform,
            "family": family or f"text {index}", "user_name": user or f"author {index}",
            "root_id": root or f"thread {index}", "created_at": f"{day.isoformat()}T{hour:02d}:30:00+08:00"}


def corpus(topic_counts, total_counts=100):
    members, universe = [], []
    for day, topic_count in enumerate(topic_counts):
        total = total_counts[day] if isinstance(total_counts, list) else total_counts
        for count in range(total):
            record = voice(f"{day}-{count}", day)
            universe.append(record)
            if count < topic_count:
                members.append(record)
    return members, universe


def analyze(members, universe, day=13):
    return analyze_history(members, universe, (START + timedelta(days=day)).isoformat())


class TemporalTests(unittest.TestCase):
    def test_gradual_normalized_rise(self):
        members, universe = corpus([1] * 7 + [2, 3, 4, 5, 6, 7, 8])
        result = analyze(members, universe)
        self.assertEqual(result["status"], "growing")
        self.assertGreater(result["stats"]["normalized_growth_ratio"], 1.35)

    def test_overall_activity_rise_does_not_make_topic_grow(self):
        members, universe = corpus([2] * 7 + [4] * 7, [100] * 7 + [200] * 7)
        result = analyze(members, universe)
        self.assertIsNone(result["status"])
        self.assertFalse(result["status_supported"])
        self.assertLess(result["stats"]["normalized_growth_ratio"], 1.1)

    def test_platform_mix_does_not_make_topic_grow(self):
        members, universe = [], []
        for platform, totals, counts in [("forum", [100] * 7 + [400] * 7, [10] * 7 + [40] * 7),
                                          ("chat", [100] * 14, [1] * 14)]:
            part, whole = corpus(counts, totals)
            for row in whole:
                row["id"] = platform + row["id"]
                row["platform"] = platform
            members.extend(part)
            universe.extend(whole)
        result = analyze(members, universe)
        self.assertIsNone(result["status"])
        self.assertLess(result["stats"]["normalized_growth_ratio"], 1.1)

    def test_onset_without_future_is_not_one_off(self):
        members, universe = corpus([0] * 13 + [15])
        result = analyze(members, universe)
        self.assertIsNone(result["status"])
        self.assertEqual(result["stats"]["observed_days_after_peak"], 0)

    def test_new_needs_multiple_days_and_prior_observation(self):
        members, universe = corpus([0] * 11 + [2, 3, 3])
        self.assertEqual(analyze(members, universe)["status"], "new")
        members, universe = corpus([3, 3])
        self.assertIsNone(analyze(members, universe, 1)["status"])

    def test_observed_spike_with_followup_is_one_off(self):
        members, universe = corpus([0] * 4 + [15, 1] + [0] * 8)
        result = analyze(members, universe)
        self.assertEqual(result["status"], "one-off")
        self.assertGreaterEqual(result["stats"]["observed_days_after_peak"], 2)

    def test_current_new_classification_precedes_a_historical_spike_shape(self):
        members, universe = corpus([0] * 8 + [15, 1, 0, 0, 0, 0])
        self.assertEqual(analyze(members, universe)["status"], "new")

    def test_direction_check_accepts_small_support_changes_but_rejects_conflicts(self):
        for prior, recent, totals, expected in (
            (10, 11, [100] * 7 + [50] * 7, "growing"),
            (11, 10, [100] * 7 + [200] * 7, "fading"),
            (10, 9, [100] * 7 + [50] * 7, None),
            (10, 10, [100] * 7 + [50] * 7, None),
            (9, 10, [100] * 7 + [200] * 7, None),
        ):
            with self.subTest(prior=prior, recent=recent, expected=expected):
                members, universe = corpus([prior] * 7 + [recent] * 7, totals)
                self.assertEqual(analyze(members, universe)["status"], expected)

    def test_ranking_is_invariant_to_thread_format_and_older_history(self):
        members, universe = corpus([1] * 7 + [4] * 7)
        before = analyze(members, universe)
        for row in members:
            row["root_id"] = "one-shared-thread"
        old = [voice(f"old-{i}", -10, user=f"old-{i}") for i in range(100)]
        after = analyze(members + old, universe + old)
        for field in ("ranking_support", "ranking_active_days", "ranking_platform_count"):
            self.assertEqual(before["stats"][field], after["stats"][field])
        self.assertEqual(before["score"], after["score"])
        self.assertNotEqual(before["stats"]["thread_count"], after["stats"]["thread_count"])

    def test_ranking_support_is_monotone_and_bounded_by_both_aggregates(self):
        _, universe = corpus([0] * 14)
        additions = [voice(f"add-{i}", 12, f"family-{i % 4}", user=f"author-{i % 7}")
                     for i in range(40)]
        previous = 0
        for count in range(len(additions) + 1):
            members = additions[:count]
            result = analyze(members, universe + members)
            self.assertGreaterEqual(result["score"], previous)
            self.assertLessEqual(result["score"], result["stats"]["provisional_author_day_support"])
            self.assertLessEqual(result["score"], result["stats"]["template_capped_support"])
            self.assertEqual(result["score"], sum(row["ranking_support"] for row in result["stats"]["daily"]
                                                 if row["included_in_status"]))
            previous = result["score"]

    def test_one_author_many_paraphrases_has_one_ranking_unit_per_day(self):
        _, universe = corpus([0] * 14)
        rows = [voice(f"prolific-{i}", 12, f"different wording {i}", user="one-author") for i in range(100)]
        self.assertEqual(analyze(rows, universe + rows)["score"], 1)
        for row in rows:
            row["family"] = "same template"
            row["user_name"] = row["id"]
        result = analyze(rows, universe + rows)
        self.assertEqual(result["score"], 3)
        self.assertEqual(result["stats"]["ranking_recent_support"], 3)

    def test_fading(self):
        members, universe = corpus([8] * 7 + [2] * 7)
        self.assertEqual(analyze(members, universe)["status"], "fading")

    def test_extreme_template_burst_is_flagged_with_caveat(self):
        members, universe = corpus([0] * 14)
        burst = [voice(f"burst-{index}", 13, "same copied template", f"author {index % 3}", hour=12)
                 for index in range(30)]
        result = analyze(burst, universe + burst)
        self.assertEqual(result["status"], "not real")
        self.assertIn("not proof", result["why_en"])
        self.assertEqual(result["stats"]["author_family_capped_support"], 3)

    def test_distinct_authors_can_form_a_concentrated_template_burst(self):
        _, universe = corpus([0] * 14)
        copies = [voice(f"coordinated-{index}", 13, "same copied praise", hour=12)
                  for index in range(30)]
        for index, row in enumerate(copies):
            stamp = (START + timedelta(days=13)).isoformat()
            row["created_at"] = f"{stamp}T12:{index // 3:02d}:{index % 3 * 20:02d}+08:00"
        result = analyze(copies, universe + copies)
        self.assertEqual(result["status"], "not real")
        self.assertEqual(result["stats"]["burst"]["largest_same_minute_count"], 3)
        self.assertEqual(result["stats"]["burst"]["largest_15_minute_count"], 30)
        self.assertEqual(result["stats"]["provisional_author_count"], 30)
        self.assertIn("genuine synchronized reaction remains possible", result["why_en"])

    def test_repetition_spread_over_time_is_not_artificial(self):
        members, universe = corpus([0] * 14)
        copies = [voice(f"copy-{index}", 13, "same concern", hour=index % 24) for index in range(30)]
        result = analyze(copies, universe + copies)
        self.assertNotEqual(result["status"], "not real")

    def test_varied_simultaneous_outage_reports_are_not_artificial(self):
        _, universe = corpus([0] * 14)
        varied = [voice(f"varied-{index}", 13, f"different outage detail {index}", hour=12)
                  for index in range(30)]
        self.assertNotEqual(analyze(varied, universe + varied)["status"], "not real")

    def test_one_author_paraphrasing_more_does_not_establish_growth(self):
        members, universe = corpus([1] * 7 + [10] * 7)
        for row in members:
            row["user_name"] = "same talkative author"
        result = analyze(members, universe)
        self.assertIsNone(result["status"])
        self.assertEqual(result["stats"]["recent_window"]["support"], 7)
        self.assertEqual(result["stats"]["prior_window"]["support"], 7)
        self.assertEqual(result["stats"]["template_capped_support"], 77)
        self.assertEqual(result["stats"]["provisional_author_day_support"], 14)

    def test_more_distinct_author_days_can_establish_growth(self):
        members, universe = corpus([1] * 7 + [10] * 7)
        result = analyze(members, universe)
        self.assertEqual(result["status"], "growing")
        self.assertEqual(result["stats"]["recent_window"]["support"], 70)

    def test_incomplete_terminal_day_excluded_but_evidence_kept(self):
        members, universe = corpus([5] * 14)
        partial = voice("partial", 14, hour=1)
        result = analyze(members + [partial], universe + [partial], 14)
        self.assertEqual(result["stats"]["boundary"]["excluded_day"], "2026-07-15")
        self.assertEqual(result["stats"]["analysis_end"], "2026-07-14")
        self.assertEqual(result["stats"]["raw_count"], len(members) + 1)
        self.assertFalse(result["stats"]["daily"][-1]["included_in_status"])
        self.assertNotEqual(result["status"], "fading")

    def test_quiet_complete_day_not_inferred_incomplete(self):
        members, universe = corpus([5] * 13 + [0], [100] * 13 + [5])
        self.assertIsNone(analyze(members, universe)["stats"]["boundary"]["excluded_day"])

    def test_future_records_cannot_change_result(self):
        members, universe = corpus([1] * 7 + [3] * 7)
        before = analyze(members, universe)
        future = [voice(f"future-{index}", 14) for index in range(200)]
        after = analyze(members + future, universe + future)
        self.assertEqual(before, after)

    def test_conflicting_future_timestamp_is_discarded(self):
        members, universe = corpus([1] * 14)
        record = voice("future-conflict", 13)
        record["created_at"] = "2026-07-15T00:00:00+08:00"
        self.assertEqual(analyze(members, universe), analyze(members + [record], universe + [record]))

    def test_utc_timestamp_uses_game_calendar_day(self):
        members, universe = corpus([1] * 14)
        record = voice("utc-game-day", 13, hour=0)
        record["created_at"] = "2026-07-13T16:30:00Z"  # July 14, 00:30 in game timezone.
        result = analyze(members + [record], universe + [record])
        self.assertEqual(result["stats"]["raw_count"], len(members) + 1)

    def test_burst_window_compares_instants_across_timezones(self):
        _, universe = corpus([0] * 14)
        copies = [voice(f"mixed-zone-{index}", 13, "same copied praise", hour=12)
                  for index in range(16)]
        for index, row in enumerate(copies):
            # Alternating serializations are the same sequence of instants.
            hour, zone = (12, "+08:00") if index % 2 else (4, "+00:00")
            row["created_at"] = f"2026-07-14T{hour:02d}:{index // 2:02d}:00{zone}"
        result = analyze(copies, universe + copies)
        self.assertEqual(result["status"], "not real")
        self.assertEqual(result["stats"]["burst"]["dominant_family_in_15_minutes_count"], 16)

    def test_missing_days_are_not_future_followup(self):
        members, universe = corpus([0] * 8 + [12])
        result = analyze(members, universe, 13)
        self.assertNotEqual(result["status"], "one-off")
        self.assertEqual(result["stats"]["boundary"]["unobserved_days_after_end"], 5)

    def test_empty_and_malformed_data(self):
        result = analyze_history([{"id": "bad", "day": "oops"}], [], "2026-07-14")
        self.assertIsNone(result["status"])
        self.assertIsNone(result["first_seen"])
        self.assertEqual(result["stats"]["raw_count"], 0)

    def test_prepared_history_matches_unprepared_for_multiple_concerns(self):
        members, universe = corpus([1] * 7 + [2, 3, 4, 5, 6, 7, 8])
        context = prepare_history(universe, "2026-07-14")
        for topic in (members, members[:4], [], universe[-10:]):
            self.assertEqual(analyze_history(topic, universe, "2026-07-14", context=context),
                             analyze_history(topic, universe, "2026-07-14"))

    def test_prepared_history_does_not_leak_mutable_result_boundary(self):
        members, universe = corpus([1] * 14)
        context = prepare_history(universe, "2026-07-14")
        first = analyze_history(members, universe, "2026-07-14", context=context)
        first["stats"]["boundary"]["reason"] = "changed by caller"
        second = analyze_history(members, universe, "2026-07-14", context=context)
        self.assertNotEqual(second["stats"]["boundary"]["reason"], "changed by caller")

    def test_prepared_history_rejects_wrong_cutoff_or_universe(self):
        members, universe = corpus([1] * 14)
        context = prepare_history(universe, "2026-07-14")
        with self.assertRaises(ValueError):
            analyze_history(members, universe, "2026-07-13", context=context)
        with self.assertRaises(ValueError):
            analyze_history(members, list(universe), "2026-07-14", context=context)
        universe.append(voice("added", 13))
        with self.assertRaises(ValueError):
            analyze_history(members, universe, "2026-07-14", context=context)

    def test_platform_history_has_cutoff_safe_dates_counts_and_author_proxies(self):
        members = [voice("q1", 0, user="shared", platform="QQ"),
                   voice("q2", 2, user="shared", platform="QQ"),
                   voice("q3", 1, user="other", platform="QQ"),
                   voice("d1", 4, user="shared", platform="Discord")]
        future = voice("future", 14, platform="Steam")
        outside = voice("outside", 1, platform="Unlisted")
        universe = members + [future]
        context = prepare_history(universe, "2026-07-14")
        result = analyze_history(members + [members[0], future, outside], universe, "2026-07-14", context=context)
        self.assertEqual(result["stats"]["platform_history"], {
            "QQ": {"first_seen": "2026-07-01", "last_seen": "2026-07-03",
                   "assigned_message_count": 3, "author_proxy_count": 2},
            "Discord": {"first_seen": "2026-07-05", "last_seen": "2026-07-05",
                        "assigned_message_count": 1, "author_proxy_count": 1},
        })
        self.assertIn("not a verified complaint census", result["stats"]["platform_history_caveat"])

    def test_empty_topic_has_empty_platform_history(self):
        self.assertEqual(analyze_history([], [], "2026-07-14")["stats"]["platform_history"], {})


if __name__ == "__main__":
    unittest.main()
