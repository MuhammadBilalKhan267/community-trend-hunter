"""Auditable, cutoff-safe measurements for an already interpreted concern.

These rules measure the supplied corpus, not the whole player population. Display
names are provisional voices, never verified people. The returned status may be
None: the five requested labels do not include stable or insufficient evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import median
from typing import Mapping, Sequence


GAME_TIMEZONE = timezone(timedelta(hours=8))


def activity_key(item: Mapping) -> tuple:
    """Order observed activity without adding overlapping weighted signals."""
    stats = item.get("history", {}).get("stats", {})
    return (-item.get("score", stats.get("ranking_support", 0)),
            -stats.get("ranking_active_days", 0),
            -stats.get("ranking_platform_count", 0),
            -item.get("ranking_engagement", 0), str(item["id"]))


def _records(records: Sequence[Mapping], cutoff: date) -> list[dict]:
    """Defensively discard malformed, future, and duplicate-ID observations."""
    result = []
    seen = set()
    for record in records:
        if not record.get("id"):
            continue
        identifier = str(record.get("id", ""))
        if not identifier or identifier in seen:
            continue
        try:
            day = date.fromisoformat(str(record.get("day", "")))
        except ValueError:
            continue
        timestamp = None
        if record.get("created_at"):
            try:
                timestamp = datetime.fromisoformat(str(record["created_at"]).replace("Z", "+00:00"))
            except ValueError:
                continue
            # The day field is the game's +08:00 calendar date, even if the
            # source timestamp was serialized in UTC or another timezone.
            timestamp = (timestamp.astimezone(GAME_TIMEZONE) if timestamp.tzinfo
                         else timestamp.replace(tzinfo=GAME_TIMEZONE))
            # Both fields must respect the boundary. Conflicting dates are not
            # silently repaired, since the correct local date is unknown.
            if timestamp.date() != day or timestamp.date() > cutoff:
                continue
        if day > cutoff:
            continue
        seen.add(identifier)
        result.append({
            "id": identifier,
            "day": day,
            "timestamp": timestamp,
            "platform": str(record.get("platform") or "unknown"),
            "author": str(record.get("user_name") or f"unknown:{identifier}"),
            "family": str(record.get("family") or f"unknown:{identifier}"),
            "root": str(record.get("root_id") or identifier),
        })
    return result


def _support(records: Sequence[Mapping]) -> tuple[Counter, Counter, Counter, Counter]:
    """Count raw, author/family, template, and distinct author/day support.

    A display name can support a text family once per platform per day. At most
    three such voices for the identical family contribute per platform/day.
    The latter is a conservative ranking weight, not an estimate of real users.
    """
    raw = Counter((r["day"], r["platform"]) for r in records)
    voices = {(r["day"], r["platform"], r["author"], r["family"]) for r in records}
    capped = Counter((day, platform) for day, platform, _, _ in voices)
    families = Counter((day, platform, family) for day, platform, _, family in voices)
    effective = Counter()
    for (day, platform, _), count in families.items():
        effective[day, platform] += min(3, count)
    author_days = {(r["day"], r["platform"], r["author"]) for r in records}
    author_support = Counter((day, platform) for day, platform, _ in author_days)
    return raw, capped, effective, author_support


def _boundary(records: Sequence[Mapping], cutoff: date) -> dict:
    counts = Counter(r["day"] for r in records)
    if not counts:
        return {"excluded_day": None, "reason": "No valid observed days.", "last_observed_day": None,
                "unobserved_days_after_end": 0, "baseline_median_count": None, "last_timestamp": None}
    last = max(counts)
    preceding = sorted(day for day in counts if last - timedelta(days=14) <= day < last)
    baseline = median(counts[day] for day in preceding) if preceding else None
    timestamps = [r["timestamp"] for r in records if r["day"] == last and r["timestamp"] is not None]
    # Compare local wall-clock values; never mix naive/aware datetime ordering.
    latest = max(timestamps, key=lambda value: value.isoformat()) if timestamps else None
    suspicious = (len(preceding) >= 5 and baseline >= 20 and counts[last] < baseline * .2
                  and latest is not None and latest.hour < 18)
    reason = ("Possible incomplete terminal collection day: total eligible volume is below 20% of the "
              "preceding observed-day median and the last timestamp is before 18:00. Excluded from "
              "status windows; observed evidence remains listed. This is a heuristic, not proof."
              if suspicious else "No strong terminal truncation signal. Day completeness is not guaranteed.")
    return {"excluded_day": last.isoformat() if suspicious else None, "reason": reason,
            "last_observed_day": last.isoformat(), "unobserved_days_after_end": (cutoff - last).days,
            "baseline_median_count": baseline, "last_timestamp": latest.isoformat() if latest else None}


def _burst(records: Sequence[Mapping]) -> dict:
    size = len(records)
    families = Counter(r["family"] for r in records)
    minutes = Counter((r["day"].isoformat(), r["timestamp"].strftime("%H:%M"))
                      for r in records if r["timestamp"] is not None)
    largest_minute = max(minutes.values(), default=0)
    largest_family = max(families.values(), default=0)
    capped = len({(r["day"], r["platform"], r["author"], r["family"]) for r in records})
    dominant_family = families.most_common(1)[0][0] if families else None

    def densest_window(timestamps):
        ordered = sorted(timestamps)
        left, best_count, best_bounds = 0, 0, None
        for right, stamp in enumerate(ordered):
            while stamp - ordered[left] > timedelta(minutes=15):
                left += 1
            if right - left + 1 > best_count:
                best_count = right - left + 1
                best_bounds = (ordered[left], stamp)
        return best_count, best_bounds

    timed = [r for r in records if r["timestamp"] is not None]
    largest_window, _ = densest_window([r["timestamp"] for r in timed])
    joint_count, bounds = densest_window([r["timestamp"] for r in timed if r["family"] == dominant_family])
    day_counts = Counter(r["day"] for r in records)
    day_fraction = max(day_counts.values(), default=0) / max(1, size)
    outside = [r for r in records if not bounds or r["timestamp"] is None
               or not bounds[0] <= r["timestamp"] <= bounds[1]]
    organic_outside = {(r["day"], r["platform"], r["family"]) for r in outside
                       if r["family"] != dominant_family}
    # Distinct display names do not validate organic independence. A burst can
    # consist of accounts posting the same template once each. Conversely, a
    # varied outage discussion is not rejected merely because it is simultaneous.
    extreme = (size >= 12 and joint_count / size >= .85 and day_fraction >= .85
               and len(families) <= max(2, int(size * .15)) and len(organic_outside) <= 2)
    return {
        "largest_same_minute_count": largest_minute,
        "largest_same_minute_fraction": round(largest_minute / max(1, size), 4),
        "largest_15_minute_count": largest_window,
        "dominant_family_in_15_minutes_count": joint_count,
        "dominant_family_in_15_minutes_fraction": round(joint_count / max(1, size), 4),
        "dominant_template_window_start": bounds[0].isoformat() if bounds else None,
        "dominant_template_window_end": bounds[1].isoformat() if bounds else None,
        "largest_day_fraction": round(day_fraction, 4),
        "organic_families_outside_burst": len(organic_outside),
        "largest_family_count": largest_family,
        "largest_family_fraction": round(largest_family / max(1, size), 4),
        "distinct_families": len(families),
        "raw_to_author_family_capped_ratio": round(size / max(1, capped), 4),
        "extreme_template_burst": extreme,
        "caveat": "Concentration can flag apparent amplification; it cannot prove coordination or fake accounts.",
    }


@dataclass(frozen=True)
class PreparedHistory:
    """Reusable cutoff-specific universe measurements for immutable input."""

    source: Sequence[Mapping]
    source_count: int
    as_of: str
    records: list[dict]
    eligible_ids: frozenset
    raw: Counter
    effective: Counter
    author_support: Counter
    boundary: dict
    observed: frozenset
    complete: frozenset
    analysis_end: date
    recent_start: date
    prior_start: date
    prior_end: date
    recent_days: frozenset
    prior_days: frozenset
    platforms: tuple


def prepare_history(universe: Sequence[Mapping], as_of: str) -> PreparedHistory:
    """Parse and count a shared universe once for many concern histories.

    Keep the source sequence unchanged until all its histories are measured.
    A different cutoff, source object or source length requires a new context.
    """
    cutoff = date.fromisoformat(as_of)
    all_records = _records(universe, cutoff)
    universe_raw, _, universe_effective, universe_author_support = _support(all_records)
    boundary = _boundary(all_records, cutoff)
    observed = frozenset(r["day"] for r in all_records)
    excluded = date.fromisoformat(boundary["excluded_day"]) if boundary["excluded_day"] else None
    complete = observed - ({excluded} if excluded else set())
    analysis_end = max(complete) if complete else cutoff
    recent_start = analysis_end - timedelta(days=6)
    prior_start = analysis_end - timedelta(days=13)
    prior_end = recent_start - timedelta(days=1)
    return PreparedHistory(
        source=universe, source_count=len(universe), as_of=as_of, records=all_records,
        eligible_ids=frozenset(r["id"] for r in all_records), raw=universe_raw,
        effective=universe_effective, author_support=universe_author_support,
        boundary=boundary, observed=observed, complete=frozenset(complete),
        analysis_end=analysis_end, recent_start=recent_start, prior_start=prior_start,
        prior_end=prior_end,
        recent_days=frozenset(day for day in complete if recent_start <= day <= analysis_end),
        prior_days=frozenset(day for day in complete if prior_start <= day <= prior_end),
        platforms=tuple(sorted({r["platform"] for r in all_records})),
    )


def analyze_history(members: Sequence[Mapping], universe: Sequence[Mapping], as_of: str,
                    *, context: PreparedHistory | None = None) -> dict:
    """Measure and classify a concern using only observations on/before as_of.

    `universe` is eligible substantive/contextual voices, not all scraped chatter.
    Member IDs absent from that universe are ignored to keep shares meaningful.
    Returned `score` is an explicit ranking heuristic, never a probability.
    """
    cutoff = date.fromisoformat(as_of)
    if context is None:
        context = prepare_history(universe, as_of)
    if not isinstance(context, PreparedHistory) or context.as_of != as_of:
        raise ValueError("Prepared history cutoff differs from requested cutoff")
    if context.source is not universe or context.source_count != len(universe):
        raise ValueError("Prepared history requires the same unchanged universe")
    eligible_ids = context.eligible_ids
    candidate_records = _records(members, cutoff)
    topic = [r for r in candidate_records if r["id"] in eligible_ids]
    raw, capped, effective, author_support = _support(topic)
    universe_raw, universe_effective, universe_author_support = context.raw, context.effective, context.author_support
    boundary = dict(context.boundary)
    observed = context.observed
    excluded = date.fromisoformat(boundary["excluded_day"]) if boundary["excluded_day"] else None
    complete, analysis_end = context.complete, context.analysis_end
    recent_start, prior_start, prior_end = context.recent_start, context.prior_start, context.prior_end
    recent_days, prior_days = context.recent_days, context.prior_days
    platforms = context.platforms

    def total(counter, days, platform=None):
        return sum(count for (day, source), count in counter.items()
                   if day in days and (platform is None or source == platform))

    platform_windows = {}
    common_platforms = []
    for platform in platforms:
        recent_denominator = total(universe_author_support, recent_days, platform)
        prior_denominator = total(universe_author_support, prior_days, platform)
        recent_numerator = total(author_support, recent_days, platform)
        prior_numerator = total(author_support, prior_days, platform)
        platform_windows[platform] = {
            "recent_support": recent_numerator, "prior_support": prior_numerator,
            "recent_universe": recent_denominator, "prior_universe": prior_denominator,
            "recent_share": recent_numerator / recent_denominator if recent_denominator else None,
            "prior_share": prior_numerator / prior_denominator if prior_denominator else None,
        }
        if recent_denominator and prior_denominator:
            common_platforms.append(platform)

    # A fixed, equal-weight platform set avoids turning platform-mix shifts into
    # topic growth. Jeffreys-style 0.5/1 pseudocounts stabilize small denominators;
    # these are descriptive smoothed shares, not inferential probabilities.
    def normalized_share(period):
        values = [((platform_windows[p][f"{period}_support"] + .5)
                   / (platform_windows[p][f"{period}_universe"] + 1)) for p in common_platforms]
        return sum(values) / len(values) if values else None

    recent_share = normalized_share("recent")
    prior_share = normalized_share("prior")
    growth = recent_share / prior_share if prior_share else None
    recent_support = total(author_support, recent_days)
    prior_support = total(author_support, prior_days)
    recent_rate = recent_support / len(recent_days) if recent_days else 0
    prior_rate = prior_support / len(prior_days) if prior_days else 0
    rate_ratio = (recent_rate + .5) / (prior_rate + .5)
    daily_effective = Counter()
    for (day, _), count in author_support.items():
        if day in complete:
            daily_effective[day] += count
    active_days = sorted(daily_effective)
    first = min((r["day"] for r in topic), default=None)
    last = max((r["day"] for r in topic), default=None)
    status = None
    reason = "Observed support does not meet a supported new/growing/one-off/fading classification."
    diagnostics = _burst(topic)
    enough_coverage = len(recent_days) >= 3 and len(prior_days) >= 3
    peak_day = max(daily_effective, key=lambda day: (daily_effective[day], -day.toordinal())) if daily_effective else None
    peak_support = daily_effective[peak_day] if peak_day else 0
    after_peak = sorted(day for day in complete if peak_day and day > peak_day)
    total_effective = sum(daily_effective.values())
    after_support = sum(daily_effective[day] for day in after_peak)
    one_off = (peak_support >= 4 and peak_support / max(1, total_effective) >= .6
               and len(active_days) <= 3 and len(after_peak) >= 2
               and after_support / len(after_peak) <= peak_support * .25)
    before_first = {day for day in complete if first and day < first}
    recent_active_days = sum(day in recent_days for day in active_days)
    if diagnostics["extreme_template_burst"]:
        status = "not real"
        reason = ("At least 85% of messages repeat one template within 15 minutes, concentrated on one "
                  "day with little distinct content outside the burst. This flags apparent amplification "
                  "and weak independent evidence, not proof of coordinated or fake accounts. A genuine "
                  "synchronized reaction remains possible.")
    elif (enough_coverage and prior_support >= 6 and growth is not None and growth <= .65
          and rate_ratio < 1 and len(active_days) >= 3):
        status = "fading"
        reason = "Both platform-normalized share and support per observed day declined across the two seven-day windows."
    elif (enough_coverage and recent_support >= 5 and prior_support >= 3
          and recent_active_days >= 2 and growth is not None and growth >= 1.35 and rate_ratio > 1):
        status = "growing"
        reason = "Both platform-normalized share and support per observed day increased, with support on multiple recent days."
    elif (first is not None and first >= recent_start and recent_support >= 3
          and recent_active_days >= 2 and len(before_first) >= 3 and prior_support == 0):
        status = "new"
        reason = "The concern first appears in the recent window after at least three earlier observed days, with support on multiple days."
    elif one_off:
        status = "one-off"
        reason = (f"Support peaked at {peak_support} provisional author-days on {peak_day.isoformat()}, "
                  f"at least 60% of observed support; {len(after_peak)} subsequent observed days show "
                  "a low tail. A short incident is supported by the observed follow-up.")

    # Each bound is monotone under additional observations. Taking their minimum
    # limits aggregate credit, without pretending to solve an exact joint
    # assignment of authors to text families.
    ranking_support = Counter({key: min(count, effective[key]) for key, count in author_support.items()})
    ranking_days = recent_days | prior_days
    ranking_recent = total(ranking_support, recent_days)
    ranking_prior = total(ranking_support, prior_days)
    ranking_records = [r for r in topic if r["day"] in ranking_days]

    daily = []
    if observed:
        day = min(observed)
        while day <= cutoff:
            active_platforms = [p for p in platforms if universe_author_support[day, p]]
            shares = {p: author_support[day, p] / universe_author_support[day, p] for p in active_platforms}
            daily.append({
                "day": day.isoformat(), "observed": day in observed,
                "included_in_status": day in complete,
                "raw_count": sum(raw[day, p] for p in platforms),
                "author_family_capped_support": sum(capped[day, p] for p in platforms),
                "template_capped_support": sum(effective[day, p] for p in platforms),
                "ranking_support": sum(ranking_support[day, p] for p in platforms),
                "provisional_author_day_support": sum(author_support[day, p] for p in platforms),
                "universe_raw_count": sum(universe_raw[day, p] for p in platforms),
                "universe_template_capped_support": sum(universe_effective[day, p] for p in platforms),
                "universe_provisional_author_day_support": sum(universe_author_support[day, p] for p in platforms),
                "platform_shares": shares,
                "mean_active_platform_share": sum(shares.values()) / len(shares) if shares else None,
            })
            day += timedelta(days=1)

    authors = {(r["platform"], r["author"]) for r in topic}
    threads = {(r["platform"], r["root"]) for r in topic}
    score_components = {"recent_ranking_support": ranking_recent, "prior_ranking_support": ranking_prior}
    score = ranking_recent + ranking_prior
    platform_history = {}
    for platform in sorted({r["platform"] for r in topic}):
        assigned = [r for r in topic if r["platform"] == platform]
        platform_history[platform] = {
            "first_seen": min(r["day"] for r in assigned).isoformat(),
            "last_seen": max(r["day"] for r in assigned).isoformat(),
            "assigned_message_count": len(assigned),
            "author_proxy_count": len({r["author"] for r in assigned}),
        }
    stats = {
        "as_of": as_of, "analysis_end": analysis_end.isoformat(),
        "raw_count": len(topic), "author_family_capped_support": sum(capped.values()),
        "template_capped_support": sum(effective.values()), "active_days": len({r["day"] for r in topic}),
        "ranking_support": score, "ranking_recent_support": ranking_recent,
        "ranking_prior_support": ranking_prior,
        "ranking_active_days": len({r["day"] for r in ranking_records}),
        "ranking_platform_count": len({r["platform"] for r in ranking_records}),
        "provisional_author_day_support": sum(author_support.values()),
        "status_active_days": len(active_days), "thread_count": len(threads),
        "provisional_author_count": len(authors),
        "author_caveat": "Display names are provisional; cross-platform names do not identify the same person. Shared display names can undercount voices.",
        "first_seen_caveat": "First seen means first in the supplied corpus, not necessarily true onset.",
        "last_seen": last.isoformat() if last else None,
        "platform_count": len({r["platform"] for r in topic}),
        "platform_history": platform_history,
        "platform_history_caveat": "Dates and counts describe cutoff-eligible assigned topic members, not a verified complaint census or true onset; author proxies are display names within each platform.",
        "recent_window": {"start": recent_start.isoformat(), "end": analysis_end.isoformat(),
                          "observed_days": len(recent_days), "support": recent_support,
                          "raw_count": total(raw, recent_days), "support_per_observed_day": recent_rate},
        "prior_window": {"start": prior_start.isoformat(), "end": prior_end.isoformat(),
                         "observed_days": len(prior_days), "support": prior_support,
                         "raw_count": total(raw, prior_days), "support_per_observed_day": prior_rate},
        "platform_windows": platform_windows, "comparison_platforms": common_platforms,
        "recent_normalized_smoothed_share": recent_share, "prior_normalized_smoothed_share": prior_share,
        "normalized_growth_ratio": growth, "smoothed_daily_support_ratio": rate_ratio,
        "template_concentration": diagnostics["largest_family_fraction"],
        "burst": diagnostics, "boundary": boundary,
        "unobserved_calendar_days": sum(not row["observed"] for row in daily),
        "ignored_members_outside_universe": len(candidate_records) - len(topic),
        "peak_day": peak_day.isoformat() if peak_day else None, "peak_support": peak_support,
        "observed_days_after_peak": len(after_peak), "daily": daily,
        "score_components": {key: round(value, 4) for key, value in score_components.items()},
        "score_caveat": "Unbounded aggregate ranking support in the paired seven-day windows; not a probability or calibrated importance estimate.",
        "ranking_policy": "Per platform/day Q=min(distinct author proxies, sum over text families of min(3, distinct author proxies)). Q14 sums Q over the two observed seven-day windows. Active days, platforms and engagement break ties; no exact joint author-family allocation is claimed.",
        "support_policy": "Status windows and normalized shares count each platform/display-name/day once in both topic and universe. Ranking takes the smaller aggregate author-day and template-capped bounds per platform/day; repeated author/family contributions count once and each family has cap 3.",
        "status_thresholds": {"growth_share_ratio": 1.35, "growth_daily_ratio_strictly_above": 1,
                              "fading_share_ratio": .65, "fading_daily_ratio_strictly_below": 1,
                              "minimum_observed_days_per_comparison_window": 3},
    }
    comparisons = (f" Recent/prior provisional-author-day support: {recent_support}/{prior_support}; "
                   f"observed days: {len(recent_days)}/{len(prior_days)}.")
    if growth is not None:
        comparisons += f" Smoothed normalized share ratio: {growth:.2f}."
    if excluded:
        comparisons += f" Possible incomplete collection on {excluded.isoformat()} is excluded from status windows."
    if boundary["unobserved_days_after_end"]:
        comparisons += (f" The last {boundary['unobserved_days_after_end']} calendar days have no eligible "
                        "observations; missing collection is not evidence of silence.")
    return {"first_seen": first.isoformat() if first else None, "status": status,
            "status_supported": status is not None, "score": round(score, 4),
            "why_en": reason + comparisons, "stats": stats}
