"""Cutoff-safe candidate ranking, capped sampling, and discussion nominations.

Vectors propose topical groups. Nearby responses remain possible engagement,
never automatic topical membership or independent complaint support.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from math import ceil, log1p
import re

import numpy as np

from .data import GAME_TIMEZONE, normalize, reaction_kind
from .temporal import activity_key, analyze_history, prepare_history

_SHORT_REACTIONS = {
    "yes", "yeah", "yea", "yep", "no", "nah", "same", "agree", "bad", "nice",
    "true", "indeed", "seems nice", "好", "坏", "壞", "是", "对", "對", "嗯", "啊",
    "不", "不是", "不好", "有道理", "确实", "確實", "赞", "贊", "同感", "+1",
}
_ALIAS_TOKEN = r"[\w\u3400-\u9fff·.-]{2,24}"
_ALIAS_PATTERNS = (
    re.compile(rf"(?P<name>{_ALIAS_TOKEN})\s*[=＝]\s*(?P<alias>{_ALIAS_TOKEN})"),
    re.compile(rf"(?P<name>{_ALIAS_TOKEN})[的]?\s*(?:昵称|暱稱|外号|外號|绰号|綽號)(?:是|叫|为|為|[:：])\s*[「『\"“]?(?P<alias>{_ALIAS_TOKEN})"),
    re.compile(rf"(?P<name>{_ALIAS_TOKEN})\s*(?:也叫|又叫|又称|又稱)\s*[「『\"“]?(?P<alias>{_ALIAS_TOKEN})"),
)


def _stamp(record):
    value = record.get("_instant") or record.get("created_at")
    try:
        value = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return value.astimezone(GAME_TIMEZONE) if value.tzinfo else value.replace(tzinfo=GAME_TIMEZONE)
    except (ValueError, TypeError):
        return None


def _visible(records, as_of):
    cutoff = date.fromisoformat(as_of)
    result, seen = [], set()
    for record in records:
        stamp = _stamp(record)
        identifier = record.get("id")
        if identifier is None or identifier in seen or stamp is None or stamp.date() > cutoff:
            continue
        if record.get("day") and str(record["day"]) != stamp.date().isoformat():
            continue
        seen.add(identifier)
        result.append(record)
    return sorted(result, key=lambda r: (_stamp(r), str(r["id"])))


def _stream(record):
    # Missing channel information cannot establish shared conversation context.
    return (record.get("platform"), record.get("channel")) if record.get("channel") else None


def _text(record):
    return record.get("text") or normalize(record.get("content") or record.get("title"))


def _short_response(record):
    text = _text(record).strip()
    bare = re.sub(r"[\s,.!！?？。~～😂😭💀]+$", "", text).lower()
    return bool(record.get("reaction") or reaction_kind(text) or bare in _SHORT_REACTIONS)


def _overall_key(candidate):
    # Spend ordinary review capacity on observable changes first. Stable groups
    # remain eligible for the independent unusual queue and later backfill;
    # an aggregate can hide a changing subtopic.
    changing = bool(candidate.get("history", {}).get("status_supported"))
    return (not changing, *activity_key(candidate))


def _positive_key(candidate):
    stats = candidate.get("history", {}).get("stats", {})
    growth = stats.get("normalized_growth_ratio")
    return (-(growth if growth is not None else 0),
            -stats.get("ranking_recent_support", stats.get("recent_window", {}).get("support", 0)),
            *activity_key(candidate))


def _interleave(*queues):
    """Build a deterministic full-pool schedule before removing reviewed IDs."""
    for position in range(max((len(queue) for queue in queues), default=0)):
        for queue in queues:
            if position < len(queue):
                yield queue[position]


def _unusual_queue(pool):
    eligible = [c for c in pool if c.get("history", {}).get("status") not in {"new", "growing"}]
    fading = sorted((c for c in eligible if c.get("history", {}).get("status") == "fading"),
                    key=lambda c: ((c.get("history", {}).get("stats", {}).get("normalized_growth_ratio")
                                    if c.get("history", {}).get("stats", {}).get("normalized_growth_ratio") is not None else 1),
                                   *activity_key(c)))
    fading_ids = {c["id"] for c in fading}
    discussion, templates, other = [], [], []
    for candidate in eligible:
        if candidate["id"] in fading_ids:
            continue
        diagnostics = candidate.get("diagnostics", {})
        if candidate.get("kind") == "burst" or diagnostics.get("discussion_bursts"):
            discussion.append(candidate)
        elif candidate.get("history", {}).get("stats", {}).get("burst", {}).get("extreme_template_burst"):
            templates.append(candidate)
        else:
            other.append(candidate)

    def discussion_strength(candidate):
        diagnostics = candidate.get("diagnostics", {})
        strengths = [diagnostics.get("engagement_score", 0)]
        strengths += [row.get("engagement_score", 0) for row in diagnostics.get("discussion_bursts", [])]
        return (-max(strengths), *activity_key(candidate))

    def template_strength(candidate):
        burst = candidate.get("history", {}).get("stats", {}).get("burst", {})
        return (-burst.get("dominant_family_in_15_minutes_fraction", 0),
                -burst.get("dominant_family_in_15_minutes_count", 0), *activity_key(candidate))

    # Detector strengths have different units. Interleave their own orderings
    # rather than inventing a conversion between response lift and repetition.
    bursts = list(_interleave(sorted(discussion, key=discussion_strength),
                              sorted(templates, key=template_strength)))
    return list(_interleave(fading, bursts, sorted(other, key=activity_key)))


def extract_aliases(records, as_of):
    """Return literal, source-backed assertions; never rewrite embedding text."""
    result, seen = [], {}
    for record in _visible(records, as_of):
        text = normalize(record.get("content") or record.get("title"))
        for pattern in _ALIAS_PATTERNS:
            for match in pattern.finditer(text):
                name, alias = match.group("name"), match.group("alias")
                key = (name, alias)
                if name == alias:
                    continue
                if key in seen:
                    seen[key]["observed_assertion_count"] += 1
                else:
                    assertion = {"name": name, "alias": alias, "source_id": record["id"],
                                   "source_text": text, "created_at": record.get("created_at"),
                                   "observed_assertion_count": 1,
                                   "caveat": "A player-supplied alias assertion, not an established global substitution."}
                    seen[key] = assertion
                    result.append(assertion)
    return result


def build_context_index(records, as_of):
    visible = _visible(records, as_of)
    by_id = {r["id"]: r for r in visible}
    streams = defaultdict(list)
    replies = defaultdict(list)
    for record in visible:
        if _stream(record):
            streams[_stream(record)].append(record)
        parent = by_id.get(record.get("parent_id"))
        if (parent is not None and _stream(record) is not None and _stream(parent) == _stream(record)
                and _stamp(parent) <= _stamp(record)):
            replies[record["parent_id"]].append(record)
    return {"as_of": as_of, "by_id": by_id, "streams": dict(streams),
            "times": {key: [_stamp(r) for r in rows] for key, rows in streams.items()},
            "replies": dict(replies)}


def _measure(members, universe, as_of, similarities=(), history_context=None):
    history = analyze_history(members, universe, as_of, context=history_context)
    days = sorted({date.fromisoformat(r["day"]) for r in members})
    span = (days[-1] - days[0]).days + 1 if days else 0
    daily = Counter(r["day"] for r in members)
    activity = np.asarray([daily[(days[0] + timedelta(days=i)).isoformat()] for i in range(span)]) if span else np.array([])
    diagnostics = {
        "active_day_fraction": len(days) / max(1, span),
        "maximum_inactive_gap_days": max(((b - a).days - 1 for a, b in zip(days, days[1:])), default=0),
        "daily_coefficient_of_variation": float(activity.std() / activity.mean()) if len(activity) and activity.mean() else 0.,
        "mean_cosine_to_center": float(np.mean(similarities)) if len(similarities) else None,
        "minimum_cosine_to_center": float(np.min(similarities)) if len(similarities) else None,
    }
    return history, diagnostics, history["score"]


def build_candidate_pool(units, vectors, vector_indices, labels, centers, records, eligible, as_of):
    """Build all ranked proposals without semantic review or future evidence."""
    index = build_context_index(records, as_of)
    by_id = index["by_id"]
    universe = _visible(eligible, as_of)
    history_context = prepare_history(universe, as_of)
    eligible_ids = {r["id"] for r in universe}
    aliases = extract_aliases(records, as_of)
    grouped, voice_clusters = defaultdict(list), defaultdict(set)
    for unit_index, unit in enumerate(units):
        if unit["voice_id"] not in eligible_ids:
            continue
        vector_index = int(vector_indices[unit_index])
        label = int(labels[vector_index])
        grouped[label].append(unit_index)
        voice_clusters[unit["voice_id"]].add(f"cluster:{label}")
    pool = []
    for label, indices in sorted(grouped.items()):
        member_ids = sorted({units[i]["voice_id"] for i in indices})
        members = [by_id[id] for id in member_ids]
        similarities = [float(vectors[int(vector_indices[i])] @ centers[label]) for i in indices]
        history, diagnostics, score = _measure(members, universe, as_of, similarities, history_context)
        replies = {r["id"]: r for id in member_ids for r in index["replies"].get(id, [])
                   if r["id"] not in member_ids}
        diagnostics.update({"explicit_reply_count": len(replies),
                            "explicit_reply_author_proxies": len({(r["platform"], r.get("user_name")) for r in replies.values()}),
                            "explicit_reply_ids": sorted(replies),
                            "image_message_count": sum("<image:" in str(r.get("content") or "") for r in members),
                            "image_evidence_caveat": "Image placeholders establish attachment presence, not image content or topic.",
                            "support_caveat": "Cluster assignments are proposed topic volume; reply engagement is separate and does not establish endorsement."})
        pool.append({"id": f"cluster:{label}", "kind": "cluster", "label": label, "indices": indices,
                     "member_ids": member_ids, "history": history, "score": score,
                     "ranking_engagement": len(replies),
                     "diagnostics": diagnostics, "center": centers[label],
                     "anchor_id": None, "related_clusters": [], "available_aliases": aliases})
    pool.extend(_burst_candidates(index, universe, as_of, voice_clusters, aliases, history_context=history_context))
    return sorted(pool, key=_overall_key)


def _burst_candidates(index, universe, as_of, voice_clusters, aliases, window_seconds=600, history_context=None):
    candidates = []
    for stream, rows in sorted(index["streams"].items()):
        times = index["times"][stream]
        short = [_short_response(r) for r in rows]
        short_prefix = [0]
        for flag in short:
            short_prefix.append(short_prefix[-1] + int(flag))
        used_until = None
        for position, anchor in enumerate(rows):
            stamp = times[position]
            if used_until and stamp <= used_until:
                continue
            if short[position] or anchor.get("trivial") or len(_text(anchor)) < 8:
                continue
            end = bisect_right(times, stamp + timedelta(seconds=window_seconds))
            response_count = short_prefix[end] - short_prefix[position + 1]
            if response_count < 4:
                continue
            baseline_start = bisect_left(times, stamp - timedelta(hours=1))
            baseline_seconds = max(window_seconds, (stamp - times[baseline_start]).total_seconds()) if baseline_start < position else 3600
            expected = (short_prefix[position] - short_prefix[baseline_start]) * window_seconds / baseline_seconds
            lift = (response_count + 1) / (expected + 1)
            fraction = response_count / max(1, end - position - 1)
            if lift < 1.5 or fraction < .35:
                continue
            responses = [rows[i] for i in range(position + 1, end) if short[i]]
            authors = {(r["platform"], r.get("user_name")) for r in responses}
            if len(authors) < 2:
                continue
            competing = [rows[i]["id"] for i in range(position + 1, end) if not short[i] and len(_text(rows[i])) >= 8]
            member_ids = [anchor["id"]]  # Reactions never become issue support.
            history, diagnostics, score = _measure([anchor], universe, as_of, history_context=history_context)
            engagement = log1p(len(responses)) * log1p(len(authors)) * min(4., lift) / (1 + len(competing))
            diagnostics.update({"possible_response_count": len(responses), "possible_response_ids": [r["id"] for r in responses],
                                "response_author_proxies": len(authors), "response_fraction": fraction,
                                "background_expected_responses": expected, "response_activity_lift": lift,
                                "competing_anchor_ids": competing, "start": stamp.isoformat(),
                                "end": times[end - 1].isoformat(), "engagement_score": engagement,
                                "support_caveat": "A discussion nomination only: proximity does not establish response targets or endorsement."})
            candidates.append({"id": f"burst:{anchor['id']}", "kind": "burst", "indices": [],
                               "member_ids": member_ids, "history": history, "score": score,
                               "ranking_engagement": len(responses),
                               "diagnostics": diagnostics,
                               "anchor_id": anchor["id"], "related_clusters": sorted(voice_clusters.get(anchor["id"], ())),
                               "available_aliases": aliases})
            used_until = times[end - 1]
    return candidates


def choose_batch(pool, reviewed, n=10, ratio=.5):
    """Reserve exploration for positive changes and unusual activity separately.

    At the default ten-candidate batch, five ordinary slots are followed by
    three new/growing slots and two unusual slots. Empty queues backfill;
    review limits belong to the caller, not this ranking function.
    """
    if n <= 0:
        return []
    if not 0 <= ratio <= 1:
        raise ValueError("ratio must be between zero and one")
    reviewed = set(reviewed)
    remaining = [c for c in pool if c["id"] not in reviewed]
    overall = sorted(remaining, key=_overall_key)
    positive = sorted((c for c in remaining if c.get("history", {}).get("status") in {"new", "growing"}),
                      key=_positive_key)
    unusual = [c for c in _unusual_queue(pool) if c["id"] not in reviewed]
    result, selected = [], set()
    ordinary_quota = ceil(n * (1 - ratio))
    exploration_quota = n - ordinary_quota
    positive_quota = round(exploration_quota * .6)
    for queue, quota in ((overall, ordinary_quota), (positive, positive_quota),
                         (unusual, exploration_quota - positive_quota)):
        added = 0
        if not quota:
            continue
        for candidate in queue:
            if candidate["id"] not in selected:
                result.append(candidate)
                selected.add(candidate["id"])
                added += 1
                if added >= quota:
                    break
    # Preserve exploration when its positive subqueue is empty; older cached
    # candidate summaries without temporal fields also follow this fallback.
    for candidate in unusual + positive + overall:
        if len(result) >= n:
            break
        if candidate["id"] not in selected:
            result.append(candidate)
            selected.add(candidate["id"])
    return result[:n]


def choose_neighbors(pool, reviewed, concepts, limit=2, neighbor_count=8):
    """Nominate bounded neighboring branches for semantic review, never merge.

    Geometry restricts ordinary nominations to the nearest ``neighbor_count``
    centers. Strong source overlap can expose a split outside that radius.
    Supported opposite directions with nontrivial support are reviewed first:
    combining such branches, if the decision model confirms a shared concern,
    may reverse the apparent trend. The scheduler tracks attempted audits per
    concept; this pure function cannot infer previously rejected attempts.
    """
    if limit <= 0 or neighbor_count <= 0:
        return []
    reviewed = set(reviewed)
    lookup = {c["id"]: c for c in pool}
    clusters = [c for c in pool if c.get("kind") == "cluster"]

    def base(identifier):
        return identifier.split(":refine:", 1)[0]

    def center(candidate):
        value = np.asarray(candidate.get("center", []), dtype=float)
        norm = float(np.linalg.norm(value))
        return value / norm if value.ndim == 1 and norm > 0 else None

    normalized = {c["id"]: center(c) for c in clusters}

    def direction(history):
        if not history.get("status_supported", bool(history.get("status"))):
            return 0
        return -1 if history.get("status") == "fading" else (1 if history.get("status") in {"new", "growing"} else 0)

    nominations = []
    for concept in concepts:
        seed_ids = {base(identifier) for identifier in concept.get("candidate_ids", [])}
        seeds = [normalized[i] for i in seed_ids if i in normalized and normalized[i] is not None]
        members = set(concept.get("member_ids", []))
        if not members:
            members = {i for identifier in seed_ids if identifier in lookup for i in lookup[identifier].get("member_ids", [])}
        candidates = []
        for candidate in clusters:
            if candidate["id"] in seed_ids:
                continue
            vector = normalized[candidate["id"]]
            cosine = max((float(vector @ seed) for seed in seeds if vector is not None and vector.shape == seed.shape), default=None)
            sources = set(candidate.get("member_ids", []))
            intersection = len(members & sources)
            union = len(members | sources)
            jaccard = intersection / max(1, union)
            smaller_fraction = intersection / max(1, min(len(members), len(sources)))
            candidates.append((candidate, cosine, intersection, jaccard, smaller_fraction))
        geometric = sorted((entry for entry in candidates if entry[1] is not None),
                           key=lambda entry: (-entry[1], entry[0]["id"]))
        ranks = {entry[0]["id"]: rank + 1 for rank, entry in enumerate(geometric)}
        nearest = {entry[0]["id"] for entry in geometric[:neighbor_count]}
        local = []
        for candidate, cosine, intersection, jaccard, smaller_fraction in candidates:
            if candidate["id"] in reviewed:
                continue
            strong_overlap = intersection >= 3 and (jaccard >= .2 or smaller_fraction >= .5)
            if candidate["id"] not in nearest and not strong_overlap:
                continue
            history = candidate.get("history", {})
            windows = history.get("stats", {})
            support = max(windows.get("recent_window", {}).get("support", 0),
                          windows.get("prior_window", {}).get("support", 0))
            meaningful = support >= 3 or len(set(candidate.get("member_ids", []))) >= 5
            opposing = meaningful and direction(concept.get("history", {})) * direction(history) == -1
            reason = "opposing_direction" if opposing else ("source_overlap" if strong_overlap else "nearby_center")
            context = {"concept_id": concept["id"], "seed_candidate_ids": sorted(seed_ids),
                       "reason": reason, "cosine": cosine, "source_jaccard": jaccard,
                       "source_overlap_count": intersection, "source_smaller_fraction": smaller_fraction,
                       "geometric_neighbor_rank": ranks.get(candidate["id"]), "opposing_direction": bool(opposing)}
            key = (not opposing, not strong_overlap, -jaccard if strong_overlap else 0,
                   -(cosine if cosine is not None else -1), -candidate.get("score", 0), candidate["id"])
            local.append((key, dict(candidate, neighbor_context=context)))
        # A large concept cannot consume the entire batch. The caller also caps
        # cumulative attempts at two when calling incrementally with limit=1.
        for key, candidate in sorted(local, key=lambda item: item[0])[:2]:
            nominations.append((key + (-concept.get("score", 0), str(concept["id"])), candidate))
    result, selected = [], set()
    for _, candidate in sorted(nominations, key=lambda item: item[0]):
        if candidate["id"] not in selected:
            result.append(candidate)
            selected.add(candidate["id"])
            if len(result) >= limit:
                break
    return result


def merge_related_bursts(pool):
    """Enrich unambiguous cluster candidates without increasing issue volume.

    A multi-cluster anchor stays a separate proposal: code cannot establish
    which of its concerns the reactions address. Safe to call repeatedly.
    """
    clusters = {}
    for candidate in pool:
        if candidate["kind"] == "cluster":
            copy = dict(candidate, diagnostics=dict(candidate["diagnostics"]))
            copy["diagnostics"]["discussion_bursts"] = list(candidate["diagnostics"].get("discussion_bursts", []))
            clusters[copy["id"]] = copy
    separate = []
    for candidate in pool:
        if candidate["kind"] != "burst":
            continue
        related = set(candidate.get("related_clusters", []))
        if len(related) != 1 or next(iter(related)) not in clusters:
            separate.append(candidate)
            continue
        cluster = clusters[next(iter(related))]
        bursts = cluster["diagnostics"]["discussion_bursts"]
        if not any(b["candidate_id"] == candidate["id"] for b in bursts):
            bursts.append(dict(candidate["diagnostics"], candidate_id=candidate["id"], anchor_id=candidate["anchor_id"]))
    for cluster in clusters.values():
        diagnostics = cluster["diagnostics"]
        bursts = diagnostics["discussion_bursts"]
        if not bursts:
            continue
        response_ids = {id for burst in bursts for id in burst.get("possible_response_ids", [])}
        diagnostics.update(discussion_burst_count=len(bursts), discussion_possible_response_count=len(response_ids),
                           discussion_engagement_caveat="Capped nomination signal, not confirmed attribution or independent reports.")
        cluster["ranking_engagement"] = max(diagnostics.get("explicit_reply_count", 0), len(response_ids))
    return sorted(list(clusters.values()) + separate, key=_overall_key)


def _packet_diagnostics(diagnostics):
    """Keep the full source-ID inventory internal, outside bounded prompts."""
    result = {}
    source_lists = {"possible_response_ids", "explicit_reply_ids", "competing_anchor_ids"}
    for key, value in diagnostics.items():
        if key in source_lists:
            result[key.removesuffix("_ids") + "_source_count"] = len(value)
        elif key == "discussion_bursts":
            ordered = sorted(value, key=lambda b: (-b.get("engagement_score", 0), b["candidate_id"]))
            result[key] = [_packet_diagnostics(b) for b in ordered[:3]]
            result["discussion_burst_count"] = len(value)
        else:
            result[key] = value
    return result


def _public(record, role, for_ids=()):
    return {"id": record["id"], "text": _text(record), "original_text": record.get("content") or record.get("title") or "",
            "title": record.get("title") or "", "platform": record.get("platform"), "channel": record.get("channel"),
            "user_name": record.get("user_name"), "created_at": record.get("created_at"),
            "parent_id": record.get("parent_id"), "role": role, "for_ids": list(for_ids)}


def sample_candidate(candidate, units, vectors, vector_indices, records, as_of, limit=None,
                     max_context=20, preceding=3, context_seconds=120, context_index=None):
    """One packet per candidate, with a fixed number of target examples."""
    index = context_index or build_context_index(records, as_of)
    if index["as_of"] != as_of:
        raise ValueError("Context index cutoff differs from requested cutoff")
    by_id = index["by_id"]
    indices = [i for i in candidate.get("indices", []) if units[i]["voice_id"] in by_id]
    budget = min(10, limit if limit is not None else 10)
    if budget < 1:
        raise ValueError("Sample limit must be positive")
    selected = []
    if indices:
        ordered = sorted(indices, key=lambda i: (units[i]["day"], units[i]["unit_id"]))
        similarities = {i: float(vectors[int(vector_indices[i])] @ candidate["center"]) for i in indices}
        central = sorted(indices, key=lambda i: (-similarities[i], units[i]["unit_id"]))
        proposals = [central[0], ordered[0], ordered[-1], min(indices, key=lambda i: similarities[i])]
        for platform in sorted({units[i]["platform"] for i in indices}):
            proposals.append(next(i for i in central if units[i]["platform"] == platform))
        # Stable pseudo-random ordering adds coverage without process-random hash.
        import hashlib
        proposals += sorted(indices, key=lambda i: hashlib.sha256(units[i]["unit_id"].encode()).digest())
        proposals += central
        seen_text, seen_voices = set(), set()
        for i in proposals:
            record = by_id[units[i]["voice_id"]]
            if units[i]["text"] in seen_text or record["id"] in seen_voices:
                continue
            selected.append((i, record))
            seen_text.add(units[i]["text"])
            seen_voices.add(record["id"])
            if len(selected) >= budget:
                break
    targets = []
    for i, record in selected:
        row = _public(record, "target")
        row.update(unit_id=units[i]["unit_id"], text=units[i]["text"])
        targets.append(row)
    if not targets and candidate.get("anchor_id") in by_id:
        targets = [_public(by_id[candidate["anchor_id"]], "potential_anchor")]
    target_ids = {r["id"] for r in targets}
    contexts = {}

    def add(record, role, target_id):
        if record["id"] in target_ids:
            return
        existing = contexts.get(record["id"])
        if existing:
            if target_id not in existing["for_ids"]:
                existing["for_ids"].append(target_id)
        elif len(contexts) < max_context:
            contexts[record["id"]] = _public(record, role, [target_id])

    bursts = list(candidate["diagnostics"].get("discussion_bursts", []))
    if candidate["kind"] == "burst":
        bursts.append(dict(candidate["diagnostics"], anchor_id=candidate["anchor_id"], candidate_id=candidate["id"]))
    bursts.sort(key=lambda b: (-b.get("engagement_score", 0), b["candidate_id"]))
    response_budget = min(10, budget)
    response_ids = set()
    # Reserve bounded context for the activity that caused the nomination.
    # Target messages still retain their own text independently of this cap.
    for burst in bursts[:3]:
        anchor_id = burst["anchor_id"]
        if anchor_id not in by_id:
            continue
        add(by_id[anchor_id], "potential_discussion_anchor", anchor_id)
        for id in burst.get("competing_anchor_ids", [])[:2]:
            if id in by_id:
                add(by_id[id], "competing_anchor", anchor_id)
        possible = [id for id in burst.get("possible_response_ids", []) if id in by_id]
        # Even spacing covers the interval instead of taking only its start.
        positions = np.linspace(0, len(possible) - 1, min(response_budget, len(possible)), dtype=int) if possible else []
        for position in positions:
            id = possible[int(position)]
            if len(response_ids) >= response_budget:
                break
            add(by_id[id], "possible_following_response", anchor_id)
            response_ids.add(id)

    for target in targets:
        record = by_id[target["id"]]
        stamp, stream = _stamp(record), _stream(record)
        parent_id, visited = record.get("parent_id"), set()
        while parent_id in by_id and parent_id not in visited and len(visited) < 3:
            visited.add(parent_id)
            parent = by_id[parent_id]
            if _stamp(parent) > stamp or _stream(parent) != stream or stream is None:
                break
            add(parent, "explicit_parent", record["id"])
            parent_id = parent.get("parent_id")
        if stream in index["streams"]:
            rows, times = index["streams"][stream], index["times"][stream]
            pos = bisect_left(times, stamp)
            nearby = [r for r in rows[max(0, pos - preceding):pos]
                      if stamp - _stamp(r) <= timedelta(seconds=context_seconds)]
            for neighbor in nearby:
                add(neighbor, "possible_preceding_context", record["id"])
        for reply in index["replies"].get(record["id"], [])[:3]:
            if _stream(reply) == stream and stream is not None:
                add(reply, "explicit_response", record["id"])
    material = "\n".join(r["text"] for r in targets + list(contexts.values()))
    aliases = [a for a in candidate.get("available_aliases", []) if a["name"] in material or a["alias"] in material]
    return {"candidate_id": candidate["id"], "kind": candidate["kind"], "as_of": as_of,
            "targets": targets, "context": list(contexts.values()), "aliases": aliases,
            "stats": candidate["history"]["stats"], "diagnostics": _packet_diagnostics(candidate["diagnostics"]),
            "related_clusters": candidate.get("related_clusters", []),
            "membership_caveat": "Samples interpret algorithmic assignments; context is not automatically supporting evidence."}
