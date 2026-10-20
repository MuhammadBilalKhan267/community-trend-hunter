"""Orchestration; language judgments cannot overwrite computed facts."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .config import (DEFAULT_CHAT_PROVIDER, DEFAULT_EMBEDDING_MODEL, DEFAULT_JEV_MODEL,
                     DEFAULT_OLLAMA_CHAT_MODEL, DEFAULT_OLLAMA_URL)
from .data import prepare, make_units, read_releases
from .discovery import unique_units, cluster_vectors
from .local_models import LocalModels
from .openai_generation import OpenAIGeneration, GenerationOutputError, GenerationServiceError
from .temporal import activity_key, analyze_history, prepare_history
from .grounding import claim_reference_material, validate_draft_references
from .aliases import remember_review_aliases, select_packet_aliases
from .decisions import BudgetExceeded, DecisionError, JevDecisions
from .review import (review_candidate, source_proposals, final_draft, final_check,
                     extractive_fallback, review_final_duplicates)

STATUSES = {"new", "growing", "one-off", "fading", "not real"}


def _concept_key(concept, accept_ratio=.8):
    """Normal acceptance outranks every provisional fallback, regardless of reach."""
    return (concept["match_ratio"] < accept_ratio, *activity_key(concept))


def validate_output(result: dict, records: list[dict]) -> None:
    """Validate the public trend schema and cited source records."""
    from datetime import date

    def calendar_day(value, field):
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a YYYY-MM-DD string")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be a valid YYYY-MM-DD date") from exc
        if parsed.isoformat() != value:
            raise ValueError(f"{field} must use exact YYYY-MM-DD formatting")
        return parsed

    if not isinstance(result, dict) or set(result) != {"as_of", "trends"}:
        raise ValueError("Output must contain exactly as_of and trends")
    as_of = calendar_day(result["as_of"], "as_of")
    if not isinstance(result["trends"], list):
        raise ValueError("trends must be a list")
    by_id = {r["id"]: r for r in records}
    if len(result["trends"]) > 8:
        raise ValueError("Output exceeds eight trends")
    fields = {"title", "title_en", "category", "status", "first_seen", "voice_ids", "why"}
    for trend in result["trends"]:
        if not isinstance(trend, dict) or set(trend) != fields:
            raise ValueError("Each trend must contain exactly the required output fields")
        for key in ("title", "title_en", "category", "why"):
            if not isinstance(trend[key], str) or not trend[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        if not isinstance(trend["status"], str) or trend["status"] not in STATUSES:
            raise ValueError("Invalid trend status")
        first_seen = calendar_day(trend["first_seen"], "first_seen")
        if first_seen > as_of:
            raise ValueError("first_seen is beyond cutoff")
        identifiers = trend["voice_ids"]
        if (not isinstance(identifiers, list) or not identifiers
                or any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers)):
            raise ValueError("voice_ids must be a nonempty list of string IDs")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("voice_ids must be unique")
        evidence_days = []
        for identifier in identifiers:
            if identifier not in by_id:
                raise ValueError("Evidence is missing or beyond cutoff")
            source = by_id[identifier]
            source_day = calendar_day(source.get("day"), "Evidence day")
            if source_day > as_of:
                raise ValueError("Evidence is missing or beyond cutoff")
            if (source.get("bot_like") or source.get("trivial")
                    or ("family" in source and not source["family"])):
                raise ValueError("Evidence must be an eligible substantive or contextual voice")
            evidence_days.append(source_day)
        if first_seen > min(evidence_days):
            raise ValueError("Evidence precedes reported first observation")



def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _cluster_snapshot(vectors, texts, target_size, cache_path, enabled=True):
    import hashlib
    key = hashlib.sha256(vectors.tobytes() + json.dumps(
        [texts, target_size, "minibatch-kmeans-v2"], ensure_ascii=False).encode()).hexdigest()
    path = Path(cache_path).parent / "clusters" / (key + ".npz")
    if enabled and path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            labels, centers = stored["labels"], stored["centers"]
        if len(labels) == len(vectors) and centers.shape[1] == vectors.shape[1]:
            return labels, centers, True
    labels, centers = cluster_vectors(vectors, target_size)
    if enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, labels=labels, centers=centers)
    return labels, centers, False


def _evidence(record):
    return {k: record.get(k) for k in (
        "id", "parent_id", "platform", "channel", "user_name", "title", "content",
        "created_at", "day", "reaction", "image_count")}


def _members_for(candidate, concern, all_concerns, units, vectors, vector_indices):
    """Sample interpretation expands computational groups, never claims a census."""
    rejected = set(concern["unrelated_ids"])
    matched = set(concern["matched_ids"])
    if candidate["kind"] == "burst":
        return set(candidate["member_ids"]) & matched
    useful = [c for c in all_concerns if c["status"] in {"accepted", "provisional"}]
    if len(useful) <= 1:
        return set(candidate["member_ids"]) - rejected
    prototypes = []
    for c in useful:
        seeds = [i for i in candidate["indices"] if units[i]["voice_id"] in c["matched_ids"]]
        vector = np.mean([vectors[vector_indices[i]] for i in seeds], axis=0) if seeds else candidate["center"]
        prototypes.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
    selected = useful.index(concern)
    members = set(matched)
    for i in candidate["indices"]:
        if int(np.argmax([vectors[vector_indices[i]] @ p for p in prototypes])) == selected:
            members.add(units[i]["voice_id"])
    return members - rejected


def _record_unit_votes(concept, targets, concern):
    """Replace focal-unit judgments, then count each original source once."""
    target_by_id = {target["id"]: target for target in targets}
    ledger = concept.setdefault("unit_votes", {})
    for identifier, vote in concern["contributions"].items():
        target = target_by_id.get(identifier, {})
        unit_id = target.get("unit_id") or identifier
        ledger.setdefault(identifier, {})[unit_id] = {
            "match": vote["match"], "role": vote.get("role", "unclear"),
            "stance": concern.get("stances", {}).get(identifier, "unclear"),
            "text": target.get("text"),
        }
    _reduce_unit_votes(concept)


def _reduce_unit_votes(concept):
    """Aggregate a focal-unit ledger without counting a source twice."""
    ledger = concept.get("unit_votes", {})
    votes, stances = {}, {}
    for identifier, units in ledger.items():
        rows = [units[key] for key in sorted(units)]
        matching = [row for row in rows if row["match"] == "match"]
        uncertain = [row for row in rows if row["match"] == "unclear"]
        relevant = matching or uncertain or rows
        # Preserve dissent as counterevidence even when the same source also
        # contributes an experience in another matching clause.
        role = ("disagreement" if matching and any(row["role"] == "disagreement" for row in matching)
                else next((row["role"] for row in relevant if row["role"] != "unclear"), "unclear"))
        votes[identifier] = {"match": "match" if matching else "unclear" if uncertain else "other", "role": role}
        if matching:
            values = {row["stance"] for row in matching}
            if "mixed" in values or {"positive", "negative"} <= values:
                stances[identifier] = "mixed"
            else:
                stances[identifier] = next((value for value in ("positive", "negative", "neutral") if value in values), "unclear")
    concept["votes"], concept["stances"] = votes, stances


def _complete_link_groups(identifiers, confirmed_pairs):
    """An unreviewed or distinct edge cannot become a transitive merge."""
    confirmed = {frozenset(pair) for pair in confirmed_pairs if len(set(pair)) == 2}
    groups = []
    for identifier in identifiers:
        group = next((group for group in groups
                      if all(frozenset((identifier, other)) in confirmed for other in group)), None)
        if group is None:
            groups.append([identifier])
        else:
            group.append(identifier)
    return [group for group in groups if len(group) > 1]


def _merge_concepts(items, by_id, eligible, as_of, context_index, releases,
                    provisional_ratio=.5, history_context=None):
    """Union confirmed topics at source level, then recompute every measurement."""
    from copy import deepcopy

    merged = deepcopy(items[0])
    merged["merged_from"] = list(dict.fromkeys(
        identifier for item in items for identifier in item.get("merged_from", [item["id"]])))
    merged["member_ids"] = set().union(*(item["member_ids"] for item in items))
    merged["unit_votes"] = {}
    for item in items:
        for identifier, units in item.get("unit_votes", {}).items():
            ledger = merged["unit_votes"].setdefault(identifier, {})
            for unit_id, incoming in units.items():
                if unit_id not in ledger:
                    ledger[unit_id] = deepcopy(incoming)
                    continue
                current = ledger[unit_id]
                judgments = [current, incoming]
                matching = [row for row in judgments if row["match"] == "match"]
                uncertain = [row for row in judgments if row["match"] == "unclear"]
                chosen = matching or uncertain or judgments
                vote = deepcopy(chosen[0])
                vote["match"] = "match" if matching else "unclear" if uncertain else "other"
                vote["role"] = ("disagreement" if matching and any(row["role"] == "disagreement" for row in matching)
                                else next((row["role"] for row in chosen if row["role"] != "unclear"), "unclear"))
                stances = {row.get("stance", "unclear") for row in matching}
                vote["stance"] = ("mixed" if "mixed" in stances or {"positive", "negative"} <= stances
                                  else next((value for value in ("positive", "negative", "neutral")
                                             if value in stances), "unclear"))
                # Preserve competing judgments for the audit while the union
                # counts a source matching either confirmed branch once.
                vote["merged_judgments"] = [deepcopy(row) for judgment in judgments
                                             for row in judgment.get("merged_judgments", [
                                                 {key: value for key, value in judgment.items()
                                                  if key != "merged_judgments"}])]
                ledger[unit_id] = vote
    _reduce_unit_votes(merged)
    merged["member_ids"].update(identifier for identifier, vote in merged["votes"].items()
                                if vote["match"] == "match")
    merged["member_ids"].difference_update(identifier for identifier, vote in merged["votes"].items()
                                           if vote["match"] == "other")
    for field in ("candidate_ids", "matched_response_ids"):
        merged[field] = list(dict.fromkeys(value for item in items for value in item.get(field, [])))
    merged["contexts"] = {}
    for item in items:
        for identifier, incoming in item.get("contexts", {}).items():
            current = merged["contexts"].get(identifier)
            if current is None:
                merged["contexts"][identifier] = deepcopy(incoming)
            else:
                current["for_ids"] = list(dict.fromkeys(current.get("for_ids", []) + incoming.get("for_ids", [])))
                if incoming.get("role") == "explicit_parent":
                    current["role"] = "explicit_parent"
                if incoming.get("alias_provenance"):
                    current["alias_provenance"] = True
    aliases, seen = [], set()
    for item in items:
        for assertion in item.get("aliases", []):
            key = json.dumps(_plain(assertion), sort_keys=True, ensure_ascii=False)
            if key not in seen:
                aliases.append(deepcopy(assertion))
                seen.add(key)
    merged["aliases"] = aliases
    merged["reviewed_release_associations"] = _canonical_release_associations(
        [event for item in items for event in item.get("reviewed_release_associations", [])], releases, as_of)
    merged["engagement"] = {
        "sampled_attribution_supported": any(item.get("engagement", {}).get("sampled_attribution_supported")
                                              for item in items),
        # These approximate response windows may overlap. Never add their sizes.
        "possible_response_count": max((item.get("engagement", {}).get("possible_response_count", 0)
                                         for item in items), default=0),
    }
    return _refresh(merged, by_id, eligible, as_of, context_index, provisional_ratio, history_context)


def _refresh(concept, by_id, eligible, as_of, context_index, provisional_ratio=.5, history_context=None):
    members = [by_id[i] for i in sorted(concept["member_ids"]) if i in by_id and not by_id[i].get("reaction")]
    history = (analyze_history(members, eligible, as_of, context=history_context)
               if history_context is not None else analyze_history(members, eligible, as_of))
    votes = concept["votes"]
    matching = [i for i, v in votes.items() if v["match"] == "match"]
    concept["match_ratio"] = len(matching) / max(1, len(votes))
    concept["matched_ids"] = matching
    concept["opposing_ids"] = [i for i in matching if votes[i]["role"] == "disagreement"]
    explicit = {r["id"]: r for i in concept["member_ids"] for r in context_index["replies"].get(i, [])
                if r["id"] not in concept["member_ids"]}
    concept["engagement"]["explicit_reply_count"] = len(explicit)
    concept["engagement"]["explicit_reply_author_proxies"] = len({
        (r["platform"], r.get("user_name")) for r in explicit.values()})
    if matching:
        concept["first_seen"] = min(by_id[i]["day"] for i in matching)
    else:
        concept["first_seen"] = history["first_seen"]
    concept["history"] = history
    concept["score"] = history["score"]
    concept["ranking_engagement"] = len(explicit)
    if concept["engagement"].get("sampled_attribution_supported"):
        concept["ranking_engagement"] = max(len(explicit), concept["engagement"].get("possible_response_count", 0))
        # A discussion can be important with one substantive anchor. This is
        # observed discussion activity, not a count of independent experiences.
        if not history["status_supported"] and concept["first_seen"]:
            from datetime import date
            age = (date.fromisoformat(as_of) - date.fromisoformat(concept["first_seen"])).days
            if age <= 7:
                history["status"] = "new"
                history["status_supported"] = True
                history["why_en"] = "A recently observed substantive post attracted a sampled, attributable discussion."
                history["stats"]["status_basis_kind"] = "attributed_recent_discussion"
    history["stats"]["sampled_topic_matches"] = len(matching)
    history["stats"]["reviewed_examples"] = len(votes)
    history["stats"]["sample_match_ratio"] = concept["match_ratio"]
    history["stats"]["sampled_contributions"] = dict(__import__("collections").Counter(v["role"] for v in votes.values()))
    history["stats"]["sampled_stances"] = dict(__import__("collections").Counter(concept["stances"].values()))
    concept["eligible"] = bool(history["status_supported"] and matching and concept["match_ratio"] >= provisional_ratio)
    return concept


def _canonical_release_associations(rows, releases, as_of=None):
    """Retain reviewed calendar identities, without inferring causal effects."""
    from datetime import date

    cutoff = date.fromisoformat(as_of) if as_of is not None else None
    calendar = {}
    for release in releases:
        try:
            day = date.fromisoformat(release.get("date", ""))
        except (TypeError, ValueError):
            continue
        if day.isoformat() != release["date"] or (cutoff is not None and day > cutoff):
            continue
        key = (release["date"], release.get("version", ""), release.get("notes", ""))
        calendar[key] = release
    selected = {}
    for row in rows if isinstance(rows, (list, tuple)) else []:
        if not isinstance(row, dict):
            continue
        key = (row.get("date"), row.get("version", ""), row.get("notes", ""))
        if key in calendar:
            selected[key] = {**calendar[key], "association": "content_association_only",
                             "relationship": "Reviewed source association; chronology does not establish causation."}
    return [selected[key] for key in sorted(selected)]


def _selected_source_ids(concept, by_id, limit=10):
    """Keep reviewed support, its defining anchor and diverse original voices."""
    from datetime import date

    ordered = sorted({identifier for identifier in concept["matched_ids"] if identifier in by_id},
                     key=lambda identifier: (by_id[identifier]["day"], identifier))
    if len(ordered) <= limit:
        return ordered
    selected = []
    counterexample = next((identifier for identifier in ordered if identifier in concept["opposing_ids"]), None)
    for identifier in (ordered[0], ordered[-1], concept.get("source_anchor_id"), counterexample):
        if identifier in ordered and identifier not in selected and len(selected) < limit:
            selected.append(identifier)
    days = {identifier: date.fromisoformat(by_id[identifier]["day"]).toordinal() for identifier in ordered}
    while len(selected) < limit:
        platforms = {by_id[identifier].get("platform") for identifier in selected}
        authors = {(by_id[identifier].get("platform"), by_id[identifier].get("user_name")) for identifier in selected}

        def diversity(identifier):
            record = by_id[identifier]
            gap = min(abs(days[identifier] - days[chosen]) for chosen in selected)
            return (-(record.get("platform") not in platforms), -gap,
                    -((record.get("platform"), record.get("user_name")) not in authors), record["day"], identifier)

        selected.append(min((identifier for identifier in ordered if identifier not in selected), key=diversity))
    return sorted(selected, key=lambda identifier: (by_id[identifier]["day"], identifier))


def _accepted_for_review(concepts, by_id, as_of):
    """Represent known topics with a few current reviewed originals, not only seeds."""
    from datetime import date, datetime
    from .data import GAME_TIMEZONE

    cutoff = date.fromisoformat(as_of)
    accepted = []
    for concept in concepts:
        records, focal_texts = {}, {}
        matched = set(concept.get("matched_ids", []))
        for identifier, units in concept.get("unit_votes", {}).items():
            texts = list(dict.fromkeys(unit["text"] for _, unit in sorted(units.items())
                                      if unit.get("match") == "match" and isinstance(unit.get("text"), str)
                                      and unit["text"].strip()))
            record = by_id.get(identifier)
            if identifier not in matched or not texts or not isinstance(record, dict) or record.get("id") != identifier:
                continue
            try:
                stamp = datetime.fromisoformat(record["created_at"])
            except (KeyError, ValueError, TypeError):
                continue
            if stamp.tzinfo is None or stamp.astimezone(GAME_TIMEZONE).date() > cutoff:
                continue
            records[identifier] = {**record, "day": stamp.astimezone(GAME_TIMEZONE).date().isoformat()}
            focal_texts[identifier] = texts
        ids = _selected_source_ids({**concept, "matched_ids": list(records),
                                    "opposing_ids": concept.get("opposing_ids", [])}, records, limit=3)
        examples = [{**_evidence(records[identifier]), "reviewed_focal_texts": focal_texts[identifier]} for identifier in ids]
        accepted.append({**concept, "reviewed_examples": examples})
    return accepted


def _state_for(concept, by_id, releases, as_of=None):
    ids = _selected_source_ids(concept, by_id)
    sources = []
    for identifier in ids:
        source = _evidence(by_id[identifier])
        reviewed_units = concept.get("unit_votes", {}).get(identifier, {})
        source["reviewed_focal_texts"] = list(dict.fromkeys(
            unit["text"] for _, unit in sorted(reviewed_units.items())
            if unit["match"] == "match" and isinstance(unit.get("text"), str) and unit["text"].strip()))
        sources.append(source)
    cutoff = as_of if as_of is not None else concept["history"]["stats"].get("as_of")
    if "reviewed_release_associations" in concept:
        related = _canonical_release_associations(concept["reviewed_release_associations"], releases, cutoff)
    else:
        # Compatibility for older stored concerns that predate release review.
        material = "\n".join((by_id[i].get("content") or "") + " " + (by_id[i].get("title") or "") for i in ids)
        mentioned = [release for release in releases if release.get("version") not in {None, "", "-"}
                     and release["version"] in material]
        related = _canonical_release_associations(mentioned, releases, cutoff)
        for association in related:
            association["relationship"] = "Source mentions version; chronology does not establish causation."
    source_ids = set(ids)
    matched_responses = set(concept.get("matched_response_ids", []))
    ordinary = {row["id"]: dict(row) for row in concept.get("contexts", {}).values()
                if row.get("id") not in source_ids and (
                    row.get("id") in matched_responses or (
                        row.get("role") in {"explicit_parent", "possible_preceding_context"}
                        and source_ids.intersection(row.get("for_ids", []))))}
    protected = {identifier for identifier, row in ordinary.items()
                 if row.get("role") == "explicit_parent" or identifier in matched_responses}
    proof_ids, aliases = [], []
    for assertion in concept.get("aliases", []):
        refs = assertion.get("evidence_ids") or [assertion.get("source_id")]
        if not refs or any(identifier not in by_id for identifier in refs):
            continue
        needed = set(refs) - source_ids
        if len(set(proof_ids) | needed | protected) > 20:
            continue
        proof_ids.extend(identifier for identifier in refs if identifier not in source_ids and identifier not in proof_ids)
        aliases.append({key: assertion[key] for key in (
            "name", "referent", "alias", "source_id", "evidence_ids", "scope", "confidence", "caveat")
            if key in assertion})
        aliases[-1]["evidence_ids"] = list(refs)
    context = []
    for identifier in proof_ids:
        row = {**_evidence(by_id[identifier]), **ordinary.get(identifier, {})}
        row.setdefault("role", "alias_provenance")
        row["alias_provenance"] = True
        context.append(row)
    remaining = [row for identifier, row in ordinary.items() if identifier not in proof_ids]
    remaining.sort(key=lambda row: row["id"] not in protected)
    context.extend(remaining)
    measurements = _compact_stats(concept["history"]["stats"])
    # The writer describes approved findings. Temporal analysis and its full
    # audit retain these detailed arrays; aggregate support remains here.
    measurements.pop("daily_activity", None)
    measurements.pop("platform_activity", None)
    return {
        "as_of": cutoff,
        "concern": {k: concept.get(k) for k in (
            "id", "definition", "scope", "uncertainty", "match_ratio", "proposal_kind",
            "source_anchor_id", "source_anchor_index", "focal_text")},
        "sources": sources,
        "counterevidence_ids": concept["opposing_ids"],
        "measurements": measurements,
        "measurement_meaning": "Topic volume is algorithmic cluster assignment interpreted through samples; engagement is separate.",
        "status": concept["history"]["status"], "status_basis": _public_status_basis(concept),
        "first_seen": concept["first_seen"], "engagement": concept["engagement"],
        "past_release_calendar": related,
        "source_release_associations": related,
        "aliases": aliases, "context": context[:20],
    }


def _public_status_basis(concept):
    """Explain computed status without exposing diagnostic ratios as product copy."""
    history = concept["history"]
    status = history["status"]
    descriptions = {
        "new": "The topic first appeared recently and recurred on multiple observed days.",
        "growing": "Discussion increased across the observed comparison periods and continued on several recent days.",
        "fading": "Discussion declined across the observed comparison periods; this does not establish that the issue was resolved.",
        "one-off": "Discussion concentrated in a brief peak, with little activity during the observed follow-up.",
        "not real": "Repeated near-identical posts dominate a short burst, so independent support remains uncertain.",
    }
    description = descriptions.get(status, "The observed activity does not yet establish a changing trend.")
    if status == "new" and history.get("stats", {}).get("status_basis_kind") == "attributed_recent_discussion":
        description = "A recent substantive post attracted a sampled, attributable discussion."
    boundary = history.get("stats", {}).get("boundary", {})
    if boundary.get("excluded_day"):
        description += " A possibly incomplete collection day is excluded from the comparison."
    if boundary.get("unobserved_days_after_end"):
        description += " Later missing observations are not evidence of silence."
    return description


def _compact_stats(stats):
    keys = ("analysis_end", "raw_count", "provisional_author_count", "provisional_author_day_support",
            "platform_count", "thread_count", "active_days", "first_seen", "last_seen",
            "recent_window", "prior_window", "normalized_growth_ratio", "growth_ratio",
            "smoothed_daily_support_ratio", "recent_normalized_smoothed_share", "prior_normalized_smoothed_share",
            "comparison_platforms", "template_capped_support", "support_policy", "ranking_policy",
            "ranking_support", "ranking_recent_support", "ranking_prior_support",
            "ranking_active_days", "ranking_platform_count",
            "platform_history", "platform_history_caveat",
            "template_concentration", "burst", "boundary",
            "sampled_topic_matches", "reviewed_examples", "sample_match_ratio", "sampled_stances")
    result = {key: stats[key] for key in keys if key in stats}
    result["daily_activity"] = stats.get("daily_activity", [
        [d["day"], d["raw_count"], d.get("provisional_author_day_support", 0)]
        for d in stats.get("daily", [])])
    result["platform_activity"] = stats.get("platform_activity", stats.get("platform_windows", {}))
    return result


def _bounded_context(packet, byte_limit=18000):
    from copy import deepcopy
    packet = deepcopy(packet)
    omitted = 0
    while len(json.dumps(packet, ensure_ascii=False).encode("utf-8")) > byte_limit and packet.get("context"):
        packet["context"].pop()
        omitted += 1
    if omitted:
        packet["omitted_context_count"] = omitted
    return packet


def _bounded_packet(packet):
    packet = {**packet, "stats": _compact_stats(packet.get("stats", {}))}
    return _bounded_context(packet)


def run(input_path: str, as_of: str, *, releases_path=None,
        cache_path=".cache/models.sqlite3", base_url=DEFAULT_OLLAMA_URL,
        embedding_model=DEFAULT_EMBEDDING_MODEL, chat_model=None, chat_provider=DEFAULT_CHAT_PROVIDER,
        max_reviews=None, max_audits=None, max_units=0, batch_size=64,
        cluster_size=80, decision_cache_path=".cache/decisions.sqlite3",
        jev_model=DEFAULT_JEV_MODEL, jev_token_target=20_000_000, jev_token_max=50_000_000,
        review_batch=10, exploration_ratio=.5, max_refinements=1, draft_retries=1,
        accept_ratio=.8, provisional_ratio=.5, chat_context=16384, chat_batch_size=64,
        embedding_num_gpu=None, embedding_residency_inputs=2048, cluster_cache=True, prepare_only=False,
        neighbor_reviews=16):
    from .candidates import (build_candidate_pool, build_context_index, choose_batch,
                             choose_neighbors, merge_related_bursts, sample_candidate)
    from .local_models import LocalGenerationError, LocalModelError

    def progress(message):
        print(message, file=sys.stderr, flush=True)

    if chat_provider not in {"ollama", "openai"}:
        raise ValueError("chat_provider must be ollama or openai")
    if chat_provider == "ollama" and chat_model is None:
        chat_model = DEFAULT_OLLAMA_CHAT_MODEL
    if chat_provider == "openai" and not prepare_only and not chat_model:
        raise ValueError("Select an explicit chat_model for OpenAI generation")
    if max_audits is not None:
        if max_reviews is not None and max_reviews != max_audits:
            raise ValueError("Use only one review-limit setting")
        max_reviews = max_audits
    if max_reviews is not None and max_reviews < 1:
        raise ValueError("max_reviews must be positive or unset")
    if max_refinements < 0 or draft_retries < 0 or review_batch < 1 or neighbor_reviews < 0:
        raise ValueError("Invalid runtime review settings")
    if embedding_residency_inputs < 0:
        raise ValueError("embedding_residency_inputs must be nonnegative")
    records, eligible, data_stats = prepare(input_path, as_of)
    by_id = {r["id"]: r for r in records}
    units = make_units(eligible)
    full_count = len(units)
    if max_units:
        units = [units[int(i)] for i in np.linspace(0, len(units)-1, min(max_units, len(units)), dtype=int)] if units else []
    result = {"as_of": as_of, "trends": []}
    diagnostics = {
        "as_of": as_of, "data": data_stats, "full_units": full_count, "processed_units": len(units),
        "sampled_development_run": bool(max_units and max_units < full_count),
        "method": "cached local embeddings, statistical candidates, source-anchored Jev review and "
                  + ("local" if chat_provider == "ollama" else "OpenAI") + " final drafting",
        "model_config": {"embedding_model": embedding_model, "chat_provider": chat_provider, "chat_model": chat_model,
                         "embedding_cache_only": chat_provider == "openai",
                         "jev_model": jev_model, "chat_context": chat_context,
                         "chat_batch_size": chat_batch_size, "embedding_num_gpu": embedding_num_gpu},
        "audits": [], "selected": [], "watchlist": [], "stop_reason": None,
        "review_policy": {"max_reviews": max_reviews, "initial_batch": review_batch,
                          "exploration_ratio": exploration_ratio, "accept_ratio": accept_ratio,
                          "provisional_ratio": provisional_ratio, "neighbor_reviews": neighbor_reviews,
                          "neighbor_pool_size": 8},
        "neighbor_reconciliation": {"budget": neighbor_reviews, "reviews": 0,
                                    "per_concept_limit": 2,
                                    "status": "pending" if neighbor_reviews else "disabled"},
        "final_duplicate_reviews": [], "final_duplicate_review_calls": 0,
    }
    if not units:
        validate_output(result, records)
        return result, diagnostics
    texts, vector_indices = unique_units(units)
    releases = read_releases(releases_path, as_of)
    context_index = build_context_index(records, as_of)
    history_context = prepare_history(eligible, as_of)
    progress(f"{as_of}: {len(records)} messages, {len(units)} clauses, {len(texts)} distinct embedding inputs.")
    options = {} if embedding_num_gpu is None else {"num_gpu": embedding_num_gpu}
    with ExitStack() as generation_stack:
        try:
            models = generation_stack.enter_context(LocalModels(
                base_url=base_url, embedding_model=embedding_model,
                chat_model=chat_model if chat_provider == "ollama" else DEFAULT_OLLAMA_CHAT_MODEL,
                cache_path=cache_path, batch_size=batch_size, embedding_options=options,
                chat_options={"num_ctx": chat_context, "num_batch": chat_batch_size},
                embedding_cache_only=chat_provider == "openai", keep_alive="30m", timeout=600))
        except LocalModelError as exc:
            diagnostics["stop_reason"] = "embedding_cache_error" if chat_provider == "openai" else "local_model_error"
            diagnostics[diagnostics["stop_reason"]] = str(exc)
            validate_output(result, records)
            return result, _plain(diagnostics)
        def unload(model):
            try:
                models.unload(model)
            except LocalModelError as exc:
                diagnostics.setdefault("local_cleanup_errors", []).append({"model": model, "message": str(exc)})

        writer = models
        vector_parts = []
        window = embedding_residency_inputs or len(texts)
        try:
            for start in range(0, len(texts), window):
                before = models.stats.get("embedding_calls", 0)
                try:
                    vector_parts.append(models.embed(texts[start:start + window]))
                finally:
                    if models.stats.get("embedding_calls", 0) > before:
                        unload(embedding_model)
        except LocalModelError as exc:
            diagnostics["stop_reason"] = "embedding_cache_error" if chat_provider == "openai" else "local_model_error"
            diagnostics[diagnostics["stop_reason"]] = str(exc)
            diagnostics["local_usage"] = dict(models.stats)
            validate_output(result, records)
            return result, _plain(diagnostics)
        vectors = np.vstack(vector_parts)
        if getattr(models, "embedding_profile", None) is not None:
            diagnostics["embedding_identity"] = models.embedding_profile
        if chat_provider == "openai" and not prepare_only:
            try:
                writer = generation_stack.enter_context(OpenAIGeneration(model=chat_model, cache_path=cache_path))
            except GenerationServiceError as exc:
                diagnostics["stop_reason"] = "generation_service_error"
                diagnostics["generation_error"] = str(exc)
                diagnostics["local_usage"] = dict(models.stats)
                return result, _plain(diagnostics)
        labels, centers, cluster_hit = _cluster_snapshot(vectors, texts, cluster_size, cache_path, cluster_cache)
        progress("Building candidate statistics and discussion nominations.")
        pool = merge_related_bursts(build_candidate_pool(
            units, vectors, vector_indices, labels, centers, records, eligible, as_of))
        diagnostics["candidate_count"] = len(pool)
        diagnostics["cluster_cache_hit"] = cluster_hit
        diagnostics["candidate_summary"] = [
            {**{k: _plain(c[k]) for k in ("id", "kind", "score", "ranking_engagement", "diagnostics")},
             "activity_rank": _plain(activity_key(c)),
             "ranking_policy": c["history"]["stats"]["ranking_policy"],
             "temporal_status": c["history"]["status"],
             "status_supported": c["history"]["status_supported"]} for c in pool]
        if prepare_only:
            diagnostics["candidate_packets"] = [_bounded_packet(sample_candidate(c, units, vectors, vector_indices,
                records, as_of, context_index=context_index)) for c in choose_batch(pool, set(), review_batch, exploration_ratio)]
            diagnostics["local_usage"] = dict(models.stats)
            return result, _plain(diagnostics)
        concepts, reviewed, refinements = {}, set(), defaultdict(int)
        approved_aliases = []
        refinement_queue = []
        review_count = 0
        neighbor_counts = defaultdict(int)
        serial = 0
        final_versions = {}
        final_duplicate_versions = {}
        with JevDecisions(cache_path=decision_cache_path, model=jev_model,
                          target_tokens=jev_token_target, max_tokens=jev_token_max) as decisions:
            def store(candidate, concern):
                nonlocal serial
                members = _members_for(candidate, concern, current_review["concerns"], units, vectors, vector_indices)
                key = concern.get("duplicate_of")
                if key not in concepts:
                    key = None
                if key is None:
                    serial += 1
                    key = f"concern:{serial}"
                    concepts[key] = {**concern, "id": key, "member_ids": set(), "votes": {}, "stances": {},
                                     "candidate_ids": [], "engagement": {}, "contexts": {}}
                item = concepts[key]
                if "reviewed_release_associations" in concern:
                    previous = item.get("reviewed_release_associations", [])
                    incoming = concern["reviewed_release_associations"]
                    item["reviewed_release_associations"] = _canonical_release_associations(
                        (previous if isinstance(previous, list) else [])
                        + (incoming if isinstance(incoming, list) else []), releases, as_of)
                item["uncertainty"] = (
                    "Topic interpretation and support are based on reviewed samples; remaining cluster assignments are approximate.")
                if item.get("proposal_kind") == "source_anchor" and "focal_text" not in item:
                    anchor = next((row for row in packet["targets"]
                                   if row["id"] == item.get("source_anchor_id")), None)
                    if anchor is not None:
                        item["focal_text"] = anchor["text"]
                item["member_ids"].update(members)
                _record_unit_votes(item, packet["targets"], concern)
                item["member_ids"].update(i for i, vote in item["votes"].items() if vote["match"] == "match")
                item["member_ids"].difference_update(
                    i for i, vote in item["votes"].items() if vote["match"] == "other")
                item["matched_response_ids"] = list(dict.fromkeys(
                    item.get("matched_response_ids", []) + concern.get("matched_response_ids", [])))
                item["candidate_ids"].append(candidate["id"])
                for row in packet.get("context", []):
                    item["contexts"][row["id"]] = row
                matched = set(concern["matched_ids"])
                alias_packet = {**packet,
                    "targets": [row for row in packet["targets"] if row["id"] in matched],
                    "context": [row for row in packet.get("context", [])
                                if not row.get("for_ids") or matched.intersection(row["for_ids"])]}
                stored_aliases = item.setdefault("aliases", [])
                keys = {(a.get("name", a.get("referent")), a.get("alias"),
                         tuple(a.get("evidence_ids") or [a.get("source_id")])) for a in stored_aliases}
                for assertion in select_packet_aliases(approved_aliases, alias_packet):
                    key = (assertion.get("name", assertion.get("referent")), assertion.get("alias"),
                           tuple(assertion.get("evidence_ids") or [assertion.get("source_id")]))
                    if key not in keys:
                        stored_aliases.append(assertion)
                        keys.add(key)
                item["engagement"]["sampled_attribution_supported"] = (
                    item["engagement"].get("sampled_attribution_supported", False)
                    or concern.get("engagement_supported", False))
                possible = candidate.get("diagnostics", {}).get("possible_response_count", 0)
                for burst in candidate.get("diagnostics", {}).get("discussion_bursts", []):
                    possible += burst.get("possible_response_count", 0)
                item["engagement"]["possible_response_count"] = max(
                    item["engagement"].get("possible_response_count", 0), possible)
                return _refresh(item, by_id, eligible, as_of, context_index,
                                provisional_ratio, history_context)

            def finalize(allow_provisional=False):
                import hashlib
                from copy import deepcopy

                selected = []

                def draft_inputs(item):
                    state = _state_for(item, by_id, releases, as_of)
                    quality = "accepted" if item["match_ratio"] >= accept_ratio else "provisional"
                    state["sample_quality"] = quality
                    if quality == "provisional":
                        state["status_basis"] += " This is a provisional topic interpretation; the sampled evidence remains mixed."
                    state = _bounded_context(state)
                    known = {row["id"] for row in state["sources"] + state.get("context", [])}
                    state["aliases"] = [assertion for assertion in state["aliases"]
                                        if set(assertion["evidence_ids"]) <= known]
                    approved = {k: item.get(k) for k in
                                ("definition", "scope", "uncertainty", "match_ratio", "proposal_kind",
                                 "source_anchor_id", "source_anchor_index", "focal_text")}
                    version = hashlib.sha256(json.dumps(_plain({
                        "votes": item["votes"], "members": sorted(item["member_ids"]),
                        "stances": item["stances"], "approved": approved, "state": state,
                    }), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    return state, approved, (item["id"], version)

                ranked = [item for item in sorted(concepts.values(), key=lambda c: _concept_key(c, accept_ratio))
                          if item["eligible"] and (item["match_ratio"] >= accept_ratio or allow_provisional)]

                def reconcile():
                    """Review only the current final set, never all discovered clusters."""
                    while True:
                        current = [item for item in sorted(concepts.values(), key=lambda c: _concept_key(c, accept_ratio))
                                   if item["eligible"] and (item["match_ratio"] >= accept_ratio or allow_provisional)]
                        window = []
                        snapshots = []
                        for item in current:
                            state, approved, key = draft_inputs(item)
                            if key in final_versions and final_versions[key] is None:
                                continue
                            window.append(item)
                            snapshots.append({"id": item["id"], "sources": state["sources"],
                                              "context": state.get("context", []),
                                              "aliases": state.get("aliases", []),
                                              "approved_candidate": approved})
                            if len(window) == 8:
                                break
                        if len(snapshots) < 2:
                            return current, window
                        fingerprint = hashlib.sha256(json.dumps(_plain(snapshots), sort_keys=True,
                            ensure_ascii=False).encode()).hexdigest()
                        if fingerprint not in final_duplicate_versions:
                            diagnostics["final_duplicate_review_calls"] += 1
                            audit = {
                                "snapshot": fingerprint, "concern_ids": [item["id"] for item in window],
                                "evidence_ids": {row["id"]: [source["id"] for source in row["sources"]]
                                                 for row in snapshots}, "review": None, "merges": []}
                            diagnostics["final_duplicate_reviews"].append(audit)
                            try:
                                check = review_final_duplicates(decisions, snapshots)
                            except (BudgetExceeded, DecisionError) as error:
                                audit["error"] = str(error)
                                raise
                            final_duplicate_versions[fingerprint] = check
                            audit["review"] = deepcopy(check)
                        else:
                            check = final_duplicate_versions[fingerprint]
                        groups = _complete_link_groups([item["id"] for item in window],
                                                       check.get("confirmed_pairs", []))
                        if not groups:
                            return current, window
                        audit = next(row for row in diagnostics["final_duplicate_reviews"]
                                     if row["snapshot"] == fingerprint)
                        for group in groups:
                            combined = _merge_concepts([concepts[key] for key in group], by_id, eligible,
                                as_of, context_index, releases, provisional_ratio, history_context)
                            concepts[group[0]] = combined
                            for key in group[1:]:
                                del concepts[key]
                            audit["merges"].append({"retained_id": group[0], "merged_ids": group,
                                "member_ids": sorted(combined["member_ids"]),
                                "matched_ids": list(combined["matched_ids"]),
                                "unit_votes": deepcopy(combined["unit_votes"]),
                                "history": deepcopy(combined["history"])})
                        # A source union may change status or ranking. Review a
                        # changed final set before writing or selecting a backfill.

                try:
                    while True:
                        ranked, window = reconcile()
                        pending = [item for item in window if draft_inputs(item)[2] not in final_versions]
                        if not pending:
                            break
                        # Reconcile again only if this version exhausts its
                        # wording attempts and exposes a replacement topic.
                        for item in pending:
                            state, approved, cache_key = draft_inputs(item)
                            failed = None
                            previous_draft = None
                            draft_attempts = []
                            for attempt in range(draft_retries + 1):
                                draft_state = deepcopy(state)
                                if failed:
                                    draft_state["previous_draft"] = deepcopy(previous_draft)
                                    draft_state["previous_draft_check"] = deepcopy(failed)
                                try:
                                    draft = final_draft(writer, approved, draft_state)
                                except (LocalGenerationError, GenerationOutputError) as error:
                                    final_versions[cache_key] = None
                                    diagnostics.setdefault("generation_failures", []).append({
                                        "concern_id": item["id"], "version": cache_key[1], "reason": str(error),
                                        "evidence_ids": [row["id"] for row in state["sources"]],
                                        "attempts": deepcopy(draft_attempts)})
                                    break
                                bad_refs = validate_draft_references(draft, state)
                                check = final_check(decisions, draft, state) if not bad_refs else {
                                    "passed": False, "failed_claims": bad_refs}
                                draft_attempts.append({"attempt": attempt + 1, "draft": deepcopy(draft),
                                    "check": deepcopy(check), "evidence_ids": [row["id"] for row in state["sources"]]})
                                if not check["passed"] and not bad_refs:
                                    fallback = extractive_fallback(draft, state, check)
                                    if fallback is not None:
                                        replacement, replacement_check = fallback
                                        if not validate_draft_references(replacement, state):
                                            draft, check = replacement, replacement_check
                                            draft_attempts[-1]["fallback_check"] = deepcopy(check)
                                if check["passed"]:
                                    ids = [r["id"] for r in state["sources"]]
                                    trend = {k: draft[k] for k in ("title", "title_en", "category", "why")}
                                    trend.update(status=item["history"]["status"], first_seen=item["first_seen"], voice_ids=ids)
                                    entry = {"concern_id": item["id"], "trend": trend, "history": item["history"],
                                             "sample_review": {"match_ratio": item["match_ratio"], "votes": item["votes"],
                                                               "quality": state["sample_quality"], "normal_acceptance_threshold": accept_ratio},
                                             "sample_quality": state["sample_quality"],
                                             "activity_rank": list(activity_key(item)),
                                             "ranking_engagement": item["ranking_engagement"],
                                             "engagement": item["engagement"], "grounding": check,
                                             "release_associations": state["source_release_associations"],
                                             "evidence": state["sources"], "draft_claims": draft["claims"],
                                             "proposed_action": draft["proposed_action"],
                                             "generation_mode": check.get("mode", "generated"),
                                             "category_mode": check.get("category_mode", "jev_checked"),
                                             "provenance": check.get("provenance"),
                                             "components": check.get("components"),
                                             "rejected_generation": check.get("rejected_generation"),
                                             "draft_attempts": deepcopy(draft_attempts),
                                             "aliases": state.get("aliases", []), "context": state.get("context", [])}
                                    final_versions[cache_key] = deepcopy(entry)
                                    break
                                previous_draft = deepcopy(draft)
                                failed = {"failed_claims": deepcopy(check["failed_claims"]),
                                          "checks": deepcopy(check.get("checks", {}))}
                                if "failure_requirements" in check:
                                    failed["failure_requirements"] = deepcopy(check["failure_requirements"])
                                material = []
                                for index, claim in enumerate(draft.get("claims", [])):
                                    prefix = f"claim_{index}"
                                    if isinstance(claim, dict) and any(
                                        label == prefix or label.startswith(prefix + ":")
                                        for label in failed["failed_claims"]
                                    ) and all(isinstance(claim.get(field), list) and all(isinstance(ref, str) for ref in claim[field])
                                              for field in ("evidence_ids", "metric_keys")):
                                        references = claim_reference_material(claim, state)
                                        # Keep reference bindings without repeating long originals.
                                        references["cited_source_ids"] = [row["id"] for row in references.pop("cited_original_sources")]
                                        material.append({"claim_index": index, "statement": claim.get("text"), **references})
                                if material:
                                    failed["failed_claim_material"] = material
                            else:
                                final_versions[cache_key] = None
                                diagnostics.setdefault("draft_failures", []).append({
                                    "concern_id": item["id"], "version": cache_key[1], "check": failed,
                                    "evidence_ids": [row["id"] for row in state["sources"]],
                                    "attempts": deepcopy(draft_attempts)})
                finally:
                    # A resource or provider failure must not discard earlier
                    # grounded entries whose current evidence is unchanged.
                    selected = []
                    ranked = [item for item in sorted(concepts.values(), key=lambda c: _concept_key(c, accept_ratio))
                              if item["eligible"] and (item["match_ratio"] >= accept_ratio or allow_provisional)]
                    chosen = set()
                    for item in ranked:
                        if len(selected) >= 8:
                            break
                        if item["id"] in chosen:
                            continue
                        _, _, cache_key = draft_inputs(item)
                        entry = final_versions.get(cache_key)
                        if entry is not None:
                            selected.append(deepcopy(entry))
                            chosen.add(item["id"])
                    rank = {item["id"]: index for index, item in enumerate(ranked)}
                    selected.sort(key=lambda entry: rank[entry["concern_id"]])
                    # Rebuild the current ranking rather than reserving slots
                    # for concerns accepted by an earlier review batch.
                    diagnostics["selected"] = selected
                    result["trends"] = [entry["trend"] for entry in selected]

            try:
                while True:
                    if max_reviews is not None and review_count >= max_reviews:
                        diagnostics["stop_reason"] = "runtime_review_limit"
                        break
                    if decisions.usage["target_exceeded"]:
                        diagnostics["stop_reason"] = "token_target"
                        break
                    if refinement_queue:
                        batch = [refinement_queue.pop(0)]
                        stage = "refinement"
                    else:
                        stage = "discovery"
                        interpreted = [item for item in sorted(concepts.values(), key=activity_key)
                                       if item["eligible"] and item["match_ratio"] >= accept_ratio]
                        if neighbor_reviews and len(interpreted) >= 8:
                            remaining_neighbors = neighbor_reviews - diagnostics["neighbor_reconciliation"]["reviews"]
                            neighbor_concepts = [item for item in interpreted[:8] if neighbor_counts[item["id"]] < 2]
                            batch = (choose_neighbors(pool, reviewed, neighbor_concepts, limit=1, neighbor_count=8)
                                     if remaining_neighbors > 0 else [])
                            if batch:
                                stage = "neighbor"
                                diagnostics["neighbor_reconciliation"]["status"] = "in_progress"
                            else:
                                diagnostics["neighbor_reconciliation"]["status"] = (
                                    "budget_exhausted" if remaining_neighbors <= 0 else "neighbors_exhausted")
                                # Only now may the writer describe a final status:
                                # neighboring branches can reverse a trend after
                                # Jev establishes that they concern the same topic.
                                finalize()
                                if len(result["trends"]) >= 8:
                                    diagnostics["stop_reason"] = "eight_supported_trends"
                                    break
                        else:
                            batch = []
                        n = review_batch if review_count == 0 else 4
                        if stage != "neighbor":
                            batch = choose_batch(pool, reviewed, n, exploration_ratio)
                    if not batch:
                        diagnostics["stop_reason"] = "candidates_exhausted"
                        break
                    for candidate in batch:
                        if max_reviews is not None and review_count >= max_reviews:
                            break
                        reviewed.add(candidate["id"])
                        packet = sample_candidate(candidate, units, vectors, vector_indices, records, as_of,
                                                  context_index=context_index)
                        packet["aliases"].extend(select_packet_aliases(approved_aliases, packet))
                        packet["past_release_calendar"] = releases
                        packet = _bounded_packet(packet)
                        progress(f"Review {review_count+1}: {candidate['id']} ({len(packet['targets'])} examples; {stage}).")
                        if stage == "neighbor":
                            diagnostics["neighbor_reconciliation"]["reviews"] += 1
                            related_concept = candidate.get("neighbor_context", {}).get("concept_id")
                            if related_concept is not None:
                                neighbor_counts[related_concept] += 1
                        current_review = review_candidate(packet, writer, decisions,
                            accepted=_accepted_for_review(concepts.values(), by_id, as_of), accept_ratio=accept_ratio,
                            provisional_ratio=provisional_ratio, proposals=source_proposals(packet))
                        approved_aliases = remember_review_aliases(
                            approved_aliases, current_review["aliases"], by_id, as_of)
                        review_count += 1
                        diagnostics["audits"].append({"candidate": candidate["id"], "packet": packet, "review": current_review,
                                                     "stage": stage,
                                                     "neighbor_context": candidate.get("neighbor_context")})
                        for concern in current_review["concerns"]:
                            if concern["status"] in {"accepted", "provisional"} or (
                                concern["status"] == "mixed" and concern["relevance"] == "relevant"
                                and concern["scope_decision"] == "supported"):
                                store(candidate, concern)
                        if current_review["status"] in {"mixed", "provisional"} and candidate["kind"] == "cluster":
                            base = candidate.get("base_id", candidate["id"])
                            sampled = {r["id"] for r in packet["targets"]}
                            remaining = [i for i in candidate["indices"] if units[i]["voice_id"] not in sampled]
                            if remaining and refinements[base] < max_refinements:
                                refinements[base] += 1
                                refinement_queue.append({**candidate, "id": f"{base}:refine:{refinements[base]}",
                                                         "base_id": base, "indices": remaining})
                    if not neighbor_reviews:
                        finalize()
                        if len(result["trends"]) >= 8:
                            diagnostics["stop_reason"] = "eight_supported_trends"
                            break
                if diagnostics["stop_reason"] in {"candidates_exhausted", "runtime_review_limit", "token_target"}:
                    finalize(allow_provisional=True)
            except BudgetExceeded:
                diagnostics["stop_reason"] = "token_budget"
            except DecisionError as exc:
                diagnostics["stop_reason"] = "decision_service_error"
                diagnostics["decision_error"] = str(exc)
            except GenerationServiceError as exc:
                diagnostics["stop_reason"] = "generation_service_error"
                diagnostics["generation_error"] = str(exc)
            except LocalModelError as exc:
                diagnostics["stop_reason"] = "local_model_error"
                diagnostics["local_model_error"] = str(exc)
            finally:
                if neighbor_reviews and diagnostics["neighbor_reconciliation"]["status"] in {"pending", "in_progress"}:
                    diagnostics["neighbor_reconciliation"]["status"] = diagnostics["stop_reason"]
                diagnostics["neighbor_reconciliation"]["reviews_by_concept"] = {
                    key: count for key, count in neighbor_counts.items() if count}
                diagnostics["jev_usage"] = decisions.usage
                diagnostics["jev_requests"] = dict(decisions.stats)
                diagnostics["jev_resolved_model"] = getattr(decisions, "resolved_model", None)
                if chat_provider == "ollama" and models.stats.get("chat_calls", 0):
                    unload(chat_model)
                diagnostics["local_usage"] = dict(models.stats)
                if chat_provider == "openai":
                    diagnostics["generation_usage"] = {"provider": "openai", **writer.stats}
        diagnostics["review_attempts"] = review_count
        diagnostics["reviewed_aliases"] = approved_aliases
        diagnostics["watchlist"] = [{"id": c["id"], "definition": c["definition"],
            "match_ratio": c["match_ratio"], "history": c["history"]} for c in concepts.values()
            if c["id"] not in {e["concern_id"] for e in diagnostics["selected"]}]
    validate_output(result, records)
    return result, _plain(diagnostics)


def write_json(path: str, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

