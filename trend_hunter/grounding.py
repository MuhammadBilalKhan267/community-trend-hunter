"""Validate factual-claim references before spending a grounding request.

Reference validity is not semantic truth. Jev still checks the prose, scope and
interpretation; this module does not parse or certify every numerical phrase.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy


def _source_ids(state: dict) -> set[str]:
    identifiers = set()
    for name in ("sources", "context"):
        rows = state.get(name, [])
        if isinstance(rows, list):
            identifiers.update(row["id"] for row in rows if isinstance(row, dict)
                               and isinstance(row.get("id"), str) and row["id"].strip())
    return identifiers


def reference_catalog(state: dict) -> dict[str, list[str]]:
    """List actual references usable by structured final generation.

    Arrays remain a single measured-series reference; their elements do not
    expand the decoding grammar. Dictionary containers and leaves can be cited.
    """
    paths = set()

    def walk(value, path, ancestors):
        if isinstance(value, dict):
            if id(value) in ancestors:
                return
            for key, child in value.items():
                if isinstance(key, str) and key:
                    child_path = f"{path}.{key}"
                    paths.add(child_path)
                    walk(child, child_path, ancestors | {id(value)})

    for root in ("measurements", "engagement"):
        if isinstance(state.get(root), dict):
            walk(state[root], root, set())
    paths.update(field for field in ("status", "first_seen", "as_of") if field in state)
    return {"evidence_ids": sorted(_source_ids(state)), "metric_keys": sorted(paths)}


def _metric_leaves(state: dict) -> dict[str, list[str]]:
    """Index unambiguous bare leaf names, preserving their full provenance."""
    leaves = defaultdict(list)

    def walk(value, path, ancestors):
        if isinstance(value, dict):
            if id(value) in ancestors:
                return
            for key, child in value.items():
                if isinstance(key, str):
                    walk(child, path + [key], ancestors | {id(value)})
        elif path:
            # Arrays are a measured series, not a separate namespace of leaf
            # aliases for every numbered item.
            leaves[path[-1]].append(".".join(path))

    for root in ("measurements", "engagement"):
        if isinstance(state.get(root), dict):
            walk(state[root], [root], set())
    for field in ("status", "first_seen", "as_of"):
        if field in state:
            leaves[field].append(field)
    return leaves


def _metric_reference(reference: str, state: dict, leaves: dict) -> str:
    if len(reference) > 512:
        return "unknown"
    if "." not in reference:
        paths = leaves.get(reference, [])
        if len(paths) == 1:
            return "known"
        return "ambiguous" if len(paths) > 1 else "unknown"
    parts = reference.split(".")
    if len(parts) < 2 or parts[0] not in {"measurements", "engagement"} or any(not part for part in parts):
        return "unknown"
    value = state
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isascii() and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            return "unknown"
    return "known"


def claim_reference_material(claim: dict, state: dict) -> dict:
    """Bind a claim directly to its cited originals and measured values."""
    sources = {}
    for collection in ("sources", "context"):
        for source in state.get(collection, []):
            if isinstance(source, dict) and isinstance(source.get("id"), str):
                sources.setdefault(source["id"], source)
    identifiers = list(dict.fromkeys(claim.get("evidence_ids", [])))
    references = list(dict.fromkeys(claim.get("metric_keys", [])))
    leaves = _metric_leaves(state)
    metrics, unresolved = [], []
    for reference in references:
        resolution = _metric_reference(reference, state, leaves)
        if resolution != "known":
            unresolved.append({"reference": reference, "reason": resolution})
            continue
        path = reference if "." in reference else leaves[reference][0]
        value = state
        for part in path.split("."):
            value = value[int(part)] if isinstance(value, list) else value[part]
        metrics.append({"reference": reference, "path": path, "value": deepcopy(value)})
    return {
        "cited_original_sources": [deepcopy(sources[identifier]) for identifier in identifiers if identifier in sources],
        "cited_measurements": metrics,
        "missing_source_ids": [identifier for identifier in identifiers if identifier not in sources],
        "unresolved_metric_references": unresolved,
    }


def validate_draft_references(draft: dict, state: dict) -> list[str]:
    """Return stable failure labels; an empty list means references are valid.

    Evidence IDs must name supplied sources or context. Metric references may
    use measurements.* or engagement.* paths, or a unique bare leaf name.
    Every declared factual claim needs at least one valid reference. This
    function does not infer missing references from the wording of a claim.
    """
    if not isinstance(draft, dict) or not isinstance(state, dict):
        return ["invalid_grounding_state"]
    claims = draft.get("claims")
    if not isinstance(claims, list) or not claims:
        return ["missing_factual_claims"]
    sources = _source_ids(state)
    leaves = _metric_leaves(state)
    failures = []
    for index, claim in enumerate(claims):
        prefix = f"claim_{index}"
        if not isinstance(claim, dict):
            failures.append(f"{prefix}:invalid_claim")
            continue
        identifiers, metrics = claim.get("evidence_ids"), claim.get("metric_keys")
        if any(not isinstance(refs, list) or any(not isinstance(value, str) or not value.strip() for value in refs)
               for refs in (identifiers, metrics)):
            failures.append(f"{prefix}:invalid_references")
            continue
        if not identifiers and not metrics:
            failures.append(f"{prefix}:missing_reference")
            continue
        if any(identifier not in sources for identifier in identifiers):
            failures.append(f"{prefix}:unknown_source_reference")
        resolutions = {_metric_reference(metric, state, leaves) for metric in metrics}
        if "unknown" in resolutions:
            failures.append(f"{prefix}:unknown_metric_reference")
        if "ambiguous" in resolutions:
            failures.append(f"{prefix}:ambiguous_metric_reference")
    return failures
