"""Source-backed alias assertions for review context, never embedding rewrites."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import re
import unicodedata

from .data import GAME_TIMEZONE


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _source(record, identifier, cutoff):
    if not isinstance(record, dict) or record.get("id", identifier) != identifier:
        return None
    try:
        stamp = datetime.fromisoformat(record["created_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if stamp.tzinfo is None or stamp.astimezone(GAME_TIMEZONE).date() > cutoff:
        return None
    content, title = record.get("content"), record.get("title")
    text = content if isinstance(content, str) and content.strip() else title if isinstance(title, str) else ""
    return {"id": identifier, "text": text, "created_at": record["created_at"],
            "platform": record.get("platform"), "channel": record.get("channel")}


def _identity(assertion: dict) -> tuple:
    name, alias = assertion.get("name", assertion.get("referent", "")), assertion.get("alias", "")
    refs = assertion.get("evidence_ids") or [assertion.get("source_id")]
    return (_normalized(name), _normalized(alias), tuple(sorted(set(ref for ref in refs if isinstance(ref, str)))))


def remember_review_aliases(registry: list[dict], approved_aliases: list[dict],
                            by_id: dict, as_of: str) -> list[dict]:
    """Return a registry with newly approved, cutoff-valid assertions appended.

    Semantic approval belongs to the existing Jev review. This helper only
    checks provenance and preserves the joint context when an interpretation
    cites several sources. Unknown or future source IDs invalidate that
    proposed assertion rather than silently dropping part of its evidence.
    """
    cutoff = date.fromisoformat(as_of)
    if cutoff.isoformat() != as_of:
        raise ValueError("Alias cutoff must use exact YYYY-MM-DD format")
    result = deepcopy(registry)
    known = {_identity(assertion) for assertion in result}
    for proposal in approved_aliases:
        if not isinstance(proposal, dict):
            continue
        name, alias = proposal.get("referent", proposal.get("name")), proposal.get("alias")
        refs = proposal.get("evidence_ids")
        if (not isinstance(name, str) or not name.strip() or not isinstance(alias, str)
                or not alias.strip() or _normalized(name) == _normalized(alias)
                or not isinstance(refs, list) or not refs
                or any(not isinstance(ref, str) or not ref for ref in refs)):
            continue
        evidence = [_source(by_id.get(identifier), identifier, cutoff) for identifier in dict.fromkeys(refs)]
        if any(source is None for source in evidence) or not any(source["text"].strip() for source in evidence):
            continue
        evidence.sort(key=lambda row: (datetime.fromisoformat(row["created_at"]), row["id"]))
        primary = next(source for source in evidence if source["text"].strip())
        days = [datetime.fromisoformat(source["created_at"]).astimezone(GAME_TIMEZONE).date().isoformat()
                for source in evidence]
        streams = list(dict.fromkeys((source["platform"], source["channel"]) for source in evidence))
        assertion = {
            "name": name.strip(), "referent": name.strip(), "alias": alias.strip(),
            "source_id": primary["id"], "source_text": primary["text"], "created_at": primary["created_at"],
            "evidence_ids": [source["id"] for source in evidence], "evidence": evidence,
            "scope": {"streams": [{"platform": platform, "channel": channel} for platform, channel in streams],
                      "observed_from": min(days), "observed_through": max(days),
                      "kind": "reviewed_context"},
            "caveat": "A relationship accepted in the supplied source context; other uses may differ. "
                      "The cited sources support the interpretation jointly, not necessarily individually.",
        }
        if "confidence" in proposal:
            assertion["confidence"] = proposal["confidence"]
        key = _identity(assertion)
        if key not in known:
            result.append(assertion)
            known.add(key)
    return result


def _mentions(material: str, term: str) -> bool:
    term = _normalized(term)
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9_ ]+", term):
        return re.search(r"(?<![a-z0-9_])" + re.escape(term) + r"(?![a-z0-9_])", material) is not None
    return term in material


def select_packet_aliases(registry: list[dict], packet: dict) -> list[dict]:
    """Select literal mentions as contextual references, retaining source scope."""
    material = "\n".join(str(row.get(field) or "") for key in ("targets", "context")
                         for row in packet.get(key, []) if isinstance(row, dict)
                         for field in ("text", "content", "title"))
    material = _normalized(material)
    cutoff = date.fromisoformat(packet["as_of"]) if packet.get("as_of") else None
    selected = []
    for assertion in registry:
        name = assertion.get("name", assertion.get("referent", ""))
        alias = assertion.get("alias", "")
        if not isinstance(name, str) or not isinstance(alias, str):
            continue
        if cutoff:
            sources = assertion.get("evidence") or [{"id": assertion.get("source_id"),
                                                     "created_at": assertion.get("created_at"),
                                                     "content": assertion.get("source_text")}]
            if any(_source(source, source.get("id"), cutoff) is None for source in sources):
                continue
        if _mentions(material, name) or _mentions(material, alias):
            selected.append(deepcopy(assertion))
    return selected
