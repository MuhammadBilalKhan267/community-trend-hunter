"""Input boundaries, conservative text preparation, and reply context."""
from __future__ import annotations

import csv
import html
import json
import re
import unicodedata
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

GAME_TIMEZONE = timezone(timedelta(hours=8))
PREPROCESS_VERSION = "own-clause-v5-short-text"
_DISCORD_EMOJI = re.compile(r"<a?:([^:\s<>]+):[0-9]+>")
REACTIONS = {
    "+1": "agree", "确实": "agree", "確實": "agree", "同意": "agree",
    "赞同": "agree", "贊同": "agree", "是的": "agree", "没错": "agree",
    "沒錯": "agree", "附议": "agree", "附議": "agree", "支持": "agree",
    "不同意": "disagree", "不认同": "disagree", "不認同": "disagree",
    "我也是": "same_experience", "俺也一样": "same_experience", "俺也一樣": "same_experience",
    "同问": "question", "同問": "question", "蹲一个": "question", "蹲一個": "question",
}
GREETINGS = {"早", "早安", "晚安", "签到", "簽到", "沙发", "沙發", "hello", "hi", "大家好", "吃了吗", "吃了嗎"}


def normalize(text: str | None) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(text or ""))
    text = re.sub(r"<emoji:([^>]+)>", r" [\1] ", text)
    text = _DISCORD_EMOJI.sub(r" [\1] ", text)
    text = re.sub(r"<image:[^>]*>", " [image unavailable] ", text)
    text = re.sub(r"@[^\s，。！？!?：:]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _available_text(text: str | None) -> str:
    """Readable source text, excluding unavailable media and emoji metadata."""
    raw = re.sub(r"<(?:image|emoji):[^>]*>", " ", html.unescape(text or ""))
    raw = _DISCORD_EMOJI.sub(" ", raw)
    return normalize(raw)


def _semantic_text(text: str | None) -> str:
    """Keep contextual emoji cues without embedding synthetic image words."""
    raw = re.sub(r"<image:[^>]*>", " ", html.unescape(text or ""))
    return normalize(raw)


def _lexical_size(text: str) -> int:
    return len(re.findall(r"[\w\u4e00-\u9fff]", text))


def reaction_kind(text: str) -> str | None:
    bare = re.sub(r"[。.!！?？\s]+$", "", text).strip().lower()
    without_emoji = re.sub(r"\[[^\]]+\]", "", bare).strip()
    if without_emoji != bare and without_emoji in REACTIONS:
        # A minimal reaction with an irony cue has no independent proposition.
        return "uncertain" if "doge" in bare else REACTIONS[without_emoji]
    return REACTIONS.get(bare)


def _is_greeting(text: str) -> bool:
    return re.sub(r"[\W_]+", "", text).lower() in GREETINGS


def _substantive(raw: str | None) -> bool:
    available = _available_text(raw)
    # Length cannot establish whether a compact contribution has useful
    # meaning. Preserve readable own text; sampled review owns relevance.
    return (bool(_lexical_size(available)) and not _is_greeting(available)
            and not reaction_kind(_semantic_text(raw)))


def prepare(path: str | Path, as_of: str) -> tuple[list[dict], list[dict], dict]:
    cutoff = date.fromisoformat(as_of)
    visible: list[dict] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            stamp = datetime.fromisoformat(record["created_at"])
            if stamp.tzinfo is None:
                raise ValueError(f"Line {line_number}: timestamp must include a timezone")
            day = stamp.astimezone(GAME_TIMEZONE).date()
            if day > cutoff:
                continue
            if record["id"] in seen:
                raise ValueError(f"Duplicate eligible message ID: {record['id']}")
            seen.add(record["id"])
            record = dict(record, day=day.isoformat(), _instant=stamp)
            record["image_refs"] = re.findall(r"<image:([^>]*)>", str(record.get("content") or ""))
            record["image_count"] = len(record["image_refs"])
            record["text"] = normalize(record.get("content"))
            own_fields = [record.get(field) for field in ("content", "title")]
            available = [_available_text(raw) for raw in own_fields]
            substantive = [_substantive(raw) for raw in own_fields]
            # Family identity includes every readable own field. A short body
            # must not collapse different titled contributions into one family.
            # It identifies duplicate own text, not a semantic topic or issue.
            family_parts = list(dict.fromkeys(_semantic_text(raw) for raw, readable in zip(own_fields, substantive)
                                              if readable))
            if not family_parts:
                family_index = next((i for i, text in enumerate(available) if _lexical_size(text)), 0)
                family_parts = [_semantic_text(own_fields[family_index])]
            record["family"] = "\n".join(family_parts)
            record["reaction"] = None if any(substantive) else reaction_kind(record["text"])
            record["bot_like"] = bool(re.search(r"(?:Q群管家|群管家)\s*[:：].*欢迎", record["text"]))
            record["trivial"] = not any(
                _lexical_size(text) and not _is_greeting(text)
                for text in available)
            record["media_only"] = bool(record["image_count"]) and not any(
                _lexical_size(text) for text in available)
            visible.append(record)
    visible.sort(key=lambda x: (x["_instant"], x["id"]))
    by_id = {r["id"]: r for r in visible}
    for record in visible:
        root = record
        visited = {root["id"]}
        context = []
        parent_id = root.get("parent_id")
        while parent_id and parent_id in by_id:
            if parent_id in visited:
                raise ValueError("Cycle in eligible reply graph")
            visited.add(parent_id)
            parent = by_id[parent_id]
            if parent["_instant"] > record["_instant"]:
                break
            if len(context) < 2:
                context.append(parent_id)
            root = parent
            parent_id = parent.get("parent_id")
        record["root_id"] = root["id"]
        record["context_ids"] = context
    eligible = [r for r in visible if not r["bot_like"] and not r["trivial"] and r["family"]]
    stats = {"visible_messages": len(visible), "eligible_messages": len(eligible),
             "bot_like_excluded": sum(r["bot_like"] for r in visible),
             "trivial_excluded": sum(r["trivial"] for r in visible),
             "ambiguous_reactions": sum(r["reaction"] == "uncertain" for r in eligible),
             "unresolved_reactions": sum(bool(r["reaction"]) and not r["context_ids"] for r in eligible)}
    stats["image_messages"] = sum(bool(r["image_count"]) for r in visible)
    stats["image_placeholders"] = sum(r["image_count"] for r in visible)
    stats["media_only_messages"] = sum(r["media_only"] for r in visible)
    stats["image_messages_with_text"] = stats["image_messages"] - stats["media_only_messages"]
    stats["image_messages_excluded_as_trivial"] = sum(bool(r["image_count"]) and r["trivial"] for r in visible)
    return visible, eligible, stats


def make_units(records: list[dict]) -> list[dict]:
    """Split substantive clauses; a source can participate in multiple concerns."""
    units = []
    for record in records:
        if record["reaction"]:
            continue
        parts = []
        # The title is the source author's own contribution, not reply context.
        # Content comes first so existing content-unit indices remain stable.
        for raw in (record.get("content") or "", record.get("title") or ""):
            # Strip complete tags before punctuation splits can cut a reference
            # containing punctuation into misleading fragments.
            raw = re.sub(r"<image:[^>]*>", " ", html.unescape(raw))
            for fragment in re.split(r"(?:[。！？!?；;\n]+|…{2,}|\.{3,})", raw):
                text = _semantic_text(fragment).strip(" ，,:：")
                if _substantive(fragment):
                    if text not in parts:
                        parts.append(text)
        if not parts:
            # Several short clauses can form a substantive whole. Synthetic
            # annotation lengths cannot make an otherwise empty source qualify.
            for raw in (record.get("content") or "", record.get("title") or ""):
                if not _substantive(raw):
                    continue
                text = _semantic_text(raw)
                parts = [text]
                break
        for index, text in enumerate(parts):
            units.append({"unit_id": f"{record['id']}:{index}", "voice_id": record["id"],
                          "text": text, "day": record["day"], "platform": record["platform"]})
    return units


def read_releases(path: str | Path | None, as_of: str) -> list[dict]:
    if path is None or not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return [r for r in csv.DictReader(source) if r["date"] <= as_of]
