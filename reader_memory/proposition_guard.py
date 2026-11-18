"""Query-blind paragraph/list safeguards; a conservative baseline, not semantic proof."""
from __future__ import annotations

import re

from .schema import fingerprint
from .selection import render_piece

LIST = re.compile(r"(?m)^\s*(?:\d{1,3}[.)]|[-*•]|[A-Za-z][.)])\s+\S")
RISKS = {
    "quantity": re.compile(r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|once|twice|thrice|half|couple|dozen)\b", re.I),
    "negation_condition": re.compile(r"\b(?:not|never|no|cannot|can't|don't|doesn't|didn't|without|unless|only|if|except)\b", re.I),
    "temporal": re.compile(r"\b(?:before|after|previously|currently|now|since|until|earlier|later|today|yesterday|tomorrow)\b|\b(?:last|next)\s+(?:day|week|month|year|time)\b", re.I),
    "speech_act_preference": re.compile(r"\?|\b(?:please|recommend|could you|can you|would you|i (?:want|need|prefer|like|love|hate|dislike)|i'm (?:looking|interested))\b", re.I),
}


def evidence_units(group):
    """Keep contiguous source paragraphs intact, including numbered-list bodies.

    These are source fragments, not automatically verified logical propositions.
    A target chunk may itself end inside a longer original paragraph.
    """
    result = []
    for span in group.metadata["spans"]:
        text = span["text"]
        bounds = [0, *[m.end() for m in re.finditer(r"\n[ \t]*\n+", text)], len(text)]
        bounds = list(dict.fromkeys(bounds))
        units = []
        for start, end in zip(bounds, bounds[1:]):
            content = text[start:end]
            if not content:
                continue
            reasons = [name for name, pattern in RISKS.items() if pattern.search(content)]
            kind = "list_block" if LIST.search(content) else "paragraph"
            if kind == "list_block":
                reasons.append("complete_list")
            units.append({"unit_id": fingerprint([span["source_id"], span["start"] + start, span["start"] + end])[:20],
                          "source_id": span["source_id"], "role": span["role"], "date": span["date"],
                          "start": span["start"] + start, "end": span["start"] + end,
                          "text": content, "kind": kind, "reasons": reasons})
        for i, unit in enumerate(units[:-1]):
            if unit["text"].rstrip().endswith(":") and units[i + 1]["kind"] == "list_block":
                unit["reasons"].append("list_heading")
        assert "".join(u["text"] for u in units) == text
        result.extend(units)
    return result


def guarded_representation(group, counter):
    units = evidence_units(group)
    selected = [u for u in units if u["reasons"]]
    suffix = "\n".join(f"[{u['source_id']}@{u['start']}:{u['end']}] [{u['date']}] {u['role']}: {u['text'].strip()}" for u in selected)
    proposed = group.variants["fact"] + ("\n\n[Verbatim evidence units]\n" + suffix if suffix else "")
    cost = lambda text: counter.count(render_piece(group.group_id, group.date, text))
    raw_cost, proposed_cost = cost(group.variants["raw"]), cost(proposed)
    fallback = proposed_cost >= raw_cost
    value = group.variants["raw"] if fallback else proposed
    retained = [u["unit_id"] for u in selected if u["text"].strip() in value]
    assert len(retained) == len(selected)
    return value, {"policy": "whole source paragraphs/lists; fallback to raw when no shorter", "units": units,
                   "selected_unit_ids": [u["unit_id"] for u in selected], "verbatim_units_retained": retained,
                   "raw_fallback": fallback, "raw_tokens": raw_cost, "proposed_tokens": proposed_cost,
                   "actual_tokens": cost(value), "semantic_completeness_verified": False}
