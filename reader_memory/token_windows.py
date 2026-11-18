"""Query-blind, lossless character-span chunks bounded by serialized reader tokens."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import re

from .retrieval import BM25
from .schema import MemoryGroup, Turn, fingerprint, history_hash
from .selection import Choice, render_evidence


def group_from_spans(spans):
    identifiers = [f"{s['source_id']}@{s['start']}:{s['end']}" for s in spans]
    raw = "\n".join(f"[{sid}] [{s['date']}] {s['role']}: {s['text']}" for sid, s in zip(identifiers, spans))
    return MemoryGroup("w" + fingerprint(identifiers)[:16], identifiers, spans[0]["date"], {"raw": raw}, {"spans": spans})


def serialize(group, representation="raw"):
    return render_evidence([Choice(group.group_id, representation, group.variants[representation], tuple(group.source_ids), group.date, 0, 0)])


def token_windows(turns, counter, limit=448):
    if not counter.exact:
        raise ValueError("Chunk construction requires the actual reader tokenizer")
    if limit < 64:
        raise ValueError("Token limit too small for provenance headers")
    if len({t.source_id for t in turns}) != len(turns):
        raise ValueError("Duplicate source turn")
    groups, pending, boundary = [], [], None

    def span(turn, start, end):
        return {"source_id": turn.source_id, "start": start, "end": end,
                "role": turn.role, "date": turn.date, "text": turn.content[start:end]}

    def fits(spans):
        return counter.count(serialize(group_from_spans(spans))) <= limit

    for turn in turns:
        current = (turn.source_id.rsplit(":", 1)[0], turn.date)
        if pending and current != boundary:
            groups.append(group_from_spans(pending))
            pending = []
        boundary = current
        entire = span(turn, 0, len(turn.content))
        if fits([*pending, entire]):
            pending.append(entire)
            continue
        if pending:
            groups.append(group_from_spans(pending))
            pending = []
        if fits([entire]):
            pending = [entire]
            continue
        start = 0
        while start < len(turn.content):
            low, high, best = start + 1, len(turn.content), start
            while low <= high:
                mid = (low + high) // 2
                if fits([span(turn, start, mid)]):
                    best, low = mid, mid + 1
                else:
                    high = mid - 1
            if best == start:
                raise ValueError("A source header plus one character exceeds the token limit")
            # Prefer a word boundary; verification below handles nonadditive BPE.
            sentence_ends = [start + m.end() for m in re.finditer(r"(?<=[.!?;])\s+", turn.content[start:best])
                             if m.end() >= (best-start)//2]
            split = max(turn.content.rfind(" ", start + (best-start)//2, best),
                        turn.content.rfind("\n", start + (best-start)//2, best))
            if sentence_ends:
                best = sentence_ends[-1]
            elif split > start:
                best = split + 1
            while best > start and not fits([span(turn, start, best)]):
                best -= 1
            if best <= start:
                raise ValueError("Unable to construct a bounded source fragment")
            groups.append(group_from_spans([span(turn, start, best)]))
            start = best
    if pending:
        groups.append(group_from_spans(pending))
    validate_partition(turns, groups, counter, limit)
    return groups


def validate_partition(turns, groups, counter, limit):
    expected = {t.source_id: t for t in turns}
    spans = defaultdict(list)
    if len({g.group_id for g in groups}) != len(groups):
        raise ValueError("Duplicate chunk")
    for group in groups:
        if counter.count(serialize(group)) > limit:
            raise ValueError("Serialized chunk exceeds the budget")
        if group.to_dict() != group_from_spans(group.metadata["spans"]).to_dict():
            raise ValueError("Chunk provenance or text changed")
        for s in group.metadata["spans"]:
            if s["source_id"] not in expected:
                raise ValueError("Unknown source span")
            turn = expected[s["source_id"]]
            if (s["text"] != turn.content[s["start"]:s["end"]] or s["role"] != turn.role or s["date"] != turn.date):
                raise ValueError("Source span mismatch")
            spans[s["source_id"]].append(s)
    if set(spans) != set(expected):
        raise ValueError("Source coverage changed")
    for sid, turn in expected.items():
        parts = sorted(spans[sid], key=lambda s: s["start"])
        cursor = 0
        for s in parts:
            if s["start"] != cursor or s["end"] < cursor or (s["end"] == cursor and turn.content):
                raise ValueError("Source gap, overlap, or empty fragment")
            cursor = s["end"]
        if cursor != len(turn.content) or "".join(s["text"] for s in parts) != turn.content:
            raise ValueError("Source content was lost")


def neutral_query(item, counter, limit=448, top_k=24):
    if len(item["haystack_sessions"]) != len(item["haystack_dates"]):
        raise ValueError("History dates do not align")
    turns = [Turn(f"s{si:04d}:t{ti:04d}", str(m["role"]), str(m["content"]), str(item["haystack_dates"][si]))
             for si in sorted(range(len(item["haystack_sessions"])), key=lambda i: (str(item["haystack_dates"][i]), i))
             for ti, m in enumerate(item["haystack_sessions"][si])]
    groups = token_windows(turns, counter, limit)
    ranked = BM25(groups).search(item["question"], top_k)
    return {"question_id": item["question_id"], "question": item["question"], "date": str(item.get("question_date", "")),
            "history_hash": history_hash(turns), "source_turns": len(turns), "chunks": len(groups),
            "candidates": [{"group": g.to_dict(), "rank": i+1, "bm25": score,
                            "raw_tokens": counter.count(serialize(g))} for i, (g, score) in enumerate(ranked)]}


SENSITIVE = re.compile(r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|once|twice|thrice|"
                       r"first|second|third|half|couple|dozen|not|never|no|cannot|can't|don't|doesn't|didn't|without|unless|"
                       r"only|if|before|after|previously|currently|now|since|until|earlier|later|last|next|"
                       r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|"
                       r"july|august|september|october|november|december)\b", re.I)


def protected_clauses(group):
    result = []
    for span in group.metadata["spans"]:
        # The content, not dates/source IDs, determines sensitivity.
        for match in re.finditer(r"[^\n]+", span["text"]):
            for sentence in re.split(r"(?<=[.!?;])\s+", match[0]):
                text = sentence.strip()
                if text and SENSITIVE.search(text):
                    result.append({"source_id": span["source_id"], "role": span["role"], "date": span["date"], "text": text})
    return result


def guarded_fact(group):
    clauses = protected_clauses(group)
    # Every detected clause is retained verbatim, even if its paraphrase is present.
    by_source = defaultdict(list)
    for c in clauses:
        by_source[c['source_id'], c['date'], c['role']].append(c['text'])
    guard = "\n".join(f"[{sid}] [{date}] {role}: " + "\n".join(texts)
                      for (sid, date, role), texts in by_source.items())
    return group.variants["fact"] + ("\n\n[Verbatim source safeguards]\n" + guard if guard else ""), clauses
