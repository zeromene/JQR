"""Benchmark adapters. History objects expose no answer-location or gold labels."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .schema import Turn


@dataclass
class Question:
    question_id: str
    question: str
    date: str
    turns: list[Turn]
    # Evaluation-only fields: never pass this object to the writer or selector.
    answer: object
    question_type: str
    gold_source_ids: list[str]


def load_longmemeval(path):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in items:
        turns, gold = [], []
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [""] * len(sessions))
        session_ids = item.get("haystack_session_ids", list(range(len(sessions))))
        if not (len(sessions) == len(dates) == len(session_ids)):
            raise ValueError("History dates/session IDs do not align")
        order = sorted(range(len(sessions)), key=lambda i: (str(dates[i]), i))
        for index in order:
            for turn_index, message in enumerate(sessions[index]):
                # Neutral IDs: do not expose the benchmark's answer/noans names.
                source_id = f"s{index:04d}:t{turn_index:04d}"
                turns.append(Turn(source_id, str(message["role"]), str(message["content"]), str(dates[index])))
                if message.get("has_answer"):
                    gold.append(source_id)
        yield Question(str(item["question_id"]), item["question"], str(item.get("question_date", "")), turns,
                       item.get("answer", ""), item.get("question_type", ""), gold)


def load_locomo(path):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    categories = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    for conversation_index, item in enumerate(items):
        conversation = item["conversation"]
        turns = []
        session_keys = [key for key in conversation if key.startswith("session_") and key[len("session_"):].isdigit()]
        for session_key in sorted(session_keys, key=lambda key: int(key.split("_")[-1])):
            date = str(conversation.get(session_key + "_date_time", ""))
            for i, message in enumerate(conversation[session_key]):
                neutral_id = f"c{conversation_index}:{session_key}:t{i}"
                turns.append(Turn(neutral_id, str(message.get("speaker", "user")), str(message.get("text", "")), date))
        for question_index, qa in enumerate(item.get("qa", [])):
            # LoCoMo has different reference/evidence conventions from LongMemEval.
            yield Question(f"locomo:{conversation_index}:{question_index}", qa["question"], "", turns,
                           qa.get("answer", qa.get("adversarial_answer", "")), categories.get(qa.get("category"), "unknown"), [])


def load_questions(kind, path, limit=None):
    loader = load_longmemeval if kind == "longmemeval" else load_locomo
    for index, question in enumerate(loader(path)):
        if limit is not None and index >= limit:
            break
        yield question
