"""Same-source, full-oracle development pilot. No automatic accuracy claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import inspect
from pathlib import Path
import time

from .datasets import load_questions
from .diagnostics import sha256, REPRESENTATIONS
from .experiment import answer_messages
from .memory import build_groups
from .schema import MemoryGroup, fingerprint, history_hash, read_jsonl, write_json
from .selection import Choice, render_evidence


def run(data, output_dir, backend, writer_path, max_questions=0):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    signature = {"data_sha256": sha256(data), "reader": backend.model_name, "writer": "Qwen3.5-4B",
                 "writer_config_sha256": sha256(Path(writer_path) / "config.json"), "window_turns": 8,
                 "fact_max_tokens": 2048, "fact_format": "compact", "answer_max_tokens": 384,
                 "answer_prompt_sha256": fingerprint(inspect.getsource(answer_messages)),
                 "memory_code_sha256": sha256(Path(__file__).with_name("memory.py")),
                 "pilot_code_sha256": sha256(__file__), "diagnostic": "all-oracle-sources-no-retrieval-no-budget-truncation"}
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != signature:
        raise ValueError("Pilot configuration changed; use a new output folder")
    write_json(config_path, signature)
    # v4 changes only parser acceptance, reusing identical per-window generation
    # caches while keeping the former writer manifest and answers intact.
    memory_path = root.parent / "oracle_writer_4b_v4.jsonl"
    memories = {row["question_id"]: row for row in read_jsonl(memory_path)} if memory_path.exists() else {}
    result_path = root / "predictions.jsonl"
    previous = list(read_jsonl(result_path)) if result_path.exists() else []
    if any(row["signature"] != fingerprint(signature) for row in previous):
        raise ValueError("Pilot run signature mismatch")
    completed = {(row["question_id"], row["representation"]) for row in previous}
    with result_path.open("a", encoding="utf-8") as stream:
        for question in load_questions("longmemeval", data, max_questions or None):
            stored = memories.get(question.question_id)
            expected_writer = fingerprint({key: signature[key] for key in
                ["data_sha256", "writer", "writer_config_sha256", "window_turns", "fact_max_tokens", "fact_format", "memory_code_sha256"]})
            if stored:
                if stored["history_hash"] != history_hash(question.turns) or stored["writer_signature"] != expected_writer:
                    raise ValueError("Incompatible shared writer cache")
                groups = [MemoryGroup.from_dict(group) for group in stored["groups"]]
            else:
                if backend.model_name != "Qwen3.5-4B":
                    raise ValueError("Build all pilot memories with the fixed 4B writer before changing readers")
                groups = build_groups(question.turns, backend, "Qwen3.5-4B", "cache/stage1_writer", fact_max_tokens=2048, fact_format="compact")
                stored = {"question_id": question.question_id, "history_hash": history_hash(question.turns),
                          "writer_signature": expected_writer, "groups": [group.to_dict() for group in groups]}
                with memory_path.open("a", encoding="utf-8") as memory_stream:
                    memory_stream.write(json.dumps(stored, ensure_ascii=False) + "\n")
                memories[question.question_id] = stored
                print("PILOT_BUILT", question.question_id, len(groups), flush=True)
            inputs, counts = {}, {}
            for representation in REPRESENTATIONS:
                evidence = render_evidence([Choice(group.group_id, representation, group.variants[representation],
                    tuple(group.source_ids), group.date, 0, 0) for group in groups])
                inputs[representation] = answer_messages(question.question, question.date, evidence)
                prompt = backend.tokenizer.apply_chat_template(inputs[representation], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                counts[representation] = len(backend.tokenizer.encode(prompt, add_special_tokens=False))
            oversized = any(count > backend.max_input_tokens for count in counts.values())
            for representation in REPRESENTATIONS:
                if (question.question_id, representation) in completed:
                    continue
                row = {"signature": fingerprint(signature), "question_id": question.question_id, "question_type": question.question_type,
                       "reader": backend.model_name, "representation": representation, "input_tokens_all_variants": counts,
                       "oracle_diagnostic_only": True, "source_ids": [source for group in groups for source in group.source_ids],
                       "history_hash": stored["history_hash"], "score_status": "not-judged"}
                if oversized:
                    row.update(status="excluded-all-representations-input-limit", hypothesis="", usage={})
                else:
                    result = backend.complete(inputs[representation], max_tokens=384)
                    row.update(status="answered", hypothesis=result["text"], usage=result["usage"])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print("PILOT_ANSWER", question.question_id, representation, row["status"], flush=True)
    rows = list(read_jsonl(result_path))
    write_json(root / "status.json", {"status": "completed", "rows": len(rows), "answered": sum(row["status"] == "answered" for row in rows),
                                     "oracle_development_only": True, "score_status": "not-judged"})


def main():
    from .__main__ import offline_environment
    from .backend import TransformersBackend
    offline_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="research/stage1/pilot_oracle.json")
    parser.add_argument("--model-path", default="models/Qwen3.5-4B")
    parser.add_argument("--writer-path", default="models/Qwen3.5-4B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args.data, args.output_dir, TransformersBackend(args.model_path), args.writer_path, args.limit)


if __name__ == "__main__":
    main()
