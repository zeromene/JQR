"""Protocol comparison: fixed v2 evidence, open answers, 384-token allowance."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import re
import time

from .capability_battery import cell, holm, solve, validate_battery
from .diagnostics import DIMENSIONS, REPRESENTATIONS, sha256
from .experiment import answer_messages
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/stage3_protocol"
VERSION = "open-protocol-v1"
DATE = "2026-10-25"


def prepare(directory=STUDY):
    directory = Path(directory)
    if (directory / "manifest.json").exists() or (directory / "items.json").exists():
        raise ValueError("Preserve frozen protocol study; use a new directory for changes")
    source = ROOT / "research/stage2/capability_probes.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    validate_battery(original)
    items = [{**{k: i[k] for k in ("id", "subset", "dimension", "family", "difficulty", "question", "variants")},
              "answer": solve(i["semantic_spec"])} for i in original]
    write_json(directory / "items.json", items)
    write_json(directory / "manifest.json", {"version": VERSION, "frozen_at": datetime.now(timezone.utc).isoformat(),
               "source_sha256": sha256(source), "data_sha256": sha256(directory / "items.json"), "code_sha256": sha256(__file__),
               "base_questions": len(items), "expected_rows_per_reader": len(expected_keys(items)),
               "source_reused": True, "max_new_tokens": 384,
               "purpose": "Post-hoc protocol comparison, not a new independent benchmark or selector evaluation.",
               "profile_subset": "calibration/raw", "audit_representations": REPRESENTATIONS,
               "readers": ["Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Phi-4-mini-instruct"]})
    print("FROZEN_PROTOCOL", len(items), len(expected_keys(items)), flush=True)


def expected_keys(items):
    return {(i["id"], rep) for i in items for rep in (("raw",) if i["subset"] == "calibration" else REPRESENTATIONS)}


def messages(item, representation):
    # Never send options, reference answers or semantic specifications.
    kind = "one integer without units" if item["dimension"] == "numeric" else "one person or city name"
    question = item["question"] + f"\nGive a concise answer. Finish with <answer>VALUE</answer>, where VALUE is {kind}. "
    question += "You may put a brief explanation before that final tag. Put only the answer value inside the tag."
    return answer_messages(question, DATE, item["variants"][representation])


def canonical(value):
    return re.sub(r"\s+", " ", value.strip().rstrip(".")).casefold()


def parse_answer(text):
    if len(re.findall(r"<answer>", text, re.I)) != 1:
        return None
    match = re.search(r"<answer>\s*([^<>\n]+?)\s*</answer>\s*$", text, re.I)
    return canonical(match[1]) if match else None


def run(data, output, backend):
    data = Path(data)
    items = json.loads(data.read_text(encoding="utf-8"))
    manifest = json.loads(data.with_name("manifest.json").read_text(encoding="utf-8"))
    if manifest["data_sha256"] != sha256(data) or manifest["code_sha256"] != sha256(__file__):
        raise ValueError("Frozen protocol code/data mismatch")
    signature = {"version": VERSION, "data_sha256": sha256(data), "code_sha256": sha256(__file__),
                 "backend_sha256": sha256(Path(__file__).with_name("backend.py")),
                 "shared_prompt_sha256": fingerprint(inspect.getsource(answer_messages)),
                 "reader": backend.model_name, "reader_manifest_sha256": sha256(Path(backend.path) / "source_manifest.json"),
                 "max_input_tokens": backend.max_input_tokens, "max_new_tokens": 384,
                 "date": DATE, "decoding": "greedy-no-thinking-bf16-sdpa"}
    sid = fingerprint(signature)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = path.with_suffix(".config.json")
    if config.exists() and fingerprint(json.loads(config.read_text(encoding="utf-8"))) != sid:
        raise ValueError("Incompatible protocol run configuration")
    rows = list(read_jsonl(path)) if path.exists() else []
    done = {(r["id"], r["representation"]) for r in rows}
    expected = expected_keys(items)
    if len(done) != len(rows) or not done <= expected or any(r["signature"] != sid for r in rows):
        raise ValueError("Incompatible or duplicate protocol rows")
    write_json(config, signature)
    with path.open("a", encoding="utf-8") as stream:
        for item in items:
            reps = ("raw",) if item["subset"] == "calibration" else REPRESENTATIONS
            for rep in reps:
                if (item["id"], rep) in done:
                    continue
                start = time.perf_counter()
                response = backend.complete(messages(item, rep), max_tokens=384)
                parsed = parse_answer(response["text"])
                row = {k: item[k] for k in ("id", "subset", "dimension", "family", "difficulty")}
                row.update(reader=backend.model_name, signature=sid, representation=rep, prediction=response["text"],
                           parsed_answer=parsed, gold_answer=canonical(item["answer"]), correct=parsed == canonical(item["answer"]),
                           usage=response.get("usage", {}), seconds=round(time.perf_counter() - start, 4),
                           synthetic_protocol_diagnostic=True)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print("OPEN_PROTOCOL", backend.model_name, item["id"], rep, row["correct"], flush=True)
    write_json(path.with_suffix(".status.json"), {"status": "completed", "rows": len(expected), "signature": sid})


def stats(values):
    # Reuse the generic binomial summary while keeping open-format validation explicit.
    return cell([{**r, "parsed_label": r["parsed_answer"]} for r in values])


def report(data, predictions, destination):
    items = json.loads(Path(data).read_text(encoding="utf-8"))
    rows = list(read_jsonl(predictions))
    index = {(r["id"], r["representation"]): r for r in rows}
    config = json.loads(Path(predictions).with_suffix(".config.json").read_text(encoding="utf-8"))
    if (len(rows) != len(index) or set(index) != expected_keys(items) or config["data_sha256"] != sha256(data)
            or any(r["signature"] != fingerprint(config) or r["reader"] != config["reader"] for r in rows)):
        raise ValueError("Expected a complete, valid, single-reader protocol run")
    for item in items:
        for rep in (("raw",) if item["subset"] == "calibration" else REPRESENTATIONS):
            row = index[item["id"], rep]
            if (row["parsed_answer"] != parse_answer(row["prediction"]) or row["gold_answer"] != canonical(item["answer"])
                    or row["correct"] != (row["parsed_answer"] == canonical(item["answer"]))
                    or any(row[k] != item[k] for k in ("subset", "dimension", "family", "difficulty"))):
                raise ValueError("Scoring/metadata mismatch")
    groups = defaultdict(list)
    for row in rows:
        groups[row["subset"], row["dimension"], row["representation"]].append(row)
    cells = [{"subset": s, "dimension": d, "representation": rep, **stats(v)} for (s, d, rep), v in sorted(groups.items())]
    pairs = []
    for dim in DIMENSIONS:
        ids = [i["id"] for i in items if i["subset"] == "audit" and i["dimension"] == dim]
        for rep in ("fact", "summary"):
            a = sum(index[i, "raw"]["correct"] and not index[i, rep]["correct"] for i in ids)
            b = sum(index[i, rep]["correct"] and not index[i, "raw"]["correct"] for i in ids)
            n = a + b
            p = min(1, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n) if n else 1.0
            pairs.append({"dimension": dim, "comparison": rep + "-raw", "n": len(ids), "raw_only_correct": a,
                          "other_only_correct": b, "accuracy_delta": (b - a) / len(ids), "mcnemar_exact_p": p})
    for pair, p in zip(pairs, holm([x["mcnemar_exact_p"] for x in pairs])):
        pair["holm_p_within_reader"] = p
    profile = {d: stats(groups["calibration", d, "raw"]) for d in DIMENSIONS}
    protocol_pairs = []
    tag = {"Qwen3.5-2B": "2b", "Qwen3.5-4B": "4b", "Qwen3.5-9B": "9b"}.get(config["reader"])
    if tag:
        old = {(r["id"], r["representation"]): r for r in read_jsonl(ROOT / f"runs/research_stage2/probes_{tag}.jsonl")}
        for dim in DIMENSIONS:
            ids = [i["id"] for i in items if i["subset"] == "calibration" and i["dimension"] == dim]
            protocol_pairs.append({"dimension": dim, "n": len(ids), "old_correct": sum(old[i, "raw"]["correct"] for i in ids),
                                   "open_correct": sum(index[i, "raw"]["correct"] for i in ids),
                                   "old_only_correct": sum(old[i, "raw"]["correct"] and not index[i, "raw"]["correct"] for i in ids),
                                   "open_only_correct": sum(index[i, "raw"]["correct"] and not old[i, "raw"]["correct"] for i in ids)})
    value = {"version": VERSION, "status": "complete-protocol-comparison-not-selector-evaluation", "reader": config["reader"],
             "rows": len(rows), "cells": cells, "audit_pairs": pairs, "calibration": profile,
             "paired_calibration_protocol_comparison": protocol_pairs,
             "invalid_outputs": sum(r["parsed_answer"] is None for r in rows),
             "generation_cap_hits": sum(r.get("usage", {}).get("completion_tokens", 0) >= 384 for r in rows),
             "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in rows),
             "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in rows),
             "limitations": ["Same v2 examples reused for a protocol check; not unseen tasks.",
                             "Open answers change both task format and generation allowance; this is a joint protocol intervention.",
                             "Output tags differ from ordinary free-form natural answers.",
                             "No evidence token budget or learned selector is evaluated here."]}
    write_json(destination, value)
    write_json(Path(destination).with_suffix(".profile.json"), {"reader": config["reader"],
               "abilities": {d: v["accuracy"] for d, v in profile.items()}, "uncertainty": profile,
               "status": "protocol-aligned-synthetic-calibration-provisional", "provenance": {
                   "run_signature": fingerprint(config), "subset": "calibration", "representation": "raw"}})
    print("REPORTED_OPEN_PROTOCOL", config["reader"], {d: v["correct"] for d, v in profile.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("run")
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", required=True)
    q = sub.add_parser("report")
    q.add_argument("--predictions", required=True)
    q.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare()
    if args.command == "report":
        return report(STUDY / "items.json", args.predictions, args.output)
    from .__main__ import offline_environment
    from .backend import TransformersBackend
    offline_environment()
    return run(STUDY / "items.json", args.output, TransformersBackend(args.model_path))


if __name__ == "__main__":
    main()
