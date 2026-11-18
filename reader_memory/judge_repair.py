"""Preset constrained Qwen9B judge, developed on old CALIBRATION cases only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .diagnostics import sha256
from .judge_validation import metrics
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/judge_repair/v2"
OUTPUT = ROOT / "runs/judge_repair_v2"
JUDGE = "Qwen3.5-9B"
FILES = ("judge_repair.py", "binary_judge.py", "judge_validation.py", "backend.py", "schema.py", "diagnostics.py")


def hashes():
    return {n: sha256(ROOT / "reader_memory" / n) for n in FILES}


def prepare():
    if STUDY.exists():
        raise FileExistsError("Preserve frozen judge repair; use a new version")
    package = ROOT / "research/judge_calibration/v1"
    selected = [c for c in read_jsonl(package / "cases.jsonl") if c["fold"] == "judge_calibration"]
    provenance = {p["case_id"]: p for p in read_jsonl(package / "provenance.private.jsonl")}
    inputs = [{"case_id": c["case_id"], "question_id": provenance[c["case_id"]]["question_id"],
               "question_type": c["question_type"], "official_judge_prompt": c["official_judge_prompt"]} for c in selected]
    if len(inputs) != 54 or len({c["question_id"] for c in inputs}) != 6:
        raise ValueError("Expected the frozen 54-case calibration partition only")
    STUDY.mkdir(parents=True)
    write_json(STUDY / "inputs.json", inputs)
    human = package / "human/rxx/annotations.rxx.jsonl"
    manifest = {"version": "judge-repair-v2", "frozen_at": datetime.now(timezone.utc).isoformat(), "judge": JUDGE,
                "code_hashes": hashes(), "inputs_sha256": sha256(STUDY / "inputs.json"), "human_sha256": sha256(human),
                "human_path": human.relative_to(ROOT).as_posix(), "decoding": "prefix-constrained yes/no plus EOS",
                "prompt": "unmodified official case prompt", "calibration_only": True, "old_audit_reused": False,
                "independent_validation_available": False, "formal_natural_scoring_approved": False,
                "selection": "one preset replacement model and output protocol; no candidate search on old audit",
                "limitations": ["54 labels from six oracle questions are calibration data, not independent validation.",
                                "Fresh temporal annotation material will cover only one new source group.",
                                "Cannot approve formal natural scoring from calibration agreement alone."]}
    write_json(STUDY / "manifest.json", manifest)
    print("JUDGE_REPAIR_PREPARED", len(inputs), flush=True)


def load():
    m = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
    if m["code_hashes"] != hashes() or m["inputs_sha256"] != sha256(STUDY / "inputs.json") or m["human_sha256"] != sha256(ROOT / m["human_path"]):
        raise ValueError("Frozen constrained-judge source/code changed")
    return m, json.loads((STUDY / "inputs.json").read_text(encoding="utf-8"))


def run():
    m, inputs = load()
    from .__main__ import offline_environment
    from .backend import TransformersBackend
    from .binary_judge import BinaryJudge
    offline_environment()
    model = ROOT / "models" / JUDGE
    config = {"study_signature": fingerprint(m), "assets": {n: sha256(model / n) for n in ("source_manifest.json", "config.json", "tokenizer.json", "tokenizer_config.json")}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError("Judge repair configuration changed")
    write_json(path, config)
    judge = BinaryJudge(TransformersBackend(model))
    prediction_path = OUTPUT / "calibration.jsonl"
    rows = list(read_jsonl(prediction_path)) if prediction_path.exists() else []
    by_id = {c["case_id"]: c for c in inputs}
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate constrained predictions")
    for r in rows:
        if (r["case_id"] not in by_id or r["signature"] != fingerprint(config) or
                r["prompt_hash"] != fingerprint(by_id[r["case_id"]]["official_judge_prompt"]) or r["parsed_label"] not in {"yes", "no"}):
            raise ValueError("Constrained prediction provenance mismatch")
    done = {r["case_id"] for r in rows}
    with prediction_path.open("a", encoding="utf-8") as stream:
        for c in inputs:
            if c["case_id"] in done:
                continue
            result = judge.complete([{"role": "user", "content": c["official_judge_prompt"]}])
            row = {"case_id": c["case_id"], "signature": fingerprint(config), "prompt_hash": fingerprint(c["official_judge_prompt"]),
                   "raw_response": result["text"], "parsed_label": result["text"], "usage": result["usage"]}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            rows.append(row)
            print("JUDGE_REPAIR", len(rows), "/", len(inputs), flush=True)
    labels = {r["case_id"]: r for r in read_jsonl(ROOT / m["human_path"])}
    value = {"status": "completed-calibration-only", "judge": JUDGE, "signature": fingerprint(config),
             "metrics": metrics(inputs, rows, labels), "invalid_output": 0,
             "independent_validation_available": False, "formal_natural_scoring_approved": False,
             "old_audit_reused": False, "limitations": m["limitations"],
             "seconds": round(sum(r["usage"]["elapsed_seconds"] for r in rows), 1)}
    value["by_type"] = {kind: metrics([c for c in inputs if c["question_type"] == kind],
                       [r for r in rows if r["case_id"] in {c["case_id"] for c in inputs if c["question_type"] == kind}], labels)
                       for kind in sorted({c["question_type"] for c in inputs})}
    write_json(OUTPUT / "report.json", value)
    print("JUDGE_REPAIR_REPORT", json.dumps(value), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.command == "prepare" else run()
