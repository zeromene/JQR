"""Import genuine human labels and validate a fixed, independent local judge.

This study creates agreement evidence only; it never promotes a judge to formal
benchmark scoring or edits human annotations. GPU inference sees official prompts
without human labels, rationales, reader names, or method names.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from .diagnostics import sha256
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "research/judge_calibration/v1"
HUMAN = PACKAGE / "human/rxx/annotations.rxx.jsonl"
STUDY = ROOT / "research/judge_validation/v1"
OUTPUT = ROOT / "runs/judge_validation_v1"
PHASES = ("judge_calibration", "judge_audit")
JUDGE = "Phi-4-mini-instruct"


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_labels(package, annotations):
    package = Path(package)
    cases = list(read_jsonl(package / "cases.jsonl"))
    provenance = list(read_jsonl(package / "provenance.private.jsonl"))
    labels = [json.loads(line) for line in Path(annotations).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = {c["case_id"] for c in cases}
    if (len(ids) != len(cases) or len({p["case_id"] for p in provenance}) != len(provenance)
            or {p["case_id"] for p in provenance} != ids
            or len(labels) != len(ids) or {r["case_id"] for r in labels} != ids):
        raise ValueError("Human labels must match every package case exactly once")
    people = {r.get("annotator_id") for r in labels}
    if len(people) != 1 or not all(isinstance(p, str) and p.strip() for p in people):
        raise ValueError("Expected one nonempty genuine annotator ID")
    for r in labels:
        if (set(r) != {"case_id", "annotator_id", "label", "rationale"}
                or r["label"] not in {"yes", "no", "uncertain"}
                or (r["rationale"] is not None and not isinstance(r["rationale"], str))
                or (r["label"] == "uncertain" and not (r["rationale"] or "").strip())):
            raise ValueError("Incomplete/invalid annotation; uncertainty requires a reason")
    by_provenance = {p["case_id"]: p for p in provenance}
    groups = defaultdict(set)
    for c in cases:
        if c["fold"] not in PHASES:
            raise ValueError("Unknown judge fold")
        groups[by_provenance[c["case_id"]]["question_id"]].add(c["fold"])
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError("Question variants cross calibration/audit folds")
    frozen = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    for phase, key in zip(PHASES, ("calibration_questions", "audit_questions")):
        actual = {p["question_id"] for p in provenance if phase in groups[p["question_id"]]}
        if actual != set(frozen[key]):
            raise ValueError("Judge fold differs from frozen package")
    return cases, by_provenance, {r["case_id"]: r for r in labels}


def import_annotations(source, package=PACKAGE, destination=HUMAN):
    source, destination, package = Path(source), Path(destination), Path(package)
    cases, provenance, labels = checked_labels(package, source)
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError("Preserve existing human labels; import revisions to a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    receipt = {"imported_at": now(), "source_filename": source.name, "sha256": sha256(destination),
               "source_claim": "Human user reported completing these labels; byte-exact browser export",
               "annotator_id": next(iter(labels.values()))["annotator_id"], "annotator_count": 1,
               "cases": len(cases), "independent_questions": len({p["question_id"] for p in provenance.values()}),
               "label_counts": dict(Counter(r["label"] for r in labels.values())),
               "by_fold": {phase: dict(Counter(labels[c["case_id"]]["label"] for c in cases if c["fold"] == phase)) for phase in PHASES},
               "package_hashes": {name: sha256(package / name) for name in ("cases.jsonl", "provenance.private.jsonl", "manifest.json")},
               "adjudicated": False, "inter_annotator_agreement_available": False, "main_benchmark_result": False}
    receipt_path = destination.with_name("receipt.json")
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["sha256"] != sha256(destination) or receipt["package_hashes"] != {name: sha256(package / name) for name in receipt["package_hashes"]}:
            raise ValueError("Existing receipt provenance mismatch")
    print("IMPORTED_HUMAN_ANNOTATIONS", json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


def freeze(package=PACKAGE, annotations=HUMAN, study=STUDY):
    package, annotations, study = Path(package), Path(annotations), Path(study)
    if study.exists():
        raise ValueError("Study exists; preserve it and use a new version")
    cases, provenance, labels = checked_labels(package, annotations)
    # Deliberately exclude human labels/rationales from GPU inference inputs.
    inputs = [{"case_id": c["case_id"], "question_id": provenance[c["case_id"]]["question_id"],
               "question_type": c["question_type"], "fold": c["fold"], "official_judge_prompt": c["official_judge_prompt"]} for c in cases]
    study.mkdir(parents=True)
    write_json(study / "inputs.json", inputs)
    manifest = {"version": "fixed-independent-judge-validation-v1", "frozen_at": now(),
                "judge": JUDGE, "max_tokens": 32, "decoding": "greedy", "prompt": "unmodified official per-case prompt",
                "parser": "entire response yes/no, case-insensitive, optional terminal period/exclamation",
                "candidate_selection": "single preset judge; no selection or prompt tuning on audit labels",
                "phase_order": list(PHASES), "inputs_sha256": sha256(study / "inputs.json"),
                "sources": [{"path": p.resolve().relative_to(ROOT).as_posix(), "sha256": sha256(p)}
                            for p in [annotations, package / "manifest.json", package / "cases.jsonl", package / "provenance.private.jsonl"]],
                "code_hashes": {name: sha256(ROOT / "reader_memory" / name)
                                for name in ("judge_validation.py", "backend.py", "schema.py", "diagnostics.py", "_vendor/longmemeval_prompt.py")},
                "cases": len(inputs), "phase_counts": dict(Counter(c["fold"] for c in inputs)),
                "independent_questions": len({c["question_id"] for c in inputs}),
                "missing_question_types": ["temporal-reasoning"], "annotator_count": 1,
                "intended_answer_reader": "Qwen3.5-4B", "natural_answers_scored": False,
                "main_benchmark_approved": False, "heldout_accessed": False,
                "limitations": ["Only ten independent oracle-development questions; audit has four groups.",
                                "No temporal-reasoning examples; normal-retrieval answers require further checks.",
                                "One human annotator; no adjudication or inter-annotator agreement.",
                                "Phi is separate from the present Qwen4B reader; future Phi answers need another independent judge."]}
    write_json(study / "manifest.json", manifest)
    print("FROZEN_JUDGE_VALIDATION", len(inputs), flush=True)
    return manifest


def load_study(study=STUDY):
    study = Path(study)
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    if sha256(study / "inputs.json") != manifest["inputs_sha256"]:
        raise ValueError("Frozen judge inputs changed")
    for item in manifest["sources"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Frozen judge source changed: " + item["path"])
    for name, digest in manifest["code_hashes"].items():
        if sha256(ROOT / "reader_memory" / name) != digest:
            raise ValueError("Frozen judge code changed: " + name)
    inputs = json.loads((study / "inputs.json").read_text(encoding="utf-8"))
    return manifest, inputs


def parse_label(text):
    match = re.fullmatch(r"\s*(yes|no)[.!]?\s*", text, re.IGNORECASE)
    return match[1].lower() if match else None


def metrics(inputs, predictions, labels, majority="yes"):
    predicted = {p["case_id"]: p for p in predictions}
    if len(predicted) != len(predictions) or set(predicted) != {c["case_id"] for c in inputs}:
        raise ValueError("Agreement requires exactly one prediction per case")
    scored = [c for c in inputs if labels[c["case_id"]]["label"] in {"yes", "no"}]
    cells = Counter((labels[c["case_id"]]["label"], predicted[c["case_id"]]["parsed_label"]) for c in scored)
    total = len(scored)
    pos, neg = sum(n for (a, _), n in cells.items() if a == "yes"), sum(n for (a, _), n in cells.items() if a == "no")
    tp, tn = cells["yes", "yes"], cells["no", "no"]
    groups = defaultdict(list)
    for c in scored:
        groups[c["question_id"]].append(labels[c["case_id"]]["label"] == predicted[c["case_id"]]["parsed_label"])
    sensitivity = tp / pos if pos else None
    specificity = tn / neg if neg else None
    return {"cases": len(inputs), "binary_human_labels": total, "uncertain_human_labels": len(inputs) - total,
            "independent_questions": len({c["question_id"] for c in inputs}), "agreements": tp + tn,
            "agreement": (tp + tn) / total if total else None,
            "question_macro_agreement": sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else None,
            "human_yes": pos, "human_no": neg, "true_yes": tp, "true_no": tn,
            "false_positive": cells["no", "yes"], "false_negative": cells["yes", "no"],
            "invalid_output": sum(p["parsed_label"] is None for p in predictions),
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_agreement": (sensitivity + specificity) / 2 if pos and neg else None,
            "calibration_majority_label": majority,
            "majority_baseline_agreement": sum(labels[c["case_id"]]["label"] == majority for c in scored) / total if total else None,
            "note": "Invalid judge outputs count as disagreement. Uncertain human labels remain explicit and unscored; repeated answers are correlated."}


def model_assets(path):
    path = Path(path)
    required = ("runtime_manifest.json", "config.json", "tokenizer.json", "tokenizer_config.json")
    return {name: sha256(path / name) for name in required}


def checked_predictions(path, expected, signature):
    rows = list(read_jsonl(path)) if Path(path).exists() else []
    by_id = {c["case_id"]: c for c in expected}
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate judge predictions")
    for row in rows:
        if (row["case_id"] not in by_id or row["signature"] != signature
                or row["prompt_hash"] != fingerprint(by_id[row["case_id"]]["official_judge_prompt"])
                or row["parsed_label"] != parse_label(row["raw_response"])):
            raise ValueError("Judge prediction provenance mismatch")
    return rows


def run(study=STUDY, output=OUTPUT, model_path=None, backend=None):
    manifest, inputs = load_study(study)
    model_path = Path(model_path or ROOT / "models" / manifest["judge"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"study_signature": fingerprint(manifest), "model": manifest["judge"], "model_assets": model_assets(model_path)}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Judge run configuration changed")
    write_json(config_path, config)
    sid = fingerprint(config)
    if backend is None:
        from .__main__ import offline_environment
        from .backend import TransformersBackend
        offline_environment()
        backend = TransformersBackend(model_path)
    if backend.model_name != manifest["judge"]:
        raise ValueError("Unexpected judge model")
    for phase in PHASES:
        if phase == "judge_audit":
            frozen = json.loads((output / "judge_frozen.json").read_text(encoding="utf-8"))
            if frozen["signature"] != sid or frozen["calibration_sha256"] != sha256(output / "judge_calibration.jsonl"):
                raise ValueError("Judge must be frozen after calibration and before audit")
        expected = [c for c in inputs if c["fold"] == phase]
        path = output / (phase + ".jsonl")
        done = {r["case_id"] for r in checked_predictions(path, expected, sid)}
        with path.open("a", encoding="utf-8") as stream:
            for case in expected:
                if case["case_id"] in done:
                    continue
                # Only the frozen official prompt reaches the model.
                response = backend.complete([{"role": "user", "content": case["official_judge_prompt"]}], max_tokens=manifest["max_tokens"])
                row = {"case_id": case["case_id"], "signature": sid, "prompt_hash": fingerprint(case["official_judge_prompt"]),
                       "raw_response": response["text"], "parsed_label": parse_label(response["text"]), "usage": response.get("usage", {})}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                done.add(case["case_id"])
                write_json(output / "status.json", {"status": "running", "phase": phase, "records": len(done), "expected_records": len(expected), "time": now()})
                print("JUDGE_VALIDATION", phase, len(done), "/", len(expected), flush=True)
        if phase == "judge_calibration":
            # The candidate/prompt are preset; calibration never searches audit results.
            write_json(output / "judge_frozen.json", {"signature": sid, "judge": manifest["judge"],
                       "calibration_sha256": sha256(path), "rule": manifest["candidate_selection"], "audit_used_for_selection": False})
    return report(study, output)


def report(study=STUDY, output=OUTPUT):
    manifest, inputs = load_study(study)
    output = Path(output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if config["study_signature"] != fingerprint(manifest):
        raise ValueError("Judge report configuration mismatch")
    frozen = json.loads((output / "judge_frozen.json").read_text(encoding="utf-8"))
    if (frozen["signature"] != fingerprint(config) or frozen["calibration_sha256"] != sha256(output / "judge_calibration.jsonl")
            or frozen["audit_used_for_selection"] is not False):
        raise ValueError("Frozen judge decision does not match calibration inputs")
    labels_path = ROOT / manifest["sources"][0]["path"]
    labels = {r["case_id"]: r for r in (json.loads(line) for line in labels_path.read_text(encoding="utf-8-sig").splitlines() if line.strip())}
    calibration_labels = [labels[c["case_id"]]["label"] for c in inputs if c["fold"] == PHASES[0] and labels[c["case_id"]]["label"] in {"yes", "no"}]
    majority = max(("yes", "no"), key=calibration_labels.count)
    value = {"status": "completed-initial-agreement-validation", "judge": manifest["judge"], "signature": fingerprint(config),
             "time": now(), "annotator_count": 1, "main_benchmark_approved": False, "natural_answers_scored": False,
             "heldout_accessed": False, "limitations": manifest["limitations"], "phases": {}}
    disagreements, all_rows = [], []
    for phase in PHASES:
        expected = [c for c in inputs if c["fold"] == phase]
        rows = checked_predictions(output / (phase + ".jsonl"), expected, fingerprint(config))
        all_rows.extend(rows)
        result = metrics(expected, rows, labels, majority)
        result["by_type"] = {kind: metrics([c for c in expected if c["question_type"] == kind],
                              [r for r in rows if r["case_id"] in {c["case_id"] for c in expected if c["question_type"] == kind}], labels, majority)
                              for kind in sorted({c["question_type"] for c in expected})}
        value["phases"][phase] = result
        by_id = {r["case_id"]: r for r in rows}
        for c in expected:
            human = labels[c["case_id"]]
            pred = by_id[c["case_id"]]
            if human["label"] != pred["parsed_label"]:
                disagreements.append({**c, "human_label": human["label"], "human_rationale": human["rationale"],
                                      "judge_label": pred["parsed_label"], "raw_response": pred["raw_response"], "review_status": "unreviewed; original human label preserved"})
    value["cost"] = {"prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in all_rows),
                     "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in all_rows),
                     "seconds": round(sum(r.get("usage", {}).get("elapsed_seconds", 0) for r in all_rows), 1),
                     "generation_cap_hits": sum(r.get("usage", {}).get("completion_tokens", 0) >= manifest["max_tokens"] for r in all_rows)}
    write_json(output / "report.json", value)
    write_json(output / "status.json", {k: v for k, v in value.items() if k != "phases"})
    (output / "disagreements.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in disagreements), encoding="utf-8")
    print("JUDGE_VALIDATION_REPORT", json.dumps(value, ensure_ascii=False), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("import", "prepare", "run", "report"))
    parser.add_argument("--annotations", default=str(HUMAN))
    parser.add_argument("--study", default=str(STUDY))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.command == "import":
        return import_annotations(args.annotations)
    if args.command == "prepare":
        return freeze(annotations=args.annotations, study=args.study)
    if args.command == "run":
        return run(args.study, args.output)
    return report(args.study, args.output)


if __name__ == "__main__":
    main()
