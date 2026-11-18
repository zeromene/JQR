"""Persona-disjoint public-data development answers and genuinely blank judge material."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re

from .diagnostics import sha256
from .fidelity_pilot import short_messages
from .natural_repair import assets
from .retrieval import BM25
from .schema import Turn, fingerprint, read_jsonl, write_json
from .selection import Choice, render_evidence
from .token_windows import token_windows
from ._vendor.longmemeval_prompt import get_anscheck_prompt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/personamem_v1"
STUDY = ROOT / "research/independent_pack/v1"
OUTPUT = ROOT / "runs/independent_pack_v1"
FILES = ("independent_pack.py", "fidelity_pilot.py", "proposition_guard.py", "natural_repair.py", "token_windows.py",
         "backend.py", "experiment.py", "memory.py", "retrieval.py", "schema.py", "selection.py", "tokenization.py", "_vendor/longmemeval_prompt.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hashes():
    return {name: sha256(ROOT / "reader_memory" / name) for name in FILES}


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def parse_options(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = ast.literal_eval(text)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("Public options must be a literal list of strings")
    return value


def source_rows(data=DATA):
    data = Path(data)
    source = read(data / "source_manifest.json")
    if source["license"] != "mit" or any(sha256(data / name) != digest for name, digest in source["files_sha256"].items()):
        raise ValueError("Public dataset assets or license changed")
    with (data / "questions_32k.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    contexts = {}
    for line in read_jsonl(data / "shared_contexts_32k.jsonl"):
        if set(line) & set(contexts):
            raise ValueError("Duplicate public context ID")
        contexts.update(line)
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate public question IDs")
    owners, long_turns = defaultdict(set), defaultdict(set)
    for row in rows:
        cid, persona = row["shared_context_id"], row["persona_id"]
        if cid not in contexts or not 0 < int(row["end_index_in_shared_context"]) <= len(contexts[cid]):
            raise ValueError("Unknown context or invalid historical prefix")
        owners[cid].add(persona)
        options = parse_options(row["all_options"])
        labels = [re.match(r"^\([a-z]\)\s+", option) for option in options]
        if len(options) != 4 or any(match is None for match in labels) or sum(option.startswith(row["correct_answer"] + " ") for option in options) != 1:
            raise ValueError("Ambiguous official option/reference")
    if any(len(v) != 1 for v in owners.values()):
        raise ValueError("Shared context crosses persona groups")
    for cid, turns in contexts.items():
        for turn in turns:
            text = normalize(turn["content"])
            if len(text) >= 200:
                long_turns[fingerprint(text)].update(owners[cid])
    if any(len(v) != 1 for v in long_turns.values()):
        raise ValueError("A substantial identical message crosses persona groups")
    return source, rows, contexts, long_turns


def select_questions(rows):
    personas = sorted({r["persona_id"] for r in rows}, key=lambda p: fingerprint(["independent-judge-v1", p]))
    audit_personas = personas[:len(personas)//2]
    selected, counts = [], Counter()
    for _ in range(2):
        for persona in audit_personas:
            choices = [r for r in rows if r["persona_id"] == persona and r not in selected]
            choice = min(choices, key=lambda r: (counts[r["question_type"]], fingerprint(["independent-question-v1", r["question_id"]])))
            selected.append(choice)
            counts[choice["question_type"]] += 1
    return {"judge_audit_personas": audit_personas, "reserved_development_personas": personas[len(personas)//2:]}, selected


def prepare(counter, model, study=STUDY, data=DATA):
    study = Path(study)
    if study.exists():
        raise FileExistsError("Independent pack is frozen; use another version")
    if not counter.exact:
        raise ValueError("Use the exact reader tokenizer")
    source, rows, contexts, long_turns = source_rows(data)
    folds, selected = select_questions(rows)
    # Audit development-source messages only; sealed evaluation answers are never used.
    dev_ids = set(read(ROOT / "research/stage1/split.json")["development_ids"])
    lme = read(ROOT / "data/longmemeval/longmemeval_s_cleaned.json")
    overlap = set()
    for q in lme:
        if q["question_id"] not in dev_ids:
            continue
        for session in q["haystack_sessions"]:
            for turn in session:
                text = normalize(turn["content"])
                if len(text) >= 200 and fingerprint(text) in long_turns:
                    overlap.add(fingerprint(text))
    if overlap:
        raise ValueError("Public-data source overlaps existing development messages")
    queries, references = [], []
    for row in selected:
        qid = "personamem1_" + row["question_id"]
        end = int(row["end_index_in_shared_context"])
        prefix = contexts[row["shared_context_id"]][:end]
        turns = [Turn(f"p{row['persona_id']}:m{i:04d}", t["role"], t["content"], "") for i, t in enumerate(prefix) if t["role"] in {"user", "assistant"}]
        groups = token_windows(turns, counter, limit=384)
        ranked = BM25(groups).search(row["user_question_or_message"], 24)
        choices = []
        for group, _ in ranked:
            proposed = [*choices, Choice(group.group_id, "raw", group.variants["raw"], tuple(group.source_ids), group.date, 0, 0)]
            if counter.count(render_evidence(proposed)) <= 2048:
                choices = proposed
        evidence = render_evidence(choices)
        if not evidence:
            raise ValueError("No fitting public-data evidence")
        queries.append({"question_id": qid, "native_question_id": row["question_id"], "source_group": "personamem1-persona-" + row["persona_id"],
                        "question_type": row["question_type"], "question": row["user_question_or_message"], "date": "unknown",
                        "context_id": row["shared_context_id"], "prefix_end_index": end, "prefix_hash": fingerprint(prefix),
                        "system_messages_removed": sum(t["role"] == "system" for t in prefix), "source_turns": len(turns), "chunks": len(groups),
                        "evidence": evidence, "evidence_tokens": counter.count(evidence), "budget": 2048,
                        "selected_chunk_ids": [c.group_id for c in choices]})
        option = next(o for o in parse_options(row["all_options"]) if o.startswith(row["correct_answer"] + " "))
        references.append({"question_id": qid, "reference": option[len(row["correct_answer"]):].strip(), "official_correct_option": row["correct_answer"],
                           "reference_origin": "published multiple-choice correct response; not a human label for this generated answer"})
        print("INDEPENDENT_PREPARED", qid, flush=True)
    study.mkdir(parents=True)
    write_json(study / "queries.json", queries)
    write_json(study / "references.private.json", references)
    manifest = {"version": "independent-pack-v1", "frozen_at": datetime.now(timezone.utc).isoformat(), "code_hashes": hashes(),
                "queries_sha256": sha256(study / "queries.json"), "references_sha256": sha256(study / "references.private.json"),
                "source_manifest_sha256": sha256(Path(data) / "source_manifest.json"), "dataset_revision": source["revision"], "license": source["license"],
                "source_questions": len(rows), "source_contexts": len(contexts), "source_personas": len({r['persona_id'] for r in rows}),
                "folds": folds, "questions": len(queries), "independent_persona_groups": len(folds["judge_audit_personas"]),
                "selected_types": dict(Counter(q["question_type"] for q in queries)), "reader": "Qwen3.5-4B", "model_assets": assets(model),
                "answer_max_tokens": 384, "short_answer_prompt": "fidelity_pilot.short_messages", "normal_history_only": True,
                "source_overlap_audit": {"identical_contexts_cross_persona": 0, "identical_messages_200plus_chars_cross_persona": 0,
                                         "identical_messages_200plus_chars_with_lme_development": 0, "near_duplicate_audit_complete": False},
                "dataset_is_synthetic": True, "native_multiple_choice_benchmark_score": False,
                "official_accuracy_available": False, "human_labels_available": False, "heldout_used": False}
    write_json(study / "manifest.json", manifest)
    print("INDEPENDENT_PACK_PREPARED", json.dumps({k: manifest[k] for k in ("questions", "independent_persona_groups", "selected_types")}), flush=True)


def load(study=STUDY):
    study = Path(study)
    manifest = read(study / "manifest.json")
    if manifest["code_hashes"] != hashes() or manifest["queries_sha256"] != sha256(study / "queries.json"):
        raise ValueError("Frozen independent source/code changed")
    queries = read(study / "queries.json")
    return manifest, queries


def run(backend, counter, study=STUDY, output=OUTPUT):
    m, queries = load(study)
    if not counter.exact or backend.model_name != m["reader"] or assets(backend.path) != m["model_assets"]:
        raise ValueError("Independent reader/tokenizer changed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"study_signature": fingerprint(m), "model_assets": assets(backend.path)}
    if (output / "config.json").exists() and read(output / "config.json") != config:
        raise ValueError("Incompatible independent resume")
    write_json(output / "config.json", config)
    rows = list(read_jsonl(output / "answers.jsonl")) if (output / "answers.jsonl").exists() else []
    if rows:
        report(study, output)
    done = {r["question_id"] for r in rows}
    with (output / "answers.jsonl").open("a", encoding="utf-8") as stream:
        for q in queries:
            if q["question_id"] in done:
                continue
            if counter.count(q["evidence"]) != q["evidence_tokens"] or q["evidence_tokens"] > q["budget"]:
                raise ValueError("Independent final-string evidence budget changed")
            messages = short_messages(q["question"], q["date"], q["evidence"])
            response = backend.complete(messages, max_tokens=m["answer_max_tokens"])
            row = {"question_id": q["question_id"], "signature": fingerprint(config), "prompt_hash": fingerprint(messages),
                   "hypothesis": response["text"], "hypothesis_hash": fingerprint(response["text"]), "usage": response.get("usage", {})}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            done.add(q["question_id"])
            write_json(output / "status.json", {"status": "running", "records": len(done), "expected_records": len(queries)})
            print("INDEPENDENT_ANSWER", len(done), "/", len(queries), flush=True)
    return report(study, output)


def report(study=STUDY, output=OUTPUT):
    m, queries = load(study)
    output = Path(output)
    config = read(output / "config.json")
    if config != {"study_signature": fingerprint(m), "model_assets": m["model_assets"]}:
        raise ValueError("Independent report configuration changed")
    rows = list(read_jsonl(output / "answers.jsonl"))
    ids, qmap = [r["question_id"] for r in rows], {q["question_id"]: q for q in queries}
    if len(set(ids)) != len(ids) or not set(ids) <= set(qmap):
        raise ValueError("Duplicate or unknown independent output")
    for r in rows:
        q = qmap[r["question_id"]]
        if (r["signature"] != fingerprint(config) or r["prompt_hash"] != fingerprint(short_messages(q["question"], q["date"], q["evidence"]))
                or r["hypothesis_hash"] != fingerprint(r["hypothesis"])):
            raise ValueError("Independent output provenance changed")
    value = {"status": "completed-answer-collection" if set(ids) == set(qmap) else "incomplete-answer-collection", "records": len(rows),
             "expected_records": len(queries), "independent_persona_groups": m["independent_persona_groups"],
             "output_cap_hits": sum(r["usage"].get("completion_tokens", 0) >= m["answer_max_tokens"] for r in rows),
             "empty_answers": sum(not r["hypothesis"].strip() for r in rows), "answer_seconds": round(sum(r["usage"].get("elapsed_seconds", 0) for r in rows), 1),
             "human_labels_available": False, "official_accuracy_available": False, "native_multiple_choice_benchmark_score": False}
    write_json(output / "report.json", value)
    write_json(output / "status.json", value)
    print("INDEPENDENT_REPORT", json.dumps(value), flush=True)
    return value


def annotation(study=STUDY, output=OUTPUT):
    m, queries = load(study)
    if report(study, output)["status"] != "completed-answer-collection":
        raise ValueError("Finish all independent answers first")
    study, output = Path(study), Path(output)
    if sha256(study / "references.private.json") != m["references_sha256"]:
        raise ValueError("Official references changed")
    references = {r["question_id"]: r for r in read(study / "references.private.json")}
    qmap = {q["question_id"]: q for q in queries}
    cases, provenance = [], []
    for r in read_jsonl(output / "answers.jsonl"):
        q, ref = qmap[r["question_id"]], references[r["question_id"]]
        case_id = "independent_" + fingerprint([q["question_id"], r["hypothesis"]])[:20]
        kind = "single-session-user" if q["question_type"].startswith("recall") else "single-session-preference"
        cases.append({"case_id": case_id, "question": q["question"], "reference_answer_or_rubric": ref["reference"], "candidate_answer": r["hypothesis"],
                      "official_judge_prompt": get_anscheck_prompt(kind, q["question"], ref["reference"], r["hypothesis"])})
        provenance.append({"case_id": case_id, "source_group": q["source_group"], "question_id": q["question_id"], "question_type": q["question_type"],
                           "output_cap_hit": r["usage"].get("completion_tokens", 0) >= m["answer_max_tokens"], "origin": "normal-retrieval open response to published PersonaMem message",
                           "native_task_conversion": "multiple-choice reference used as open-response rubric; human may mark uncertain"})
    old_dir = ROOT / "runs/natural_repair_v2/fresh_annotation"
    for c in read_jsonl(old_dir / "cases.jsonl"):
        cases.append(c)
        provenance.append({"case_id": c["case_id"], "source_group": "lme-f420262-temporal", "origin": "previously unannotated v2 temporal case"})
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate independent case")
    dest = output / "annotation"
    dest.mkdir(exist_ok=True)
    for name, rows in (("cases.jsonl", cases), ("provenance.private.jsonl", provenance),
                       ("annotations.blank.jsonl", [{"case_id": c["case_id"], "annotator_id": None, "label": None, "rationale": None} for c in cases])):
        (dest / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_json(dest / "manifest.json", {"cases": len(cases), "new_public_data_cases": len(queries), "retained_unannotated_temporal_cases": len(cases)-len(queries),
               "source_groups": len({p["source_group"] for p in provenance}), "human_labels_available": False,
               "cases_sha256": sha256(dest / "cases.jsonl"), "provenance_sha256": sha256(dest / "provenance.private.jsonl"),
               "limitation": "PersonaMem synthetic histories and adapted open-response task; genuine labels required, clustered uncertainty remains"})
    from tools.build_annotation_ui import build
    build(dest / "cases.jsonl", dest / "annotate.html")
    print("INDEPENDENT_ANNOTATION_PREPARED", len(cases), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "annotation"))
    args = parser.parse_args()
    if args.command == "report":
        return report()
    if args.command == "annotation":
        return annotation()
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    model = ROOT / "models/Qwen3.5-4B"
    counter = TokenCounter(model)
    if args.command == "prepare":
        return prepare(counter, model)
    from .backend import TransformersBackend
    return run(TransformersBackend(model), counter)


if __name__ == "__main__":
    main()
