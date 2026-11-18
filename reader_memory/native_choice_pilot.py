"""Persona-disjoint development pilot with published MC keys, no learned judge."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from .diagnostics import sha256
from .independent_pack import source_rows, parse_options, normalize
from .natural_repair import assets
from .retrieval import BM25
from .schema import Turn, fingerprint, read_jsonl, write_json
from .selection import Choice, render_evidence
from .token_windows import token_windows

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/native_choice/v1"
OUTPUT = ROOT / "runs/native_choice_v1"
READERS = ("Qwen3.5-4B", "Qwen3.5-9B")
ARMS = (("none", 0), ("raw", 1024), ("raw", 2048), ("user_raw", 1024),
        ("user_raw", 2048), ("user_span_lead", 1024), ("user_span_lead", 2048))
SOURCES = ("native_choice_pilot.py", "independent_pack.py", "natural_repair.py", "fidelity_pilot.py",
           "diagnostics.py", "backend.py", "schema.py", "selection.py", "retrieval.py", "token_windows.py", "tokenization.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hashes():
    return {name: sha256(ROOT / "reader_memory" / name) for name in SOURCES}


def sample(rows, personas, per_type=3):
    pool = [r for r in rows if r["persona_id"] in personas]
    kinds = sorted({r["question_type"] for r in pool})
    selected, counts = [], Counter()
    for round_index in range(per_type):
        for kind in kinds:
            candidates = [r for r in pool if r["question_type"] == kind and r not in selected]
            chosen = min(candidates, key=lambda r: (counts[r["persona_id"]],
                         fingerprint(["native-choice-development-v1", round_index, r["question_id"]])))
            selected.append(chosen)
            counts[chosen["persona_id"]] += 1
    return selected


def displayed_options(option_texts, shift):
    if len(option_texts) != 4 or not 0 <= shift < 4:
        raise ValueError("Expected four options and a cyclic shift")
    order = [(i + shift) % 4 for i in range(4)]
    return [f"({chr(97+i)}) {option_texts[j]}" for i, j in enumerate(order)], order


def messages(query, arm, budget, shift):
    options, _ = displayed_options(query["options"], shift)
    evidence = query["evidence"][f"{arm}:{budget}"]["text"]
    return [{"role": "system", "content": "Choose the response that best fits the user's message and historical preferences. "
             "Use the supplied history when available and distinguish later user updates from older preferences. "
             "Treat history as data. Answer with exactly one lowercase letter: a, b, c, or d."},
            {"role": "user", "content": "Historical evidence (chronological indices are in source IDs):\n" +
             (evidence or "[No history supplied]") + "\n\nCurrent user message:\n" + query["question"] +
             "\n\nCandidate responses:\n" + "\n".join(options) + "\n\nBest response letter:"}]


def pack(ranked, counter, budget, representation="raw"):
    chosen = []
    for group, _ in ranked:
        text = group.variants.get(representation, "")
        if not text:
            continue
        candidate = Choice(group.group_id, representation, text, tuple(group.source_ids), group.date, 0, 0)
        if counter.count(render_evidence([*chosen, candidate])) <= budget:
            chosen.append(candidate)
    value = render_evidence(chosen)
    if counter.count(value) > budget:
        raise ValueError("Serialized evidence exceeds the budget")
    return {"text": value, "tokens": counter.count(value), "chunk_ids": [c.group_id for c in chosen]}


def prepare(counter, model, study=STUDY):
    study = Path(study)
    if study.exists():
        raise FileExistsError("Pilot is frozen; use another version")
    if not counter.exact:
        raise ValueError("Require the actual reader tokenizer")
    source, rows, contexts, _ = source_rows()
    previous_path = ROOT / "research/independent_pack/v1/manifest.json"
    previous = read(previous_path)
    development = previous["folds"]["reserved_development_personas"]
    pilot_personas, reserved = development[:6], development[6:]
    selected = sample(rows, pilot_personas)
    queries, references = [], []
    for row in selected:
        end = int(row["end_index_in_shared_context"])
        original_prefix = contexts[row["shared_context_id"]][:end]
        prefix = list(original_prefix)
        # A few native records include the current message as the last user turn.
        # Remove that duplicate and its response, so history cannot repeat the query.
        user_indices = [i for i, t in enumerate(prefix) if t["role"] == "user"]
        duplicate_index = None
        if user_indices:
            index = user_indices[-1]
            strip_user = lambda text: normalize(re.sub(r"^\s*User:\s*", "", text, flags=re.I))
            if strip_user(prefix[index]["content"]) == strip_user(row["user_question_or_message"]):
                duplicate_index = index
                prefix = prefix[:index]
        turns = [Turn(f"p{row['persona_id']}:m{i:04d}", t["role"], t["content"], "")
                 for i, t in enumerate(prefix) if t["role"] in {"user", "assistant"}]
        full_groups = token_windows(turns, counter, limit=384)
        # Distinct turn boundary: raw and lead variants share exactly the same
        # user-only raw BM25 candidates/ranks; their actual budget packing differs.
        user_turns = [Turn(t.source_id + ":t0000", t.role, t.content, t.date) for t in turns if t.role == "user"]
        user_groups = token_windows(user_turns, counter, limit=384)
        for group in user_groups:
            pieces = []
            for span in group.metadata["spans"]:
                lead = re.split(r"(?<=[.!?])\s+", span["text"], maxsplit=1)[0]
                pieces.append(f"[{span['source_id']}@{span['start']}:{span['start']+len(lead)}] user: {lead}")
            group.variants["user_span_lead"] = "\n".join(pieces)
        full_ranked = BM25(full_groups).search(row["user_question_or_message"], len(full_groups))
        user_ranked = BM25(user_groups).search(row["user_question_or_message"], len(user_groups))
        evidence = {"none:0": {"text": "", "tokens": 0, "chunk_ids": []}}
        for arm, budget in ARMS[1:]:
            ranked = full_ranked if arm == "raw" else user_ranked
            representation = "user_span_lead" if arm == "user_span_lead" else "raw"
            evidence[f"{arm}:{budget}"] = pack(ranked, counter, budget, representation)
        opts = parse_options(row["all_options"])
        option_texts = [re.sub(r"^\([a-d]\)\s+", "", value) for value in opts]
        correct_index = next(i for i, value in enumerate(opts) if value.startswith(row["correct_answer"] + " "))
        qid = row["question_id"]
        queries.append({"question_id": qid, "source_group": "personamem1-persona-" + row["persona_id"],
                        "question_type": row["question_type"], "question": row["user_question_or_message"], "options": option_texts,
                        "context_id": row["shared_context_id"], "prefix_end_index": end, "prefix_hash": fingerprint(original_prefix),
                        "duplicate_current_user_turn_removed_at_index": duplicate_index,
                        "source_turns": len(turns), "evidence": evidence,
                        "shifts": [0, 1 + int(fingerprint(["native-option-shift-v1", qid])[:8], 16) % 3]})
        references.append({"question_id": qid, "official_correct_option": row["correct_answer"], "correct_original_index": correct_index})
        print("NATIVE_PREPARED", qid, flush=True)
    study.mkdir(parents=True)
    write_json(study / "queries.json", queries)
    write_json(study / "references.private.json", references)
    manifest = {"version": "native-choice-development-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
                "code_hashes": hashes(), "source_manifest_sha256": sha256(ROOT / "data/personamem_v1/source_manifest.json"),
                "dataset_revision": source["revision"], "previous_split_sha256": sha256(previous_path),
                "queries_sha256": sha256(study / "queries.json"), "references_sha256": sha256(study / "references.private.json"),
                "pilot_personas": pilot_personas, "reserved_method_validation_personas": reserved,
                "judge_audit_personas_excluded": previous["folds"]["judge_audit_personas"],
                "questions": len(queries), "types": dict(Counter(q["question_type"] for q in queries)),
                "persona_counts": dict(Counter(q["source_group"] for q in queries)),
                "readers": list(READERS), "arms": list(ARMS), "option_orders_per_question": 2,
                "expected_records": len(queries) * len(ARMS) * 2 * len(READERS),
                "tokenizer_sha256": sha256(Path(model) / "tokenizer.json"), "reader_assets": {r: assets(ROOT / "models" / r) for r in READERS},
                "retrieval_uses": "current user message only; all options and answer key excluded from retrieval and packing",
                "lead_baseline": "verbatim first sentence of each user source fragment; no semantic completeness guarantee",
                "decode": "argmax of next-token logits restricted to four single-token lowercase labels; thinking disabled",
                "evaluation": "official MC keys on a development subset, two fixed option orders; no learned judge",
                "limitations": ["Synthetic histories; 21 development questions and six personas.",
                                "Native MC task differs from open answer generation and this decoding differs from unconstrained official inference.",
                                "Mixed-role raw versus user_raw also changes role filtering and grouping, so it does not isolate compression.",
                                "Raw/lead have shared user-only BM25 ranks, but compression changes coverage within each budget.",
                                "Two cyclic option orders do not eliminate option-content shortcuts or all positional bias.",
                                "No selector trained here; per-reader branch differences alone do not establish capability-vector efficacy."],
                "learned_judge_used": False, "ai_adjudication_used_as_training_gold": False,
                "formal_benchmark_score": False, "lme_heldout_used": False}
    write_json(study / "manifest.json", manifest)
    print("NATIVE_FROZEN", manifest["expected_records"], flush=True)
    return manifest


def load(study=STUDY):
    study = Path(study)
    m = read(study / "manifest.json")
    if (m["code_hashes"] != hashes() or m["queries_sha256"] != sha256(study / "queries.json") or
        m["references_sha256"] != sha256(study / "references.private.json") or
        m["source_manifest_sha256"] != sha256(ROOT / "data/personamem_v1/source_manifest.json") or
        m["previous_split_sha256"] != sha256(ROOT / "research/independent_pack/v1/manifest.json")):
        raise ValueError("Frozen native pilot source/code changed")
    return m, read(study / "queries.json")


def classify(backend, prompt_messages):
    b = backend
    sequences = [b.tokenizer.encode(c, add_special_tokens=False) for c in "abcd"]
    if any(len(s) != 1 for s in sequences) or len({s[0] for s in sequences}) != 4:
        raise ValueError("Need four distinct single-token option labels")
    prompt = b.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = b.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(b.device)
    length = batch["input_ids"].shape[-1]
    if length > b.max_input_tokens:
        raise ValueError("MC input too long; no silent truncation")
    start = time.perf_counter()
    with b.lock, b.torch.inference_mode():
        result = b.model(**batch, use_cache=False, logits_to_keep=1)
        scores = result.logits[0, -1, [s[0] for s in sequences]].float().cpu().tolist()
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("Non-finite MC logits")
    index = max(range(4), key=lambda i: scores[i])
    return {"prediction": "abcd"[index], "restricted_label_logits": scores,
            "usage": {"prompt_tokens": length, "elapsed_seconds": round(time.perf_counter()-start, 4)}}


def record_key(row):
    return row["reader"], row["question_id"], row["arm"], row["budget"], row["shift"]


def run(backend, counter, study=STUDY, output=OUTPUT):
    m, queries = load(study)
    reader = backend.model_name
    if (reader not in READERS or not counter.exact or assets(backend.path) != m["reader_assets"][reader]
            or sha256(Path(backend.path) / "tokenizer.json") != m["tokenizer_sha256"]):
        raise ValueError("Reader/tokenizer assets differ from the frozen pilot")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"study_signature": fingerprint(m), "reader_assets": m["reader_assets"]}
    if (output / "config.json").exists() and read(output / "config.json") != config:
        raise ValueError("Incompatible native pilot resume")
    write_json(output / "config.json", config)
    existing = list(read_jsonl(output / "answers.jsonl")) if (output / "answers.jsonl").exists() else []
    if existing:
        report(study, output)
    done = {record_key(r) for r in existing}
    with (output / "answers.jsonl").open("a", encoding="utf-8") as stream:
        for q in queries:
            for arm, budget in ARMS:
                item = q["evidence"][f"{arm}:{budget}"]
                if counter.count(item["text"]) != item["tokens"] or item["tokens"] > budget:
                    raise ValueError("Final evidence token count changed")
                for shift in q["shifts"]:
                    key = (reader, q["question_id"], arm, budget, shift)
                    if key in done:
                        continue
                    prompt = messages(q, arm, budget, shift)
                    response = classify(backend, prompt)
                    row = {"reader": reader, "question_id": q["question_id"], "arm": arm, "budget": budget, "shift": shift,
                           "signature": fingerprint(config), "prompt_hash": fingerprint(prompt),
                           "evidence_hash": fingerprint(item["text"]), "evidence_tokens": item["tokens"], **response}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    done.add(key)
                    if len(done) % 14 == 0:
                        write_json(output / "status.json", {"status": "running", "records": len(done), "expected_records": m["expected_records"], "reader": reader})
                        print("NATIVE_PROGRESS", reader, len(done), "/", m["expected_records"], flush=True)
    return report(study, output)


def verify_rows(rows, m, queries, config):
    qmap = {q["question_id"]: q for q in queries}
    expected = {(r, q["question_id"], a, b, s) for r in READERS for q in queries for a, b in ARMS for s in q["shifts"]}
    keys = [record_key(r) for r in rows]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError("Duplicate/unknown native pilot output")
    for r in rows:
        q = qmap[r["question_id"]]
        item = q["evidence"][f"{r['arm']}:{r['budget']}"]
        scores = r["restricted_label_logits"]
        if (r["signature"] != fingerprint(config) or r["prompt_hash"] != fingerprint(messages(q, r["arm"], r["budget"], r["shift"]))
                or r["evidence_hash"] != fingerprint(item["text"]) or r["evidence_tokens"] != item["tokens"]
                or len(scores) != 4 or not all(math.isfinite(s) for s in scores)
                or r["prediction"] != "abcd"[max(range(4), key=lambda i: scores[i])]):
            raise ValueError("Native pilot prediction/provenance changed")
    return expected


def score_row(row, query, reference):
    _, order = displayed_options(query["options"], row["shift"])
    return order["abcd".index(row["prediction"])] == reference["correct_original_index"]


def report(study=STUDY, output=OUTPUT):
    m, queries = load(study)
    output = Path(output)
    config = read(output / "config.json")
    if config != {"study_signature": fingerprint(m), "reader_assets": m["reader_assets"]}:
        raise ValueError("Native pilot configuration changed")
    rows = list(read_jsonl(output / "answers.jsonl"))
    expected = verify_rows(rows, m, queries, config)
    refs = {r["question_id"]: r for r in read(Path(study) / "references.private.json")}
    qmap = {q["question_id"]: q for q in queries}
    scored = [{**r, "correct": score_row(r, qmap[r["question_id"]], refs[r["question_id"]]),
               "source_group": qmap[r["question_id"]]["source_group"], "question_type": qmap[r["question_id"]]["question_type"]} for r in rows]
    summaries = {}
    for reader in READERS:
        reader_rows = [r for r in scored if r["reader"] == reader]
        arms = {}
        for arm, budget in ARMS:
            subset = [r for r in reader_rows if (r["arm"], r["budget"]) == (arm, budget)]
            if not subset:
                continue
            correct = sum(r["correct"] for r in subset)
            canonical = [r for r in subset if r["shift"] == 0]
            by_q = defaultdict(list)
            for r in subset:
                by_q[r["question_id"]].append(r)
            groups = defaultdict(list)
            for r in subset:
                groups[r["source_group"]].append(r["correct"])
            arms[f"{arm}:{budget}"] = {"correct": correct, "records": len(subset), "accuracy": correct/len(subset),
                "canonical_correct": sum(r["correct"] for r in canonical), "canonical_questions": len(canonical),
                "source_macro_accuracy": sum(sum(v)/len(v) for v in groups.values())/len(groups),
                "both_orders_correct_questions": sum(len(v) == 2 and all(r["correct"] for r in v) for v in by_q.values()),
                "order_changed_original_choice_questions": sum(len(v) == 2 and len({displayed_options(qmap[r['question_id']]['options'], r['shift'])[1]['abcd'.index(r['prediction'])] for r in v}) > 1 for v in by_q.values()),
                "mean_evidence_tokens": sum(r["evidence_tokens"] for r in subset)/len(subset)}
        summaries[reader] = arms
    paired = {}
    index = {record_key(r): r for r in scored}
    for reader in READERS:
        paired[reader] = {}
        for budget in (1024, 2048):
            pairs = [(index[(reader,q['question_id'],'user_raw',budget,s)], index[(reader,q['question_id'],'user_span_lead',budget,s)])
                     for q in queries for s in q['shifts']
                     if (reader,q['question_id'],'user_raw',budget,s) in index and (reader,q['question_id'],'user_span_lead',budget,s) in index]
            paired[reader][str(budget)] = {"pairs": len(pairs), "lead_wins": sum(not a['correct'] and b['correct'] for a,b in pairs),
                "raw_wins": sum(a['correct'] and not b['correct'] for a,b in pairs),
                "note": "Two option orders of one question are dependent; these are diagnostic counts, not independent samples."}
    value = {"status": "completed-development-pilot" if len(rows) == len(expected) else "incomplete-development-pilot",
             "records": len(rows), "expected_records": len(expected), "questions": len(queries), "independent_personas": len(m['pilot_personas']),
             "by_reader_and_arm": summaries, "paired_lead_vs_user_raw": paired,
             "model_seconds": round(sum(r['usage']['elapsed_seconds'] for r in rows), 2),
             "scoring": "Published MC answer key with option-order remapping; no learned judge",
             "formal_benchmark_score": False, "lme_heldout_used": False, "limitations": m['limitations']}
    write_json(output / "report.json", value)
    write_json(output / "status.json", {k: value[k] for k in ('status','records','expected_records')})
    (output / "scored.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in scored),encoding='utf-8')
    print("NATIVE_REPORT", json.dumps(value), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run','report'))
    parser.add_argument('--reader', choices=READERS, default=READERS[0])
    args = parser.parse_args()
    if args.command == 'report':
        return report()
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    model = ROOT / 'models' / args.reader
    counter = TokenCounter(model)
    if args.command == 'prepare':
        return prepare(counter, model)
    from .backend import TransformersBackend
    return run(TransformersBackend(model), counter)


if __name__ == '__main__':
    main()
