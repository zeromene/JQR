"""Frozen training-only six-way comparison with source-unit guards and short answers."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from . import natural_repair as prior
from .diagnostics import sha256
from .experiment import answer_messages
from .proposition_guard import guarded_representation
from .schema import MemoryGroup, fingerprint, read_jsonl, write_json
from .selection import Choice, render_evidence

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/fidelity_pilot/v3"
OUTPUT = ROOT / "runs/fidelity_pilot_v3"
BRANCHES = (*prior.BRANCHES, "unit_guarded")
SHORT_ANSWER = ("Return only the final answer, at most 90 words. Do not repeat the question, narrate your search, "
                "or provide a chain of reasoning. Include every requested item needed for a complete answer. "
                "For recommendations, give at most three specific options with brief preference-based reasons. "
                "If evidence is insufficient, state the specific missing information briefly.")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hashes():
    return {name: sha256(ROOT / "reader_memory" / name) for name in (*prior.DEPENDENCIES, "fidelity_pilot.py", "proposition_guard.py")}


def short_messages(question, date, evidence):
    messages = answer_messages(question, date, evidence)
    messages[0]["content"] += "\n" + SHORT_ANSWER
    return messages


def paired_context(query, target, budget, counter):
    def choose(g, name):
        return Choice(g.group_id, name, g.variants[name], tuple(g.source_ids), g.date, 0, 0)
    def variants(background):
        return {"none": render_evidence(background), **{name: render_evidence([*background, choose(target, name)]) for name in BRANCHES[1:]}}
    if any(counter.count(s) > budget for s in variants([]).values()):
        return {"status": "excluded-all-branches-target-budget", "branches": {}}
    background = []
    for candidate in query["candidates"]:
        group = MemoryGroup.from_dict(candidate["group"])
        if group.group_id == target.group_id:
            continue
        proposed = [*background, choose(group, "raw")]
        if all(counter.count(s) <= budget for s in variants(proposed).values()):
            background = proposed
    texts = variants(background)
    return {"status": "ready", "background_group_ids": [c.group_id for c in background],
            "background_hash": fingerprint(texts["none"]), "branches": {name: {"evidence": s, "evidence_tokens": counter.count(s)} for name, s in texts.items()}}


def prepare(counter, model, study=STUDY):
    study = Path(study)
    if study.exists():
        raise FileExistsError("Preserve the frozen fidelity pilot; use another version")
    if not counter.exact:
        raise ValueError("Exact reader tokenizer required")
    old_manifest, old_queries = prior.load(ROOT / "research/natural_repair/v2")
    old_output = ROOT / "runs/natural_repair_v2"
    if prior.report(ROOT / "research/natural_repair/v2", old_output)["status"] != "completed-answer-collection":
        raise ValueError("Complete and verify the prior collection first")
    queries = [q for q in old_queries if q["purpose"] == "training-pilot"]
    folds = read(ROOT / "research/development/folds.json")
    if not {q["question_id"] for q in queries} <= set(folds["ids"]["selector_train"]):
        raise ValueError("Fidelity pilot must remain training-only")
    targets, contexts, prior_hashes = {}, [], {}
    cache_groups = [read(p)["group"] for p in (old_output / "writer_cache").glob("*.json") if not p.name.endswith(".responses.json")]
    for query in queries:
        qid = query["question_id"]
        target_file = old_output / "target_memories" / (qid + ".json")
        value = read(target_file)
        if value not in cache_groups or value["variants"]["raw"] != query["candidates"][0]["group"]["variants"]["raw"]:
            raise ValueError("Prior target differs from its writer cache or raw retrieval input")
        prior_hashes[qid] = sha256(target_file)
        target = MemoryGroup.from_dict(value)
        target.variants["unit_guarded"], target.metadata["unit_guard"] = guarded_representation(target, counter)
        targets[qid] = target.to_dict()
        for budget in query["budgets"]:
            paired = paired_context(query, target, budget, counter)
            old_pair = read(old_output / "contexts" / (fingerprint([qid, budget]) + ".json"))
            paired.update(question_id=qid, budget=budget, pair_id=fingerprint([qid, budget]),
                          background_same_as_v2=paired.get("background_hash") == old_pair.get("background_hash"))
            contexts.append(paired)
    study.mkdir(parents=True)
    for name, value in (("queries.json", queries), ("targets.json", targets), ("contexts.json", contexts)):
        write_json(study / name, value)
    manifest = {"version": "fidelity-pilot-v3", "frozen_at": datetime.now(timezone.utc).isoformat(), "code_hashes": hashes(),
                "input_hashes": {n: sha256(study / n) for n in ("queries.json", "targets.json", "contexts.json")},
                "prior_manifest_signature": fingerprint(old_manifest), "prior_target_hashes": prior_hashes,
                "model_assets": prior.assets(model), "reader": "Qwen3.5-4B", "branches": list(BRANCHES), "answer_max_tokens": 384,
                "answer_word_instruction": 90, "short_answer_instruction": SHORT_ANSWER, "expected_records": len(contexts) * len(BRANCHES),
                "training_questions": len(queries), "background_pairs_changed_from_v2": sum(not c["background_same_as_v2"] for c in contexts),
                "raw_fallback_targets": sum(g["metadata"]["unit_guard"]["raw_fallback"] for g in targets.values()),
                "heldout_used": False, "writer_calls": 0, "official_accuracy_available": False,
                "scope": "training development; cached writer controls; not an independent efficacy test"}
    write_json(study / "manifest.json", manifest)
    print("FIDELITY_PREPARED", json.dumps({k: manifest[k] for k in ("expected_records", "training_questions", "raw_fallback_targets", "background_pairs_changed_from_v2")}), flush=True)


def load(study=STUDY):
    study = Path(study)
    manifest = read(study / "manifest.json")
    if manifest["code_hashes"] != hashes() or any(sha256(study / n) != digest for n, digest in manifest["input_hashes"].items()):
        raise ValueError("Frozen fidelity code or inputs changed")
    return manifest, read(study / "queries.json"), read(study / "contexts.json")


def run(backend, counter, study=STUDY, output=OUTPUT):
    m, queries, contexts = load(study)
    if not counter.exact or backend.model_name != m["reader"] or prior.assets(backend.path) != m["model_assets"]:
        raise ValueError("Fidelity reader/tokenizer assets changed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"study_signature": fingerprint(m), "model_assets": prior.assets(backend.path)}
    if (output / "config.json").exists() and read(output / "config.json") != config:
        raise ValueError("Incompatible fidelity resume")
    write_json(output / "config.json", config)
    old = list(read_jsonl(output / "answers.jsonl")) if (output / "answers.jsonl").exists() else []
    if old:
        report(study, output)
    done = {(r["pair_id"], r["branch"]) for r in old}
    qmap = {q["question_id"]: q for q in queries}
    with (output / "answers.jsonl").open("a", encoding="utf-8") as stream:
        for context in contexts:
            q = qmap[context["question_id"]]
            for branch in BRANCHES:
                key = (context["pair_id"], branch)
                if key in done:
                    continue
                value = context["branches"].get(branch, {})
                messages = short_messages(q["question"], q["date"], value.get("evidence", ""))
                row = {"pair_id": key[0], "branch": branch, "question_id": q["question_id"], "budget": context["budget"],
                       "signature": fingerprint(config), "context_hash": fingerprint(context), "prompt_hash": fingerprint(messages),
                       "background_hash": context.get("background_hash"), "status": context["status"], "hypothesis": "", "usage": {},
                       "evidence_tokens": value.get("evidence_tokens"), "output_cap_hit": False}
                if context["status"] == "ready":
                    if counter.count(value["evidence"]) != value["evidence_tokens"] or value["evidence_tokens"] > context["budget"]:
                        raise ValueError("Final fidelity evidence budget changed")
                    response = backend.complete(messages, max_tokens=m["answer_max_tokens"])
                    row.update(status="answered", hypothesis=response["text"], usage=response.get("usage", {}),
                               output_cap_hit=response.get("usage", {}).get("completion_tokens", 0) >= m["answer_max_tokens"])
                row["hypothesis_hash"] = fingerprint(row["hypothesis"])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                done.add(key)
                write_json(output / "status.json", {"status": "running", "records": len(done), "expected_records": m["expected_records"]})
                print("FIDELITY_ANSWER", len(done), "/", m["expected_records"], q["question_id"], context["budget"], branch, flush=True)
    return report(study, output)


def report(study=STUDY, output=OUTPUT):
    m, queries, contexts = load(study)
    output = Path(output)
    config = read(output / "config.json")
    if config != {"study_signature": fingerprint(m), "model_assets": m["model_assets"]}:
        raise ValueError("Fidelity report configuration mismatch")
    rows = list(read_jsonl(output / "answers.jsonl"))
    cmap, qmap = {c["pair_id"]: c for c in contexts}, {q["question_id"]: q for q in queries}
    expected = {(c["pair_id"], b) for c in contexts for b in BRANCHES}
    keys = [(r["pair_id"], r["branch"]) for r in rows]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError("Duplicate or unknown fidelity output")
    for r in rows:
        c, q = cmap[r["pair_id"]], qmap[r["question_id"]]
        b = c["branches"].get(r["branch"], {})
        prompt = short_messages(q["question"], q["date"], b.get("evidence", ""))
        if (r["signature"] != fingerprint(config) or r["context_hash"] != fingerprint(c) or r["prompt_hash"] != fingerprint(prompt)
                or r["hypothesis_hash"] != fingerprint(r["hypothesis"]) or r["background_hash"] != c.get("background_hash")
                or r["budget"] != c["budget"] or r["question_id"] != c["question_id"] or r["evidence_tokens"] != b.get("evidence_tokens")
                or r["output_cap_hit"] != (r["usage"].get("completion_tokens", 0) >= m["answer_max_tokens"])):
            raise ValueError("Fidelity output provenance changed")
        if r["status"] == "answered" and (c["status"] != "ready" or r["evidence_tokens"] > r["budget"]):
            raise ValueError("Fidelity status/budget mismatch")
        if r["status"] != "answered" and r["status"] != c["status"]:
            raise ValueError("Invalid fidelity exclusion")
    marginal = {b: [c["branches"][b]["evidence_tokens"] - c["branches"]["none"]["evidence_tokens"] for c in contexts if c["status"] == "ready"] for b in BRANCHES[1:]}
    result = {"status": "completed-answer-collection" if set(keys) == expected else "incomplete-answer-collection", "records": len(rows),
              "expected_records": m["expected_records"], "statuses": dict(Counter(r["status"] for r in rows)),
              "output_cap_hits": sum(r["output_cap_hit"] for r in rows), "over_90_words": sum(len(r["hypothesis"].split()) > 90 for r in rows),
              "empty_answers": sum(r["status"] == "answered" and not r["hypothesis"].strip() for r in rows),
              "answer_seconds": round(sum(r["usage"].get("elapsed_seconds", 0) for r in rows), 1),
              "raw_fallback_targets": m["raw_fallback_targets"], "training_questions": m["training_questions"],
              "background_pairs_changed_from_v2": m["background_pairs_changed_from_v2"], "writer_calls": 0,
              "mean_marginal_evidence_tokens": {b: round(statistics.mean(v), 1) for b, v in marginal.items() if v},
              "official_accuracy_available": False, "heldout_used": False}
    write_json(output / "report.json", result)
    write_json(output / "status.json", result)
    print("FIDELITY_REPORT", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    args = parser.parse_args()
    if args.command == "report":
        return report()
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
