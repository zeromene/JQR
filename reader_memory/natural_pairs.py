"""Natural TRAIN-only window interventions; collect answers, not uncalibrated labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .diagnostics import REPRESENTATIONS, sha256
from .experiment import answer_messages
from .memory import build_groups, make_windows, render_turns
from .retrieval import BM25
from .schema import MemoryGroup, Turn, fingerprint, history_hash, read_jsonl, write_json
from .selection import Choice, render_evidence

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/natural_pairs/v1"
OUTPUT = ROOT / "runs/natural_pairs_v1"
BRANCHES = ("none", *REPRESENTATIONS)
DEPENDENCIES = ("backend.py", "experiment.py", "memory.py", "retrieval.py", "schema.py", "selection.py", "tokenization.py", "_vendor/simplemem_prompt.py")


def code_hashes():
    base = Path(__file__).parent
    return {name: sha256(base / name) for name in ("natural_pairs.py", *DEPENDENCIES)}


def select_ids(items, folds, count):
    """Round-robin types with deterministic ordering, at most one question/source group."""
    if count <= 0:
        raise ValueError("Pilot question count must be positive")
    train = set(folds["ids"]["selector_train"])
    groups = {qid: group["id"] for group in folds["groups"] for qid in group["question_ids"]}
    by_type = defaultdict(list)
    for item in items:
        if item["question_id"] in train:
            by_type[item["question_type"]].append(item["question_id"])
    for values in by_type.values():
        values.sort(key=lambda qid: fingerprint(["natural-pairs-v1", qid]))
    selected, used = [], set()
    while len(selected) < count:
        before = len(selected)
        for kind in sorted(by_type):
            while by_type[kind] and groups[by_type[kind][0]] in used:
                by_type[kind].pop(0)
            if by_type[kind] and len(selected) < count:
                qid = by_type[kind].pop(0)
                selected.append(qid)
                used.add(groups[qid])
        if len(selected) == before:
            raise ValueError("Not enough independent training source groups")
    return selected


def sanitized_query(item, top_k=24):
    """Discard references, original session IDs and has_answer before retrieval/writing."""
    sessions, dates = item["haystack_sessions"], item["haystack_dates"]
    if len(sessions) != len(dates):
        raise ValueError("History dates do not align")
    turns = [Turn(f"s{si:04d}:t{ti:04d}", str(message["role"]), str(message["content"]), str(dates[si]))
             for si in sorted(range(len(sessions)), key=lambda i: (str(dates[i]), i))
             for ti, message in enumerate(sessions[si])]
    groups, windows = [], {}
    for window in make_windows(turns, 8):
        gid = "m" + fingerprint([t.source_id for t in window])[:16]
        group = MemoryGroup(gid, [t.source_id for t in window], window[0].date, {"raw": render_turns(window)})
        groups.append(group)
        windows[gid] = [asdict(t) for t in window]
    ranked = BM25(groups).search(item["question"], top_k)
    return {"question_id": str(item["question_id"]), "question": item["question"], "date": str(item.get("question_date", "")),
            "history_hash": history_hash(turns), "history_turns": len(turns), "history_windows": len(groups),
            "candidates": [{"group": group.to_dict(), "turns": windows[group.group_id], "rank": i + 1, "bm25": score}
                           for i, (group, score) in enumerate(ranked)]}


def prepare(data, folds_path, outer_path, destination=STUDY, count=12):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Preserve the frozen natural study; choose a new version")
    folds = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    outer = json.loads(Path(outer_path).read_text(encoding="utf-8"))
    if folds["outer_split_sha256"] != sha256(outer_path) or outer["s_sha256"] != sha256(data):
        raise ValueError("Natural source/split changed")
    train = set(folds["ids"]["selector_train"])
    if not train <= set(outer["development_ids"]) or train & set(outer["heldout_ids"]):
        raise ValueError("Training/heldout isolation failed")
    # Parsing the source container does not use heldout references for selection.
    # Filter by the frozen training IDs before reading questions or answer fields.
    source = [item for item in json.loads(Path(data).read_text(encoding="utf-8")) if item["question_id"] in train]
    if len({item["question_id"] for item in source}) != len(source) or {item["question_id"] for item in source} != train:
        raise ValueError("Missing/duplicate training records")
    by_id = {item["question_id"]: item for item in source}
    identifiers = select_ids(source, folds, count)
    queries = [sanitized_query(by_id[qid]) for qid in identifiers]
    references = [{"question_id": qid, "answer": by_id[qid]["answer"], "question_type": by_id[qid]["question_type"]}
                  for qid in identifiers]
    write_json(destination / "queries.json", queries)
    write_json(destination / "references.private.json", references)
    manifest = {"version": "natural-window-pairs-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
                "data_sha256": sha256(data), "folds_sha256": sha256(folds_path), "outer_split_sha256": sha256(outer_path),
                "queries_sha256": sha256(destination / "queries.json"), "references_sha256": sha256(destination / "references.private.json"),
                "code_hashes": code_hashes(), "question_ids": identifiers,
                "type_counts": dict(Counter(by_id[qid]["question_type"] for qid in identifiers)),
                "independent_source_groups": len(identifiers), "question_count": len(identifiers),
                "writer": "Qwen3.5-4B", "reader": "Qwen3.5-4B", "top_k": 24, "window_turns": 8,
                "target_count_per_question": 2, "target_max_raw_tokens": 512, "budgets": [1024, 2048],
                "fact_max_tokens": 2048, "fact_format": "compact", "answer_max_tokens": 384,
                "source": "longmemeval_s_cleaned/train-only; normal BM25 candidates, no oracle selection",
                "construction": "query-blind writing of query-retrieved targets on demand; not full offline memory construction",
                "labels": "not collected until independent judge calibration or genuine human labels",
                "heldout_accessed_for_model_selection": False, "main_benchmark_result": False}
    write_json(destination / "manifest.json", manifest)
    print("PREPARED_NATURAL_PAIRS", json.dumps({k: manifest[k] for k in ("question_count", "type_counts", "independent_source_groups")}), flush=True)
    return manifest


def load_study(study):
    study = Path(study)
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    if manifest["code_hashes"] != code_hashes() or manifest["queries_sha256"] != sha256(study / "queries.json"):
        raise ValueError("Frozen natural code or queries changed")
    queries = json.loads((study / "queries.json").read_text(encoding="utf-8"))
    if [query["question_id"] for query in queries] != manifest["question_ids"]:
        raise ValueError("Natural question list changed")
    return manifest, queries


def model_assets(model_path):
    root = Path(model_path)
    if not (root / "source_manifest.json").is_file() or not (root / "config.json").is_file():
        raise ValueError("Missing model provenance")
    names = ("source_manifest.json", "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    return {name: sha256(root / name) for name in names if (root / name).is_file()}


def choice(group, representation):
    return Choice(group.group_id, representation, group.variants[representation], tuple(group.source_ids), group.date, 0, 0)


def make_plan(manifest, queries, counter, assets):
    if not counter.exact:
        raise ValueError("Natural interventions require the actual reader tokenizer")
    targets = []
    for query in queries:
        selected, costs = [], {}
        for candidate in query["candidates"]:
            group = MemoryGroup.from_dict(candidate["group"])
            cost = counter.count(render_evidence([choice(group, "raw")]))
            costs[group.group_id] = cost
            if cost <= manifest["target_max_raw_tokens"] and len(selected) < manifest["target_count_per_question"]:
                selected.append(group.group_id)
        targets.append({"question_id": query["question_id"], "target_group_ids": selected, "raw_candidate_tokens": costs,
                        "status": "targets-selected" if selected else "no-eligible-target-within-top-k"})
    return {"study_signature": fingerprint(manifest), "model_assets": assets, "reader": manifest["reader"], "targets": targets,
            "expected_max_answer_records": sum(len(t["target_group_ids"]) for t in targets) * len(manifest["budgets"]) * len(BRANCHES)}


def save_plan(study, output, counter, model_path):
    manifest, queries = load_study(study)
    plan = make_plan(manifest, queries, counter, model_assets(model_path))
    path = Path(output) / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Frozen target plan changed")
    write_json(path, plan)
    print("PLANNED_NATURAL_PAIRS", len(queries), plan["expected_max_answer_records"], flush=True)
    return plan


def context(query, target, budget, counter):
    """Keep C fixed across all branches, verifying whole serialized BPE costs."""
    if not counter.exact:
        raise ValueError("Exact reader tokens required")
    target_sources = set(target.source_ids)
    background = []

    def evidence(choices):
        return {"none": render_evidence(choices), **{rep: render_evidence([*choices, choice(target, rep)]) for rep in REPRESENTATIONS}}

    if any(counter.count(text) > budget for text in evidence([]).values()):
        return {"status": "excluded-all-branches-target-exceeds-budget", "branches": {}}
    for candidate in query["candidates"]:
        group = MemoryGroup.from_dict(candidate["group"])
        if group.group_id == target.group_id or target_sources & set(group.source_ids):
            continue
        proposed = [*background, choice(group, "raw")]
        if all(counter.count(text) <= budget for text in evidence(proposed).values()):
            background = proposed
    variants = evidence(background)
    return {"status": "ready", "background_group_ids": [c.group_id for c in background],
            "background_hash": fingerprint(variants["none"]), "background_evidence": variants["none"],
            "target_source_ids": target.source_ids, "target_group_sha256": fingerprint(target.to_dict()),
            "branches": {branch: {"evidence": text, "evidence_tokens": counter.count(text)} for branch, text in variants.items()}}


def expected_keys(plan, manifest):
    return {(t["question_id"], gid, budget, branch) for t in plan["targets"] for gid in t["target_group_ids"]
            for budget in manifest["budgets"] for branch in BRANCHES}


def append_json(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()


def run(study, output, backend, counter):
    manifest, queries = load_study(study)
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if backend.model_name != manifest["reader"] or plan != make_plan(manifest, queries, counter, model_assets(backend.path)):
        raise ValueError("Reader/target plan mismatch")
    signature = {"study_signature": fingerprint(manifest), "plan_sha256": sha256(output / "plan.json"),
                 "reader": backend.model_name, "backend_class": type(backend).__name__, "max_input_tokens": backend.max_input_tokens,
                 "answer_max_tokens": manifest["answer_max_tokens"], "decoding": "greedy-no-thinking-bf16-sdpa"}
    sid = fingerprint(signature)
    config = output / "config.json"
    if config.exists() and json.loads(config.read_text(encoding="utf-8")) != signature:
        raise ValueError("Natural run configuration changed")
    write_json(config, signature)
    path = output / "answers.jsonl"
    rows = list(read_jsonl(path)) if path.exists() else []
    keys = [(row["question_id"], row["target_group_id"], row["budget"], row["branch"]) for row in rows]
    expected = expected_keys(plan, manifest)
    if len(keys) != len(set(keys)) or not set(keys) <= expected or any(row["signature"] != sid for row in rows):
        raise ValueError("Invalid/duplicate natural answer records")
    done = set(keys)
    context_path = output / "contexts.jsonl"
    previous_contexts = list(read_jsonl(context_path)) if context_path.exists() else []
    contexts = {row["pair_id"]: row for row in previous_contexts}
    if len(contexts) != len(previous_contexts) or any(row["signature"] != sid for row in previous_contexts):
        raise ValueError("Invalid natural contexts")
    query_map = {query["question_id"]: query for query in queries}
    with path.open("a", encoding="utf-8") as stream, context_path.open("a", encoding="utf-8") as context_stream:
        for target_plan in plan["targets"]:
            qid = target_plan["question_id"]
            query = query_map[qid]
            by_group = {c["group"]["group_id"]: c for c in query["candidates"]}
            for gid in target_plan["target_group_ids"]:
                if all((qid, gid, budget, branch) in done for budget in manifest["budgets"] for branch in BRANCHES):
                    continue
                candidate = by_group[gid]
                turns = [Turn(**turn) for turn in candidate["turns"]]
                groups = build_groups(turns, backend, manifest["writer"], output / "writer_cache", window_turns=8,
                                      fact_max_tokens=manifest["fact_max_tokens"], fact_format=manifest["fact_format"])
                if len(groups) != 1 or groups[0].group_id != gid or groups[0].variants["raw"] != candidate["group"]["variants"]["raw"]:
                    raise ValueError("Target writer changed source window")
                target = groups[0]
                write_json(output / "target_memories" / f"{qid}.{gid}.json", target.to_dict())
                for budget in manifest["budgets"]:
                    pair_id = fingerprint([qid, gid, budget])
                    paired = context(query, target, budget, counter)
                    paired.update(pair_id=pair_id, question_id=qid, target_group_id=gid, budget=budget, signature=sid)
                    if paired["status"] == "ready":
                        for branch, value in paired["branches"].items():
                            prompt = answer_messages(query["question"], query["date"], value["evidence"])
                            serialized = backend.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                            value["prompt_tokens"] = len(backend.tokenizer.encode(serialized, add_special_tokens=False))
                            value["prompt_hash"] = fingerprint(prompt)
                        if any(v["prompt_tokens"] > backend.max_input_tokens for v in paired["branches"].values()):
                            paired["status"] = "excluded-all-branches-prompt-limit"
                    if pair_id in contexts:
                        if contexts[pair_id] != paired:
                            raise ValueError("Paired background changed during resume")
                    else:
                        append_json(context_stream, paired)
                        contexts[pair_id] = paired
                    for branch in BRANCHES:
                        if (qid, gid, budget, branch) in done:
                            continue
                        value = paired["branches"].get(branch, {})
                        row = {"signature": sid, "pair_id": pair_id, "context_hash": fingerprint(paired), "question_id": qid,
                               "target_group_id": gid, "budget": budget, "branch": branch, "reader": backend.model_name,
                               "background_hash": paired.get("background_hash"), "evidence_tokens": value.get("evidence_tokens"),
                               "prompt_tokens": value.get("prompt_tokens"), "score_status": "awaiting-calibrated-independent-judge-or-human-labels",
                               "natural_training_pilot_only": True}
                        start = time.perf_counter()
                        if paired["status"] == "ready":
                            response = backend.complete(answer_messages(query["question"], query["date"], value["evidence"]), max_tokens=384)
                            row.update(status="answered", hypothesis=response["text"], usage=response.get("usage", {}))
                        else:
                            row.update(status=paired["status"], hypothesis="", usage={})
                        row["seconds"] = round(time.perf_counter() - start, 4)
                        append_json(stream, row)
                        done.add((qid, gid, budget, branch))
                        write_json(output / "status.json", {"status": "running", "records": len(done), "expected_records": len(expected),
                                   "question_id": qid, "score_status": row["score_status"]})
                        print("NATURAL_PAIR", qid, gid, budget, branch, row["status"], flush=True)
    return report(study, output)


def report(study, output):
    manifest, queries = load_study(study)
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    contexts = list(read_jsonl(output / "contexts.jsonl")) if (output / "contexts.jsonl").exists() else []
    by_context = {row["pair_id"]: row for row in contexts}
    rows = list(read_jsonl(output / "answers.jsonl")) if (output / "answers.jsonl").exists() else []
    expected = expected_keys(plan, manifest)
    keys = [(row["question_id"], row["target_group_id"], row["budget"], row["branch"]) for row in rows]
    sid = fingerprint(config)
    if (plan["study_signature"] != fingerprint(manifest) or config["study_signature"] != fingerprint(manifest) or
            config["plan_sha256"] != sha256(output / "plan.json") or len(keys) != len(set(keys)) or not set(keys) <= expected or
            len(contexts) != len(by_context)):
        raise ValueError("Invalid natural run provenance/records")
    counts = Counter()
    for row in rows:
        paired = by_context[row["pair_id"]]
        value = paired["branches"].get(row["branch"], {})
        if (row["pair_id"] != fingerprint([row["question_id"], row["target_group_id"], row["budget"]]) or
                any(row[key] != paired[key] for key in ("question_id", "target_group_id", "budget")) or
                row["reader"] != manifest["reader"] or row["signature"] != sid or paired["signature"] != sid or row["context_hash"] != fingerprint(paired) or
                row["background_hash"] != paired.get("background_hash") or row["evidence_tokens"] != value.get("evidence_tokens") or
                row["prompt_tokens"] != value.get("prompt_tokens") or
                (paired["status"] == "ready" and (row["status"] != "answered" or row["evidence_tokens"] > row["budget"] or
                                                   paired["background_hash"] != fingerprint(paired["background_evidence"]))) or
                (paired["status"] != "ready" and row["status"] != paired["status"])):
            raise ValueError("Natural pair context/budget mismatch")
        counts[row["status"]] += 1
    complete = set(keys) == expected
    value = {"status": "completed-answer-collection" if complete else "incomplete-answer-collection", "records": len(rows),
             "expected_records": len(expected), "questions": len(queries), "questions_with_targets": sum(bool(t["target_group_ids"]) for t in plan["targets"]),
             "status_counts": dict(counts), "paired_interventions": len(contexts), "budgets": manifest["budgets"],
             "generation_cap_hits": sum(row.get("usage", {}).get("completion_tokens", 0) >= 384 for row in rows),
             "prompt_tokens": sum(row.get("usage", {}).get("prompt_tokens", 0) for row in rows),
             "completion_tokens": sum(row.get("usage", {}).get("completion_tokens", 0) for row in rows),
             "seconds_answer_calls": round(sum(row["seconds"] for row in rows), 1),
             "score_status": "awaiting-calibrated-independent-judge-or-human-labels", "utility_labels_available": False,
             "effectiveness_evaluated": False, "heldout_used": False, "construction": manifest["construction"]}
    write_json(output / "report.json", value)
    write_json(output / "status.json", value)
    print("NATURAL_PAIR_REPORT", json.dumps(value), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "plan", "run", "report"])
    parser.add_argument("--study", default=str(STUDY))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--model-path", default=str(ROOT / "models/Qwen3.5-4B"))
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(ROOT / "data/longmemeval/longmemeval_s_cleaned.json", ROOT / "research/development/folds.json",
                       ROOT / "research/stage1/split.json", args.study)
    if args.command == "report":
        return report(args.study, args.output)
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    counter = TokenCounter(args.model_path)
    if args.command == "plan":
        return save_plan(args.study, args.output, counter, args.model_path)
    from .backend import TransformersBackend
    return run(args.study, args.output, TransformersBackend(args.model_path), counter)


if __name__ == "__main__":
    main()
