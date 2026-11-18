"""Post-hoc label-position check, separate from the frozen v2 measurement."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .diagnostics import DIMENSIONS, parse_label, probe_messages, sha256
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/stage2_position_audit"
RUN = ROOT / "runs/research_stage2/position_audit"


def prepare():
    source = ROOT / "research/stage2/capability_probes.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    selected = [i for i in original if i["subset"] == "calibration" and int(i["id"].rsplit(":", 1)[1]) < 2]
    rows = []
    for item in selected:
        values = list(item["options"].values())
        target = item["options"][item["answer_label"]]
        for shift in range(4):
            rotated = values[shift:] + values[:shift]
            row = {**item, "original_id": item["id"], "shift": shift,
                   "id": item["id"] + f":shift{shift}", "options": dict(zip("ABCD", rotated)),
                   "answer_label": "ABCD"[rotated.index(target)]}
            rows.append(row)
    random.Random(202610252).shuffle(rows)
    if len(selected) != 48 or len(rows) != 192:
        raise ValueError("Unexpected position audit size")
    path = STUDY / "items.json"
    if path.exists():
        raise ValueError("Preserve frozen position audit")
    write_json(path, rows)
    write_json(STUDY / "manifest.json", {"version": "position-audit-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                "items_sha256": sha256(path), "source_sha256": sha256(source), "code_sha256": sha256(__file__),
                "base_questions": 48, "presentations_per_reader": 192, "representation": "raw",
                "reason": "Post-hoc: uneven correct-label accuracy observed in the completed 2B v2 run.",
                "selection": "Calibration only; first two indices in every family/difficulty cell; no selection by correctness.",
                "purpose": "Measurement diagnostic, not a replacement profile or a positive-result search."})
    print("FROZEN_POSITION_AUDIT", len(rows), flush=True)


def run_reader(tag, items):
    from .backend import TransformersBackend
    from .__main__ import offline_environment
    offline_environment()
    output = RUN / f"{tag}.jsonl"
    model = ROOT / "models" / f"Qwen3.5-{tag.upper()}"
    signature = {"items_sha256": sha256(STUDY / "items.json"), "code_sha256": sha256(__file__),
                 "backend_sha256": sha256(Path(__file__).with_name("backend.py")),
                 "reader_manifest_sha256": sha256(model / "source_manifest.json"),
                 "prompt_sha256": fingerprint(inspect.getsource(probe_messages)), "reader": model.name,
                 "max_input_tokens": 12288, "max_new_tokens": 16, "decoding": "greedy-no-thinking-bf16-sdpa"}
    sid = fingerprint(signature)
    previous = list(read_jsonl(output)) if output.exists() else []
    done = {r["id"] for r in previous}
    if len(done) != len(previous) or any(r["signature"] != sid for r in previous):
        raise ValueError("Incompatible position run")
    if not done <= {i["id"] for i in items}:
        raise ValueError("Unexpected rows")
    write_json(output.with_suffix(".config.json"), signature)
    if len(done) == len(items):
        return
    backend = TransformersBackend(model)
    with output.open("a", encoding="utf-8") as stream:
        for item in items:
            if item["id"] in done:
                continue
            result = backend.complete(probe_messages(item, "raw"), max_tokens=16)
            label = parse_label(result["text"])
            stream.write(json.dumps({"id": item["id"], "original_id": item["original_id"], "shift": item["shift"],
                         "dimension": item["dimension"], "reader": model.name, "signature": sid, "gold_label": item["answer_label"],
                         "prediction": result["text"], "parsed_label": label, "semantic_prediction": item["options"].get(label),
                         "correct": label == item["answer_label"], "usage": result["usage"]}) + "\n")
            stream.flush()
            print("POSITION_AUDIT", tag, item["id"], label == item["answer_label"], flush=True)


def report():
    lines = ["# 选项位置追加诊断", "", "这是看到 2B 的 v2 位置差异后追加的探索性检查。原题库、原画像和原检验全部保留。",
             "每个 calibration 模板/档位选前两题，共 48 个基础问题，逐题循环换位四次；只测原文。四个位置对应完全相同的基础题集合。",
             "追加检查不能变成替代画像，也不计作 192 个独立问题。", "",
             "| 读者 | A 为正确选项 | B 为正确选项 | C 为正确选项 | D 为正确选项 | 四次语义答案一致 | 四次全对 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    reports = {}
    for tag in ("2b", "4b", "9b"):
        path = RUN / f"{tag}.jsonl"
        rows = list(read_jsonl(path))
        if len(rows) != 192 or len({r["id"] for r in rows}) != 192 or len({r["signature"] for r in rows}) != 1:
            raise ValueError("Expected complete position audit")
        groups = defaultdict(list)
        for r in rows:
            groups[r["original_id"]].append(r)
        stable = sum(len({r["semantic_prediction"] for r in values}) == 1 and values[0]["semantic_prediction"] is not None
                     for values in groups.values())
        all_correct = sum(all(r["correct"] for r in values) for values in groups.values())
        counts = {label: {"n": sum(r["gold_label"] == label for r in rows),
                          "correct": sum(r["correct"] for r in rows if r["gold_label"] == label)} for label in "ABCD"}
        if len(groups) != 48 or any(c["n"] != 48 for c in counts.values()):
            raise ValueError("Unequal question sets at label positions")
        originals = {r["id"]: r for r in read_jsonl(ROOT / f"runs/research_stage2/probes_{tag}.jsonl") if r["representation"] == "raw"}
        repeats = [r for r in rows if r["shift"] == 0]
        same = sum(r["prediction"] == originals[r["original_id"]]["prediction"] for r in repeats)
        per_dimension = {d: {"n_base_questions": sum(values[0]["dimension"] == d for values in groups.values()),
                             "n_presentations": sum(r["dimension"] == d for r in rows),
                             "correct_presentations": sum(r["correct"] for r in rows if r["dimension"] == d)} for d in DIMENSIONS}
        reports[tag] = {"by_gold_position": counts, "base_questions": 48, "stable_semantic_answer": stable,
                        "all_four_correct": all_correct, "unchanged_prompt_exact_repeat": {"same": same, "n": 48},
                        "prediction_labels": dict(Counter(r["parsed_label"] for r in rows)),
                        "invalid_outputs": sum(r["parsed_label"] is None for r in rows), "dimensions": per_dimension}
        lines.append(f"| {tag.upper()} | " + " | ".join(f"{counts[l]['correct']}/48" for l in "ABCD") + f" | {stable}/48 | {all_correct}/48 |")
        lines.extend([])
    lines.extend(["", "四次语义一致表示换位后仍选择同一个答案内容，不要求都对；四次全对是更严格的指标。不同位置的记录相互相关，不作独立样本检验。",
                  "", "该结果只刻画固定 16 token、直接选字母、关闭思考协议下的行为，不能直接等同于自然对话中允许 384 token 的读取能力。后续画像须检验输出协议与位置敏感性，不能只用模型规模解释差异。", ""])
    write_json(RUN / "report.json", {"status": "completed-posthoc-diagnostic", "readers": reports})
    (ROOT / "research/POSITION_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def all_readers():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="8")
    RUN.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
    if sha256(__file__) != manifest["code_sha256"] or sha256(STUDY / "items.json") != manifest["items_sha256"]:
        raise ValueError("Frozen position audit changed")
    deadline = time.monotonic() + 6 * 3600
    while True:
        status = json.loads((ROOT / "runs/research_stage2/queue_status.json").read_text(encoding="utf-8"))
        if status["status"] == "completed":
            active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            if not active:
                break
        if time.monotonic() > deadline:
            raise TimeoutError("Waiting for stage2 completion / free GPU")
        time.sleep(15)
    items = json.loads((STUDY / "items.json").read_text(encoding="utf-8"))
    for tag in ("2b", "4b", "9b"):
        # A process per reader releases all GPU allocations before switching.
        subprocess.run([sys.executable, "-m", "reader_memory.position_audit", "reader", tag], cwd=ROOT, check=True)
    report()
    print("POSITION_AUDIT_COMPLETED", flush=True)


if __name__ == "__main__":
    command = sys.argv[1:]
    if command == ["prepare"]:
        prepare()
    elif command == ["all"]:
        all_readers()
    elif len(command) == 2 and command[0] == "reader" and command[1] in {"2b", "4b", "9b"}:
        run_reader(command[1], json.loads((STUDY / "items.json").read_text(encoding="utf-8")))
    elif command == ["detach"]:
        RUN.mkdir(parents=True, exist_ok=True)
        with (RUN / "queue.log").open("a", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-m", "reader_memory.position_audit", "all"], cwd=ROOT,
                         stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (RUN / "queue.pid").write_text(str(process.pid) + "\n")
        print("STARTED_POSITION_AUDIT", process.pid, flush=True)
    else:
        raise SystemExit("Use prepare | all | reader 2b/4b/9b | detach")
