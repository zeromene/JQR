"""Validate live protocol snapshots and report only completed experiments."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .diagnostics import DIMENSIONS, sha256
from .open_protocol import canonical, expected_keys, parse_answer, report as protocol_report
from .schema import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
READERS = (("2b", "Qwen3.5-2B"), ("4b", "Qwen3.5-4B"), ("9b", "Qwen3.5-9B"),
           ("phi", "Phi-4-mini-instruct"))


def validate_snapshot(items, rows, config, data_sha256):
    expected = expected_keys(items)
    keys = [(row["id"], row["representation"]) for row in rows]
    if (len(keys) != len(set(keys)) or not set(keys) <= expected or
            config["data_sha256"] != data_sha256):
        raise ValueError("Invalid or duplicate protocol snapshot")
    by_id = {item["id"]: item for item in items}
    signature = fingerprint(config)
    for row in rows:
        item = by_id[row["id"]]
        if (row["signature"] != signature or row["reader"] != config["reader"] or
                row["parsed_answer"] != parse_answer(row["prediction"]) or
                row["gold_answer"] != canonical(item["answer"]) or
                row["correct"] != (row["parsed_answer"] == canonical(item["answer"])) or
                any(row[k] != item[k] for k in ("subset", "dimension", "family", "difficulty"))):
            raise ValueError("Protocol snapshot score/signature mismatch")
    return {"rows": len(rows), "expected_rows": len(expected), "complete": set(keys) == expected,
            "invalid_outputs": sum(row["parsed_answer"] is None for row in rows),
            "cap_hits": sum(row.get("usage", {}).get("completion_tokens", 0) >= 384 for row in rows),
            "seconds": round(sum(row["seconds"] for row in rows), 1)}


def build(root=ROOT):
    root = Path(root)
    data = root / "research/stage3_protocol/items.json"
    items = json.loads(data.read_text(encoding="utf-8"))
    manifest = json.loads(data.with_name("manifest.json").read_text(encoding="utf-8"))
    if manifest["data_sha256"] != sha256(data):
        raise ValueError("Protocol source mismatch")
    run = root / "runs/research_stage3"
    progress, reports = {}, {}
    for tag, reader in READERS:
        path = run / f"open_{tag}.jsonl"
        if not path.exists():
            progress[reader] = {"rows": 0, "expected_rows": len(expected_keys(items)), "complete": False}
            continue
        config = json.loads(path.with_suffix(".config.json").read_text(encoding="utf-8"))
        if config["reader"] != reader or config["code_sha256"] != manifest["code_sha256"]:
            raise ValueError("Wrong protocol reader or implementation")
        # A live append may leave an unfinished final record. Only consume lines
        # ending in a newline; completed-run validation still requires all keys.
        content = path.read_bytes()
        content = content[:content.rfind(b"\n") + 1] if content else content
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        progress[reader] = validate_snapshot(items, rows, config, sha256(data))
        if progress[reader]["complete"]:
            destination = path.with_suffix(".report.json")
            protocol_report(data, path, destination)
            reports[reader] = json.loads(destination.read_text(encoding="utf-8"))

    now = datetime.now(timezone.utc).isoformat()
    value = {"updated_at": now, "readers": progress, "protocol_complete": len(reports) == len(READERS)}
    text = ["# 开放答案与学习型表示选择进度", "", f"更新时间：{now}。", "",
            "本轮是合成开发诊断。复用旧题、去掉选项并将生成上限改为 384 token；协议变化的收益不能记为记忆方法收益。",
            "不包含自然长期记忆主成绩、窗口边际效用或预算分配验证。未完成的读者只报告进度，不报告其局部准确率。", "",
            "| 读者 | 完整记录数 | 状态 | 无效输出 | 触及生成上限 |",
            "| --- | ---: | --- | ---: | ---: |"]
    for reader, status in progress.items():
        text.append(f"| {reader} | {status['rows']}/{status['expected_rows']} | "
                    f"{'完整' if status['complete'] else '尚未完成'} | {status.get('invalid_outputs', '—')} | {status.get('cap_hits', '—')} |")
    if reports:
        text += ["", "## 已完成读者的独立 calibration", "",
                 "每维 48 题，画像只使用原文。数值分数提高也只能说明联合回答协议发生变化。", "",
                 "| 读者 | 否定 | 条件 | 时间 | 数值 |", "| --- | ---: | ---: | ---: | ---: |"]
        for reader, report in reports.items():
            text.append("| " + reader + " | " + " | ".join(f"{report['calibration'][d]['correct']}/48" for d in DIMENSIONS) + " |")
        text += ["", "## 已完成读者的 audit 表示对照", "", "每操作/表示 48 题，无证据预算。", "",
                 "| 读者 | 操作 | 原文 | 事实 | 摘要 |", "| --- | --- | ---: | ---: | ---: |"]
        for reader, report in reports.items():
            index = {(c["dimension"], c["representation"]): c for c in report["cells"] if c["subset"] == "audit"}
            for d in DIMENSIONS:
                text.append(f"| {reader} | {d} | " + " | ".join(f"{index[d, rep]['correct']}/48" for rep in ("raw", "fact", "summary")) + " |")
            pairs = [p for p in report["audit_pairs"] if p["holm_p_within_reader"] < .05]
            text += ["", f"{reader}：单读者八比较 Holm p<0.05 的项目：" +
                     ("；".join(f"{p['dimension']} {p['comparison']} Δ={p['accuracy_delta']*100:+.1f}pp, p={p['holm_p_within_reader']:.4g}" for p in pairs) or "无") + "。", ""]
        text += ["该校正不覆盖跨读者或后续分析，也不是能力与表示的交互检验。"]

    selector_path = run / "selector_controlled/report.json"
    if selector_path.exists():
        if len(reports) != len(READERS):
            raise ValueError("Selector result exists before complete protocol inputs")
        selector = json.loads(selector_path.read_text(encoding="utf-8"))
        config = json.loads(selector_path.with_name("config.json").read_text(encoding="utf-8"))
        if (selector["signature"] != fingerprint(config) or
                config["code_sha256"] != sha256(root / "reader_memory/controlled_selector.py") or
                config["folds_sha256"] != sha256(root / "research/selector_controlled/folds.json") or
                any(sha256(root / source["path"]) != source["sha256"] for source in config["sources"])):
            raise ValueError("Selector result provenance mismatch")
        value["selector_complete"] = True
        for name, title in (("unseen_reader", "留一读者与留出问题（主要比较）"), ("seen_reader", "熟悉读者与留出问题（次要比较）")):
            result = selector[name]
            text += ["", f"## {title}", "", "| 方法 | 正确/决策数 | 准确率 |", "| --- | ---: | ---: |"]
            for method, metric in result["metrics"].items():
                text.append(f"| {method} | {metric['correct']}/{metric['n_decisions']} | {metric['accuracy']*100:.2f}% |")
            delta = result["primary_vector_minus_no_profile"]
            lo, hi = delta["question_cluster_bootstrap_95_ci"]
            text += ["", f"能力向量减无画像：{delta['accuracy_delta']*100:+.2f} 个百分点，按基础问题聚类的 95% bootstrap 区间 [{lo*100:+.2f}, {hi*100:+.2f}]。",
                     "每个读者 96 道测试题；四位读者共享这些题，统计聚类数为 96。区间仅描述这些固定读者与模板，不支持普遍迁移。"]
        primary = selector["unseen_reader"]["primary_vector_minus_no_profile"]
        text += ["", "当前解释：" + ("主要比较的区间高于零，出现能力画像的额外收益信号；仍须核对总体能力与类型条件策略，并在自然任务验证。"
                 if primary["question_cluster_bootstrap_95_ci"][0] > 0 else
                 "主要比较尚未支持能力画像具有稳定的额外收益；不能宣称核心假设成立，也不根据本测试集再调特征或正则强度。")]
    else:
        value["selector_complete"] = False
        text += ["", "学习型选择比较尚未完成。代码及 72/24/96 基础问题划分已冻结；四位读者输入全部完整并验证后才执行。",
                 "比较固定、类型条件、无画像、模型 ID、总体能力和四维能力向量，并保留错配画像检查；不以测试表现修改预设方法。"]
    text += ["", "设计与来源：[协议](stage3_protocol/PLAN.md)、[选择器设计](selector_controlled/PLAN.md)、[实现冻结记录](selector_controlled/implementation.json)。",
             "自然任务仍需真实人工判分校准、正常检索背景下的配对效用监督及近期系统基线；440 道留出未用于本轮。", ""]
    write_json(run / "progress_report.json", value)
    (root / "research/STAGE3_REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print("STAGE3_REPORT", json.dumps(value), flush=True)
    return value


if __name__ == "__main__":
    build()
