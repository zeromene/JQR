"""Exploratory learned WHOLE-PACKAGE selector; no budget or window claims."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from .diagnostics import DIMENSIONS, REPRESENTATIONS, sha256
from .open_protocol import report as validate_report
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
MODES = ("no_profile", "model_id", "overall", "vector")


def validate_folds(folds, source_items):
    names = ("inner_fit_ids", "inner_tune_ids", "train_ids", "test_ids")
    groups = {name: set(folds[name]) for name in names}
    if any(len(groups[name]) != len(folds[name]) for name in names):
        raise ValueError("Duplicate selector fold IDs")
    if (groups["inner_fit_ids"] & groups["inner_tune_ids"] or
            groups["train_ids"] & groups["test_ids"] or
            groups["inner_fit_ids"] | groups["inner_tune_ids"] != groups["train_ids"]):
        raise ValueError("Selector training/tuning/test isolation failed")
    audit = {item["id"] for item in source_items if item["subset"] == "audit"}
    if groups["train_ids"] | groups["test_ids"] != audit:
        raise ValueError("Selector folds must partition audit questions only")
    if len(folds["readers"]) != len(set(folds["readers"])) or len(folds["readers"]) < 2:
        raise ValueError("Invalid reader list")
    if not folds["ridge_grid"] or any(value <= 0 for value in folds["ridge_grid"]):
        raise ValueError("Ridge penalties must be positive")


def feature(item, representation, reader, profiles, readers, seen_readers, mode):
    """Use only pre-answer text, public operation type and calibration profiles."""
    raw, text = item["variants"]["raw"], item["variants"][representation]
    operation = np.zeros(len(DIMENSIONS))
    operation[DIMENSIONS.index(item["dimension"])] = 1
    rep = np.zeros(len(REPRESENTATIONS))
    rep[REPRESENTATIONS.index(representation)] = 1
    type_rep = np.outer(operation, rep).ravel()
    # Continuous structural features are shared by every method.
    structure = np.array([np.log1p(len(raw)) / 8, np.log1p(len(text)) / 8, len(text) / max(1, len(raw)),
                          np.log1p(len(item["question"])) / 6,
                          np.log1p(len(re.findall(r"\d+", raw))) / 5,
                          np.log1p(len(re.findall(r"\b(?:not|false|rejected|unless|NOT)\b", raw))) / 5,
                          np.log1p(raw.count("\n")) / 5])
    shared = np.r_[1., type_rep, np.outer(structure, rep).ravel()]
    if mode == "no_profile":
        return shared
    if mode == "model_id":
        # A genuinely unseen ID is always all-zero, even if its name is known.
        context = np.array([float(name == reader and reader in seen_readers) for name in readers])
    elif mode == "overall":
        context = np.array([np.mean([profiles[reader][d] for d in DIMENSIONS])])
    elif mode == "vector":
        context = np.array([profiles[reader][d] for d in DIMENSIONS])
    else:
        raise ValueError(mode)
    return np.r_[shared, np.outer(context, type_rep).ravel()]


class RidgeUtility:
    def __init__(self, mode, strength, profiles, readers, seen_readers):
        self.mode, self.strength = mode, strength
        self.profiles, self.readers, self.seen_readers = profiles, readers, set(seen_readers)

    def x(self, item, representation, reader, profile_reader=None):
        profiles = self.profiles
        if profile_reader is not None:
            profiles = {**profiles, reader: profiles[profile_reader]}
        return feature(item, representation, reader, profiles, self.readers, self.seen_readers, self.mode)

    def fit(self, items, identifiers, readers, outcomes):
        x, y = [], []
        for reader in readers:
            for identifier in identifiers:
                for rep in REPRESENTATIONS:
                    x.append(self.x(items[identifier], rep, reader))
                    y.append(float(outcomes[reader, identifier, rep]))
        x, y = np.array(x), np.array(y)
        self.center, self.scale = x.mean(axis=0), x.std(axis=0)
        self.center[0], self.scale[0] = 0, 1
        self.scale[self.scale < 1e-12] = 1
        z = (x - self.center) / self.scale
        penalty = np.eye(z.shape[1]) * self.strength
        penalty[0, 0] = 0
        self.weights = np.linalg.solve(z.T @ z + penalty, z.T @ y)
        return self

    def choose(self, item, reader, profile_reader=None):
        scores = [float(((self.x(item, rep, reader, profile_reader) - self.center) / self.scale) @ self.weights)
                  for rep in REPRESENTATIONS]
        return REPRESENTATIONS[int(np.argmax(scores))]

    def parameters(self):
        return {"mode": self.mode, "strength": self.strength, "seen_readers": sorted(self.seen_readers),
                "center": self.center.tolist(), "scale": self.scale.tolist(), "weights": self.weights.tolist()}


def tune_and_fit(mode, items, folds, train_readers, profiles, outcomes):
    choices = []
    for strength in folds["ridge_grid"]:
        model = RidgeUtility(mode, strength, profiles, folds["readers"], train_readers)
        model.fit(items, folds["inner_fit_ids"], train_readers, outcomes)
        correct = sum(outcomes[r, i, model.choose(items[i], r)] for r in train_readers for i in folds["inner_tune_ids"])
        choices.append((int(correct), strength))
    _, strength = max(choices)  # ties prefer stronger regularization
    model = RidgeUtility(mode, strength, profiles, folds["readers"], train_readers)
    model.fit(items, folds["train_ids"], train_readers, outcomes)
    return model, [{"strength": strength, "correct": correct,
                    "decisions": len(train_readers) * len(folds["inner_tune_ids"])} for correct, strength in choices]


def best_fixed(identifiers, readers, outcomes):
    scores = [sum(outcomes[r, i, rep] for r in readers for i in identifiers) for rep in REPRESENTATIONS]
    return REPRESENTATIONS[int(np.argmax(scores))]


def evaluate(items, folds, profiles, outcomes, transfer):
    readers = folds["readers"]
    predictions, artifacts = [], []
    tests = folds["test_ids"]
    outer = [[r] for r in readers] if transfer else [readers]
    for test_readers in outer:
        train_readers = [r for r in readers if r not in test_readers] if transfer else readers
        fixed = best_fixed(folds["train_ids"], train_readers, outcomes)
        by_type = {d: best_fixed([i for i in folds["train_ids"] if items[i]["dimension"] == d], train_readers, outcomes)
                   for d in DIMENSIONS}
        by_reader = {r: best_fixed(folds["train_ids"], [r], outcomes) for r in test_readers} if not transfer else {}
        models = {}
        for mode in MODES:
            model, tuning = tune_and_fit(mode, items, folds, train_readers, profiles, outcomes)
            models[mode] = model
            artifacts.append({"test_readers": test_readers, "train_readers": train_readers,
                              "model": model.parameters(), "tuning": tuning})
        for reader in test_readers:
            wrong_profile = readers[(readers.index(reader) + 1) % len(readers)]
            for identifier in tests:
                item = items[identifier]
                selected = {"global_fixed": fixed, "type_fixed": by_type[item["dimension"]]}
                if not transfer:
                    selected["reader_fixed"] = by_reader[reader]
                selected.update({mode: model.choose(item, reader) for mode, model in models.items()})
                selected["vector_wrong_profile"] = models["vector"].choose(item, reader, wrong_profile)
                for method, rep in selected.items():
                    predictions.append({"reader": reader, "id": identifier, "dimension": item["dimension"],
                                        "method": method, "representation": rep, "correct": bool(outcomes[reader, identifier, rep])})
    return predictions, artifacts


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    metrics = {}
    for method, values in groups.items():
        metrics[method] = {"n_decisions": len(values), "n_base_questions": len({r["id"] for r in values}),
                           "correct": sum(r["correct"] for r in values), "accuracy": np.mean([r["correct"] for r in values]).item(),
                           "selection_counts": dict(Counter(r["representation"] for r in values)),
                           "by_reader": {reader: {"n": sum(r["reader"] == reader for r in values),
                                                  "correct": sum(r["correct"] for r in values if r["reader"] == reader)}
                                         for reader in sorted({r["reader"] for r in values})}}
    index = {(r["reader"], r["id"], r["method"]): r for r in rows}
    per_question = defaultdict(list)
    for (reader, identifier, method), row in index.items():
        if method == "vector":
            per_question[identifier].append(int(row["correct"]) - int(index[reader, identifier, "no_profile"]["correct"]))
    differences = np.array([np.mean(values) for _, values in sorted(per_question.items())])
    rng = np.random.default_rng(20261026)
    boot = differences[rng.integers(0, len(differences), size=(2000, len(differences)))].mean(axis=1)
    return {"metrics": metrics, "primary_vector_minus_no_profile": {
                "accuracy_delta": float(differences.mean()), "question_cluster_bootstrap_95_ci": np.quantile(boot, [.025, .975]).tolist(),
                "clusters": len(differences), "draws": 2000,
                "scope": "Conditional on four fixed readers and these shared hand-written template families; exploratory."}}


def main():
    study = ROOT / "research/selector_controlled"
    folds = json.loads((study / "folds.json").read_text(encoding="utf-8"))
    data = ROOT / "research/stage3_protocol/items.json"
    if sha256(data) != folds["source_sha256"]:
        raise ValueError("Controlled selector source changed")
    implementation = json.loads((study / "implementation.json").read_text(encoding="utf-8"))
    if implementation["code_sha256"] != sha256(__file__):
        raise ValueError("Frozen selector implementation changed")
    if implementation["folds_sha256"] != sha256(study / "folds.json"):
        raise ValueError("Frozen selector folds changed")
    source_items = json.loads(data.read_text(encoding="utf-8"))
    validate_folds(folds, source_items)
    rows_by_reader, profiles, outcomes, sources = {}, {}, {}, []
    tags = ("2b", "4b", "9b", "phi")
    with tempfile.TemporaryDirectory() as temporary:
        for tag, reader in zip(tags, folds["readers"]):
            path = ROOT / f"runs/research_stage3/open_{tag}.jsonl"
            validated = Path(temporary) / f"{tag}.json"
            validate_report(data, path, validated)
            report = json.loads(validated.read_text(encoding="utf-8"))
            if report["reader"] != reader:
                raise ValueError("Unexpected reader")
            rows_by_reader[reader] = list(read_jsonl(path))
            profiles[reader] = {d: report["calibration"][d]["accuracy"] for d in DIMENSIONS}
            for row in rows_by_reader[reader]:
                if row["subset"] == "audit":
                    outcomes[reader, row["id"], row["representation"]] = bool(row["correct"])
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    # Answers and model responses are excluded from the selector feature objects.
    items = {i["id"]: {k: i[k] for k in ("id", "dimension", "question", "variants")}
             for i in source_items}
    destination = ROOT / "runs/research_stage3/selector_controlled"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "report.json").exists():
        raise FileExistsError("Preserve the previous controlled selector result")
    config = {"folds_sha256": sha256(study / "folds.json"), "code_sha256": sha256(__file__), "sources": sources,
              "profiles": profiles, "numpy_version": np.__version__, "kind": "whole-package-no-budget-synthetic-diagnostic"}
    write_json(destination / "config.json", config)
    results = {}
    for name, transfer in (("unseen_reader", True), ("seen_reader", False)):
        rows, artifacts = evaluate(items, folds, profiles, outcomes, transfer)
        results[name] = summarize(rows)
        (destination / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        write_json(destination / f"{name}.models.json", artifacts)
    write_json(destination / "report.json", {"status": "complete-exploratory-whole-package-diagnostic", "signature": fingerprint(config),
               **results, "natural_task_accuracy_claim": False, "budget_allocation_evaluated": False})
    print("CONTROLLED_SELECTOR_COMPLETED", json.dumps({name: r["primary_vector_minus_no_profile"] for name, r in results.items()}), flush=True)


if __name__ == "__main__":
    main()
