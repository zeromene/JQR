"""Versioned, solver-checked synthetic measurement; no benchmark claims."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import inspect
import itertools
import json
import math
from pathlib import Path
import random
import time

from .diagnostics import DIMENSIONS, REPRESENTATIONS, parse_label, probe_messages, sha256, wilson
from .schema import fingerprint, read_jsonl, write_json

VERSION = "capability-battery-v2"
FAMILIES = {
    "negation": {"calibration": ("direct_status", "claim_verification"),
                 "audit": ("rejected_claim", "negative_lookup")},
    "condition": {"calibration": ("all_required", "either_route"),
                  "audit": ("unless_veto", "alternate_with_veto")},
    "temporal": {"calibration": ("effective_changes", "profile_updates"),
                 "audit": ("dated_intervals", "recorded_vs_effective")},
    "numeric": {"calibration": ("balance", "shopping"),
                "audit": ("inventory", "travel")},
}
PEOPLE = ("Mira", "Leon", "Nora", "Owen", "Ravi", "Tara", "Iris", "Hugo")
PLACES = ("Oslo", "Kyoto", "Lima", "Perth", "Bern", "Dakar", "Turin", "Busan")
FIELDS = ("badge", "training", "booking", "clearance", "payment", "invitation", "insurance")


def policy_value(family, bits):
    if family == "all_required":
        return all(bits)
    if family == "either_route":
        return (bits[0] or bits[1]) and all(bits[2:])
    if family == "unless_veto":
        return bits[0] and not bits[1] and all(bits[2:])
    if family == "alternate_with_veto":
        return ((bits[0] and bits[1]) or bits[2]) and not bits[3] and all(bits[4:])
    raise ValueError(family)


def solve(spec):
    """Reference solver over the public semantic specification, never prompted."""
    kind = spec["kind"]
    if kind == "negation":
        matches = [person for person, values in spec["states"].items()
                   if values[spec["property"]] == spec["wanted"]]
        answer, = matches
    elif kind == "condition":
        answer, = [person for person, bits in spec["states"].items() if policy_value(spec["family"], bits)]
    elif kind == "temporal":
        cutoff = spec["cutoff"]
        if spec["family"] == "dated_intervals":
            answer, = [e["value"] for e in spec["events"] if e["start"] <= cutoff < e["end"]]
        else:
            eligible = [e for e in spec["events"] if e["field"] == "city" and e["effective"] <= cutoff]
            answer = max(eligible, key=lambda e: e["effective"])["value"]
    elif spec["family"] in {"balance", "inventory"}:
        answer = spec["initial"] + sum(e["change"] for e in spec["events"] if e["completed"])
    elif spec["family"] == "shopping":
        answer = sum(e["quantity"] * e["unit_price"] for e in spec["items"]) - spec["discount"]
    elif spec["family"] == "travel":
        answer = sum(e["rides"] * e["fare"] for e in spec["legs"]) + spec["fee"] - spec["refund"]
    else:
        raise ValueError(spec)
    return str(answer)


def variants(raw, spec, summary):
    # Structured facts serialize all semantic fields, including the query fields.
    # They contain no solved answer, gold label or derived intermediate results.
    return {"raw": raw, "fact": json.dumps(spec, ensure_ascii=False, separators=(",", ":")), "summary": summary}


def negation(rng, family, level):
    people = rng.sample(PEOPLE, 4)
    properties = rng.sample(["approved", "registered", "active", "trained", "insured"], (1, 3, 5)[level - 1])
    prop = rng.choice(properties)
    wanted = family != "negative_lookup"
    target = rng.choice(people)
    states = {p: {key: bool(rng.getrandbits(1)) for key in properties} for p in people}
    for p in people:
        states[p][prop] = wanted if p == target else not wanted
    spec = {"kind": "negation", "family": family, "states": states, "property": prop, "wanted": wanted}
    records = [(p, key, states[p][key]) for p in people for key in properties]
    rng.shuffle(records)
    raw, short = [], []
    for p, key, value in records:
        if family in {"direct_status", "negative_lookup"}:
            raw.append(f"{p} is {'not ' if not value else ''}{key}.")
            short.append(f"{p}:{key}={'yes' if value else 'no'}")
        else:
            negative = bool(rng.getrandbits(1))
            claim = f"{p} is {'not ' if negative else ''}{key}"
            truth = value != negative
            if family == "claim_verification":
                raw.append(f"Claim: '{claim}'. Verification: the claim is {'true' if truth else 'false'}.")
                short.append(f"'{claim}' verified {'true' if truth else 'false'}")
            else:
                raw.append(f"The reviewer {'accepted as true' if truth else 'rejected as false'} the claim '{claim}'.")
                short.append(f"'{claim}' {'accepted=true' if truth else 'rejected=false'}")
    question = f"Which person is {'not ' if not wanted else ''}{prop}?"
    return spec, variants("All checks are authoritative and concern the same time.\n" + "\n".join(raw), spec,
                          "Authoritative checks at the same time: " + "; ".join(short) + "."), question, people


def condition(rng, family, level):
    sizes = {"all_required": (2, 3, 5), "either_route": (3, 4, 6),
             "unless_veto": (2, 4, 6), "alternate_with_veto": (4, 5, 7)}
    fields = list(FIELDS[:sizes[family][level - 1]])
    all_states = list(itertools.product((False, True), repeat=len(fields)))
    good = [v for v in all_states if policy_value(family, v)]
    bad = [v for v in all_states if not policy_value(family, v)]
    vectors = [rng.choice(good), *rng.choices(bad, k=3)]
    rng.shuffle(vectors)
    people = rng.sample(PEOPLE, 4)
    states = dict(zip(people, vectors))
    f = fields
    if family == "all_required":
        rule = " AND ".join(f)
    elif family == "either_route":
        rule = f"({f[0]} OR {f[1]}) AND " + " AND ".join(f[2:])
    elif family == "unless_veto":
        rule = f"{f[0]} AND NOT {f[1]}" + (" AND " + " AND ".join(f[2:]) if f[2:] else "")
    else:
        rule = f"(({f[0]} AND {f[1]}) OR {f[2]}) AND NOT {f[3]}" + (" AND " + " AND ".join(f[4:]) if f[4:] else "")
    prefix = f"Permission holds if and only if {rule}. All statuses below are current."
    sentences = [f"{p} has {key}." if value else f"{p} does not have {key}."
                 for p, bits in states.items() for key, value in zip(fields, bits)]
    rng.shuffle(sentences)
    spec = {"kind": "condition", "family": family, "rule": rule, "fields": fields, "states": states}
    summary = prefix + " " + "; ".join(f"{p}: " + ",".join(f"{key}={'yes' if v else 'no'}" for key, v in zip(fields, bits))
                                         for p, bits in states.items())
    return spec, variants(prefix + "\n" + "\n".join(sentences), spec, summary), "Who has permission?", people


def temporal(rng, family, level):
    n = (4, 7, 11)[level - 1]
    person = rng.choice(PEOPLE)
    start = date(2026, 1, 1)
    events = []
    for i in range(n):
        effective = start + timedelta(days=i * 5)
        value = rng.choice(PLACES)
        if family == "dated_intervals":
            events.append({"start": effective.isoformat(), "end": (effective + timedelta(days=5)).isoformat(), "value": value})
        else:
            event = {"field": "city", "effective": effective.isoformat(), "value": value}
            if family == "recorded_vs_effective":
                event["recorded"] = (effective + timedelta(days=rng.randint(1, 50))).isoformat()
            events.append(event)
            if family == "profile_updates":
                events.append({"field": "hobby", "effective": effective.isoformat(), "value": rng.choice(["chess", "hiking", "painting"])})
    cutoff = start + timedelta(days=5 * rng.randrange(n) + 2)
    rng.shuffle(events)
    spec = {"kind": "temporal", "family": family, "person": person, "events": events, "cutoff": cutoff.isoformat()}
    prefix = f"This is {person}'s authoritative history. The entries are unsorted. "
    if family == "dated_intervals":
        prefix += "Each stay includes the start date and excludes the end date."
        rows = [f"From {e['start']} inclusive until {e['end']} exclusive, {person} lived in {e['value']}." for e in events]
        compact = [f"[{e['start']},{e['end']}):{e['value']}" for e in events]
    else:
        prefix += "Use effective dates to determine the state, not record order or recording dates."
        rows = [f"Effective {e['effective']}, {person}'s {e['field']} became {e['value']}." +
                (f" This was recorded on {e['recorded']}." if "recorded" in e else "") for e in events]
        compact = [f"{e['field']}={e['value']} effective={e['effective']}" +
                   (f" recorded={e['recorded']}" if "recorded" in e else "") for e in events]
    target = solve(spec)
    options = [target, *rng.sample([p for p in PLACES if p != target], 3)]
    return spec, variants(prefix + "\n" + "\n".join(rows), spec, prefix + " " + "; ".join(compact)), \
        f"Which city did {person} live in on {cutoff.isoformat()}?", options


def numeric(rng, family, level):
    n = (1, 3, 6)[level - 1]
    spec = {"kind": "numeric", "family": family}
    if family in {"balance", "inventory"}:
        spec["initial"] = rng.randint(50, 150)
        spec["events"] = [{"change": rng.choice([-1, 1]) * rng.randint(2, 20), "completed": True} for _ in range(n)]
        spec["events"].append({"change": rng.randint(2, 20), "completed": False})
        rng.shuffle(spec["events"])
        unit = "credits" if family == "balance" else "units"
        prefix = f"The initial {'account balance' if family == 'balance' else 'stock'} was {spec['initial']} {unit}. Signed totals may be negative."
        rows = [f"Event {i + 1}: {'received' if e['change'] > 0 else 'removed'} {abs(e['change'])} {unit}; " +
                ("completed." if e["completed"] else "planned only, not completed.") for i, e in enumerate(spec["events"])]
        compact = prefix + " Changes: " + "; ".join(f"{e['change']:+d} {'completed' if e['completed'] else 'pending'}" for e in spec["events"])
        question = "What is the final total after completed events only?"
    elif family == "shopping":
        spec["items"] = [{"quantity": rng.randint(1, 4), "unit_price": rng.randint(3, 20)} for _ in range(n)]
        spec["discount"] = rng.randint(1, min(5, sum(e["quantity"] * e["unit_price"] for e in spec["items"]) - 1))
        prefix = "All prices are in whole credits. The listed items were all purchased."
        rows = [f"Item {i + 1}: bought {e['quantity']} units at {e['unit_price']} credits per unit." for i, e in enumerate(spec["items"])]
        rows.append(f"One fixed discount of {spec['discount']} credits applies to the entire order. No other fees.")
        compact = "Purchased quantity x unit price: " + "; ".join(f"{e['quantity']} x {e['unit_price']}" for e in spec["items"])
        compact += f". One order-wide discount: {spec['discount']} credits; no fees."
        question = "How many credits were paid in total?"
    else:
        spec["legs"] = [{"rides": rng.choice([1, 2]), "fare": rng.randint(3, 15)} for _ in range(n)]
        spec["fee"], spec["refund"] = rng.randint(1, 5), rng.randint(1, 5)
        prefix = "All journeys below were completed. Each fare is for one ride; a return journey uses two rides."
        rows = [f"Journey {i + 1}: {'one-way (one ride)' if e['rides'] == 1 else 'return (two rides)'}, {e['fare']} credits per ride."
                for i, e in enumerate(spec["legs"])]
        rows.append(f"Add one booking fee of {spec['fee']} credits for the entire trip, then subtract one refund of {spec['refund']} credits.")
        compact = "Completed rides x fare: " + "; ".join(f"{e['rides']} x {e['fare']}" for e in spec["legs"])
        compact += f". One trip-wide fee +{spec['fee']}; one refund -{spec['refund']} credits."
        question = "What is the net total cost in credits?"
    target = int(solve(spec))
    options = [str(target), *[str(target + offset) for offset in rng.sample([i for i in range(-12, 13) if i], 3)]]
    return spec, variants(prefix + "\n" + "\n".join(rows), spec, compact), question, options


def make_battery(per_cell=8, seed=202610251):
    if per_cell < 4 or per_cell % 4:
        raise ValueError("per_cell must be a positive multiple of four, at least four")
    generators = {"negation": negation, "condition": condition, "temporal": temporal, "numeric": numeric}
    items = []
    for dimension in DIMENSIONS:
        for subset in ("calibration", "audit"):
            for family in FAMILIES[dimension][subset]:
                for level in (1, 2, 3):
                    for index in range(per_cell):
                        identifier = f"{subset}:{dimension}:{family}:L{level}:{index:03d}"
                        rng = random.Random(f"{seed}:{identifier}")
                        spec, representations, question, options = generators[dimension](rng, family, level)
                        target = solve(spec)
                        options = [target, *[v for v in options if v != target]]
                        if len(options) != 4 or len(set(options)) != 4:
                            raise ValueError("Expected four distinct options")
                        rng.shuffle(options)
                        position = index % 4
                        current = options.index(target)
                        options[current], options[position] = options[position], options[current]
                        items.append({"id": identifier, "subset": subset, "dimension": dimension, "family": family,
                                      "difficulty": level, "question": question, "options": dict(zip("ABCD", options)),
                                      "answer_label": "ABCD"[position], "variants": representations, "semantic_spec": spec,
                                      "generation": VERSION, "seed": seed, "fidelity": "deterministic-rendering-not-human-validated"})
    random.Random(seed).shuffle(items)
    validate_battery(items)
    return items


def validate_battery(items):
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs")
    for item in items:
        if item["family"] not in FAMILIES[item["dimension"]][item["subset"]]:
            raise ValueError("Unexpected family/subset assignment")
        if json.loads(item["variants"]["fact"]) != json.loads(json.dumps(item["semantic_spec"])):
            raise ValueError("Structured facts differ from semantic specification")
        if item["options"][item["answer_label"]] != solve(item["semantic_spec"]):
            raise ValueError("Gold does not match reference solver")
        if len(set(item["options"].values())) != 4:
            raise ValueError("Options not unique")
    for dimension in DIMENSIONS:
        a = {i["family"] for i in items if i["dimension"] == dimension and i["subset"] == "calibration"}
        b = {i["family"] for i in items if i["dimension"] == dimension and i["subset"] == "audit"}
        if a & b:
            raise ValueError("Calibration/audit template overlap")


def prepare(directory, per_cell=8):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    items = make_battery(per_cell)
    data = root / "capability_probes.json"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["data_sha256"] != sha256(data) or fingerprint(items) != fingerprint(json.loads(data.read_text(encoding="utf-8"))):
            raise ValueError("Frozen battery changed; create a new study version")
        print("FROZEN_DATA_UNCHANGED", len(items))
        return
    if data.exists():
        raise ValueError("Unmanifested battery exists; preserve it and use a new directory")
    write_json(data, items)
    write_json(manifest_path, {"version": VERSION, "data_sha256": sha256(data), "items": len(items),
                              "expected_rows_per_reader": len(items) * 3, "per_family_difficulty_subset": per_cell,
                              "families": FAMILIES, "generator_sha256": sha256(__file__),
                              "frozen_at": datetime.now(timezone.utc).isoformat(),
                              "status": "frozen-before-v2-inference", "external_data_used": False})
    print("FROZEN", len(items), sha256(data))


def run(data, output, backend):
    items = json.loads(Path(data).read_text(encoding="utf-8"))
    validate_battery(items)
    signature = {"version": VERSION, "data_sha256": sha256(data), "reader": backend.model_name,
                 "reader_manifest_sha256": sha256(Path(backend.path) / "source_manifest.json"),
                 "max_input_tokens": backend.max_input_tokens, "max_new_tokens": 16,
                 "decoding": "greedy-no-thinking-bf16-sdpa", "representations": REPRESENTATIONS,
                 "prompt_sha256": fingerprint(inspect.getsource(probe_messages)),
                 "runner_sha256": sha256(__file__), "backend_sha256": sha256(Path(__file__).with_name("backend.py"))}
    signature_id = fingerprint(signature)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = path.with_suffix(".config.json")
    if config.exists() and fingerprint(json.loads(config.read_text(encoding="utf-8"))) != signature_id:
        raise ValueError("Run settings changed; use a new output")
    previous = list(read_jsonl(path)) if path.exists() else []
    done = {(r["id"], r["representation"]) for r in previous}
    if len(done) != len(previous) or any(r["signature"] != signature_id for r in previous):
        raise ValueError("Duplicate or incompatible existing rows")
    expected = {(i["id"], r) for i in items for r in REPRESENTATIONS}
    if not done <= expected:
        raise ValueError("Unexpected existing rows")
    write_json(config, signature)
    with path.open("a", encoding="utf-8") as stream:
        for item in items:
            # Counterbalance representation order, fixed across readers.
            order = list(REPRESENTATIONS)
            random.Random(item["id"]).shuffle(order)
            for representation in order:
                if (item["id"], representation) in done:
                    continue
                start = time.perf_counter()
                response = backend.complete(probe_messages(item, representation), max_tokens=16)
                parsed = parse_label(response["text"])
                row = {key: item[key] for key in ("id", "subset", "dimension", "family", "difficulty")}
                row.update(signature=signature_id, reader=backend.model_name, representation=representation,
                           prediction=response["text"], parsed_label=parsed, gold_label=item["answer_label"],
                           correct=parsed == item["answer_label"], usage=response.get("usage", {}),
                           seconds=round(time.perf_counter() - start, 4), synthetic_development_only=True)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print("V2_PROBE", backend.model_name, item["id"], representation, row["correct"], flush=True)
    write_json(path.with_suffix(".status.json"), {"status": "completed", "rows": len(expected),
                                                "signature": signature_id, "final_test_accessed": False})


def cell(values):
    n = len(values)
    k = sum(r["correct"] for r in values)
    return {"n": n, "correct": k, "accuracy": k / n if n else None, "wilson_95_ci": wilson(k, n),
            "invalid_outputs": sum(r["parsed_label"] is None for r in values)}


def holm(pvalues):
    adjusted, running = [None] * len(pvalues), 0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, min(1, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def report(data, predictions, output):
    items = json.loads(Path(data).read_text(encoding="utf-8"))
    validate_battery(items)
    rows = list(read_jsonl(predictions))
    index = {(r["id"], r["representation"]): r for r in rows}
    expected = {(i["id"], r) for i in items for r in REPRESENTATIONS}
    if len(index) != len(rows) or set(index) != expected or len({r["signature"] for r in rows}) != 1:
        raise ValueError("A complete, unique, single-signature battery is required")
    config = json.loads(Path(predictions).with_suffix(".config.json").read_text(encoding="utf-8"))
    if config["data_sha256"] != sha256(data) or any(r["signature"] != fingerprint(config) for r in rows):
        raise ValueError("Configuration/data signature mismatch")
    for item in items:
        for rep in REPRESENTATIONS:
            row = index[item["id"], rep]
            if (row["gold_label"] != item["answer_label"] or row["parsed_label"] != parse_label(row["prediction"])
                    or row["correct"] != (row["parsed_label"] == item["answer_label"])
                    or row["reader"] != config["reader"]
                    or any(row[k] != item[k] for k in ("subset", "dimension", "family", "difficulty"))):
                raise ValueError("Prediction metadata or scoring mismatch")
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["subset"], r["dimension"], r["representation"]].append(r)
    cells = [{"subset": s, "dimension": d, "representation": r, **cell(v)} for (s, d, r), v in sorted(grouped.items())]
    details = []
    for keys in ("difficulty", "family"):
        groups = defaultdict(list)
        for r in rows:
            groups[r["subset"], r["dimension"], r["representation"], r[keys]].append(r)
        details.extend({"subset": s, "dimension": d, "representation": r, keys: value, **cell(v)}
                       for (s, d, r, value), v in sorted(groups.items()))
    pairs = []
    for dimension in DIMENSIONS:
        ids = [i["id"] for i in items if i["subset"] == "audit" and i["dimension"] == dimension]
        for rep in ("fact", "summary"):
            a = sum(index[i, "raw"]["correct"] and not index[i, rep]["correct"] for i in ids)
            b = sum(index[i, rep]["correct"] and not index[i, "raw"]["correct"] for i in ids)
            n = a + b
            p = min(1, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n) if n else 1.0
            pairs.append({"dimension": dimension, "comparison": f"{rep}-raw", "n": len(ids),
                          "raw_only_correct": a, "other_only_correct": b, "accuracy_delta": (b - a) / len(ids),
                          "mcnemar_exact_p": p})
    for pair, p in zip(pairs, holm([p["mcnemar_exact_p"] for p in pairs])):
        pair["holm_p_within_reader_8_comparisons"] = p
    profile, stability = {}, {}
    for dimension in DIMENSIONS:
        values = grouped["calibration", dimension, "raw"]
        estimate = cell(values)
        profile[dimension] = estimate["accuracy"]
        # Blocks of four contain A/B/C/D once; never split on label parity.
        halves = [cell([r for r in values if (int(r["id"].rsplit(":", 1)[1]) // 4) % 2 == parity]) for parity in (0, 1)]
        families = {f: cell([r for r in values if r["family"] == f]) for f in FAMILIES[dimension]["calibration"]}
        flags = []
        if estimate["accuracy"] >= .95:
            flags.append("possible-ceiling")
        if estimate["accuracy"] <= .35:
            flags.append("near-chance-or-floor")
        if any(h["n"] == 0 for h in halves):
            flags.append("insufficient-balanced-halves")
        elif abs(halves[0]["accuracy"] - halves[1]["accuracy"]) > .25:
            flags.append("split-half-sensitive")
        if max(v["accuracy"] for v in families.values()) - min(v["accuracy"] for v in families.values()) > .25:
            flags.append("template-sensitive")
        stability[dimension] = {**estimate, "deterministic_split_halves": halves, "families": families, "flags": flags}
    write_json(output, {"version": VERSION, "reader": config["reader"], "rows": len(rows), "cells": cells,
                        "details": details, "audit_pairs": pairs, "calibration_stability": stability,
                        "status": "complete-synthetic-measurement-not-validated-ability-benchmark",
                        "limitations": ["Four hand-written families per operation; dimensions can be correlated.",
                                        "Difficulty levels describe task structure, not empirically calibrated difficulty.",
                                        "Wilson intervals and McNemar tests treat generated instances as units, not independent template families.",
                                        "Split halves are descriptive, not a test-retest reliability estimate.",
                                        "No natural-dialogue or unseen-family transfer conclusion follows from these probes."]})
    write_json(Path(output).with_suffix(".profile.json"), {"reader": config["reader"], "abilities": profile,
                "status": "synthetic-calibration-only-provisional", "uncertainty": stability,
                "provenance": {"run_signature": fingerprint(config), "data_sha256": sha256(data),
                               "subset": "calibration", "representation": "raw", "source": str(predictions)}})
    print("REPORTED", config["reader"], json.dumps(profile), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--directory", default="research/stage2")
    p.add_argument("--per-cell", type=int, default=8)
    r = sub.add_parser("run")
    r.add_argument("--data", default="research/stage2/capability_probes.json")
    r.add_argument("--model-path", required=True)
    r.add_argument("--output", required=True)
    q = sub.add_parser("report")
    q.add_argument("--data", default="research/stage2/capability_probes.json")
    q.add_argument("--predictions", required=True)
    q.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args.directory, args.per_cell)
    if args.command == "report":
        return report(args.data, args.predictions, args.output)
    from .__main__ import offline_environment
    from .backend import TransformersBackend
    offline_environment()
    return run(args.data, args.output, TransformersBackend(args.model_path))


if __name__ == "__main__":
    main()
