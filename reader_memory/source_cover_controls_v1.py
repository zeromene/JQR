"""Separate predicted source choice from ordinary one-sentence-per-source packing."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import statistics

from . import bridge_identity_controls_v1 as parent
from . import anchor_guard_controls_v1 as guard
from .schema import fingerprint, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/source_cover_controls_v1'
RUN = ROOT / 'runs/source_cover_controls_v1'
load, sha, immutable, recomp = parent.load, parent.sha, parent.immutable, parent.recomp
packing = guard.packing
BASE, FULL, INDEPENDENT = parent.BASE, parent.FULL, parent.INDEPENDENT
COVER_PATH, COVER_INDEPENDENT = 'path_source_cover', 'independent_source_cover'
METHODS = (BASE, FULL, INDEPENDENT, COVER_PATH, COVER_INDEPENDENT)
NEW_METHODS = (COVER_PATH, COVER_INDEPENDENT)
PAIRS = ((COVER_PATH, FULL), (COVER_INDEPENDENT, INDEPENDENT),
         (COVER_PATH, COVER_INDEPENDENT), (FULL, INDEPENDENT),
         (FULL, BASE), (COVER_PATH, BASE), (COVER_INDEPENDENT, BASE))
CONFIG = dict(version='source-cover-controls-v1', questions=128, sources=64, budget=256,
    methods=list(METHODS), max_conditions=640, max_new_reader_calls=256,
    reader='Qwen3.5-9B', reader_variant='format_only', output_tokens=2048,
    decoding='greedy-thinking_false', retries=0,
    source_choice='unchanged first blind Beam path versus independent max-Qwen sources; same predicted count',
    scores='unchanged Qwen3-Reranker-4B FP32 scores, public title plus original sentence',
    protected_anchor='highest-ranked complete sentence actually retained by original global baseline',
    cover='atomically reserve anchor and the highest-ranked complete sentence in every predicted source',
    fallback='predicted count one or joint reserve over target: exact corresponding original context',
    fill='scan all sentences by common Qwen order; accept full rendered unions within global actual token target',
    presentation='best retained Qwen rank per source, original sentence order within source',
    min_group_mean_length_ratio=0.95, primary='source choice by packing two-by-two control at 256',
    training=False, new_retrieval=False, new_encoder_calls=0, learned_dependency_protection=False,
    ordinary_coverage_control=True, independent_confirmation=False,
    protected_final_data_accessed=False, automatic_followup=False)


def arm(method):
    if method not in METHODS:
        raise ValueError('Unregistered source coverage method')
    return method + '__b256'


def cover_plan(source, scores, seed_ids, baseline, previous, counter, method):
    """Public, deterministic atomic reservation; never inspect labels or reader outcomes."""
    if method not in NEW_METHODS or not counter.exact:
        raise ValueError('Invalid method or inexact counter')
    order = packing.sentence_order(source, scores)
    by = {p['source_id']: p for p in source['paragraphs']}
    if not seed_ids or len(seed_ids) != len(set(seed_ids)) or not set(seed_ids) <= set(by):
        raise ValueError('Invalid public predicted sources')
    target = baseline['evidence_tokens']
    if not 0 < target <= 256 or previous['evidence_tokens'] > target:
        raise ValueError('Invalid original actual-token target')
    retained = set(baseline['selected_spans'])
    anchor = next((u for u in order if u in retained), None)
    if anchor is None or anchor not in previous['selected_spans']:
        raise ValueError('Missing original protected anchor')
    best = [next(u for u in order if u.split(':')[0] == pid) for pid in seed_ids]
    reserved = list(dict.fromkeys([anchor, *best]))
    selected = {}
    for uid in reserved:
        pid, _, index = uid.split(':')
        span = by[pid]['spans'][int(index)]
        selected[pid] = recomp.merge_ranges(by[pid]['text'],
            selected.get(pid, []) + [[span['start'], span['end']]])
    reserve_tokens = counter.count(recomp.render_ranges(source,
        packing.canonical_ranges(source, scores, selected)))
    reason = ('single_predicted_source' if len(seed_ids) == 1 else
              'joint_reserve_over_target' if reserve_tokens > target else 'joint_reserve_fits')
    trace = dict(seed_ids=list(seed_ids), predicted_source_count=len(seed_ids), anchor=anchor,
        reserved_spans=reserved, reserve_tokens=reserve_tokens, target_tokens=target,
        reason=reason, joint_reserve_feasible=reserve_tokens <= target,
        original_sources_kept=sorted(set(seed_ids) & set(previous['selected_ranges'])))
    fill_trace = []
    if reason != 'joint_reserve_fits':
        result = deepcopy(previous)
        result.update(method=method, arm=arm(method), evidence_budget=256)
    else:
        for uid in order:
            pid, _, index = uid.split(':')
            span = by[pid]['spans'][int(index)]
            proposal = deepcopy(selected)
            proposal[pid] = recomp.merge_ranges(by[pid]['text'],
                proposal.get(pid, []) + [[span['start'], span['end']]])
            if proposal == selected:
                continue
            n = counter.count(recomp.render_ranges(source,
                packing.canonical_ranges(source, scores, proposal)))
            fill_trace.append(dict(uid=uid, proposed_tokens=n, accepted=n <= target))
            if n <= target:
                selected = proposal
        result = packing.make_plan(source, scores, selected, counter, arm(method), target)
        result.update(method=method, evidence_budget=256)
        if not set(reserved) <= set(result['selected_spans']):
            raise ValueError('Atomic source reservation was lost')
    if anchor not in result['selected_spans'] or result['evidence_tokens'] > target:
        raise ValueError('Anchor or shared token target violated')
    trace.update(actual_tokens=result['evidence_tokens'],
        selected_sources_kept=sorted(set(seed_ids) & set(result['selected_ranges'])),
        changed=result['evidence_text'] != previous['evidence_text'], fill_trace=fill_trace)
    return result, trace


def plans_for(source, scores, blind_path, baseline, full, independent, counter):
    old = [baseline, full, independent]
    if any(p['question_id'] != source['question_id'] or p['method'] != m or
           p['arm'] != arm(m) or p['evidence_budget'] != 256
           for p, m in zip(old, METHODS[:3])):
        raise ValueError('Historical control identity mismatch')
    independent_ids = packing.independent_seeds(source, scores, len(blind_path))
    path, pt = cover_plan(source, scores, blind_path, baseline, full, counter, COVER_PATH)
    indep, it = cover_plan(source, scores, independent_ids, baseline, independent, counter, COVER_INDEPENDENT)
    return [deepcopy(p) for p in old] + [path, indep], [
        dict(question_id=source['question_id'], method=COVER_PATH, **pt),
        dict(question_id=source['question_id'], method=COVER_INDEPENDENT, **it)]


def length_statistics(plans):
    rows = []
    by = {(p['question_id'], p['method']): p for p in plans}
    for scope in ('overall', 'multi', 'single'):
        for method in METHODS:
            group = [p for p in plans if p['method'] == method and (scope == 'overall' or p['mode'] == scope)]
            gaps = [by[p['question_id'], BASE]['evidence_tokens'] - p['evidence_tokens'] for p in group]
            mean_n = statistics.mean(p['evidence_tokens'] for p in group)
            mean_base = statistics.mean(by[p['question_id'], BASE]['evidence_tokens'] for p in group)
            rows.append(dict(method=method, scope=scope, n=len(group), mean_tokens=mean_n,
                mean_ratio=mean_n / mean_base, max_gap_tokens=max(gaps),
                within_12_tokens=sum(abs(g) <= 12 for g in gaps),
                passed=all(g >= 0 for g in gaps) and mean_n / mean_base >= 0.95))
    return rows


def replay_cache():
    marker = load(PLAN / 'preparation.json')['reader_marker_sha256']
    result = {}
    for item in load(PLAN / 'private/replay_cache.json'):
        recomp.interface.parent.check_cache_entry(item, marker)
        if item['key'] in result:
            raise ValueError('Duplicate historical request')
        result[item['key']] = item['entry']
    return result


def freeze():
    prep = load(PLAN / 'preparation.json')
    names = ['reader_memory/source_cover_controls_v1.py', 'tools/prepare_source_cover_controls_v1.py',
        'scripts/source_cover_controls_v1_queue.py', 'tests/test_source_cover_controls_v1.py',
        'tools/status_source_cover_controls_v1.py', 'research/bridge_identity_controls_v1/manifest.json',
        *prep['server_source_files']]
    names += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
        and 'snapshots' not in p.relative_to(PLAN).parts and p.suffix != '.pyc'
        and p.name not in {'manifest.json', 'CURRENT_STATUS.md', 'submission_receipt.json',
                          'STATUS_LATEST.json', 'SNAPSHOT_LATEST.json'}]
    value = dict(config=CONFIG, parent_signature=parent.verify(),
        files={n: sha(ROOT / n) for n in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_SOURCE_COVER', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Source coverage experiment changed')
    if parent.verify() != value['parent_signature']:
        raise ValueError('Frozen parent changed')
    for name, digest in value['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return value['signature']


def preflight():
    sid = verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    prep = load(PLAN / 'preparation.json')
    marker = prep['reader_marker_sha256']
    if sha(ROOT / 'models/Qwen3.5-9B/.verified.json') != marker:
        raise ValueError('Reader model changed')
    sources, blind = load(PLAN / 'public/source_inputs.json'), load(PLAN / 'public/blind_paths.json')
    plans = load(PLAN / 'public/qa_plans.json')
    by = {(p['question_id'], p['arm']): p for p in plans}
    expected = {(s['question_id'], arm(m)) for s in sources for m in METHODS}
    if len(sources) != 128 or len({s['native_id'] for s in sources}) != 64 or len(plans) != 640 or set(by) != expected:
        raise ValueError('Incomplete fixed panel')
    counter, traces = ExactCounter(), []
    for s in sources:
        q = s['question_id']
        cached = load(ROOT / 'runs/reranker_budget_v1/selected' / (q.replace(':', '_') + '.json'))
        if cached['source_sha256'] != fingerprint(s) or blind[q]['source_sha256'] != fingerprint(s):
            raise ValueError('Source/path/score provenance changed')
        if blind[q]['predicted_hops'] != len(blind[q]['seed_ids']):
            raise ValueError('Blind predicted count changed')
        generated, ts = plans_for(s, cached['scores'], blind[q]['seed_ids'],
            by[q, arm(BASE)], by[q, arm(FULL)], by[q, arm(INDEPENDENT)], counter)
        if any(p != by[q, p['arm']] for p in generated):
            raise ValueError('Server coverage plan reproduction mismatch')
        traces.extend(ts)
    if traces != load(PLAN / 'public/cover_traces.json'):
        raise ValueError('Coverage trace reproduction mismatch')
    lengths = length_statistics(plans)
    if lengths != prep['length_statistics'] or not all(x['passed'] for x in lengths):
        raise ValueError('Actual-length gate failed')
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/Qwen3.5-9B'), local_files_only=True, trust_remote_code=False)
    cache, keys, maximum = replay_cache(), set(), 0
    for p in plans:
        target = by[p['question_id'], arm(BASE)]['evidence_tokens']
        if counter.count(p['evidence_text']) != p['evidence_tokens'] or not 0 < p['evidence_tokens'] <= target or fingerprint(p['evidence_text']) != p['evidence_sha256']:
            raise ValueError('Evidence identity or token budget changed')
        messages = recomp.messages_for(p)
        key = recomp.interface.parent.request_key(messages, marker)
        keys.add(key)
        if p['method'] not in NEW_METHODS and key not in cache:
            raise ValueError('Original conditions must be reused')
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        maximum = max(maximum, counter.count(text))
    new = len(keys - set(cache))
    if new > 256 or new != prep['new_reader_calls_upper_bound'] or maximum + 2048 > 32768:
        raise ValueError('Inference cap or prompt capacity failed')
    immutable(RUN / 'qa_plans.json', plans)
    immutable(RUN / 'selection_complete.json', dict(signature=sid, conditions=640,
        plans_sha256=sha(RUN / 'qa_plans.json'), uncached_requests=new))
    immutable(RUN / 'preflight.json', dict(signature=sid, passed=True, questions=128, sources=64,
        conditions=640, unique_requests=len(keys), uncached_requests=new, full_prompt_max_tokens=maximum,
        server_selection_reproduced=True, actual_length_gate_passed=True,
        new_training=False, new_encoder_calls=0, protected_final_data_accessed=False))
    print('SOURCE_COVER_PREFLIGHT_OK', new, 'uncached calls', flush=True)


def qa(endpoint):
    sid, ledger = verify(), recomp.ledger
    old = ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for
    ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = RUN, lambda: sid, replay_cache, recomp.messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = old


def summarize(rows):
    summaries, comparisons, interactions = [], [], []
    for scope in ('overall', 'multi', 'single'):
        def by(method):
            return {r['question_id']: r for r in rows if r['method'] == method and (scope == 'overall' or r['mode'] == scope)}
        for method in METHODS:
            group = list(by(method).values())
            summaries.append(dict(method=method, budget=256, scope=scope, n=len(group),
                f1_points=100 * statistics.mean(r['local_native_f1'] for r in group),
                full_f1_count=sum(r['local_native_f1'] == 1 for r in group),
                mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in group),
                statuses=dict(Counter(r['status'] for r in group))))
        for a, b in PAIRS:
            comparisons.append(dict(method=a, comparator=b, budget=256, scope=scope,
                **recomp.interface.paired(by(a), by(b))))
        a, b = {}, {}
        for q, full in by(FULL).items():
            four = [by(m)[q] for m in (COVER_PATH, COVER_INDEPENDENT, FULL, INDEPENDENT)]
            normal = all(r['status'] == 'ok' for r in four)
            delta = four[0]['local_native_f1'] - four[1]['local_native_f1'] - four[2]['local_native_f1'] + four[3]['local_native_f1']
            a[q] = dict(native_id=full['native_id'], status='ok' if normal else 'failed', local_native_f1=delta)
            b[q] = dict(native_id=full['native_id'], status='ok' if normal else 'failed', local_native_f1=0.0)
        interactions.append(dict(scope=scope, contrast='(cover_path-cover_independent)-(original_path-original_independent)',
            **recomp.interface.paired(a, b)))
    return summaries, comparisons, interactions


def score():
    sid = verify()
    rows = recomp.interface.parent.score_rows(load(RUN / 'qa_plans.json'),
        list(read_jsonl(RUN / 'qa/predictions.jsonl')), load(PLAN / 'private/references.json'), sid)
    cache = replay_cache()
    for row in rows:
        if row['method'] not in NEW_METHODS:
            entry = cache[row['request_key']]
            if not row['reused'] or any(row[k] != entry[k] for k in ('hypothesis', 'status', 'error')):
                raise ValueError('Historical output changed')
    summaries, comparisons, interactions = summarize(rows)
    calls = list(read_jsonl(RUN / 'qa/answer_calls.jsonl')) if (RUN / 'qa/answer_calls.jsonl').exists() else []
    if len(calls) != load(PLAN / 'preparation.json')['new_reader_calls_upper_bound']:
        raise ValueError('Call accounting mismatch')
    immutable(RUN / 'scored.json', rows)
    immutable(RUN / 'report.json', dict(signature=sid, complete=True, scores=summaries,
        comparisons=comparisons, factorial_interactions=interactions,
        new_reader_calls=len(calls), reused_conditions=sum(r['reused'] for r in rows),
        reader_cost={k: sum(c['result'].get('usage', {}).get(k, 0) for c in calls) for k in ('prompt_tokens', 'completion_tokens')},
        new_reader_seconds=sum(c['seconds'] for c in calls), config=CONFIG,
        length_statistics=load(PLAN / 'preparation.json')['length_statistics'],
        paper_ready=False, independent_confirmation=False, automatic_followup=False))
    print('SOURCE_COVER_COMPLETE', [s for s in summaries if s['scope'] == 'overall'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'verify', 'preflight', 'qa', 'score'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8033')
    args = parser.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    else:
        value = globals()[args.mode]()
        if value:
            print(value, flush=True)
