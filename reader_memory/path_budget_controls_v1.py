"""Cached-path packing controls with per-question actual-token targets."""
import argparse
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import statistics

from . import path_retrieval_execution_v1 as previous
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/path_budget_controls_v1'
RUN = ROOT / 'runs/path_budget_controls_v1'
recomp = previous.parent.recomp
load, sha, immutable = previous.load, previous.sha, previous.immutable
OLD_ARMS = ('recomp_sentence', 'beam_path_fixed')
NEW_ARMS = ('path_seed_fill', 'independent_seed_fill', 'recomp_short')
ARMS = OLD_ARMS + NEW_ARMS
CONFIG = dict(version='path-budget-controls-v1', questions=128, sources=64,
    arms=list(ARMS), max_conditions=640, max_new_reader_calls=384,
    evidence_cap=2048, full_target='per-question cached RECOMP actual evidence tokens',
    short_target='per-question cached path actual evidence tokens',
    reader='Qwen3.5-9B', reader_variant='format_only', output_tokens=2048,
    decoding='greedy-thinking_false', retries=0, training=False, new_retrieval=False,
    seeds='cached first blind path; independent paragraph max cached sentence score, same predicted count',
    fill='same cached RECOMP sentence scan; exact source-offset union; no repeated or dummy padding',
    seed_overflow='ordinary RECOMP sentence compression inside the same seed paragraphs',
    order='best retained sentence rank per source, native text order within source',
    length_gate=dict(min_mean_ratio=0.98, min_fraction_at_95_percent=0.95),
    independent_confirmation=False, dependency_supervision=False, protected_final_data_accessed=False,
    automatic_followup=False)


def sentence_order(source, scores):
    units = recomp.units(source)
    if set(scores) != {u for u, _ in units} or not all(math.isfinite(s) for s in scores.values()):
        raise ValueError('Missing/nonfinite shared sentence scores')
    position = {u: i for i, (u, _) in enumerate(units)}
    return sorted(scores, key=lambda u: (-scores[u], position[u]))


def canonical_ranges(source, scores, ranges):
    by = {p['source_id']: p for p in source['paragraphs']}
    ranks = {u: i for i, u in enumerate(sentence_order(source, scores))}
    result = {}
    keys = {}
    for pid, values in ranges.items():
        result[pid] = recomp.merge_ranges(by[pid]['text'], values)
        kept = [ranks[pid + ':span:' + str(i)] for i, span in enumerate(by[pid]['spans'])
                if any(a <= span['start'] and b >= span['end'] for a, b in result[pid])]
        if not kept:
            raise ValueError('Cannot render a partial or empty source sentence')
        keys[pid] = min(kept)
    return {pid: result[pid] for pid in sorted(result, key=lambda pid: keys[pid])}


def make_plan(source, scores, ranges, counter, arm, budget):
    selected = canonical_ranges(source, scores, ranges)
    text = recomp.render_ranges(source, selected)
    if not text.strip() or counter.count(text) > budget:
        raise ValueError('Empty or over-budget selection; never slice sentences')
    by = {p['source_id']: p for p in source['paragraphs']}
    spans = [uid for uid, _ in recomp.units(source) if any(
        a <= by[uid.split(':')[0]]['spans'][int(uid.split(':')[2])]['start'] and
        b >= by[uid.split(':')[0]]['spans'][int(uid.split(':')[2])]['end']
        for a, b in selected.get(uid.split(':')[0], []))]
    full = [pid for pid, values in selected.items() if len(values) == 1 and
            not by[pid]['text'][:values[0][0]].strip() and not by[pid]['text'][values[0][1]:].strip()]
    return dict(**{k: source[k] for k in ('question_id', 'native_id', 'mode', 'question')},
        arm=arm, reader_variant='format_only', selected_ids=['quote:' + pid for pid in selected],
        selected_ranges=selected, selected_spans=spans, fully_preserved_sources=full,
        evidence_text=text, evidence_tokens=counter.count(text), evidence_sha256=fingerprint(text),
        private_inputs_read=False, diagnostic_only=False)


def independent_seeds(source, scores, count):
    if not 1 <= count <= len(source['paragraphs']):
        raise ValueError('Invalid predicted seed count')
    ordered, seen = sentence_order(source, scores), []
    for uid in ordered:
        pid = uid.split(':')[0]
        if pid not in seen:
            seen.append(pid)
        if len(seen) == count:
            return seen
    raise ValueError('Insufficient public paragraphs')


def seed_fill(source, scores, seed_ids, counter, arm, budget):
    if not counter.exact or not 0 < budget <= 2048:
        raise ValueError('Inexact counter or illegal budget')
    by = {p['source_id']: p for p in source['paragraphs']}
    if len(seed_ids) != len(set(seed_ids)) or not set(seed_ids) <= set(by):
        raise ValueError('Duplicate or invalid seed')
    selected = {pid: [[0, len(by[pid]['text'])]] for pid in seed_ids}
    seed_tokens = counter.count(recomp.render_ranges(source, canonical_ranges(source, scores, selected))) if selected else 0
    overflow = seed_tokens > budget
    if overflow:
        sub = dict(source, paragraphs=[by[pid] for pid in seed_ids])
        allowed = {uid for uid, _ in recomp.units(sub)}
        compressed, _ = recomp.pack(sub, {u: scores[u] for u in allowed}, 'recomp_sentence', counter, budget)
        selected = compressed['selected_ranges']
    trace = []
    for uid in sentence_order(source, scores):
        pid, _, index = uid.split(':')
        span = by[pid]['spans'][int(index)]
        proposal = deepcopy(selected)
        proposal[pid] = recomp.merge_ranges(by[pid]['text'], proposal.get(pid, []) + [[span['start'], span['end']]])
        if proposal == selected:
            continue
        n = counter.count(recomp.render_ranges(source, canonical_ranges(source, scores, proposal)))
        trace.append(dict(uid=uid, proposed_tokens=n, accepted=n <= budget))
        if n <= budget:
            selected = proposal
    plan = make_plan(source, scores, selected, counter, arm, budget)
    if not overflow and any(pid not in plan['fully_preserved_sources'] for pid in seed_ids):
        raise ValueError('Filling deleted a seed paragraph')
    return plan, dict(seed_ids=seed_ids, seed_tokens=seed_tokens, seed_overflow=overflow,
                     target_tokens=budget, actual_tokens=plan['evidence_tokens'], trace=trace)


def replay_cache():
    marker = load(PLAN / 'preparation.json')['reader_marker_sha256']
    cache = {}
    for item in load(PLAN / 'private/replay_cache.json'):
        recomp.interface.parent.check_cache_entry(item, marker)
        if item['key'] in cache:
            raise ValueError('Duplicate historical request')
        cache[item['key']] = item['entry']
    return cache


def freeze():
    parent_signature = previous.verify()
    preparation = load(PLAN / 'preparation.json')
    names = ['reader_memory/path_budget_controls_v1.py', 'tools/prepare_path_budget_controls_v1.py',
             'scripts/path_budget_controls_v1_queue.py', 'tests/test_path_budget_controls_v1.py',
             'research/path_retrieval_execution_v1/manifest.json']
    names += list(preparation['server_source_files'])
    names += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
              and 'snapshots' not in p.relative_to(PLAN).parts and p.suffix != '.pyc'
              and p.name not in {'manifest.json', 'CURRENT_STATUS.md', 'submission_receipt.json'}]
    value = dict(config=CONFIG, parent_signature=parent_signature,
                 files={n: sha(ROOT / n) for n in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_PATH_BUDGET_CONTROLS', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Scientific identity changed')
    if previous.verify() != value['parent_signature']:
        raise ValueError('Completed parent changed')
    for name, digest in value['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return value['signature']


def preflight():
    sid = verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter = ExactCounter()
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/Qwen3.5-9B'), local_files_only=True, trust_remote_code=False)
    preparation = load(PLAN / 'preparation.json')
    marker = preparation['reader_marker_sha256']
    if sha(ROOT / 'models/Qwen3.5-9B/.verified.json') != marker:
        raise ValueError('Reader identity changed')
    for name, digest in preparation['server_source_files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Completed source evidence changed: ' + name)
    plans = load(PLAN / 'public/qa_plans.json')
    positions = {p['question_id'] for p in plans}
    if len(plans) != 640 or {(p['question_id'], p['arm']) for p in plans} != {(q, a) for q in positions for a in ARMS}:
        raise ValueError('Incomplete fixed 128-by-5 comparison')
    cache = replay_cache()
    keys, prompts = set(), []
    for p in plans:
        if counter.count(p['evidence_text']) != p['evidence_tokens'] or p['evidence_tokens'] > 2048 or fingerprint(p['evidence_text']) != p['evidence_sha256']:
            raise ValueError('Evidence changed or tokenizer mismatch')
        messages = recomp.messages_for(p)
        key = recomp.interface.parent.request_key(messages, marker)
        keys.add(key)
        if p['arm'] in OLD_ARMS and key not in cache:
            raise ValueError('Historical outcome must be reused, including failure')
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        n = counter.count(prompt)
        if n + 2048 > 32768:
            raise ValueError('Prompt exceeds reader capacity; refuse truncation')
        prompts.append(dict(question_id=p['question_id'], arm=p['arm'], request_key=key, full_prompt_tokens=n))
    new = len(keys - set(cache))
    if new > 384 or new != preparation['new_reader_calls_upper_bound'] or not preparation['full_length_gate_passed']:
        raise ValueError('Call cap or full-context length gate failed')
    immutable(RUN / 'qa_plans.json', plans)
    immutable(RUN / 'selection_complete.json', dict(signature=sid, plans_sha256=sha(RUN / 'qa_plans.json'),
        conditions=640, uncached_requests=new, source='CPU prepared; cached retriever and sentence scores'))
    immutable(RUN / 'preflight.json', dict(signature=sid, passed=True, conditions=640,
        unique_requests=len(keys), uncached_requests=new, full_prompt_max_tokens=max(p['full_prompt_tokens'] for p in prompts),
        length_statistics=preparation['length_statistics'], full_length_gate_passed=True, prompt_identities=prompts,
        new_training=False, protected_final_data_accessed=False))
    print('PATH_BUDGET_PREFLIGHT_OK', new, 'uncached calls', flush=True)


def qa(endpoint):
    sid = verify()
    ledger = recomp.ledger
    old = ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for
    ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = RUN, lambda: sid, replay_cache, recomp.messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = old


def score():
    sid = verify()
    refs = load(previous.parent.PLAN / 'private/development_references.json')
    plans = load(RUN / 'qa_plans.json')
    scored = recomp.interface.parent.score_rows(plans, list(read_jsonl(RUN / 'qa/predictions.jsonl')), refs, sid)
    old = {(r['question_id'], r['arm']): r for r in load(PLAN / 'private/baseline_outcomes.json')}
    for r in scored:
        if r['arm'] in OLD_ARMS and (not r['reused'] or any(r[k] != old[r['question_id'], r['arm']][k] for k in ('hypothesis', 'status', 'local_native_f1'))):
            raise ValueError('Original baseline result changed')
    summaries, comparisons = [], []
    pairs = [('path_seed_fill', 'recomp_sentence'), ('path_seed_fill', 'independent_seed_fill'),
             ('beam_path_fixed', 'recomp_short'), ('independent_seed_fill', 'recomp_sentence'),
             ('path_seed_fill', 'beam_path_fixed')]
    for scope in ('overall', 'multi', 'single'):
        by = lambda a: {r['question_id']: r for r in scored if r['arm'] == a and (scope == 'overall' or r['mode'] == scope)}
        for arm in ARMS:
            rows = list(by(arm).values())
            tokens = [r['evidence_tokens'] for r in rows]
            summaries.append(dict(arm=arm, scope=scope, n=len(rows),
                f1_points=100 * statistics.mean(r['local_native_f1'] for r in rows),
                mean_evidence_tokens=statistics.mean(tokens), min_evidence_tokens=min(tokens),
                median_evidence_tokens=statistics.median(tokens), max_evidence_tokens=max(tokens),
                statuses=dict(Counter(r['status'] for r in rows)),
                all_support_sources_touched=sum(set(map(str, refs[r['question_id']]['support_indices'])) <=
                    {u.split(':')[0] for u in r['selected_spans']} for r in rows)))
        for a, b in pairs:
            comparisons.append(dict(arm=a, comparator=b, scope=scope, **recomp.interface.paired(by(a), by(b))))
    get = lambda scope, other='recomp_sentence': next(c for c in comparisons if c['arm'] == 'path_seed_fill' and c['comparator'] == other and c['scope'] == scope)
    passed = (get('overall')['delta_points'] > 0 and get('multi')['delta_points'] > 0 and
              get('multi')['wins'] >= 3 and get('single')['delta_points'] >= -3.125 and
              get('overall')['normal_pair_delta_points'] is not None and get('overall')['normal_pair_delta_points'] > 0)
    calls_path = RUN / 'qa/answer_calls.jsonl'
    calls = list(read_jsonl(calls_path)) if calls_path.exists() else []
    if len(calls) > 384:
        raise ValueError('New-call cap exceeded')
    immutable(RUN / 'scored.json', scored)
    immutable(RUN / 'report.json', dict(signature=sid, complete=True, scores=summaries, comparisons=comparisons,
        new_reader_calls=len(calls), reused_conditions=sum(r['reused'] for r in scored),
        reader_cost={k: sum(c['result'].get('usage', {}).get(k, 0) for c in calls) for k in ('prompt_tokens', 'completion_tokens')},
        new_reader_seconds=sum(c['seconds'] for c in calls), preparation=load(PLAN / 'preparation.json'),
        resource_continue_signal=passed,
        positive_path_vs_independent_point_estimate=(get('overall', 'independent_seed_fill')['delta_points'] > 0 and get('multi', 'independent_seed_fill')['delta_points'] > 0),
        novelty_proven=False, independent_confirmation=False, paper_ready=False,
        new_training=False, automatic_followup=False, config=CONFIG))
    print('PATH_BUDGET_RESULTS', summaries, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'verify', 'preflight', 'qa', 'score'])
    p.add_argument('--endpoint', default='http://127.0.0.1:8025')
    args = p.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    else:
        result = globals()[args.mode]()
        if result:
            print(result, flush=True)
