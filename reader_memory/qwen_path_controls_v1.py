"""Shared frozen Qwen sentence scores for ordinary path packing controls."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

from . import reranker_budget_v1 as parent
from . import path_budget_curve_v1 as packing
from .schema import fingerprint, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/qwen_path_controls_v1'
RUN = ROOT / 'runs/qwen_path_controls_v1'
load, sha, immutable, recomp = parent.load, parent.sha, parent.immutable, parent.recomp
BUDGETS = (256, 512, 1024, 2048)
BASE = 'qwen_rerank_sentence'
PATH = 'qwen_path_compress'
FILL = 'qwen_path_seed_fill'
OLD = 'beam_path_compress'
METHODS = (BASE, PATH, FILL, OLD)
NEW_METHODS = (PATH, FILL)
PAIRS = ((PATH, BASE), (FILL, BASE), (FILL, PATH), (PATH, OLD), (FILL, OLD))
CONFIG = dict(version='qwen-path-controls-v1', questions=128, sources=64,
    budgets=list(BUDGETS), methods=list(METHODS), max_conditions=2048,
    max_new_reader_calls=1024, reader='Qwen3.5-9B', reader_variant='format_only',
    output_tokens=2048, decoding='greedy-thinking_false', retries=0,
    sentence_scores='unchanged cached Qwen3-Reranker-4B FP32 yes-minus-no over title and original sentence',
    paths='unchanged cached first blind Beam path; predicted hop count only; no gold routing',
    path_packing='retain full path if it fits; otherwise common whole-sentence greedy packing within path',
    fill_packing='same path seed packing followed by global Qwen sentence scan, without deleting retained seed text',
    fill_target='per-question/budget frozen Qwen global actual evidence tokens',
    presentation='source order by best retained Qwen sentence rank; original sentence order within source',
    primary='256 path versus global; report filled control separately', secondary='512; report1024/2048',
    filled_min_mean_length_ratio=0.95, training=False, new_retrieval=False, new_encoder_calls=0,
    dependency_protection_implemented=False, independent_confirmation=False,
    protected_final_data_accessed=False, automatic_followup=False)


def arm(method, budget):
    if method not in METHODS or budget not in BUDGETS:
        raise ValueError('Unregistered shared-score control')
    return method + '__b' + str(budget)


def plans_for(source, scores, blind_path, counter, budget):
    """Selection sees public text, fixed semantic scores and the prior blind path."""
    arm(BASE, budget)
    packing.parent.sentence_order(source, scores)
    ids = {p['source_id'] for p in source['paragraphs']}
    if not blind_path or len(blind_path) != len(set(blind_path)) or not set(blind_path) <= ids:
        raise ValueError('Invalid public predicted path')
    base = parent.sentence_plan(source, scores, counter, budget)
    path, path_trace = packing.compressed_path(source, scores, blind_path, counter, budget)
    fill, fill_trace = packing.parent.seed_fill(
        source, scores, blind_path, counter, FILL, base['evidence_tokens'])
    path.update(arm=arm(PATH, budget), method=PATH, evidence_budget=budget)
    fill.update(arm=arm(FILL, budget), method=FILL, evidence_budget=budget)
    if not set(path['selected_ranges']) <= set(blind_path):
        raise ValueError('Path-only packing escaped predicted path')
    if path['evidence_tokens'] > budget or fill['evidence_tokens'] > base['evidence_tokens']:
        raise ValueError('Shared budget contract violated')
    return [base, path, fill], [dict(method=PATH, budget=budget, **path_trace),
                              dict(method=FILL, budget=budget, **fill_trace)]


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
    preparation = load(PLAN / 'preparation.json')
    parents = dict(reranker=parent.verify(), path=packing.parent.previous.verify())
    names = ['reader_memory/qwen_path_controls_v1.py', 'tools/prepare_qwen_path_controls_v1.py',
        'scripts/qwen_path_controls_v1_queue.py', 'tests/test_qwen_path_controls_v1.py',
        'reader_memory/path_budget_curve_v1.py', 'reader_memory/path_budget_controls_v1.py',
        'tools/prepare_replacement_learning_v1.py', 'tools/status_qwen_path_controls_v1.py',
        'tests/test_path_budget_curve_v1.py']
    names += list(preparation['server_source_files'])
    names += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
        and 'snapshots' not in p.relative_to(PLAN).parts and p.suffix != '.pyc'
        and p.name not in {'manifest.json', 'CURRENT_STATUS.md', 'submission_receipt.json', 'STATUS_LATEST.json', 'SNAPSHOT_LATEST.json'}]
    value = dict(config=CONFIG, parent_signatures=parents,
                 files={n: sha(ROOT / n) for n in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_QWEN_PATH_CONTROLS', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Shared-score experiment identity changed')
    parents = dict(reranker=parent.verify(), path=packing.parent.previous.verify())
    if parents != value['parent_signatures']:
        raise ValueError('Frozen parent changed')
    for name, digest in value['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return value['signature']


def preflight():
    sid = verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    preparation = load(PLAN / 'preparation.json')
    marker = preparation['reader_marker_sha256']
    if sha(ROOT / 'models/Qwen3.5-9B/.verified.json') != marker:
        raise ValueError('Reader identity changed')
    sources = load(PLAN / 'public/source_inputs.json')
    blind = load(PLAN / 'public/blind_paths.json')
    controls = load(PLAN / 'public/qa_plans.json')
    by = {(p['question_id'], p['arm']): p for p in controls}
    if (len(sources) != 128 or len({s['native_id'] for s in sources}) != 64
            or len(controls) != 2048 or len(by) != 2048):
        raise ValueError('Incomplete fixed development panel')
    expected = {(s['question_id'], arm(m, b)) for s in sources for b in BUDGETS for m in METHODS}
    if set(by) != expected:
        raise ValueError('Incomplete condition matrix')
    counter = ExactCounter()
    # Recompute all new public selections on the server before any reader call.
    for source in sources:
        q = source['question_id']
        scores = load(ROOT / 'runs/reranker_budget_v1/selected' / (q.replace(':', '_') + '.json'))
        if scores['source_sha256'] != fingerprint(source) or blind[q]['source_sha256'] != fingerprint(source):
            raise ValueError('Source/path/score provenance mismatch')
        if blind[q]['predicted_hops'] != len(blind[q]['seed_ids']):
            raise ValueError('Invalid blind predicted hop count')
        for budget in BUDGETS:
            plans, _ = plans_for(source, scores['scores'], blind[q]['seed_ids'], counter, budget)
            for plan in plans:
                if plan != by[q, plan['arm']]:
                    raise ValueError('Prepared selection does not reproduce on server')
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/Qwen3.5-9B'), local_files_only=True, trust_remote_code=False)
    cache, keys, maximum = replay_cache(), set(), 0
    for plan in controls:
        if (counter.count(plan['evidence_text']) != plan['evidence_tokens']
                or plan['evidence_tokens'] > plan['evidence_budget']
                or fingerprint(plan['evidence_text']) != plan['evidence_sha256']):
            raise ValueError('Evidence identity or budget changed')
        messages = recomp.messages_for(plan)
        key = recomp.interface.parent.request_key(messages, marker)
        if plan['method'] not in NEW_METHODS and key not in cache:
            raise ValueError('Original outcomes must be reused, including failure')
        keys.add(key)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        maximum = max(maximum, counter.count(prompt))
    new = len(keys - set(cache))
    if (new > 1024 or new != preparation['new_reader_calls_upper_bound']
            or maximum + 2048 > 32768 or not preparation['filled_length_gate_passed']):
        raise ValueError('Inference cap, capacity or actual-length gate failed')
    immutable(RUN / 'qa_plans.json', controls)
    immutable(RUN / 'selection_complete.json', dict(signature=sid,
        plans_sha256=sha(RUN / 'qa_plans.json'), conditions=2048, uncached_requests=new))
    immutable(RUN / 'preflight.json', dict(signature=sid, passed=True, questions=128, sources=64,
        conditions=2048, unique_requests=len(keys), uncached_requests=new, full_prompt_max_tokens=maximum,
        filled_length_gate_passed=True, server_selection_reproduced=True, new_training=False,
        new_encoder_calls=0, protected_final_data_accessed=False))
    print('QWEN_PATH_PREFLIGHT_OK', new, 'uncached calls', flush=True)


def qa(endpoint):
    sid = verify()
    ledger = recomp.ledger
    old = ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for
    ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = RUN, lambda: sid, replay_cache, recomp.messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = old


def summarize(rows):
    summaries, comparisons = [], []
    for budget in BUDGETS:
        for scope in ('overall', 'multi', 'single'):
            def by(method):
                return {r['question_id']: r for r in rows if r['method'] == method
                    and r['evidence_budget'] == budget and (scope == 'overall' or r['mode'] == scope)}
            for method in METHODS:
                selected = list(by(method).values())
                summaries.append(dict(method=method, budget=budget, scope=scope, n=len(selected),
                    f1_points=100*statistics.mean(r['local_native_f1'] for r in selected),
                    full_f1_count=sum(r['local_native_f1'] == 1 for r in selected),
                    mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in selected),
                    statuses=dict(Counter(r['status'] for r in selected))))
            for a, b in PAIRS:
                comparisons.append(dict(method=a, comparator=b, budget=budget, scope=scope,
                    **recomp.interface.paired(by(a), by(b))))
    return summaries, comparisons


def score():
    sid = verify()
    rows = recomp.interface.parent.score_rows(load(RUN / 'qa_plans.json'),
        list(read_jsonl(RUN / 'qa/predictions.jsonl')), load(PLAN / 'private/references.json'), sid)
    summaries, comparisons = summarize(rows)
    cache = replay_cache()
    for row in rows:
        if row['method'] not in NEW_METHODS:
            entry = cache[row['request_key']]
            if not row['reused'] or any(row[k] != entry[k] for k in ('hypothesis', 'status', 'error')):
                raise ValueError('Original control output changed')
    calls = list(read_jsonl(RUN / 'qa/answer_calls.jsonl')) if (RUN / 'qa/answer_calls.jsonl').exists() else []
    if len(calls) != load(PLAN / 'preparation.json')['new_reader_calls_upper_bound']:
        raise ValueError('New call accounting mismatch')
    immutable(RUN / 'scored.json', rows)
    immutable(RUN / 'report.json', dict(signature=sid, complete=True, scores=summaries,
        comparisons=comparisons, new_reader_calls=len(calls), reused_conditions=sum(r['reused'] for r in rows),
        reader_cost={k: sum(c['result'].get('usage', {}).get(k, 0) for c in calls)
            for k in ('prompt_tokens', 'completion_tokens')},
        new_reader_seconds=sum(c['seconds'] for c in calls),
        length_statistics=load(PLAN / 'preparation.json')['length_statistics'],
        config=CONFIG, paper_ready=False, independent_confirmation=False, automatic_followup=False))
    print('QWEN_PATH_CONTROLS_COMPLETE', [s for s in summaries if s['scope'] == 'overall'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'verify', 'preflight', 'qa', 'score'])
    p.add_argument('--endpoint', default='http://127.0.0.1:8029')
    args = p.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    else:
        result = globals()[args.mode]()
        if result:
            print(result, flush=True)
