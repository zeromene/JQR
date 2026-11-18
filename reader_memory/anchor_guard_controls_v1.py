"""Test loss of the strongest retained global sentence as a packing failure."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import statistics

from . import qwen_path_controls_v1 as parent
from .schema import fingerprint, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/anchor_guard_controls_v1'
RUN = ROOT / 'runs/anchor_guard_controls_v1'
load, sha, immutable, recomp = parent.load, parent.sha, parent.immutable, parent.recomp
packing = parent.packing.parent
BUDGETS = parent.BUDGETS
BASE, PREVIOUS = parent.BASE, parent.FILL
MAIN, INDEPENDENT = 'anchor_path_fill', 'anchor_independent_fill'
METHODS = (BASE, PREVIOUS, MAIN, INDEPENDENT)
NEW_METHODS = (MAIN, INDEPENDENT)
PAIRS = ((MAIN, BASE), (MAIN, PREVIOUS), (MAIN, INDEPENDENT), (INDEPENDENT, BASE))
CONFIG = dict(version='anchor-guard-controls-v1', questions=128, sources=64,
    budgets=list(BUDGETS), methods=list(METHODS), max_conditions=2048, max_new_reader_calls=1024,
    reader='Qwen3.5-9B', reader_variant='format_only', output_tokens=2048,
    decoding='greedy-thinking_false', retries=0,
    sentence_scores='unchanged cached Qwen3-Reranker-4B FP32 title+sentence scores',
    paths='unchanged first blind Beam path; independent seeds use same predicted count',
    anchor='one highest-ranked complete sentence retained by frozen global baseline, original tie order',
    trigger='anchor absent from ordinary filled context; if present retain evidence byte-for-byte',
    repair='anchor first; retain old selected sentences by common Qwen order if they fit; then global sentence fill',
    target='per-question/budget frozen global actual evidence tokens',
    presentation='source order by best retained Qwen rank, original sentence order within source',
    primary='256 guarded path versus ordinary path fill and strong global; independent seed control',
    secondary='512; report1024/2048', filled_min_mean_length_ratio=0.95,
    training=False, new_retrieval=False, new_encoder_calls=0,
    dependency_protection_implemented=False, reader_utility_model=False,
    independent_confirmation=False, protected_final_data_accessed=False, automatic_followup=False)


def arm(method, budget):
    if method not in METHODS or budget not in BUDGETS:
        raise ValueError('Unregistered anchor control')
    return method + '__b' + str(budget)


def guarded_plan(source, scores, baseline, candidate, counter, budget, method):
    """Use public selected offsets and one global anchor, never outcomes or labels."""
    arm(method, budget)
    order = packing.sentence_order(source, scores)
    baseline_spans = set(baseline['selected_spans'])
    anchor = next((u for u in order if u in baseline_spans), None)
    if anchor is None:
        raise ValueError('Empty global baseline')
    target = baseline['evidence_tokens']
    if not counter.exact or not 0 < target <= budget:
        raise ValueError('Invalid exact-token target')
    triggered = anchor not in candidate['selected_spans']
    if not triggered:
        result = deepcopy(candidate)
        result.update(method=method, arm=arm(method, budget), evidence_budget=budget)
        return result, dict(anchor=anchor, triggered=False, target_tokens=target,
            actual_tokens=result['evidence_tokens'], removed_spans=[], added_spans=[], trace=[])
    by = {p['source_id']: p for p in source['paragraphs']}
    selected, trace = {}, []

    def attempt(uid, phase):
        nonlocal selected
        pid, _, index = uid.split(':')
        span = by[pid]['spans'][int(index)]
        proposal = deepcopy(selected)
        proposal[pid] = recomp.merge_ranges(by[pid]['text'],
            proposal.get(pid, []) + [[span['start'], span['end']]])
        if proposal == selected:
            return
        n = counter.count(recomp.render_ranges(source, packing.canonical_ranges(source, scores, proposal)))
        trace.append(dict(uid=uid, phase=phase, proposed_tokens=n, accepted=n <= target))
        if n <= target:
            selected = proposal

    attempt(anchor, 'anchor')
    if not selected:
        raise ValueError('Global retained sentence cannot fit its own baseline budget')
    old = set(candidate['selected_spans'])
    for uid in order:
        if uid in old:
            attempt(uid, 'old_context')
    for uid in order:
        attempt(uid, 'global_fill')
    result = packing.make_plan(source, scores, selected, counter, arm(method, budget), target)
    result.update(method=method, evidence_budget=budget)
    if anchor not in result['selected_spans'] or result['evidence_tokens'] > target:
        raise ValueError('Anchor or budget contract violated')
    return result, dict(anchor=anchor, triggered=True, target_tokens=target,
        actual_tokens=result['evidence_tokens'],
        removed_spans=sorted(old-set(result['selected_spans'])),
        added_spans=sorted(set(result['selected_spans'])-old), trace=trace)


def plans_for(source, scores, blind_path, counter, budget):
    arm(BASE, budget)
    old, _ = parent.plans_for(source, scores, blind_path, counter, budget)
    baseline, original = old[0], old[2]
    seeds = packing.independent_seeds(source, scores, len(blind_path))
    independent, seed_trace = packing.seed_fill(source, scores, seeds, counter, INDEPENDENT,
                                               baseline['evidence_tokens'])
    main, mt = guarded_plan(source, scores, baseline, original, counter, budget, MAIN)
    indep, it = guarded_plan(source, scores, baseline, independent, counter, budget, INDEPENDENT)
    return [baseline, original, main, indep], [dict(method=MAIN, budget=budget, **mt),
        dict(method=INDEPENDENT, budget=budget, independent_seed_ids=seeds,
             seed_packing=seed_trace, **it)]


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
    parent_signature = parent.verify()
    preparation = load(PLAN / 'preparation.json')
    names = ['reader_memory/anchor_guard_controls_v1.py', 'tools/prepare_anchor_guard_controls_v1.py',
        'scripts/anchor_guard_controls_v1_queue.py', 'tests/test_anchor_guard_controls_v1.py',
        'tests/test_path_budget_curve_v1.py', 'tools/status_anchor_guard_controls_v1.py',
        'research/qwen_path_controls_v1/manifest.json']
    names += list(preparation['server_source_files'])
    names += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
        and 'snapshots' not in p.relative_to(PLAN).parts and p.suffix != '.pyc'
        and p.name not in {'manifest.json', 'CURRENT_STATUS.md', 'submission_receipt.json', 'STATUS_LATEST.json', 'SNAPSHOT_LATEST.json'}]
    value = dict(config=CONFIG, parent_signature=parent_signature,
                 files={n: sha(ROOT / n) for n in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_ANCHOR_GUARD_CONTROLS', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k:v for k,v in value.items() if k!='signature'}):
        raise ValueError('Anchor experiment identity changed')
    if parent.verify() != value['parent_signature']:
        raise ValueError('Frozen parent changed')
    for n, digest in value['files'].items():
        if sha(ROOT / n) != digest:
            raise ValueError('Frozen input changed: ' + n)
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
    plans = load(PLAN / 'public/qa_plans.json')
    by = {(p['question_id'],p['arm']):p for p in plans}
    expected = {(s['question_id'],arm(m,b)) for s in sources for b in BUDGETS for m in METHODS}
    if len(sources)!=128 or len({s['native_id'] for s in sources})!=64 or len(plans)!=2048 or set(by)!=expected:
        raise ValueError('Incomplete fixed development matrix')
    counter = ExactCounter()
    for source in sources:
        q = source['question_id']
        scores = load(ROOT/'runs/reranker_budget_v1/selected'/(q.replace(':','_')+'.json'))
        if scores['source_sha256']!=fingerprint(source) or blind[q]['source_sha256']!=fingerprint(source):
            raise ValueError('Source/path/score provenance mismatch')
        if blind[q]['predicted_hops']!=len(blind[q]['seed_ids']):
            raise ValueError('Invalid blind predicted count')
        for budget in BUDGETS:
            generated,_ = plans_for(source,scores['scores'],blind[q]['seed_ids'],counter,budget)
            for p in generated:
                if p!=by[q,p['arm']]:
                    raise ValueError('Server plan reproduction mismatch')
    tok = AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    cache,keys,maximum = replay_cache(),set(),0
    for p in plans:
        target = by[p['question_id'],arm(BASE,p['evidence_budget'])]['evidence_tokens']
        if counter.count(p['evidence_text'])!=p['evidence_tokens'] or p['evidence_tokens']>target or fingerprint(p['evidence_text'])!=p['evidence_sha256']:
            raise ValueError('Budget or evidence identity changed')
        messages = recomp.messages_for(p)
        key = recomp.interface.parent.request_key(messages,marker)
        keys.add(key)
        if p['method'] not in NEW_METHODS and key not in cache:
            raise ValueError('Historical control must be reused')
        text = tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        maximum = max(maximum,counter.count(text))
    new = len(keys-set(cache))
    if new>1024 or new!=preparation['new_reader_calls_upper_bound'] or maximum+2048>32768 or not preparation['filled_length_gate_passed']:
        raise ValueError('Call cap, prompt capacity or length gate failed')
    immutable(RUN/'qa_plans.json',plans)
    immutable(RUN/'selection_complete.json',dict(signature=sid,plans_sha256=sha(RUN/'qa_plans.json'),conditions=2048,uncached_requests=new))
    immutable(RUN/'preflight.json',dict(signature=sid,passed=True,questions=128,sources=64,conditions=2048,
        unique_requests=len(keys),uncached_requests=new,full_prompt_max_tokens=maximum,
        server_selection_reproduced=True,filled_length_gate_passed=True,new_training=False,
        new_encoder_calls=0,protected_final_data_accessed=False))
    print('ANCHOR_GUARD_PREFLIGHT_OK',new,'uncached calls',flush=True)


def qa(endpoint):
    sid = verify()
    ledger = recomp.ledger
    old = ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for
    ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for = RUN,lambda:sid,replay_cache,recomp.messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for = old


def summarize(rows):
    summaries,comparisons = [],[]
    for budget in BUDGETS:
        for scope in ('overall','multi','single'):
            def by(method):
                return {r['question_id']:r for r in rows if r['method']==method and r['evidence_budget']==budget
                        and (scope=='overall' or r['mode']==scope)}
            for method in METHODS:
                group = list(by(method).values())
                summaries.append(dict(method=method,budget=budget,scope=scope,n=len(group),
                    f1_points=100*statistics.mean(r['local_native_f1'] for r in group),
                    full_f1_count=sum(r['local_native_f1']==1 for r in group),
                    mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in group),
                    statuses=dict(Counter(r['status'] for r in group))))
            for a,b in PAIRS:
                comparisons.append(dict(method=a,comparator=b,budget=budget,scope=scope,
                    **recomp.interface.paired(by(a),by(b))))
    return summaries,comparisons


def score():
    sid = verify()
    rows = recomp.interface.parent.score_rows(load(RUN/'qa_plans.json'),list(read_jsonl(RUN/'qa/predictions.jsonl')),
                                              load(PLAN/'private/references.json'),sid)
    summaries,comparisons = summarize(rows)
    cache = replay_cache()
    for r in rows:
        if r['method'] not in NEW_METHODS:
            entry = cache[r['request_key']]
            if not r['reused'] or any(r[k]!=entry[k] for k in ('hypothesis','status','error')):
                raise ValueError('Historical output changed')
    calls = list(read_jsonl(RUN/'qa/answer_calls.jsonl')) if (RUN/'qa/answer_calls.jsonl').exists() else []
    if len(calls)!=load(PLAN/'preparation.json')['new_reader_calls_upper_bound']:
        raise ValueError('New-call accounting mismatch')
    immutable(RUN/'scored.json',rows)
    immutable(RUN/'report.json',dict(signature=sid,complete=True,scores=summaries,comparisons=comparisons,
        new_reader_calls=len(calls),reused_conditions=sum(r['reused'] for r in rows),
        reader_cost={k:sum(c['result'].get('usage',{}).get(k,0) for c in calls) for k in ('prompt_tokens','completion_tokens')},
        new_reader_seconds=sum(c['seconds'] for c in calls),config=CONFIG,
        length_statistics=load(PLAN/'preparation.json')['length_statistics'],
        paper_ready=False,independent_confirmation=False,automatic_followup=False))
    print('ANCHOR_GUARD_COMPLETE',[s for s in summaries if s['scope']=='overall'],flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','verify','preflight','qa','score'])
    p.add_argument('--endpoint',default='http://127.0.0.1:8030')
    a = p.parse_args()
    if a.mode=='qa':
        qa(a.endpoint)
    else:
        result = globals()[a.mode]()
        if result:
            print(result,flush=True)
