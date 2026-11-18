"""Frozen source-block deletion diagnostics for the 256-token candidate."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import statistics
import unicodedata

from . import anchor_guard_controls_v1 as parent
from .schema import fingerprint, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/anchor_intervention_v1'
RUN = ROOT / 'runs/anchor_intervention_v1'
load, sha, immutable, recomp = parent.load, parent.sha, parent.immutable, parent.recomp
BASE, INDEPENDENT, FULL = parent.BASE, parent.INDEPENDENT, parent.MAIN
DROP_A, DROP_B, DROP_AB = 'without_anchor_source', 'without_other_path_sources', 'without_anchor_and_path'
METHODS = (BASE, INDEPENDENT, FULL, DROP_A, DROP_B, DROP_AB)
NEW_METHODS = (DROP_A, DROP_B, DROP_AB)
BUDGETS = (256,)
PAIRS = ((FULL, BASE), (FULL, INDEPENDENT), (FULL, DROP_A), (FULL, DROP_B), (FULL, DROP_AB))
CONFIG = dict(version='anchor-intervention-v1', questions=128, sources=64,
    budgets=[256], methods=list(METHODS), max_conditions=768, max_new_reader_calls=384,
    reader='Qwen3.5-9B', reader_variant='format_only', output_tokens=2048,
    decoding='greedy-thinking_false', retries=0,
    anchor='highest Qwen-ranked complete sentence actually retained by original global baseline',
    block_a='all selected sentences in anchor source; literal duplicate closure has first priority',
    block_b='all other selected blind-path source sentences; literal duplicate closure excluding A',
    literal_copy_key='NFKC casefold and whitespace normalization of full original sentence, no semantic equivalence claim',
    background='remaining selected sentences; unchanged across the four factorial cells',
    intervention='delete A, delete B, delete both; never fill, summarize, reorder surviving sources, or cut sentences',
    empty_context='empty evidence string, exact zero evidence tokens; same reader question and prompt',
    groups='public selected offsets, cached Qwen scores and predicted blind path only; no answers or support IDs',
    primary='observed two-block dependence with full exact and all three deletions nonexact; all four calls must be ok',
    strict_secondary='full exact, all three deletions zero F1; all four calls must be ok',
    interaction='F1(full)-F1(without_A)-F1(without_B)+F1(without_both); descriptive, not calibrated probability',
    scope='source-block reader dependence; not sufficient-context certification or same-length method ranking',
    training=False, new_retrieval=False, new_encoder_calls=0, semantic_copy_removal=False,
    independent_confirmation=False, protected_final_data_accessed=False, automatic_followup=False,
    paper_ready=False)


def arm(method, budget=256):
    if method not in METHODS or budget != 256:
        raise ValueError('Unregistered deletion diagnostic')
    return method + '__b256'


def literal_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def groups_for(source, scores, blind_path, full, baseline):
    order = parent.packing.sentence_order(source, scores)
    pids = {p['source_id'] for p in source['paragraphs']}
    if not blind_path or len(set(blind_path)) != len(blind_path) or not set(blind_path) <= pids:
        raise ValueError('Invalid public blind path')
    if full['question_id'] != source['question_id'] or baseline['question_id'] != source['question_id']:
        raise ValueError('Question identity mismatch')
    selected = set(full['selected_spans'])
    if not selected or not selected <= set(order):
        raise ValueError('Invalid selected original spans')
    anchor = next((u for u in order if u in set(baseline['selected_spans'])), None)
    if anchor not in selected:
        raise ValueError('Frozen full candidate is missing anchor')
    anchor_pid = anchor.split(':')[0]
    texts = dict(recomp.units(source))
    keys = {u: literal_key(texts[u]) for u in selected}
    if any(not k for k in keys.values()):
        raise ValueError('Empty original sentence')
    raw_a = {u for u in selected if u.split(':')[0] == anchor_pid}
    a_keys = {keys[u] for u in raw_a}
    a = {u for u in selected if keys[u] in a_keys}
    raw_b = {u for u in selected-a if u.split(':')[0] in set(blind_path)-{anchor_pid}}
    b_keys = {keys[u] for u in raw_b}
    b = {u for u in selected-a if keys[u] in b_keys}
    c = selected-a-b
    duplicate_classes = [sorted(u for u in selected if keys[u] == k) for k in sorted(set(keys.values()))
                         if sum(keys[u] == k for u in selected) > 1]
    return dict(anchor=anchor, anchor_source=anchor_pid, blind_path=list(blind_path),
        a=sorted(a), b=sorted(b), background=sorted(c),
        literal_a_expansion=sorted(a-raw_a), literal_b_expansion=sorted(b-raw_b),
        literal_duplicate_classes=duplicate_classes, both_blocks_present=bool(a and b))


def filtered_plan(source, full, remove, counter, method):
    arm(method)
    if not counter.exact or full['evidence_budget'] != 256:
        raise ValueError('Exact tokenizer and frozen 256 candidate required')
    selected = set(full['selected_spans'])
    if not set(remove) <= selected:
        raise ValueError('Deletion outside frozen context')
    if not remove:
        result = deepcopy(full)
        result.update(method=method, arm=arm(method), diagnostic_only=True)
        return result
    by = {p['source_id']: p for p in source['paragraphs']}
    ranges = {}
    for pid in full['selected_ranges']:
        pieces = [[s['start'], s['end']] for i, s in enumerate(by[pid]['spans'])
                  if pid+':span:'+str(i) in selected-set(remove)]
        if pieces:
            ranges[pid] = recomp.merge_ranges(by[pid]['text'], pieces)
    text = recomp.render_ranges(source, ranges)
    tokens = counter.count(text)
    if tokens > 256 or tokens > full['evidence_tokens']:
        raise ValueError('Deletion unexpectedly increased token cost')
    result = deepcopy(full)
    result.update(method=method, arm=arm(method), diagnostic_only=True,
        selected_ids=['quote:'+pid for pid in ranges], selected_ranges=ranges,
        selected_spans=[u for u in full['selected_spans'] if u not in remove],
        fully_preserved_sources=[pid for pid in ranges if len(ranges[pid]) == 1
            and not by[pid]['text'][:ranges[pid][0][0]].strip()
            and not by[pid]['text'][ranges[pid][0][1]:].strip()],
        evidence_text=text, evidence_tokens=tokens, evidence_sha256=fingerprint(text))
    return result


def plans_for(source, scores, blind_path, full, baseline, independent, counter):
    if any(p['evidence_budget'] != 256 for p in (full, baseline, independent)):
        raise ValueError('Only the predeclared primary budget is permitted')
    groups = groups_for(source, scores, blind_path, full, baseline)
    drops = [filtered_plan(source, full, set(groups['a']), counter, DROP_A),
             filtered_plan(source, full, set(groups['b']), counter, DROP_B),
             filtered_plan(source, full, set(groups['a']) | set(groups['b']), counter, DROP_AB)]
    return [deepcopy(baseline), deepcopy(independent), deepcopy(full), *drops], groups


def replay_cache():
    marker = load(PLAN/'preparation.json')['reader_marker_sha256']
    cache = {}
    for item in load(PLAN/'private/replay_cache.json'):
        recomp.interface.parent.check_cache_entry(item, marker)
        if item['key'] in cache:
            raise ValueError('Duplicate historical request')
        cache[item['key']] = item['entry']
    return cache


def freeze():
    preparation = load(PLAN/'preparation.json')
    names = ['reader_memory/anchor_intervention_v1.py', 'tools/prepare_anchor_intervention_v1.py',
        'scripts/anchor_intervention_v1_queue.py', 'tests/test_anchor_intervention_v1.py',
        'tests/test_path_budget_curve_v1.py', 'tools/status_anchor_intervention_v1.py',
        'research/anchor_guard_controls_v1/manifest.json', *preparation['server_source_files']]
    names += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
        and 'snapshots' not in p.relative_to(PLAN).parts and p.suffix != '.pyc'
        and p.name not in {'manifest.json', 'CURRENT_STATUS.md', 'submission_receipt.json', 'STATUS_LATEST.json', 'SNAPSHOT_LATEST.json'}]
    value = dict(config=CONFIG, parent_signature=parent.verify(),
                 files={n:sha(ROOT/n) for n in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN/'manifest.json', value)
    print('FROZEN_ANCHOR_INTERVENTION', value['signature'], flush=True)


def verify():
    v = load(PLAN/'manifest.json')
    if v['config'] != CONFIG or v['signature'] != fingerprint({k:x for k,x in v.items() if k!='signature'}):
        raise ValueError('Diagnostic identity changed')
    if parent.verify() != v['parent_signature']:
        raise ValueError('Frozen parent changed')
    for n,d in v['files'].items():
        if sha(ROOT/n) != d:
            raise ValueError('Frozen input changed: '+n)
    return v['signature']


def preflight():
    sid = verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    prep = load(PLAN/'preparation.json')
    marker = prep['reader_marker_sha256']
    if sha(ROOT/'models/Qwen3.5-9B/.verified.json') != marker:
        raise ValueError('Reader model changed')
    sources, blind = load(PLAN/'public/source_inputs.json'), load(PLAN/'public/blind_paths.json')
    plans = load(PLAN/'public/qa_plans.json')
    by = {(p['question_id'],p['arm']):p for p in plans}
    expected = {(s['question_id'],arm(m)) for s in sources for m in METHODS}
    if len(sources)!=128 or len({s['native_id'] for s in sources})!=64 or len(plans)!=768 or set(by)!=expected:
        raise ValueError('Incomplete fixed panel')
    counter, traces = ExactCounter(), []
    for s in sources:
        q = s['question_id']
        cached = load(ROOT/'runs/reranker_budget_v1/selected'/(q.replace(':','_')+'.json'))
        if cached['source_sha256'] != fingerprint(s) or blind[q]['source_sha256'] != fingerprint(s):
            raise ValueError('Source or path provenance mismatch')
        if blind[q]['predicted_hops'] != len(blind[q]['seed_ids']):
            raise ValueError('Blind count changed')
        generated, trace = plans_for(s,cached['scores'],blind[q]['seed_ids'],
            by[q,arm(FULL)],by[q,arm(BASE)],by[q,arm(INDEPENDENT)],counter)
        if any(p != by[q,p['arm']] for p in generated):
            raise ValueError('Server intervention reproduction mismatch')
        traces.append(dict(question_id=q,**trace))
    if traces != load(PLAN/'public/intervention_traces.json'):
        raise ValueError('Source block definitions changed')
    tok = AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    cache, keys, maximum = replay_cache(), set(), 0
    for p in plans:
        if counter.count(p['evidence_text'])!=p['evidence_tokens'] or not 0<=p['evidence_tokens']<=256 or fingerprint(p['evidence_text'])!=p['evidence_sha256']:
            raise ValueError('Evidence identity or token contract changed')
        messages = recomp.messages_for(p)
        key = recomp.interface.parent.request_key(messages,marker)
        keys.add(key)
        if p['method'] not in NEW_METHODS and key not in cache:
            raise ValueError('Original controls must be reused')
        text = tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        maximum = max(maximum,counter.count(text))
    new = len(keys-set(cache))
    if new>384 or new!=prep['new_reader_calls_upper_bound'] or maximum+2048>32768:
        raise ValueError('Call cap or prompt capacity failed')
    immutable(RUN/'qa_plans.json',plans)
    immutable(RUN/'selection_complete.json',dict(signature=sid,plans_sha256=sha(RUN/'qa_plans.json'),conditions=768,uncached_requests=new))
    immutable(RUN/'preflight.json',dict(signature=sid,passed=True,questions=128,sources=64,conditions=768,
        unique_requests=len(keys),uncached_requests=new,full_prompt_max_tokens=maximum,
        server_selection_reproduced=True,empty_contexts=sum(not p['selected_spans'] for p in plans),
        new_training=False,new_encoder_calls=0,protected_final_data_accessed=False))
    print('ANCHOR_INTERVENTION_PREFLIGHT_OK',new,'uncached calls',flush=True)


def qa(endpoint):
    sid = verify()
    ledger = recomp.ledger
    old = ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for
    ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for = RUN,lambda:sid,replay_cache,recomp.messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for = old


def summarize(rows):
    summaries, comparisons = [], []
    for scope in ('overall','multi','single'):
        def by(method):
            return {r['question_id']:r for r in rows if r['method']==method and (scope=='overall' or r['mode']==scope)}
        for method in METHODS:
            group = list(by(method).values())
            summaries.append(dict(method=method,budget=256,scope=scope,n=len(group),
                f1_points=100*statistics.mean(r['local_native_f1'] for r in group),
                full_f1_count=sum(r['local_native_f1']==1 for r in group),
                mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in group),
                statuses=dict(Counter(r['status'] for r in group))))
        for a,b in PAIRS:
            comparisons.append(dict(method=a,comparator=b,budget=256,scope=scope,
                **recomp.interface.paired(by(a),by(b))))
    return summaries,comparisons


def mechanism_rows(rows, traces):
    by = {(r['question_id'],r['method']):r for r in rows}
    result = []
    for trace in traces:
        q = trace['question_id']
        full,a,b,ab = [by[q,m] for m in (FULL,DROP_A,DROP_B,DROP_AB)]
        normal = all(r['status']=='ok' for r in (full,a,b,ab))
        exact = full['local_native_f1']==1
        eligible = trace['both_blocks_present']
        hurt_a,hurt_b = a['local_native_f1']<1,b['local_native_f1']<1
        if not eligible: category='no_other_path_block'
        elif not normal: category='unresolved_failure'
        elif not exact: category='full_not_exact'
        elif ab['local_native_f1']==1: category='background_alone_exact'
        elif hurt_a and hurt_b: category='both_deletions_hurt'
        elif hurt_a: category='only_anchor_deletion_hurts'
        elif hurt_b: category='only_other_path_deletion_hurts'
        else: category='neither_deletion_hurts'
        result.append(dict(question_id=q,native_id=full['native_id'],mode=full['mode'],
            eligible=eligible,all_four_normal=normal,full_exact=exact,category=category,
            strict_joint_zero=eligible and normal and exact and a['local_native_f1']==0 and b['local_native_f1']==0 and ab['local_native_f1']==0,
            f1={m:by[q,m]['local_native_f1'] for m in (FULL,DROP_A,DROP_B,DROP_AB)},
            evidence_tokens={m:by[q,m]['evidence_tokens'] for m in (FULL,DROP_A,DROP_B,DROP_AB)},
            interaction_f1=full['local_native_f1']-a['local_native_f1']-b['local_native_f1']+ab['local_native_f1'],
            interaction_exact=int(exact)-int(a['local_native_f1']==1)-int(b['local_native_f1']==1)+int(ab['local_native_f1']==1),
            full_vs_global_delta=full['local_native_f1']-by[q,BASE]['local_native_f1'],
            full_vs_independent_delta=full['local_native_f1']-by[q,INDEPENDENT]['local_native_f1']))
    return result


def mechanism_summary(rows):
    summaries = []
    for scope in ('overall','multi','single'):
        group = [r for r in rows if scope=='overall' or r['mode']==scope]
        eligible_normal = [r for r in group if r['eligible'] and r['all_four_normal']]
        summaries.append(dict(scope=scope,n=len(group),eligible=sum(r['eligible'] for r in group),
            eligible_all_four_normal=len(eligible_normal),categories=dict(Counter(r['category'] for r in group)),
            strict_joint_zero=sum(r['strict_joint_zero'] for r in group),
            mean_interaction_f1_eligible_normal=statistics.mean(r['interaction_f1'] for r in eligible_normal) if eligible_normal else None,
            global_gain_questions=sum(r['full_vs_global_delta']>0 for r in group),
            joint_witness_among_global_gains=sum(r['full_vs_global_delta']>0 and r['category']=='both_deletions_hurt' for r in group)))
    return summaries


def score():
    sid = verify()
    rows = recomp.interface.parent.score_rows(load(RUN/'qa_plans.json'),list(read_jsonl(RUN/'qa/predictions.jsonl')),
        load(PLAN/'private/references.json'),sid)
    cache = replay_cache()
    for r in rows:
        if r['method'] not in NEW_METHODS:
            entry = cache[r['request_key']]
            if not r['reused'] or any(r[k]!=entry[k] for k in ('hypothesis','status','error')):
                raise ValueError('Historical control changed')
    summaries,comparisons = summarize(rows)
    mechanisms = mechanism_rows(rows,load(PLAN/'public/intervention_traces.json'))
    calls = list(read_jsonl(RUN/'qa/answer_calls.jsonl')) if (RUN/'qa/answer_calls.jsonl').exists() else []
    if len(calls)!=load(PLAN/'preparation.json')['new_reader_calls_upper_bound']:
        raise ValueError('New-call accounting mismatch')
    immutable(RUN/'scored.json',rows)
    immutable(RUN/'mechanism_rows.json',mechanisms)
    immutable(RUN/'report.json',dict(signature=sid,complete=True,scores=summaries,comparisons=comparisons,
        mechanism_summary=mechanism_summary(mechanisms),new_reader_calls=len(calls),reused_conditions=sum(r['reused'] for r in rows),
        reader_cost={k:sum(c['result'].get('usage',{}).get(k,0) for c in calls) for k in ('prompt_tokens','completion_tokens')},
        new_reader_seconds=sum(c['seconds'] for c in calls),config=CONFIG,
        paper_ready=False,independent_confirmation=False,automatic_followup=False))
    print('ANCHOR_INTERVENTION_COMPLETE',mechanism_summary(mechanisms),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','verify','preflight','qa','score'])
    p.add_argument('--endpoint',default='http://127.0.0.1:8031')
    a=p.parse_args()
    if a.mode=='qa':qa(a.endpoint)
    else:
        result=globals()[a.mode]()
        if result:print(result,flush=True)
