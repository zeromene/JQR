"""Equal-token natural-material swaps separate old background from new source material."""
import argparse
from collections import Counter
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import re
import statistics

from . import source_cover_controls_v1 as parent
from . import anchor_intervention_v1 as deletion
from .schema import fingerprint, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/context_factorial_v1'
RUN = ROOT / 'runs/context_factorial_v1'
load, sha, immutable, recomp = parent.load, parent.sha, parent.immutable, parent.recomp
packing = parent.packing
FULL, COVER = parent.FULL, parent.COVER_PATH
B0D0, B1D0 = 'low_background_low_other', 'old_background_low_other'
B0D1, B1D1 = 'low_background_new_other', 'old_background_new_other'
NEW_METHODS = (B0D0, B1D0, B0D1, B1D1)
METHODS = (FULL, COVER, *NEW_METHODS)
PAIRS = ((B1D0, B0D0), (B1D1, B0D1), (B0D0, B0D1), (B1D0, B1D1))
CONFIG = dict(version='context-factorial-v1', questions=128, sources=64,
    methods=list(METHODS), conditions=768, max_new_reader_calls=512,
    historical_budget=256, diagnostic_budget=512, reader='Qwen3.5-9B',
    reader_variant='format_only', output_tokens=2048, decoding='greedy-thinking_false', retries=0,
    background='old-only sentences in sources retained by both old and covered path',
    new_other='new-only sentences from sources absent in old protected path',
    fixed_core='union of old and covered selected spans excluding those two blocks',
    replacements='whole original sentences, bottom half Qwen ranks, outside all original/covered sources and duplicate titles',
    match='same source count per slot; exact rendered slot tokens and exact complete evidence tokens in all four cells',
    pool='up to three contiguous eligible sentences; lowest-max-score first 256 blocks; at most two sources per slot; first16 matches',
    no_match='all diagnostic cells copy original full context, marked inapplicable',
    selection_uses_answers=False, synthetic_diagnostic_only=True,
    training=False, new_retrieval=False, new_encoder_calls=0,
    independent_confirmation=False, protected_final_data_accessed=False, automatic_followup=False)


def arm(method):
    if method not in METHODS:
        raise ValueError('Unregistered factorial arm')
    return method + ('__b512' if method in NEW_METHODS else '__b256')


def render_units(source, units):
    """Render genuine offsets in original paragraph order, including honest source titles."""
    units = set(units)
    valid = {u for u, _ in recomp.units(source)}
    if not units <= valid:
        raise ValueError('Unknown original sentence')
    ranges = {}
    for para in source['paragraphs']:
        pid = para['source_id']
        pieces = [[s['start'], s['end']] for i, s in enumerate(para['spans']) if pid + ':span:' + str(i) in units]
        if pieces:
            ranges[pid] = recomp.merge_ranges(para['text'], pieces)
    return recomp.render_ranges(source, ranges)


def evidence(core, background, other):
    return '[Fixed material]\n' + core + '\n\n[Additional material A]\n' + background + '\n\n[Additional material B]\n' + other


def blocks_for(source, old, covered):
    o, n = set(old['selected_spans']), set(covered['selected_spans'])
    shared_sources = set(old['selected_ranges']) & set(covered['selected_ranges'])
    b = {u for u in o - n if u.split(':')[0] in shared_sources}
    d = {u for u in n - o if u.split(':')[0] not in old['selected_ranges']}
    k = (o | n) - b - d
    if b & d or b & k or d & k or not (o & n) <= k:
        raise ValueError('Factor blocks or common facts overlap')
    return dict(core=sorted(k), background=sorted(b), other=sorted(d),
        original_only=sorted(o - n), covered_only=sorted(n - o),
        core_common_spans=sorted(o & n))


def control_pool(source, scores, excluded, counter, maximum):
    order = packing.sentence_order(source, scores)
    low = set(order[(len(order) + 1) // 2:])
    by_uid = dict(recomp.units(source))
    selected_sources = {u.split(':')[0] for u in excluded}
    titles = {deletion.literal_key(p['title']) for p in source['paragraphs'] if p['source_id'] in selected_sources}
    copy_keys = {deletion.literal_key(by_uid[u]) for u in excluded}
    choices = []
    for pi, para in enumerate(source['paragraphs']):
        pid = para['source_id']
        if pid in selected_sources or deletion.literal_key(para['title']) in titles:
            continue
        for start in range(len(para['spans'])):
            units = []
            for end in range(start, min(start + 3, len(para['spans']))):
                uid = pid + ':span:' + str(end)
                if uid not in low or deletion.literal_key(by_uid[uid]) in copy_keys:
                    break
                units.append(uid)
                text = render_units(source, units)
                tokens = counter.count(text)
                if tokens <= maximum:
                    choices.append(dict(units=list(units), source=pid, text=text, tokens=tokens,
                        max_score=max(scores[u] for u in units), paragraph_order=pi, start=start, end=end))
    choices.sort(key=lambda x: (x['max_score'], x['paragraph_order'], x['start'], x['end']))
    return choices[:256]


def matching_controls(source, pool, target_tokens, source_count, counter):
    if source_count not in (1, 2):
        return []
    result = []
    if source_count == 1:
        candidates = ((x,) for x in pool)
    else:
        candidates = ((x, y) for x, y in combinations(pool, 2) if x['source'] != y['source']
                      and abs(x['tokens'] + y['tokens'] - target_tokens) <= 16)
    for parts in candidates:
        units = sorted({u for p in parts for u in p['units']})
        text = render_units(source, units)
        if counter.count(text) == target_tokens:
            result.append(dict(units=units, sources=sorted({p['source'] for p in parts}), text=text,
                tokens=target_tokens, max_score=max(p['max_score'] for p in parts)))
            if len(result) == 16:
                break
    return result


def view_plan(full, method, text, trace, slots, counter):
    result = deepcopy(full)
    result.update(method=method, arm=arm(method), evidence_budget=512, diagnostic_only=True,
        synthetic_counterfactual=True, selected_ids=[], selected_ranges={}, selected_spans=[],
        fully_preserved_sources=[], evidence_text=text, evidence_tokens=counter.count(text),
        evidence_sha256=fingerprint(text), fixed_core_sha256=trace.get('core_sha256'),
        factorial_eligible=trace['eligible'], diagnostic_slot_spans=slots,
        offsets_reference_original=True, natural_task_main_result=False)
    return result


def plans_for(source, scores, full, covered, counter):
    if not counter.exact or any(p['question_id'] != source['question_id'] or p['evidence_budget'] != 256 for p in (full, covered)):
        raise ValueError('Exact counter and original question contexts required')
    packing.sentence_order(source, scores)
    groups = blocks_for(source, full, covered)
    trace = dict(question_id=source['question_id'], eligible=False, reason=None, **groups)
    selected = None
    if not groups['background']:
        trace['reason'] = 'no_old_same_source_background'
    elif not groups['other']:
        trace['reason'] = 'no_new_source_material'
    else:
        core, bg, other = [render_units(source, groups[k]) for k in ('core', 'background', 'other')]
        original_text = evidence(core, bg, other)
        target = counter.count(original_text)
        counts = [len({u.split(':')[0] for u in groups[k]}) for k in ('background', 'other')]
        trace.update(core_sha256=fingerprint(core), original_background_sha256=fingerprint(bg),
            original_other_sha256=fingerprint(other), target_tokens=target,
            background_tokens=counter.count(bg), other_tokens=counter.count(other),
            slot_source_counts=counts)
        if target > 512:
            trace['reason'] = 'combined_diagnostic_over_512'
        elif max(counts) > 2:
            trace['reason'] = 'slot_more_than_two_sources'
        else:
            pool = control_pool(source, scores, set(full['selected_spans']) | set(covered['selected_spans']),
                counter, max(counter.count(bg), counter.count(other)))
            bm = matching_controls(source, pool, counter.count(bg), counts[0], counter)
            dm = matching_controls(source, pool, counter.count(other), counts[1], counter)
            trace.update(control_pool_size=len(pool), background_matches=len(bm), other_matches=len(dm))
            trace['reason'] = 'no_disjoint_exact_token_controls'
            for b in bm:
                for d in dm:
                    if set(b['sources']) & set(d['sources']):
                        continue
                    cells = [evidence(core, b['text'], d['text']), evidence(core, bg, d['text']),
                             evidence(core, b['text'], other), original_text]
                    if not all(counter.count(x) == target for x in cells):
                        continue
                    trace.update(eligible=True, reason='exact_matched_natural_slots',
                        background_control=b, other_control=d)
                    selected = cells
                    break
                if selected is not None:
                    break
    applicable = selected is not None
    if selected is None:
        selected = [full['evidence_text']] * 4
    slots = [dict(core=groups['core'], background=trace.get('background_control', {}).get('units', []),
                  other=trace.get('other_control', {}).get('units', [])),
             dict(core=groups['core'], background=groups['background'],
                  other=trace.get('other_control', {}).get('units', [])),
             dict(core=groups['core'], background=trace.get('background_control', {}).get('units', []), other=groups['other']),
             dict(core=groups['core'], background=groups['background'], other=groups['other'])]
    if not applicable:
        slots = [{} for _ in NEW_METHODS]
    new = [view_plan(full, m, text, trace, slot, counter) for m, text, slot in zip(NEW_METHODS, selected, slots)]
    if len({p['evidence_tokens'] for p in new}) != 1 or any(p['evidence_tokens'] > 512 for p in new):
        raise ValueError('Complete factorial evidence is not exactly token matched')
    return [deepcopy(full), deepcopy(covered), *new], trace


def content_for(source, units):
    by = dict(recomp.units(source))
    pids = {u.split(':')[0] for u in units}
    return '\n'.join([*(p['title'] for p in source['paragraphs'] if p['source_id'] in pids), *(by[u] for u in units)])


def contains_answer(text, reference):
    return any(a.strip() and re.search(r'(?<!\w)' + re.escape(a.strip()) + r'(?!\w)', text, re.I)
               for a in [reference['answer'], *reference['aliases']])


def reference_qc(source, trace, reference):
    """Post-construction analysis flags only; never change an entity, control or plan."""
    if not trace['eligible']:
        return dict(question_id=source['question_id'], public_eligible=False,
                    reference_literal_in_core=False, control_answer_overlap=False)
    controls = trace['background_control']['units'] + trace['other_control']['units']
    return dict(question_id=source['question_id'], public_eligible=True,
        reference_literal_in_core=contains_answer(content_for(source, trace['core']), reference),
        control_answer_overlap=contains_answer(content_for(source, controls), reference))


def preflight():
    sid = verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    prep = load(PLAN / 'preparation.json')
    marker = prep['reader_marker_sha256']
    if sha(ROOT / 'models/Qwen3.5-9B/.verified.json') != marker:
        raise ValueError('Reader model changed')
    sources, plans = load(PLAN / 'public/source_inputs.json'), load(PLAN / 'public/qa_plans.json')
    by = {(p['question_id'], p['method']): p for p in plans}
    expected = {(s['question_id'], m) for s in sources for m in METHODS}
    if len(sources) != 128 or len({s['native_id'] for s in sources}) != 64 or len(plans) != 768 or set(by) != expected:
        raise ValueError('Incomplete fixed panel')
    counter, traces = ExactCounter(), []
    for source in sources:
        q = source['question_id']
        scores = load(ROOT / 'runs/reranker_budget_v1/selected' / (q.replace(':', '_') + '.json'))
        if scores['source_sha256'] != fingerprint(source):
            raise ValueError('Source/score provenance changed')
        generated, trace = plans_for(source, scores['scores'], by[q, FULL], by[q, COVER], counter)
        if any(p != by[q, p['method']] for p in generated):
            raise ValueError('Server factorial construction mismatch')
        traces.append(trace)
    if traces != load(PLAN / 'public/factorial_traces.json'):
        raise ValueError('Factorial trace reproduction mismatch')
    refs = load(PLAN / 'private/references.json')
    if [reference_qc(s, t, refs[s['question_id']]) for s, t in zip(sources, traces)] != load(PLAN / 'private/qc_flags.json'):
        raise ValueError('Post-construction QC mismatch')
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/Qwen3.5-9B'), local_files_only=True, trust_remote_code=False)
    cache, keys, maximum = replay_cache(), set(), 0
    for p in plans:
        if counter.count(p['evidence_text']) != p['evidence_tokens'] or not 0 < p['evidence_tokens'] <= p['evidence_budget'] or fingerprint(p['evidence_text']) != p['evidence_sha256']:
            raise ValueError('Evidence identity or budget changed')
        if p['method'] in NEW_METHODS:
            others = [by[p['question_id'], m] for m in NEW_METHODS]
            if len({x['evidence_tokens'] for x in others}) != 1:
                raise ValueError('Exact full-evidence token matching failed')
        messages = recomp.messages_for(p)
        key = recomp.interface.parent.request_key(messages, marker)
        keys.add(key)
        if p['method'] not in NEW_METHODS and key not in cache:
            raise ValueError('Historical conditions must be reused')
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        maximum = max(maximum, counter.count(text))
    new = len(keys - set(cache))
    if new > 512 or new != prep['new_reader_calls_upper_bound'] or maximum + 2048 > 32768:
        raise ValueError('Inference cap or prompt capacity failed')
    immutable(RUN / 'qa_plans.json', plans)
    immutable(RUN / 'selection_complete.json', dict(signature=sid, conditions=768,
        plans_sha256=sha(RUN / 'qa_plans.json'), uncached_requests=new))
    immutable(RUN / 'preflight.json', dict(signature=sid, passed=True, questions=128, sources=64,
        conditions=768, unique_requests=len(keys), uncached_requests=new, full_prompt_max_tokens=maximum,
        server_selection_reproduced=True, exact_diagnostic_token_matching=True,
        eligible_questions=sum(t['eligible'] for t in traces), new_training=False,
        new_encoder_calls=0, protected_final_data_accessed=False))
    print('CONTEXT_FACTORIAL_PREFLIGHT_OK', new, 'uncached calls', flush=True)


def summarize(rows, traces, qc):
    summaries, comparisons, interactions = [], [], []
    trace_by = {t['question_id']: t for t in traces}
    qc_by = {t['question_id']: t for t in qc}
    for population in ('all_conditions', 'public_matched', 'core_answer_present_controls_clear'):
        allowed = {t['question_id'] for t in traces if population == 'all_conditions' or
                   t['eligible'] and (population == 'public_matched' or
                   qc_by[t['question_id']]['reference_literal_in_core'] and not qc_by[t['question_id']]['control_answer_overlap'])}
        for scope in ('overall', 'multi', 'single'):
            def by(method):
                return {r['question_id']: r for r in rows if r['method'] == method and r['question_id'] in allowed
                        and (scope == 'overall' or r['mode'] == scope)}
            methods = METHODS if population == 'all_conditions' else NEW_METHODS
            for method in methods:
                group = list(by(method).values())
                summaries.append(dict(population=population, method=method,
                    budget=512 if method in NEW_METHODS else 256, scope=scope, n=len(group),
                    f1_points=100 * statistics.mean(r['local_native_f1'] for r in group) if group else None,
                    full_f1_count=sum(r['local_native_f1'] == 1 for r in group),
                    mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in group) if group else None,
                    statuses=dict(Counter(r['status'] for r in group))))
            # No-op-heavy aggregates and old 256 contexts are not mechanism comparisons.
            if population == 'all_conditions' or not by(B0D0):
                continue
            for a, b in PAIRS:
                comparisons.append(dict(population=population, method=a, comparator=b, scope=scope,
                    **recomp.interface.paired(by(a), by(b))))
            a, b = {}, {}
            for q, row in by(B0D0).items():
                four = [by(m)[q] for m in NEW_METHODS]
                delta = four[3]['local_native_f1'] - four[2]['local_native_f1'] - four[1]['local_native_f1'] + four[0]['local_native_f1']
                status = 'ok' if all(r['status'] == 'ok' for r in four) else 'failed'
                a[q] = dict(native_id=row['native_id'], status=status, local_native_f1=delta)
                b[q] = dict(native_id=row['native_id'], status=status, local_native_f1=0.0)
            interactions.append(dict(population=population, scope=scope,
                contrast='B1D1-B0D1-B1D0+B0D0', **recomp.interface.paired(a, b)))
    mechanisms = []
    by = {(r['question_id'], r['method']): r for r in rows}
    for q, trace in trace_by.items():
        four = [by[q, m] for m in NEW_METHODS]
        values = [r['local_native_f1'] for r in four]
        normal = all(r['status'] == 'ok' for r in four)
        flags = qc_by[q]
        clean = trace['eligible'] and flags['reference_literal_in_core'] and not flags['control_answer_overlap'] and normal
        mechanisms.append(dict(question_id=q, native_id=four[0]['native_id'], mode=four[0]['mode'],
            public_eligible=trace['eligible'], reason=trace['reason'], all_four_normal=normal,
            reference_literal_in_core=flags['reference_literal_in_core'], control_answer_overlap=flags['control_answer_overlap'],
            original_answer_f1=dict(zip(NEW_METHODS, values)), clean_candidate=clean,
            background_rescue=clean and values[1] == 1 and values[0] < 1,
            new_material_hurts_with_background=clean and values[1] == 1 and values[3] < 1,
            strict_background_buffers=clean and values[0] == values[1] == values[3] == 1 and values[2] < 1,
            original_full_f1=by[q, FULL]['local_native_f1'],
            original_cover_f1=by[q, COVER]['local_native_f1']))
    return summaries, comparisons, interactions, mechanisms


def score():
    sid = verify()
    rows = recomp.interface.parent.score_rows(load(RUN / 'qa_plans.json'),
        list(read_jsonl(RUN / 'qa/predictions.jsonl')), load(PLAN / 'private/references.json'), sid)
    cache = replay_cache()
    for row in rows:
        if row['method'] not in NEW_METHODS:
            entry = cache[row['request_key']]
            if not row['reused'] or any(row[k] != entry[k] for k in ('hypothesis', 'status', 'error')):
                raise ValueError('Historical outcome changed')
    summaries, comparisons, interactions, mechanisms = summarize(rows,
        load(PLAN / 'public/factorial_traces.json'), load(PLAN / 'private/qc_flags.json'))
    calls = list(read_jsonl(RUN / 'qa/answer_calls.jsonl')) if (RUN / 'qa/answer_calls.jsonl').exists() else []
    if len(calls) != load(PLAN / 'preparation.json')['new_reader_calls_upper_bound']:
        raise ValueError('New call accounting mismatch')
    immutable(RUN / 'scored.json', rows)
    immutable(RUN / 'mechanism_rows.json', mechanisms)
    immutable(RUN / 'report.json', dict(signature=sid, complete=True, scores=summaries,
        comparisons=comparisons, factorial_interactions=interactions,
        mechanism_summary=dict(public_eligible=sum(t['public_eligible'] for t in mechanisms),
            clean_candidates=sum(t['clean_candidate'] for t in mechanisms),
            background_rescue=sum(t['background_rescue'] for t in mechanisms),
            new_material_hurts_with_background=sum(t['new_material_hurts_with_background'] for t in mechanisms),
            strict_background_buffers=sum(t['strict_background_buffers'] for t in mechanisms)),
        new_reader_calls=len(calls), reused_conditions=sum(r['reused'] for r in rows),
        reader_cost={k: sum(c['result'].get('usage', {}).get(k, 0) for c in calls) for k in ('prompt_tokens', 'completion_tokens')},
        new_reader_seconds=sum(c['seconds'] for c in calls), config=CONFIG,
        synthetic_diagnostic_only=True, paper_ready=False, independent_confirmation=False, automatic_followup=False))
    print('CONTEXT_FACTORIAL_COMPLETE', flush=True)


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
    names = ['reader_memory/context_factorial_v1.py', 'tools/prepare_context_factorial_v1.py',
        'scripts/context_factorial_v1_queue.py', 'tests/test_context_factorial_v1.py',
        'tools/status_context_factorial_v1.py', 'research/source_cover_controls_v1/manifest.json',
        *prep['server_source_files']]
    names += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
        and 'snapshots' not in p.relative_to(PLAN).parts and p.suffix != '.pyc'
        and p.name not in {'manifest.json', 'CURRENT_STATUS.md', 'submission_receipt.json',
                          'STATUS_LATEST.json', 'SNAPSHOT_LATEST.json'}]
    value = dict(config=CONFIG, parent_signature=parent.verify(),
        files={n: sha(ROOT / n) for n in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_CONTEXT_FACTORIAL', value['signature'], flush=True)


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


def qa(endpoint):
    sid, ledger = verify(), recomp.ledger
    old = ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for
    ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = RUN, lambda: sid, replay_cache, recomp.messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = old


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'verify', 'preflight', 'qa', 'score'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8034')
    args = parser.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    else:
        value = globals()[args.mode]()
        if value:
            print(value, flush=True)
