"""Fixed 256/512/1024 evidence budgets; cached paths, shared sentence ranking."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

from . import path_budget_controls_v1 as parent
from .schema import fingerprint, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/path_budget_curve_v1'
RUN = ROOT / 'runs/path_budget_curve_v1'
recomp = parent.recomp
load, sha, immutable = parent.load, parent.sha, parent.immutable
BUDGETS = (256, 512, 1024)
METHODS = ('recomp_sentence', 'recomp_paragraph_fill', 'independent_seed_fill',
           'path_seed_fill', 'beam_path_compress')
FILLED = METHODS[:-1]
CONFIG = dict(version='path-budget-curve-v1', sources=64, questions=128,
    budgets=list(BUDGETS), methods=list(METHODS), max_conditions=1920,
    max_new_reader_calls=1920, reader='Qwen3.5-9B', reader_variant='format_only',
    output_tokens=2048, decoding='greedy-thinking_false', retries=0,
    primary='three-budget macro-average; also report each fixed budget and multi/single separately',
    actual_length_target='each question/budget RECOMP sentence actual evidence tokens',
    sentence_scores='same cached NQ RECOMP FP32 scores over original20 paragraphs',
    paragraph_control='greedy complete paragraphs by max sentence rank, then same sentence fill',
    independent_control='max-sentence-score paragraph ranking, same predicted count and packing',
    path_overflow='ordinary RECOMP sentence compression inside predicted path; no relation protection',
    diagnostic='beam_path_compress has no external fill; variable actual tokens, not length-matched primary',
    source_order='best retained sentence rank; native text order inside source',
    training=False, new_retrieval=False, new_encoder_calls=0,
    independent_confirmation=False, dependency_protection_implemented=False,
    protected_final_data_accessed=False, automatic_followup=False)


def arm(method, budget):
    if method not in METHODS or budget not in BUDGETS:
        raise ValueError('Unknown fixed method or budget')
    return method + '__b' + str(budget)


def paragraph_seeds(source, scores, counter, budget):
    """Ordinary coarse-first control; reject whole blocks that cannot fit."""
    by = {p['source_id']: p for p in source['paragraphs']}
    seen, selected, decisions = set(), {}, []
    for uid in parent.sentence_order(source, scores):
        pid = uid.split(':')[0]
        if pid in seen:
            continue
        seen.add(pid)
        proposal = dict(selected)
        proposal[pid] = [[0, len(by[pid]['text'])]]
        ordered = parent.canonical_ranges(source, scores, proposal)
        n = counter.count(recomp.render_ranges(source, ordered))
        decisions.append(dict(source_id=pid, proposed_tokens=n, accepted=n <= budget))
        if n <= budget:
            selected = proposal
    return list(selected), decisions


def compressed_path(source, scores, seed_ids, counter, budget):
    if not seed_ids or len(seed_ids) != len(set(seed_ids)):
        raise ValueError('Invalid blind path')
    by = {p['source_id']: p for p in source['paragraphs']}
    ranges = {pid: [[0, len(by[pid]['text'])]] for pid in seed_ids}
    n = counter.count(recomp.render_ranges(source, parent.canonical_ranges(source, scores, ranges)))
    overflow = n > budget
    if overflow:
        sub = dict(source, paragraphs=[by[pid] for pid in seed_ids])
        units = {u for u, _ in recomp.units(sub)}
        compact, _ = recomp.pack(sub, {u: scores[u] for u in units}, 'recomp_sentence', counter, budget)
        ranges = compact['selected_ranges']
    plan = parent.make_plan(source, scores, ranges, counter, 'beam_path_compress', budget)
    return plan, dict(seed_ids=seed_ids, seed_tokens=n, seed_overflow=overflow,
                     actual_tokens=plan['evidence_tokens'], no_external_fill=True)


def plans_for(source, scores, blind_path, counter, budget):
    """No references, true hop counts, answers or prior F1 values are inputs."""
    if budget not in BUDGETS:
        raise ValueError('Unregistered fixed budget')
    baseline, _ = recomp.pack(source, scores, 'recomp_sentence', counter, budget)
    target = baseline['evidence_tokens']
    normal = parent.make_plan(source, scores, baseline['selected_ranges'], counter, 'recomp_sentence', target)
    if normal != baseline:
        raise ValueError('Sentence baseline presentation differs')
    independent = parent.independent_seeds(source, scores, len(blind_path))
    paragraphs, coarse_trace = paragraph_seeds(source, scores, counter, target)
    result, traces = [baseline], []
    for name, seeds in [('recomp_paragraph_fill', paragraphs),
                        ('independent_seed_fill', independent), ('path_seed_fill', blind_path)]:
        p, trace = parent.seed_fill(source, scores, seeds, counter, name, target)
        if name == 'recomp_paragraph_fill':
            trace['paragraph_admission'] = coarse_trace
        result.append(p)
        traces.append(dict(method=name, budget=budget, **trace))
    p, trace = compressed_path(source, scores, blind_path, counter, budget)
    result.append(p)
    traces.append(dict(method='beam_path_compress', budget=budget, **trace))
    for p in result:
        method = p['arm']
        p.update(arm=arm(method, budget), method=method, evidence_budget=budget)
        if not p['evidence_tokens'] <= budget:
            raise ValueError('Evidence exceeds fixed budget')
    return result, traces


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
    previous_signature = parent.verify()
    preparation = load(PLAN / 'preparation.json')
    for name, expected in preparation['server_source_files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Prepared snapshot and server source differ: ' + name)
    names = ['reader_memory/path_budget_curve_v1.py', 'tools/prepare_path_budget_curve_v1.py',
             'tests/test_path_budget_curve_v1.py', 'scripts/path_budget_curve_v1_queue.py',
             'research/path_budget_controls_v1/manifest.json']
    names += list(preparation['server_source_files'])
    names += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
              and 'snapshots' not in p.relative_to(PLAN).parts and p.suffix != '.pyc'
              and p.name not in {'manifest.json', 'CURRENT_STATUS.md', 'submission_receipt.json'}]
    value = dict(config=CONFIG, parent_signature=previous_signature,
                 files={n: sha(ROOT / n) for n in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_PATH_BUDGET_CURVE', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Budget-curve identity changed')
    if parent.verify() != value['parent_signature']:
        raise ValueError('Frozen prior experiment changed')
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
        raise ValueError('Reader changed')
    plans = load(PLAN / 'public/qa_plans.json')
    questions = {p['question_id'] for p in plans}
    expected = {(q, arm(m, b)) for q in questions for b in BUDGETS for m in METHODS}
    if len(questions) != 128 or len(plans) != 1920 or {(p['question_id'], p['arm']) for p in plans} != expected:
        raise ValueError('Incomplete 128-by-3-by-5 matrix')
    cache, keys, prompts = replay_cache(), set(), []
    for p in plans:
        if p['arm'] != arm(p['method'], p['evidence_budget']) or counter.count(p['evidence_text']) != p['evidence_tokens'] or p['evidence_tokens'] > p['evidence_budget'] or fingerprint(p['evidence_text']) != p['evidence_sha256']:
            raise ValueError('Evidence budget, tokenization or identity mismatch')
        messages = recomp.messages_for(p)
        key = recomp.interface.parent.request_key(messages, marker)
        keys.add(key)
        full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        n = counter.count(full)
        if n + 2048 > 32768:
            raise ValueError('Reader capacity overflow; no truncation')
        prompts.append(dict(question_id=p['question_id'], arm=p['arm'], request_key=key, full_prompt_tokens=n))
    new = len(keys - set(cache))
    if new != preparation['new_reader_calls_upper_bound'] or new > 1920 or not preparation['filled_length_gate_passed']:
        raise ValueError('Prepared request or length audit mismatch')
    immutable(RUN / 'qa_plans.json', plans)
    immutable(RUN / 'selection_complete.json', dict(signature=sid, plans_sha256=sha(RUN / 'qa_plans.json'),
        conditions=1920, uncached_requests=new))
    immutable(RUN / 'preflight.json', dict(signature=sid, passed=True, questions=128, conditions=1920,
        budgets=list(BUDGETS), methods=list(METHODS), uncached_requests=new, unique_requests=len(keys),
        length_statistics=preparation['length_statistics'], filled_length_gate_passed=True,
        prompt_identities=prompts, full_prompt_max_tokens=max(p['full_prompt_tokens'] for p in prompts),
        new_training=False, protected_final_data_accessed=False))
    print('PATH_BUDGET_CURVE_PREFLIGHT_OK', new, 'uncached calls', flush=True)


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
    refs = load(parent.previous.parent.PLAN / 'private/development_references.json')
    plans = load(RUN / 'qa_plans.json')
    scored = recomp.interface.parent.score_rows(plans, list(read_jsonl(RUN / 'qa/predictions.jsonl')), refs, sid)
    summaries, comparisons = [], []
    pairs = [('path_seed_fill', m) for m in METHODS if m != 'path_seed_fill']
    pairs += [('beam_path_compress', 'recomp_sentence'), ('recomp_paragraph_fill', 'recomp_sentence')]
    for budget in (*BUDGETS, 'macro'):
        for scope in ('overall', 'multi', 'single'):
            def by(method):
                return {(r['question_id'], r['evidence_budget']): r for r in scored if r['method'] == method
                        and (budget == 'macro' or r['evidence_budget'] == budget)
                        and (scope == 'overall' or r['mode'] == scope)}
            for method in METHODS:
                rows = list(by(method).values())
                tokens = [r['evidence_tokens'] for r in rows]
                summaries.append(dict(method=method, budget=budget, scope=scope, n=len(rows),
                    f1_points=100*statistics.mean(r['local_native_f1'] for r in rows),
                    full_f1_count=sum(r['local_native_f1'] == 1 for r in rows),
                    mean_evidence_tokens=statistics.mean(tokens), min_evidence_tokens=min(tokens),
                    median_evidence_tokens=statistics.median(tokens), max_evidence_tokens=max(tokens),
                    statuses=dict(Counter(r['status'] for r in rows)),
                    all_support_sources_touched=sum(set(map(str, refs[r['question_id']]['support_indices'])) <=
                        {u.split(':')[0] for u in r['selected_spans']} for r in rows)))
            for a, b in pairs:
                comparisons.append(dict(method=a, comparator=b, budget=budget, scope=scope,
                                        **recomp.interface.paired(by(a), by(b))))
    path = RUN / 'qa/answer_calls.jsonl'
    calls = list(read_jsonl(path)) if path.exists() else []
    preparation = load(PLAN / 'preparation.json')
    if len(calls) > preparation['new_reader_calls_upper_bound']:
        raise ValueError('New inference budget exceeded')
    immutable(RUN / 'scored.json', scored)
    immutable(RUN / 'report.json', dict(signature=sid, complete=True, scores=summaries, comparisons=comparisons,
        new_reader_calls=len(calls), reused_conditions=sum(r['reused'] for r in scored),
        reader_cost={k: sum(c['result'].get('usage', {}).get(k, 0) for c in calls) for k in ('prompt_tokens', 'completion_tokens')},
        new_reader_seconds=sum(c['seconds'] for c in calls), preparation=preparation,
        same_supervision_independent_control=False, novelty_proven=False,
        independent_confirmation=False, paper_ready=False, automatic_followup=False, config=CONFIG))
    print('PATH_BUDGET_CURVE_COMPLETE', [r for r in summaries if r['scope'] == 'overall'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'verify', 'preflight', 'qa', 'score'])
    p.add_argument('--endpoint', default='http://127.0.0.1:8026')
    args = p.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    else:
        value = globals()[args.mode]()
        if value:
            print(value, flush=True)
