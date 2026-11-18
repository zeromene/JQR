"""Frozen, answer-free construction of matched support repair diagnostics."""
from copy import deepcopy
from itertools import combinations
import math
import statistics
from . import matched_quote_fourarm_v2 as previous

parent = previous.parent
ROOT = previous.ROOT
BASE = ROOT / 'research/oracle_evidence_repair_v1'
RUN = ROOT / 'runs/oracle_evidence_repair_v1'
load, sha, fp, immutable, write_json = previous.load, previous.sha, previous.fp, previous.immutable, previous.write_json
READERS, BUDGETS = previous.READERS, previous.BUDGETS
ARMS = ('direct', 'oracle_patch', 'ranked_patch', 'support_only')
CONFIG = dict(version='oracle-evidence-repair-v1', sources=480, arms=list(ARMS), readers=list(READERS),
    budgets=list(BUDGETS), reader_max_input=4096, reader_max_output=128, max_reader_conditions=7680,
    new_proposer_calls=0, training=False, automatic_expansion=False, seed=30, bootstrap_replicates=10000,
    patch_sentences=1, length_tolerance_tokens=8, retain_at_least_one_baseline_interval=True,
    primary_minimum=.02, matched_control_minimum=.01, minimum_patch_sources=96,
    resource_caps_mib=dict(qa4=10500, qa9=19500), resource_headroom_mib=dict(qa4=1000, qa9=1000))


def verify():
    m = load(BASE / 'manifest.json')
    assert m['config'] == CONFIG and m['signature'] == fp({k: v for k, v in m.items() if k != 'signature'})
    assert m['parent_signature'] == previous.verify()
    for name, digest in m['files'].items(): assert sha(ROOT / name) == digest, name
    return m['signature']


def sources():
    ss = load(BASE / 'public/sources.json')
    assert len(ss) == len({s['question_id'] for s in ss}) == CONFIG['sources']
    return ss


def rank_key(s, unit, ranks):
    para = next(p for p in s['paragraphs'] if p['source_id'] == unit['source'])
    hits = [ranks[para['source_id'] + ':span:' + str(i)] for i, sp in enumerate(para['spans'])
            if sp['start'] < unit['end'] and unit['start'] < sp['end']]
    assert hits
    return min(hits), int(unit['source']), unit['start'], unit['end']


def omitted(s, unit, ranges):
    text = s['paragraphs'][int(unit['source'])]['text']
    return not any(not text[i].isspace() and any(a <= i < b for a, b in ranges.get(unit['source'], []))
                   for i in range(unit['start'], unit['end']))


def render(s, units, counter):
    ranges = {}
    for u in units:
        text = s['paragraphs'][int(u['source'])]['text']
        assert text[u['start']:u['end']] == u['text']
        ranges.setdefault(u['source'], []).append([u['start'], u['end']])
    ranges = {pid: parent.merge_ranges(s['paragraphs'][int(pid)]['text'], rr) for pid, rr in ranges.items()}
    text = parent.render_ranges(s, ranges) if ranges else ''
    return dict(selected_ranges=ranges, evidence_text=text, evidence_tokens=counter.count(text))


def base_units(s, baseline):
    return [dict(source=pid, start=a, end=b, text=s['paragraphs'][int(pid)]['text'][a:b])
            for pid, ranges in baseline['selected_ranges'].items() for a, b in ranges]


def material(s, budget, arm, rendered, trace):
    assert 0 < rendered['evidence_tokens'] <= budget
    return dict(question_id=s['question_id'], budget=budget, arm=arm, **rendered, selected_ids=[], trace=trace,
                status='ready', reason=None, diagnostic_gold_used=arm != 'direct')


def direct_plan(baseline):
    p = deepcopy(baseline); p['arm'] = 'direct'
    return p


def construct(s, baseline, native, labels, scores, counter):
    """No answer or model prediction arguments. Controls may naturally select gold."""
    budget = baseline['budget']; assert baseline['question_id'] == s['question_id'] and baseline['arm'] == 'direct'
    original = base_units(s, baseline); original_render = render(s, original, counter)
    assert original_render['evidence_text'] == baseline['evidence_text']
    assert original_render['evidence_tokens'] == baseline['evidence_tokens'] <= budget
    order = parent.development.parent.packing.sentence_order(s, scores)
    ranks = {key: i for i, key in enumerate(order)}
    key = lambda u: rank_key(s, u, ranks)
    by = {u['id']: u for u in native}; assert len(by) == len(native)
    gold = sorted([by[uid] for uid in labels], key=key)
    assert gold
    missing = [u for u in gold if omitted(s, u, baseline['selected_ranges'])]
    candidates = sorted([u for u in native if omitted(s, u, baseline['selected_ranges'])], key=key)
    trace = dict(rule='one_fully_omitted_native_support_sentence', missing_support_count=len(missing),
        target_id=missing[0]['id'] if missing else None, changed=False, reason='no_fully_omitted_support_sentence',
        common_removed_ranges=[], common_retained_ranges=baseline['selected_ranges'], added_id=None,
        gold_selected_condition_and_target_length=True, answer_values_used=False)
    patches = None
    if missing:
        target = missing[0]
        # Source-agnostic removal order: reverse existing relevance, never gold labels.
        removal = sorted(range(len(original)), key=lambda i: (key(original[i]), i), reverse=True)
        for n in range(len(original)):
            removed_indices = set(removal[:n])
            kept = [u for i, u in enumerate(original) if i not in removed_indices]
            assert kept
            oracle = render(s, kept + [target], counter)
            if oracle['evidence_tokens'] > budget: continue
            feasible = []
            for unit in candidates:
                trial = render(s, kept + [unit], counter)
                if trial['evidence_tokens'] <= budget and abs(trial['evidence_tokens'] - oracle['evidence_tokens']) <= CONFIG['length_tolerance_tokens']:
                    feasible.append((unit, trial))
            assert feasible and any(u['id'] == target['id'] for u, _ in feasible)
            chosen, ordinary = feasible[0]
            shared = dict(trace, changed=True, reason='patched',
                common_removed_ranges=[[original[i]['source'], original[i]['start'], original[i]['end']] for i in removal[:n]],
                common_retained_ranges=render(s, kept, counter)['selected_ranges'],
                length_gap_tokens=oracle['evidence_tokens'] - ordinary['evidence_tokens'],
                feasible_control_candidates=len(feasible), original_evidence_tokens=baseline['evidence_tokens'])
            patches = [material(s, budget, 'oracle_patch', oracle, dict(shared, added_id=target['id'])),
                       material(s, budget, 'ranked_patch', ordinary, dict(shared, added_id=chosen['id']))]
            break
        if patches is None: trace['reason'] = 'target_cannot_fit_while_retaining_a_baseline_interval'
    if patches is None:
        patches = [material(s, budget, arm, original_render, trace) for arm in ('oracle_patch', 'ranked_patch')]

    # Budget-constrained annotation reference, not a mathematical performance bound.
    # Exhaustive only over <=6 marked support sentences, no answer scores consulted.
    best, chosen, chosen_render = None, None, None
    for count in range(1, len(gold) + 1):
        for indices in combinations(range(len(gold)), count):
            selected = [gold[i] for i in indices]; rr = render(s, selected, counter)
            if rr['evidence_tokens'] > budget: continue
            objective = (count, -sum(key(u)[0] for u in selected), -rr['evidence_tokens'], tuple(-i for i in indices))
            if best is None or objective > best: best, chosen, chosen_render = objective, selected, rr
    if chosen is None:
        chosen = []; chosen_render = original_render
    reference = material(s, budget, 'support_only', chosen_render,
        dict(rule='maximum_count_annotated_sentences_then_existing_rank', selected_support_ids=[u['id'] for u in chosen],
             total_support_sentences=len(gold), all_support_fit=len(chosen) == len(gold),
             fallback_to_direct=not bool(chosen), answer_values_used=False))
    return [direct_plan(baseline), *patches, reference]


def validate_plans(plans, qids):
    assert len(qids) == len(set(qids)) == CONFIG['sources']
    expected = {(q, b, arm) for q in qids for b in BUDGETS for arm in ARMS}
    assert len(plans) == len(expected) and {(p['question_id'], p['budget'], p['arm']) for p in plans} == expected
    assert all(p['status'] == 'ready' and 0 < p['evidence_tokens'] <= p['budget'] for p in plans)
    ix = {(p['question_id'], p['budget'], p['arm']): p for p in plans}
    for q in qids:
        for b in BUDGETS:
            oracle, ranked = ix[q, b, 'oracle_patch'], ix[q, b, 'ranked_patch']
            assert abs(oracle['evidence_tokens'] - ranked['evidence_tokens']) <= CONFIG['length_tolerance_tokens']
            for field in ('changed', 'target_id', 'common_removed_ranges', 'common_retained_ranges'):
                assert oracle['trace'][field] == ranked['trace'][field]


def preflight():
    from transformers import AutoTokenizer
    sid = verify()
    for name, digest in load(BASE / 'manifest.json')['model_files'].items(): assert sha(ROOT / name) == digest, name
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/Qwen3.5-9B'), local_files_only=True, trust_remote_code=False)
    ss = sources(); byq = {s['question_id']: s for s in ss}; plans = load(BASE / 'public/plans.json')
    validate_plans(plans, byq); lengths = []
    for p in plans:
        msg = parent.role.old.common.messages(byq[p['question_id']]['question'], p['evidence_text'])
        prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        lengths.append(len(tok.encode(prompt, add_special_tokens=False)))
        assert len(tok.encode(p['evidence_text'], add_special_tokens=False)) == p['evidence_tokens']
    assert max(lengths) <= CONFIG['reader_max_input']
    immutable(RUN / 'preflight.json', dict(signature=sid, materials=len(plans), maximum_prompt=max(lengths),
              answer_values_read=False, gold_used_only_in_frozen_material_construction=True))


def assemble():
    sid = verify(); plans = load(BASE / 'public/plans.json'); validate_plans(plans, [s['question_id'] for s in sources()])
    immutable(RUN / 'plans.json', dict(signature=sid, selection_before_qa=True, plans=plans))


def qa(reader):
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid = verify(); bundle = load(RUN / 'plans.json'); assert bundle['signature'] == sid and bundle['selection_before_qa']
    assert (RUN / 'assemble_complete.json').exists()
    questions = {s['question_id']: s['question'] for s in sources()}; validate_plans(bundle['plans'], questions)
    replay = load(BASE / 'replay/reader_cache.json'); torch_setup()
    backend = TransformersBackend(ROOT / 'models' / reader, max_input_tokens=4096)
    for i, p in enumerate(sorted(bundle['plans'], key=lambda p: fp([30, p['question_id'], p['budget'], p['arm']])), 1):
        q, b, arm = p['question_id'], p['budget'], p['arm']; out = RUN / 'responses' / reader / (fp([reader, q, b, arm]) + '.json')
        assert not out.exists()
        msg = parent.role.old.common.messages(questions[q], p['evidence_text']); content = fp([reader, msg, 128]); key = fp([sid, content])
        prompt = backend.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert len(backend.tokenizer.encode(prompt, add_special_tokens=False)) <= 4096
        assert len(backend.tokenizer.encode(p['evidence_text'], add_special_tokens=False)) == p['evidence_tokens']
        cache = RUN / 'request_cache' / reader / (key + '.json'); start = cache.with_suffix('.started.json')
        reused = cache.exists(); origin = 'within_run' if reused else None
        if reused:
            saved = load(cache); assert saved['signature'] == sid and saved['content_key'] == content; result = saved['result']
        elif content in replay:
            entry = replay[content]; assert (entry['reader'], entry['messages'], entry['max_output']) == (reader, msg, 128)
            result = entry['result']; reused = True; origin = 'frozen_parent_request'
            immutable(cache, dict(signature=sid, content_key=content, result=result, replayed_from=entry['parent_signature']))
        else:
            assert not start.exists(), 'Unknown request retry forbidden'
            immutable(start, dict(signature=sid, content_key=content, time=__import__('time').time()))
            result = generate(backend, msg, 128); immutable(cache, dict(signature=sid, content_key=content, result=result))
        immutable(out, dict(signature=sid, reader=reader, question_id=q, budget=b, arm=arm, plan_hash=fp(p),
            status='ok' if result['text'].strip() else 'empty', request_key=key, reused=reused, reuse_origin=origin, **result))
        print('QA', reader, i, '/' + str(len(bundle['plans'])), flush=True)


def paired(values):
    import numpy as np
    assert values and all(math.isfinite(v) for v in values)
    x = np.array(values); rng = np.random.default_rng(30)
    means = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
    return dict(n=len(x), mean_delta_f1=float(x.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
        wins=sum(v > 0 for v in values), ties=sum(v == 0 for v in values), losses=sum(v < 0 for v in values))


PAIRS = (('oracle_patch', 'direct'), ('oracle_patch', 'ranked_patch'), ('ranked_patch', 'direct'), ('support_only', 'direct'))


def analyze_records(records, qids, plans):
    ix = {(v['reader'], v['question_id'], v['budget'], v['arm']): v for v in records}
    assert len(ix) == len(records) == len(qids) * 16
    def values(a, c, qq=qids, rr=READERS, bb=BUDGETS):
        return [statistics.mean(ix[r, q, b, a]['f1'] - ix[r, q, b, c]['f1'] for r in rr for b in bb) for q in qq]
    comparisons = [dict(arm=a, control=c, **paired(values(a, c))) for a, c in PAIRS]
    per_reader = {r: [dict(arm=a, control=c, **paired(values(a, c, rr=(r,)))) for a, c in PAIRS[:2]] for r in READERS}
    per_budget = {str(b): [dict(arm=a, control=c, **paired(values(a, c, bb=(b,)))) for a, c in PAIRS[:2]] for b in BUDGETS}
    patch_ids = [q for q in qids if any(p['question_id'] == q and p['arm'] == 'oracle_patch' and p['trace']['changed'] for p in plans)]
    # Defined only by pre-QA material feasibility, never by predictions or answer scores.
    changed_subset = [dict(arm=a, control=c, **paired(values(a, c, qq=patch_ids))) for a, c in PAIRS[:2]] if patch_ids else []
    groups = []
    for reader in ('pooled', *READERS):
        for budget in ('pooled', *BUDGETS):
            for arm in ARMS:
                rows = [v for v in records if v['arm'] == arm and (reader == 'pooled' or v['reader'] == reader) and (budget == 'pooled' or v['budget'] == budget)]
                groups.append(dict(reader=reader, budget=budget, arm=arm, f1_points=100*statistics.mean(v['f1'] for v in rows),
                    em_points=100*statistics.mean(v['em'] for v in rows), mean_tokens=statistics.mean(v['evidence_tokens'] for v in rows)))
    first, second = comparisons[:2]
    criteria = dict(two_points_over_direct=first['mean_delta_f1'] >= .02 and first['ci95'][0] > 0,
        one_point_over_matched_ranked=second['mean_delta_f1'] >= .01 and second['ci95'][0] > 0,
        reader_budget_directions_nonnegative=all(v['mean_delta_f1'] >= 0 for group in list(per_reader.values()) + list(per_budget.values()) for v in group),
        sufficient_intervention_sources=len(patch_ids) >= CONFIG['minimum_patch_sources'])
    decision = 'evidence_repair_opportunity_requires_deployable_method' if all(criteria.values()) else 'stop_single_sentence_repair_expansion_review_capacity'
    return dict(comparisons=comparisons, per_reader=per_reader, per_budget=per_budget, groups=groups,
        changed_source_subset=changed_subset, patch_source_ids=patch_ids, criteria=criteria, decision=decision,
        oracle_diagnostic=True, deployable_method=False, publication_ready=False, automatic_expansion=False)


def score():
    sid = verify(); bundle = load(RUN / 'plans.json'); assert bundle['signature'] == sid and bundle['selection_before_qa']
    plans = bundle['plans']; qids = [s['question_id'] for s in sources()]; validate_plans(plans, qids); responses = {}
    for r in READERS:
        for p in plans:
            q, b, arm = p['question_id'], p['budget'], p['arm']; v = load(RUN / 'responses' / r / (fp([r, q, b, arm]) + '.json'))
            assert (v['signature'], v['reader'], v['question_id'], v['budget'], v['arm'], v['plan_hash']) == (sid, r, q, b, arm, fp(p))
            assert v['status'] in ('ok', 'empty'); responses[r, q, b, arm] = v
    assert len(responses) == len(qids)*16
    refs = load(BASE / 'private/references.json'); assert set(refs) == set(qids); metric = parent.official_metrics(); records = []
    for r in READERS:
        for p in plans:
            q, b, arm = p['question_id'], p['budget'], p['arm']; v = responses[r, q, b, arm]
            records.append(dict(reader=r, question_id=q, budget=b, arm=arm, f1=float(metric['f1_score'](v['text'], refs[q]['answer'])[0]),
                em=float(metric['exact_match_score'](v['text'], refs[q]['answer'])), evidence_tokens=p['evidence_tokens'],
                status=v['status'], reused=v['reused'], usage=v['usage']))
    fresh = [v for v in records if not v['reused']]
    summary = dict(signature=sid, complete=True, sources=len(qids), conditions=len(records), records=records,
        exposed_development=True, new_proposer_calls=0, new_reader_calls=len(fresh),
        new_reader_costs={k: sum(v['usage'].get(k, 0) for v in fresh) for k in ('prompt_tokens', 'completion_tokens', 'elapsed_seconds')},
        preparation=load(BASE / 'preparation.json'), **analyze_records(records, qids, plans))
    immutable(RUN / 'summary.json', summary); print('SCORE_COMPLETE', summary['decision'], flush=True)


def phase(name):
    if name in ('qa4', 'qa9'): qa('Qwen3.5-' + ('4B' if name == 'qa4' else '9B'))
    else: globals()[name]()
