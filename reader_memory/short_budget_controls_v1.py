"""Supplementary controls: reselect baseline sentences under the compact cap."""
from collections import Counter
import statistics
from . import anchored_stop_replication_v1 as parent
from . import full_comparison_v1 as original

ROOT = parent.ROOT
BASE = ROOT / 'research/short_budget_controls_v1'
PLAN = parent.PLAN
RUN = ROOT / 'runs/short_budget_controls_v1'
load, sha, fp, immutable, write_json = parent.load, parent.sha, parent.fp, parent.immutable, parent.write_json
READERS, BUDGETS = parent.READERS, parent.BUDGETS
ARMS = ('short_global', 'short_ap3')
CONFIG = dict(version='short-budget-controls-v1', sources=96, readers=list(READERS), budgets=list(BUDGETS),
    arms=list(ARMS), max_answer_conditions=768, reader_output=128,
    cap='Each source/budget anchored-stop actual evidence tokens; original complete-sentence packing',
    interpretation='Supplementary same-panel mechanism control; length cap depends on the proposer, not a free deployment policy',
    criterion='Anchor minus short AP3 and anchor minus short global: both source-pooled lower CI >0 and both reader point differences >0',
    resource_caps_mib=dict(qa4=10500, qa9=19500), resource_headroom_mib=dict(qa4=1000, qa9=1000))


def verify():
    m = load(BASE / 'manifest.json')
    assert m['config'] == CONFIG and fp({k:v for k,v in m.items() if k != 'signature'}) == m['signature']
    assert parent.verify() == m['parent_signature']
    for n, d in m['files'].items():
        if sha(ROOT / n) != d:
            raise ValueError('Frozen file changed: ' + n)
    return m['signature']


def short_plans(source, scores, seed_ids, counter, cap, outer_budget):
    """Same pack/seed/guard functions, with arbitrary public content cap.

    Keep the registered outer budget on AP3 metadata; the guard uses its global
    plan's actual size internally. No edit to the frozen baseline modules.
    """
    if not isinstance(cap, int) or not 0 <= cap <= outer_budget:
        raise ValueError('Invalid public evidence cap')
    try:
        baseline, _ = original.ranker.recomp.pack(source, scores, 'recomp_sentence', counter, cap)
    except ValueError as exc:
        if not original.empty_packing_error(exc):
            raise
        return [original.fail_plan(source, m, outer_budget, str(exc))
                for m in (original.ranker.NEW_METHOD, original.anchor.MAIN)]
    try:
        candidate, _ = original.anchor.packing.seed_fill(source, scores, seed_ids, counter,
            original.anchor.PREVIOUS, baseline['evidence_tokens'])
        main, _ = original.anchor.guarded_plan(source, scores, baseline, candidate, counter,
            outer_budget, original.anchor.MAIN)
    except ValueError as exc:
        if not original.empty_packing_error(exc):
            raise
        main = original.fail_plan(source, original.anchor.MAIN, outer_budget, str(exc))
    for plan in (baseline, main):
        assert counter.count(plan['evidence_text']) == plan['evidence_tokens'] <= cap
    return [baseline, main]


def runtime():
    m = parent.runtime()
    m.BASE, m.RUN, m.SHARED_CACHE = BASE, RUN, RUN / 'request_cache'
    m.ARMS, m.verify = ARMS, verify
    m.CONFIG = dict(m.CONFIG, version=CONFIG['version'], arms=list(ARMS))
    return m


def assemble():
    signature = verify()
    bundle = load(parent.RUN / 'plans.json')
    assert bundle['signature'] == parent.verify() and bundle['selection_before_qa']
    old = {(p['question_id'], p['budget'], p['arm']):p for p in bundle['plans']}
    assert len(old) == 960
    counter = parent.parent.base.old.common.CounterTokens()
    plans = []
    for s in load(PLAN / 'public/sources.json'):
        q = s['question_id']
        ranked = load(parent.RUN / 'ranking' / (q + '.json'))
        retrieved = load(parent.RUN / 'retrieval' / (q + '.json'))
        for r in (ranked, retrieved):
            assert r['signature'] == parent.verify() and r['source_hash'] == fp(s)
        seeds = [str(x) for x in retrieved['paths'][0]]
        for b in BUDGETS:
            cap = old[q, b, 'quote_anchor']['evidence_tokens']
            # Same-cap identity establishes these are the original algorithms.
            full = short_plans(s, ranked['scores'], seeds, counter, b, b)
            for p, a in zip(full, ('global', 'ap3')):
                assert p['evidence_text'] == old[q, b, a]['evidence_text']
            short = short_plans(s, ranked['scores'], seeds, counter, cap, b)
            for a, p in zip(ARMS, short):
                plans.append(dict(question_id=q, budget=b, arm=a,
                    evidence_text=p['evidence_text'], evidence_tokens=p['evidence_tokens'],
                    status='selection_failed' if p.get('selection_error') else 'ready', reason=p.get('selection_error'),
                    selected_ids=p.get('selected_spans', []), selected_ranges=p.get('selected_ranges', {}),
                    trace=dict(content_cap=cap, unused_cap=cap-p['evidence_tokens'],
                               parent_anchor_hash=fp(old[q,b,'quote_anchor'])), private_inputs_read=False))
    assert len(plans) == 384
    immutable(RUN / 'plans.json', dict(signature=signature, parent_signature=parent.verify(),
        parent_plan_bundle_hash=fp(bundle), selection_before_qa=True, plans=plans))
    print('ASSEMBLE_COMPLETE', len(plans), flush=True)


def score():
    from .full_comparison_v1 import answer_scores
    from .discriminative_pilot_v1 import paired
    signature = verify()
    bundle = load(RUN / 'plans.json')
    plans = bundle['plans']
    answers = {(r,p['question_id'],p['budget'],p['arm']): load(RUN/'responses'/r/
        (fp([r,p['question_id'],p['budget'],p['arm']])+'.json')) for r in READERS for p in plans}
    assert len(plans) == 384 and len(answers) == 768
    old = load(parent.RUN / 'summary.json')
    assert old['complete'] and old['conditions'] == 1920 and old['signature'] == parent.verify()
    assert bundle['parent_plan_bundle_hash'] == fp(load(parent.RUN/'plans.json'))
    # Private reference access only after both matrices complete.
    refs = load(PLAN / 'private/references.json')
    records = []
    for r in READERS:
        for p in plans:
            q, b, a = p['question_id'], p['budget'], p['arm']
            response = answers[r,q,b,a]
            assert response['signature'] == signature and response['plan_hash'] == fp(p)
            em, f1 = answer_scores(response['text'], refs[q])
            records.append(dict(reader=r, question_id=q, budget=b, arm=a, em=em, f1=f1,
                evidence_tokens=p['evidence_tokens'], content_cap=p['trace']['content_cap'],
                status=response['status'], reused=response.get('reused',False), usage=response['usage']))
    idx = {(r['reader'],r['question_id'],r['budget'],r['arm']):r for r in records+old['records']}
    comparisons, groups = [], []
    for reader in (*READERS, 'pooled'):
        rr = READERS if reader == 'pooled' else (reader,)
        for budget in (*BUDGETS, 'pooled'):
            bb = BUDGETS if budget == 'pooled' else (budget,)
            for control in ARMS:
                ds = [statistics.mean(idx[r,q,b,'quote_anchor']['f1']-idx[r,q,b,control]['f1']
                      for r in rr for b in bb) for q in refs]
                comparisons.append(dict(reader=reader,budget=budget,control=control,**paired(ds)))
            for arm in ('quote_anchor', 'ap3', *ARMS):
                rows = [idx[r,q,b,arm] for r in rr for q in refs for b in bb]
                groups.append(dict(reader=reader,budget=budget,arm=arm,
                    f1_points=100*statistics.mean(x['f1'] for x in rows),
                    mean_tokens=statistics.mean(x['evidence_tokens'] for x in rows)))
    pooled = [c for c in comparisons if c['reader'] == c['budget'] == 'pooled']
    per_reader = [c for c in comparisons if c['reader'] != 'pooled' and c['budget'] == 'pooled']
    passed = all(c['ci95'][0]>0 for c in pooled) and all(c['mean_delta_f1']>0 for c in per_reader)
    fresh = [r for r in records if not r['reused'] and r['status'] in ('ok','empty')]
    summary = dict(signature=signature,parent_signature=parent.verify(),complete=True,conditions=768,
        records=records,comparisons=comparisons,groups=groups,specific_selection_criterion_met=passed,
        unique_new_reader_calls=len(fresh),reader_costs={k:sum(r['usage'].get(k,0) for r in fresh)
            for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
        statuses=dict(Counter(r['status'] for r in records)),
        caveats='Same 96-source supplementary control, not independent replication. Equal public content caps, not exact actual length. Cap derived from proposer; cannot claim a free adaptive length policy. Original parent criterion unchanged.')
    immutable(RUN/'summary.json',summary)
    print('SCORE_COMPLETE', pooled, 'selection_criterion',passed,flush=True)


def phase(name):
    if name == 'score':
        score()
    elif name == 'assemble':
        assemble()
    elif name == 'preflight':
        runtime().preflight()
    elif name in ('qa4','qa9'):
        runtime().qa('Qwen3.5-'+('4B' if name=='qa4' else '9B'))
    else:
        raise ValueError(name)
