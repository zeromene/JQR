"""Fresh two-hop replication of fixed-proposer anchored stopping."""
from collections import Counter
import json
import statistics
from . import matched_proposer_v1 as parent

ROOT = parent.ROOT
BASE = ROOT / 'research/anchored_stop_replication_v1'
PLAN = BASE / 'confirmation'
RUN = ROOT / 'runs/anchored_stop_replication_v1'
load, sha, fp, immutable, write_json = parent.load, parent.sha, parent.fp, parent.immutable, parent.write_json
READERS = parent.READERS
BUDGETS = (256, 512)
ARMS = ('global', 'ap3', 'quote_joint', 'quote_anchor', 'quote_cover')
CONFIG = dict(version='anchored-stop-replication-v1', sources=96, hops=2, proposer='Qwen3.5-9B',
    readers=list(READERS), budgets=list(BUDGETS), arms=list(ARMS), primary_arm='quote_anchor',
    max_proposer_calls=192, max_answer_conditions=1920, proposer_output=2048, reader_output=128,
    criterion='Source primary >= .02, lower CI >0, both reader differences >0, both budget mean token ratios <=.8',
    resource_caps_mib=dict(retrieve=4000, rank=9000, propose9=20500, qa4=10500, qa9=19500),
    resource_headroom_mib=dict(retrieve=1000, rank=1000, propose9=500, qa4=1000, qa9=1000))


def verify():
    m = load(BASE / 'manifest.json')
    assert m['config'] == CONFIG and fp({k: v for k, v in m.items() if k != 'signature'}) == m['signature']
    for n, d in m['files'].items():
        if sha(ROOT / n) != d:
            raise ValueError('Frozen file changed: ' + n)
    return m['signature']


def runtime():
    m = parent.runtime('Qwen3.5-9B')
    m.BASE, m.PLAN, m.RUN, m.PANEL = BASE, PLAN, RUN, 'confirmation'
    m.CONFIG = dict(m.CONFIG, version=CONFIG['version'], questions={'development': 96, 'confirmation': 96}, arms=list(ARMS))
    m.ARMS, m.verify, m.SHARED_CACHE = ARMS, verify, RUN / 'request_cache'
    return m


def score():
    from .full_comparison_v1 import answer_scores
    common = parent.base.old.common
    signature = verify()
    plans = load(RUN / 'plans.json')['plans']
    responses = {(reader, p['question_id'], p['budget'], p['arm']): load(RUN / 'responses' / reader /
        (fp([reader, p['question_id'], p['budget'], p['arm']]) + '.json')) for reader in READERS for p in plans}
    assert len(plans) == 960 and len(responses) == 1920
    refs = load(PLAN / 'private/references.json')
    assert len(refs) == 96 and all(r['hops'] == 2 for r in refs.values())
    records = []
    for reader in READERS:
        for p in plans:
            q = p['question_id']
            r = responses[reader, q, p['budget'], p['arm']]
            assert r['signature'] == signature and r['plan_hash'] == fp(p)
            em, f1 = answer_scores(r['text'], refs[q])
            records.append(dict(reader=reader, question_id=q, budget=p['budget'], arm=p['arm'], em=em, f1=f1,
                evidence_tokens=p['evidence_tokens'], status=r['status'], reused=r.get('reused', False), usage=r['usage'],
                finish_reason=r.get('finish_reason'), fallback=p['trace'].get('fallback')))
    idx = {(r['reader'], r['question_id'], r['budget'], r['arm']): r for r in records}
    def values(control, readers=READERS, budgets=BUDGETS):
        return [statistics.mean(idx[r, q, b, 'quote_anchor']['f1'] - idx[r, q, b, control]['f1']
                for r in readers for b in budgets) for q in refs]
    primary = common.paired(values('ap3'))
    per_reader = {r: common.paired(values('ap3', (r,))) for r in READERS}
    mechanism = {a: common.paired(values(a)) for a in ('quote_joint', 'quote_cover')}
    token_ratios = {str(b): statistics.mean(p['evidence_tokens'] for p in plans if p['budget'] == b and p['arm'] == 'quote_anchor') /
                    statistics.mean(p['evidence_tokens'] for p in plans if p['budget'] == b and p['arm'] == 'ap3') for b in BUDGETS}
    passed = primary['mean_delta_f1'] >= .02 and primary['ci95'][0] > 0 and all(
        p['mean_delta_f1'] > 0 for p in per_reader.values()) and all(x <= .8 for x in token_ratios.values())
    groups, comparisons = [], []
    for reader in READERS:
        for budget in (*BUDGETS, 'pooled'):
            bs = BUDGETS if budget == 'pooled' else (budget,)
            for arm in ARMS:
                rs = [r for r in records if r['reader'] == reader and r['budget'] in bs and r['arm'] == arm]
                groups.append(dict(reader=reader, budget=budget, arm=arm,
                    f1_points=100*statistics.mean(r['f1'] for r in rs), em_points=100*statistics.mean(r['em'] for r in rs),
                    mean_tokens=statistics.mean(r['evidence_tokens'] for r in rs)))
            for control in ('ap3', 'global', 'quote_joint', 'quote_cover'):
                comparisons.append(dict(reader=reader, budget=budget, control=control,
                    **common.paired(values(control, (reader,), bs))))
    calls = [r for r in records if not r['reused'] and r['status'] in ('ok', 'empty')]
    summary = dict(signature=signature, complete=True, conditions=len(records), records=records, primary=primary,
        per_reader_primary=per_reader, mechanisms=mechanism, token_ratios=token_ratios, efficacy_compactness_passed=passed,
        group_means=groups, paired_comparisons=comparisons, proposal_gate=load(RUN / 'proposal_gate.json'),
        unique_new_reader_calls=len(calls), reader_costs={k: sum(r['usage'].get(k, 0) for r in calls)
            for k in ('prompt_tokens', 'completion_tokens', 'elapsed_seconds')},
        status_counts=dict(Counter(r['status'] for r in records)), finish_reasons=dict(Counter(r['finish_reason'] for r in records)),
        caveats='New source-isolated TWO-HOP train-file panel only. Candidate chosen after prior test; present protocol frozen before new outputs. Same-family readers; descriptive unadjusted intervals. Mechanisms are separate claims.')
    immutable(RUN / 'summary.json', summary)
    print('SCORE_COMPLETE', json.dumps({k: summary[k] for k in ('primary', 'per_reader_primary', 'token_ratios', 'efficacy_compactness_passed')}), flush=True)


def phase(name):
    if name == 'score':
        score()
    elif name in ('qa4', 'qa9'):
        runtime().qa('Qwen3.5-' + ('4B' if name == 'qa4' else '9B'))
    elif name == 'propose9':
        runtime().propose()
    elif name in ('preflight', 'retrieve', 'rank', 'assemble'):
        getattr(runtime(), name)()
    else:
        raise ValueError(name)
