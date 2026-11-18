"""Separately frozen exploratory QA of all 72 sources after the original gate failed."""
from copy import deepcopy
import math
import statistics
import time
from . import planning_effectiveness_v2 as original

parent = original.parent
ROOT = original.ROOT
BASE = ROOT / 'research/planning_effectiveness_supplement_v1'
INPUT_BASE = original.BASE
PLAN = BASE
RUN = ROOT / 'runs/planning_effectiveness_supplement_v1'
load, sha, fp, immutable, write_json = original.load, original.sha, original.fp, original.immutable, original.write_json
READERS, BUDGETS, MODES = original.READERS, original.BUDGETS, original.MODES
OLD_ARMS, ARMS, PAIRS = original.OLD_ARMS, original.ARMS, original.PAIRS
sources, pack, validate_plan_matrix, paired = original.sources, original.pack, original.validate_plan_matrix, original.paired
CONFIG = dict(version='planning-effectiveness-supplement-v1', sources=72, modes=list(MODES), arms=list(ARMS),
    readers=list(READERS), budgets=list(BUDGETS), inherited_proposals=216, max_proposer_calls=0,
    max_reader_conditions=1728, seed=30, bootstrap_replicates=10000, repairs=0, training=False,
    original_behavior_gate_passed=False, post_behavior_exploratory=True, automatic_expansion=False,
    reader_max_output=128, reader_max_input=4096,
    resource_caps_mib=dict(qa4=10500, qa9=19500), resource_headroom_mib=dict(qa4=1000, qa9=1000))


def verify():
    m = load(BASE / 'manifest.json')
    assert m['config'] == CONFIG and m['signature'] == fp({k: v for k, v in m.items() if k != 'signature'})
    assert m['parent_signature'] == original.verify()
    for name, digest in m['files'].items():
        assert sha(ROOT / name) == digest, name
    return m['signature']


def inherited_snapshot():
    return load(BASE / 'basis/snapshot.json')


def validity_sets(snapshot, qids):
    valid = {m: [q for q in qids if snapshot['proposals/' + fp([q, m]) + '.json']['inspection']['schema_ok']
                 and snapshot['proposals/' + fp([q, m]) + '.json']['inspection']['usable']] for m in MODES}
    return dict(valid_by_mode=valid, all_three_valid=[q for q in qids if all(q in valid[m] for m in MODES)],
        pairwise={a + '-' + b: [q for q in qids if q in valid[a] and q in valid[b]]
                  for a, b in PAIRS[:3]})


def preflight():
    from tools.summarize_planning_effectiveness_v2 import audit_snapshot
    sid = verify()
    for name, digest in load(BASE / 'manifest.json')['model_files'].items():
        assert sha(ROOT / name) == digest, name
    snapshot = inherited_snapshot()
    audited = audit_snapshot(snapshot, original.verify(), sources())
    assert audited['audit_passed'] and not audited['proposal_gate']['passed']
    assert not audited['effectiveness_tested'] and audited['new_reader_calls'] == 0
    assert validity_sets(snapshot, [s['question_id'] for s in sources()]) == load(BASE / 'validity_sets.json')
    immutable(RUN / 'preflight.json', dict(signature=sid, parent_signature=original.verify(),
        original_snapshot_sha256=sha(BASE / 'basis/snapshot.json'), inherited_proposals=216,
        new_proposer_calls=0, original_gate_passed=False, original_zero_qa_verified=True, reference_values_read=False))
    immutable(RUN / 'proposal_gate.json', snapshot['proposal_gate.json'])


def assemble():
    sid = verify()
    parent_sid = original.verify()
    counter = parent.CounterTokens()
    cached = load(INPUT_BASE / 'public/inherited_inputs.json')
    plans = deepcopy(load(INPUT_BASE / 'public/fixed_plans.json'))
    snapshot = inherited_snapshot()
    for s in sources():
        q = s['question_id']
        assert cached[q]['source_hash'] == fp(s)
        for mode in MODES:
            row = snapshot['proposals/' + fp([q, mode]) + '.json']
            assert row['signature'] == parent_sid and row['source_hash'] == fp(s)
            for budget in BUDGETS:
                plans.append(dict(pack(s, cached[q], row['inspection'], budget, counter), arm=mode))
    validate_plan_matrix(plans, [s['question_id'] for s in sources()])
    immutable(RUN / 'plans.json', dict(signature=sid, selection_before_qa=True, plans=plans))


def subset_pair(values):
    import numpy as np
    n = len(values)
    if not n:
        return dict(n=0, mean_delta_f1=None, ci95=None, wins=0, ties=0, losses=0)
    assert all(math.isfinite(v) for v in values)
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(30)
    ci = np.quantile(x[rng.integers(0, n, (10000, n))].mean(axis=1), [.025, .975]).tolist()
    return dict(n=n, mean_delta_f1=float(x.mean()), ci95=ci,
        wins=sum(v > 0 for v in values), ties=sum(v == 0 for v in values), losses=sum(v < 0 for v in values))


def sensitivity(records):
    qids = [s['question_id'] for s in sources()]
    sets = load(BASE / 'validity_sets.json')
    ix = {(v['reader'], v['question_id'], v['budget'], v['arm']): v for v in records}
    output = dict(fallback_sources={m: [q for q in qids if q not in sets['valid_by_mode'][m]] for m in MODES}, pairs={})
    for a, b in PAIRS[:3]:
        key = a + '-' + b
        valid = sets['pairwise'][key]
        affected = [q for q in qids if q not in valid]
        delta = {q: statistics.mean(ix[r, q, budget, a]['f1'] - ix[r, q, budget, b]['f1']
                  for r in READERS for budget in BUDGETS) for q in qids}
        output['pairs'][key] = dict(common_valid=subset_pair([delta[q] for q in valid]),
            all_three_valid=subset_pair([delta[q] for q in sets['all_three_valid']]),
            fallback_affected=subset_pair([delta[q] for q in affected]),
            common_valid_contribution_to_full_mean=sum(delta[q] for q in valid) / 72,
            fallback_contribution_to_full_mean=sum(delta[q] for q in affected) / 72,
            fallback_contribution_absolute_bound=len(affected) / 72)
    return output


def exploratory_decision(comparisons, per_reader, per_budget, proposal_groups, diagnostic):
    result = original.decide(comparisons, per_reader, per_budget, proposal_groups)
    stable = diagnostic['pairs']['plan-rationale']['common_valid']['mean_delta_f1']
    result['criteria']['primary_common_valid_nonnegative'] = stable is not None and stable >= 0
    if not all(result['criteria'].values()):
        result['decision'] = 'stop_expansion_of_this_mechanism_claim'
    elif result['plan_over_rationale_output_ratio'] > 1.25:
        result['decision'] = 'exploratory_signal_requires_effort_matched_followup'
    else:
        result['decision'] = 'exploratory_signal_warrants_fresh_validation'
    return result


def phase(name):
    if name in ('qa4', 'qa9'):
        qa('Qwen3.5-' + ('4B' if name == 'qa4' else '9B'))
    else:
        globals()[name]()


# The QA identity/cache and full-matrix scoring logic are inherited unchanged.
def qa(reader):
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid=verify();bundle=load(RUN/'plans.json');assert bundle['signature']==sid and bundle['selection_before_qa']
    assert (RUN/'assemble_complete.json').exists()
    questions={s['question_id']:s['question'] for s in sources()};validate_plan_matrix(bundle['plans'],questions)
    replay=load(INPUT_BASE/'replay/reader_cache.json');torch_setup()
    backend=TransformersBackend(ROOT/'models'/reader,max_input_tokens=4096)
    for i,p in enumerate(sorted(bundle['plans'],key=lambda p:fp([30,p['question_id'],p['budget'],p['arm']])),1):
        q,b,a=p['question_id'],p['budget'],p['arm'];out=RUN/'responses'/reader/(fp([reader,q,b,a])+'.json');assert not out.exists()
        base=dict(signature=sid,reader=reader,question_id=q,budget=b,arm=a,plan_hash=fp(p))
        if p['status']!='ready':immutable(out,dict(base,status='selection_failed',text='',usage={}));continue
        assert len(backend.tokenizer.encode(p['evidence_text'],add_special_tokens=False))==p['evidence_tokens']
        msg=parent.role.old.common.messages(questions[q],p['evidence_text']);content=fp([reader,msg,128])
        prompt=backend.tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert len(backend.tokenizer.encode(prompt,add_special_tokens=False))<=4096
        key=fp([sid,content]);cache=RUN/'request_cache'/reader/(key+'.json');start=cache.with_suffix('.started.json')
        reused=cache.exists();origin='within_run' if reused else None
        if reused:
            saved=load(cache);assert saved['signature']==sid and saved['content_key']==content;result=saved['result']
        elif content in replay:
            e=replay[content];assert e['reader']==reader and e['messages']==msg and e['max_output']==128
            result=e['result'];reused=True;origin='frozen_hotpot_confirmation_v1'
            immutable(cache,dict(signature=sid,content_key=content,result=result,replayed_from=e['parent_signature']))
        else:
            assert not start.exists(),'Unknown-call replay forbidden'
            immutable(start,dict(signature=sid,content_key=content,time=time.time()));result=generate(backend,msg,128)
            immutable(cache,dict(signature=sid,content_key=content,result=result))
        immutable(out,dict(base,status='ok' if result['text'].strip() else 'empty',request_key=key,reused=reused,reuse_origin=origin,**result))
        print('QA',reader,i,'/864',flush=True)


def score():
    sid=verify();gate=load(RUN/'proposal_gate.json');assert gate == inherited_snapshot()['proposal_gate.json']
    bundle=load(RUN/'plans.json');assert bundle['signature']==sid and bundle['selection_before_qa']
    plans=bundle['plans'];qids=[s['question_id'] for s in sources()];validate_plan_matrix(plans,qids);responses={}
    for r in READERS:
        for p in plans:
            q,b,a=p['question_id'],p['budget'],p['arm'];v=load(RUN/'responses'/r/(fp([r,q,b,a])+'.json'))
            assert v['signature']==sid and v['plan_hash']==fp(p) and v['status'] in ('ok','empty','selection_failed')
            assert (v['reader'],v['question_id'],v['budget'],v['arm'])==(r,q,b,a);responses[r,q,b,a]=v
    assert len(responses)==1728
    refs=load(INPUT_BASE/'private/references.json');assert set(refs)==set(qids);metric=parent.official_metrics();records=[]
    for r in READERS:
        for p in plans:
            q,b,a=p['question_id'],p['budget'],p['arm'];v=responses[r,q,b,a];gold=refs[q]['answer']
            records.append(dict(reader=r,question_id=q,budget=b,arm=a,f1=float(metric['f1_score'](v['text'],gold)[0]),
                em=float(metric['exact_match_score'](v['text'],gold)),evidence_tokens=p['evidence_tokens'],status=v['status'],
                reused=v.get('reused',False),usage=v['usage'],finish_reason=v.get('finish_reason')))
    ix={(v['reader'],v['question_id'],v['budget'],v['arm']):v for v in records}
    def values(a,c,rr=READERS,bb=BUDGETS):return [statistics.mean(ix[r,q,b,a]['f1']-ix[r,q,b,c]['f1'] for r in rr for b in bb) for q in qids]
    comparisons=[dict(arm=a,control=c,**paired(values(a,c))) for a,c in PAIRS]
    per_reader={r:paired(values('plan','rationale',(r,))) for r in READERS}
    per_budget={str(b):paired(values('plan','rationale',bb=(b,))) for b in BUDGETS}
    groups=[]
    for r in ('pooled',*READERS):
        for b in ('pooled',*BUDGETS):
            for a in ARMS:
                rs=[v for v in records if v['arm']==a and (r=='pooled' or v['reader']==r) and (b=='pooled' or v['budget']==b)]
                groups.append(dict(reader=r,budget=b,arm=a,f1_points=100*statistics.mean(v['f1'] for v in rs),
                    em_points=100*statistics.mean(v['em'] for v in rs),mean_tokens=statistics.mean(v['evidence_tokens'] for v in rs)))
    fresh=[v for v in records if not v['reused'] and v['status'] in ('ok','empty')]
    diagnostic=sensitivity(records)
    result=dict(signature=sid,post_behavior_exploratory=True,original_behavior_gate_passed=False,new_proposer_calls=0,inherited_proposals=216,sensitivity=diagnostic,complete=True,conditions=len(records),sources=72,exposed_development=True,effectiveness_tested=True,
        comparisons=comparisons,groups=groups,per_reader=per_reader,per_budget=per_budget,records=records,proposal_gate=gate,
        new_reader_calls=len(fresh),new_reader_costs={k:sum(v['usage'].get(k,0) for v in fresh) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
        inherited_costs=load(INPUT_BASE/'replay/costs.json'),**exploratory_decision(comparisons,per_reader,per_budget,gate['groups'],diagnostic))
    immutable(RUN/'summary.json',result);print('SCORE_COMPLETE',result['decision'],comparisons[:2],flush=True)

