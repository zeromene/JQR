"""Exposed-panel ablation: proposal content versus missing-role fallback."""
from collections import Counter
from copy import deepcopy
import statistics
from . import granularity_factorial_v1 as parent

ROOT=parent.ROOT
BASE=ROOT/'research/coverage_gate_diagnostic_v1'
PLAN=BASE/'development'
RUN=ROOT/'runs/coverage_gate_diagnostic_v1'
load,sha,fp,immutable,write_json=parent.load,parent.sha,parent.fp,parent.immutable,parent.write_json
READERS=parent.READERS
BUDGETS=(256,512)
ARMS=('role_gate','role_no_gate','plain_own_gate','plain_role_gate')
CONFIG=dict(version='coverage-gate-diagnostic-v1',sources=96,exposed_development_panel=True,
    readers=list(READERS),budgets=list(BUDGETS),arms=list(ARMS),reader_output=128,
    max_new_reader_calls=92,max_answer_conditions=1536,new_proposer_calls=0,
    primary='role_gate minus role_no_gate, source mean over both readers and budgets',
    criterion='Descriptive development signal only: primary lower CI >0 and both reader point differences >0',
    resource_caps_mib=dict(qa4=10500,qa9=19500),resource_headroom_mib=dict(qa4=1000,qa9=1000))


def verify():
    m=load(BASE/'manifest.json')
    assert m['config']==CONFIG and fp({k:v for k,v in m.items() if k!='signature'})==m['signature']
    assert parent.verify()==m['parent_signature']
    for name,digest in m['files'].items():
        assert sha(ROOT/name)==digest, 'Frozen file changed: '+name
    return m['signature']


def covered(proposal):
    cs=proposal['candidates']
    represented=set().union(*(set(c['roles']) for c in cs)) if cs else set()
    return bool(cs) and represented==set(range(len(proposal['roles'])))


def without_coverage_gate(proposal):
    # Change only metadata used by the gate. Raw quote ranges and text survive.
    result=deepcopy(proposal)
    result['roles']=[dict(need='A container for the existing quotes',parents=[])] if result['candidates'] else []
    for c in result['candidates']:c['roles']=[0]
    return result


def make_plans():
    cached=load(parent.PLAN/'public/cached_inputs.json')
    old={(p['question_id'],p['budget'],p['arm']):p for p in load(PLAN/'public/parent_plans.json')}
    counter=parent.parent.parent.base.old.common.CounterTokens()
    plans=[]
    for s in load(PLAN/'public/sources.json'):
        q=s['question_id'];pr=cached[q]['proposal'];complete=covered(pr)
        for b in BUDGETS:
            # Preserve old baseline selected_spans needed by the anchor packer.
            baseline,ap3=parent.short.short_plans(s,cached[q]['ranking']['scores'],
                [str(x) for x in cached[q]['retrieval']['paths'][0]],counter,b,b)
            for p,a in ((baseline,'global'),(ap3,'ap3')):
                assert p['evidence_text']==old[q,b,a]['evidence_text']
            no_gate,tr=parent.anchored.quote_plan(s,cached[q]['ranking']['scores'],baseline,ap3,
                without_coverage_gate(pr),counter,'quote_anchor')
            if complete:assert no_gate['evidence_text']==old[q,b,'anchored']['evidence_text']
            for arm in ARMS:
                if arm=='role_no_gate':
                    p=dict(question_id=q,budget=b,arm=arm,evidence_text=no_gate['evidence_text'],
                        evidence_tokens=counter.count(no_gate['evidence_text']),status='ready',reason=None,
                        selected_ids=no_gate.get('selected_spans',[]),selected_ranges=no_gate.get('selected_ranges',{}),
                        trace=dict(tr),private_inputs_read=False)
                else:
                    origin='anchored' if arm=='role_gate' else 'ap3' if arm=='plain_role_gate' and not complete else 'plain_anchor'
                    p=deepcopy(old[q,b,origin]);p['arm']=arm
                    p['trace']=dict(p['trace'],parent_arm=origin)
                    if arm=='plain_role_gate' and not complete:p['trace']['fallback']='transferred_missing_role_gate'
                p['trace']['role_coverage_gate_passed']=complete
                assert counter.count(p['evidence_text'])==p['evidence_tokens']<=b and not p['private_inputs_read']
                plans.append(p)
    assert len(plans)==768
    return plans


def runtime():
    m=parent.runtime()
    m.BASE,m.PLAN,m.RUN,m.PANEL=BASE,PLAN,RUN,'development'
    m.ARMS,m.BUDGETS,m.verify,m.SHARED_CACHE=ARMS,BUDGETS,verify,RUN/'request_cache'
    m.CONFIG=dict(m.CONFIG,version=CONFIG['version'],arms=list(ARMS),budgets=list(BUDGETS))
    return m


def assemble():
    plans=make_plans()
    assert fp(plans)==load(PLAN/'allocation.json')['plans_hash']
    immutable(RUN/'plans.json',dict(signature=verify(),plans=plans,selection_before_qa=True))


def score():
    from .full_comparison_v1 import answer_scores
    sid=verify();plans=load(RUN/'plans.json')['plans']
    responses={(r,p['question_id'],p['budget'],p['arm']):load(RUN/'responses'/r/(fp([r,p['question_id'],p['budget'],p['arm']])+'.json')) for r in READERS for p in plans}
    assert len(responses)==1536
    # No reference access until the entire matrix exists.
    refs=load(PLAN/'private/references.json');records=[]
    for r in READERS:
        for p in plans:
            q,b,a=p['question_id'],p['budget'],p['arm'];response=responses[r,q,b,a]
            assert response['signature']==sid and response['plan_hash']==fp(p)
            em,f1=answer_scores(response['text'],refs[q])
            records.append(dict(reader=r,question_id=q,budget=b,arm=a,em=em,f1=f1,
                evidence_tokens=p['evidence_tokens'],status=response['status'],reused=response.get('reused',False),
                usage=response['usage'],fallback=p['trace'].get('fallback')))
    idx={(r['reader'],r['question_id'],r['budget'],r['arm']):r for r in records}
    comparisons=[];groups=[]
    pairs=(('role_gate','role_no_gate'),('plain_role_gate','plain_own_gate'),('role_gate','plain_role_gate'),('role_gate','plain_own_gate'))
    for reader in (*READERS,'pooled'):
        rr=READERS if reader=='pooled' else (reader,)
        for budget in (*BUDGETS,'pooled'):
            bb=BUDGETS if budget=='pooled' else (budget,)
            for a,c in pairs:
                ds=[statistics.mean(idx[r,q,b,a]['f1']-idx[r,q,b,c]['f1'] for r in rr for b in bb) for q in refs]
                comparisons.append(dict(reader=reader,budget=budget,arm=a,control=c,**parent.parent.parent.base.old.common.paired(ds)))
            for a in ARMS:
                rows=[idx[r,q,b,a] for r in rr for q in refs for b in bb]
                groups.append(dict(reader=reader,budget=budget,arm=a,f1_points=100*statistics.mean(x['f1'] for x in rows),
                    em_points=100*statistics.mean(x['em'] for x in rows),mean_tokens=statistics.mean(x['evidence_tokens'] for x in rows)))
    primary=next(x for x in comparisons if x['reader']=='pooled' and x['budget']=='pooled' and x['control']=='role_no_gate')
    individual=[x for x in comparisons if x['reader']!='pooled' and x['budget']=='pooled' and x['control']=='role_no_gate']
    passed=primary['ci95'][0]>0 and all(x['mean_delta_f1']>0 for x in individual)
    fresh=[r for r in records if not r['reused'] and r['status'] in ('ok','empty')]
    assert len(fresh)<=92
    summary=dict(signature=sid,complete=True,conditions=1536,records=records,primary=primary,comparisons=comparisons,
        groups=groups,development_rule_passed=passed,new_reader_calls=len(fresh),
        reader_costs={k:sum(r['usage'].get(k,0) for r in fresh) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
        statuses=dict(Counter(r['status'] for r in records)),
        caveats='Post-factorial exposed-panel diagnostic. Syntactic missing-role gate, not semantic verifier. Transfer uses BOTH proposer outputs and is not a free deployable plain baseline. No independent confirmation.')
    immutable(RUN/'summary.json',summary)
    print('SCORE_COMPLETE',primary,'rule',passed,flush=True)


def phase(name):
    if name=='preflight':runtime().preflight()
    elif name=='assemble':assemble()
    elif name in ('qa4','qa9'):runtime().qa('Qwen3.5-'+('4B' if name=='qa4' else '9B'))
    elif name=='score':score()
    else:raise ValueError(name)
