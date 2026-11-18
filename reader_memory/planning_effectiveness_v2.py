"""Fixed 72-source effect screen after the separately completed behavior pilot."""
from copy import deepcopy
import math
import statistics
import time
from . import planning_behavior_v2 as behavior
previous=behavior.previous;parent=behavior.parent
ROOT=parent.ROOT;BASE=ROOT/'research/planning_effectiveness_v2';PLAN=BASE;RUN=ROOT/'runs/planning_effectiveness_v2'
load,sha,fp,immutable,write_json=parent.load,parent.sha,parent.fp,parent.immutable,parent.write_json
READERS=parent.READERS;BUDGETS=parent.BUDGETS;MODES=behavior.MODES
OLD_ARMS=('original','role_common','budget_plain');ARMS=MODES+OLD_ARMS
CONFIG=dict(version='planning-effectiveness-v2',sources=72,modes=list(MODES),arms=list(ARMS),readers=list(READERS),budgets=list(BUDGETS),
    max_proposer_calls=216,max_reader_conditions=1728,seed=30,bootstrap_replicates=10000,generation_cap=2048,repairs=0,
    proposer_max_input=32768,reader_max_output=128,reader_max_input=4096,training=False,automatic_expansion=False,
    minimum_compliant_sources=69,minimum_quote_match=.9,
    resource_caps_mib=dict(propose9=20500,qa4=10500,qa9=19500),resource_headroom_mib=dict(propose9=500,qa4=1000,qa9=1000))
PAIRS=(('plan','rationale'),('plan','direct'),('rationale','direct'),('plan','role_common'),('plan','budget_plain'),
    ('original','direct'),('role_common','direct'),('original','budget_plain'))
messages=behavior.messages


def verify():
    m=load(BASE/'manifest.json')
    assert m['config']==CONFIG and m['signature']==fp({k:v for k,v in m.items() if k!='signature'})
    assert m['parent_signature']==behavior.verify()
    for n,h in m['files'].items():assert sha(ROOT/n)==h,n
    return m['signature']


def sources():
    ss=load(BASE/'public/sources.json');assert len(ss)==len({s['question_id'] for s in ss})==72
    return [parent.execution_source(s) for s in ss]


def technical_gate(rows,qids):
    expected={(q,m) for q in qids for m in MODES}
    assert len(qids)==len(set(qids))==72 and len(rows)==len(expected)==216
    assert {(r['question_id'],r['mode']) for r in rows}==expected
    groups=behavior.groups(rows)
    criteria={m:dict(schema=g['schema_ok']>=69,usable=g['usable']>=69,style=g['style_ok']>=69,
        quote_matching=g['quote_match_fraction']>=.9) for m,g in groups.items()}
    return dict(groups=groups,criteria=criteria,passed=all(all(v.values()) for v in criteria.values()),actual_calls=len(rows),
        costs={k:sum(r['result']['usage'].get(k,0) for r in rows) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},reference_values_read=False)


def preflight():
    from transformers import AutoTokenizer
    sid=verify()
    for n,h in load(BASE/'manifest.json')['model_files'].items():assert sha(ROOT/n)==h,n
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    lengths=[]
    for s in sources():
        for m in MODES:
            msg=messages(s,m);assert msg==behavior.messages(s,m)
            text=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert text.count(behavior.MODE_RULES[m])==2
            lengths.append(len(tok.encode(text,add_special_tokens=False)))
    assert len(lengths)==216 and max(lengths)<=32768
    fixed=load(BASE/'public/fixed_plans.json');assert len(fixed)==432;counter=parent.CounterTokens()
    for p in fixed:assert counter.count(p['evidence_text'])==p['evidence_tokens']<=p['budget']
    immutable(RUN/'preflight.json',dict(signature=sid,prompts=216,maximum_prompt=max(lengths),prompt_identical_to_behavior=True,reference_values_read=False))


def propose():
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid=verify();torch_setup();backend=TransformersBackend(ROOT/'models/Qwen3.5-9B',max_input_tokens=32768);rows=[]
    ss=sources()
    for s in ss:
        q=s['question_id']
        for mode in sorted(MODES,key=lambda m:fp([30,q,m])):
            key=fp([q,mode]);path=RUN/'proposals'/(key+'.json');start=RUN/'requests'/(key+'.started.json')
            assert not path.exists() and not start.exists(),'Unknown request retry forbidden'
            msg=messages(s,mode);identity=dict(signature=sid,question_id=q,source_hash=fp(s),mode=mode,max_output=2048,prompt_hash=fp(msg))
            immutable(start,dict(identity,time=time.time()))
            try:result=generate(backend,msg,2048)
            except Exception as exc:
                immutable(RUN/'failures'/(key+'.json'),dict(identity,error=repr(exc)));raise
            row=dict(identity,result=result,inspection=behavior.inspect(s,result));immutable(path,row);rows.append(row)
            print('PROPOSE',len(rows),'/216',mode,'style_ok',behavior.style_ok(row['inspection'],mode),flush=True)
    immutable(RUN/'proposal_gate.json',dict(signature=sid,**technical_gate(rows,[s['question_id'] for s in ss])))


def should_skip(name):
    if name not in ('assemble','qa4','qa9'):return False
    gate=load(RUN/'proposal_gate.json');assert gate['signature']==verify()
    return not gate['passed']


def skip_phase(name):
    assert should_skip(name)
    immutable(RUN/(name+'_skipped.json'),dict(signature=verify(),reason='behavior_gate_failed',new_model_calls=0))
    print('SKIPPED',name,'behavior_gate_failed',flush=True)


def pack(s,cached,inspection,b,counter):
    proposal=dict(status='valid' if inspection['schema_ok'] and inspection['usable'] else 'invalid',candidates=inspection['candidates'])
    # No use of note compliance, notes or parent indices in selection.
    return previous.pack(s,cached,proposal,b,counter)


def validate_plan_matrix(plans,qids):
    expected={(q,b,a) for q in qids for b in BUDGETS for a in ARMS}
    assert len(qids)==len(set(qids))==72 and len(plans)==len(expected)==864
    assert {(p['question_id'],p['budget'],p['arm']) for p in plans}==expected
    assert all(0<=p['evidence_tokens']<=p['budget'] for p in plans)


def assemble():
    if should_skip('assemble'):skip_phase('assemble');return
    sid=verify();counter=parent.CounterTokens();cached=load(BASE/'public/inherited_inputs.json');plans=deepcopy(load(BASE/'public/fixed_plans.json'))
    for s in sources():
        q=s['question_id'];assert cached[q]['source_hash']==fp(s)
        for mode in MODES:
            p=load(RUN/'proposals'/(fp([q,mode])+'.json'));assert p['signature']==sid and p['source_hash']==fp(s)
            for b in BUDGETS:plans.append(dict(pack(s,cached[q],p['inspection'],b,counter),arm=mode))
    validate_plan_matrix(plans,[s['question_id'] for s in sources()])
    immutable(RUN/'plans.json',dict(signature=sid,selection_before_qa=True,plans=plans))


def qa(reader):
    phase_name='qa4' if reader=='Qwen3.5-4B' else 'qa9'
    if should_skip(phase_name):skip_phase(phase_name);return
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid=verify();bundle=load(RUN/'plans.json');assert bundle['signature']==sid and bundle['selection_before_qa']
    assert (RUN/'assemble_complete.json').exists()
    questions={s['question_id']:s['question'] for s in sources()};validate_plan_matrix(bundle['plans'],questions)
    replay=load(BASE/'replay/reader_cache.json');torch_setup()
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


def paired(values):
    import numpy as np
    assert len(values)==72 and all(math.isfinite(v) for v in values)
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(30)
    means=x[rng.integers(0,72,(10000,72))].mean(axis=1)
    return dict(n=72,mean_delta_f1=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
        wins=sum(v>0 for v in values),ties=sum(v==0 for v in values),losses=sum(v<0 for v in values))


def decide(comparisons,per_reader,per_budget,proposal_groups):
    c={(v['arm'],v['control']):v for v in comparisons}
    criteria=dict(primary_two_points=c['plan','rationale']['mean_delta_f1']>=.02,
        each_reader_nonnegative=all(v['mean_delta_f1']>=0 for v in per_reader.values()),
        each_budget_nonnegative=all(v['mean_delta_f1']>=0 for v in per_budget.values()),
        not_below_direct=c['plan','direct']['mean_delta_f1']>=0,
        close_to_existing_role=c['plan','role_common']['mean_delta_f1']>=-.02)
    ratio=proposal_groups['plan']['total_output_tokens']/max(1,proposal_groups['rationale']['total_output_tokens'])
    decision='stop_expansion_of_this_mechanism_claim' if not all(criteria.values()) else 'needs_effort_matched_followup_before_mechanism_claim' if ratio>1.25 else 'eligible_for_mechanism_confirmation'
    return dict(criteria=criteria,decision=decision,plan_over_rationale_output_ratio=ratio,
        equal_realized_compute_established=False,mechanism_confirmed=False,publication_ready=False,automatic_expansion=False)


def score():
    sid=verify();gate=load(RUN/'proposal_gate.json');assert gate['signature']==sid
    if not gate['passed']:
        assert not (RUN/'plans.json').exists() and not any((RUN/'responses').rglob('*.json'))
        for name in ('assemble','qa4','qa9'):
            marker=load(RUN/(name+'_skipped.json'));assert marker['signature']==sid and marker['new_model_calls']==0
        immutable(RUN/'summary.json',dict(signature=sid,complete=True,conditions=0,sources=72,proposal_gate=gate,
            decision='stop_before_qa_for_behavior_review',effectiveness_tested=False,new_reader_calls=0,
            reference_values_read=False,publication_ready=False,mechanism_confirmed=False,automatic_expansion=False));return
    bundle=load(RUN/'plans.json');assert bundle['signature']==sid and bundle['selection_before_qa']
    plans=bundle['plans'];qids=[s['question_id'] for s in sources()];validate_plan_matrix(plans,qids);responses={}
    for r in READERS:
        for p in plans:
            q,b,a=p['question_id'],p['budget'],p['arm'];v=load(RUN/'responses'/r/(fp([r,q,b,a])+'.json'))
            assert v['signature']==sid and v['plan_hash']==fp(p) and v['status'] in ('ok','empty','selection_failed')
            assert (v['reader'],v['question_id'],v['budget'],v['arm'])==(r,q,b,a);responses[r,q,b,a]=v
    assert len(responses)==1728
    refs=load(BASE/'private/references.json');assert set(refs)==set(qids);metric=parent.official_metrics();records=[]
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
    result=dict(signature=sid,complete=True,conditions=len(records),sources=72,exposed_development=True,effectiveness_tested=True,
        comparisons=comparisons,groups=groups,per_reader=per_reader,per_budget=per_budget,records=records,proposal_gate=gate,
        new_reader_calls=len(fresh),new_reader_costs={k:sum(v['usage'].get(k,0) for v in fresh) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
        inherited_costs=load(BASE/'replay/costs.json'),**decide(comparisons,per_reader,per_budget,gate['groups']))
    immutable(RUN/'summary.json',result);print('SCORE_COMPLETE',result['decision'],comparisons[:2],flush=True)


def phase(name):
    if name=='propose9':propose()
    elif name in ('qa4','qa9'):qa('Qwen3.5-'+('4B' if name=='qa4' else '9B'))
    else:globals()[name]()
