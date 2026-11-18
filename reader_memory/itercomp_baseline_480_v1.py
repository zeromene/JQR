"""Late-added closest-baseline development comparison; frozen before new outputs."""
from collections import Counter
from copy import deepcopy
import importlib.util
import math
import statistics
import time
from . import selector_distill_v1 as parent
from .itercomp_engine_v1 import units, render, judge_messages

ROOT=parent.ROOT; BASE=ROOT/'research/itercomp_baseline_480_v1'; PLAN=BASE; RUN=ROOT/'runs/itercomp_baseline_480_v1'
load,sha,fp,immutable,write_json=parent.load,parent.sha,parent.fp,parent.immutable,parent.write_json
READERS=parent.READERS; BUDGETS=(256,512)
CAPPED=('original','budget_plain','role_common','exit_title','itercomp_cap')
NATIVE=('itercomp_native','itercomp_first')
CONFIG=dict(version='itercomp-baseline-480-v1',sources=480,seed=30,bootstrap_replicates=10000,
    max_steps=5,max_judge_calls=2400,max_input=32768,max_output=512,encoder_batch=8,
    reader_max_output=128,reader_max_input=4096,max_reader_conditions=11520,
    capped=list(CAPPED),native=list(NATIVE),encoder_device='cpu',training=False,
    resource_caps_mib=dict(pipeline=20500,qa4=10500,qa9=19500),
    resource_headroom_mib=dict(pipeline=500,qa4=1000,qa9=1000),automatic_expansion=False)


def verify():
    m=load(BASE/'manifest.json')
    assert m['config']==CONFIG and m['signature']==fp({k:v for k,v in m.items() if k!='signature'})
    assert m['parent_signature']==parent.verify()
    assert m['source_parent_signature']==parent.teacher.parent.verify()
    for n,h in m['files'].items():assert sha(ROOT/n)==h,n
    return m['signature']


def sources():
    ss=load(BASE/'public/sources.json');assert len(ss)==len({s['question_id'] for s in ss})==CONFIG['sources']
    return [parent.teacher.parent.execution_source(s) for s in ss]


def native_runtime():
    spec=importlib.util.spec_from_file_location('reader_memory._itercomp_comparison_native',ROOT/'reader_memory/itercomp_technical_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.BASE,m.RUN,m.CONFIG,m.verify,m.sources=BASE,RUN,CONFIG,verify,sources
    return m


def preflight():
    from transformers import AutoTokenizer
    sid=verify()
    for n,h in load(BASE/'manifest.json')['model_files'].items():assert sha(ROOT/n)==h,n
    compat=load(BASE/'replay/encoder_compatibility.json');assert compat['passed']
    native_runtime().preflight()
    counter=parent.teacher.parent.CounterTokens();fixed=load(BASE/'public/fixed_plans.json')
    assert len(fixed)==3840 and len({(p['question_id'],p['budget'],p['arm']) for p in fixed})==3840
    maxima={}
    for reader in READERS:
        tok=AutoTokenizer.from_pretrained(str(ROOT/'models'/reader),local_files_only=True,trust_remote_code=False)
        lens=[]
        for s in sources():
            # Every native material is a subset; validate complete original candidate rendering too.
            cs=units(s);text=render(cs,list(range(len(cs))))
            msg=parent.teacher.parent.role.old.common.messages(s['question'],text)
            lens.append(len(tok.encode(tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)))
        maxima[reader]=max(lens)
        assert max(lens)<=CONFIG['reader_max_input'],'Native input must not silently truncate'
    for p in fixed:assert counter.count(p['evidence_text'])==p['evidence_tokens']<=p['budget']
    immutable(RUN/'reader_preflight.json',dict(signature=sid,full_candidate_maximum_prompt=maxima,private_references_read=False))


def cap_native(s,row,budget,counter):
    """Controlled adaptation, not native IterCOMP: best score over executed queries."""
    cs=units(s);eligible=row['selected']
    scores=[max(step['scores'][i] for step in row['steps']) for i in range(len(cs))]
    p=parent.pack(cs,scores,budget,counter,eligible=eligible)
    p['trace'].update(adaptation='native selected pool; max score across executed queries; whole sentences; original-order rendering',
        native_status=row['status'],native_tokens=row['evidence_tokens'],native_indices=eligible,
        answerability_certified_for_capped_text=False)
    return p


def assemble():
    sid=verify();counter=parent.teacher.parent.CounterTokens();plans=deepcopy(load(BASE/'public/fixed_plans.json'))
    for s in sources():
        q=s['question_id'];r=load(RUN/'results'/(q+'.json'))
        assert r['signature']==sid and r['source_hash']==fp(s)
        for b in BUDGETS:
            p=cap_native(s,r,b,counter)
            plans.append(dict(p,question_id=q,budget=b,arm='itercomp_cap',private_inputs_read=False))
        for a,selected in [('itercomp_native',r['selected']),('itercomp_first',r['steps'][0]['selected'])]:
            text=render(units(s),selected)
            plans.append(dict(question_id=q,budget=0,arm=a,evidence_text=text,evidence_tokens=counter.count(text),
                selected_indices=selected,selected_ids=[units(s)[i]['uid'] for i in selected],selected_ranges={},
                trace=dict(fallback=None,native_status=r['status'],uncapped=True,answerability_is_self_report=True),
                private_inputs_read=False,status='ready' if text else 'selection_failed',reason=None))
    validate_plan_matrix(plans,[s['question_id'] for s in sources()])
    immutable(RUN/'plans.json',dict(signature=sid,selection_before_qa=True,plans=plans))


def validate_plan_matrix(plans,qids):
    expected={(q,b,a) for q in qids for b in BUDGETS for a in CAPPED}|{(q,0,a) for q in qids for a in NATIVE}
    assert len(plans)==len(expected)==5760
    assert {(p['question_id'],p['budget'],p['arm']) for p in plans}==expected
    assert all(p['evidence_tokens']<=p['budget'] for p in plans if p['arm'] in CAPPED)


def qa(reader):
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid=verify();bundle=load(RUN/'plans.json');assert bundle['signature']==sid and bundle['selection_before_qa']
    assert (RUN/'assemble_complete.json').exists()
    questions={s['question_id']:s['question'] for s in sources()};validate_plan_matrix(bundle['plans'],questions)
    replay=load(BASE/'replay/reader_cache.json');torch_setup()
    backend=TransformersBackend(ROOT/'models'/reader,max_input_tokens=CONFIG['reader_max_input'])
    for i,p in enumerate(sorted(bundle['plans'],key=lambda p:fp([30,p['question_id'],p['budget'],p['arm']])),1):
        q,b,a=p['question_id'],p['budget'],p['arm'];out=RUN/'responses'/reader/(fp([reader,q,b,a])+'.json')
        assert not out.exists()
        base=dict(signature=sid,reader=reader,question_id=q,budget=b,arm=a,plan_hash=fp(p))
        if p['status']!='ready':immutable(out,dict(base,status='selection_failed',text='',usage={}));continue
        assert len(backend.tokenizer.encode(p['evidence_text'],add_special_tokens=False))==p['evidence_tokens']
        msg=parent.teacher.parent.role.old.common.messages(questions[q],p['evidence_text']);content=fp([reader,msg,128])
        prompt=backend.tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert len(backend.tokenizer.encode(prompt,add_special_tokens=False))<=CONFIG['reader_max_input']
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
            immutable(start,dict(signature=sid,content_key=content,time=time.time()))
            result=generate(backend,msg,128)
            immutable(cache,dict(signature=sid,content_key=content,result=result))
        immutable(out,dict(base,status='ok' if result['text'].strip() else 'empty',request_key=key,reused=reused,reuse_origin=origin,**result))
        print('QA',reader,i,'/5760',flush=True)


def paired(values):
    import numpy as np
    assert len(values)==CONFIG['sources'] and all(math.isfinite(v) for v in values)
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(30)
    bootstrap=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    return dict(n=len(x),mean_delta_f1=float(x.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
        wins=sum(v>0 for v in values),ties=sum(v==0 for v in values),losses=sum(v<0 for v in values))


def decision(primary,per_reader,per_budget,native_delta,length_ratio,technical_failures):
    criteria=dict(primary_at_least_two_points=primary['mean_delta_f1']>=.02,
        primary_positive_interval=primary['ci95'][0]>0,native_positive_interval=native_delta['ci95'][0]>0,
        each_reader_nonnegative=all(v['mean_delta_f1']>=0 for v in per_reader.values()),
        each_budget_nonnegative=all(v['mean_delta_f1']>=0 for v in per_budget.values()),
        native_does_not_dominate=not(native_delta['mean_delta_f1']<=-.02 and length_ratio>=1),
        no_technical_judge_failures=technical_failures==0)
    if technical_failures:result='stop_for_adapter_review'
    elif all(criteria.values()):result='closest_baseline_advantage_supported'
    else:result='stop_expansion_for_competitiveness_review'
    return dict(criteria=criteria,decision=result,publication_ready=False,automatic_expansion=False)


def score():
    sid=verify();bundle=load(RUN/'plans.json');assert bundle['signature']==sid and bundle['selection_before_qa']
    plans=bundle['plans'];ss=sources();qids=[s['question_id'] for s in ss];validate_plan_matrix(plans,qids)
    responses={}
    for r in READERS:
        for p in plans:
            q,b,a=p['question_id'],p['budget'],p['arm'];row=load(RUN/'responses'/r/(fp([r,q,b,a])+'.json'))
            assert row['signature']==sid and row['plan_hash']==fp(p) and row['status'] in ('ok','empty','selection_failed')
            responses[r,q,b,a]=row
    assert len(responses)==11520
    # Private references are opened only after the complete response matrix is validated.
    refs=load(BASE/'private/references.json');assert set(refs)==set(qids)
    metric=parent.teacher.parent.official_metrics();records=[]
    for r in READERS:
        for p in plans:
            q,b,a=p['question_id'],p['budget'],p['arm'];v=responses[r,q,b,a];gold=refs[q]['answer']
            records.append(dict(reader=r,question_id=q,budget=b,arm=a,f1=float(metric['f1_score'](v['text'],gold)[0]),
                em=float(metric['exact_match_score'](v['text'],gold)),evidence_tokens=p['evidence_tokens'],
                status=v['status'],reused=v.get('reused',False),usage=v['usage'],finish_reason=v.get('finish_reason')))
    ix={(v['reader'],v['question_id'],v['budget'],v['arm']):v for v in records}
    def contrast(a,c,rr=READERS,bb=BUDGETS):
        return paired([statistics.mean(ix[r,q,0 if a in NATIVE else b,a]['f1']-ix[r,q,0 if c in NATIVE else b,c]['f1'] for r in rr for b in bb) for q in qids])
    comparisons=[dict(arm=a,control=c,**contrast(a,c)) for a,c in [('original','itercomp_cap'),('original','itercomp_native'),
        ('original','budget_plain'),('role_common','budget_plain'),('original','exit_title'),('itercomp_native','itercomp_first'),('itercomp_cap','itercomp_native')]]
    per_reader={r:contrast('original','itercomp_cap',(r,)) for r in READERS}
    per_budget={str(b):contrast('original','itercomp_cap',bb=(b,)) for b in BUDGETS}
    groups=[]
    for r in ('pooled',*READERS):
        for a in (*CAPPED,*NATIVE):
            for b in (('pooled',*BUDGETS) if a in CAPPED else (0,)):
                rs=[v for v in records if v['arm']==a and (r=='pooled' or v['reader']==r) and (b=='pooled' or v['budget']==b)]
                groups.append(dict(reader=r,budget=b,arm=a,f1_points=100*statistics.mean(v['f1'] for v in rs),
                    em_points=100*statistics.mean(v['em'] for v in rs),mean_tokens=statistics.mean(v['evidence_tokens'] for v in rs)))
    native=[load(RUN/'results'/(q+'.json')) for q in qids];counts=Counter(v['status'] for v in native)
    failures=sum(v for k,v in counts.items() if k not in ('answerable','iteration_limit'))
    original_length=statistics.mean(p['evidence_tokens'] for p in plans if p['arm']=='original')
    native_length=statistics.mean(p['evidence_tokens'] for p in plans if p['arm']=='itercomp_native')
    fresh=[v for v in records if not v['reused'] and v['status'] in ('ok','empty')]
    result=dict(signature=sid,complete=True,conditions=len(records),source_components=480,late_added_comparison=True,previously_exposed_sources=True,
        comparisons=comparisons,groups=groups,per_reader=per_reader,per_budget=per_budget,records=records,
        native_status_counts=dict(counts),actual_judge_calls=sum(len(v['steps']) for v in native),
        judge_costs={k:sum(step['result']['usage'].get(k,0) for v in native for step in v['steps']) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
        pipeline_seconds=sum(v['seconds'] for v in native),encoder_device='cpu',latency_speedup_established=False,
        inherited_costs=load(BASE/'replay/costs.json'),new_reader_calls=len(fresh),
        new_reader_costs={k:sum(v['usage'].get(k,0) for v in fresh) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
        original_over_native_length=original_length/max(1,native_length),
        native_over_budget={str(b):sum(v['evidence_tokens']>b for v in native) for b in BUDGETS},
        capped_changed_from_native={str(b):sum(p['evidence_text']!=next(v['evidence_text'] for v in native if v['question_id']==p['question_id']) for p in plans if p['arm']=='itercomp_cap' and p['budget']==b) for b in BUDGETS},
        **decision(comparisons[0],per_reader,per_budget,comparisons[1],original_length/max(1,native_length),failures))
    assert result['actual_judge_calls']<=2400
    immutable(RUN/'summary.json',result);print('SCORE_COMPLETE',result['decision'],comparisons,flush=True)


def phase(name):
    if name=='pipeline':native_runtime().pipeline()
    elif name in ('qa4','qa9'):qa('Qwen3.5-'+('4B' if name=='qa4' else '9B'))
    else:globals()[name]()
