"""Conditional Phi transfer test on fixed public evidence, without reselection."""
from collections import Counter
import statistics
import time
from . import short_budget_controls_v1 as parent

ROOT=parent.ROOT
BASE=ROOT/'research/cross_family_compact_v1'
PLAN=parent.PLAN
RUN=ROOT/'runs/cross_family_compact_v1'
load,sha,fp,immutable,write_json=parent.load,parent.sha,parent.fp,parent.immutable,parent.write_json
READER='Phi-4-mini-instruct'
BUDGETS=parent.BUDGETS
ARMS=('ap3','quote_anchor','short_global','short_ap3')
CONFIG=dict(version='cross-family-compact-v1',sources=96,reader=READER,budgets=list(BUDGETS),arms=list(ARMS),
    conditions=768,reader_output=128,max_input=4096,
    activation='Both parent efficacy_compactness_passed and short control specific_selection_criterion_met',
    evidence='Identical strings selected by Qwen pipeline; 256/512 are Qwen evidence token caps, Phi counts separately reported',
    criterion='Anchor minus AP3 source mean >=.02 and CI lower >0; each budget Phi evidence mean ratio <=.8',
    resource_caps_mib=dict(qa_phi=9500),resource_headroom_mib=dict(qa_phi=1000))


def verify():
    m=load(BASE/'manifest.json')
    assert m['config']==CONFIG and m['signature']==fp({k:v for k,v in m.items() if k!='signature'})
    assert parent.verify()==m['parent_signature']
    for n,d in m['files'].items():
        if sha(ROOT/n)!=d:raise ValueError('Frozen file changed: '+n)
    return m['signature']


def preflight():
    from transformers import AutoTokenizer
    signature=verify()
    m=load(ROOT/'models'/READER/'runtime_manifest.json')
    for item in m['files']:
        p=ROOT/'models'/READER/item['path']
        assert p.stat().st_size==item['size'] and sha(p)==item['sha256'],item['path']
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models'/READER),local_files_only=True,trust_remote_code=False)
    assert tok is not None
    immutable(RUN/'preflight.json',dict(signature=signature,actual_model_bytes_verified=True))


def assemble():
    from transformers import AutoTokenizer
    signature=verify()
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models'/READER),local_files_only=True,trust_remote_code=False)
    bundles=[load(parent.parent.RUN/'plans.json'),load(parent.RUN/'plans.json')]
    assert bundles[0]['signature']==parent.parent.verify() and bundles[1]['signature']==parent.verify()
    sources={s['question_id']:s for s in load(PLAN/'public/sources.json')}
    plans=[]
    for bundle in bundles:
        assert bundle['selection_before_qa']
        for p in bundle['plans']:
            if p['arm'] not in ARMS:continue
            messages=parent.parent.parent.base.old.common.messages(sources[p['question_id']]['question'],p['evidence_text'])
            prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompt_tokens=len(tok.encode(prompt,add_special_tokens=False))
            assert prompt_tokens<=CONFIG['max_input']
            plans.append(dict(p,origin_signature=bundle['signature'],origin_plan_hash=fp(p),
                phi_evidence_tokens=len(tok.encode(p['evidence_text'],add_special_tokens=False)),phi_prompt_tokens=prompt_tokens))
    assert len(plans)==768 and len({(p['question_id'],p['budget'],p['arm']) for p in plans})==768
    immutable(RUN/'plans.json',dict(signature=signature,selection_before_qa=True,source_bundle_hashes=list(map(fp,bundles)),plans=plans))
    print('ASSEMBLE_COMPLETE',len(plans),flush=True)


def qa():
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    signature=verify()
    bundle=load(RUN/'plans.json')
    assert bundle['signature']==signature
    sources={s['question_id']:s for s in load(PLAN/'public/sources.json')}
    torch_setup()
    backend=TransformersBackend(ROOT/'models'/READER,max_input_tokens=CONFIG['max_input'])
    for i,p in enumerate(sorted(bundle['plans'],key=lambda p:fp([30,p['question_id'],p['budget'],p['arm']])),1):
        output=RUN/'responses'/(fp([READER,p['question_id'],p['budget'],p['arm']])+'.json')
        if output.exists():raise ValueError('Existing answer; no replay')
        base=dict(signature=signature,reader=READER,question_id=p['question_id'],budget=p['budget'],arm=p['arm'],plan_hash=fp(p))
        if p['status']!='ready':
            immutable(output,dict(base,status='selection_failed',text='',usage={}))
            continue
        assert len(backend.tokenizer.encode(p['evidence_text'],add_special_tokens=False))==p['phi_evidence_tokens']
        messages=parent.parent.parent.base.old.common.messages(sources[p['question_id']]['question'],p['evidence_text'])
        prompt=backend.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert len(backend.tokenizer.encode(prompt,add_special_tokens=False))==p['phi_prompt_tokens']<=CONFIG['max_input']
        key=fp([signature,READER,messages,CONFIG['reader_output']])
        cache=RUN/'request_cache'/(key+'.json')
        started=cache.with_suffix('.started.json')
        reused=cache.exists()
        if reused:
            saved=load(cache)
            assert saved['signature']==signature and saved['request_key']==key
            result=saved['result']
        else:
            if started.exists():raise ValueError('Unknown inference request; no replay')
            immutable(started,dict(signature=signature,request_key=key,time=time.time()))
            try:result=generate(backend,messages,CONFIG['reader_output'])
            except Exception as exc:
                immutable(output,dict(base,status='generation_failed',text='',usage={},error=repr(exc)))
                raise
            immutable(cache,dict(signature=signature,request_key=key,result=result))
        immutable(output,dict(base,status='ok' if result['text'].strip() else 'empty',request_key=key,reused=reused,**result))
        print('QA',i,'/ 768',flush=True)


def score():
    from .full_comparison_v1 import answer_scores
    from .discriminative_pilot_v1 import paired
    signature=verify()
    plans=load(RUN/'plans.json')['plans']
    answers={(p['question_id'],p['budget'],p['arm']):load(RUN/'responses'/
        (fp([READER,p['question_id'],p['budget'],p['arm']])+'.json')) for p in plans}
    assert len(plans)==len(answers)==768
    refs=load(PLAN/'private/references.json')
    records=[]
    for p in plans:
        q,b,a=p['question_id'],p['budget'],p['arm']
        r=answers[q,b,a]
        assert r['signature']==signature and r['plan_hash']==fp(p)
        em,f1=answer_scores(r['text'],refs[q])
        records.append(dict(question_id=q,budget=b,arm=a,em=em,f1=f1,status=r['status'],usage=r['usage'],
            reused=r.get('reused',False),qwen_evidence_tokens=p['evidence_tokens'],phi_evidence_tokens=p['phi_evidence_tokens']))
    idx={(r['question_id'],r['budget'],r['arm']):r for r in records}
    comparisons=[]
    for budget in (*BUDGETS,'pooled'):
        bb=BUDGETS if budget=='pooled' else (budget,)
        for control in ('ap3','short_global','short_ap3'):
            ds=[statistics.mean(idx[q,b,'quote_anchor']['f1']-idx[q,b,control]['f1'] for b in bb) for q in refs]
            comparisons.append(dict(budget=budget,control=control,**paired(ds)))
    primary=next(x for x in comparisons if x['budget']=='pooled' and x['control']=='ap3')
    ratios={str(b):statistics.mean(idx[q,b,'quote_anchor']['phi_evidence_tokens'] for q in refs)/
        statistics.mean(idx[q,b,'ap3']['phi_evidence_tokens'] for q in refs) for b in BUDGETS}
    passed=primary['mean_delta_f1']>=.02 and primary['ci95'][0]>0 and all(v<=.8 for v in ratios.values())
    groups=[]
    for budget in (*BUDGETS,'pooled'):
        for arm in ARMS:
            rows=[r for r in records if r['arm']==arm and (budget=='pooled' or r['budget']==budget)]
            groups.append(dict(budget=budget,arm=arm,f1_points=100*statistics.mean(r['f1'] for r in rows),
                qwen_tokens=statistics.mean(r['qwen_evidence_tokens'] for r in rows),phi_tokens=statistics.mean(r['phi_evidence_tokens'] for r in rows)))
    fresh=[r for r in records if not r['reused'] and r['status'] in ('ok','empty')]
    immutable(RUN/'summary.json',dict(signature=signature,complete=True,conditions=768,records=records,
        primary=primary,comparisons=comparisons,phi_token_ratios=ratios,criterion_met=passed,groups=groups,
        new_reader_calls=len(fresh),reader_costs={k:sum(r['usage'].get(k,0) for r in fresh) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
        status_counts=dict(Counter(r['status'] for r in records)),
        caveats='One prespecified other model family on the same 96 two-hop sources. Identical evidence strings, budgets measured with Qwen tokenizer; actual Phi lengths separate. No reselection or prompt tuning.'))
    print('SCORE_COMPLETE',primary,ratios,passed,flush=True)


def phase(name):
    if name=='qa_phi':qa()
    elif name in ('preflight','assemble','score'):globals()[name]()
    else:raise ValueError(name)
