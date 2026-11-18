"""Eight fixed training sources, native uncapped IterCOMP technical validation only."""
import time
from collections import Counter
from . import selector_teacher_pilot_v1 as parent
from .itercomp_engine_v1 import BGEEncoder, compress, units, render, judge_messages
ROOT=parent.ROOT; BASE=ROOT/'research/itercomp_technical_v1'; RUN=ROOT/'runs/itercomp_technical_v1'
load,sha,fp,immutable,write_json=parent.load,parent.sha,parent.fp,parent.immutable,parent.write_json
CONFIG=dict(version='itercomp-technical-v1',sources=8,max_steps=5,max_judge_calls=40,judge='Qwen3.5-9B',
            max_input=32768,max_output=512,encoder_device='cpu',encoder_dtype='float32',encoder_batch=8,
            resource_caps_mib=dict(pipeline=20500),resource_headroom_mib=dict(pipeline=500),
            private_labels=False,answer_generation=False,training=False,performance_comparison=False)


def verify():
    m=load(BASE/'manifest.json');assert m['config']==CONFIG and m['signature']==fp({k:v for k,v in m.items() if k!='signature'})
    assert m['parent_signature']==parent.verify()
    for n,h in m['files'].items():assert sha(ROOT/n)==h,n
    return m['signature']


def sources():
    ss=load(BASE/'public/sources.json');assert len(ss)==8
    return [parent.parent.execution_source(s) for s in ss]


def prerequisite():
    # Different jobs keep their existing phases. This technical job starts after the teacher pilot.
    deadline=time.monotonic()+7200
    while True:
        p=parent.RUN/'queue_status.json'
        state=load(p) if p.exists() else {}
        if state.get('status')=='completed':break
        assert state.get('status')!='failed','Teacher pilot failed; inspect first'
        assert time.monotonic()<deadline,'Teacher prerequisite wait limit'
        print('WAITING_FOR_TEACHER_COMPLETION',state.get('phase'),flush=True);time.sleep(30)
    compat=load(ROOT/'research/itercomp_adapter_v1/encoder_compatibility.json')
    assert compat['passed'] and compat['dataset_calls']==0 and compat['dense_exact'] and compat['sparse_exact']
    immutable(RUN/'prerequisite.json',dict(signature=verify(),teacher_completed=True,encoder_compatibility_sha256=sha(ROOT/'research/itercomp_adapter_v1/encoder_compatibility.json')))


def preflight():
    from transformers import AutoTokenizer
    verify();m=load(BASE/'manifest.json')
    for n,h in m['model_files'].items():assert sha(ROOT/n)==h,n
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    bge=AutoTokenizer.from_pretrained(str(ROOT/'models/bge-m3'),local_files_only=True,trust_remote_code=False)
    lengths=[];max_unit=0
    for s in sources():
        cs=units(s);msg=judge_messages(s['question'],render(cs,list(range(len(cs)))))
        lengths.append(len(tokenizer.encode(tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)))
        max_unit=max(max_unit,max(len(bge.encode(t,add_special_tokens=True)) for t in [s['question']]+[c['text'] for c in cs]))
    assert max(lengths)<=CONFIG['max_input'] and max_unit<=8192
    immutable(RUN/'preflight.json',dict(signature=verify(),maximum_judge_prompt=max(lengths),maximum_encoder_input=max_unit,private_labels_read=False))


def pipeline():
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid=verify();torch_setup()
    encoder=BGEEncoder(ROOT/'models/bge-m3',batch_size=CONFIG['encoder_batch'])
    backend=TransformersBackend(ROOT/'models/Qwen3.5-9B',max_input_tokens=CONFIG['max_input'])
    for s in sources():
        q=s['question_id'];attempt=[0]
        def judge(messages):
            step=attempt[0];attempt[0]+=1
            assert step<CONFIG['max_steps']
            prompt=backend.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert len(backend.tokenizer.encode(prompt,add_special_tokens=False))<=CONFIG['max_input']
            path=RUN/'requests'/q/(str(step)+'.json');start=path.with_suffix('.started.json')
            assert not start.exists() and not path.exists(),'No unknown-call replay'
            identity=dict(signature=sid,source_hash=fp(s),prompt_hash=fp(messages),max_tokens=CONFIG['max_output'])
            immutable(start,dict(identity,time=time.time()))
            result=generate(backend,messages,CONFIG['max_output'])
            immutable(path,dict(identity,result=result));return result
        def record(row):immutable(RUN/'steps'/q/(str(row['step'])+'.json'),dict(signature=sid,source_hash=fp(s),**row))
        start=time.perf_counter();r=compress(s,encoder,judge,record,CONFIG['max_steps'])
        r.update(signature=sid,question_id=q,source_hash=fp(s),seconds=time.perf_counter()-start,
                 evidence_tokens=len(backend.tokenizer.encode(r['evidence_text'],add_special_tokens=False)),
                 source_sentences=len(units(s)),selected_sentences=len(r['selected']))
        immutable(RUN/'results'/(q+'.json'),r)
        print('ITERCOMP_SOURCE',q,r['status'],'steps',len(r['steps']),'tokens',r['evidence_tokens'],flush=True)


def summarize():
    sid=verify();rows=[load(RUN/'results'/(s['question_id']+'.json')) for s in sources()]
    assert all(r['signature']==sid for r in rows)
    counts=Counter(r['status'] for r in rows);calls=sum(len(r['steps']) for r in rows)
    assert calls<=40
    result=dict(signature=sid,complete=True,status_counts=dict(counts),judge_calls=calls,
        engineering_passed=all(r['status'] in ('answerable','iteration_limit') for r in rows),
        reader_accuracy_evaluated=False,semantic_sufficiency_established=False,training=False,
        latency_benchmark=False,encoder_device='cpu',
        mean_selected_tokens=sum(r['evidence_tokens'] for r in rows)/8,
        judge_usage={k:sum(step['result']['usage'].get(k,0) for r in rows for step in r['steps']) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')})
    immutable(RUN/'summary.json',result);print(result,flush=True)


def phase(name):globals()[name]()
