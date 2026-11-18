"""Bounded two-dataset competition with exact historical material/request reuse."""
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import argparse,json,os,time,subprocess,traceback
from . import length_repack_qa_v1 as u
from . import hotpot_confirmation_v1 as hp
from . import matched_quote_fourarm_v2 as direct
from . import selective_evidence_repair_v1 as repair
from . import itercomp_engine_v1 as it
from . import itercomp_baseline_480_v1 as itcap

ROOT=u.ROOT;BASE=ROOT/'research/paper_competition_v1';RUN=ROOT/'runs/paper_competition_v1'
ARMS=['quote_anchor','rank_add','strong_rerank','exit_budgeted','compact_budgeted','itercomp_cap','longrefiner_repair']
CONFIG=dict(sources=96,per_dataset=48,budgets=[256,512],arms=ARMS,reader='Qwen3.5-9B',seed=30,
    hotpot_output=128,musique_output=2048,reader_max_input=8192,proposer_max_input=16384,
    proposer_output=2048,itercomp_output=512,itercomp_steps=5,gpu_cap_mib=10500,admission_mib=11700,
    memory_wait_seconds=7200,wall_limit_seconds=86400,training=False,automatic_followup=False,logical_conditions=1536)

def sources():return u.load(BASE/'public/sources.json')
def inputs():return u.load(BASE/'inherited/selected_assets.json')
def state(status,**fields):
    v=dict(status=status,time=datetime.now(timezone.utc).isoformat(),**fields)
    p=RUN/'status.tmp';p.write_text(json.dumps(v,indent=2),encoding='utf-8');os.replace(p,RUN/'status.json')
    print(json.dumps(v),flush=True)

def normalize(dataset,method,p,origin):
    budget=p.get('budget',p.get('evidence_budget',0))
    failed=p.get('status')=='selection_failed' or bool(p.get('selection_error'))
    return dict(dataset=dataset,question_id=p['question_id'],method=method,budget=budget,
        evidence_text=p['evidence_text'],evidence_tokens=p['evidence_tokens'],
        status='selection_failed' if failed else 'ready',reason=p.get('reason',p.get('selection_error')),
        origin=origin,parent_plan_hash=u.fp(p),trace=p.get('trace',{}))

def qa_request(p,s):
    if p['dataset']=='hotpot':
        messages=hp.role.old.common.messages(s['question'],p['evidence_text']);limit=128
    else:
        messages=hp.short.original.recomp.messages_for(dict(question=s['question'],evidence_text=p['evidence_text']));limit=2048
    key=u.fp([CONFIG['reader'],messages,limit])
    return dict(key=key,reader=CONFIG['reader'],messages=messages,max_output=limit,evidence_tokens=p['evidence_tokens'])

def proposal_messages(s,kind):
    payload=dict(question=s['question'],sources=[dict(id=p['source_id'],title=p['title'],text=p['text']) for p in s['paragraphs']])
    text=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    if kind=='role':return [dict(role='system',content=hp.role_prompt.PROMPT),dict(role='user',content=text)]
    assert kind=='direct'
    return [dict(role='system',content=direct.COMMON_PROMPT+'\n'+direct.MODE_RULES[kind]+'\n'+direct.format_example(kind)),
            dict(role='user',content=text+'\nEND OF SOURCE DATA. Output reminder: '+direct.MODE_RULES[kind])]

def verify(models=False):
    m=u.load(BASE/'manifest.json');assert m['config']==CONFIG
    for name,digest in m['files'].items():assert u.sha(ROOT/name)==digest,name
    if models:
        for name,digest in m['models'].items():assert u.sha(ROOT/name)==digest,name
    return u.sha(BASE/'manifest.json')

def setup_gpu():
    import torch
    torch.set_num_threads(8);torch.manual_seed(30);torch.cuda.manual_seed_all(30)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_cap_mib']*2**20/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    return torch

def qwen():
    from .length_repack_offload_v1 import OffloadBackend
    return OffloadBackend()

def call(backend,messages,limit,identity,phase):
    from .mainline_service_v1 import generate
    key=u.fp([phase,identity]);path=RUN/'calls'/phase/(key+'.json');start=path.with_suffix('.started.json')
    assert not path.exists() and not start.exists(),'No unknown request retries'
    prompt=backend.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    n=len(backend.tokenizer.encode(prompt,add_special_tokens=False))
    assert n<= (CONFIG['reader_max_input'] if phase=='qa' else CONFIG['proposer_max_input'])
    u.immutable(start,dict(signature=u.sha(BASE/'manifest.json'),identity=identity,messages=messages,max_output=limit,time=time.time()))
    result=generate(backend,messages,limit)
    u.immutable(path,dict(signature=u.sha(BASE/'manifest.json'),identity=identity,messages=messages,max_output=limit,result=result))
    return result

def propose():
    backend=qwen();ss=[x for x in sources() if x['dataset']=='musique']
    for i,x in enumerate(ss,1):
        s=x['source'];q=s['question_id']
        for kind in ['role','direct']:
            msg=proposal_messages(s,kind);attempts=[]
            for attempt in range(2 if kind=='role' else 1):
                result=call(backend,msg,2048,[q,kind,attempt],'propose')
                parsed=hp.parse_result(s,'role',result) if kind=='role' else direct.inspect(s,result)
                attempts.append(dict(result=result,messages=msg,parsed=parsed))
                if kind!='role' or parsed['status']!='invalid':break
                correction='Repair your output once. Return the complete valid JSON object only. Validation errors: '+json.dumps(parsed['errors'])+'. Use only exact quotes from the original sources; do not invent facts. If no supported evidence is identifiable, return {"roles":[]}.'
                msg=proposal_messages(s,kind)+[dict(role='assistant',content=result['text']),dict(role='user',content=correction)]
            u.immutable(RUN/'proposals'/(u.fp([q,kind])+'.json'),dict(question_id=q,source_hash=u.fp(s),kind=kind,parsed=parsed,attempts=attempts))
        state('running',phase='propose',completed=i,expected=48)

def itercomp():
    backend=qwen();encoder=it.BGEEncoder(ROOT/'models/bge-m3',device='cpu',batch_size=8)
    counter=hp.CounterTokens()
    for i,x in enumerate([r for r in sources() if r['dataset']=='musique'],1):
        s=x['source'];q=s['question_id'];index=[0]
        def judge(msg):
            j=index[0];index[0]+=1
            return call(backend,msg,512,[q,j],'itercomp')
        def record(step):u.immutable(RUN/'itercomp_steps'/q/(str(step['step'])+'.json'),step)
        row=it.compress(s,encoder,judge,record,5);row.update(question_id=q,source_hash=u.fp(s),evidence_tokens=counter.count(row['evidence_text']))
        u.immutable(RUN/'itercomp'/(q+'.json'),row)
        state('running',phase='itercomp',completed=i,expected=48)

def trace_generator(engine,label):
    original=engine.generate_one
    def traced(prompt,max_tokens,adapter=None,logprobs=None):
        key=u.fp([label,prompt,max_tokens,adapter,logprobs]);path=RUN/'compressor_calls'/label/(key+'.json')
        if path.exists():return u.load(path)['result']
        started=path.with_suffix('.started.json');assert not started.exists(),'Unknown compressor request'
        u.immutable(started,dict(prompt=prompt,max_tokens=max_tokens,adapter=adapter,logprobs=logprobs,time=time.time()))
        begin=time.perf_counter();value=original(prompt,max_tokens,adapter,logprobs)
        u.immutable(path,dict(result=value,seconds=time.perf_counter()-begin,prompt=prompt,max_tokens=max_tokens,adapter=adapter,logprobs=logprobs))
        return value
    engine.generate_one=traced

def compressors(phase):
    from experiments.recent_baselines_v1 import engines,core
    from experiments import longrefiner_repair_v1 as fixed
    if phase=='compact':
        from .paper_competition_resources_v1 import CompactOffloadEngine
        worker=engines.CompAct.__new__(engines.CompAct)
        engines.Base.__init__(worker,'compact_budgeted',RUN/'generation_cache/compact')
        worker.engine=CompactOffloadEngine(worker.cache/'decode')
        worker.utils=engines.official_functions(ROOT/'third_party/recent_compact/utils.py',('create_prompt','parse_output_without_sentence'))
        engine=worker.engine;method='compact_budgeted'
    else:
        import types
        worker=engines.LongRefiner(RUN/'generation_cache/longrefiner')
        worker.method='longrefiner_repair';method='longrefiner_repair'
        official=type(worker.worker).__mro__[1]
        corrected=fixed.corrected_class(official.run_doc_structuring.__globals__)
        for name in ['run_doc_structuring','run_all_search','select_by_budget']:
            setattr(worker.worker,name,types.MethodType(getattr(corrected,name),worker.worker))
        engine=worker.worker.model
    trace_generator(engine,phase)
    # Preserve original algorithm, only request the two registered budgets.
    core.BUDGETS=(256,512)
    from scripts.recent_baselines_technical_gate_v2 import fixture
    smoke=worker(fixture())
    assert len(smoke)==2 and any(p['evidence_text'].strip() for p in smoke),'Empty synthetic compressor output'
    for p in smoke:
        assert len(worker.reader_tokenizer.encode(p['evidence_text'],add_special_tokens=False))==p['evidence_tokens']<=p['evidence_budget']
    u.immutable(RUN/'technical_smoke'/(phase+'.json'),dict(passed=True,plans=smoke,criterion='nonempty material and schema/budget, no QA scoring'))
    for i,x in enumerate([r for r in sources() if r['dataset']=='hotpot'],1):
        s=x['source'];start=time.perf_counter()
        try:plans=worker(s)
        except (ValueError,KeyError,IndexError) as error:
            # Same declared malformed-output convention as the historical adapters.
            plans=core.failed_plans(s,method,str(error))
        for p in plans:
            assert p['method']==method and p['evidence_budget'] in (256,512)
            assert len(worker.reader_tokenizer.encode(p['evidence_text'],add_special_tokens=False))==p['evidence_tokens']<=p['evidence_budget']
            assert core.fp(p['evidence_text'])==p['evidence_sha256']
        u.immutable(RUN/'compressions'/phase/(s['question_id']+'.json'),dict(source_hash=u.fp(s),plans=plans,seconds=time.perf_counter()-start))
        state('running',phase=phase,completed=i,expected=48)

def assemble():
    from transformers import AutoTokenizer
    counter=hp.CounterTokens();bundle=inputs();plans=deepcopy(u.load(BASE/'public/fixed_plans.json'))
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.5-9B',local_files_only=True,trust_remote_code=False)
    for x in sources():
        s=x['source'];q=s['question_id'];d=x['dataset']
        if d=='hotpot':
            for phase,method in [('compact','compact_budgeted'),('longrefiner','longrefiner_repair')]:
                row=u.load(RUN/'compressions'/phase/(q+'.json'));assert row['source_hash']==u.fp(s)
                for p in row['plans']:plans.append(normalize(d,method,p,'new_'+phase))
            continue
        cached=bundle['inputs'][q];scores=cached['ranking']['scores'];seeds=[str(v) for v in cached['retrieval']['paths'][0]]
        role=u.load(RUN/'proposals'/(u.fp([q,'role'])+'.json'))['parsed']
        raw=u.load(RUN/'proposals'/(u.fp([q,'direct'])+'.json'))['parsed']
        ir=u.load(RUN/'itercomp'/(q+'.json'));native=[dict(v,id=v['uid']) for v in it.units(s)]
        for b in [256,512]:
            global_plan,ap3=hp.short.short_plans(s,scores,seeds,counter,b,b)
            if global_plan.get('selection_error') or ap3.get('selection_error'):
                p,trace=ap3,dict(fallback='baseline_selection_failure')
            else:p,trace=hp.anchored.quote_plan(s,scores,global_plan,ap3,role,counter,'quote_anchor')
            plans.append(normalize(d,'quote_anchor',dict(p,question_id=q,budget=b,trace=trace),'new_role_proposal'))
            direct_plan=dict(direct.pack(s,cached,raw,b,counter),arm='direct')
            if direct_plan['status']=='selection_failed':
                rp=direct_plan
            else:
                candidates=repair.candidate_bundle(s,direct_plan,native,scores,counter)
                rp=repair.material(direct_plan,candidates,'rank_add')
            plans.append(normalize(d,'rank_add',rp,'new_direct_proposal'))
            ip=itcap.cap_native(s,ir,b,counter)
            plans.append(normalize(d,'itercomp_cap',dict(ip,question_id=q,budget=b),'new_itercomp'))
        plans.append(normalize(d,'itercomp_native',dict(question_id=q,budget=0,evidence_text=ir['evidence_text'],evidence_tokens=ir['evidence_tokens'],trace=dict(native_status=ir['status'])),'new_itercomp'))
    expected={(x['dataset'],x['source']['question_id'],b,m) for x in sources() for b in [256,512] for m in ARMS}
    expected|={(x['dataset'],x['source']['question_id'],0,m) for x in sources() for m in ['itercomp_native','uncompressed']}
    assert len(plans)==len(expected)==1536
    assert {(p['dataset'],p['question_id'],p['budget'],p['method']) for p in plans}==expected
    source_by={x['source']['question_id']:x['source'] for x in sources()};requests={};mapping=[]
    for p in plans:
        assert counter.count(p['evidence_text'])==p['evidence_tokens']
        assert p['budget']==0 or p['evidence_tokens']<=p['budget']
        key=None
        if p['status']=='ready':
            req=qa_request(p,source_by[p['question_id']])
            prompt=tok.apply_chat_template(req['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            n=len(tok.encode(prompt,add_special_tokens=False))
            if n>CONFIG['reader_max_input']:
                assert p['method']=='uncompressed','Compressed request exceeds reader input cap'
                p['status']='input_infeasible';p['reason']='uncompressed exceeds8192; no truncation'
            else:key=req['key'];requests[key]=req
        mapping.append(dict(dataset=p['dataset'],question_id=p['question_id'],budget=p['budget'],method=p['method'],plan_hash=u.fp(p),key=key))
    replay=u.load(BASE/'public/replay.json')
    u.immutable(RUN/'plans.json',plans);u.immutable(RUN/'requests.json',requests);u.immutable(RUN/'mapping.json',mapping)
    u.immutable(RUN/'assembly.json',dict(signature=u.sha(BASE/'manifest.json'),plans=1536,unique_requests=len(requests),
        replay_requests=len(set(requests)&set(replay)),new_requests=len(set(requests)-set(replay)),references_read=False))
    state('assembled',**u.load(RUN/'assembly.json'))

def qa():
    requests=u.load(RUN/'requests.json');replay=u.load(BASE/'public/replay.json');backend=None
    for i,(key,r) in enumerate(sorted(requests.items()),1):
        if key in replay:
            v=replay[key];assert v['request_identity']==u.fp([r['reader'],r['messages'],r['max_output']])
            result=v['result'];origin='exact_historical_replay'
        else:
            if backend is None:backend=qwen()
            result=call(backend,r['messages'],r['max_output'],key,'qa');origin='new_inference'
        u.immutable(RUN/'outputs'/(key+'.json'),dict(signature=u.sha(BASE/'manifest.json'),key=key,origin=origin,result=result))
        if i%20==0 or i==len(requests):state('running',phase='qa',completed=i,expected=len(requests))

def worker(phase):
    import fcntl
    verify();RUN.mkdir(parents=True,exist_ok=True)
    if phase=='assemble':assemble();return
    with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
        deadline=time.monotonic()+CONFIG['memory_wait_seconds']
        while True:
            acquired=False;free=u.free_memory()
            if free>=CONFIG['admission_mib']:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                except BlockingIOError:pass
            if acquired:
                if u.free_memory()>=CONFIG['admission_mib']:break
                fcntl.flock(lock,fcntl.LOCK_UN)
            state('waiting_for_memory',phase=phase,free_mib=free,required_mib=CONFIG['admission_mib'])
            if time.monotonic()>deadline:raise TimeoutError('Admission timeout')
            time.sleep(20)
        torch=setup_gpu();start=time.perf_counter();okay=False
        state('running',phase=phase,completed=0)
        try:
            if phase in ['compact','longrefiner']:compressors(phase)
            else:globals()[phase]()
            okay=True
        finally:
            import resource
            u.immutable(RUN/'resources'/(phase+'.json'),dict(completed=okay,phase_seconds=time.perf_counter()-start,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                cpu_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))

def run():
    RUN.mkdir(parents=True,exist_ok=True);sid=verify(models=True)
    assert not (RUN/'assembly.json').exists() and not list((RUN/'calls').rglob('*.json'))
    from .paper_competition_resources_v1 import preflight
    preflight()
    deadline=time.monotonic()+CONFIG['wall_limit_seconds']
    for phase in ['longrefiner','compact','propose','itercomp','assemble','qa']:
        with (RUN/(phase+'.log')).open('a') as log:
            p=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.paper_competition_v1','worker',phase],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (RUN/'active_worker.json').write_text(json.dumps(dict(phase=phase,pid=p.pid)))
            try:code=p.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                raise TimeoutError('Registered total runtime ceiling reached')
            if code!=0:raise RuntimeError('Phase failed: '+phase)
    req=u.load(RUN/'requests.json');assert len(list((RUN/'outputs').glob('*.json')))==len(req)
    u.immutable(RUN/'completion.json',dict(status='completed',signature=sid,time=datetime.now(timezone.utc).isoformat(),logical_conditions=1536,unique_requests=len(req),references_read=False))
    state('completed',logical_conditions=1536)
    from .paper_competition_analysis_v1 import analyze
    analyze()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);parser.add_argument('phase',nargs='?');args=parser.parse_args()
    try:
        if args.mode=='run':run()
        else:worker(args.phase)
    except Exception as error:
        RUN.mkdir(parents=True,exist_ok=True);state('failed',phase=args.phase,error=repr(error),traceback=traceback.format_exc());raise
