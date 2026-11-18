"""Durable phase execution, full GPU weights, no queue expiration or score peeking."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,subprocess,time,traceback,math,threading
from . import paper_confirmation_v1 as n
from . import full_comparison_v1 as full
u=n.u;ROOT=n.ROOT;BASE=n.BASE;RUN=n.RUN
PHASES=('retrieve','rank','propose','assemble','qa_qwen','qa_phi')

def state(status,**fields):
    RUN.mkdir(parents=True,exist_ok=True)
    row=dict(status=status,time=datetime.now(timezone.utc).isoformat(),expires_at=None,**fields)
    tmp=RUN/'status.tmp';tmp.write_text(json.dumps(row,indent=2),encoding='utf-8');os.replace(tmp,RUN/'status.json')
    print(json.dumps(row),flush=True)

def put(path,value):
    # Atomic visibility: an interrupted write is never a completed model result.
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert u.load(path)==value;return
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)

def tokenizer(reader):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT/'models'/reader,local_files_only=True,trust_remote_code=False)

def prompt_tokens(tok,messages):
    return len(tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))

def preflight():
    sid=n.verify();ss=n.sources();tok=tokenizer('Qwen3.5-9B');rtok=tokenizer('Qwen3-Reranker-4B')
    btok=full.beam.tokenizer();maximum=0;rankmax=0;contexts={};sentences=0
    for x in ss:
        s=x['source'];d=x['dataset'];contexts.setdefault(d,[]).append(len(s['paragraphs']))
        assert set(s)=={'question_id','native_id','mode','question','paragraphs'}
        assert 0<len(btok.encode(s['question'].removesuffix('?'),add_special_tokens=False))<256
        for kind in ('role','direct'):
            length=prompt_tokens(tok,n.old.proposal_messages(s,kind));maximum=max(maximum,length)
            assert length<=n.CONFIG['proposer_max_input'],s['question_id']
        for uid,text in full.ranker.documents(s):
            ids=full.ranker.rank_input_ids(rtok,s['question'],text);rankmax=max(rankmax,len(ids));sentences+=1
    # Loading Phi tokenizer now detects incompatible checkpoint assets before any model call.
    tokenizer('Phi-4-mini-instruct')
    result=dict(signature=sid,passed=True,sources=len(ss),max_proposer_input=maximum,max_rank_input=rankmax,
                rank_forward_calls_expected=sentences,native_context_ranges={d:[min(v),max(v)] for d,v in contexts.items()},
                new_model_calls=0,references_read=False)
    put(RUN/'preflight.json',result);print(json.dumps(result),flush=True)

def result_path(phase,identity):return RUN/'cache'/phase/(u.fp(identity)+'.json')

def cached(phase,identity,source=None):
    path=result_path(phase,identity)
    if not path.exists():return None
    value=u.load(path);assert value['signature']==u.sha(BASE/'manifest.json') and value['identity']==identity
    if source is not None:assert value['source_hash']==u.fp(source)
    return value

def durable(phase,identity,fn,source=None):
    previous=cached(phase,identity,source)
    if previous:return previous
    attempt=RUN/'attempts'/os.environ['CONFIRMATION_ATTEMPT'];key=u.fp([phase,identity])
    for old in (RUN/'attempts').glob('*/calls/'+key+'.started.json'):
        failure=old.parent.parent/'resource_failure.json'
        assert failure.exists() and u.load(failure)['active_key']==key,'Unknown interrupted call: '+str(old)
    start=dict(signature=u.sha(BASE/'manifest.json'),phase=phase,identity=identity,key=key,time=time.time())
    put(attempt/'calls'/(key+'.started.json'),start)
    os.environ['CONFIRMATION_ACTIVE_KEY']=key
    begin=time.perf_counter();value=fn()
    result=dict(signature=start['signature'],identity=identity,source_hash=u.fp(source) if source is not None else None,
                value=value,seconds=time.perf_counter()-begin,attempt=attempt.name)
    put(result_path(phase,identity),result)
    os.environ.pop('CONFIRMATION_ACTIVE_KEY',None)
    return result

def gen(backend,messages,limit,identity):
    from .mainline_service_v1 import generate
    phase='generation';key=[backend.model_name,messages,limit]
    previous=cached(phase,key)
    if previous:return previous['value']
    assert prompt_tokens(backend.tokenizer,messages)<=n.CONFIG['proposer_max_input' if identity=='propose' else 'reader_max_input']
    return durable(phase,key,lambda:generate(backend,messages,limit))['value']

def retrieve(engine):
    torch,tok,model=engine
    for i,x in enumerate(n.sources(),1):
        s=x['source'];q=s['question_id']
        def compute():
            encoded,audit=full.beam.codes(tok,s,'cuda');costs,handle=full.beam.encoder_meter(model)
            try:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=full.beam.count_logits(torch,model,encoded['q_codes'][0]).float()
                    raw=int(logits.argmax(dim=-1).item())+1;hops=min(raw,len(s['paragraphs']))
                    prediction=model(**encoded,sf_idx=[[]],hop=hops)
            finally:handle.remove()
            torch.cuda.synchronize();paths=prediction['current_preds']
            assert paths and all(len(p)==hops and len(set(p))==hops and all(0<=j<len(s['paragraphs']) for j in p) for p in paths)
            return dict(paths=paths,predicted_hops=hops,raw_predicted_hops=raw,hop_logits=logits[0].cpu().tolist(),tokenization=audit,costs=costs)
        durable('retrieve',q,compute,s)
        if i%10==0 or i==1935:state('running',phase='retrieve',completed=i,expected=1935)

def rank(engine):
    torch,tok,model=engine
    for i,x in enumerate(n.sources(),1):
        s=x['source'];q=s['question_id']
        def compute():
            scores={};lengths=[]
            with torch.inference_mode():
                for uid,text in full.ranker.documents(s):
                    ids=full.ranker.rank_input_ids(tok,s['question'],text)
                    scores[uid]=float(full.ranker.two_head_scores(torch,tok,model,ids)['fp32']);lengths.append(len(ids))
            assert all(math.isfinite(v) for v in scores.values())
            return dict(scores=scores,costs=dict(input_tokens=sum(lengths),forward_calls=len(lengths)))
        durable('rank',q,compute,s)
        if i%5==0 or i==1935:state('running',phase='rank',completed=i,expected=1935)

def propose(backend):
    for i,x in enumerate(n.sources(),1):
        s=x['source'];q=s['question_id']
        for kind in ('role','direct'):
            if cached('proposal',[q,kind],s):continue
            messages=n.old.proposal_messages(s,kind);attempts=[]
            for j in range(2 if kind=='role' else 1):
                result=gen(backend,messages,2048,'propose')
                parsed=n.old.hp.parse_result(s,'role',result) if kind=='role' else n.old.direct.inspect(s,result)
                attempts.append(dict(messages=messages,result=result,parsed=parsed))
                if kind!='role' or parsed['status']!='invalid':break
                correction='Repair your output once. Return the complete valid JSON object only. Validation errors: '+json.dumps(parsed['errors'])+'. Use only exact quotes from the original sources; do not invent facts. If no supported evidence is identifiable, return {"roles":[]}.'
                messages=n.old.proposal_messages(s,kind)+[dict(role='assistant',content=result['text']),dict(role='user',content=correction)]
            # No model call inside this final materialization: it is safely resumable.
            put(result_path('proposal',[q,kind]),dict(signature=u.sha(BASE/'manifest.json'),identity=[q,kind],source_hash=u.fp(s),
                value=dict(parsed=parsed,attempts=attempts)))
        if i%5==0 or i==1935:state('running',phase='propose',completed=i,expected=1935)

def assemble():
    from copy import deepcopy
    sid=n.verify();tok=tokenizer('Qwen3.5-9B');phi=tokenizer('Phi-4-mini-instruct')
    class Count:
        exact=True
        def count(self,text):return len(tok.encode(text,add_special_tokens=False))
    count=Count();plans=[];mapping=[];requests={};token_counts={};cpu=[]
    hotpot_native=u.load(BASE/'public/hotpot_native_units.json')
    def register(p,s,reader):
        p=deepcopy(p);p['reader']=reader;p['source_hash']=u.fp(s)
        key=None
        if p['status']=='ready':
            req=n.qa_request(p,s,reader);key=req['key'];requests[key]=req
            tt=tok if reader=='Qwen3.5-9B' else phi
            length=prompt_tokens(tt,req['messages']);assert length<=n.CONFIG['reader_max_input']
            token_counts[key]=length;p['actual_reader_evidence_tokens']=len(tt.encode(p['evidence_text'],add_special_tokens=False))
        plans.append(p);mapping.append(dict(plan_hash=u.fp(p),key=key))
    for i,x in enumerate(n.sources(),1):
        started=time.perf_counter();s=x['source'];q=s['question_id'];d=x['dataset']
        ranking=cached('rank',q,s)['value'];retrieved=cached('retrieve',q,s)['value'];scores=ranking['scores'];seeds=[str(v) for v in retrieved['paths'][0]]
        proposals={k:cached('proposal',[q,k],s)['value']['parsed'] for k in ('role','direct')}
        global_plan,ap3=n.old.hp.short.short_plans(s,scores,seeds,count,512,512)
        if global_plan.get('selection_error') or ap3.get('selection_error'):quote,trace=deepcopy(ap3),dict(fallback='baseline_selection_failure')
        else:quote,trace=n.old.hp.anchored.quote_plan(s,scores,global_plan,ap3,proposals['role'],count,'quote_anchor')
        direct=dict(n.old.direct.pack(s,dict(ranking=ranking,retrieval=retrieved),proposals['direct'],512,count),arm='direct')
        if direct['status']=='selection_failed':added=direct
        else:
            native=hotpot_native[q] if d=='hotpot' else [dict(v,id=v['uid']) for v in n.old.it.units(s)]
            candidates=n.old.repair.candidate_bundle(s,direct,native,scores,count)
            added=n.old.repair.material(direct,candidates,'rank_add')
        pp={'quote_anchor':dict(quote,trace=trace),'rank_add':added,'ap3':ap3,'strong_rerank':global_plan}
        for method,p in pp.items():
            p=n.old.normalize(d,method,dict(p,question_id=q,budget=512),'fresh_frozen_system')
            p.update(study='complete',extractor=None)
            assert count.count(p['evidence_text'])==p['evidence_tokens']<=512
            for reader in n.CONFIG['readers']:register(p,s,reader)
        if x['diagnostic']:
            for kind in ('role','direct'):
                dd=n.retention.make_plans(s,scores,proposals[kind],kind,count.count)
                for p in dd:
                    p.update(study='retention',dataset=d,method=p['arm'])
                    register(p,s,'Qwen3.5-9B')
                a,b=[n.qa_request(dict(v,dataset=d),s,'Qwen3.5-9B')['key'] for v in (dd[0],dd[3])]
                assert token_counts[a]==token_counts[b],'Primary full prompt lengths differ'
        cpu.append(dict(question_id=q,packing_seconds=time.perf_counter()-started))
        if i%25==0:state('assembling',completed=i,expected=1935)
    assert len(plans)==len(mapping)==n.CONFIG['logical_conditions']
    assert len({(p['dataset'],p['question_id'],p['study'],p['reader'],p['extractor'],p['method']) for p in plans})==len(plans)
    for name,value in [('plans',plans),('mapping',mapping),('requests',requests),('packing_costs',cpu),('prompt_lengths',token_counts)]:put(RUN/(name+'.json'),value)
    put(RUN/'assembly.json',dict(signature=sid,logical_conditions=len(plans),unique_requests=len(requests),
        exact_qwen_primary_pairs=1024,phi_evidence='identical Qwen-budgeted strings; actual Phi lengths differ',references_read=False))

def qa(backend):
    requests=u.load(RUN/'requests.json');selected=[(k,r) for k,r in sorted(requests.items()) if r['reader']==backend.model_name]
    sid=u.sha(BASE/'manifest.json')
    for i,(key,r) in enumerate(selected,1):
        path=RUN/'outputs'/(key+'.json')
        if path.exists():
            out=u.load(path);assert out['signature']==sid and out['key']==key
        else:
            result=gen(backend,r['messages'],r['max_output'],'qa')
            put(path,dict(signature=sid,key=key,result=result,request_identity=u.fp([r['reader'],r['messages'],r['max_output']])))
        if i%10==0 or i==len(selected):state('running',phase='qa',reader=backend.model_name,completed=i,expected=len(selected))

def preload(phase):
    import torch
    from transformers import AutoModelForCausalLM
    if phase=='retrieve':
        tok,model=full.beam.make_model('cpu');model.float()
        model.load_state_dict(torch.load(ROOT/'runs/path_retrieval_execution_v1/best.pt',map_location='cpu',weights_only=True),strict=True)
        return (torch,tok,model.eval())
    if phase=='rank':
        model=AutoModelForCausalLM.from_pretrained(full.ranker.MODEL,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        return torch,tokenizer('Qwen3-Reranker-4B'),model
    if phase in ('propose','qa_qwen'):
        from .paper_competition_fastqueue_v1 import preload as qwen
        return qwen()
    from .backend import TransformersBackend
    reader='Phi-4-mini-instruct';v=TransformersBackend.__new__(TransformersBackend)
    v.model=AutoModelForCausalLM.from_pretrained(ROOT/'models'/reader,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    v.tokenizer=tokenizer(reader);v.path=str(ROOT/'models'/reader);v.model_name=reader;v.torch=torch;v.device='cuda';v.max_input_tokens=8192;v.lock=threading.Lock()
    return v

def worker(phase):
    n.verify()
    if phase=='assemble':assemble();return 0
    import torch,fcntl,resource
    from .paper_competition_fastqueue_v1 import NvmlMemory
    attempt=RUN/'attempts'/os.environ['CONFIRMATION_ATTEMPT'];okay=False;begin=time.perf_counter()
    torch.set_num_threads(8);torch.manual_seed(30);torch.cuda.manual_seed_all(30)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state('preloading_cpu_weights',phase=phase,attempt=attempt.name)
    try:
        engine=preload(phase);model=engine[2] if isinstance(engine,tuple) else engine.model
        torch.cuda.set_per_process_memory_fraction(n.CONFIG['cap_mib'][phase]*2**20/torch.cuda.get_device_properties(0).total_memory)
        monitor=NvmlMemory();torch.cuda.reset_peak_memory_stats()
        with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
            last=-float('inf');checks=0
            while True:
                free=monitor.free_mib();checks+=1;acquired=False
                if free>=n.CONFIG['admission_mib'][phase]:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                    except BlockingIOError:pass
                if acquired:
                    try:
                        if monitor.free_mib()>=n.CONFIG['admission_mib'][phase]:
                            amount=max(0,n.CONFIG['reserve_mib'][phase]*2**20-torch.cuda.memory_reserved())
                            reservation=torch.empty(amount,dtype=torch.uint8,device='cuda');del reservation
                            break
                    except torch.OutOfMemoryError:torch.cuda.empty_cache()
                    fcntl.flock(lock,fcntl.LOCK_UN)
                if time.monotonic()-last>=20:
                    state('waiting_for_memory',phase=phase,attempt=attempt.name,free_mib=round(free),required_mib=n.CONFIG['admission_mib'][phase],cpu_weights_ready=True,poll_seconds=.1)
                    last=time.monotonic()
                time.sleep(.1)
            put(attempt/'reservation.json',dict(phase=phase,checks=checks,free_mib_before=free,claimed_mib=torch.cuda.memory_reserved()/2**20))
            model.to('cuda').eval();assert all(v.device.type=='cuda' for v in model.parameters())
            if phase!='retrieve':assert all(v.dtype==torch.bfloat16 for v in model.parameters())
            state('running',phase=phase,attempt=attempt.name,all_weights_cuda=True)
            if phase=='retrieve':retrieve(engine)
            elif phase=='rank':rank(engine)
            elif phase=='propose':propose(engine)
            else:qa(engine)
            okay=True
    except torch.OutOfMemoryError as exc:
        put(attempt/'resource_failure.json',dict(kind='cuda_out_of_memory',active_key=os.environ.get('CONFIRMATION_ACTIVE_KEY'),error=str(exc),time=time.time()))
        state('requeuing_after_memory_failure',phase=phase);return 75
    finally:
        put(attempt/'resources.json',dict(phase=phase,completed=okay,elapsed_including_wait=time.perf_counter()-begin,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0,
            cpu_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    return 0

def run():
    import fcntl
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);sid=n.verify(models=True)
        if not (RUN/'preflight.json').exists():preflight()
        assert u.load(RUN/'preflight.json')['signature']==sid
        for phase in PHASES:
            done=RUN/'phases'/(phase+'.json')
            if done.exists():assert u.load(done)['signature']==sid;continue
            while True:
                number=len(list((RUN/'attempts').glob('*')))+1;name=f'{number:06d}';folder=RUN/'attempts'/name;folder.mkdir(parents=True)
                with (folder/'worker.log').open('x') as log:
                    child=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.paper_confirmation_runtime_v1','worker',phase],
                        cwd=ROOT,env=dict(os.environ,CONFIRMATION_ATTEMPT=name),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    active=dict(phase=phase,pid=child.pid,coordinator=os.getpid(),attempt=name)
                    (RUN/'active_worker.json').write_text(json.dumps(active),encoding='utf-8');code=child.wait()
                if code==0:put(done,dict(signature=sid,phase=phase,attempt=name,completed_at=datetime.now(timezone.utc).isoformat()));break
                if code==75 and (folder/'resource_failure.json').exists():time.sleep(1);continue
                raise RuntimeError('Non-resource failure: '+str(folder/'worker.log'))
        requests=u.load(RUN/'requests.json');assert {p.stem for p in (RUN/'outputs').glob('*.json')}==set(requests)
        put(RUN/'completion.json',dict(signature=sid,status='completed',time=datetime.now(timezone.utc).isoformat(),
                                     logical_conditions=n.CONFIG['logical_conditions'],unique_requests=len(requests),references_read=False))
        state('completed',unique_requests=len(requests))
    from .paper_confirmation_analysis_v1 import analyze
    analyze()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','run','worker']);p.add_argument('phase',nargs='?');args=p.parse_args()
    try:
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run()
        else:raise SystemExit(worker(args.phase))
    except Exception as exc:state('failed',error=repr(exc),traceback=traceback.format_exc());raise
