"""Durable, no-expiry full-GPU QA queue for frozen postprocessing plans."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,subprocess,time,traceback
from . import proposal_retention_v1 as n
u=n.u;ROOT=n.ROOT;BASE=n.BASE;RUN=n.RUN


def state(status,**fields):
    RUN.mkdir(parents=True,exist_ok=True)
    row=dict(status=status,time=datetime.now(timezone.utc).isoformat(),**fields)
    tmp=RUN/'status.tmp';tmp.write_text(json.dumps(row,indent=2),encoding='utf-8');os.replace(tmp,RUN/'status.json')
    print(json.dumps(row),flush=True)


def preflight():
    from transformers import AutoTokenizer
    sid=n.verify();tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.5-9B',local_files_only=True,trust_remote_code=False)
    pp=u.load(BASE/'public/plans.json');rr=u.load(BASE/'public/requests.json');mm=u.load(BASE/'public/mapping.json')
    assert len(pp)==len(mm)==n.CONFIG['logical_conditions']
    pi={(v['dataset'],v['question_id'],v['extractor'],v['arm']):v for v in pp}
    mi={(v['dataset'],v['question_id'],v['extractor'],v['arm']):v for v in mm}
    assert len(pi)==len(mi)==len(pp) and set(pi)==set(mi)
    lengths={}
    for k,r in rr.items():
        assert k==r['key']==u.fp([r['reader'],r['messages'],r['max_output']])
        assert r['reader']==n.CONFIG['reader'] and r['max_output'] in (128,2048)
        prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        lengths[k]=len(tok.encode(prompt,add_special_tokens=False));assert lengths[k]<=n.CONFIG['max_input']
    for identity,v in pi.items():
        assert mi[identity]['plan_hash']==u.fp(v)
        assert len(tok.encode(v['evidence_text'],add_special_tokens=False))==v['evidence_tokens']<=512
        if identity[-1]=='retain_fill':
            other=identity[:-1]+('knapsack_matched',)
            assert v['evidence_tokens']==pi[other]['evidence_tokens']
            assert lengths[mi[identity]['key']]==lengths[mi[other]['key']],'Full reader input is not length matched'
    preparation=u.load(BASE/'preparation.json')
    result=dict(signature=sid,passed=True,logical_conditions=len(pp),unique_requests=len(rr),
        exact_primary_evidence_pairs=192,exact_primary_full_prompt_pairs=192,max_prompt=max(lengths.values()),
        initial_sparse_pair_warning=not preparation['passed'],references_read=False,new_model_calls=0)
    u.immutable(RUN/'preflight.json',result);print(json.dumps(result))


def validate_outputs(sid,requests):
    for path in (RUN/'outputs').glob('*.json'):
        v=u.load(path);k=path.stem
        assert k in requests and v['key']==k and v['signature']==sid
        assert v['request_identity']==u.fp([requests[k]['reader'],requests[k]['messages'],requests[k]['max_output']])
        assert isinstance(v['result']['text'],str) and v['result']['finish_reason'] in ('stop','length')


def worker():
    import fcntl
    import torch
    from .paper_competition_fastqueue_v1 import preload,NvmlMemory
    from .mainline_service_v1 import generate
    sid=n.verify();attempt=RUN/'attempts'/os.environ['RETENTION_ATTEMPT'];requests=u.load(BASE/'public/requests.json')
    validate_outputs(sid,requests)
    pending=sorted(k for k in requests if not (RUN/'outputs'/(k+'.json')).exists())
    if not pending:return 0
    current=None;begin=time.perf_counter();okay=False
    torch.set_num_threads(8);torch.manual_seed(30);torch.cuda.manual_seed_all(30)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state('preloading_cpu_weights',attempt=attempt.name,pending=len(pending),expires_at=None)
    try:
        backend=preload()
        torch.cuda.set_per_process_memory_fraction(n.CONFIG['gpu_cap_mib']*2**20/torch.cuda.get_device_properties(0).total_memory)
        warm=torch.empty(1,dtype=torch.uint8,device='cuda');del warm
        torch.cuda.reset_peak_memory_stats();monitor=NvmlMemory()
        with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
            last=-float('inf');checks=0;reservation_ooms=0
            while True:
                free=monitor.free_mib();checks+=1;acquired=False
                if free>=n.CONFIG['admission_mib']:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                    except BlockingIOError:pass
                if acquired:
                    try:
                        if monitor.free_mib()>=n.CONFIG['admission_mib']:
                            amount=max(0,n.CONFIG['reserve_mib']*2**20-torch.cuda.memory_reserved())
                            allocation=torch.empty(amount,dtype=torch.uint8,device='cuda');del allocation
                            assert torch.cuda.memory_reserved()>=n.CONFIG['reserve_mib']*2**20
                            break
                    except torch.OutOfMemoryError:
                        reservation_ooms+=1;torch.cuda.empty_cache()
                    fcntl.flock(lock,fcntl.LOCK_UN)
                if time.monotonic()-last>=20:
                    state('waiting_for_memory',attempt=attempt.name,cpu_weights_ready=True,free_mib=round(free),
                        required_mib=n.CONFIG['admission_mib'],pending=len(pending),poll_seconds=.1,expires_at=None)
                    last=time.monotonic()
                time.sleep(n.CONFIG['poll_seconds'])
            u.immutable(attempt/'reservation.json',dict(free_mib_before=free,checks=checks,reservation_ooms=reservation_ooms,time=time.time()))
            state('moving_weights_to_gpu',attempt=attempt.name,pending=len(pending))
            backend.model=backend.model.to('cuda').eval()
            assert all(v.device.type=='cuda' and v.dtype==torch.bfloat16 for v in backend.model.parameters())
            u.immutable(attempt/'placement.json',dict(all_weights_cuda=True,precision='bfloat16',quantization=False,
                allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20))
            state('running',attempt=attempt.name,completed=len(requests)-len(pending),expected=len(requests))
            for i,k in enumerate(pending,1):
                current=k;r=requests[k]
                prompt=backend.tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                assert len(backend.tokenizer.encode(prompt,add_special_tokens=False))<=n.CONFIG['max_input']
                u.immutable(attempt/'calls'/(k+'.started.json'),dict(signature=sid,key=k,request_identity=k,time=time.time()))
                result=generate(backend,r['messages'],r['max_output'])
                receipt=dict(signature=sid,key=k,request_identity=k,result=result,origin='new_inference',attempt=attempt.name)
                u.immutable(attempt/'calls'/(k+'.json'),receipt)
                u.immutable(RUN/'outputs'/(k+'.json'),receipt)
                current=None
                if i%5==0 or i==len(pending):state('running',attempt=attempt.name,completed=len(requests)-len(pending)+i,expected=len(requests))
            okay=True
    except torch.OutOfMemoryError as exc:
        u.immutable(attempt/'resource_failure.json',dict(kind='cuda_out_of_memory',key=current,error=str(exc),time=time.time()))
        state('requeuing_after_memory_failure',attempt=attempt.name,expires_at=None)
        return 75
    finally:
        import resource
        u.immutable(attempt/'resources.json',dict(completed=okay,elapsed_including_wait=time.perf_counter()-begin,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0,
            cpu_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    return 0


def run():
    import fcntl
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sid=n.verify(models=True);assert not (RUN/'completion.json').exists()
        if not (RUN/'preflight.json').exists():preflight()
        assert u.load(RUN/'preflight.json')['signature']==sid
        requests=u.load(BASE/'public/requests.json');replay=u.load(BASE/'public/replay.json')
        # Every known completed call is durable before the next call begins.
        validate_outputs(sid,requests)
        for directory in sorted((RUN/'attempts').glob('*')):
            incomplete=[]
            for path in (directory/'calls').glob('*.started.json'):
                key=path.name.removesuffix('.started.json');done=directory/'calls'/(key+'.json')
                if done.exists():
                    v=u.load(done);assert v['signature']==sid and v['key']==key
                    target=RUN/'outputs'/(key+'.json')
                    if not target.exists():u.immutable(target,v)
                else:incomplete.append(key)
            if incomplete:
                failure=u.load(directory/'resource_failure.json')
                assert failure['kind']=='cuda_out_of_memory' and incomplete==[failure['key']],'Unknown interrupted call; do not replay'
        for k,v in replay.items():
            assert v['request_identity']==k and k in requests
            target=RUN/'outputs'/(k+'.json')
            if not target.exists():u.immutable(target,dict(signature=sid,key=k,request_identity=k,origin='exact_historical_replay',result=v['result']))
        number=len(list((RUN/'attempts').glob('*')))
        while len(list((RUN/'outputs').glob('*.json')))<len(requests):
            number+=1;name=f'{number:06d}';directory=RUN/'attempts'/name;directory.mkdir(parents=True)
            with (directory/'worker.log').open('x') as log:
                child=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.proposal_retention_runtime_v1','worker'],
                    cwd=ROOT,env=dict(os.environ,RETENTION_ATTEMPT=name),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                (RUN/'active_worker.json').write_text(json.dumps(dict(pid=child.pid,attempt=name)))
                code=child.wait()
            if code==0:break
            if code==75 and (directory/'resource_failure.json').exists():
                state('waiting_to_requeue',next_attempt=number+1,expires_at=None);time.sleep(n.CONFIG['retry_delay_seconds']);continue
            raise RuntimeError('Non-resource failure; inspect '+str(directory/'worker.log'))
        validate_outputs(sid,requests)
        assert {v.stem for v in (RUN/'outputs').glob('*.json')}==set(requests)
        u.immutable(RUN/'completion.json',dict(signature=sid,status='completed',time=datetime.now(timezone.utc).isoformat(),
            logical_conditions=n.CONFIG['logical_conditions'],unique_requests=len(requests),references_read=False))
        state('completed',unique_requests=len(requests))
    # GPU worker has exited before scoring begins.
    from .proposal_retention_analysis_v1 import analyze
    analyze()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','run','worker']);args=parser.parse_args()
    try:
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run()
        else:raise SystemExit(worker())
    except Exception as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc());raise
