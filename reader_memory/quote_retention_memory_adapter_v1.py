"""Resource-only admission adjustment after exact cached-output replay at low memory."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import subprocess
import time
from . import quote_retention_mechanism_v2 as n
from . import paper_component_controls_v2 as component

ROOT=n.ROOT
BASE=ROOT/'research/quote_retention_mechanism_v2/memory_review'
RUN=n.RUN/'memory_adapter_v1'
CONFIG=dict(version='quote-retention-memory-adapter-v1',reader='Qwen3.5-9B',
    admission_mib=20250,reserve_mib=19000,cap_mib=19500,
    fallback_admission_mib=26000,fallback_reserve_mib=24000,fallback_cap_mib=24500,
    model='unchanged full Qwen3.5-9B',precision='bfloat16',attention='sdpa',
    placement='all_weights_cuda',seed=30,wait_timeout=None,wall_timeout=None,
    old_threshold_is_reservation_policy_not_measured_minimum=True,
    probe_requests='three cached Qwen requests: longest Hotpot prompt, longest MuSiQue prompt, longest observed MuSiQue output',
    acceptance='all three replayed completion token sequences exactly match historical output; all weights remain BF16 on CUDA',
    resource_retry='After first actual worker OOM return to original 26000/24000/24500 policy indefinitely',
    science_configuration_changed=False)
FILES=['reader_memory/quote_retention_memory_adapter_v1.py',
       'tools/quote_retention_memory_adapter_v1_remote.py',
       'research/quote_retention_mechanism_v2/memory_review/RESOURCE_AMENDMENT.md']


def freeze():
    sid=n.verify()
    n.put(BASE/'adapter_manifest.json',dict(config=CONFIG,parent_signature=sid,
        files={f:n.u.sha(ROOT/f) for f in FILES},created_at=datetime.now(timezone.utc).isoformat()))


def verify():
    value=n.u.load(BASE/'adapter_manifest.json')
    assert value['config']==CONFIG and n.verify()==value['parent_signature']
    for f,h in value['files'].items():assert n.u.sha(ROOT/f)==h,f
    return n.u.sha(BASE/'adapter_manifest.json')


def probe():
    import fcntl
    import torch
    from .paper_competition_fastqueue_v1 import NvmlMemory
    from .mainline_service_v1 import generate
    sig=verify();RUN.mkdir(parents=True,exist_ok=True)
    inventory=n.u.load(BASE/'inventory.json')
    candidates=inventory['cached_probe_candidates']
    assert len(candidates)==3
    requests=n.u.load(n.RUN/'requests.json');reuse=n.u.load(n.RUN/'reuse.json')
    tok=n.runtime.tokenizer(CONFIG['reader'])
    pending=[r for k,r in requests.items() if r['reader']==CONFIG['reader'] and k not in reuse]
    max_prompt=max(n.runtime.prompt_tokens(tok,r['messages']) for r in pending)
    assert max(c['prompt_tokens'] for c in candidates)>=max_prompt
    torch.set_num_threads(8);torch.manual_seed(30);torch.cuda.manual_seed_all(30)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    print(json.dumps(dict(status='preloading_probe',requests=len(candidates),pending_max_prompt=max_prompt)),flush=True)
    backend=n.runtime.preload('qa_qwen')
    torch.cuda.set_per_process_memory_fraction(CONFIG['cap_mib']*2**20/torch.cuda.get_device_properties(0).total_memory)
    monitor=NvmlMemory();results=[];stage='admission';started=time.perf_counter()
    try:
        with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            free=monitor.free_mib()
            assert free>=CONFIG['admission_mib'],f'Not enough free memory for bounded probe: {free}'
            claim=torch.empty(int(CONFIG['reserve_mib']*2**20),dtype=torch.uint8,device='cuda');del claim
            stage='loading';backend.model.to('cuda').eval()
            assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in backend.model.parameters())
            torch.cuda.reset_peak_memory_stats()
            loaded=dict(allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20)
            for candidate in candidates:
                key=candidate['key'];stage='request:'+key
                req=requests[key];old=n.u.load(ROOT/reuse[key])['result']
                assert req['reader']==CONFIG['reader'] and n.u.fp([req['reader'],req['messages'],req['max_output']])==key
                torch.cuda.reset_peak_memory_stats()
                result=generate(backend,req['messages'],req['max_output'])
                identical=(result['completion_token_ids']==old['completion_token_ids']
                           and result['text']==old['text'] and result['finish_reason']==old['finish_reason'])
                row=dict(**candidate,identical_to_cache=identical,
                    prompt_tokens_actual=result['usage']['prompt_tokens'],output_tokens=result['usage']['completion_tokens'],
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,free_mib_after=monitor.free_mib())
                n.put(RUN/'probe_outputs'/(key+'.json'),dict(key=key,result=result,resource_test_only=True))
                results.append(row);print(json.dumps(row),flush=True)
                assert identical,'Replay mismatch; preserve experiment and do not change admission'
            n.put(RUN/'probe.json',dict(adapter_signature=sig,passed=True,loaded=loaded,
                free_mib_before=free,results=results,all_weights_cuda=True,precision='bfloat16',
                elapsed_seconds=time.perf_counter()-started,science_outputs_written=False))
    except Exception as exc:
        n.put(RUN/'probe_failure.json',dict(adapter_signature=sig,stage=stage,error=repr(exc),results=results,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        raise


def worker(attempt,policy):
    verify()
    assert n.u.load(RUN/'probe.json')['passed']
    for filename in ['input_hashes.json','reuse_hashes.json']:
        for relative,digest in n.u.load(n.RUN/filename).items():assert n.u.sha(ROOT/relative)==digest,relative
    # Only substitute the three resource constants in the frozen, previously used worker.
    # The model loader, inference calls, decoder, request ordering, and output persistence are identical.
    source=inspect.getsource(component.qa_worker)
    old="isq=reader=='Qwen3.5-9B';admission=26000 if isq else 12000;reserve=24000 if isq else 10500;cap=24500 if isq else 11000"
    assert source.count(old)==1
    admission,reserve,cap=(20250,19000,19500) if policy=='compact' else (26000,24000,24500)
    replacement=f"isq=reader=='Qwen3.5-9B';admission={admission} if isq else 12000;reserve={reserve} if isq else 10500;cap={cap} if isq else 11000"
    source=source.replace(old,replacement)
    namespace=dict(component.__dict__)
    namespace.update(RUN=n.RUN,verify=n.verify,state=n.state)
    exec(compile(source,str(Path(__file__))+':resource_constants_only','exec'),namespace)
    n.put(n.RUN/'attempts'/attempt/'resource_policy.json',dict(adapter_signature=verify(),policy=policy,
        admission_mib=admission,reserve_mib=reserve,cap_mib=cap,
        generated_worker_sha256=hashlib.sha256(source.encode()).hexdigest(),science_configuration_changed=False))
    return namespace['qa_worker'](CONFIG['reader'],attempt)


def run():
    import fcntl
    verify();assert n.u.load(RUN/'probe.json')['passed']
    with (n.RUN/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert (n.RUN/'completed_readers/Phi-4-mini-instruct.json').exists()
        policy='compact'
        prior=list((n.RUN/'attempts').glob('*/resource_policy.json'))
        if any((p.parent/'oom.json').exists() for p in prior):policy='original'
        while not (n.RUN/'completed_readers'/(CONFIG['reader']+'.json')).exists():
            attempt=max([int(p.name) for p in (n.RUN/'attempts').iterdir() if p.name.isdigit()]+[0])+1
            name=f'{attempt:06d}';folder=n.RUN/'attempts'/name;folder.mkdir(parents=True)
            with (folder/'worker.log').open('x') as log:
                proc=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.quote_retention_memory_adapter_v1',
                    'worker','--attempt',name,'--policy',policy],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                (n.RUN/'active_worker.json').write_text(json.dumps(dict(pid=proc.pid,reader=CONFIG['reader'],attempt=name,
                    execution='memory_adapter_v1',policy=policy)),encoding='utf-8')
                code=proc.wait()
            if code==75:
                policy='original';continue
            assert code==0,'Resource-adapted worker failed: '+str(folder/'worker.log')
        if not (n.RUN/'analysis.json').exists():n.analyze()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','probe','run','worker'])
    p.add_argument('--attempt');p.add_argument('--policy',choices=['compact','original']);a=p.parse_args()
    if a.action=='worker':raise SystemExit(worker(a.attempt,a.policy))
    globals()[a.action]()
