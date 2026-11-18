"""Preload CPU weights and atomically claim needed CUDA allocator capacity."""
from datetime import datetime,timezone
import argparse,ctypes,json,os,subprocess,threading,time,traceback
from . import paper_competition_queue_v1 as q
p=q.p;u=q.u;ROOT=p.ROOT;RUN=p.RUN
PARENT_BASE=ROOT/'research/paper_competition_queue_v1'
BASE=ROOT/'research/paper_competition_fastqueue_v1';EXEC=RUN/'fast_queue_v1'
CONFIG=dict(placement='all_weights_cuda',precision='bfloat16',attention='sdpa',seed=30,
    poll_seconds=.1,retry_delay_seconds=1,admission_mib=26000,reserve_mib=24000,gpu_cap_mib=24500,
    admission_timeout_seconds=None,wall_timeout_seconds=None,resource_retry_limit=None,
    cpu_preload=True,hold_model_between_phases=True)
_backend=None


def verify(models=False):
    parent=u.load(PARENT_BASE/'manifest.json')
    assert parent['previous_execution_signature']==q.old.verify(models=models)
    for name,digest in parent['files'].items():assert u.sha(ROOT/name)==digest,name
    m=u.load(BASE/'manifest.json');assert m['config']==CONFIG
    assert m['previous_execution_signature']==u.sha(PARENT_BASE/'manifest.json')
    for name,digest in m['files'].items():assert u.sha(ROOT/name)==digest,name
    return u.sha(BASE/'manifest.json')


class NvmlMemory:
    class Info(ctypes.Structure):
        _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    def __init__(self):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1');assert self.lib.nvmlInit_v2()==0
        self.handle=ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))==0
    def free_mib(self):
        value=self.Info();assert self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(value))==0
        return value.free/2**20


def preload():
    import torch
    from transformers import AutoTokenizer,AutoModelForImageTextToText
    from .backend import TransformersBackend
    path=ROOT/'models/Qwen3.5-9B'
    model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    assert all(x.device.type=='cpu' and x.dtype==torch.bfloat16 for x in model.parameters())
    value=TransformersBackend.__new__(TransformersBackend)
    value.model=model;value.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    value.path=str(path.resolve());value.model_name=path.name;value.torch=torch
    value.device='cuda';value.max_input_tokens=16384;value.lock=threading.Lock()
    return value


def reserve_when_free(torch,monitor,lock):
    import fcntl
    last_log=-float('inf');checks=0;ooms=0
    while True:
        free=monitor.free_mib();checks+=1;acquired=False
        if free>=CONFIG['admission_mib']:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
            except BlockingIOError:pass
        if acquired:
            try:
                if monitor.free_mib()>=CONFIG['admission_mib']:
                    amount=max(0,CONFIG['reserve_mib']*2**20-torch.cuda.memory_reserved())
                    allocation=torch.empty(amount,dtype=torch.uint8,device='cuda')
                    del allocation  # Keep its memory in this process's caching allocator.
                    reserved=torch.cuda.memory_reserved()
                    assert reserved>=CONFIG['reserve_mib']*2**20
                    return dict(checks=checks,reservation_ooms=ooms,free_mib_before=free,reserved_mib=reserved/2**20)
            except torch.OutOfMemoryError:
                ooms+=1;torch.cuda.empty_cache()
            except BaseException:
                fcntl.flock(lock,fcntl.LOCK_UN);raise
            fcntl.flock(lock,fcntl.LOCK_UN)
        now=time.monotonic()
        if now-last_log>=20:
            p.state('waiting_for_memory',execution='fast_queue_v1',phase='propose',cpu_weights_ready=True,
                free_mib=round(free),required_mib=CONFIG['admission_mib'],poll_seconds=CONFIG['poll_seconds'],
                checks=checks,reservation_ooms=ooms,expires_at=None)
            last_log=now
        time.sleep(CONFIG['poll_seconds'])


def backend():
    assert _backend is not None
    return _backend


def worker():
    global _backend
    import torch
    verify();q.BASE=BASE;q.EXEC=EXEC;q.check_previous_calls()
    directory=q.attempt_dir();started=time.perf_counter();okay=False
    p.state('preloading_cpu_weights',execution='fast_queue_v1',attempt=os.environ['COMPETITION_ATTEMPT'],expires_at=None)
    value=preload()
    try:
        torch=p.setup_gpu()
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_cap_mib']*2**20/torch.cuda.get_device_properties(0).total_memory)
        # Initialize the CUDA context before the free-memory window opens.
        warm=torch.empty(1,dtype=torch.uint8,device='cuda');del warm
        torch.cuda.reset_peak_memory_stats();monitor=NvmlMemory()
    except torch.OutOfMemoryError as exc:
        u.immutable(directory/'resource_failure.json',dict(kind='cuda_out_of_memory',error=str(exc),time=time.time(),
            phase='cuda_initialization',uncompleted_calls={},traceback=traceback.format_exc()))
        return 75
    with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
        try:
            receipt=reserve_when_free(torch,monitor,lock)
            u.immutable(directory/'reservation.json',dict(receipt,time=time.time(),execution_signature=u.sha(BASE/'manifest.json')))
            p.state('moving_preloaded_weights_to_gpu',execution='fast_queue_v1',**receipt)
            value.model=value.model.to('cuda').eval()
            assert all(x.device.type=='cuda' and x.dtype==torch.bfloat16 for x in value.model.parameters())
            _backend=value
            u.immutable(directory/'placement.json',dict(all_weights_cuda=True,precision='bfloat16',quantization=False,
                execution_signature=u.sha(BASE/'manifest.json'),reserved_mib=torch.cuda.memory_reserved()/2**20,
                allocated_mib=torch.cuda.memory_allocated()/2**20))
            q.backend=backend;q.previous.full_backend=backend;q.previous.resume_call=q.call;p.qwen=backend;p.call=q.call
            for phase,fn in [('propose',q.previous.resume_propose),('itercomp',q.itercomp),('assemble',p.assemble),('qa',q.qa)]:
                os.environ['COMPETITION_PHASE']=phase
                if phase=='assemble' and (RUN/'assembly.json').exists():continue
                p.state('running',phase=phase,execution='fast_queue_v1',placement=CONFIG['placement'])
                begin=time.perf_counter();fn()
                u.immutable(directory/('completed_'+phase+'.json'),dict(phase=phase,seconds=time.perf_counter()-begin))
            okay=True
        except torch.OutOfMemoryError as exc:
            pending={x.relative_to(RUN).as_posix():u.sha(x) for x in (directory/'calls').rglob('*.started.json')
                     if not x.with_name(x.name.replace('.started.json','.json')).exists()}
            u.immutable(directory/'resource_failure.json',dict(kind='cuda_out_of_memory',error=str(exc),time=time.time(),
                phase=os.environ.get('COMPETITION_PHASE','loading'),uncompleted_calls=pending,traceback=traceback.format_exc()))
            p.state('requeuing_after_memory_failure',execution='fast_queue_v1',expires_at=None)
            return 75
        finally:
            import resource
            record=dict(completed=okay,execution_signature=u.sha(BASE/'manifest.json'),placement=CONFIG['placement'],
                elapsed_including_wait_seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,cpu_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            u.immutable(directory/'resources.json',record)
            u.immutable(RUN/'resources'/('fast_queue_'+os.environ['COMPETITION_ATTEMPT']+'.json'),record)
    return 0


def run():
    import fcntl
    with (EXEC/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signature=verify(models=True);snapshot=u.load(EXEC/'restart_snapshot.json')
        for name,digest in snapshot['completed_files'].items():assert u.sha(RUN/name)==digest,name
        number=0
        while True:
            number+=1;name=f'{number:06d}';directory=EXEC/'attempts'/name;directory.mkdir(parents=True)
            with (directory/'worker.log').open('x') as log:
                child=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.paper_competition_fastqueue_v1','worker'],
                    cwd=ROOT,env=dict(os.environ,COMPETITION_ATTEMPT=name),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                (RUN/'active_worker.json').write_text(json.dumps(dict(pid=child.pid,phase='queued',attempt=name,execution='fast_queue_v1')))
                code=child.wait()
            if code==0:break
            if q.recoverable(code,directory):
                p.state('waiting_to_requeue',execution='fast_queue_v1',next_attempt=number+1,expires_at=None)
                time.sleep(CONFIG['retry_delay_seconds']);continue
            raise RuntimeError('Non-resource failure; inspect '+str(directory/'worker.log'))
        for name,digest in snapshot['completed_files'].items():assert u.sha(RUN/name)==digest,name
        req=u.load(RUN/'requests.json');assert len(list((RUN/'outputs').glob('*.json')))==len(req)
        u.immutable(RUN/'completion.json',dict(status='completed',signature=u.sha(p.BASE/'manifest.json'),execution_signature=signature,
            time=datetime.now(timezone.utc).isoformat(),logical_conditions=1536,unique_requests=len(req),references_read=False))
        from .paper_competition_analysis_v1 import analyze
        analyze()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);args=parser.parse_args()
    try:
        if args.mode=='run':run()
        else:raise SystemExit(worker())
    except Exception as exc:p.state('failed',execution='fast_queue_v1',error=repr(exc),traceback=traceback.format_exc());raise
