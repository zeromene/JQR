"""Persistent full-GPU queue with explicit OOM recovery and exact result reuse."""
from datetime import datetime,timezone
import argparse,json,os,subprocess,time,traceback
from . import paper_competition_noexpiry_v1 as old
p=old.p;u=p.u;previous=old.previous;ROOT=p.ROOT;RUN=p.RUN
BASE=ROOT/'research/paper_competition_queue_v1';EXEC=RUN/'persistent_queue_v1'
CONFIG=dict(placement='all_weights_cuda',precision='bfloat16',attention='sdpa',seed=30,
    admission_mib=26000,gpu_cap_mib=24500,admission_timeout_seconds=None,wall_timeout_seconds=None,
    resource_retry_limit=None,retry_delay_seconds=30,hold_model_between_phases=True)
_backend=None


def verify(models=False):
    parent=old.verify(models=models);m=u.load(BASE/'manifest.json');assert m['config']==CONFIG
    assert m['previous_execution_signature']==parent
    for name,digest in m['files'].items():assert u.sha(ROOT/name)==digest,name
    return u.sha(BASE/'manifest.json')


def attempt_dir():return EXEC/'attempts'/os.environ['COMPETITION_ATTEMPT']


def backend():
    global _backend
    if _backend is None:
        from .backend import TransformersBackend
        p.state('loading_model',placement=CONFIG['placement'],attempt=os.environ['COMPETITION_ATTEMPT'])
        value=TransformersBackend(ROOT/'models/Qwen3.5-9B',device='cuda',max_input_tokens=16384)
        assert all(x.device.type=='cuda' and x.dtype==value.torch.bfloat16 for x in value.model.parameters())
        u.immutable(attempt_dir()/'placement.json',dict(execution_signature=u.sha(BASE/'manifest.json'),
            all_weights_cuda=True,precision='bfloat16',quantization=False,parameters=sum(x.numel() for x in value.model.parameters())))
        _backend=value
    return _backend


def recoverable(exit_code,attempt):
    marker=attempt/'resource_failure.json'
    return exit_code==75 and marker.exists() and u.load(marker)['kind']=='cuda_out_of_memory'


def check_previous_calls():
    # Never retry a request with an unknown outcome. OOM is a specifically
    # recorded failed execution, with its identity and started-file hash retained.
    for directory in sorted((EXEC/'attempts').glob('*')):
        pending=[x for x in (directory/'calls').rglob('*.started.json')
                 if not x.with_name(x.name.replace('.started.json','.json')).exists()]
        if not pending:continue
        marker=u.load(directory/'resource_failure.json')
        assert marker['kind']=='cuda_out_of_memory'
        for path in pending:
            assert marker['uncompleted_calls'][path.relative_to(RUN).as_posix()]==u.sha(path)


def call(model,messages,limit,identity,phase):
    key=u.fp([phase,identity]);canonical=RUN/'calls'/phase/(key+'.json')
    if canonical.exists():return previous.checked_old_call(u.load(canonical),messages,limit,identity)
    # Recover an already written response even if committing its canonical copy
    # was interrupted. This does not invoke the model again.
    for directory in sorted((EXEC/'attempts').glob('*')):
        saved=directory/'calls'/phase/(key+'.json')
        if saved.exists():
            value=u.load(saved);result=previous.checked_old_call(value,messages,limit,identity)
            u.immutable(canonical,value);return result
    original_start=canonical.with_suffix('.started.json')
    if original_start.exists():
        receipt=u.load(RUN/'fullgpu/transition.json');name=original_start.relative_to(RUN).as_posix()
        assert receipt['interrupted_calls'].get(name)==u.sha(original_start)
        original=u.load(original_start)
        assert original['identity']==identity and original['messages']==messages and original['max_output']==limit
    dest=attempt_dir()/'calls'/phase/(key+'.json');started=dest.with_suffix('.started.json')
    assert not dest.exists() and not started.exists()
    prompt=model.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    n=len(model.tokenizer.encode(prompt,add_special_tokens=False))
    assert n<=(p.CONFIG['reader_max_input'] if phase=='qa' else p.CONFIG['proposer_max_input'])
    u.immutable(started,dict(identity=identity,messages=messages,max_output=limit,time=time.time(),
        execution_signature=u.sha(BASE/'manifest.json')))
    from .mainline_service_v1 import generate
    result=generate(model,messages,limit)
    value=dict(signature=u.sha(p.BASE/'manifest.json'),execution_signature=u.sha(BASE/'manifest.json'),
        identity=identity,messages=messages,max_output=limit,result=result,placement=CONFIG['placement'])
    u.immutable(dest,value);u.immutable(canonical,value)
    return result


def itercomp():
    remaining=[x for x in p.sources() if x['dataset']=='musique' and not (RUN/'itercomp'/(x['source']['question_id']+'.json')).exists()]
    if not remaining:return
    encoder=p.it.BGEEncoder(ROOT/'models/bge-m3',device='cpu',batch_size=8);counter=p.hp.CounterTokens()
    total=48-len(remaining)
    for x in remaining:
        s=x['source'];q=s['question_id'];index=[0]
        def judge(messages):
            i=index[0];index[0]+=1
            return call(backend(),messages,512,[q,i],'itercomp')
        def record(step):
            path=RUN/'itercomp_steps'/q/(str(step['step'])+'.json')
            if path.exists():
                before=u.load(path)
                # CPU timing fields may change when reconstructing a partial
                # trajectory; the evidence, queries, scores and answers may not.
                assert {k:v for k,v in before.items() if k!='filtering_seconds'}=={k:v for k,v in step.items() if k!='filtering_seconds'}
            else:u.immutable(path,step)
        row=p.it.compress(s,encoder,judge,record,5)
        row.update(question_id=q,source_hash=u.fp(s),evidence_tokens=counter.count(row['evidence_text']))
        u.immutable(RUN/'itercomp'/(q+'.json'),row)
        total+=1;p.state('running',phase='itercomp',completed=total,expected=48,placement=CONFIG['placement'])


def qa():
    requests=u.load(RUN/'requests.json');replay=u.load(p.BASE/'public/replay.json')
    for i,(key,req) in enumerate(sorted(requests.items()),1):
        path=RUN/'outputs'/(key+'.json')
        if path.exists():
            value=u.load(path);assert value['key']==key and value['signature']==u.sha(p.BASE/'manifest.json')
            continue
        if key in replay:
            value=replay[key];assert value['request_identity']==u.fp([req['reader'],req['messages'],req['max_output']])
            result=value['result'];origin='exact_historical_replay'
        else:result=call(backend(),req['messages'],req['max_output'],key,'qa');origin='new_inference'
        u.immutable(path,dict(signature=u.sha(p.BASE/'manifest.json'),key=key,origin=origin,result=result))
        if i%20==0 or i==len(requests):p.state('running',phase='qa',completed=i,expected=len(requests),placement=CONFIG['placement'])


def worker():
    import torch,fcntl
    verify();check_previous_calls();directory=attempt_dir();started=time.perf_counter();okay=False
    with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
        os.environ['COMPETITION_PHASE']='propose'
        old.admit(lock)  # Unlimited admission wait; original 26,000 MiB threshold.
        torch=p.setup_gpu()
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_cap_mib']*2**20/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        try:
            model=backend()  # Load before reporting any new inference progress.
            previous.full_backend=backend;previous.resume_call=call;p.qwen=backend;p.call=call
            for phase,fn in [('propose',previous.resume_propose),('itercomp',itercomp),('assemble',p.assemble),('qa',qa)]:
                os.environ['COMPETITION_PHASE']=phase
                if phase=='assemble' and (RUN/'assembly.json').exists():continue
                p.state('running',phase=phase,placement=CONFIG['placement'],attempt=os.environ['COMPETITION_ATTEMPT'])
                begin=time.perf_counter();fn()
                u.immutable(directory/('completed_'+phase+'.json'),dict(phase=phase,seconds=time.perf_counter()-begin))
            okay=True
        except torch.OutOfMemoryError as exc:
            pending={x.relative_to(RUN).as_posix():u.sha(x) for x in (directory/'calls').rglob('*.started.json')
                     if not x.with_name(x.name.replace('.started.json','.json')).exists()}
            u.immutable(directory/'resource_failure.json',dict(kind='cuda_out_of_memory',error=str(exc),
                phase=os.environ['COMPETITION_PHASE'],time=time.time(),uncompleted_calls=pending,
                traceback=traceback.format_exc(),will_requeue=True))
            p.state('requeuing_after_memory_failure',phase=os.environ['COMPETITION_PHASE'],attempt=os.environ['COMPETITION_ATTEMPT'],
                expires_at=None,resource_retry_limit=None)
            return 75
        finally:
            import resource
            value=dict(completed=okay,execution_signature=u.sha(BASE/'manifest.json'),attempt=os.environ['COMPETITION_ATTEMPT'],
                placement=CONFIG['placement'],elapsed_including_wait_seconds=time.perf_counter()-started,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                cpu_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            u.immutable(directory/'resources.json',value)
            u.immutable(RUN/'resources'/('persistent_queue_'+os.environ['COMPETITION_ATTEMPT']+'.json'),value)
    return 0


def run():
    import fcntl
    with (EXEC/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p.state('verifying_restart',execution='persistent_queue_v1',expires_at=None)
        signature=verify(models=True);snapshot=u.load(EXEC/'restart_snapshot.json')
        for name,digest in snapshot['completed_files'].items():assert u.sha(RUN/name)==digest,name
        number=0
        while True:
            number+=1;name=f'{number:06d}';directory=EXEC/'attempts'/name;directory.mkdir(parents=True)
            env=dict(os.environ,COMPETITION_ATTEMPT=name)
            with (directory/'worker.log').open('x') as log:
                child=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.paper_competition_queue_v1','worker'],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                (RUN/'active_worker.json').write_text(json.dumps(dict(pid=child.pid,phase='queued',attempt=name,execution='persistent_queue_v1')))
                code=child.wait()  # No expiry, regardless of queue or inference time.
            if code==0:break
            if recoverable(code,directory):
                p.state('waiting_to_requeue',attempt=name,next_attempt=number+1,expires_at=None)
                time.sleep(CONFIG['retry_delay_seconds']);continue
            raise RuntimeError('Non-resource failure; inspect '+str(directory/'worker.log'))
        for name,digest in snapshot['completed_files'].items():assert u.sha(RUN/name)==digest,name
        req=u.load(RUN/'requests.json');assert len(list((RUN/'outputs').glob('*.json')))==len(req)
        u.immutable(RUN/'completion.json',dict(status='completed',signature=u.sha(p.BASE/'manifest.json'),execution_signature=signature,
            time=datetime.now(timezone.utc).isoformat(),logical_conditions=1536,unique_requests=len(req),references_read=False))
        p.state('completed',logical_conditions=1536)
        from .paper_competition_analysis_v1 import analyze
        analyze()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);args=parser.parse_args()
    try:
        if args.mode=='run':run()
        else:raise SystemExit(worker())
    except Exception as exc:
        p.state('failed',execution='persistent_queue_v1',error=repr(exc),traceback=traceback.format_exc());raise
