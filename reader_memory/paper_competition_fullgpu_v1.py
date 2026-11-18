"""User-requested placement change; retain the frozen scientific experiment."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,os,json,time,subprocess,traceback
from . import paper_competition_v1 as p
u=p.u;ROOT=p.ROOT;BASE=ROOT/'research/paper_competition_fullgpu_v1';RUN=p.RUN
CONFIG=dict(placement='all_weights_cuda',precision='bfloat16',attention='sdpa',gpu_cap_mib=24500,
    admission_mib=26000,wait_seconds=7200,phases=['propose','itercomp','assemble','qa'],
    seed=30,alter_scientific_protocol=False,repeat_completed_calls=False)


def verify(models=False):
    sid=p.verify(models=models);m=u.load(BASE/'manifest.json')
    assert m['parent_signature']==sid and m['config']==CONFIG
    for name,digest in m['files'].items():assert u.sha(ROOT/name)==digest,name
    return u.sha(BASE/'manifest.json')


def full_backend():
    from .backend import TransformersBackend
    backend=TransformersBackend(ROOT/'models/Qwen3.5-9B',device='cuda',max_input_tokens=16384)
    assert all(v.device.type=='cuda' and v.dtype==backend.torch.bfloat16 for v in backend.model.parameters())
    path=RUN/'fullgpu'/('placement_'+os.environ['COMPETITION_PHASE']+'.json')
    u.immutable(path,dict(execution_signature=u.sha(BASE/'manifest.json'),all_weights_cuda=True,
        cpu_offload=False,quantization=False,precision='bfloat16',attention='sdpa',parameters=sum(v.numel() for v in backend.model.parameters())))
    return backend


def checked_old_call(row,messages,limit,identity):
    assert row['signature']==u.sha(p.BASE/'manifest.json')
    assert row['identity']==identity and row['messages']==messages and row['max_output']==limit
    return row['result']


def resume_call(backend,messages,limit,identity,phase):
    key=u.fp([phase,identity]);old=RUN/'calls'/phase/(key+'.json');start=old.with_suffix('.started.json')
    if old.exists():return checked_old_call(u.load(old),messages,limit,identity)
    # Only specifically identified requests interrupted by this switch may retry.
    receipt=u.load(RUN/'fullgpu/transition.json')
    if start.exists():
        rel=start.relative_to(RUN).as_posix()
        assert rel in receipt['interrupted_calls'],'Unattributed incomplete request'
        assert u.sha(start)==receipt['interrupted_calls'][rel]
        prior=u.load(start)
        assert prior['identity']==identity and prior['messages']==messages and prior['max_output']==limit
    dest=RUN/'fullgpu/calls'/phase/(key+'.json');begun=dest.with_suffix('.started.json')
    assert not dest.exists() and not begun.exists(),'No blind retry in new execution'
    from .mainline_service_v1 import generate
    prompt=backend.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    n=len(backend.tokenizer.encode(prompt,add_special_tokens=False))
    assert n<=(p.CONFIG['reader_max_input'] if phase=='qa' else p.CONFIG['proposer_max_input'])
    u.immutable(begun,dict(identity=identity,messages=messages,max_output=limit,time=time.time(),
        execution_signature=u.sha(BASE/'manifest.json'),interrupted_predecessor=str(start.relative_to(RUN)) if start.exists() else None))
    result=generate(backend,messages,limit)
    value=dict(signature=u.sha(p.BASE/'manifest.json'),execution_signature=u.sha(BASE/'manifest.json'),
        identity=identity,messages=messages,max_output=limit,result=result,placement='all_weights_cuda')
    u.immutable(dest,value)
    # The frozen assembly/analysis can read its normal path; old results never overwritten.
    u.immutable(old,value)
    return result


def resume_propose():
    backend=None
    for i,x in enumerate([x for x in p.sources() if x['dataset']=='musique'],1):
        s=x['source'];q=s['question_id']
        for kind in ['role','direct']:
            dest=RUN/'proposals'/(u.fp([q,kind])+'.json')
            if dest.exists():
                old=u.load(dest);assert old['source_hash']==u.fp(s) and old['question_id']==q and old['kind']==kind
                continue
            if backend is None:backend=full_backend()
            msg=p.proposal_messages(s,kind);attempts=[]
            for attempt in range(2 if kind=='role' else 1):
                result=resume_call(backend,msg,2048,[q,kind,attempt],'propose')
                parsed=p.hp.parse_result(s,'role',result) if kind=='role' else p.direct.inspect(s,result)
                attempts.append(dict(result=result,messages=msg,parsed=parsed))
                if kind!='role' or parsed['status']!='invalid':break
                correction='Repair your output once. Return the complete valid JSON object only. Validation errors: '+json.dumps(parsed['errors'])+'. Use only exact quotes from the original sources; do not invent facts. If no supported evidence is identifiable, return {"roles":[]}.'
                msg=p.proposal_messages(s,kind)+[dict(role='assistant',content=result['text']),dict(role='user',content=correction)]
            u.immutable(dest,dict(question_id=q,source_hash=u.fp(s),kind=kind,parsed=parsed,attempts=attempts))
        p.state('running',phase='propose',completed=i,expected=48,placement='all_weights_cuda')


def worker(phase):
    import fcntl
    verify();os.environ['COMPETITION_PHASE']=phase
    if phase=='assemble':p.assemble();return
    with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
        deadline=time.monotonic()+CONFIG['wait_seconds']
        while True:
            free=u.free_memory();acquired=False
            if free>=CONFIG['admission_mib']:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                except BlockingIOError:pass
            if acquired:
                if u.free_memory()>=CONFIG['admission_mib']:break
                fcntl.flock(lock,fcntl.LOCK_UN)
            p.state('waiting_for_memory',phase=phase,placement='all_weights_cuda',free_mib=free,required_mib=CONFIG['admission_mib'])
            if time.monotonic()>deadline:raise TimeoutError('Full GPU admission timeout')
            time.sleep(20)
        torch=p.setup_gpu()
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_cap_mib']*2**20/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter();okay=False
        p.qwen=full_backend;p.call=resume_call
        p.state('running',phase=phase,placement='all_weights_cuda')
        try:
            if phase=='propose':resume_propose()
            else:getattr(p,phase)()
            okay=True
        finally:
            import resource
            value=dict(completed=okay,placement='all_weights_cuda',execution_signature=u.sha(BASE/'manifest.json'),
                phase_seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,cpu_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            u.immutable(RUN/'fullgpu/resources'/(phase+'.json'),value)
            u.immutable(RUN/'resources'/(phase+'.json'),value)


def run():
    execution=verify(models=True);transition=u.load(RUN/'fullgpu/transition.json')
    for name,digest in transition['completed_files'].items():assert u.sha(RUN/name)==digest,name
    assert not (RUN/'assembly.json').exists() and not list((RUN/'itercomp').glob('*.json'))
    # Preserve the original 24-hour experiment ceiling; do not reset its budget.
    deadline=u.load(RUN/'submission.json')['time']+p.CONFIG['wall_limit_seconds']
    for phase in CONFIG['phases']:
        with (RUN/'fullgpu'/(phase+'.log')).open('x') as log:
            child=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.paper_competition_fullgpu_v1','worker',phase],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            (RUN/'active_worker.json').write_text(json.dumps(dict(phase=phase,pid=child.pid,placement='all_weights_cuda')))
            try:code=child.wait(timeout=max(1,deadline-time.time()))
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise TimeoutError('Original total runtime ceiling reached')
            if code:raise RuntimeError('Full GPU phase failed: '+phase)
    for name,digest in transition['completed_files'].items():assert u.sha(RUN/name)==digest,'Completed result changed: '+name
    req=u.load(RUN/'requests.json');assert len(list((RUN/'outputs').glob('*.json')))==len(req)
    u.immutable(RUN/'completion.json',dict(status='completed',signature=u.sha(p.BASE/'manifest.json'),
        execution_signature=execution,time=datetime.now(timezone.utc).isoformat(),logical_conditions=1536,
        unique_requests=len(req),references_read=False))
    p.state('completed',logical_conditions=1536,placement='all_weights_cuda')
    from .paper_competition_analysis_v1 import analyze
    analyze()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);parser.add_argument('phase',nargs='?');args=parser.parse_args()
    try:
        if args.mode=='run':run()
        else:worker(args.phase)
    except Exception as error:
        p.state('failed',phase=args.phase,placement='all_weights_cuda',error=repr(error),traceback=traceback.format_exc());raise
