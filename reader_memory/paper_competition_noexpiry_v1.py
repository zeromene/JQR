"""Resume the authorized full-GPU experiment without admission or wall expiry."""
from datetime import datetime,timezone
import argparse,json,os,subprocess,time,traceback
from . import paper_competition_fullgpu_v1 as previous

p=previous.p;u=p.u;ROOT=p.ROOT;RUN=p.RUN
BASE=ROOT/'research/paper_competition_noexpiry_v1'
EXEC=RUN/'noexpiry_v1'
CONFIG=dict(placement='all_weights_cuda',precision='bfloat16',attention='sdpa',gpu_cap_mib=24500,
    admission_mib=26000,admission_timeout_seconds=None,wall_timeout_seconds=None,
    phases=['propose','itercomp','assemble','qa'],seed=30,repeat_completed_calls=False)


def verify(models=False):
    old=previous.verify(models=models);m=u.load(BASE/'manifest.json')
    assert m['previous_execution_signature']==old and m['config']==CONFIG
    for name,digest in m['files'].items():assert u.sha(ROOT/name)==digest,name
    return u.sha(BASE/'manifest.json')


def admit(lock):
    import fcntl
    while True:
        free=u.free_memory();acquired=False
        if free>=CONFIG['admission_mib']:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
            except BlockingIOError:pass
        if acquired:
            if u.free_memory()>=CONFIG['admission_mib']:return
            fcntl.flock(lock,fcntl.LOCK_UN)
        p.state('waiting_for_memory',phase=os.environ['COMPETITION_PHASE'],placement=CONFIG['placement'],
            free_mib=free,required_mib=CONFIG['admission_mib'],admission_timeout_seconds=None,wall_timeout_seconds=None)
        time.sleep(20)


def worker(phase):
    signature=verify();os.environ['COMPETITION_PHASE']=phase
    if phase=='assemble':p.assemble();return
    with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
        admit(lock)
        torch=p.setup_gpu()
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_cap_mib']*2**20/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter();okay=False
        # Existing resumption logic retains request identities, completed outputs,
        # and the one specifically documented interrupted request. Only its
        # execution signature now points to this user-authorized timing amendment.
        previous.BASE=BASE
        p.qwen=previous.full_backend;p.call=previous.resume_call
        p.state('running',phase=phase,placement=CONFIG['placement'],admission_timeout_seconds=None,wall_timeout_seconds=None)
        try:
            if phase=='propose':previous.resume_propose()
            else:getattr(p,phase)()
            okay=True
        finally:
            import resource
            value=dict(completed=okay,placement=CONFIG['placement'],execution_signature=signature,
                phase_seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,cpu_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                admission_timeout_seconds=None,wall_timeout_seconds=None)
            u.immutable(EXEC/'resources'/(phase+'.json'),value)
            u.immutable(RUN/'resources'/(phase+'.json'),value)


def wait_stage(child,phase):
    # Intentionally no wait(timeout=...) and no wall-clock expiry.
    if child.wait()!=0:raise RuntimeError('Full GPU phase failed: '+phase)


def run():
    import fcntl
    with (EXEC/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p.state('verifying_restart',placement=CONFIG['placement'],admission_timeout_seconds=None,wall_timeout_seconds=None)
        signature=verify(models=True);snapshot=u.load(EXEC/'restart_snapshot.json')
        for name,digest in snapshot['completed_files'].items():assert u.sha(RUN/name)==digest,name
        assert not (RUN/'assembly.json').exists() and not list((RUN/'itercomp').glob('*.json'))
        for phase in CONFIG['phases']:
            with (EXEC/(phase+'.log')).open('x') as log:
                child=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.paper_competition_noexpiry_v1','worker',phase],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                (RUN/'active_worker.json').write_text(json.dumps(dict(phase=phase,pid=child.pid,
                    placement=CONFIG['placement'],execution='noexpiry_v1')))
                wait_stage(child,phase)
        for name,digest in snapshot['completed_files'].items():assert u.sha(RUN/name)==digest,name
        req=u.load(RUN/'requests.json');assert len(list((RUN/'outputs').glob('*.json')))==len(req)
        u.immutable(RUN/'completion.json',dict(status='completed',signature=u.sha(p.BASE/'manifest.json'),
            execution_signature=signature,time=datetime.now(timezone.utc).isoformat(),logical_conditions=1536,
            unique_requests=len(req),references_read=False,admission_timeout_seconds=None,wall_timeout_seconds=None))
        p.state('completed',logical_conditions=1536,placement=CONFIG['placement'])
        from .paper_competition_analysis_v1 import analyze
        analyze()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);parser.add_argument('phase',nargs='?');args=parser.parse_args()
    try:
        if args.mode=='run':run()
        else:worker(args.phase)
    except Exception as error:
        p.state('failed',phase=args.phase,placement=CONFIG['placement'],error=repr(error),traceback=traceback.format_exc(),
            admission_timeout_seconds=None,wall_timeout_seconds=None)
        raise
