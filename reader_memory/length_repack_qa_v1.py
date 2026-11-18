"""Frozen-reader QA only; all evidence choices are prepared before this worker."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research/length_repack_v1'
RUN=ROOT/'runs/length_repack_v1'
CONFIG=dict(readers=['Qwen3.5-4B','Qwen3.5-9B'],max_input=4096,max_output=128,seed=30,
    dtype='bfloat16',attention='sdpa',thinking=False,do_sample=False,
    caps_mib={'Qwen3.5-4B':10500,'Qwen3.5-9B':19500},headroom_mib=1000,
    pre_cuda_margin_mib=512,wait_seconds=7200)


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()


def fp(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def immutable(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def state(status,**extra):
    value=dict(status=status,time=datetime.now(timezone.utc).isoformat(),**extra)
    temp=RUN/'status.tmp';temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temp,RUN/'status.json')
    print(json.dumps(value),flush=True)


def verify(models=False):
    m=load(BASE/'manifest.json')
    assert m['config']==CONFIG
    for name,digest in m['inference_files'].items():assert sha(ROOT/name)==digest,name
    if models:
        for name,digest in m['model_files'].items():assert sha(ROOT/name)==digest,name
    return sha(BASE/'manifest.json')


def validate_request(key,r):
    assert set(r)=={'key','reader','messages','max_output','evidence_tokens'}
    assert r['key']==key==fp([r['reader'],r['messages'],r['max_output']])
    assert r['reader'] in CONFIG['readers'] and r['max_output']==128
    assert [m['role'] for m in r['messages']]==['system','user']


def preflight():
    from transformers import AutoTokenizer
    sid=verify(models=True)
    requests=load(BASE/'public/requests.json');replay=load(BASE/'public/replay.json')
    assert set(replay)<=set(requests)
    lengths={}
    for reader in CONFIG['readers']:
        tok=AutoTokenizer.from_pretrained(str(ROOT/'models'/reader),local_files_only=True,trust_remote_code=False)
        reader_lengths=[]
        for key,r in requests.items():
            validate_request(key,r)
            if r['reader']!=reader:continue
            prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            n=len(tok.encode(prompt,add_special_tokens=False));assert n<=4096
            # Fixed schema allows unambiguous extraction of the rendered evidence.
            user=r['messages'][1]['content'];assert user.startswith('Evidence:\n')
            evidence=user[len('Evidence:\n'):].rsplit('\n\nQuestion: ',1)[0]
            assert len(tok.encode(evidence,add_special_tokens=False))==r['evidence_tokens']
            if key in replay:assert replay[key]['result']['usage']['prompt_tokens']==n
            reader_lengths.append(n)
        lengths[reader]=dict(requests=len(reader_lengths),max_input=max(reader_lengths))
    immutable(RUN/'preflight.json',dict(signature=sid,lengths=lengths,new_requests=len(requests)-len(replay),
        private_references_read=False,versions={n:md.version(n) for n in ['torch','transformers','tokenizers']}))


def free_memory():
    return int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())


def worker(reader):
    import fcntl
    sid=verify();requests=load(BASE/'public/requests.json');replay=load(BASE/'public/replay.json')
    keys=sorted(k for k,r in requests.items() if r['reader']==reader)
    pending=[]
    for key in keys:
        path=RUN/'outputs'/reader/(key+'.json');assert not path.exists()
        if key in replay:
            immutable(path,dict(signature=sid,key=key,reader=reader,origin='exact_historical_replay',
                replay_origin=replay[key]['origin'],result=replay[key]['result']))
        else:pending.append(key)
    assert not (RUN/'resources'/(reader+'.json')).exists()
    if not pending:
        immutable(RUN/'resources'/(reader+'.json'),dict(status='completed',reader=reader,new_calls=0,replayed=len(keys)))
        return
    cap=CONFIG['caps_mib'][reader];required=cap+CONFIG['headroom_mib']+CONFIG['pre_cuda_margin_mib']
    deadline=time.monotonic()+CONFIG['wait_seconds']
    with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
        while True:
            acquired=False;free=free_memory()
            if free>=required:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                except BlockingIOError:pass
            if acquired:
                free=free_memory()
                if free>=required:break
                fcntl.flock(lock,fcntl.LOCK_UN)
            state('waiting_for_memory',reader=reader,free_mib=free,required_mib=required,
                  pending_new_requests=len(pending),replayed_requests=len(keys)-len(pending),cuda_initialized=False)
            if time.monotonic()>=deadline:raise TimeoutError('Two-hour admission limit')
            time.sleep(20)
        import torch
        from .backend import TransformersBackend
        from .mainline_service_v1 import generate
        torch.set_num_threads(8);torch.manual_seed(30);torch.cuda.manual_seed_all(30)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(cap*2**20/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        state('loading',reader=reader,free_mib_at_admission=free,new_requests=len(pending))
        start=time.perf_counter();status='failed';finished=0
        try:
            backend=TransformersBackend(ROOT/'models'/reader,max_input_tokens=4096)
            for key in pending:
                request=requests[key]
                started=RUN/'started'/reader/(key+'.json')
                assert not started.exists(),'Unknown request retry forbidden'
                immutable(started,dict(signature=sid,key=key,reader=reader,time=datetime.now(timezone.utc).isoformat()))
                result=generate(backend,request['messages'],128)
                assert result['usage']['prompt_tokens']<=4096 and 0<result['usage']['completion_tokens']<=128
                immutable(RUN/'outputs'/reader/(key+'.json'),dict(signature=sid,key=key,reader=reader,
                    origin='new_inference',result=result))
                finished+=1
                if finished%20==0 or finished==len(pending):
                    state('running',reader=reader,new_completed=finished,new_expected=len(pending),
                          seconds=time.perf_counter()-start)
            status='completed'
        finally:
            immutable(RUN/'resources'/(reader+'.json'),dict(status=status,reader=reader,new_calls=finished,
                replayed=len(keys)-len(pending),phase_seconds=time.perf_counter()-start,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,free_mib_at_admission=free,
                gpu=torch.cuda.get_device_name(0)))
            fcntl.flock(lock,fcntl.LOCK_UN)


def run():
    RUN.mkdir(parents=True,exist_ok=True)
    assert not list((RUN/'outputs').rglob('*.json')),'Run already started'
    state('preflight');preflight()
    for reader in CONFIG['readers']:
        state('starting_reader',reader=reader)
        with (RUN/(reader+'.log')).open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','reader_memory.length_repack_qa_v1','worker',reader],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active=dict(reader=reader,pid=child.pid)
            (RUN/'active_worker.json').write_text(json.dumps(active),encoding='utf-8')
            rc=child.wait()
        if rc:
            raise RuntimeError(f'{reader} worker exited with code {rc}; inspect its log')
    requests=load(BASE/'public/requests.json')
    outputs={}
    for key,r in requests.items():
        v=load(RUN/'outputs'/r['reader']/(key+'.json'))
        assert v['key']==key and v['reader']==r['reader']
        outputs[key]=v
    immutable(RUN/'completion.json',dict(status='completed',time=datetime.now(timezone.utc).isoformat(),
        signature=sha(BASE/'manifest.json'),unique_requests=len(outputs),
        new_calls=sum(v['origin']=='new_inference' for v in outputs.values()),
        replayed=sum(v['origin']=='exact_historical_replay' for v in outputs.values()),private_references_read=False))
    state('completed',unique_requests=len(outputs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('reader',nargs='?')
    a=p.parse_args()
    try:
        if a.mode=='run':run()
        else:worker(a.reader)
    except Exception as exc:
        RUN.mkdir(parents=True,exist_ok=True)
        state('failed',reader=a.reader,error=repr(exc),traceback=traceback.format_exc());raise
