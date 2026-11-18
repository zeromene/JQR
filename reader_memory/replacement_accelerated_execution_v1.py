"""Two independent serial readers; frozen training, selection, and final serial QA unchanged."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
import os
from pathlib import Path
import time

from . import replacement_learning_v1 as frozen
from . import baseline_bootstrap_v2 as base
from .backend import TransformersBackend
from .mainline_service_v1 import generate
from .schema import fingerprint,read_jsonl,write_json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/replacement_learning_v1_accel1'


def key(row):return row['question_id'],row.get('arm',row.get('variant'))


def verify_execution():
    meta=frozen.load(RUN/'execution_manifest.json')
    if fingerprint({k:v for k,v in meta.items() if k!='signature'})!=meta['signature']:
        raise ValueError('Execution manifest signature mismatch')
    if meta['scientific_signature']!=frozen.verify():raise ValueError('Scientific identity changed')
    for path,digest in meta['files'].items():
        if base.sha(ROOT/path)!=digest:raise ValueError('Changed execution input: '+path)
    return meta


def ledger(folder,plans,sid):
    predictions=list(read_jsonl(folder/'predictions.jsonl')) if (folder/'predictions.jsonl').exists() else []
    calls=list(read_jsonl(folder/'answer_calls.jsonl')) if (folder/'answer_calls.jsonl').exists() else []
    done={key(p):p for p in predictions};expected={key(p):p for p in plans}
    if len(done)!=len(predictions) or len(predictions)!=len(calls):raise ValueError('Invalid committed ledger')
    for pred,call in zip(predictions,calls,strict=True):
        if key(pred)!=key(call) or key(pred) not in expected:raise ValueError('Unexpected committed key')
        request=fingerprint(expected[key(pred)])
        if any(r['signature']!=sid or r['request_sha256']!=request for r in [pred,call]):
            raise ValueError('Committed request identity changed')
    return done


BACKEND=None


def worker_initialize():
    global BACKEND
    BACKEND=TransformersBackend(ROOT/'models/Qwen3.5-9B',max_input_tokens=32768)
    print('SERIAL_REPLICA_READY',os.getpid(),flush=True)


def worker_generate(plan,signature,raw_path):
    tokens=len(BACKEND.tokenizer.encode(plan['evidence_text'],add_special_tokens=False))
    if tokens!=plan['evidence_tokens'] or tokens>2048:raise ValueError('Frozen reader text/token mismatch')
    messages=[dict(role='system',content=frozen.SYSTEM),dict(role='user',
        content=frozen.evidence.PROMPT(None,plan['question'],plan['evidence_text']))]
    started=time.perf_counter();fatal=None
    try:result=generate(BACKEND,messages,2048)
    except Exception as exc:fatal=f'{type(exc).__name__}: {exc}';result={}
    payload=dict(signature=signature,row=plan,result=result,result_sha256=fingerprint(result),
                 seconds=time.perf_counter()-started,worker_pid=os.getpid(),fatal=fatal)
    write_json(raw_path,payload)
    return raw_path


def commit_saved_result(folder,payload,done,sid,total):
    if payload['signature']!=sid or fingerprint(payload['result'])!=payload['result_sha256']:
        raise ValueError('Saved result scientific identity or hash mismatch')
    plan,result=payload['row'],payload['result']
    if key(plan) in done:return
    answer,status,error='','failed',None
    try:answer=frozen.parse_answer(base.require_complete(result));status='ok'
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    messages=[dict(role='system',content=frozen.SYSTEM),dict(role='user',
        content=frozen.evidence.PROMPT(None,plan['question'],plan['evidence_text']))]
    digest=fingerprint(plan)
    execution=dict(worker_pid=payload['worker_pid'],worker_count=2,generation_batch_size=1,
                   raw_result_sha256=payload['result_sha256'])
    base.append(folder/'answer_calls.jsonl',dict(question_id=plan['question_id'],variant=plan['variant'],
        signature=sid,request_sha256=digest,messages=messages,max_tokens=2048,result=result,
        status=status,error=error,seconds=payload['seconds'],execution_parallel=execution))
    pred=dict(**plan,signature=sid,request_sha256=digest,hypothesis=answer,status=status,error=error,
              execution_parallel=execution)
    base.append(folder/'predictions.jsonl',pred);done[key(plan)]=pred
    write_json(folder/'status.json',dict(status='running',completed=len(done),expected=total,
               workers=2,generation_batch_size=1,phase='rewards'))
    print('PARALLEL_SERIAL_REWARD',len(done),'/',total,*key(plan),status,flush=True)


def rewards():
    meta=verify_execution();sid=meta['scientific_signature']
    if meta['reader_workers']!=2 or meta['generation_batch_size']!=1:
        raise ValueError('Unexpected reader replica configuration')
    plans=frozen.load(frozen.PLAN/'public/reward_inputs.json');folder=RUN/'rewards'
    folder.mkdir(parents=True,exist_ok=True);done=ledger(folder,plans,sid)
    if (folder/'inflight.json').exists():raise ValueError('Unexpected serial inflight request')
    for intent in sorted((folder/'intents').glob('*.json')):
        pending=frozen.load(intent);raw=folder/'raw_results'/(pending['request_sha256']+'.json')
        if not raw.exists():raise ValueError('Uncommitted worker result; never silently regenerate')
        payload=frozen.load(raw)
        if payload['row']!=pending['row']:raise ValueError('Saved request/result mismatch')
        commit_saved_result(folder,payload,done,sid,len(plans));intent.unlink()
    remaining=iter(p for p in plans if key(p) not in done)
    started=time.perf_counter();initial=len(done)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_initialize) as executor:
        validation=RUN/'replica_validation.json'
        if not validation.exists():
            marker=RUN/'replica_validation_started.json'
            if marker.exists():raise ValueError('Replica validation interrupted; inspect saved diagnostics')
            seeds=frozen.load(ROOT/'research/reader_acceleration_20261028/probe_started.json')['prompts']
            by_hash={fingerprint(p):p for p in plans}
            write_json(marker,dict(requests=[p['request_sha256'] for p in seeds],inference_calls=4,
                                  reward_updates=0))
            checks=[];validation_started=time.perf_counter()
            for seed in seeds:
                row=by_hash[seed['request_sha256']]
                raw=RUN/'diagnostics'/(seed['request_sha256']+'.json')
                checks.append((seed,executor.submit(worker_generate,row,sid,str(raw))))
            results=[]
            for seed,future in checks:
                payload=frozen.load(Path(future.result()))
                results.append(dict(request_sha256=seed['request_sha256'],worker_pid=payload['worker_pid'],
                    exact_token_match=payload['result'].get('completion_token_ids')==seed['original_tokens'],
                    seconds=payload['seconds'],fatal=payload['fatal']))
            value=dict(results=results,seconds=time.perf_counter()-validation_started,
                       generation_batch_size=1,reward_updates=0,diagnostic_inference_calls=4,
                       all_exact_token_matches=all(r['exact_token_match'] for r in results))
            write_json(validation,value)
            if not value['all_exact_token_matches']:raise ValueError('Serial replica changed diagnostic output')
            print('SERIAL_REPLICA_VALIDATION',json.dumps(value),flush=True)
        pending={}
        def submit():
            row=next(remaining,None)
            if row is None:return False
            digest=fingerprint(row);intent=folder/'intents'/(digest+'.json')
            raw=folder/'raw_results'/(digest+'.json')
            if raw.exists():raise ValueError('Raw result exists without handled intent')
            write_json(intent,dict(signature=sid,request_sha256=digest,row=row))
            future=executor.submit(worker_generate,row,sid,str(raw))
            pending[future]=(row,intent)
            return True
        submit();submit()
        while pending:
            completed,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                row,intent=pending.pop(future)
                payload=frozen.load(Path(future.result()))
                if payload['row']!=row:raise ValueError('Worker returned another request')
                commit_saved_result(folder,payload,done,sid,len(plans));intent.unlink()
                if payload['fatal']:raise RuntimeError('Worker failure retained without retry: '+payload['fatal'])
                elapsed=time.perf_counter()-started
                write_json(folder/'parallel_progress.json',dict(initial=initial,completed_new=len(done)-initial,
                    completed=len(done),expected=len(plans),seconds=elapsed,workers=2,
                    requests_per_second=(len(done)-initial)/elapsed))
                submit()
    if set(done)!={key(p) for p in plans}:raise ValueError('Incomplete reward coverage')
    write_json(folder/'status.json',dict(status='completed',completed=len(done),expected=len(plans),
                                       workers=2,generation_batch_size=1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','rewards','score_rewards','train','select','qa','score'])
    parser.add_argument('--endpoint',default='http://127.0.0.1:8018')
    args=parser.parse_args();verify_execution();frozen.RUN=RUN
    assert frozen.parse_answer('{"answer":"Berlin"}')=='Berlin'
    assert base.native_f1('Berlin','Berlin',2)==1.
    if args.mode=='rewards':rewards()
    elif args.mode=='qa':frozen.qa(args.endpoint)
    else:getattr(frozen,args.mode)()


if __name__=='__main__':main()
