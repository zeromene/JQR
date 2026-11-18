"""Frozen 9B abstractive-summary control on the existing 480 public-dev sources."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import statistics
import subprocess
import time
import traceback
from . import paper_benchmark_extension_v1 as parent

u=parent.u;rt=parent.rt;ROOT=parent.ROOT
BASE=ROOT/'research/paper_same_model_summary_v1'
RUN=ROOT/'runs/paper_same_model_summary_v1'
METHODS=['qwen_summary_cap','qwen_summary_native']
PROMPT=(
    'Write a concise, question-focused summary of the supplied source documents. '
    'Preserve the facts and relationships needed to answer the question, including '
    'relevant intermediate connections between documents. Use only information in '
    'the supplied sources; do not invent facts. You may paraphrase and combine '
    'information across sources. Keep the summary within 512 tokens. '
    'Return only the summary text, without a preamble, JSON, or a separate final answer.'
)
CONFIG=dict(version='paper-same-model-summary-v1',sources=480,reader='Qwen3.5-9B',
    compressor='Qwen3.5-9B',methods=METHODS,budget=512,summary_max_input=16384,
    summary_max_output=2048,reader_max_input=8192,reader_max_output=2048,
    summary_calls_per_source=1,technical_smoke_calls=1,seed=30,bootstrap=10000,
    decoding='Original full-GPU BF16 SDPA greedy, enable_thinking=False; common mainline generate',
    prompt=PROMPT,source_serialization='Exact question/sources JSON user payload of frozen role proposer',
    primary='quote_anchor minus qwen_summary_cap',
    secondary=['quote_anchor minus qwen_summary_native','qwen_summary_cap minus brief_pro_auto_cap',
               'qwen_summary_native minus brief_pro_auto_native'],
    scope='Supplement after seeing prior 480-dev results; no new independent confirmation or isolated mechanism claim',
    training=False,prompt_search=False,adaptive_expansion=False,wait_timeout=None,wall_timeout=None,
    admission_mib=26000,reserve_mib=24000,cap_mib=24500,poll_seconds=.1)

def put(path,value):rt.put(path,value)

def state(status,**fields):
    RUN.mkdir(parents=True,exist_ok=True)
    value=dict(status=status,time=datetime.now(timezone.utc).isoformat(),expires_at=None,**fields)
    tmp=RUN/'status.tmp';tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,RUN/'status.json');print(json.dumps(value,ensure_ascii=False),flush=True)

def summary_messages(source):
    payload=dict(question=source['question'],sources=[dict(id=p['source_id'],title=p['title'],text=p['text']) for p in source['paragraphs']])
    return [dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(payload,ensure_ascii=False,separators=(',',':')))]

def request(messages,limit):
    return dict(key=u.fp([CONFIG['reader'],messages,limit]),reader=CONFIG['reader'],messages=messages,max_output=limit)

def freeze():
    sid=parent.verify();assert u.load(parent.RUN/'analysis.json')['complete']
    files=['reader_memory/paper_same_model_summary_v1.py','research/paper_same_model_summary_v1/PROTOCOL.md',
           'tests/test_paper_same_model_summary_v1.py','reader_memory/mainline_service_v1.py',
           'reader_memory/paper_confirmation_runtime_v1.py','reader_memory/paper_competition_fastqueue_v1.py',
           'experiments/recent_baselines_v1/core.py']
    predecessors=[parent.RUN/n for n in ['analysis.json','plans.json','mapping.json','requests.json','reuse.json','legacy_replay.json']]
    put(BASE/'manifest.json',dict(config=CONFIG,parent_signature=sid,
        files={p:u.sha(ROOT/p) for p in files},inherited_files={str(p.relative_to(ROOT)):u.sha(p) for p in predecessors}))

def verify():
    m=u.load(BASE/'manifest.json');assert m['config']==CONFIG and parent.verify()==m['parent_signature']
    for path,digest in {**m['files'],**m['inherited_files']}.items():assert u.sha(ROOT/path)==digest,path
    return u.sha(BASE/'manifest.json')

def result_path(phase,key):return RUN/'calls'/phase/(key+'.json')

def durable_call(phase,req,backend,attempt):
    from .mainline_service_v1 import generate
    key=req['key'];dest=result_path(phase,key)
    if dest.exists():
        previous=u.load(dest);assert previous['request']==req
        return previous['result']
    # Known CUDA OOM is the only permitted retry of a started, unsaved call.
    for started in (RUN/'attempts').glob('*/calls/'+key+'.json'):
        failure=started.parent.parent/'oom.json'
        assert failure.exists() and u.load(failure)['active_key']==key,'Unknown interrupted model call'
    os.environ['SUMMARY_ACTIVE_KEY']=key
    put(RUN/'attempts'/attempt/'calls'/(key+'.json'),dict(key=key,phase=phase,started=time.time()))
    result=generate(backend,req['messages'],req['max_output'])
    assert result['usage']['prompt_tokens']==rt.prompt_tokens(backend.tokenizer,req['messages'])
    assert result['usage']['completion_tokens']==len(result['completion_token_ids'])
    put(dest,dict(signature=u.sha(BASE/'manifest.json'),request=req,result=result))
    os.environ.pop('SUMMARY_ACTIVE_KEY',None)
    return result

def cache_candidates():
    cache=u.load(parent.RUN/'legacy_replay.json')
    for key,row in u.load(parent.RUN/'reuse.json').items():
        if key in cache:assert cache[key]['result']['completion_token_ids']==row['result']['completion_token_ids']
        cache[key]=row
    for key,req in u.load(parent.RUN/'requests.json').items():
        path=parent.RUN/'outputs'/(key+'.json')
        if not path.exists():continue
        row=u.load(path);assert row['key']==key and row['signature']==u.sha(parent.BASE/'manifest.json')
        if key in cache:assert cache[key]['result']['completion_token_ids']==row['result']['completion_token_ids']
        cache[key]=dict(result=row['result'],messages=req['messages'],max_output=req['max_output'],origin=str(path.relative_to(ROOT)))
    return cache

def exact_reuse(req,cache):
    row=cache.get(req['key'])
    if row is None:return None
    assert row['messages']==req['messages'] and row['max_output']==req['max_output']
    return row

def prepare():
    sid=verify();tok=rt.tokenizer(CONFIG['reader']);ss=parent.sources();assert len(ss)==480
    requests={};mapping={};lengths=[]
    for s in ss:
        messages=summary_messages(s)
        assert messages[1]==parent.screen.proposal_messages(s,'role')[1],'Source serialization differs'
        req=request(messages,2048);n=rt.prompt_tokens(tok,messages)
        assert n<=16384,'Summary input too long; no clipping'
        requests[req['key']]=req;mapping[s['question_id']]=req['key'];lengths.append(n)
    assert len(mapping)==len(requests)==480
    cache=cache_candidates();reuse={}
    for key,req in requests.items():
        row=exact_reuse(req,cache)
        if row is not None:reuse[key]=row
    for name,value in [('summary_requests',requests),('summary_mapping',mapping),('summary_reuse',reuse)]:put(RUN/(name+'.json'),value)
    put(RUN/'preparation.json',dict(signature=sid,sources=480,unique_summary_requests=len(requests),exact_summary_reuse=len(reuse),
        new_summaries=len(requests)-len(reuse),max_summary_prompt_tokens=max(lengths),mean_summary_prompt_tokens=statistics.mean(lengths),
        available_cache_requests=len(cache),parent_all_8160_conditions_reused=True,new_reader_logical_conditions_max=960,
        new_retrieval_calls=0,new_reranking_calls=0,references_read=False,source_serialization_matches_all_role_inputs=True))
    state('prepared',summaries_to_generate=len(requests)-len(reuse),sources=480)

def compress(backend,attempt):
    fixture=dict(question='Which city is the museum in?',paragraphs=[dict(source_id='0',title='Museum',text='The museum is in Brookfield.')])
    smoke=RUN/'technical_smoke.json'
    if not smoke.exists():
        req=request(summary_messages(fixture),2048);result=durable_call('smoke',req,backend,attempt)
        assert result['text'].strip() and result['finish_reason']=='stop','Summary interface smoke failed'
        put(smoke,dict(passed=True,synthetic_only=True,result=result))
    requests=u.load(RUN/'summary_requests.json');reuse=u.load(RUN/'summary_reuse.json')
    for i,(key,req) in enumerate(sorted(requests.items()),1):
        if key not in reuse:durable_call('summary',req,backend,attempt)
        if i%10==0 or i==len(requests):state('running',phase='summary',completed=i,expected=len(requests))

def make_plans(source,result,tokenizer):
    from experiments.recent_baselines_v1.core import cap_text
    text=result['text'];capped,applied=cap_text(text,512,tokenizer)
    return [dict(dataset='musique',question_id=source['question_id'],question=source['question'],method=method,
        budget=512 if method==METHODS[0] else 0,evidence_text=value,
        evidence_tokens=len(tokenizer.encode(value,add_special_tokens=False)),status='ready' if value.strip() else 'empty_summary',
        trace=dict(summary_finish=result['finish_reason'],cap_applied=applied if method==METHODS[0] else False,
                   native_tokens=len(tokenizer.encode(text,add_special_tokens=False))))
        for method,value in zip(METHODS,[capped,text])]

def assemble():
    verify();started=time.perf_counter();tok=rt.tokenizer(CONFIG['reader'])
    requests=u.load(RUN/'summary_requests.json');mapping=u.load(RUN/'summary_mapping.json');reuse=u.load(RUN/'summary_reuse.json')
    assert all(k in reuse or result_path('summary',k).exists() for k in requests)
    plans=[];qa_requests={};qa_mapping=[];by={s['question_id']:s for s in parent.sources()}
    for q,s in by.items():
        key=mapping[q];row=reuse[key] if key in reuse else u.load(result_path('summary',key))
        if key not in reuse:assert row['request']==requests[key]
        for p in make_plans(s,row['result'],tok):
            req=parent.screen.qa_request(p,s)
            if p['status']=='empty_summary':qa_mapping.append(None)
            elif rt.prompt_tokens(tok,req['messages'])>8192:
                p['status']='input_infeasible';qa_mapping.append(None)
            else:qa_mapping.append(req['key']);qa_requests[req['key']]=req
            plans.append(p)
    assert len(plans)==len(qa_mapping)==960 and len({(p['question_id'],p['method']) for p in plans})==960
    cache=cache_candidates();qa_reuse={}
    for key,req in qa_requests.items():
        row=exact_reuse(req,cache)
        if row is not None:qa_reuse[key]=row
    for name,value in [('plans',plans),('qa_mapping',qa_mapping),('qa_requests',qa_requests),('qa_reuse',qa_reuse)]:put(RUN/(name+'.json'),value)
    put(RUN/'assembly.json',dict(logical_conditions=960,unique_qa_requests=len(qa_requests),reused_qa_requests=len(qa_reuse),
        new_qa_requests=len(qa_requests)-len(qa_reuse),plan_statuses=dict(Counter(p['status'] for p in plans)),
        packing_and_cache_seconds=time.perf_counter()-started,references_read=False))
    state('assembled',new_reader_requests=len(qa_requests)-len(qa_reuse),conditions=960)

def qa(backend,attempt):
    requests=u.load(RUN/'qa_requests.json');reuse=u.load(RUN/'qa_reuse.json')
    pending=[(k,r) for k,r in sorted(requests.items()) if k not in reuse]
    for i,(key,req) in enumerate(pending,1):
        durable_call('qa',req,backend,attempt)
        if i%10==0 or i==len(pending):state('running',phase='qa',completed=i,expected=len(pending))

def worker(phase,attempt):
    import fcntl
    import torch
    from .paper_competition_fastqueue_v1 import NvmlMemory
    sid=verify();started=time.perf_counter();folder=RUN/'attempts'/attempt;okay=False
    torch.set_num_threads(8);torch.manual_seed(30);torch.cuda.manual_seed_all(30)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        state('preloading_cpu',phase=phase,attempt=attempt)
        backend=rt.preload('qa_qwen')
        torch.cuda.set_per_process_memory_fraction(CONFIG['cap_mib']*2**20/torch.cuda.get_device_properties(0).total_memory)
        monitor=NvmlMemory()
        with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
            last=-float('inf')
            while True:
                free=monitor.free_mib();held=False
                if free>=CONFIG['admission_mib']:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);held=True
                    except BlockingIOError:pass
                if held:
                    try:
                        if monitor.free_mib()>=CONFIG['admission_mib']:
                            reservation=torch.empty(max(0,int(CONFIG['reserve_mib']*2**20-torch.cuda.memory_reserved())),dtype=torch.uint8,device='cuda')
                            del reservation;break
                    except torch.OutOfMemoryError:torch.cuda.empty_cache()
                    fcntl.flock(lock,fcntl.LOCK_UN)
                if time.monotonic()-last>=20:
                    state('waiting_for_memory',phase=phase,free_mib=round(free),required_mib=CONFIG['admission_mib'],cpu_weights_ready=True)
                    last=time.monotonic()
                time.sleep(CONFIG['poll_seconds'])
            backend.model.to('cuda').eval()
            assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in backend.model.parameters())
            put(folder/'placement.json',dict(all_weights_cuda=True,precision='bfloat16',quantization=False))
            state('running',phase=phase,attempt=attempt)
            (compress if phase=='summary' else qa)(backend,attempt)
        put(RUN/'completed_phases'/(phase+'.json'),dict(signature=sid,phase=phase));okay=True
        return 0
    except torch.OutOfMemoryError as exc:
        put(folder/'oom.json',dict(active_key=os.environ.get('SUMMARY_ACTIVE_KEY'),error=repr(exc),phase=phase));return 75
    finally:
        put(folder/'resources.json',dict(phase=phase,complete=okay,seconds_with_wait=time.perf_counter()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0,includes_active_reservation=True))

def analyze():
    from .evidence_acquisition_v2 import parse_answer
    from .paper_confirmation_analysis_v1 import paired
    sid=verify();plans=u.load(RUN/'plans.json');mapping=u.load(RUN/'qa_mapping.json')
    requests=u.load(RUN/'qa_requests.json');reuse=u.load(RUN/'qa_reuse.json')
    assert all(k in reuse or result_path('qa',k).exists() for k in requests)
    ss=parent.sources();wanted={s['question_id'] for s in ss}
    refs={q:r for q,r in u.load(parent.historical.PLAN/'private/references.json').items() if q in wanted};assert len(refs)==480
    records=[]
    with parent.historical.bind():
        for p,key in zip(plans,mapping):
            f1=em=0;status=p['status'];usage={}
            if key is not None:
                row=reuse[key] if key in reuse else u.load(result_path('qa',key))
                if key not in reuse:assert row['signature']==sid and row['request']==requests[key]
                result=row['result'];usage=result['usage']
                if result['finish_reason']!='stop':status='output_limit'
                else:
                    try:hypothesis=parse_answer(result['text'])
                    except (ValueError,KeyError,TypeError):status='parse_failure'
                    else:em,f1=parent.old.answer_scores(hypothesis,refs[p['question_id']]);status='ok'
            records.append(dict(question_id=p['question_id'],method=p['method'],budget=p['budget'],f1=f1,em=em,status=status,
                evidence_tokens=p['evidence_tokens'],usage=usage,key=key,reused=key in reuse if key else False))
    previous=u.load(parent.RUN/'analysis.json');all_records=previous['records']+records
    ix={(r['question_id'],r['method']):r for r in all_records};assert len(ix)==len(all_records)==9120
    clusters=parent.old.source_clusters(refs);hops={s['question_id']:int(s['native_id'][0]) for s in ss}
    contrasts=[('quote_anchor',METHODS[0]),('quote_anchor',METHODS[1]),(METHODS[0],'brief_pro_auto_cap'),(METHODS[1],'brief_pro_auto_native')]
    comparisons=[];hopcomparisons=[];groups=[];hopgroups=[]
    for m in parent.CONFIG['methods']+METHODS:
        rr=[ix[q,m] for q in wanted]
        groups.append(dict(method=m,n=480,f1=100*statistics.mean(r['f1'] for r in rr),em=100*statistics.mean(r['em'] for r in rr),
            evidence_tokens=statistics.mean(r['evidence_tokens'] for r in rr),statuses=dict(Counter(r['status'] for r in rr)),
            reader_input_tokens=statistics.mean(r['usage'].get('prompt_tokens',0) for r in rr),reader_output_tokens=statistics.mean(r['usage'].get('completion_tokens',0) for r in rr)))
        for h in (2,3,4):
            hh=[r for r in rr if hops[r['question_id']]==h]
            hopgroups.append(dict(method=m,hops=h,n=len(hh),f1=100*statistics.mean(r['f1'] for r in hh),em=100*statistics.mean(r['em'] for r in hh)))
    for a,b in contrasts:
        for metric in ('f1','em'):
            comparisons.append(dict(method=a,control=b,metric=metric,**paired({q:ix[q,a][metric]-ix[q,b][metric] for q in wanted},clusters)))
        if a=='quote_anchor':
            for h in (2,3,4):
                hopcomparisons.append(dict(method=a,control=b,hops=h,**paired({q:ix[q,a]['f1']-ix[q,b]['f1'] for q in wanted if hops[q]==h},clusters)))
    sr=u.load(RUN/'summary_reuse.json');sreq=u.load(RUN/'summary_requests.json')
    results=[(sr[k] if k in sr else u.load(result_path('summary',k)))['result'] for k in sreq]
    costs=dict(summary_calls=480,mean_summary_input_tokens=statistics.mean(r['usage']['prompt_tokens'] for r in results),
        mean_summary_output_tokens=statistics.mean(r['usage']['completion_tokens'] for r in results),
        mean_observed_summary_seconds=statistics.mean(r['usage']['elapsed_seconds'] for r in results),
        summary_finish_reasons=dict(Counter(r['finish_reason'] for r in results)),empty_summaries=sum(not r['text'].strip() for r in results),
        hard_capped_sources=sum(p['trace']['cap_applied'] for p in plans if p['method']==METHODS[0]),
        new_retrieval_calls=0,new_reranking_calls=0,
        note='Summary needs no reranker/Beam; quote retains its original retrieval/ranking/proposal deployment costs. Shared GPU times are observational.')
    primary=next(r for r in comparisons if (r['method'],r['control'],r['metric'])==('quote_anchor',METHODS[0],'f1'))
    interpretation='inconclusive_direction' if primary['ci95_points'][0]<=0<=primary['ci95_points'][1] else ('quote_advantage_supported' if primary['ci95_points'][0]>0 else 'summary_advantage_supported')
    report=dict(signature=sid,complete=True,scope=CONFIG['scope'],groups=groups,comparisons=comparisons,hop_groups=hopgroups,
        hop_comparisons=hopcomparisons,new_records=records,costs=costs,primary_interpretation=interpretation,
        assembly=u.load(RUN/'assembly.json'),original_confirmation_decision_unchanged=True,automatic_expansion=False)
    put(RUN/'analysis.json',report)
    put(RUN/'completion.json',dict(signature=sid,new_conditions=960,total_conditions_with_inherited=9120,complete=True))
    lines=['# 同9B普通摘要对照完成报告','','范围：原480题已暴露开发集；单项描述性来源聚类95%区间。原判断不变。','',
        '|方法|F1|EM|平均证据token|','|---|---:|---:|---:|']
    for g in groups:lines.append(f"|{g['method']}|{g['f1']:.2f}|{g['em']:.2f}|{g['evidence_tokens']:.1f}|")
    lines+=['','|比较|指标|差值|95%区间|','|---|---|---:|---|']
    for r in comparisons:lines.append(f"|{r['method']} − {r['control']}|{r['metric']}|{r['delta_points']:+.2f}|[{r['ci95_points'][0]:+.2f}, {r['ci95_points'][1]:+.2f}]|")
    lines+=['',f"主比较：{interpretation}。跨零不等于等效，原生参考不用于替换主比较。",'',
        f"摘要每题平均输入/输出token：{costs['mean_summary_input_tokens']:.1f}/{costs['mean_summary_output_tokens']:.1f}；512裁切{costs['hard_capped_sources']}/480题。",'',
        '相同9B模型、原文和单次输出额度不等于实际总成本相同；引用方案仍有排序/检索及机械修复。全部分层、失败和成本见analysis.json。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state('complete',new_conditions=960,total_conditions=9120,primary_interpretation=interpretation)

def run():
    import fcntl
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify()
        if not (RUN/'preparation.json').exists():prepare()
        attempt=len(list((RUN/'attempts').glob('*')))
        for phase in ('summary','qa'):
            if phase=='qa' and not (RUN/'assembly.json').exists():assemble()
            while not (RUN/'completed_phases'/(phase+'.json')).exists():
                attempt+=1;name=f'{attempt:06d}';folder=RUN/'attempts'/name;folder.mkdir(parents=True)
                with (folder/'worker.log').open('a') as log:
                    proc=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.paper_same_model_summary_v1','worker','--phase',phase,'--attempt',name],
                        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    ticks=int((Path('/proc')/str(proc.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
                    (RUN/'active_worker.json').write_text(json.dumps(dict(pid=proc.pid,start_ticks=ticks,phase=phase,attempt=name)))
                    code=proc.wait()
                if code==75:
                    state('requeued_after_oom',phase=phase);continue
                assert code==0,'Worker failed; inspect '+str(folder/'worker.log')
        if not (RUN/'completion.json').exists():analyze()

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','run','worker','analyze'])
    p.add_argument('--phase',choices=['summary','qa']);p.add_argument('--attempt');args=p.parse_args()
    try:
        if args.action=='worker':raise SystemExit(worker(args.phase,args.attempt))
        globals()[args.action]()
    except Exception as exc:state('failed',error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
