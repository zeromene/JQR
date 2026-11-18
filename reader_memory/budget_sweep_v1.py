"""Frozen fixed-budget JQR/reranker sweep, two readers and exact request reuse."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType
import argparse
import json
import os
import statistics
import subprocess
import time
import traceback
from . import compact_reranker_control_v1 as previous
from . import budget_sweep_core_v1 as core

parent, runtime, u, ROOT = previous.parent, previous.runtime, previous.u, previous.ROOT
NAME='budget_sweep_v1'
BASE, RUN = ROOT/'research'/NAME, ROOT/'runs'/NAME
CONFIG=dict(version=NAME,sources=1935,budgets=list(core.BUDGETS),methods=list(core.METHODS),
    readers=list(parent.CONFIG['readers']),reader_order=['Phi-4-mini-instruct','Qwen3.5-9B'],
    logical_conditions=30960,seed=30,bootstrap=10000,proposer_calls=0,ranking_calls=0,training=False,
    cap='External fixed nominal Qwen token budget; never JQR realized length',
    scope='Supplementary fixed-budget strategy comparison on the existing exposed panels',
    simultaneous_f1_family=16,simultaneous_budget_interaction_family=12,
    wait_timeout=None,wall_timeout=None,automatic_expansion=False,
    qwen_memory_mib=[26000,24000,24500],phi_memory_mib=[12000,10500,11000])
FILES=[f'reader_memory/{NAME}.py','reader_memory/budget_sweep_core_v1.py',
    f'tests/test_{NAME}.py',f'tools/{NAME}_remote.py',f'tools/plot_{NAME}.py',
    f'research/{NAME}/PROTOCOL.md','reader_memory/short_budget_controls_v1.py',
    'reader_memory/anchored_quote_v1.py','reader_memory/conjunctive_quote_plan.py',
    'reader_memory/anchor_guard_controls_v1.py','reader_memory/path_budget_controls_v1.py',
    'reader_memory/relation_preserving_pilot_v1.py','reader_memory/full_comparison_v1.py',
    'reader_memory/paper_component_controls_v2.py','reader_memory/paper_confirmation_analysis_v1.py',
    'reader_memory/paper_confirmation_runtime_v1.py','reader_memory/mainline_service_v1.py']
CACHE_RUNS=[parent.RUN,previous.old.COMP,previous.old.RUN,previous.RUN]


def put(path,value):runtime.put(path,value)


def state(status,**kwargs):
    RUN.mkdir(parents=True,exist_ok=True)
    v=dict(status=status,time=datetime.now(timezone.utc).isoformat(),**kwargs)
    v.setdefault('expires_at',None)
    p=RUN/'status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
    print(json.dumps(v,ensure_ascii=False),flush=True)


def freeze():
    signature=previous.verify()
    paths=[]
    for folder in CACHE_RUNS:
        paths += [folder/name for name in ['plans.json','requests.json']]
        paths += [folder/name for name in ['reuse.json','reuse_hashes.json'] if (folder/name).exists()]
    paths += [parent.RUN/'analysis.json',parent.BASE/'allocation.json']
    put(BASE/'manifest.json',dict(config=CONFIG,parent_signature=signature,
        files={p:u.sha(ROOT/p) for p in FILES},inputs={str(p.relative_to(ROOT)):u.sha(p) for p in paths},
        frozen_at=datetime.now(timezone.utc).isoformat()))
    state('frozen',signature=u.sha(BASE/'manifest.json'))


def verify():
    m=u.load(BASE/'manifest.json')
    assert m['config']==CONFIG and previous.verify()==m['parent_signature']
    for p,digest in (m['files']|m['inputs']).items():assert u.sha(ROOT/p)==digest,p
    if (RUN/'material_manifest.json').exists():
        for p,digest in u.load(RUN/'material_manifest.json')['files'].items():assert u.sha(ROOT/p)==digest,p
    return u.sha(BASE/'manifest.json')


def prepare():
    sid=verify();tokens={r:runtime.tokenizer(r) for r in CONFIG['readers']}
    class Count:
        exact=True
        def __init__(self):self.memo={}
        def count(self,text):
            if text not in self.memo:self.memo[text]=len(tokens['Qwen3.5-9B'].encode(text,add_special_tokens=False))
            return self.memo[text]
    counter=Count()
    oldplans={(p['question_id'],p['reader'],p['method']):p for p in u.load(parent.RUN/'plans.json')
              if p['study']=='complete' and p['method'] in core.METHODS}
    cache={};signatures={u.sha(ROOT/'research'/folder.name/'manifest.json') for folder in CACHE_RUNS}
    for folder in CACHE_RUNS:
        rr=u.load(folder/'reuse.json') if (folder/'reuse.json').exists() else {}
        for key,req in u.load(folder/'requests.json').items():
            dest=ROOT/rr[key] if key in rr else folder/'outputs'/(key+'.json')
            if key not in cache:cache[key]=(req,dest)
            else:
                v=cache[key][0]
                assert (v['reader'],v['messages'],v['max_output'])==(req['reader'],req['messages'],req['max_output'])
    plans,mapping,requests,reuse,reuse_hashes,input_hashes,audits=[],[],{},{},{},{},[]
    for i,x in enumerate(parent.sources(),1):
        started=time.perf_counter();s=x['source'];q=s['question_id'];d=x['dataset'];counter.memo.clear()
        scores=runtime.cached('rank',q,s)['value']['scores']
        ret=runtime.cached('retrieve',q,s)['value']
        prop=runtime.cached('proposal',[q,'role'],s)['value']['parsed']
        for phase,ident in [('rank',q),('retrieve',q),('proposal',[q,'role'])]:
            p=runtime.result_path(phase,ident);input_hashes[str(p.relative_to(ROOT))]=u.sha(p)
        for budget in core.BUDGETS:
            built,trace=core.construct(s,scores,[str(k) for k in ret['paths'][0]],prop,counter,budget)
            audit=dict(dataset=d,question_id=q,budget=budget,source_hash=u.fp(s),proposal_hash=u.fp(prop),
                admitted=trace['admitted'],fallback=trace.get('fallback'),effective_budget=trace['effective_budget'],
                anchor=trace.get('anchor'),trace=trace,methods={})
            for method,value in built.items():
                failed=bool(value.get('selection_error'))
                audit['methods'][method]=dict(evidence_tokens=value['evidence_tokens'],
                    evidence_sha256=u.fp(value['evidence_text']),selection_failed=failed)
                for reader in CONFIG['readers']:
                    tok=tokens[reader]
                    p=dict(dataset=d,question_id=q,reader=reader,method=method,budget=budget,
                        effective_budget=trace['effective_budget'],evidence_text=value['evidence_text'],
                        evidence_tokens=value['evidence_tokens'],actual_reader_evidence_tokens=len(tok.encode(value['evidence_text'],add_special_tokens=False)),
                        status='selection_failed' if failed else 'ready',selection_error=value.get('selection_error'),
                        admitted=trace['admitted'] if method=='quote_anchor' else None,
                        fallback=trace.get('fallback') if method=='quote_anchor' else None)
                    assert counter.count(p['evidence_text'])==p['evidence_tokens']<=budget
                    if budget==512:
                        old=oldplans[q,reader,method]
                        assert p['evidence_text']==old['evidence_text'] and p['status']==old['status'],('512 reproduction',q,reader,method)
                    key=None
                    if not failed:
                        req=parent.qa_request(p,s,reader);key=req['key']
                        assert key==u.fp([reader,req['messages'],req['max_output']])
                        p['prompt_tokens']=runtime.prompt_tokens(tok,req['messages'])
                        assert p['prompt_tokens']<=parent.CONFIG['reader_max_input']
                        if key not in requests:
                            if key in cache:
                                oldreq,dest=cache[key]
                                assert (req['reader'],req['messages'],req['max_output'])==(oldreq['reader'],oldreq['messages'],oldreq['max_output'])
                                out=u.load(dest)
                                assert out['key']==out['request_identity']==key and out['signature'] in signatures
                                assert out['result']['usage']['prompt_tokens']==p['prompt_tokens']
                                reuse[key]=str(dest.relative_to(ROOT));reuse_hashes[reuse[key]]=u.sha(dest)
                            requests[key]=req
                        if budget==512:assert key in reuse,'512 output must be reusable'
                    plans.append(p);mapping.append(key)
            audits.append(audit)
        if i%25==0 or i==CONFIG['sources']:
            state('preparing',completed=i,expected=CONFIG['sources'],new_unique_requests=len(requests)-len(reuse),last_source_seconds=time.perf_counter()-started)
    assert len(plans)==len(mapping)==CONFIG['logical_conditions']
    assert len({(p['question_id'],p['reader'],p['method'],p['budget']) for p in plans})==len(plans)
    summaries=[]
    for d in ['hotpot','musique']:
        for b in core.BUDGETS:
            rows=[a for a in audits if a['dataset']==d and a['budget']==b]
            summaries.append(dict(dataset=d,budget=b,n=len(rows),admitted=sum(a['admitted'] for a in rows),
                admission_percent=100*statistics.mean(a['admitted'] for a in rows),
                fallback_reasons=dict(Counter(a['fallback'] for a in rows if not a['admitted'])),
                effective_limit=previous.summary([a['effective_budget'] for a in rows]),
                methods={m:dict(mean_tokens=statistics.mean(a['methods'][m]['evidence_tokens'] for a in rows),
                    selection_failures=sum(a['methods'][m]['selection_failed'] for a in rows)) for m in core.METHODS}))
    prep=dict(signature=sid,logical_conditions=len(plans),unique_requests=len(requests),exact_reused_requests=len(reuse),
        new_requests_by_reader={r:sum(k not in reuse and v['reader']==r for k,v in requests.items()) for r in CONFIG['readers']},
        original_512_reproductions=7740,references_read=False,proposer_calls=0,ranking_calls=0,summaries=summaries)
    for name,val in [('plans',plans),('mapping',mapping),('requests',requests),('reuse',reuse),('reuse_hashes',reuse_hashes),
                     ('input_hashes',input_hashes),('construction_audit',audits),('preparation',prep)]:put(RUN/(name+'.json'),val)
    names=['plans','mapping','requests','reuse','reuse_hashes','input_hashes','construction_audit','preparation']
    put(RUN/'material_manifest.json',dict(signature=sid,files={str((RUN/(n+'.json')).relative_to(ROOT)):u.sha(RUN/(n+'.json')) for n in names}))
    state('prepared',**prep)


def qa_worker(reader,attempt):
    from . import paper_component_controls_v2 as worker
    verify()
    for name in ['input_hashes','reuse_hashes']:
        for path,digest in u.load(RUN/(name+'.json')).items():assert u.sha(ROOT/path)==digest,path
    fn=worker.qa_worker
    bound=FunctionType(fn.__code__,dict(fn.__globals__,RUN=RUN,verify=verify,state=state,put=put),fn.__name__,fn.__defaults__,fn.__closure__)
    return bound(reader,attempt)


def analyze():
    import numpy as np
    from . import paper_confirmation_analysis_v1 as analysis
    from .evidence_acquisition_v2 import parse_answer
    sid=verify();requests=u.load(RUN/'requests.json');reuse=u.load(RUN/'reuse.json');hashes=u.load(RUN/'reuse_hashes.json')
    assert all(k in reuse or (RUN/'outputs'/(k+'.json')).exists() for k in requests)
    outputs={}
    for key,req in requests.items():
        path=ROOT/reuse[key] if key in reuse else RUN/'outputs'/(key+'.json');out=u.load(path)
        assert key==out['key']==out['request_identity']==u.fp([req['reader'],req['messages'],req['max_output']])
        if key in reuse:assert u.sha(path)==hashes[reuse[key]]
        else:assert out['signature']==sid
        outputs[key]=out['result']
    refs=analysis.references();metric=parent.old.hp.official_metrics();records=[]
    plans=u.load(RUN/'plans.json');mapping=u.load(RUN/'mapping.json')
    assert len(plans)==len(mapping)==CONFIG['logical_conditions']
    for p,key in zip(plans,mapping):
        em=f1=0.;hypothesis='';status=p['status'];finish=None;usage={}
        if key is not None:
            result=outputs[key];finish=result['finish_reason'];usage=result['usage']
            assert usage['prompt_tokens']==p['prompt_tokens']
            if p['dataset']=='hotpot':
                hypothesis=result['text'];status='ok' if hypothesis.strip() else 'empty'
                em=float(metric['exact_match_score'](hypothesis,refs[p['question_id']]['answer']))
                f1=float(metric['f1_score'](hypothesis,refs[p['question_id']]['answer'])[0])
            elif finish!='stop':status='output_limit'
            else:
                try:hypothesis=parse_answer(result['text'])
                except (ValueError,KeyError,TypeError):status='parse_failure'
                else:em,f1=analysis.mu.answer_scores(hypothesis,refs[p['question_id']]);status='ok'
        records.append({k:v for k,v in p.items() if k!='evidence_text'}|dict(key=key,f1=f1,em=em,hypothesis=hypothesis,
            status=status,finish_reason=finish,valid=status=='ok' and finish=='stop',usage=usage,reused=key in reuse))
    idx={(p['dataset'],p['reader'],p['question_id'],p['budget'],p['method']):p for p in records}
    replay=0
    for p in u.load(parent.RUN/'analysis.json')['records']:
        if p['study']=='complete' and p['method'] in core.METHODS:
            now=idx[p['dataset'],p['reader'],p['question_id'],512,p['method']]
            assert abs(now['f1']-p['f1'])<1e-12 and abs(now['em']-p['em'])<1e-12
            replay+=1
    assert replay==7740
    meta={('hotpot:'+p['native_id'] if p['dataset']=='hotpot' else p['native_id']+':multi'):p for p in u.load(parent.BASE/'allocation.json')['selected']}
    groups=[];labels=[];interlabels=[];estimates=[];boots=[];interestimates=[];interboots=[];emstats=[];sensitivity=[]
    for di,d in enumerate(['hotpot','musique']):
        ids=sorted(q for q,v in meta.items() if v['dataset']==d)
        clusters=[meta[q]['component_id' if d=='hotpot' else 'support_document_component'] for q in ids]
        values=[];emvalues=[]
        for q in ids:
            values.append([idx[d,r,q,b,'quote_anchor']['f1']-idx[d,r,q,b,'strong_rerank']['f1'] for r in CONFIG['readers'] for b in core.BUDGETS])
            emvalues.append([idx[d,r,q,b,'quote_anchor']['em']-idx[d,r,q,b,'strong_rerank']['em'] for r in CONFIG['readers'] for b in core.BUDGETS])
        est,boot=core.bootstrap_matrix(values,clusters,CONFIG['bootstrap'],CONFIG['seed']+di)
        ee,eb=core.bootstrap_matrix(emvalues,clusters,CONFIG['bootstrap'],CONFIG['seed']+di)
        estimates.append(est);boots.append(boot)
        ee_ci=np.quantile(eb,[.025,.975],axis=0)
        intercols=[(j,j//4*4+3) for j in range(8) if j%4!=3]
        interestimates.append(np.array([est[j]-est[k] for j,k in intercols]))
        interboots.append(np.stack([boot[:,j]-boot[:,k] for j,k in intercols],axis=1))
        for j,(r,b) in enumerate((r,b) for r in CONFIG['readers'] for b in core.BUDGETS):
            labels.append(dict(dataset=d,reader=r,budget=b,n=len(ids),clusters=len(set(clusters))))
            emstats.append(dict(delta_points=float(100*ee[j]),ci95_points=(100*ee_ci[:,j]).tolist()))
            if b!=512:interlabels.append(dict(dataset=d,reader=r,budget=b,reference_budget=512,n=len(ids),clusters=len(set(clusters))))
            for m in core.METHODS:
                rr=[idx[d,r,q,b,m] for q in ids]
                groups.append(dict(dataset=d,reader=r,budget=b,method=m,n=len(rr),
                    f1=100*statistics.mean(v['f1'] for v in rr),em=100*statistics.mean(v['em'] for v in rr),
                    valid_percent=100*statistics.mean(v['valid'] for v in rr),statuses=dict(Counter(v['status'] for v in rr)),
                    evidence_tokens=statistics.mean(v['evidence_tokens'] for v in rr),
                    actual_reader_evidence_tokens=statistics.mean(v['actual_reader_evidence_tokens'] for v in rr),
                    prompt_tokens=statistics.mean(v.get('prompt_tokens',0) for v in rr)))
            valid=[q for q in ids if all(idx[d,r,q,b,m]['valid'] for m in core.METHODS)]
            sensitivity.append(dict(dataset=d,reader=r,budget=b,n=len(valid),total=len(ids),
                f1_delta_points=100*statistics.mean(idx[d,r,q,b,'quote_anchor']['f1']-idx[d,r,q,b,'strong_rerank']['f1'] for q in valid) if valid else None,
                em_delta_points=100*statistics.mean(idx[d,r,q,b,'quote_anchor']['em']-idx[d,r,q,b,'strong_rerank']['em'] for q in valid) if valid else None))
    results,critical=core.interval_family(np.concatenate(estimates),np.concatenate(boots,axis=1))
    interaction,intercritical=core.interval_family(np.concatenate(interestimates),np.concatenate(interboots,axis=1))
    comparisons=[dict(**label,f1=stat,em=e) for label,stat,e in zip(labels,results,emstats)]
    interactions=[dict(**label,f1=stat) for label,stat in zip(interlabels,interaction)]
    value=dict(signature=sid,complete=True,logical_conditions=len(records),groups=groups,comparisons=comparisons,
        interactions=interactions,both_valid_sensitivity=sensitivity,records=records,original_512_score_reproductions=replay,
        preparation=u.load(RUN/'preparation.json'),scope=CONFIG['scope'],automatic_expansion=False,
        bootstrap=dict(draws=CONFIG['bootstrap'],joint_f1_family_size=16,interaction_family_size=12,
                       f1_max_standardized_critical=critical,interaction_max_standardized_critical=intercritical),
        original_primary_decisions_unchanged=True)
    put(RUN/'analysis.json',value)
    lines=['# 固定预算扫描：全部完成','',CONFIG['scope'],'',
        '|数据|读者|预算|JQR F1|重排 F1|ΔF1|95% 区间|族内同时 95% 区间|','|---|---|---:|---:|---:|---:|---|---|']
    for c in comparisons:
        gg={g['method']:g for g in groups if all(g[k]==c[k] for k in ['dataset','reader','budget'])}
        f=c['f1'];lines.append(f"|{c['dataset']}|{c['reader']}|{c['budget']}|{gg['quote_anchor']['f1']:.2f}|{gg['strong_rerank']['f1']:.2f}|{f['delta_points']:+.2f}|{f['ci95_points']}|{f['simultaneous_ci95_points']}|")
    lines+=['','## 相对 512 的预算交互','', '|数据|读者|预算|Δ(B)−Δ(512)|同时 95% 区间|','|---|---|---:|---:|---|']
    for c in interactions:
        f=c['f1'];lines.append(f"|{c['dataset']}|{c['reader']}|{c['budget']}|{f['delta_points']:+.2f}|{f['simultaneous_ci95_points']}|")
    lines+=['','## 接纳与长度','', '|数据|预算|接纳率%|JQR长度|重排长度|回退原因|','|---|---:|---:|---:|---:|---|']
    for c in value['preparation']['summaries']:
        lines.append(f"|{c['dataset']}|{c['budget']}|{c['admission_percent']:.2f}|{c['methods']['quote_anchor']['mean_tokens']:.2f}|{c['methods']['strong_rerank']['mean_tokens']:.2f}|{c['fallback_reasons']}|")
    lines+=['','完整 F1/EM、实际读者长度、失败及双方均有效敏感性结果见 analysis.json。全部 7,740 个 512-token 条件的 F1/EM 精确复现。固定名义预算不等于相同实际长度或总成本；不替换原主要检验。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state('complete',logical_conditions=len(records),unique_requests=len(requests),original_512_score_reproductions=replay)


def run():
    import fcntl
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        if not (RUN/'preparation.json').exists():prepare()
        prep=u.load(RUN/'preparation.json')
        for reader in CONFIG['reader_order']:
            while not (RUN/'completed_readers'/(reader+'.json')).exists():
                if not prep['new_requests_by_reader'][reader]:
                    put(RUN/'completed_readers'/(reader+'.json'),dict(reader=reader,all_reused=True));break
                attempt=max([int(p.name) for p in (RUN/'attempts').glob('*') if p.name.isdigit()]+[0])+1
                name=f'{attempt:06d}';folder=RUN/'attempts'/name;folder.mkdir(parents=True)
                with (folder/'worker.log').open('x') as log:
                    child=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.budget_sweep_v1','worker','--reader',reader,'--attempt',name],
                        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    (RUN/'active_worker.json').write_text(json.dumps(dict(pid=child.pid,reader=reader,attempt=name,
                        start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]))),encoding='utf-8')
                    code=child.wait()
                if code==75:continue
                assert code==0,'Worker failed; preserve interrupted requests: '+str(folder/'worker.log')
        if not (RUN/'analysis.json').exists():analyze()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','run','worker','analyze'])
    p.add_argument('--reader');p.add_argument('--attempt');a=p.parse_args()
    try:
        if a.action=='worker':raise SystemExit(qa_worker(a.reader,a.attempt))
        globals()[a.action]()
    except Exception as exc:state('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
