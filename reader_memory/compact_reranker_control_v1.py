"""Frozen full-candidate production reranker at each question's JQR token cap."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import math
import os
import statistics
import subprocess
import time
import traceback
from . import quote_retention_mechanism_v2 as old
from . import compact_reranker_control_core_v1 as core

parent, runtime, u, ROOT = old.parent, old.runtime, old.u, old.ROOT
NAME = 'compact_reranker_control_v1'
BASE, RUN = ROOT/'research'/NAME, ROOT/'runs'/NAME
ARMS = ('A_jqr','E_full_pool_tight')
CONFIG = dict(version=NAME,sources=1935,arms=list(ARMS),readers=list(parent.CONFIG['readers']),
    logical_conditions=7740,nominal_budget=512,cap='same-question A actual Qwen evidence tokens',
    selection='unchanged production all-native-sentence reranker packer; no quote candidates or routing',
    reader_order=['Phi-4-mini-instruct','Qwen3.5-9B'],seed=30,bootstrap=10000,
    focal='MuSiQue/Qwen A minus E: >=2 F1 points and 95% lower bound >0',
    scope='Supplementary exposed-panel same-cap strategy comparison; not exact-length or independent confirmation',
    wait_timeout=None,wall_timeout=None,training=False,proposer_calls=0,ranking_calls=0,
    qwen_resource_policy=dict(admission=20250,reserve=19000,cap=19500,after_oom=[26000,24000,24500]),
    automatic_expansion=False)
FILES = ['reader_memory/compact_reranker_control_v1.py','reader_memory/compact_reranker_control_core_v1.py',
    'tools/compact_reranker_control_v1_remote.py','tests/test_compact_reranker_control_v1.py',
    'research/compact_reranker_control_v1/PROTOCOL.md',
    'reader_memory/relation_preserving_pilot_v1.py','reader_memory/paper_component_controls_v2.py',
    'reader_memory/paper_confirmation_runtime_v1.py','reader_memory/mainline_service_v1.py',
    'reader_memory/paper_confirmation_analysis_v1.py']


def put(path,value): runtime.put(path,value)


def state(status,**kwargs):
    RUN.mkdir(parents=True,exist_ok=True)
    row=dict(status=status,time=datetime.now(timezone.utc).isoformat(),**kwargs)
    row.setdefault('expires_at',None)
    p=RUN/'status.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def freeze():
    signature=old.verify()
    from . import quote_retention_memory_adapter_v1 as adapter
    resource_signature=adapter.verify()
    assert u.load(adapter.RUN/'probe.json')['passed']
    paths=[old.RUN/n for n in ['plans.json','requests.json','reuse.json','reuse_hashes.json',
        'construction_audit.json','analysis.json','material_manifest.json']]
    paths += [parent.RUN/n for n in ['plans.json','requests.json']]
    paths += [old.COMP/n for n in ['requests.json','reuse.json']]
    paths += [adapter.RUN/'probe.json',adapter.BASE/'adapter_manifest.json']
    value=dict(config=CONFIG,parent_signature=signature,resource_signature=resource_signature,
        files={p:u.sha(ROOT/p) for p in FILES},inputs={str(p.relative_to(ROOT)):u.sha(p) for p in paths},
        frozen_at=datetime.now(timezone.utc).isoformat())
    put(BASE/'manifest.json',value)
    state('frozen',signature=u.sha(BASE/'manifest.json'))


def verify():
    m=u.load(BASE/'manifest.json')
    assert m['config']==CONFIG and old.verify()==m['parent_signature']
    for path,digest in (m['files']|m['inputs']).items(): assert u.sha(ROOT/path)==digest,path
    if (RUN/'material_manifest.json').exists():
        for path,digest in u.load(RUN/'material_manifest.json')['files'].items():
            assert u.sha(ROOT/path)==digest,path
    return u.sha(BASE/'manifest.json')


def summary(values):
    ordered=sorted(values)
    return dict(mean=statistics.mean(values),median=statistics.median(values),
        p90=ordered[min(len(ordered)-1,math.ceil(.9*len(ordered))-1)],max=max(values),min=min(values))


def prepare():
    sid=verify()
    tokens={r:runtime.tokenizer(r) for r in CONFIG['readers']}
    class Count:
        exact=True
        def count(self,text): return len(tokens['Qwen3.5-9B'].encode(text,add_special_tokens=False))
    counter=Count()
    oldplans=u.load(old.RUN/'plans.json')
    prior={(p['question_id'],p['reader'],p['method']):p for p in oldplans}
    original={(p['question_id'],p['reader']):p for p in u.load(parent.RUN/'plans.json')
              if p['study']=='complete' and p['method']=='strong_rerank'}
    oldaudit={r['question_id']:r for r in u.load(old.RUN/'construction_audit.json')}
    cache={}
    valid_signatures={u.sha(parent.BASE/'manifest.json'),u.sha(ROOT/'research/paper_component_controls_v2/manifest.json'),
                      u.sha(old.BASE/'manifest.json')}
    for folder in [parent.RUN,old.COMP,old.RUN]:
        reqs=u.load(folder/'requests.json')
        rr=u.load(folder/'reuse.json') if (folder/'reuse.json').exists() else {}
        for key,req in reqs.items():
            dest=ROOT/rr[key] if key in rr else folder/'outputs'/(key+'.json')
            if key in cache:
                oldreq,_=cache[key]
                assert (req['reader'],req['messages'],req['max_output'])==(oldreq['reader'],oldreq['messages'],oldreq['max_output'])
            else: cache[key]=(req,dest)
    plans,mapping,requests,reuse,reuse_hashes,input_hashes,audits=[],[],{},{},{},{},[]
    for i,x in enumerate(parent.sources(),1):
        started=time.perf_counter();s=x['source'];q=s['question_id'];d=x['dataset']
        scores=runtime.cached('rank',q,s)['value']['scores']
        ret=runtime.cached('retrieve',q,s)['value']
        proposal=runtime.cached('proposal',[q,'role'],s)['value']['parsed']
        for phase,identity in [('rank',q),('retrieve',q),('proposal',[q,'role'])]:
            path=runtime.result_path(phase,identity);input_hashes[str(path.relative_to(ROOT))]=u.sha(path)
        global_plan,fallback=parent.old.hp.short.short_plans(s,scores,[str(k) for k in ret['paths'][0]],counter,512,512)
        for r in CONFIG['readers']:
            assert global_plan['evidence_text']==original[q,r]['evidence_text'],('512 production reproduction',q)
        if global_plan.get('selection_error') or fallback.get('selection_error'):
            a,trace=fallback,dict(fallback='baseline_selection_failure')
        else:
            a,trace=parent.old.hp.anchored.quote_plan(s,scores,global_plan,fallback,proposal,counter,'quote_anchor')
        for r in CONFIG['readers']:
            assert a['evidence_text']==prior[q,r,ARMS[0]]['evidence_text'],('A reproduction',q)
        admitted=not trace.get('fallback');cap=a['evidence_tokens']
        assert admitted==oldaudit[q]['admitted']
        anchor=old.core.span_for(s,trace['anchor']) if admitted else None
        e_error=None
        try:
            e,e_trace=core.full_pool_plan(s,scores,counter,cap,anchor)
        except ValueError as exc:
            if not parent.old.hp.short.original.empty_packing_error(exc): raise
            e=dict(evidence_text='',evidence_tokens=0,selected_ranges={},selected_spans=[])
            e_trace=[];e_error=str(exc)
        a_ranges=old.plan_ranges(s,a);a=dict(a,selected_ranges=a_ranges)
        diagnostic=None
        if admitted:
            pool=old.core.candidate_pool(s,scores,proposal,fallback)
            c=old.core.pack(s,scores,pool,old.core.extend({},anchor),cap,counter.count)
            assert c['evidence_text']==prior[q,CONFIG['readers'][0],'C_reselect_tight']['evidence_text'],('C reproduction',q)
            diagnostic=core.construction_audit(s,scores,proposal,fallback,a,c,anchor,cap,counter.count)
        audit=dict(question_id=q,dataset=d,source_hash=u.fp(s),admitted=admitted,fallback=trace.get('fallback'),
            selection_cap=cap,A_tokens=cap,E_tokens=e['evidence_tokens'],length_shortfall=cap-e['evidence_tokens'],
            A_text_hash=u.fp(a['evidence_text']),E_text_hash=u.fp(e['evidence_text']),
            changed_vs_A=e['evidence_text']!=a['evidence_text'],
            changed_vs_C=e['evidence_text']!=prior[q,CONFIG['readers'][0],'C_reselect_tight']['evidence_text'],
            all_candidate_sentences=len(scores),old_common_pool_candidates=len(oldaudit[q]['pool']),
            E_selected_spans=e.get('selected_spans',[]),E_trace=e_trace,E_error=e_error,
            E_geometry=core.geometry(s,e['selected_ranges']),C_diagnostic=diagnostic)
        for arm,value in [(ARMS[0],a),(ARMS[1],e)]:
            retained=old.core.retention(proposal,old.plan_ranges(s,value))
            for reader in CONFIG['readers']:
                tok=tokens[reader]
                p=dict(dataset=d,question_id=q,reader=reader,method=arm,budget=512,
                    effective_budget=global_plan['evidence_tokens'],selection_cap=cap,
                    evidence_text=value['evidence_text'],evidence_tokens=value['evidence_tokens'],
                    actual_reader_evidence_tokens=len(tok.encode(value['evidence_text'],add_special_tokens=False)),
                    status=prior[q,reader,ARMS[0]]['status'] if arm==ARMS[0] else ('selection_failed' if e_error else 'ready'),
                    admitted=admitted,fallback=trace.get('fallback'),**retained)
                assert counter.count(p['evidence_text'])==p['evidence_tokens']<=cap<=512
                key=None
                if p['status']=='ready':
                    req=parent.qa_request(p,s,reader);key=req['key']
                    assert key==u.fp([reader,req['messages'],req['max_output']])
                    p['prompt_tokens']=runtime.prompt_tokens(tok,req['messages'])
                    assert p['prompt_tokens']<=parent.CONFIG['reader_max_input']
                    if key not in requests:
                        if key in cache:
                            oldreq,dest=cache[key]
                            assert (req['reader'],req['messages'],req['max_output'])==(oldreq['reader'],oldreq['messages'],oldreq['max_output'])
                            out=u.load(dest)
                            assert out['key']==out['request_identity']==key and out['signature'] in valid_signatures
                            assert out['result']['usage']['prompt_tokens']==p['prompt_tokens']
                            reuse[key]=str(dest.relative_to(ROOT));reuse_hashes[reuse[key]]=u.sha(dest)
                        requests[key]=req
                    if arm==ARMS[0]: assert key in reuse,'A must be reused exactly'
                plans.append(p);mapping.append(key)
        audit['seconds']=time.perf_counter()-started;audits.append(audit)
        if i%25==0 or i==1935:
            state('preparing',completed=i,expected=1935,new_unique_requests=len(requests)-len(reuse))
    assert len(plans)==len(mapping)==CONFIG['logical_conditions']
    assert len({(p['question_id'],p['reader'],p['method']) for p in plans})==len(plans)
    summaries=[]
    for d in ['hotpot','musique']:
        rr=[a for a in audits if a['dataset']==d];adm=[a for a in rr if a['admitted']]
        lost=[a for a in adm if a['C_diagnostic']['lost_quotes']]
        witnesses=[a for a in lost if a['C_diagnostic']['single_sentence_replacement_witnesses']]
        summaries.append(dict(dataset=d,n=len(rr),admitted=len(adm),C_lost_quote_questions=len(lost),
            long_sentence_replacement_witness_questions=len(witnesses),
            same_score_precedence_witness_questions=sum(any(w['later_contained_same_score_quote_indices']
                for w in a['C_diagnostic']['single_sentence_replacement_witnesses']) for a in witnesses),
            C_loss_quotes_fit_with_anchor=sum(q['fits_with_anchor'] for a in lost for q in a['C_diagnostic']['lost_quotes']),
            C_loss_quotes=sum(len(a['C_diagnostic']['lost_quotes']) for a in lost),
            A_mean_partly_included_sentences=statistics.mean(a['C_diagnostic']['A_geometry']['partly_included_sentences'] for a in adm),
            C_mean_partly_included_sentences=statistics.mean(a['C_diagnostic']['C_geometry']['partly_included_sentences'] for a in adm),
            mean_A_tokens=statistics.mean(a['A_tokens'] for a in rr),mean_E_tokens=statistics.mean(a['E_tokens'] for a in rr),
            length_shortfall=summary([a['length_shortfall'] for a in rr]),
            exact_length_questions=sum(a['length_shortfall']==0 for a in rr),
            shortfall_at_most_8_questions=sum(a['length_shortfall']<=8 for a in rr),
            E_selection_failures=sum(a['E_error'] is not None for a in rr),
            changed_vs_A=sum(a['changed_vs_A'] for a in rr),changed_vs_C=sum(a['changed_vs_C'] for a in rr),
            max_absolute_A_E_F1_delta_bound=100*sum(a['changed_vs_A'] for a in rr)/len(rr)))
    prep=dict(signature=sid,logical_conditions=len(plans),unique_requests=len(requests),exact_reused_requests=len(reuse),
        new_requests_by_reader={r:sum(k not in reuse and v['reader']==r for k,v in requests.items()) for r in CONFIG['readers']},
        original_A_reproductions=1935,full_budget_production_reproductions=1935,
        original_C_reproductions=sum(a['admitted'] for a in audits),references_read=False,
        summaries=summaries,scope=CONFIG['scope'])
    for name,value in [('plans',plans),('mapping',mapping),('requests',requests),('reuse',reuse),
        ('reuse_hashes',reuse_hashes),('input_hashes',input_hashes),('construction_audit',audits),('preparation',prep)]:
        put(RUN/(name+'.json'),value)
    names=['plans','mapping','requests','reuse','reuse_hashes','input_hashes','construction_audit','preparation']
    put(RUN/'material_manifest.json',dict(signature=sid,files={str((RUN/(n+'.json')).relative_to(ROOT)):u.sha(RUN/(n+'.json')) for n in names}))
    state('prepared',**prep)


def qa_worker(reader,attempt,policy):
    from . import paper_component_controls_v2 as component
    verify()
    for name in ['input_hashes.json','reuse_hashes.json']:
        for relative,digest in u.load(RUN/name).items(): assert u.sha(ROOT/relative)==digest,relative
    source=inspect.getsource(component.qa_worker)
    before="isq=reader=='Qwen3.5-9B';admission=26000 if isq else 12000;reserve=24000 if isq else 10500;cap=24500 if isq else 11000"
    assert source.count(before)==1
    admission,reserve,cap=(20250,19000,19500) if policy=='compact' else (26000,24000,24500)
    after=f"isq=reader=='Qwen3.5-9B';admission={admission} if isq else 12000;reserve={reserve} if isq else 10500;cap={cap} if isq else 11000"
    source=source.replace(before,after)
    namespace=dict(component.__dict__);namespace.update(RUN=RUN,verify=verify,state=state)
    exec(compile(source,str(Path(__file__))+':validated_worker_resource_constants','exec'),namespace)
    put(RUN/'attempts'/attempt/'resource_policy.json',dict(reader=reader,policy=policy,
        qwen_admission_mib=admission,qwen_reserve_mib=reserve,qwen_cap_mib=cap,
        generated_worker_sha256=hashlib.sha256(source.encode()).hexdigest(),science_configuration_changed=False))
    return namespace['qa_worker'](reader,attempt)


def analyze():
    from . import paper_confirmation_analysis_v1 as analysis
    from .evidence_acquisition_v2 import parse_answer
    sid=verify();requests=u.load(RUN/'requests.json');reuse=u.load(RUN/'reuse.json')
    assert all(k in reuse or (RUN/'outputs'/(k+'.json')).exists() for k in requests)
    outputs={};hashes=u.load(RUN/'reuse_hashes.json')
    for key,req in requests.items():
        path=ROOT/reuse[key] if key in reuse else RUN/'outputs'/(key+'.json');out=u.load(path)
        assert out['key']==out['request_identity']==key==u.fp([req['reader'],req['messages'],req['max_output']])
        if key in reuse: assert u.sha(path)==hashes[reuse[key]]
        else: assert out['signature']==sid
        outputs[key]=out['result']
    refs=analysis.references();metric=parent.old.hp.official_metrics()
    plans=u.load(RUN/'plans.json');mapping=u.load(RUN/'mapping.json');records=[]
    for p,key in zip(plans,mapping):
        em=f1=0.;hypothesis='';status=p['status'];finish=None;usage={}
        if key is not None:
            result=outputs[key];finish=result['finish_reason'];usage=result['usage']
            assert usage['prompt_tokens']==p['prompt_tokens']
            if p['dataset']=='hotpot':
                hypothesis=result['text'];status='ok' if hypothesis.strip() else 'empty'
                em=float(metric['exact_match_score'](hypothesis,refs[p['question_id']]['answer']))
                f1=float(metric['f1_score'](hypothesis,refs[p['question_id']]['answer'])[0])
            elif finish!='stop': status='output_limit'
            else:
                try: hypothesis=parse_answer(result['text'])
                except (ValueError,KeyError,TypeError): status='parse_failure'
                else: em,f1=analysis.mu.answer_scores(hypothesis,refs[p['question_id']]);status='ok'
        records.append({k:v for k,v in p.items() if k!='evidence_text'}|dict(key=key,f1=f1,em=em,
            hypothesis=hypothesis,status=status,finish_reason=finish,usage=usage,reused=key in reuse))
    idx={(r['dataset'],r['reader'],r['question_id'],r['method']):r for r in records}
    oldrecords=u.load(old.RUN/'analysis.json')['records']
    for r in oldrecords:
        if r['method']=='A_jqr':
            now=idx[r['dataset'],r['reader'],r['question_id'],'A_jqr']
            assert abs(now['f1']-r['f1'])<1e-12 and abs(now['em']-r['em'])<1e-12
        elif r['method']=='C_reselect_tight': idx[r['dataset'],r['reader'],r['question_id'],r['method']]=r
    meta={('hotpot:'+r['native_id'] if r['dataset']=='hotpot' else r['native_id']+':multi'):r for r in u.load(parent.BASE/'allocation.json')['selected']}
    groups=[];comparisons=[]
    for d in ['hotpot','musique']:
        for reader in CONFIG['readers']:
            for scope in ['all','originally_admitted','originally_fallback']:
                rr=[r for r in records if r['dataset']==d and r['reader']==reader and r['method']==ARMS[0]
                    and (scope=='all' or r['admitted']==(scope=='originally_admitted'))]
                ids=sorted(r['question_id'] for r in rr)
                if not ids: continue
                clusters={q:meta[q]['component_id' if d=='hotpot' else 'support_document_component'] for q in ids}
                for arm in (*ARMS,'C_reselect_tight'):
                    rr=[idx[d,reader,q,arm] for q in ids]
                    groups.append(dict(dataset=d,reader=reader,scope=scope,method=arm,n=len(rr),
                        f1=100*statistics.mean(r['f1'] for r in rr),em=100*statistics.mean(r['em'] for r in rr),
                        evidence_tokens=statistics.mean(r['evidence_tokens'] for r in rr),
                        actual_reader_evidence_tokens=statistics.mean(r['actual_reader_evidence_tokens'] for r in rr),
                        prompt_tokens=statistics.mean(r.get('prompt_tokens',0) for r in rr),
                        statuses=dict(Counter(r['status'] for r in rr))))
                for method,control in [(ARMS[0],ARMS[1]),(ARMS[1],'C_reselect_tight')]:
                    paired={m:analysis.paired({q:idx[d,reader,q,method][m]-idx[d,reader,q,control][m] for q in ids},clusters) for m in ['f1','em']}
                    comparisons.append(dict(dataset=d,reader=reader,scope=scope,method=method,control=control,**paired,
                        mean_length_difference=statistics.mean(idx[d,reader,q,method]['evidence_tokens']-idx[d,reader,q,control]['evidence_tokens'] for q in ids)))
    focus=next(c for c in comparisons if c['dataset']=='musique' and c['reader']=='Qwen3.5-9B' and c['scope']=='all' and c['method']==ARMS[0])
    panel=[c for c in comparisons if c['scope']=='all' and c['method']==ARMS[0]]
    passed=lambda c:c['f1']['delta_points']>=2 and c['f1']['ci95_points'][0]>0
    decision='focal_advantage_supported_same_cap_only' if passed(focus) else 'focal_advantage_not_established_restrict_compact_selection_claim'
    value=dict(signature=sid,complete=True,groups=groups,comparisons=comparisons,records=records,
        focal=focus,focal_criterion_met=passed(focus),all_four_criterion_met=all(passed(c) for c in panel),decision=decision,
        scope=CONFIG['scope'],original_A_F1_EM_reproductions=3870,original_decisions_unchanged=True,
        construction_summary=u.load(RUN/'preparation.json')['summaries'],automatic_expansion=False)
    put(RUN/'analysis.json',value)
    lines=['# 全候选紧凑重排序对照','',CONFIG['scope'],'',
        '|数据|读者|比较|ΔF1|95%区间|ΔEM|平均长度差|','|---|---|---|---:|---|---:|---:|']
    for c in comparisons:
        if c['scope']=='all':
            lines.append(f"|{c['dataset']}|{c['reader']}|{c['method']} − {c['control']}|{c['f1']['delta_points']:+.2f}|{c['f1']['ci95_points']}|{c['em']['delta_points']:+.2f}|{c['mean_length_difference']:+.2f}|")
    lines+=['',f"焦点标准通过：{passed(focus)}；四组均通过：{all(passed(c) for c in panel)}。",'',
        'A/E 使用相同逐题预算上限，实际长度并非严格相同；不作精确等长或纯保留因果主张。旧 C 的失效与新生产对照的表现分开解释。没有扩样或追加调优。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state('complete',logical_conditions=len(records),decision=decision,focal_criterion_met=passed(focus))


def run():
    import fcntl
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify()
        if not (RUN/'preparation.json').exists(): prepare()
        prep=u.load(RUN/'preparation.json')
        for reader in CONFIG['reader_order']:
            while not (RUN/'completed_readers'/(reader+'.json')).exists():
                if not prep['new_requests_by_reader'][reader]:
                    put(RUN/'completed_readers'/(reader+'.json'),dict(reader=reader,all_reused=True));break
                policies=list((RUN/'attempts').glob('*/resource_policy.json'))
                policy='original' if any(u.load(p)['reader']==reader and (p.parent/'oom.json').exists() for p in policies) else 'compact'
                attempt=max([int(p.name) for p in (RUN/'attempts').glob('*') if p.name.isdigit()]+[0])+1
                name=f'{attempt:06d}';folder=RUN/'attempts'/name;folder.mkdir(parents=True)
                with (folder/'worker.log').open('x') as log:
                    child=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.compact_reranker_control_v1',
                        'worker','--reader',reader,'--attempt',name,'--policy',policy],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    receipt=dict(pid=child.pid,reader=reader,attempt=name,policy=policy,
                        start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]))
                    (RUN/'active_worker.json').write_text(json.dumps(receipt),encoding='utf-8')
                    code=child.wait()
                if code==75: continue
                assert code==0,'Worker failed; preserve interrupted requests: '+str(folder/'worker.log')
        if not (RUN/'analysis.json').exists(): analyze()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','run','worker','analyze'])
    p.add_argument('--reader');p.add_argument('--attempt');p.add_argument('--policy',choices=['compact','original'],default='compact')
    a=p.parse_args()
    try:
        if a.action=='worker': raise SystemExit(qa_worker(a.reader,a.attempt,a.policy))
        globals()[a.action]()
    except Exception as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__': main()
