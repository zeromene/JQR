"""Frozen production-quotation mechanism experiment, with exact historical QA reuse."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
import os
import statistics
import subprocess
import time
import traceback
from . import paper_confirmation_v1 as parent
from . import paper_confirmation_runtime_v1 as runtime
from . import quote_retention_mechanism_core_v1 as core

u = parent.u
ROOT = parent.ROOT
BASE = ROOT / 'research/quote_retention_mechanism_v2'
RUN = ROOT / 'runs/quote_retention_mechanism_v2'
COMP = ROOT / 'runs/paper_component_controls_v2'
CONFIG = dict(version='quote-retention-mechanism-v2', sources=1935, budget=512,
    arms=list(core.ARMS), readers=list(parent.CONFIG['readers']), logical_conditions=15480,
    proposer_calls=0, ranking_calls=0, training=False, automatic_expansion=False,
    primary='HotpotQA/Qwen B_protect_fill minus D_reselect_full: >=2 F1 and 95% lower bound >0',
    contrasts=[['B_protect_fill','D_reselect_full'], ['A_jqr','C_reselect_tight'],
               ['A_jqr','B_protect_fill'], ['A_jqr','D_reselect_full']],
    seed=30, bootstrap=10000, wait_timeout=None, wall_timeout=None,
    reader_order=['Phi-4-mini-instruct','Qwen3.5-9B'],
    scope='Post-confirmation fixed-proposal quotation diagnostic; not independent replication or exact-length matching')


def put(path, value):
    runtime.put(path, value)


def state(status, **kwargs):
    row = dict(status=status, time=datetime.now(timezone.utc).isoformat(), **kwargs)
    row.setdefault('expires_at', None)
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / 'status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)
    print(json.dumps(row), flush=True)


def freeze():
    from . import paper_component_controls_v2 as component
    p_sig, c_sig = parent.verify(), component.verify()
    assert (parent.RUN / 'completion.json').exists() and (COMP / 'analysis.json').exists()
    paths = ['reader_memory/quote_retention_mechanism_core_v1.py',
             'reader_memory/quote_retention_mechanism_v2.py',
             'research/quote_retention_mechanism_v2/PROTOCOL.md',
             'tests/test_quote_retention_mechanism_v2.py',
             'tools/quote_retention_mechanism_v2_remote.py']
    inputs = [parent.RUN / x for x in ['plans.json','requests.json','mapping.json']]
    inputs += [COMP / x for x in ['plans.json','requests.json','mapping.json','reuse.json']]
    value = dict(config=CONFIG, parent_signature=p_sig, component_signature=c_sig,
        files={p:u.sha(ROOT / p) for p in paths},
        inputs={str(p.relative_to(ROOT)):u.sha(p) for p in inputs},
        frozen_at=datetime.now(timezone.utc).isoformat())
    put(BASE / 'manifest.json', value)
    state('frozen', signature=u.sha(BASE / 'manifest.json'))


def verify():
    from . import paper_component_controls_v2 as component
    m = u.load(BASE / 'manifest.json')
    assert m['config'] == CONFIG
    assert parent.verify() == m['parent_signature']
    cm = u.load(ROOT/'research/paper_component_controls_v2/manifest.json')
    assert cm['config']==component.CONFIG and cm['parent_signature']==m['parent_signature']
    for path,digest in cm['files'].items():
        assert u.sha(ROOT/path)==digest,path
    for name,digest in cm['input_hashes'].items():
        assert u.sha(parent.RUN/name)==digest,name
    assert u.sha(ROOT/'research/paper_component_controls_v2/manifest.json')==m['component_signature']
    for path, digest in (m['files'] | m['inputs']).items():
        assert u.sha(ROOT / path) == digest, path
    material = RUN / 'material_manifest.json'
    if material.exists():
        for path, digest in u.load(material)['files'].items():
            assert u.sha(ROOT / path) == digest, path
    return u.sha(BASE / 'manifest.json')


def plan_ranges(source, plan):
    from .relation_preserving_pilot_v1 import merge_ranges
    if 'selected_ranges' in plan:
        return plan['selected_ranges']
    rr = {}
    for uid in plan.get('selected_spans', []):
        rr = core.extend(rr, core.span_for(source, uid))
    by = {p['source_id']:p for p in source['paragraphs']}
    return {pid:merge_ranges(by[pid]['text'], spans) for pid,spans in rr.items()}


def prepare():
    sid = verify()
    tok = runtime.tokenizer('Qwen3.5-9B')
    phi = runtime.tokenizer('Phi-4-mini-instruct')
    class Count:
        exact = True
        def count(self, text):
            return len(tok.encode(text, add_special_tokens=False))
    counter = Count()
    oldplans = u.load(parent.RUN / 'plans.json')
    original = {(p['question_id'],p['reader']):p for p in oldplans
                if p['study']=='complete' and p['method']=='quote_anchor'}
    component_plans = u.load(COMP / 'plans.json')
    previous_fill = {(p['question_id'],p['reader']):p for p in component_plans if p['method']=='quote_cover'}
    oldrequests = u.load(parent.RUN / 'requests.json')
    comprequests = u.load(COMP / 'requests.json')
    compreuse = u.load(COMP / 'reuse.json')
    cache_signatures = {u.sha(parent.BASE / 'manifest.json'), u.sha(ROOT / 'research/paper_component_controls_v2/manifest.json')}
    plans, mapping, requests, reuse, reuse_hashes, audit, input_hashes = [], [], {}, {}, {}, [], {}
    for i, x in enumerate(parent.sources(), 1):
        started = time.perf_counter()
        s, d = x['source'], x['dataset']
        q = s['question_id']
        rank = runtime.cached('rank', q, s)['value']
        ret = runtime.cached('retrieve', q, s)['value']
        proposal = runtime.cached('proposal', [q,'role'], s)['value']['parsed']
        for phase, identity in [('rank',q),('retrieve',q),('proposal',[q,'role'])]:
            path = runtime.result_path(phase, identity)
            input_hashes[str(path.relative_to(ROOT))] = u.sha(path)
        global_plan, fallback = parent.old.hp.short.short_plans(
            s, rank['scores'], [str(k) for k in ret['paths'][0]], counter, 512, 512)
        if global_plan.get('selection_error') or fallback.get('selection_error'):
            a, b = deepcopy(fallback), deepcopy(fallback)
            trace = dict(fallback='baseline_selection_failure')
        else:
            a, trace = parent.old.hp.anchored.quote_plan(s,rank['scores'],global_plan,fallback,proposal,counter,'quote_anchor')
            b, _ = parent.old.hp.anchored.quote_plan(s,rank['scores'],global_plan,fallback,proposal,counter,'quote_cover')
        for reader in CONFIG['readers']:
            assert a['evidence_text'] == original[q,reader]['evidence_text'], ('A reproduction',q)
            assert b['evidence_text'] == previous_fill[q,reader]['evidence_text'], ('B reproduction',q)
        controls, info = core.make_controls(s,rank['scores'],proposal,fallback,a,b,trace,
                                           global_plan['evidence_tokens'],counter.count)
        audit_row = dict(question_id=q,dataset=d,L=global_plan['evidence_tokens'],B=512,
                         proposal_hash=u.fp(proposal),source_hash=u.fp(s),**info,arms={})
        for arm, value in controls.items():
            ranges = plan_ranges(s,value)
            retain = core.retention(proposal,ranges)
            cap = a['evidence_tokens'] if info['admitted'] and arm==core.ARMS[2] else global_plan['evidence_tokens']
            assert value['evidence_tokens'] == counter.count(value['evidence_text']) <= 512
            if info['admitted']:
                assert value['evidence_tokens'] <= cap
            audit_row['arms'][arm] = dict(evidence_tokens=value['evidence_tokens'],cap=cap,
                text_hash=u.fp(value['evidence_text']),**retain)
            for reader in CONFIG['readers']:
                t = tok if reader=='Qwen3.5-9B' else phi
                p = dict(dataset=d,question_id=q,reader=reader,method=arm,budget=512,
                    effective_budget=global_plan['evidence_tokens'],selection_cap=cap,
                    evidence_text=value['evidence_text'],evidence_tokens=value['evidence_tokens'],
                    actual_reader_evidence_tokens=len(t.encode(value['evidence_text'],add_special_tokens=False)),
                    status=original[q,reader]['status'], admitted=info['admitted'],fallback=info['fallback'],**retain)
                key = None
                if p['status']=='ready':
                    req = parent.qa_request(p,s,reader)
                    key = req['key']
                    assert key == u.fp([reader,req['messages'],req['max_output']])
                    p['prompt_tokens'] = runtime.prompt_tokens(t,req['messages'])
                    assert p['prompt_tokens'] <= parent.CONFIG['reader_max_input']
                    if key not in requests:
                        dest = None
                        if key in oldrequests:
                            oldreq = oldrequests[key]
                            dest = parent.RUN / 'outputs' / (key+'.json')
                        elif key in comprequests:
                            oldreq = comprequests[key]
                            dest = ROOT / compreuse[key] if key in compreuse else COMP / 'outputs' / (key+'.json')
                        if dest is not None:
                            assert (req['messages'],reader,req['max_output']) == (oldreq['messages'],oldreq['reader'],oldreq['max_output'])
                            out = u.load(dest)
                            assert out['key'] == out['request_identity'] == key
                            assert out['signature'] in cache_signatures
                            assert out['result']['usage']['prompt_tokens'] == p['prompt_tokens']
                            reuse[key] = str(dest.relative_to(ROOT))
                            reuse_hashes[reuse[key]] = u.sha(dest)
                        requests[key] = req
                    if arm in core.ARMS[:2]:
                        assert key in reuse, 'A/B must reuse exact original requests'
                plans.append(p)
                mapping.append(key)
        audit_row['seconds'] = time.perf_counter()-started
        audit.append(audit_row)
        if i%50==0 or i==1935:
            state('preparing',completed=i,expected=1935,new_unique_requests=len(requests)-len(reuse))
    assert len(plans) == len(mapping) == CONFIG['logical_conditions']
    assert len({(p['question_id'],p['reader'],p['method']) for p in plans}) == len(plans)
    summaries = []
    for d in ('hotpot','musique'):
        rows = [r for r in audit if r['dataset']==d]
        admitted = [r for r in rows if r['admitted']]
        for arm in core.ARMS:
            rr = [r['arms'][arm] for r in admitted]
            summaries.append(dict(dataset=d,method=arm,n=len(rows),admitted=len(admitted),
                fallback_reasons=dict(Counter(r['fallback'] for r in rows if not r['admitted'])),
                mean_evidence_tokens=statistics.mean(r['arms'][arm]['evidence_tokens'] for r in rows),
                admitted_quote_retention=statistics.mean(r['quote_retention'] for r in rr) if rr else None,
                admitted_all_quotes_retained=sum(r['all_quotes_retained'] for r in rr),
                changed_vs_A=sum(r['arms'][arm]['text_hash']!=r['arms'][core.ARMS[0]]['text_hash'] for r in rows),
                changed_vs_B=sum(r['arms'][arm]['text_hash']!=r['arms'][core.ARMS[1]]['text_hash'] for r in rows),
                mean_length_minus_A=statistics.mean(r['arms'][arm]['evidence_tokens']-r['arms'][core.ARMS[0]]['evidence_tokens'] for r in rows),
                mean_length_minus_B=statistics.mean(r['arms'][arm]['evidence_tokens']-r['arms'][core.ARMS[1]]['evidence_tokens'] for r in rows)))
    losses = sum(not r['arms'][arm]['all_quotes_retained'] for r in audit if r['admitted'] for arm in core.ARMS[2:])
    result = dict(signature=sid,logical_conditions=len(plans),unique_requests=len(requests),
        exact_reused_requests=len(reuse),new_requests_by_reader={reader:sum(
            key not in reuse and req['reader']==reader for key,req in requests.items()) for reader in CONFIG['readers']},
        original_reproductions=1935,fill_reproductions=1935,references_read=False,
        conditions_with_member_loss=losses,informative=losses>0,summaries=summaries)
    for name, value in [('plans',plans),('mapping',mapping),('requests',requests),('reuse',reuse),
                        ('reuse_hashes',reuse_hashes),('input_hashes',input_hashes),('construction_audit',audit),('preparation',result)]:
        put(RUN / (name+'.json'), value)
    names = ['plans','mapping','requests','reuse','reuse_hashes','input_hashes','construction_audit','preparation']
    put(RUN / 'material_manifest.json',dict(signature=sid,files={str((RUN/(n+'.json')).relative_to(ROOT)):u.sha(RUN/(n+'.json')) for n in names}))
    state('prepared',**result)


def qa_worker(reader, attempt):
    # Reuse the exact already validated BF16/SDPA worker and resource admission behavior.
    from . import paper_component_controls_v2 as component
    verify()
    for relative,digest in u.load(RUN/'reuse_hashes.json').items():
        assert u.sha(ROOT/relative)==digest, relative
    for relative,digest in u.load(RUN/'input_hashes.json').items():
        assert u.sha(ROOT/relative)==digest, relative
    component.RUN = RUN
    component.verify = verify
    component.state = state
    return component.qa_worker(reader,attempt)


def analyze():
    from . import paper_confirmation_analysis_v1 as analysis
    from .evidence_acquisition_v2 import parse_answer
    sid = verify()
    requests, reuse = u.load(RUN/'requests.json'), u.load(RUN/'reuse.json')
    assert all(key in reuse or (RUN/'outputs'/(key+'.json')).exists() for key in requests)
    outputs = {}
    reuse_hashes = u.load(RUN/'reuse_hashes.json')
    for key, req in requests.items():
        path = ROOT/reuse[key] if key in reuse else RUN/'outputs'/(key+'.json')
        out = u.load(path)
        assert out['key']==out['request_identity']==key==u.fp([req['reader'],req['messages'],req['max_output']])
        if key in reuse:
            assert u.sha(path)==reuse_hashes[reuse[key]]
        else:
            assert out['signature']==sid
        outputs[key]=out['result']
    # Read answers only after the entire request matrix is complete.
    refs = analysis.references()
    metric = parent.old.hp.official_metrics()
    plans, mapping = u.load(RUN/'plans.json'), u.load(RUN/'mapping.json')
    metadata = {('hotpot:'+r['native_id'] if r['dataset']=='hotpot' else r['native_id']+':multi'):r
                for r in u.load(parent.BASE/'allocation.json')['selected']}
    records = []
    for p, key in zip(plans,mapping):
        f1=em=0.; hypothesis='';status=p['status'];finish=None;usage={}
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
        records.append({k:v for k,v in p.items() if k!='evidence_text'}|dict(
            key=key,f1=f1,em=em,hypothesis=hypothesis,status=status,finish_reason=finish,usage=usage,reused=key in reuse))
    groups, comparisons = [], []
    index = {(r['dataset'],r['reader'],r['question_id'],r['method']):r for r in records}
    for d in ('hotpot','musique'):
        for reader in CONFIG['readers']:
            for scope in ('all','originally_admitted'):
                ids=sorted(r['question_id'] for r in records if r['dataset']==d and r['reader']==reader
                           and r['method']==core.ARMS[0] and (scope=='all' or r['admitted']))
                if not ids:continue
                clusters={q:metadata[q]['component_id' if d=='hotpot' else 'support_document_component'] for q in ids}
                for arm in core.ARMS:
                    rr=[index[d,reader,q,arm] for q in ids]
                    eligible=[r for r in rr if r['quote_retention'] is not None]
                    lengths=[r['evidence_tokens'] for r in rr]
                    groups.append(dict(dataset=d,reader=reader,scope=scope,method=arm,n=len(rr),
                        f1=100*statistics.mean(r['f1'] for r in rr),em=100*statistics.mean(r['em'] for r in rr),
                        evidence_tokens=statistics.mean(lengths),median_evidence_tokens=statistics.median(lengths),
                        reader_evidence_tokens=statistics.mean(r['actual_reader_evidence_tokens'] for r in rr),
                        prompt_tokens=statistics.mean(r.get('prompt_tokens',0) for r in rr),
                        quote_retention=statistics.mean(r['quote_retention'] for r in eligible) if eligible else None,
                        all_quotes_retained=sum(r['all_quotes_retained'] for r in eligible),retention_denominator=len(eligible),
                        statuses=dict(Counter(r['status'] for r in rr)),finish_reasons=dict(Counter(r['finish_reason'] for r in rr))))
                for method,control in CONFIG['contrasts']:
                    paired={metric_name:analysis.paired({q:index[d,reader,q,method][metric_name]-index[d,reader,q,control][metric_name] for q in ids},clusters) for metric_name in ('f1','em')}
                    comparisons.append(dict(dataset=d,reader=reader,scope=scope,method=method,control=control,
                        **paired,mean_length_difference=statistics.mean(index[d,reader,q,method]['evidence_tokens']-index[d,reader,q,control]['evidence_tokens'] for q in ids)))
    primary=next(c for c in comparisons if c['dataset']=='hotpot' and c['reader']=='Qwen3.5-9B'
                 and c['scope']=='all' and [c['method'],c['control']]==CONFIG['contrasts'][0])
    result=dict(signature=sid,complete=True,groups=groups,comparisons=comparisons,records=records,
        primary=primary,primary_criterion_met=primary['f1']['delta_points']>=2 and primary['f1']['ci95_points'][0]>0,
        scope=CONFIG['scope'],original_confirmation_decision_unchanged=True,automatic_expansion=False)
    put(RUN/'analysis.json',result)
    lines=['# 正式引文级保留诊断','',CONFIG['scope'],'',
           '|数据|读者|比较|ΔF1|95%区间|ΔEM|平均长度差|','|---|---|---|---:|---|---:|---:|']
    for c in comparisons:
        if c['scope']=='all':
            lines.append(f"|{c['dataset']}|{c['reader']}|{c['method']} − {c['control']}|{c['f1']['delta_points']:+.2f}|{c['f1']['ci95_points']}|{c['em']['delta_points']:+.2f}|{c['mean_length_difference']:+.1f}|")
    lines += ['', '同预算上限不是精确等长；主结果保留全部回退和失败。新结果不改变原句子级等长诊断判定。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state('complete',logical_conditions=len(records),primary_criterion_met=result['primary_criterion_met'])


def run():
    import fcntl
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify()
        prep=u.load(RUN/'preparation.json')
        assert prep['informative'], 'No member deletion: report non-discriminating control without new inference'
        attempt=len(list((RUN/'attempts').glob('*')))
        for reader in CONFIG['reader_order']:
            while not (RUN/'completed_readers'/(reader+'.json')).exists():
                if not prep['new_requests_by_reader'][reader]:
                    put(RUN/'completed_readers'/(reader+'.json'),dict(reader=reader,all_reused=True));break
                attempt+=1;name=f'{attempt:06d}';folder=RUN/'attempts'/name;folder.mkdir(parents=True)
                with (folder/'worker.log').open('x') as log:
                    child=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.quote_retention_mechanism_v2',
                        'worker','--reader',reader,'--attempt',name],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    (RUN/'active_worker.json').write_text(json.dumps(dict(pid=child.pid,reader=reader,attempt=name)),encoding='utf-8')
                    code=child.wait()
                if code==75:continue
                assert code==0,'Worker failed: '+str(folder/'worker.log')
        if not (RUN/'analysis.json').exists():analyze()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','prepare','run','worker','analyze'])
    parser.add_argument('--reader');parser.add_argument('--attempt')
    args=parser.parse_args()
    try:
        if args.action=='worker':raise SystemExit(qa_worker(args.reader,args.attempt))
        globals()[args.action]()
    except Exception as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
