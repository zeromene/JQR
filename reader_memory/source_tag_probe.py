"""Same-source, exact-budget comparison of neutral and source-grounded cues."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from itertools import combinations
import json
import math
from pathlib import Path

from .choice_protocol_audit import save
from .diagnostics import sha256
from .four_reader_probe import READERS,SharedCounter,load as load_original,probe_assets
from .native_choice_pilot import classify,messages,read,record_key,score_row
from .personamem2 import load_history
from .retrieval_answer_probe import prompt_entry
from .schema import fingerprint,read_jsonl
from .shared_retrieval_audit import load as load_retrieval,query_content
from .source_tags import build_memory,render,visible_cues
from .token_windows import token_windows

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'research/source_tag_probe/v1'
OUTPUT=ROOT/'runs/source_tag_probe_v1'
ARMS=('plain','neutral','typed')
BUDGET=2048


def expected(queries):
    return {(r,q['question_id'],a,BUDGET,s) for r in READERS for q in queries for a in ARMS for s in q['shifts']}


def pack(groups,order,memory,counter):
    selected=[]
    for i in order[:24]:
        proposal=[*selected,groups[i]]
        if all(counter.count(render(proposal,memory,a))<=BUDGET for a in ARMS): selected=proposal
    evidence={f'{a}:{BUDGET}':{'text':render(selected,memory,a),
            'tokens_by_reader':counter.costs(render(selected,memory,a)),
            'chunk_ids':[g.group_id for g in selected]} for a in ARMS}
    return evidence,selected


def prepare(counters,study=STUDY):
    study=Path(study)
    if study.exists(): raise FileExistsError('Preserve frozen source cue study')
    common=SharedCounter(counters)
    if set(counters)!=set(READERS) or not all(c.exact for c in counters.values()):
        raise ValueError('Every actual reader tokenizer required')
    original,queries=load_original();retrieval,inputs=load_retrieval();byid={q['question_id']:q for q in inputs}
    histories={};memories={};prepared=[]
    # Complete history memories are written without ever accepting a query.
    for q in queries:
        if q['history_path'] in histories: continue
        turns,audit=load_history(q);memory=build_memory(turns)
        groups=token_windows(turns,common,limit=384)
        histories[q['history_path']]=(groups,memory,audit)
        memories[q['history_path']]={'history_audit':audit,'memory':memory}
    for q in queries:
        groups,memory,audit=histories[q['history_path']];parent=byid[q['question_id']]
        if (any(q[k]!=parent[k] for k in ('question_id','source_group','history_path','persona_id','question'))
            or query_content(q['question'])!=parent['normalized_query']): raise ValueError('Wrong source ranking')
        rank=parent['rankings']['dense'];positions={g.group_id:i for i,g in enumerate(groups)}
        if set(positions)!={x['chunk_id'] for x in rank}: raise ValueError('Inherited source windows changed')
        order=[positions[x['chunk_id']] for x in rank]
        evidence,selected=pack(groups,order,memory,common)
        spans=[s for g in selected for s in g.metadata['spans']];cues=visible_cues(spans,memory)
        prepared.append(({k:q[k] for k in ('question_id','persona_id','source_group','history_path','options','shifts')})|
            {'question':query_content(q['question']),'evidence':evidence,'history_audit':audit,
             'selected_sources':[g.to_dict() for g in selected],'visible_cues':cues,
             'inherited_dense_rank_hash':fingerprint(rank)})
        print('SOURCE_TAG_PREPARED',len(prepared),'/',len(queries),flush=True)
    prompts={};maximum={}
    for reader,c in counters.items():
        if probe_assets(ROOT/'models'/reader)!=original['reader_assets'][reader]: raise ValueError('Reader assets changed')
        labels=[c.tokenizer.encode(x,add_special_tokens=False) for x in 'abcd']
        if any(len(x)!=1 for x in labels) or len({x[0] for x in labels})!=4: raise ValueError('Invalid four choice labels')
        prompts[reader]={};maximum[reader]=0
        for q in prepared:
            entries={}
            for a in ARMS:
                item=q['evidence'][f'{a}:{BUDGET}']
                if c.count(item['text'])!=item['tokens_by_reader'][reader] or item['tokens_by_reader'][reader]>BUDGET:
                    raise ValueError('Cue branch budget exceeded')
                for s in q['shifts']:
                    entry=prompt_entry(c,q,a,BUDGET,s);entries[f'{a}:{BUDGET}:{s}']=entry
                    maximum[reader]=max(maximum[reader],entry['tokens'])
            prompts[reader][q['question_id']]=entries
    study.mkdir(parents=True)
    for name,value in (('queries.json',prepared),('memories.json',memories),('prompts.json',prompts),
                       ('references.private.json',read(ROOT/'research/four_reader_probe/v1/references.private.json'))):
        save(study/name,value)
    paths=('research/SOURCE_TAG_PROBE_PLAN.md','reader_memory/source_tags.py','reader_memory/source_tag_probe.py',
           'tools/audit_source_tag_inputs.py','reader_memory/native_choice_pilot.py','reader_memory/tokenization.py',
           'reader_memory/token_windows.py',
           'reader_memory/retrieval_answer_probe.py','research/four_reader_probe/v1/manifest.json',
           'research/shared_retrieval_audit/v1/manifest.json','research/shared_retrieval_audit/v1/inputs.json')
    m={'version':'source-tag-probe-v1','frozen_at':datetime.now(timezone.utc).isoformat(),
       'file_hashes':{p:sha256(ROOT/p) for p in paths},
       'study_files':{p:sha256(study/p) for p in ('queries.json','memories.json','prompts.json','references.private.json')},
       'reader_assets':original['reader_assets'],'readers':list(READERS),'arms':list(ARMS),'budget':BUDGET,
       'questions':len(prepared),'source_groups':len(memories),'expected_records':len(expected(prepared)),
       'maximum_preflight_prompt_tokens':maximum,'oracle_features_used':False,'reserved_validation_used':False,
       'writer_input':'complete ordered history turns only','retrieval':'unchanged dense ranking; same top24 pool',
       'source_control':'identical chosen spans in all arms/readers; no extra coverage bought with savings',
       'selector_trained':False,'formal_benchmark_score':False}
    save(study/'manifest.json',m);print('SOURCE_TAG_FROZEN',m['expected_records'],flush=True)
    return m


def load(study=STUDY):
    study=Path(study);m=read(study/'manifest.json')
    if any(sha256(ROOT/p)!=h for p,h in m['file_hashes'].items()) or any(sha256(study/p)!=h for p,h in m['study_files'].items()):
        raise ValueError('Frozen source cue study changed')
    load_original();load_retrieval()
    queries=read(study/'queries.json')
    if len(expected(queries))!=m['expected_records']: raise ValueError('Cue matrix changed')
    return m,queries,read(study/'prompts.json')


def verify(rows,queries,m,prompts):
    keys=expected(queries);qmap={q['question_id']:q for q in queries};seen={}
    for row in rows:
        key=record_key(row)
        if key in seen or key not in keys: raise ValueError('Duplicate/unknown cue output')
        r,qid,a,b,s=key;q=qmap[qid];item=q['evidence'][f'{a}:{b}'];entry=prompts[r][qid][f'{a}:{b}:{s}']
        scores=row['restricted_label_logits']
        if (row['signature']!=fingerprint(m) or row['prompt_hash']!=fingerprint(messages(q,a,b,s))
            or row['prompt_hash']!=entry['messages_hash'] or row['prompt_token_ids_hash']!=entry['token_ids_hash']
            or row['usage']['prompt_tokens']!=entry['tokens'] or row['evidence_hash']!=fingerprint(item['text'])
            or row['evidence_tokens']!=item['tokens_by_reader'][r] or row['evidence_tokens']>BUDGET
            or len(scores)!=4 or not all(math.isfinite(x) for x in scores)
            or row['prediction']!='abcd'[max(range(4),key=lambda i:scores[i])]): raise ValueError('Cue output provenance changed')
        if row['prediction_origin']=='cache-identical-prompt':
            prior=seen.get(tuple(row['cached_from_key']))
            if (prior is None or prior['reader']!=r or prior['prompt_hash']!=row['prompt_hash']
                or prior['prompt_token_ids_hash']!=row['prompt_token_ids_hash'] or prior['restricted_label_logits']!=scores
                or row['usage']['elapsed_seconds']!=0): raise ValueError('Invalid cue prompt cache')
        elif row['prediction_origin']!='model-forward' or row['cached_from_key'] is not None: raise ValueError('Unknown output origin')
        seen[key]=row
    return keys


def run(backend,counter,study=STUDY,output=OUTPUT):
    m,queries,prompts=load(study);reader=backend.model_name;output=Path(output)
    if reader not in READERS or not counter.exact or probe_assets(backend.path)!=m['reader_assets'][reader]: raise ValueError('Wrong reader')
    output.mkdir(parents=True,exist_ok=True)
    rows=list(read_jsonl(output/'answers.jsonl')) if (output/'answers.jsonl').exists() else []
    verify(rows,queries,m,prompts);done={record_key(r) for r in rows}
    cache={(r['reader'],r['prompt_hash'],r['prompt_token_ids_hash']):r for r in rows}
    with (output/'answers.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
        for q in queries:
            for arm in ARMS:
                item=q['evidence'][f'{arm}:{BUDGET}']
                if counter.count(item['text'])!=item['tokens_by_reader'][reader]: raise ValueError('Actual cue cost changed')
                for shift in q['shifts']:
                    key=(reader,q['question_id'],arm,BUDGET,shift)
                    if key in done: continue
                    entry=prompt_entry(counter,q,arm,BUDGET,shift)
                    if entry!=prompts[reader][q['question_id']][f'{arm}:{BUDGET}:{shift}']: raise ValueError('Actual cue prompt changed')
                    ck=(reader,entry['messages_hash'],entry['token_ids_hash']);prior=cache.get(ck)
                    result=({'prediction':prior['prediction'],'restricted_label_logits':prior['restricted_label_logits'],
                        'usage':{**prior['usage'],'elapsed_seconds':0.0}} if prior else classify(backend,messages(q,arm,BUDGET,shift)))
                    if result['usage']['prompt_tokens']!=entry['tokens']: raise ValueError('Model prompt length differs')
                    row={'reader':reader,'question_id':q['question_id'],'arm':arm,'budget':BUDGET,'shift':shift,
                         'signature':fingerprint(m),'prompt_hash':entry['messages_hash'],'prompt_token_ids_hash':entry['token_ids_hash'],
                         'evidence_hash':fingerprint(item['text']),'evidence_tokens':item['tokens_by_reader'][reader],
                         'prediction_origin':'cache-identical-prompt' if prior else 'model-forward',
                         'cached_from_key':list(record_key(prior)) if prior else None,**result}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();done.add(key);cache[ck]=row
                    if len(done)%96==0:
                        save(output/'status.json',{'status':'running','reader':reader,'records':len(done),'expected_records':m['expected_records']})
                        print('SOURCE_TAG_PROGRESS',reader,len(done),'/',m['expected_records'],flush=True)
    return report(study,output)


def report(study=STUDY,output=OUTPUT):
    m,queries,prompts=load(study);output=Path(output);rows=list(read_jsonl(output/'answers.jsonl'))
    keys=verify(rows,queries,m,prompts);qmap={q['question_id']:q for q in queries}
    refs={r['question_id']:r for r in read(Path(study)/'references.private.json')}
    scored=[{**r,'source_group':qmap[r['question_id']]['source_group'],
        'correct':score_row(r,qmap[r['question_id']],refs[r['question_id']]),
        'semantic_choice':(r['shift']+'abcd'.index(r['prediction']))%4} for r in rows]
    summaries={}
    for reader in READERS:
        summaries[reader]={}
        for arm in ARMS:
            selected=[r for r in scored if r['reader']==reader and r['arm']==arm];byq=defaultdict(list)
            for r in selected: byq[r['question_id']].append(r)
            complete=[v for v in byq.values() if len(v)==4]
            summaries[reader][arm]={'presentations':len(selected),'correct':sum(r['correct'] for r in selected),
                'accuracy':sum(r['correct'] for r in selected)/len(selected) if selected else None,
                'mean_evidence_tokens':sum(r['evidence_tokens'] for r in selected)/len(selected) if selected else None,
                'semantic_choice_varies_across_orders':sum(len({r['semantic_choice'] for r in v})>1 for v in complete),
                'all_four_orders_correct_questions':sum(all(r['correct'] for r in v) for v in complete),
                'label_histogram':dict(Counter(r['prediction'] for r in selected)),
                'by_position':{str(s):{'presentations':sum(r['shift']==s for r in selected),
                    'correct':sum(r['correct'] and r['shift']==s for r in selected)} for s in range(4)}}
    value={'status':'completed-source-tag-probe' if len(rows)==len(keys) else 'incomplete-source-tag-probe',
        'records':len(rows),'expected_records':len(keys),'questions':len(queries),'source_groups':m['source_groups'],
        'actual_model_forward_calls':sum(r['prediction_origin']=='model-forward' for r in rows),
        'identical_prompt_cache_reuses':sum(r['prediction_origin']=='cache-identical-prompt' for r in rows),
        'model_seconds':sum(r['usage']['elapsed_seconds'] for r in rows),'by_reader':summaries,
        'visible_cue_questions':sum(bool(q['visible_cues']) for q in queries),
        'visible_cue_counts':dict(Counter(c['kind'] for q in queries for c in q['visible_cues'])),
        'formal_benchmark_score':False,'reserved_validation_used':False,'capability_vector_efficacy_established':False}
    save(output/'report.json',value);save(output/'status.json',{k:value[k] for k in ('status','records','expected_records')})
    (output/'scored.jsonl').write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored).encode('utf-8'))
    print('SOURCE_TAG_REPORT',value['status'],len(rows),flush=True);return value


def analyze(study=STUDY,output=OUTPUT):
    result=report(study,output)
    if result['status']!='completed-source-tag-probe': raise ValueError('Complete all readers before analysis')
    import numpy as np
    m,queries,_=load(study);output=Path(output);rows=list(read_jsonl(output/'scored.jsonl'));index={record_key(r):r for r in rows}
    groups=sorted({q['source_group'] for q in queries});draws=np.random.default_rng(20261026).integers(0,len(groups),size=(20000,len(groups)))
    def differences(reader,candidate,baseline):
        return [{'question_id':q['question_id'],'source_group':q['source_group'],
            'delta':sum(int(index[(reader,q['question_id'],candidate,BUDGET,s)]['correct'])-
                int(index[(reader,q['question_id'],baseline,BUDGET,s)]['correct']) for s in q['shifts'])/4} for q in queries]
    def summary(values):
        sums=np.array([sum(v['delta'] for v in values if v['source_group']==g) for g in groups])
        sizes=np.array([sum(v['source_group']==g for v in values) for g in groups])
        lo,hi=np.quantile(sums[draws].sum(axis=1)/sizes[draws].sum(axis=1),[.025,.975])*100
        return {'difference_pp':float(sums.sum()/sizes.sum()*100),'exploratory_group_bootstrap_95ci_pp':[float(lo),float(hi)],
            'positive_questions':sum(v['delta']>0 for v in values),'negative_questions':sum(v['delta']<0 for v in values),
            'unchanged_questions':sum(v['delta']==0 for v in values)}
    comparisons=[];primary={}
    for reader in READERS:
        for candidate,baseline in (('typed','neutral'),('typed','plain'),('neutral','plain')):
            values=differences(reader,candidate,baseline)
            comparisons.append({'reader':reader,'candidate':candidate,'baseline':baseline,'primary':baseline=='neutral',
                **summary(values),'question_differences':values})
            if baseline=='neutral': primary[reader]=values
    interactions=[]
    for a,b in combinations(READERS,2):
        values=[{**v,'delta':v['delta']-w['delta']} for v,w in zip(primary[a],primary[b])]
        interactions.append({'first_reader':a,'second_reader':b,**summary(values)})
    value={'version':'source-tag-analysis-v1','manifest_sha256':sha256(Path(study)/'manifest.json'),
        'answers_sha256':sha256(output/'answers.jsonl'),'paired_comparisons':comparisons,'reader_interactions':interactions,
        'bootstrap':{'resamples':20000,'seed':20261026,'unit':'complete persona source groups','multiple_comparison_correction':False},
        'formal_benchmark_score':False,'reserved_validation_used':False,'conditional_selector_trained':False}
    save(output/'analysis.json',value);print('SOURCE_TAG_ANALYZED',len(comparisons),flush=True);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','run','report','analyze'))
    parser.add_argument('--reader',choices=READERS,default=READERS[0]);args=parser.parse_args()
    if args.command=='report': return report()
    if args.command=='analyze': return analyze()
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    if args.command=='prepare': return prepare({r:TokenCounter(ROOT/'models'/r) for r in READERS})
    from .backend import TransformersBackend
    model=ROOT/'models'/args.reader;return run(TransformersBackend(model),TokenCounter(model))


if __name__=='__main__': main()
