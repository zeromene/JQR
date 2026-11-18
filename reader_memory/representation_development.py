"""Frozen source-only semantic writing pilot; no task answering or selector training."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import re

from .choice_protocol_audit import save
from .diagnostics import sha256
from .four_reader_probe import READERS,probe_assets
from .independent_pack import normalize
from .native_choice_pilot import read
from .personamem2 import REVISION,load_history
from .schema import Turn,fingerprint,read_jsonl
from .source_records import (parse_records,parse_review,render_records,review_sample,reviewer_messages,
                            select_units,source_text,writer_messages)

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'research/representation_development/v1'
OUTPUT=ROOT/'runs/representation_development_v1'
ALLOCATION=ROOT/'runs/research_viability_v1/source_allocation.json'
WRITER='Qwen3.5-4B'
REVIEWER='Qwen3.5-9B'
WRITER_CAP=1024
REVIEWER_CAP=768


def prompt_entry(counter,messages):
    text=counter.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=counter.tokenizer.encode(text,add_special_tokens=False)
    if len(ids)>12288: raise ValueError('Source-record prompt exceeds context')
    return {'messages_hash':fingerprint(messages),'token_ids_hash':fingerprint(ids),'tokens':len(ids)}


def allocated_sources():
    a=read(ALLOCATION);split=read(ROOT/'research/personamem_v2/v1/split.json')
    if a['existing_split_sha256']!=sha256(ROOT/'research/personamem_v2/v1/split.json'):
        raise ValueError('Allocated source split changed')
    pilot=[g for g in a['source_allocations'] if g['role']=='representation_development']
    confirm=[g for g in a['source_allocations'] if g['role']=='prospective_confirmation']
    groups={g['source_group'] for g in pilot}
    if (len(pilot)!=len(groups) or len(groups)!=16 or not groups<=set(split['train_groups'])
        or groups&(set(split['probe_groups'])|set(split['reserved_internal_validation_groups']))
        or {q['history_path'] for g in pilot for q in g['queries']}&{q['history_path'] for g in confirm for q in g['queries']}):
        raise ValueError('Invalid development source isolation')
    return pilot


def load_sources():
    source=read(OUTPUT/'source_manifest.json');pilot=allocated_sources()
    if (source['status']!='download-completed' or source['revision']!=REVISION or source['allocation_sha256']!=sha256(ALLOCATION)
        or source['existing_source_manifest_sha256']!=sha256(ROOT/'data/personamem_v2/source_manifest.json')):
        raise ValueError('Source identity changed')
    histories={};audits=[]
    for g in pilot:
        names={q['history_path'] for q in g['queries']};pids={q['persona_id'] for q in g['queries']}
        if len(names)!=1 or len(pids)!=1: raise ValueError('Unexpected allocation history join')
        name=names.pop();pid=pids.pop();path=OUTPUT/'sources'/name;entry=source['files'][name]
        if sha256(path)!=entry['sha256'] or path.stat().st_size!=entry['size']: raise ValueError('Source bytes changed')
        value=read(path)
        if str(value['metadata']['persona_id'])!=pid: raise ValueError('Wrong history persona')
        messages=value['chat_history']
        if any(m['role'] not in {'user','assistant','system'} or not isinstance(m['content'],str) for m in messages):
            raise ValueError('Unexpected source content')
        turns=[Turn(f'p{pid}:m{i:04d}:t0000',m['role'],m['content'],'') for i,m in enumerate(messages) if m['role']!='system']
        forbidden={q['actual_query_fingerprint'] for q in g['queries']}
        if any(fingerprint(normalize(t.content)) in forbidden for t in turns if t.role=='user'):
            raise ValueError('Allocated current query duplicates historical source')
        histories[g['source_group']]=(turns,pid)
        audits.append({'source_group':g['source_group'],'persona_id':pid,'history_path':name,
            'history_sha256':sha256(path),'history_body_fingerprint':fingerprint([(t.role,t.content) for t in turns]),
            'retained_turns':len(turns),'removed_system_messages':sum(m['role']=='system' for m in messages)})
    if set(source['files'])!={v['history_path'] for v in audits}: raise ValueError('Unallocated source downloaded')
    if len({a['history_body_fingerprint'] for a in audits})!=16: raise ValueError('Duplicate development histories')
    return histories,audits


def overlap_audit(histories,audits):
    prior=read(ROOT/'research/four_reader_probe/v1/queries.json');seen=set();old_bodies=set();old_messages=set()
    for q in prior:
        if q['history_path'] in seen: continue
        seen.add(q['history_path']);turns,_=load_history(q)
        old_bodies.add(fingerprint([(t.role,t.content) for t in turns]))
        old_messages.update(fingerprint(normalize(t.content)) for t in turns if t.role=='user' and len(normalize(t.content))>=200)
    # Read only context lines named by the already exposed v1 experiment.
    prefixes={}
    for name in ('research/native_choice/v1/queries.json','research/window_utility/v1/queries.json'):
        for q in read(ROOT/name):
            if q['source_group'].rsplit('-',1)[-1] in {'8','15','9','12'}: raise ValueError('Reserved v1 context entered overlap audit')
            prefixes[q['context_id']]=max(prefixes.get(q['context_id'],0),q['prefix_end_index'])
    old_v1=set()
    with (ROOT/'data/personamem_v1/shared_contexts_32k.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            match=re.match(r'\s*\{\s*"([^\"]+)"\s*:',line)
            if match is None or match[1] not in prefixes: continue
            context=json.loads(line)[match[1]][:prefixes[match[1]]]
            old_v1.update(fingerprint(normalize(t['content'])) for t in context
                if t['role']=='user' and len(normalize(t['content']))>=200)
    values=[]
    for a in audits:
        turns,_=histories[a['source_group']]
        long={fingerprint(normalize(t.content)) for t in turns if t.role=='user' and len(normalize(t.content))>=200}
        if a['history_body_fingerprint'] in old_bodies: raise ValueError('New history duplicates an exposed v2 history')
        values.append({'source_group':a['source_group'],'distinct_long_user_messages':len(long),
            'exact_long_user_message_overlap_with_exposed_v2':len(long&old_messages),
            'exact_long_user_message_overlap_with_exposed_v1':len(long&old_v1)})
    return {'rows':values,'exposed_v2_histories':len(seen),'exposed_v1_context_prefixes':len(prefixes),
        'old_reserved_v1_bodies_used':False,'exact_text_only':True,'persona_identity_across_datasets_guaranteed':False,
        'selection_changed_by_overlap_or_scores':False}


def prepare(counter):
    if STUDY.exists(): raise FileExistsError('Preserve frozen representation-development inputs')
    if not counter.exact: raise ValueError('Actual writer tokenizer required')
    histories,audits=load_sources();units=[];sampling=[]
    for group,(turns,pid) in histories.items():
        selected,audit=select_units(turns,group,pid,counter);units.extend(selected);sampling.append({'source_group':group,**audit})
    review_ids=review_sample(units)
    if len(units)!=64 or len(set(review_ids))!=32: raise ValueError('Incorrect pilot sample size')
    prompts={u['unit_id']:prompt_entry(counter,writer_messages(u)) for u in units}
    source_audit={'histories':audits,'sampling':sampling,'overlap':overlap_audit(histories,audits),
        'source_only_sampling':True,'task_answers_generated':False,'reserved_validation_used':False}
    STUDY.mkdir(parents=True)
    for name,value in (('units.json',units),('review_unit_ids.json',review_ids),('writer_prompts.json',prompts),('source_audit.json',source_audit)):
        save(STUDY/name,value)
    dependencies=('research/REPRESENTATION_DEVELOPMENT_PLAN.md','reader_memory/source_records.py',
        'reader_memory/representation_development.py','tools/audit_representation_development.py',
        'tools/run_representation_development.py','reader_memory/backend.py','reader_memory/schema.py',
        'reader_memory/personamem2.py','runs/research_viability_v1/source_allocation.json',
        'runs/representation_development_v1/source_manifest.json','research/personamem_v2/v1/split.json',
        'tools/prepare_personamem2.py','research/sources/expanded_data_20261026/ImplicitPersona.metadata.json')
    m={'version':'representation-development-v1','frozen_at':datetime.now(timezone.utc).isoformat(),
        'file_hashes':{p:sha256(ROOT/p) for p in dependencies},
        'input_hashes':{p:sha256(STUDY/p) for p in ('units.json','review_unit_ids.json','writer_prompts.json','source_audit.json')},
        'source_file_hashes':{'runs/representation_development_v1/sources/'+a['history_path']:a['history_sha256'] for a in audits},
        'writer':WRITER,'reviewer':REVIEWER,'writer_cap':WRITER_CAP,'reviewer_cap':REVIEWER_CAP,
        'writer_assets':probe_assets(ROOT/'models'/WRITER),'reviewer_assets':probe_assets(ROOT/'models'/REVIEWER),
        'source_groups':16,'writer_units':64,'review_units':32,'writer_input':'historical source unit only',
        'selector_trained':False,'human_gold':False,'formal_benchmark_score':False,'reserved_validation_used':False,
        'confirmation_sources_read':False}
    save(STUDY/'manifest.json',m);print('RECORD_PILOT_FROZEN',64,32,flush=True)


def load():
    m=read(STUDY/'manifest.json')
    if (any(sha256(ROOT/p)!=v for p,v in m['file_hashes'].items())
        or any(sha256(STUDY/p)!=v for p,v in m['input_hashes'].items())
        or any(sha256(ROOT/p)!=v for p,v in m['source_file_hashes'].items())):
        raise ValueError('Frozen record pilot changed')
    return m,read(STUDY/'units.json'),read(STUDY/'writer_prompts.json')


def decode_writer(response,unit):
    if response['usage']['completion_tokens']>=WRITER_CAP: return {'status':'output-cap','derived':None,'error':'writer output cap reached'}
    try: return {'status':'accepted','derived':parse_records(response['text'],unit),'error':None}
    except (ValueError,TypeError,KeyError) as exc: return {'status':'mechanical-rejection','derived':None,'error':repr(exc)}


def verify_writer(rows,m,units,prompts):
    byid={u['unit_id']:u for u in units};seen=set()
    for row in rows:
        uid=row['unit_id']
        if uid in seen or uid not in byid: raise ValueError('Duplicate/unknown writer output')
        unit=byid[uid];entry=prompts[uid];usage=row['response']['usage']
        expected={'unit_id':uid,'study_signature':fingerprint(m),'unit_hash':fingerprint(unit),
            'prompt_entry':entry,'response':row['response'],**decode_writer(row['response'],unit)}
        if (row!=expected or entry['messages_hash']!=fingerprint(writer_messages(unit))
            or usage['prompt_tokens']!=entry['tokens'] or not math.isfinite(usage['elapsed_seconds'])
            or usage['elapsed_seconds']<0 or not 0<=usage['completion_tokens']<=WRITER_CAP):
            raise ValueError('Writer output/derived provenance changed')
        seen.add(uid)
    return seen


def run_writer(backend,counter):
    m,units,prompts=load()
    if backend.model_name!=WRITER or probe_assets(backend.path)!=m['writer_assets']: raise ValueError('Wrong writer')
    path=OUTPUT/'writer.jsonl';rows=list(read_jsonl(path)) if path.exists() else []
    done=verify_writer(rows,m,units,prompts)
    with path.open('a',encoding='utf-8',newline='\n') as stream:
        for unit in units:
            uid=unit['unit_id']
            if uid in done: continue
            prompt=writer_messages(unit);entry=prompt_entry(counter,prompt)
            if entry!=prompts[uid]: raise ValueError('Writer actual input IDs changed')
            response=backend.complete(prompt,max_tokens=WRITER_CAP)
            row={'unit_id':uid,'study_signature':fingerprint(m),'unit_hash':fingerprint(unit),'prompt_entry':entry,
                 'response':response,**decode_writer(response,unit)}
            verify_writer([row],m,units,prompts)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();done.add(uid)
            save(OUTPUT/'status.json',{'status':'writing','units':len(done),'expected_units':64})
            print('RECORD_WRITTEN',len(done),'/',64,row['status'],flush=True)


def freeze_review(counter):
    m,units,prompts=load();rows=list(read_jsonl(OUTPUT/'writer.jsonl'))
    if len(verify_writer(rows,m,units,prompts))!=64: raise ValueError('Complete writer collection first')
    byid={u['unit_id']:u for u in units};written={r['unit_id']:r for r in rows};entries={}
    for uid in read(STUDY/'review_unit_ids.json'):
        row=written[uid]
        entries[uid]=(prompt_entry(counter,reviewer_messages(byid[uid],row['derived']['records'])) if row['status']=='accepted' else None)
    value={'study_signature':fingerprint(m),'writer_sha256':sha256(OUTPUT/'writer.jsonl'),
        'reviewer_assets':probe_assets(ROOT/'models'/REVIEWER),'entries':entries}
    target=OUTPUT/'review_inputs.json'
    if target.exists() and read(target)!=value: raise ValueError('Preserve frozen reviewer inputs')
    if value['reviewer_assets']!=m['reviewer_assets']: raise ValueError('Reviewer assets changed')
    if not target.exists(): save(target,value)
    print('RECORD_REVIEW_INPUTS_FROZEN',len(entries),flush=True)


def decode_review(response,records):
    if response['usage']['completion_tokens']>=REVIEWER_CAP: return {'status':'output-cap','review':None,'error':'reviewer output cap reached'}
    try: return {'status':'reviewed','review':parse_review(response['text'],records),'error':None}
    except (ValueError,KeyError,TypeError) as exc: return {'status':'review-format-rejection','review':None,'error':repr(exc)}


def verify_review(rows,m,units,written,inputs):
    byid={u['unit_id']:u for u in units};seen=set()
    if inputs['study_signature']!=fingerprint(m) or inputs['writer_sha256']!=sha256(OUTPUT/'writer.jsonl'):
        raise ValueError('Reviewer no longer uses the complete frozen writer outputs')
    for row in rows:
        uid=row['unit_id']
        if uid in seen or uid not in inputs['entries']: raise ValueError('Duplicate/unknown reviewer output')
        base={'unit_id':uid,'study_signature':fingerprint(m),'writer_row_hash':fingerprint(written[uid]),
              'review_inputs_sha256':sha256(OUTPUT/'review_inputs.json'),'prompt_entry':inputs['entries'][uid]}
        if written[uid]['status']!='accepted':
            expected={**base,'response':None,'status':'skipped-writer-unavailable','review':None,'error':written[uid]['error']}
        else:
            entry=inputs['entries'][uid];response=row['response'];usage=response['usage']
            if (entry['messages_hash']!=fingerprint(reviewer_messages(byid[uid],written[uid]['derived']['records']))
                or usage['prompt_tokens']!=entry['tokens'] or not math.isfinite(usage['elapsed_seconds'])
                or usage['elapsed_seconds']<0 or not 0<=usage['completion_tokens']<=REVIEWER_CAP):
                raise ValueError('Reviewer actual input/cost changed')
            expected={**base,'response':response,**decode_review(response,written[uid]['derived']['records'])}
        if row!=expected: raise ValueError('Reviewer provenance/derivation changed')
        seen.add(uid)
    return seen


def run_review(backend,counter):
    m,units,prompts=load();writer=list(read_jsonl(OUTPUT/'writer.jsonl'));verify_writer(writer,m,units,prompts)
    written={r['unit_id']:r for r in writer};inputs=read(OUTPUT/'review_inputs.json');byid={u['unit_id']:u for u in units}
    if backend.model_name!=REVIEWER or probe_assets(backend.path)!=m['reviewer_assets']: raise ValueError('Wrong reviewer')
    path=OUTPUT/'reviewer.jsonl';rows=list(read_jsonl(path)) if path.exists() else []
    done=verify_review(rows,m,units,written,inputs)
    with path.open('a',encoding='utf-8',newline='\n') as stream:
        for uid in read(STUDY/'review_unit_ids.json'):
            if uid in done: continue
            row={'unit_id':uid,'study_signature':fingerprint(m),'writer_row_hash':fingerprint(written[uid]),
                 'review_inputs_sha256':sha256(OUTPUT/'review_inputs.json'),'prompt_entry':inputs['entries'][uid]}
            if written[uid]['status']!='accepted':
                row.update(response=None,status='skipped-writer-unavailable',review=None,error=written[uid]['error'])
            else:
                records=written[uid]['derived']['records'];prompt=reviewer_messages(byid[uid],records)
                if prompt_entry(counter,prompt)!=inputs['entries'][uid]: raise ValueError('Reviewer actual token IDs changed')
                response=backend.complete(prompt,max_tokens=REVIEWER_CAP)
                row.update(response=response,**decode_review(response,records))
            verify_review([row],m,units,written,inputs)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();done.add(uid)
            save(OUTPUT/'status.json',{'status':'ai-source-pre-review','units':len(done),'expected_units':32})
            print('RECORD_REVIEWED',len(done),'/',32,row['status'],flush=True)


def report(counters):
    m,units,prompts=load();writer=list(read_jsonl(OUTPUT/'writer.jsonl'));reviewer=list(read_jsonl(OUTPUT/'reviewer.jsonl'))
    if len(verify_writer(writer,m,units,prompts))!=64: raise ValueError('Incomplete writer matrix')
    written={r['unit_id']:r for r in writer};inputs=read(OUTPUT/'review_inputs.json')
    if len(verify_review(reviewer,m,units,written,inputs))!=32: raise ValueError('Incomplete review matrix')
    valid=[r for r in writer if r['status']=='accepted'];records=[v for r in valid for v in r['derived']['records']]
    judged=[v for r in reviewer if r['status']=='reviewed' for v in r['review']['reviews']]
    byid={u['unit_id']:u for u in units};lengths=[];human=[]
    for row in valid:
        uid=row['unit_id'];unit=byid[uid];rs=row['derived']['records']
        texts={'raw':source_text(unit['spans']),'table':render_records(rs,'table'),'compact':render_records(rs,'compact')}
        lengths.append({'unit_id':uid,'records':len(rs),'tokens_by_reader':{reader:{a:c.count(t) for a,t in texts.items()} for reader,c in counters.items()}})
    for uid in read(STUDY/'review_unit_ids.json'):
        row=written[uid]
        human.append({'case_id':'blind-'+fingerprint(['source-record-human-v1',uid])[:20],
            'source_group':byid[uid]['source_group'],'source_unit':byid[uid]['spans'],
            'candidate_records':row['derived']['records'] if row['status']=='accepted' else [],
            'writer_available':row['status']=='accepted','all_claims_supported':None,'attribution_correct':None,
            'conditions_and_state_preserved':None,'important_omissions':None,'notes':None})
    human.sort(key=lambda c:c['case_id'])
    save(OUTPUT/'lengths.json',lengths);save(OUTPUT/'human_review.blank.json',human)
    value={'status':'completed-source-record-quality-pilot','source_groups':16,'writer_units':64,'review_units':32,
        'writer_status_counts':dict(Counter(r['status'] for r in writer)),
        'accepted_empty_units':sum(not r['derived']['records'] for r in valid),'accepted_nonempty_units':sum(bool(r['derived']['records']) for r in valid),
        'writer_parse_or_cap_failures':64-len(valid),'records':len(records),'record_kind_counts':dict(Counter(r['kind'] for r in records)),
        'literal_citations_checked':sum(len(r['evidence']) for r in records),
        'review_status_counts':dict(Counter(r['status'] for r in reviewer)),
        'ai_record_verdict_counts':dict(Counter(v['verdict'] for v in judged)),
        'ai_issue_counts':dict(Counter(x for v in judged for x in v['issues'])),
        'actual_writer_calls':64,'actual_reviewer_calls':sum(r['response'] is not None for r in reviewer),
        'writer_seconds':sum(r['response']['usage']['elapsed_seconds'] for r in writer),
        'reviewer_seconds':sum(r['response']['usage']['elapsed_seconds'] for r in reviewer if r['response'] is not None),
        'writer_completion_tokens':sum(r['response']['usage']['completion_tokens'] for r in writer),
        'reviewer_completion_tokens':sum(r['response']['usage']['completion_tokens'] for r in reviewer if r['response'] is not None),
        'mean_tokens_on_accepted_units':{reader:{arm:sum(v['tokens_by_reader'][reader][arm] for v in lengths)/len(lengths) if lengths else None
            for arm in ('raw','table','compact')} for reader in counters},
        'semantic_precision_measured_against_humans':False,'semantic_recall_measured':False,'lossless_compression_established':False,
        'ai_review_is_human_gold':False,'task_accuracy_measured':False,'selector_trained':False,'formal_benchmark_score':False,
        'confirmation_sources_read':False,'reserved_validation_used':False,
        'study_manifest_sha256':sha256(STUDY/'manifest.json'),'writer_sha256':sha256(OUTPUT/'writer.jsonl'),
        'reviewer_sha256':sha256(OUTPUT/'reviewer.jsonl')}
    save(OUTPUT/'report.json',value);print('RECORD_QUALITY_REPORT',json.dumps(value,ensure_ascii=False),flush=True)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','writer','freeze-review','review','report'));args=parser.parse_args()
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    if args.command=='prepare': return prepare(TokenCounter(ROOT/'models'/WRITER))
    if args.command=='freeze-review': return freeze_review(TokenCounter(ROOT/'models'/REVIEWER))
    if args.command=='report': return report({r:TokenCounter(ROOT/'models'/r) for r in READERS})
    from .backend import TransformersBackend
    reader=WRITER if args.command=='writer' else REVIEWER
    backend=TransformersBackend(ROOT/'models'/reader);counter=TokenCounter(ROOT/'models'/reader)
    return run_writer(backend,counter) if args.command=='writer' else run_review(backend,counter)


if __name__=='__main__': main()
