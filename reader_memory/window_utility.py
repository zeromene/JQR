"""Official-key window interventions on new questions from exposed development personas."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from .diagnostics import sha256
from .independent_pack import source_rows, parse_options, normalize
from .native_choice_pilot import READERS, SOURCES as PRIOR_SOURCES, classify, displayed_options, messages, pack, read, record_key, score_row
from .natural_repair import assets
from .retrieval import BM25
from .schema import Turn, fingerprint, read_jsonl, write_json
from .token_windows import token_windows
from .update_memory import group_order, intervention_plan, link_target, variants

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'research/window_utility/v1'
OUTPUT = ROOT/'runs/window_utility_v1'
VARIANTS = ('anchor','bundle_raw','bundle_lead','bundle_guard')
CONDITIONS = (('global_none',0),('user_bm25',1024),('user_bm25',2048),
              *((a,b) for b in (1024,2048) for a in ('without',*VARIANTS)))


def hashes():
    return {n:sha256(ROOT/'reader_memory'/n) for n in (*PRIOR_SOURCES,'update_memory.py','window_utility.py')}


def select_new(rows, personas, used_ids, per_type=6):
    pool=[r for r in rows if r['persona_id'] in personas and r['question_id'] not in used_ids]
    selected, counts=[],Counter()
    for kind in sorted({r['question_type'] for r in pool}):
        candidates=[r for r in pool if r['question_type']==kind]
        for _ in range(min(per_type,len(candidates))):
            chosen=min(candidates,key=lambda r:(counts[r['persona_id']],fingerprint(['window-utility-v1',r['question_id']])))
            selected.append(chosen);counts[chosen['persona_id']]+=1;candidates.remove(chosen)
    return selected


def historical_prefix(row, contexts):
    original=contexts[row['shared_context_id']][:int(row['end_index_in_shared_context'])]
    prefix=list(original)
    indices=[i for i,t in enumerate(prefix) if t['role']=='user']
    duplicate=None
    clean=lambda text:normalize(re.sub(r'^\s*User:\s*','',text,flags=re.I))
    if indices and clean(prefix[indices[-1]]['content'])==clean(row['user_question_or_message']):
        duplicate=indices[-1];prefix=prefix[:duplicate]
    turns=[Turn(f"p{row['persona_id']}:m{i:04d}:t0000",t['role'],t['content'],'')
           for i,t in enumerate(prefix) if t['role']=='user']
    return original,turns,duplicate


def prepare(counter, model, study=STUDY):
    study=Path(study)
    if study.exists():
        raise FileExistsError('Preserve frozen window inputs; use another version')
    if not counter.exact:
        raise ValueError('Exact reader tokenizer required')
    source,rows,contexts,_=source_rows()
    previous_path=ROOT/'research/native_choice/v1/manifest.json'
    previous=read(previous_path)
    previous_queries=ROOT/'research/native_choice/v1/queries.json'
    used={q['question_id'] for q in read(previous_queries)}
    selected=select_new(rows,previous['pilot_personas'],used)
    queries,references=[],[]
    for row in selected:
        original,turns,duplicate=historical_prefix(row,contexts)
        groups=variants(token_windows(turns,counter,limit=384),counter)
        ranked=BM25(groups).search(row['user_question_or_message'],len(groups))
        if not ranked:
            raise ValueError('No history candidate for a selected question')
        anchor=ranked[0][0]
        target,trace=link_target(anchor,groups)
        evidence={'global_none:0':{'text':'','tokens':0,'chunk_ids':[]}}
        plans={}
        for budget in (1024,2048):
            evidence[f'user_bm25:{budget}']=pack(ranked,counter,budget)
            plan=intervention_plan(ranked,target,anchor,counter,budget)
            plan['background_hash']=fingerprint(plan['background'])
            plans[str(budget)]=plan
            for branch,item in plan['branches'].items():
                evidence[f'{branch}:{budget}']=item
        opts=parse_options(row['all_options'])
        queries.append({'question_id':row['question_id'],'question':row['user_question_or_message'],
                        'source_group':'personamem1-persona-'+row['persona_id'],'question_type':row['question_type'],
                        'options':[re.sub(r'^\([a-d]\)\s+','',v) for v in opts], 'shifts':[0,1,2,3],
                        'context_id':row['shared_context_id'],'prefix_end_index':int(row['end_index_in_shared_context']),
                        'prefix_hash':fingerprint(original),'duplicate_current_user_turn_removed_at_index':duplicate,
                        'source_user_turns':len(turns),'evidence':evidence,'plans':plans,'link_trace':trace,
                        'target_groups':[g.to_dict() for g in target], 'anchor_id':anchor.group_id})
        references.append({'question_id':row['question_id'],'correct_original_index':next(i for i,v in enumerate(opts) if v.startswith(row['correct_answer']+' ')),
                           'official_correct_option':row['correct_answer']})
        print('WINDOW_PREPARED',row['question_id'],trace['linked'],flush=True)
    study.mkdir(parents=True)
    write_json(study/'queries.json',queries);write_json(study/'references.private.json',references)
    manifest={'version':'window-utility-v1','frozen_at':datetime.now(timezone.utc).isoformat(),'code_hashes':hashes(),
        'queries_sha256':sha256(study/'queries.json'),'references_sha256':sha256(study/'references.private.json'),
        'source_manifest_sha256':sha256(ROOT/'data/personamem_v1/source_manifest.json'),'dataset_revision':source['revision'],
        'previous_manifest_sha256':sha256(previous_path),'previous_queries_sha256':sha256(previous_queries),
        'plan_sha256':sha256(ROOT/'research/WINDOW_UTILITY_PLAN.md'),
        'pilot_personas':previous['pilot_personas'],'reserved_method_validation_personas':previous['reserved_method_validation_personas'],
        'excluded_previous_question_ids':sorted(used),'questions':len(queries),
        'types':dict(Counter(q['question_type'] for q in queries)), 'persona_counts':dict(Counter(q['source_group'] for q in queries)),
        'readers':list(READERS),'conditions':list(CONDITIONS),'option_orders_per_question':4,
        'expected_records':len(queries)*len(CONDITIONS)*4*len(READERS),
        'reader_assets':{r:assets(ROOT/'models'/r) for r in READERS},'tokenizer_sha256':sha256(Path(model)/'tokenizer.json'),
        'target_policy':'BM25 rank one user fragment plus one latest lexically related update; no answer-source oracle',
        'paired_policy':'all five branches share one exact background reserved for the largest complete serialized branch',
        'query_blind_variants':True,'retrieval_excludes_options_and_reference':True,
        'linked_questions':sum(q['link_trace']['linked'] for q in queries),
        'semantic_link_verification_complete':False,'primary_measure':'bundle_raw minus without; four-order question mean, source-group uncertainty',
        'secondary_measures':['bundle_guard versus bundle_lead','bundle_raw versus anchor on linked questions','user_bm25 versus global_none'],
        'learned_judge_used':False,'ai_adjudication_used_as_gold':False,'formal_benchmark_score':False,'lme_heldout_used':False,
        'limitations':['Only six already exposed source personas; new questions are not an independent source validation.',
                       'Cyclic shifts balance each option across labels but do not cover all 24 permutations.',
                       'Lexical update links may be unrelated; guards are extractive baselines without semantic guarantees.',
                       'Target fragment interventions depend on one fixed background and are not global additive utilities.',
                       'Two same-family readers cannot establish capability vector efficacy or family transfer.']}
    write_json(study/'manifest.json',manifest)
    print('WINDOW_FROZEN',json.dumps({k:manifest[k] for k in ('questions','linked_questions','expected_records')}),flush=True)
    return manifest


def load(study=STUDY):
    study=Path(study);m=read(study/'manifest.json')
    paths={'queries_sha256':study/'queries.json','references_sha256':study/'references.private.json',
        'source_manifest_sha256':ROOT/'data/personamem_v1/source_manifest.json',
        'previous_manifest_sha256':ROOT/'research/native_choice/v1/manifest.json',
        'previous_queries_sha256':ROOT/'research/native_choice/v1/queries.json','plan_sha256':ROOT/'research/WINDOW_UTILITY_PLAN.md'}
    if m['code_hashes']!=hashes() or any(m[k]!=sha256(p) for k,p in paths.items()):
        raise ValueError('Frozen window study source/code changed')
    return m,read(study/'queries.json')


def verify_rows(rows, queries, config):
    qmap={q['question_id']:q for q in queries}
    expected={(r,q['question_id'],a,b,s) for r in READERS for q in queries for a,b in CONDITIONS for s in q['shifts']}
    keys=[record_key(r) for r in rows]
    if len(keys)!=len(set(keys)) or not set(keys)<=expected:
        raise ValueError('Duplicate or unknown window output')
    checked={}
    for row in rows:
        q=qmap[row['question_id']];arm,budget=row['arm'],row['budget'];item=q['evidence'][f'{arm}:{budget}']
        scores=row['restricted_label_logits']
        if (row['signature']!=fingerprint(config) or row['prompt_hash']!=fingerprint(messages(q,arm,budget,row['shift']))
            or row['evidence_hash']!=fingerprint(item['text']) or row['evidence_tokens']!=item['tokens']
            or item['tokens']>budget or len(scores)!=4 or not all(math.isfinite(s) for s in scores)
            or row['prediction']!='abcd'[max(range(4),key=lambda i:scores[i])]):
            raise ValueError('Window prediction or input provenance differs')
        expected_background=fingerprint(q['plans'][str(budget)]['background']) if arm in ('without',*VARIANTS) else None
        if row['background_hash']!=expected_background:
            raise ValueError('Paired background changed')
        if row['prediction_origin']=='cache-identical-prompt':
            source=checked.get(tuple(row['cached_from_key']))
            if source is None or source['reader']!=row['reader'] or source['prompt_hash']!=row['prompt_hash'] or source['restricted_label_logits']!=scores:
                raise ValueError('Cached window output is not the same model input')
        elif row['prediction_origin']!='model-forward':
            raise ValueError('Unknown prediction origin')
        checked[record_key(row)]=row
    return expected


def run(backend, counter, study=STUDY, output=OUTPUT):
    m,queries=load(study);reader=backend.model_name
    if (reader not in READERS or not counter.exact or assets(backend.path)!=m['reader_assets'][reader]
        or sha256(Path(backend.path)/'tokenizer.json')!=m['tokenizer_sha256']):
        raise ValueError('Reader/tokenizer assets changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config={'study_signature':fingerprint(m),'reader_assets':m['reader_assets']}
    if (output/'config.json').exists() and read(output/'config.json')!=config:
        raise ValueError('Incompatible window resume')
    write_json(output/'config.json',config)
    rows=list(read_jsonl(output/'answers.jsonl')) if (output/'answers.jsonl').exists() else []
    verify_rows(rows,queries,config)
    done={record_key(r) for r in rows};cache={(r['reader'],r['prompt_hash']):r for r in rows}
    with (output/'answers.jsonl').open('a',encoding='utf-8') as stream:
        for q in queries:
            for arm,budget in CONDITIONS:
                item=q['evidence'][f'{arm}:{budget}']
                if counter.count(item['text'])!=item['tokens'] or item['tokens']>budget:
                    raise ValueError('Exact final evidence cost changed')
                for shift in q['shifts']:
                    key=(reader,q['question_id'],arm,budget,shift)
                    if key in done:continue
                    prompt=messages(q,arm,budget,shift);ph=fingerprint(prompt);cached=cache.get((reader,ph))
                    if cached:
                        result={'prediction':cached['prediction'],'restricted_label_logits':cached['restricted_label_logits'],
                                'usage':{**cached['usage'],'elapsed_seconds':0.0}}
                    else:result=classify(backend,prompt)
                    row={'reader':reader,'question_id':q['question_id'],'arm':arm,'budget':budget,'shift':shift,
                        'signature':fingerprint(config),'prompt_hash':ph,'evidence_hash':fingerprint(item['text']),
                        'evidence_tokens':item['tokens'], 'background_hash':q['plans'][str(budget)]['background_hash'] if arm in ('without',*VARIANTS) else None,
                        'prediction_origin':'cache-identical-prompt' if cached else 'model-forward',
                        'cached_from_key':list(record_key(cached)) if cached else None,**result}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();done.add(key);cache[(reader,ph)]=row
                    if len(done)%52==0:
                        write_json(output/'status.json',{'status':'running','reader':reader,'records':len(done),'expected_records':m['expected_records']})
                        print('WINDOW_PROGRESS',reader,len(done),'/',m['expected_records'],flush=True)
    return report(study,output)


def log_probability(row, correct_index):
    scores=row['restricted_label_logits'];m=max(scores)
    displayed=(correct_index-row['shift'])%4
    return scores[displayed]-m-math.log(sum(math.exp(v-m) for v in scores))


def summary(rows, query_count):
    byq=defaultdict(list)
    for r in rows:byq[r['question_id']].append(r)
    complete=[v for v in byq.values() if len(v)==4]
    return {'correct_presentations':sum(r['correct'] for r in rows),'presentations':len(rows),
        'mean_accuracy':sum(r['correct'] for r in rows)/len(rows) if rows else None,
        'complete_questions':len(complete),'four_orders_correct_questions':sum(all(r['correct'] for r in v) for v in complete),
        'semantic_choice_changed_questions':sum(len({(int(r['shift'])+'abcd'.index(r['prediction']))%4 for r in v})>1 for v in complete),
        'mean_evidence_tokens':sum(r['evidence_tokens'] for r in rows)/len(rows) if rows else None,
        'expected_questions':query_count}


def report(study=STUDY, output=OUTPUT):
    m,queries=load(study);output=Path(output);config=read(output/'config.json')
    if config!={'study_signature':fingerprint(m),'reader_assets':m['reader_assets']}:
        raise ValueError('Window configuration differs')
    rows=list(read_jsonl(output/'answers.jsonl'));expected=verify_rows(rows,queries,config)
    refs={r['question_id']:r for r in read(Path(study)/'references.private.json')};qmap={q['question_id']:q for q in queries}
    scored=[{**r,'correct':score_row(r,qmap[r['question_id']],refs[r['question_id']]),
             'correct_option_conditional_log_probability':log_probability(r,refs[r['question_id']]['correct_original_index'])} for r in rows]
    index={record_key(r):r for r in scored};utilities=[]
    for reader in READERS:
        for q in queries:
            for budget in (1024,2048):
                for variant in VARIANTS:
                    pairs=[(index[(reader,q['question_id'],'without',budget,s)],index[(reader,q['question_id'],variant,budget,s)])
                           for s in q['shifts'] if (reader,q['question_id'],'without',budget,s) in index and (reader,q['question_id'],variant,budget,s) in index]
                    if len(pairs)!=4:continue
                    utilities.append({'reader':reader,'question_id':q['question_id'],'source_group':q['source_group'],'budget':budget,
                        'representation':variant,'background_hash':q['plans'][str(budget)]['background_hash'],
                        'target_chunk_ids':q['evidence'][f'{variant}:{budget}']['target_chunk_ids'],
                        'delta_mean_accuracy':sum(int(b['correct'])-int(a['correct']) for a,b in pairs)/4,
                        'delta_conditional_log_probability':sum(b['correct_option_conditional_log_probability']-a['correct_option_conditional_log_probability'] for a,b in pairs)/4,
                        'without_correct':sum(a['correct'] for a,b in pairs),'variant_correct':sum(b['correct'] for a,b in pairs),
                        'orders':4,'linked':q['link_trace']['linked'],'training_only':True})
    summaries={r:{f'{a}:{b}':summary([x for x in scored if (x['reader'],x['arm'],x['budget'])==(r,a,b)],len(queries))
                  for a,b in CONDITIONS} for r in READERS}
    value={'status':'completed-window-utility' if len(rows)==len(expected) else 'incomplete-window-utility',
        'records':len(rows),'expected_records':len(expected),'questions':len(queries),'source_personas':len(m['pilot_personas']),
        'actual_model_forward_calls':sum(r['prediction_origin']=='model-forward' for r in rows),
        'identical_prompt_cache_reuses':sum(r['prediction_origin']=='cache-identical-prompt' for r in rows),
        'model_seconds':round(sum(r['usage']['elapsed_seconds'] for r in rows),2),'linked_questions':m['linked_questions'],
        'by_reader_condition':summaries,'utility_records':len(utilities),
        'nonzero_accuracy_utility_records':sum(abs(u['delta_mean_accuracy'])>1e-12 for u in utilities),
        'negative_accuracy_utility_records':sum(u['delta_mean_accuracy']<0 for u in utilities),
        'formal_benchmark_score':False,'capability_vector_efficacy_established':False,'lme_heldout_used':False,'limitations':m['limitations']}
    write_json(output/'report.json',value);write_json(output/'status.json',{k:value[k] for k in ('status','records','expected_records')})
    for name,data in (('scored.jsonl',scored),('utilities.train_only.jsonl',utilities)):
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
    print('WINDOW_REPORT',json.dumps({k:value[k] for k in ('status','records','actual_model_forward_calls','utility_records','nonzero_accuracy_utility_records')}),flush=True)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--reader',choices=READERS,default=READERS[0]);args=parser.parse_args()
    if args.command=='report':return report()
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment();model=ROOT/'models'/args.reader;counter=TokenCounter(model)
    if args.command=='prepare':return prepare(counter,model)
    from .backend import TransformersBackend
    return run(TransformersBackend(model),counter)


if __name__=='__main__':main()
