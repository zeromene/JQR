"""Frozen answer-level controls on four previously audited retrieval policies."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path

from .choice_protocol_audit import save
from .diagnostics import sha256
from .four_reader_probe import READERS, load as load_probe, probe_assets
from .native_choice_pilot import classify, messages, read, record_key, score_row
from .schema import fingerprint, read_jsonl
from .shared_retrieval_audit import load as load_retrieval, query_content

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'research/retrieval_answer_probe/v1'
OUTPUT = ROOT/'runs/retrieval_answer_probe_v1'
POLICIES = ('lexical','dense','hybrid','hybrid_neighbors')
BUDGETS = (1024,2048)
ARMS = (('none',0),*((p,b) for b in BUDGETS for p in POLICIES))
CONTRASTS = (('dense','lexical'),('hybrid','lexical'),('hybrid_neighbors','hybrid'))


def expected(queries):
    return {(r,q['question_id'],a,b,s) for r in READERS for q in queries for a,b in ARMS for s in q['shifts']}


def make_query(original,retrieved):
    for key in ('question_id','source_group','persona_id','history_path'):
        if original[key]!=retrieved[key]: raise ValueError('Cross-source retrieval join')
    clean = query_content(original['question'])
    if clean!=retrieved['normalized_query'] or original['question']!=retrieved['question']:
        raise ValueError('Question normalization changed')
    evidence = {'none:0':{'text':'','tokens_by_reader':{r:0 for r in READERS},'chunk_ids':[]}}
    for policy in POLICIES:
        for budget in BUDGETS:
            key = f'{policy}:{budget}';source = retrieved['branches'][key]
            evidence[key] = {k:source[k] for k in ('text','tokens_by_reader','chunk_ids')}
    return ({k:original[k] for k in ('question_id','source_group','persona_id','history_path','options','shifts')}
            | {'question':clean,'original_question_hash':fingerprint(original['question']),'evidence':evidence})


def prompt_entry(counter,q,arm,budget,shift):
    prompt = messages(q,arm,budget,shift)
    text = counter.tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids = counter.tokenizer.encode(text,add_special_tokens=False)
    if len(ids)>12288: raise ValueError('Complete answer probe prompt exceeds reader limit')
    return {'messages_hash':fingerprint(prompt),'token_ids_hash':fingerprint(ids),'tokens':len(ids)}


def prepare(counters,study=STUDY):
    study = Path(study)
    if study.exists(): raise FileExistsError('Preserve frozen answer probe')
    if set(counters)!=set(READERS) or not all(c.exact for c in counters.values()):
        raise ValueError('Every reader requires its actual tokenizer')
    old,original = load_probe();retrieval,items = load_retrieval()
    integrity = read(ROOT/'runs/shared_retrieval_audit_v1/input_integrity.json')
    if integrity['status']!='passed' or integrity['inputs_sha256']!=retrieval['inputs_sha256']:
        raise ValueError('Source retrieval audit not verified')
    byid = {q['question_id']:q for q in items}
    if set(byid)!={q['question_id'] for q in original}: raise ValueError('Full exposed query set required')
    queries = [make_query(q,byid[q['question_id']]) for q in original]
    prompts = {};maximum = {}
    for reader,counter in counters.items():
        if probe_assets(ROOT/'models'/reader)!=old['reader_assets'][reader]: raise ValueError('Reader assets changed')
        labels = [counter.tokenizer.encode(x,add_special_tokens=False) for x in 'abcd']
        if any(len(v)!=1 for v in labels) or len({v[0] for v in labels})!=4: raise ValueError('Four distinct single-token labels required')
        prompts[reader] = {};maximum[reader] = 0
        for q in queries:
            entries = {}
            for arm,budget in ARMS:
                item = q['evidence'][f'{arm}:{budget}']
                if counter.count(item['text'])!=item['tokens_by_reader'][reader] or item['tokens_by_reader'][reader]>budget:
                    raise ValueError('Inherited evidence token budget changed')
                for shift in q['shifts']:
                    entry = prompt_entry(counter,q,arm,budget,shift)
                    entries[f'{arm}:{budget}:{shift}'] = entry
                    maximum[reader] = max(maximum[reader],entry['tokens'])
            prompts[reader][q['question_id']] = entries
        print('ANSWER_PROBE_PREFLIGHT',reader,maximum[reader],flush=True)
    study.mkdir(parents=True)
    save(study/'queries.json',queries);save(study/'prompts.json',prompts)
    refs = read(ROOT/'research/four_reader_probe/v1/references.private.json')
    save(study/'references.private.json',refs)
    paths = ('research/RETRIEVAL_ANSWER_PROBE_PLAN.md','reader_memory/retrieval_answer_probe.py',
             'research/shared_retrieval_audit/v1/manifest.json','research/shared_retrieval_audit/v1/inputs.json',
             'runs/shared_retrieval_audit_v1/input_integrity.json','research/four_reader_probe/v1/manifest.json',
             'research/four_reader_probe/v1/queries.json','research/four_reader_probe/v1/references.private.json')
    m = {'version':'retrieval-answer-probe-v1','frozen_at':datetime.now(timezone.utc).isoformat(),
         'file_hashes':{p:sha256(ROOT/p) for p in paths},
         'study_files':{name:sha256(study/name) for name in ('queries.json','prompts.json','references.private.json')},
         'reader_assets':old['reader_assets'],'readers':list(READERS),'arms':list(ARMS),
         'questions':len(queries),'source_groups':len({q['source_group'] for q in queries}),
         'expected_records':len(expected(queries)),'maximum_preflight_prompt_tokens':maximum,
         'inherited_retrieval_inputs_sha256':retrieval['inputs_sha256'],'new_retrieval_or_parameter_tuning':False,
         'representation':'raw only; inherited exact evidence, all readers share strings',
         'question_format':'actual user content in every arm; old wrapped-query answers are not reused',
         'reserved_validation_used':False,'learned_selector_trained':False,'formal_benchmark_score':False}
    save(study/'manifest.json',m)
    print('ANSWER_PROBE_FROZEN',m['expected_records'],flush=True)
    return m


def load(study=STUDY):
    study = Path(study);m = read(study/'manifest.json')
    if any(sha256(ROOT/p)!=h for p,h in m['file_hashes'].items()) or any(sha256(study/p)!=h for p,h in m['study_files'].items()):
        raise ValueError('Frozen answer probe changed')
    load_probe();load_retrieval()
    queries = read(study/'queries.json')
    if len(expected(queries))!=m['expected_records']: raise ValueError('Frozen matrix changed')
    return m,queries,read(study/'prompts.json')


def verify(rows,queries,m,prompts):
    valid_keys = expected(queries);qmap = {q['question_id']:q for q in queries};seen = {}
    for row in rows:
        key = record_key(row)
        if key in seen or key not in valid_keys: raise ValueError('Duplicate or unknown answer probe output')
        reader,qid,arm,budget,shift = key;q = qmap[qid];item = q['evidence'][f'{arm}:{budget}']
        entry = prompts[reader][qid][f'{arm}:{budget}:{shift}'];scores = row['restricted_label_logits']
        if (row['signature']!=fingerprint(m) or row['prompt_hash']!=entry['messages_hash']
            or row['prompt_hash']!=fingerprint(messages(q,arm,budget,shift)) or row['prompt_token_ids_hash']!=entry['token_ids_hash']
            or row['usage']['prompt_tokens']!=entry['tokens'] or row['evidence_hash']!=fingerprint(item['text'])
            or row['evidence_tokens']!=item['tokens_by_reader'][reader] or row['evidence_tokens']>budget
            or len(scores)!=4 or not all(math.isfinite(v) for v in scores)
            or row['prediction']!='abcd'[max(range(4),key=lambda i:scores[i])]):
            raise ValueError('Answer probe provenance or prediction changed')
        if row['prediction_origin']=='cache-identical-prompt':
            prior = seen.get(tuple(row['cached_from_key']))
            if (prior is None or prior['reader']!=reader or prior['prompt_hash']!=row['prompt_hash']
                or prior['prompt_token_ids_hash']!=row['prompt_token_ids_hash'] or prior['restricted_label_logits']!=scores
                or row['usage']['elapsed_seconds']!=0): raise ValueError('Invalid answer probe cache reuse')
        elif row['prediction_origin']!='model-forward' or row['cached_from_key'] is not None:
            raise ValueError('Unknown answer probe output origin')
        seen[key] = row
    return valid_keys


def run(backend,counter,study=STUDY,output=OUTPUT):
    m,queries,prompts = load(study);output = Path(output);reader = backend.model_name
    if reader not in READERS or not counter.exact or probe_assets(backend.path)!=m['reader_assets'][reader]:
        raise ValueError('Wrong reader/tokenizer')
    output.mkdir(parents=True,exist_ok=True)
    rows = list(read_jsonl(output/'answers.jsonl')) if (output/'answers.jsonl').exists() else []
    verify(rows,queries,m,prompts);done = {record_key(r) for r in rows}
    cache = {(r['reader'],r['prompt_hash'],r['prompt_token_ids_hash']):r for r in rows}
    with (output/'answers.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
        for q in queries:
            for arm,budget in ARMS:
                item = q['evidence'][f'{arm}:{budget}']
                if counter.count(item['text'])!=item['tokens_by_reader'][reader]: raise ValueError('Reader token cost changed')
                for shift in q['shifts']:
                    key = (reader,q['question_id'],arm,budget,shift)
                    if key in done: continue
                    entry = prompt_entry(counter,q,arm,budget,shift)
                    if entry!=prompts[reader][q['question_id']][f'{arm}:{budget}:{shift}']:
                        raise ValueError('Actual tokenized prompt changed')
                    ph = entry['messages_hash'];th = entry['token_ids_hash'];prior = cache.get((reader,ph,th))
                    result = ({'prediction':prior['prediction'],'restricted_label_logits':prior['restricted_label_logits'],
                               'usage':{**prior['usage'],'elapsed_seconds':0.0}} if prior else classify(backend,messages(q,arm,budget,shift)))
                    if result['usage']['prompt_tokens']!=entry['tokens']: raise ValueError('Actual model input length changed')
                    row = {'reader':reader,'question_id':q['question_id'],'arm':arm,'budget':budget,'shift':shift,
                           'signature':fingerprint(m),'prompt_hash':ph,'prompt_token_ids_hash':th,
                           'evidence_hash':fingerprint(item['text']),'evidence_tokens':item['tokens_by_reader'][reader],
                           'prediction_origin':'cache-identical-prompt' if prior else 'model-forward',
                           'cached_from_key':list(record_key(prior)) if prior else None,**result}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();done.add(key);cache[(reader,ph,th)] = row
                    if len(done)%144==0:
                        save(output/'status.json',{'status':'running','reader':reader,'records':len(done),'expected_records':m['expected_records']})
                        print('ANSWER_PROBE_PROGRESS',reader,len(done),'/',m['expected_records'],flush=True)
    return report(study,output)


def report(study=STUDY,output=OUTPUT):
    m,queries,prompts = load(study);output = Path(output);rows = list(read_jsonl(output/'answers.jsonl'))
    keys = verify(rows,queries,m,prompts);qmap = {q['question_id']:q for q in queries}
    refs = {r['question_id']:r for r in read(Path(study)/'references.private.json')}
    scored = [{**r,'source_group':qmap[r['question_id']]['source_group'],
               'correct':score_row(r,qmap[r['question_id']],refs[r['question_id']]),
               'semantic_choice':(r['shift']+'abcd'.index(r['prediction']))%4} for r in rows]
    summaries = {}
    for reader in READERS:
        summaries[reader] = {}
        for arm,budget in ARMS:
            selected = [r for r in scored if (r['reader'],r['arm'],r['budget'])==(reader,arm,budget)]
            byq = defaultdict(list)
            for r in selected: byq[r['question_id']].append(r)
            complete = [v for v in byq.values() if len(v)==4]
            summaries[reader][f'{arm}:{budget}'] = {
                'presentations':len(selected),'correct':sum(r['correct'] for r in selected),
                'accuracy':sum(r['correct'] for r in selected)/len(selected) if selected else None,
                'mean_evidence_tokens':sum(r['evidence_tokens'] for r in selected)/len(selected) if selected else None,
                'complete_questions':len(complete),'semantic_choice_varies_across_orders':sum(len({r['semantic_choice'] for r in v})>1 for v in complete),
                'all_four_orders_correct_questions':sum(all(r['correct'] for r in v) for v in complete),
                'label_histogram':dict(Counter(r['prediction'] for r in selected)),
                'by_position':{str(s):{'presentations':sum(r['shift']==s for r in selected),
                                     'correct':sum(r['correct'] and r['shift']==s for r in selected)} for s in range(4)}}
    value = {'status':'completed-retrieval-answer-probe' if len(rows)==len(keys) else 'incomplete-retrieval-answer-probe',
             'records':len(rows),'expected_records':len(keys),'questions':len(queries),'source_groups':m['source_groups'],
             'actual_model_forward_calls':sum(r['prediction_origin']=='model-forward' for r in rows),
             'identical_prompt_cache_reuses':sum(r['prediction_origin']=='cache-identical-prompt' for r in rows),
             'model_seconds':sum(r['usage']['elapsed_seconds'] for r in rows),'by_reader':summaries,
             'formal_benchmark_score':False,'reserved_validation_used':False,'reader_capability_efficacy_established':False}
    save(output/'report.json',value)
    save(output/'status.json',{k:value[k] for k in ('status','records','expected_records')})
    (output/'scored.jsonl').write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored).encode('utf-8'))
    print('ANSWER_PROBE_REPORT',value['status'],len(rows),flush=True)
    return value


def analyze(study=STUDY,output=OUTPUT):
    result = report(study,output)
    if result['status']!='completed-retrieval-answer-probe': raise ValueError('Complete all readers before interpreting comparisons')
    import numpy as np
    m,queries,_ = load(study);output = Path(output);rows = list(read_jsonl(output/'scored.jsonl'))
    index = {record_key(r):r for r in rows};groups = sorted({q['source_group'] for q in queries})
    draws = np.random.default_rng(20261026).integers(0,len(groups),size=(20000,len(groups)))
    def delta(reader,candidate,cb,baseline,bb):
        return [{'question_id':q['question_id'],'source_group':q['source_group'],
                 'delta':sum(int(index[(reader,q['question_id'],candidate,cb,s)]['correct'])-
                             int(index[(reader,q['question_id'],baseline,bb,s)]['correct']) for s in q['shifts'])/4} for q in queries]
    def summarize(values):
        # Persona counts are equal here; use weighted resampling so the formula
        # still retains all questions if a later input audit changes that fact.
        sizes = np.array([sum(v['source_group']==g for v in values) for g in groups])
        sums = np.array([sum(v['delta'] for v in values if v['source_group']==g) for g in groups])
        if not all(sizes): raise ValueError('Missing source in paired comparison')
        samples = sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
        lo,hi = np.quantile(samples,[.025,.975])*100
        return {'difference_pp':float(sums.sum()/sizes.sum()*100),'exploratory_group_bootstrap_95ci_pp':[float(lo),float(hi)],
                'questions':len(values),'source_groups':len(groups),'positive_questions':sum(v['delta']>0 for v in values),
                'negative_questions':sum(v['delta']<0 for v in values),'unchanged_questions':sum(v['delta']==0 for v in values)}
    comparisons = [];contrasts = {};budget_effects = [];interactions = []
    for reader in READERS:
        for budget in BUDGETS:
            for candidate,baseline in (*CONTRASTS,*((p,'none') for p in POLICIES)):
                values = delta(reader,candidate,budget,baseline,0 if baseline=='none' else budget)
                contrasts[(reader,candidate,baseline,budget)] = values
                comparisons.append({'reader':reader,'candidate':candidate,'baseline':baseline,'budget':budget,
                    'primary':(candidate,baseline)==('dense','lexical'),**summarize(values),'question_differences':values})
        for policy in POLICIES:
            budget_effects.append({'reader':reader,'policy':policy,'candidate_budget':2048,'baseline_budget':1024,
                                   **summarize(delta(reader,policy,2048,policy,1024))})
    for budget in BUDGETS:
        for candidate,baseline in CONTRASTS:
            for a,b in combinations(READERS,2):
                first = contrasts[(a,candidate,baseline,budget)];second = contrasts[(b,candidate,baseline,budget)]
                values = [{**v,'delta':v['delta']-w['delta']} for v,w in zip(first,second)]
                interactions.append({'first_reader':a,'second_reader':b,'candidate':candidate,'baseline':baseline,
                                     'budget':budget,**summarize(values)})
    value = {'version':'retrieval-answer-analysis-v1','manifest_sha256':sha256(Path(study)/'manifest.json'),
             'answers_sha256':sha256(output/'answers.jsonl'),'paired_comparisons':comparisons,'budget_effects':budget_effects,
             'exploratory_reader_interactions':interactions,'bootstrap':{'resamples':20000,'seed':20261026,'unit':'complete persona source group',
                'four_positions_averaged_within_question':True,'multiple_comparison_correction':False},
             'formal_benchmark_score':False,'reserved_validation_used':False,'conditional_selector_trained':False,
             'reader_capability_efficacy_established':False}
    save(output/'analysis.json',value)
    print('ANSWER_PROBE_ANALYZED',len(comparisons),'comparisons',flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','run','report','analyze'))
    parser.add_argument('--reader',choices=READERS,default=READERS[0]);args = parser.parse_args()
    if args.command=='report': return report()
    if args.command=='analyze': return analyze()
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    if args.command=='prepare': return prepare({r:TokenCounter(ROOT/'models'/r) for r in READERS})
    from .backend import TransformersBackend
    model = ROOT/'models'/args.reader
    return run(TransformersBackend(model),TokenCounter(model))


if __name__=='__main__': main()
