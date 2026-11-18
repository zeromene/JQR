"""Frozen comparison of restricted next-token choice and strict free generation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from .diagnostics import sha256
from .four_reader_probe import READERS, load as load_probe, probe_assets
from .native_choice_pilot import displayed_options, messages, read, record_key
from .schema import fingerprint, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'research/choice_protocol_audit/v1'
OUTPUT = ROOT/'runs/choice_protocol_audit_v1'
OLD = ROOT/'runs/four_reader_probe_v1/answers.jsonl'
ARMS = (('none', 0), ('raw', 1024))
MAX_NEW_TOKENS = 16


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    temporary.replace(path)


def sample(queries, groups=8):
    by_group = defaultdict(list)
    for q in queries:
        by_group[q['source_group']].append(q)
    if len(by_group) < groups:
        raise ValueError('Insufficient already exposed source groups')
    selected = sorted(by_group, key=lambda g: fingerprint(['choice-protocol-groups-v1', g]))[:groups]
    return [min(by_group[g], key=lambda q: fingerprint(['choice-protocol-query-v1', q['question_id']])) for g in selected]


def parse(text, truncated=False):
    stripped = text.strip()
    return stripped if not truncated and re.fullmatch('[abcd]', stripped) else None


def expected(queries):
    return {(r,q['question_id'],a,b,s) for r in READERS for q in queries for a,b in ARMS for s in q['shifts']}


def prepare(study=STUDY):
    study = Path(study)
    if study.exists():
        raise FileExistsError('Preserve frozen choice audit')
    old, queries = load_probe(); chosen = sample(queries)
    # The input selector does not read old predictions or correct labels.
    study.mkdir(parents=True)
    save(study/'queries.json', chosen)
    paths = ['research/CHOICE_PROTOCOL_AUDIT_PLAN.md','reader_memory/choice_protocol_audit.py',
             'research/four_reader_probe/v1/manifest.json','research/four_reader_probe/v1/queries.json',
             'research/four_reader_probe/v1/references.private.json', 'runs/four_reader_probe_v1/answers.jsonl']
    m = {'version':'choice-protocol-audit-v1','frozen_at':datetime.now(timezone.utc).isoformat(),
         'file_hashes':{p:sha256(ROOT/p) for p in paths}, 'queries_sha256':sha256(study/'queries.json'),
         'readers':list(READERS),'reader_assets':old['reader_assets'],'questions':len(chosen),
         'source_groups':len({q['source_group'] for q in chosen}),'expected_records':len(expected(chosen)),
         'arms':list(ARMS),'max_new_tokens':MAX_NEW_TOKENS,'parser':'fullmatch [abcd] after whitespace strip; reject length truncation',
         'sampling_uses_predictions_or_labels':False,'reserved_validation_used':False,
         'prompt_or_evidence_modified':False,'formal_benchmark_score':False}
    save(study/'manifest.json',m)
    print('CHOICE_AUDIT_FROZEN',m['expected_records'],flush=True)
    return m


def load(study=STUDY):
    study = Path(study); m = read(study/'manifest.json')
    if any(sha256(ROOT/p)!=h for p,h in m['file_hashes'].items()) or sha256(study/'queries.json')!=m['queries_sha256']:
        raise ValueError('Frozen choice audit source changed')
    load_probe()
    queries = read(study/'queries.json')
    if len(expected(queries))!=m['expected_records']:
        raise ValueError('Choice audit sample changed')
    return m, queries


def generate(backend, prompt_messages):
    b = backend
    prompt = b.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = b.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(b.device)
    length = batch['input_ids'].shape[-1]
    if length>b.max_input_tokens:
        raise ValueError('Choice audit context exceeds limit')
    label_ids = [b.tokenizer.encode(x,add_special_tokens=False) for x in 'abcd']
    if any(len(x)!=1 for x in label_ids) or len({x[0] for x in label_ids})!=4:
        raise ValueError('Labels require distinct single tokens')
    eos = b.model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos,list) else [eos] if eos is not None else []
    started = time.perf_counter()
    with b.lock, b.torch.inference_mode():
        output = b.model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=b.tokenizer.pad_token_id if b.tokenizer.pad_token_id is not None else b.tokenizer.eos_token_id,
                    return_dict_in_generate=True, output_logits=True)
    ids = output.sequences[0,length:].tolist()
    logits = output.logits[0][0].float()
    scores = logits[[x[0] for x in label_ids]]
    probs = b.torch.softmax(logits,dim=0)
    top = b.torch.topk(probs,8)
    stopped = bool(ids and ids[-1] in eos_ids)
    truncated = len(ids)>=MAX_NEW_TOKENS and not stopped
    text = b.tokenizer.decode(ids,skip_special_tokens=True)
    return {'text':text,'raw_text':b.tokenizer.decode(ids,skip_special_tokens=False),'generated_token_ids':ids,
            'eos_token_ids':eos_ids,'eos_stopped':stopped,'length_truncated':truncated,'prediction':parse(text,truncated),
            'first_step_label_logits':scores.cpu().tolist(),
            'first_step_label_argmax':'abcd'[int(scores.argmax().item())],
            'first_step_label_probability_mass':float(probs[[x[0] for x in label_ids]].sum().item()),
            'first_step_label_vocab_ranks':[int((logits>v).sum().item())+1 for v in scores],
            'first_step_top_tokens':[{'id':i,'text':b.tokenizer.decode([i],skip_special_tokens=False),'probability':p}
                                     for i,p in zip(top.indices.cpu().tolist(),top.values.cpu().tolist())],
            'usage':{'prompt_tokens':length,'completion_tokens':len(ids),'elapsed_seconds':round(time.perf_counter()-started,4)}}


def verify(rows, queries, m, old):
    qmap = {q['question_id']:q for q in queries}; seen = set(); valid_keys = expected(queries)
    for row in rows:
        key = record_key(row)
        if key in seen or key not in valid_keys or key not in old:
            raise ValueError('Duplicate/unknown choice audit output')
        r,qid,a,b,s = key; q = qmap[qid]; prior = old[key]
        if (row['signature']!=fingerprint(m) or row['prompt_hash']!=fingerprint(messages(q,a,b,s))
            or row['prompt_hash']!=prior['prompt_hash'] or row['evidence_hash']!=prior['evidence_hash']
            or row['old_prediction']!=prior['prediction'] or row['prediction']!=parse(row['text'],row['length_truncated'])
            or row['usage']['completion_tokens']!=len(row['generated_token_ids'])
            or row['usage']['completion_tokens']>MAX_NEW_TOKENS
            or row['eos_stopped']!=bool(row['generated_token_ids'] and row['generated_token_ids'][-1] in row['eos_token_ids'])
            or row['length_truncated']!=(len(row['generated_token_ids'])>=MAX_NEW_TOKENS and not row['eos_stopped'])
            or len(row['first_step_label_logits'])!=4
            or row['first_step_label_argmax']!='abcd'[max(range(4),key=lambda i:row['first_step_label_logits'][i])]):
            raise ValueError('Choice audit provenance or parsed prediction changed')
        seen.add(key)
    return valid_keys


def run(backend, study=STUDY, output=OUTPUT):
    m, queries = load(study); output = Path(output); reader = backend.model_name
    if reader not in READERS or probe_assets(backend.path)!=m['reader_assets'][reader]:
        raise ValueError('Reader assets changed')
    old = {record_key(r):r for r in read_jsonl(OLD)}
    output.mkdir(parents=True,exist_ok=True)
    rows = list(read_jsonl(output/'answers.jsonl')) if (output/'answers.jsonl').exists() else []
    verify(rows,queries,m,old); done = {record_key(r) for r in rows}
    with (output/'answers.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
        for q in queries:
            for a,b in ARMS:
                for s in q['shifts']:
                    key = (reader,q['question_id'],a,b,s)
                    if key in done: continue
                    prompt = messages(q,a,b,s); prior = old[key]
                    if fingerprint(prompt)!=prior['prompt_hash']:
                        raise ValueError('Old prompt pairing failed')
                    result = generate(backend,prompt)
                    row = {'reader':reader,'question_id':q['question_id'],'arm':a,'budget':b,'shift':s,
                           'signature':fingerprint(m),'prompt_hash':prior['prompt_hash'],'evidence_hash':prior['evidence_hash'],
                           'old_prediction':prior['prediction'],**result}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();done.add(key)
                    if len(done)%16==0:
                        save(output/'status.json',{'status':'running','reader':reader,'records':len(done),'expected_records':m['expected_records']})
                        print('CHOICE_AUDIT_PROGRESS',reader,len(done),'/',m['expected_records'],flush=True)
    return report(study,output)


def semantic(label,q,shift):
    return displayed_options(q['options'],shift)[1][ord(label)-97] if label is not None else None


def report(study=STUDY,output=OUTPUT):
    m,queries = load(study); output = Path(output)
    rows = list(read_jsonl(output/'answers.jsonl'));old = {record_key(r):r for r in read_jsonl(OLD)}
    valid_keys = verify(rows,queries,m,old)
    refs = {x['question_id']:x for x in read(ROOT/'research/four_reader_probe/v1/references.private.json')}
    qmap = {q['question_id']:q for q in queries}; scored = []
    for r in rows:
        q = qmap[r['question_id']]; correct = refs[r['question_id']]['correct_original_index']
        scored.append({**r,'source_group':q['source_group'],'semantic_prediction':semantic(r['prediction'],q,r['shift']),
                       'correct':semantic(r['prediction'],q,r['shift'])==correct,
                       'old_correct':semantic(r['old_prediction'],q,r['shift'])==correct})
    summaries = {}
    for reader in READERS:
        summaries[reader] = {}
        for a,b in ARMS:
            selected = [r for r in scored if (r['reader'],r['arm'],r['budget'])==(reader,a,b)]
            byq = defaultdict(list)
            for r in selected: byq[r['question_id']].append(r)
            complete = [v for v in byq.values() if len(v)==4]
            all_valid = [v for v in complete if all(r['prediction'] is not None for r in v)]
            summaries[reader][f'{a}:{b}'] = {
                'presentations':len(selected),'old_correct':sum(r['old_correct'] for r in selected),
                'generated_correct':sum(r['correct'] for r in selected),'valid':sum(r['prediction'] is not None for r in selected),
                'length_truncated':sum(r['length_truncated'] for r in selected),
                'agreement_with_old':sum(r['prediction']==r['old_prediction'] for r in selected),
                'first_step_restricted_argmax_agrees_with_old':sum(r['first_step_label_argmax']==r['old_prediction'] for r in selected),
                'mean_first_step_label_probability_mass':sum(r['first_step_label_probability_mass'] for r in selected)/len(selected) if selected else None,
                'complete_questions':len(complete),'all_valid_questions':len(all_valid),
                'generated_semantically_stable_questions':sum(len({r['semantic_prediction'] for r in v})==1 for v in all_valid),
                'old_semantically_stable_questions':sum(len({semantic(r['old_prediction'],qmap[r['question_id']],r['shift']) for r in v})==1 for v in complete),
                'generated_label_histogram':dict(Counter(r['prediction'] or 'INVALID' for r in selected)),
                'old_label_histogram':dict(Counter(r['old_prediction'] for r in selected))}
    value = {'status':'completed-choice-protocol-audit' if len(rows)==len(valid_keys) else 'incomplete-choice-protocol-audit',
             'records':len(rows),'expected_records':len(valid_keys),'questions':len(queries),'source_groups':m['source_groups'],
             'by_reader':summaries,'actual_generate_calls':len(rows),'model_seconds':sum(r['usage']['elapsed_seconds'] for r in rows),
             'invalid_counted_in_denominator':True,'judge_used':False,'reserved_validation_used':False,
             'future_protocol_selected_on_these_scores':False,'formal_benchmark_score':False}
    save(output/'report.json',value)
    save(output/'status.json',{k:value[k] for k in ('status','records','expected_records')})
    (output/'scored.jsonl').write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored).encode('utf-8'))
    print('CHOICE_AUDIT_REPORT',value['status'],len(rows),flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--reader',choices=READERS,default=READERS[0]); args = parser.parse_args()
    if args.command=='prepare': return prepare()
    if args.command=='report': return report()
    from .__main__ import offline_environment
    from .backend import TransformersBackend
    offline_environment()
    return run(TransformersBackend(ROOT/'models'/args.reader))


if __name__=='__main__': main()
