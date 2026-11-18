"""Four-reader, common-source representation probe on new training personas.

This stage measures whether reader/representation interactions exist before
training another conditional selector. It is a development experiment only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from .diagnostics import sha256
from .native_choice_pilot import SOURCES, classify, messages, read, record_key, score_row
from .personamem2 import DATA, STUDY as INPUT_STUDY, load_history
from .retrieval import BM25
from .schema import fingerprint, read_jsonl, write_json
from .selection import Choice, render_evidence
from .token_windows import token_windows

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'research/four_reader_probe/v1'
OUTPUT = ROOT/'runs/four_reader_probe_v1'
READERS = ('Qwen3.5-2B','Qwen3.5-4B','Qwen3.5-9B','Phi-4-mini-instruct')
ARMS = (('none',0),*((arm,b) for b in (1024,2048) for arm in ('raw','compact','lead')))


class SharedCounter:
    exact = True

    def __init__(self, counters):
        if not counters or not all(c.exact for c in counters.values()):
            raise ValueError('Every reader requires its own exact tokenizer')
        self.counters = counters

    def count(self, text):
        return max(c.count(text) for c in self.counters.values())

    def costs(self, text):
        return {r:c.count(text) for r,c in self.counters.items()}


def serialize(groups, arm):
    if arm == 'raw':
        return render_evidence([Choice(g.group_id,'raw',g.variants['raw'],tuple(g.source_ids),g.date,0,0) for g in groups])
    if arm not in {'compact','lead'}:
        raise ValueError('Unknown representation')
    pieces = []
    for g in groups:
        for s in g.metadata['spans']:
            text = s['text'] if arm == 'compact' else re.split(r'(?<=[.!?])\s+',s['text'],maxsplit=1)[0]
            date = f" [{s['date']}]" if s['date'] else ''
            pieces.append(f"[{s['source_id']}@{s['start']}:{s['start']+len(text)}]{date} {s['role']}: {text}")
    return '\n'.join(pieces)


def pack(ranked, counter, budget):
    chosen = []
    for g,_ in ranked:
        proposal = [*chosen,g]
        # Greedy BM25 retrieval is shared. All branches receive exactly these
        # same spans in this same order; savings cannot buy extra coverage.
        if all(counter.count(serialize(proposal,a)) <= budget for a in ('raw','compact','lead')):
            chosen = proposal
    values = {}
    for arm in ('raw','compact','lead'):
        text = serialize(chosen,arm)
        values[arm] = {'text':text,'tokens_by_reader':counter.costs(text),
                       'chunk_ids':[g.group_id for g in chosen]}
        if any(c>budget for c in values[arm]['tokens_by_reader'].values()):
            raise ValueError('Final serialization exceeds a reader budget')
    return values, chosen


def hashes():
    return {n:sha256(ROOT/'reader_memory'/n) for n in (*SOURCES,'personamem2.py','four_reader_probe.py')}


def probe_assets(model):
    """Qwen has a separate template file; Phi embeds it in tokenizer config."""
    model=Path(model)
    names=('source_manifest.json','config.json','tokenizer.json','tokenizer_config.json')
    value={n:sha256(model/n) for n in names}
    template=model/'chat_template.jinja'
    value['chat_template.jinja']=sha256(template) if template.is_file() else None
    if not template.is_file() and not read(model/'tokenizer_config.json').get('chat_template'):
        raise ValueError('Reader has no auditable chat template')
    return value


def prepare(counters, study=STUDY):
    study=Path(study)
    if study.exists():
        raise FileExistsError('Preserve frozen probe inputs')
    if set(counters) != set(READERS):
        raise ValueError('Missing reader tokenizer')
    reader_assets={r:probe_assets(ROOT/'models'/r) for r in READERS}
    common = SharedCounter(counters)
    inputs = read(INPUT_STUDY/'probe_inputs.json')
    split = read(INPUT_STUDY/'split.json')
    references = read(INPUT_STUDY/'references.private.json')
    if any(q['source_group'] not in split['train_groups'] for q in inputs):
        raise ValueError('Reserved persona leaked into probe')
    if len({q['question_id'] for q in inputs}) != len(inputs):
        raise ValueError('Duplicate query')
    queries, cache = [], {}
    for q in inputs:
        if q['history_path'] not in cache:
            turns,audit=load_history(q)
            groups=token_windows(turns,common,limit=384)
            cache[q['history_path']]=(groups,audit,turns)
        groups,audit,turns=cache[q['history_path']]
        # Audit every query even when its history chunks were cached.
        _,again=load_history(q)
        if again != audit:
            raise ValueError('History audit changed')
        ranked=BM25(groups).search(q['question'],top_k=24)
        evidence={'none:0':{'text':'','tokens_by_reader':{r:0 for r in READERS},'chunk_ids':[]}}
        selected={}
        for budget in (1024,2048):
            packed,chosen=pack(ranked,common,budget)
            selected[str(budget)]=[g.to_dict() for g in chosen]
            for arm,item in packed.items():
                evidence[f'{arm}:{budget}']=item
        queries.append({**q,'history_audit':audit,'evidence':evidence,'selected_sources':selected})
        print('FOUR_READER_PREPARED',len(queries),'/',len(inputs),flush=True)
    # Confirm label tokenization and every complete chat prompt before inference.
    prompt_max={}
    for reader,c in counters.items():
        labels=[c.tokenizer.encode(x,add_special_tokens=False) for x in 'abcd']
        if any(len(x)!=1 for x in labels) or len({x[0] for x in labels})!=4:
            raise ValueError('Reader lacks distinct single-token lowercase labels')
        maximum=0
        for q in queries:
            for arm,budget in ARMS:
                for shift in q['shifts']:
                    text=c.tokenizer.apply_chat_template(messages(q,arm,budget,shift),tokenize=False,
                            add_generation_prompt=True,enable_thinking=False)
                    maximum=max(maximum,c.count(text))
        if maximum>12288:
            raise ValueError('A complete prompt exceeds the configured reader context')
        prompt_max[reader]=maximum
    study.mkdir(parents=True)
    write_json(study/'queries.json',queries)
    write_json(study/'references.private.json',references)
    paths={name:ROOT/name for name in ('research/FOUR_READER_PROBE_PLAN.md',
        'research/personamem_v2/v1/probe_inputs.json','research/personamem_v2/v1/split.json',
        'research/personamem_v2/v1/references.private.json','data/personamem_v2/source_manifest.json')}
    manifest={'version':'four-reader-probe-v1','frozen_at':datetime.now(timezone.utc).isoformat(),
        'code_hashes':hashes(),'file_hashes':{p:sha256(v) for p,v in paths.items()},
        'queries_sha256':sha256(study/'queries.json'),'references_sha256':sha256(study/'references.private.json'),
        'readers':list(READERS),'arms':list(ARMS),'questions':len(queries),
        'source_groups':len({q['source_group'] for q in queries}),
        'expected_records':len(queries)*len(READERS)*len(ARMS)*4,
        'reader_assets':reader_assets,
        'maximum_preflight_prompt_tokens':prompt_max,
        'packing':'same top-24 BM25 source pool and selected spans in all branches/readers; every final string <= budget under every tokenizer',
        'compact_fidelity':'Every selected source character, role, source ID, offset and nonempty date retained in order; redundant group wrappers removed.',
        'lead_fidelity':'First sentence per identical source fragment; intentionally lossy diagnostic control.',
        'semantic_update_linking_used':False,'learned_selector_trained':False,
        'official_val_or_benchmark_used':False,'old_reserved_personas_or_lme440_used':False,
        'formal_benchmark_score':False,'capability_vector_efficacy_established':False,
        'limitations':['Public LLM-generated training data and 32 source groups; not a held-out benchmark result.',
            'Compact isolates formatting/metadata reduction; it does not validate semantic summarization.',
            'Restricted four-label next-token decoding differs from free generation.',
            'Four cyclic orders do not enumerate all 24 permutations.',
            'Conditional selector, capability vectors and joint memory-budget allocation are not tested here.',
            'Four readers from two model families cannot by themselves identify a general capability representation.']}
    write_json(study/'manifest.json',manifest)
    print('FOUR_READER_FROZEN',manifest['expected_records'],flush=True)
    return manifest


def load(study=STUDY):
    study=Path(study);m=read(study/'manifest.json')
    if m['code_hashes']!=hashes() or any(sha256(ROOT/p)!=h for p,h in m['file_hashes'].items()):
        raise ValueError('Frozen code/source/plan changed')
    if sha256(study/'queries.json')!=m['queries_sha256'] or sha256(study/'references.private.json')!=m['references_sha256']:
        raise ValueError('Frozen inputs changed')
    return m,read(study/'queries.json')


def verify(rows, queries, config):
    qmap={q['question_id']:q for q in queries}
    expected={(r,q['question_id'],a,b,s) for r in READERS for q in queries for a,b in ARMS for s in q['shifts']}
    checked={}
    for row in rows:
        key=record_key(row)
        if key in checked or key not in expected:
            raise ValueError('Duplicate or unknown probe output')
        r,qid,a,b,s=key;q=qmap[qid];item=q['evidence'][f'{a}:{b}'];scores=row['restricted_label_logits']
        if (row['signature']!=fingerprint(config) or row['prompt_hash']!=fingerprint(messages(q,a,b,s))
            or row['evidence_hash']!=fingerprint(item['text']) or row['evidence_tokens']!=item['tokens_by_reader'][r]
            or row['evidence_tokens']>b or len(scores)!=4 or not all(math.isfinite(x) for x in scores)
            or row['prediction']!='abcd'[max(range(4),key=lambda i:scores[i])]):
            raise ValueError('Probe provenance or prediction changed')
        if row['prediction_origin']=='cache-identical-prompt':
            prior=checked.get(tuple(row['cached_from_key']))
            if prior is None or prior['reader']!=r or prior['prompt_hash']!=row['prompt_hash'] or prior['restricted_label_logits']!=scores:
                raise ValueError('Cross-input or cross-reader cache reuse')
        elif row['prediction_origin']!='model-forward':
            raise ValueError('Unknown prediction source')
        checked[key]=row
    return expected


def run(backend,counter,study=STUDY,output=OUTPUT):
    m,queries=load(study);reader=backend.model_name;output=Path(output)
    if reader not in READERS or not counter.exact or probe_assets(backend.path)!=m['reader_assets'][reader]:
        raise ValueError('Wrong reader/tokenizer assets')
    output.mkdir(parents=True,exist_ok=True)
    config={'study_signature':fingerprint(m),'reader_assets':m['reader_assets']}
    if (output/'config.json').exists() and read(output/'config.json')!=config:
        raise ValueError('Incompatible probe resume')
    write_json(output/'config.json',config)
    rows=list(read_jsonl(output/'answers.jsonl')) if (output/'answers.jsonl').exists() else []
    verify(rows,queries,config)
    done={record_key(r) for r in rows};cache={(r['reader'],r['prompt_hash']):r for r in rows}
    with (output/'answers.jsonl').open('a',encoding='utf-8') as stream:
        for q in queries:
            for arm,budget in ARMS:
                item=q['evidence'][f'{arm}:{budget}']
                if counter.count(item['text'])!=item['tokens_by_reader'][reader]:
                    raise ValueError('Actual reader evidence cost changed')
                for shift in q['shifts']:
                    key=(reader,q['question_id'],arm,budget,shift)
                    if key in done:continue
                    prompt=messages(q,arm,budget,shift);ph=fingerprint(prompt);prior=cache.get((reader,ph))
                    result=({'prediction':prior['prediction'],'restricted_label_logits':prior['restricted_label_logits'],
                        'usage':{**prior['usage'],'elapsed_seconds':0.0}} if prior else classify(backend,prompt))
                    row={'reader':reader,'question_id':q['question_id'],'arm':arm,'budget':budget,'shift':shift,
                        'signature':fingerprint(config),'prompt_hash':ph,'evidence_hash':fingerprint(item['text']),
                        'evidence_tokens':item['tokens_by_reader'][reader],
                        'prediction_origin':'cache-identical-prompt' if prior else 'model-forward',
                        'cached_from_key':list(record_key(prior)) if prior else None,**result}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    done.add(key);cache[(reader,ph)]=row
                    if len(done)%112==0:
                        write_json(output/'status.json',{'status':'running','reader':reader,'records':len(done),'expected_records':m['expected_records']})
                        print('FOUR_READER_PROGRESS',reader,len(done),'/',m['expected_records'],flush=True)
    return report(study,output)


def report(study=STUDY,output=OUTPUT):
    m,queries=load(study);output=Path(output);config=read(output/'config.json')
    if config!={'study_signature':fingerprint(m),'reader_assets':m['reader_assets']}:
        raise ValueError('Probe configuration differs')
    rows=list(read_jsonl(output/'answers.jsonl'));expected=verify(rows,queries,config)
    refs={r['question_id']:r for r in read(Path(study)/'references.private.json')};qmap={q['question_id']:q for q in queries}
    scored=[{**r,'source_group':qmap[r['question_id']]['source_group'],
             'correct':score_row(r,qmap[r['question_id']],refs[r['question_id']])} for r in rows]
    summaries={}
    for reader in READERS:
        summaries[reader]={}
        for arm,budget in ARMS:
            selected=[r for r in scored if (r['reader'],r['arm'],r['budget'])==(reader,arm,budget)]
            summaries[reader][f'{arm}:{budget}']={'presentations':len(selected),'correct':sum(r['correct'] for r in selected),
                'accuracy':sum(r['correct'] for r in selected)/len(selected) if selected else None,
                'mean_evidence_tokens':sum(r['evidence_tokens'] for r in selected)/len(selected) if selected else None}
    value={'status':'completed-four-reader-probe' if len(rows)==len(expected) else 'incomplete-four-reader-probe',
        'records':len(rows),'expected_records':len(expected),'questions':m['questions'],'source_groups':m['source_groups'],
        'actual_model_forward_calls':sum(r['prediction_origin']=='model-forward' for r in rows),
        'model_seconds':sum(r['usage']['elapsed_seconds'] for r in rows),'by_reader':summaries,
        'formal_benchmark_score':False,'capability_vector_efficacy_established':False,'limitations':m['limitations']}
    write_json(output/'report.json',value)
    (output/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored),encoding='utf-8')
    write_json(output/'status.json',{k:value[k] for k in ('status','records','expected_records')})
    print('FOUR_READER_REPORT',value['status'],len(rows),'/',len(expected),flush=True)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--reader',choices=READERS,default=READERS[0]);args=parser.parse_args()
    if args.command=='report':return report()
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    if args.command=='prepare':return prepare({r:TokenCounter(ROOT/'models'/r) for r in READERS})
    from .backend import TransformersBackend
    model=ROOT/'models'/args.reader
    return run(TransformersBackend(model),TokenCounter(model))


if __name__=='__main__':main()
