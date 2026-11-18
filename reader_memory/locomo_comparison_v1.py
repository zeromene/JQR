"""Frozen LoCoMo raw-history transfer of the ten MuSiQue comparison conditions.

Development is the existing 64 questions. Heldout is every category 1--4 QA in
the eight reserved histories, opened only after code freeze and the development
technical gate. No compact-memory or official full-SimpleMem replication claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import statistics
import time

from . import full_comparison_v1 as engine
from . import baseline_bootstrap_v2 as native
from .schema import fingerprint as fp, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'research/locomo_comparison_v1'
RUN = ROOT/'runs/locomo_comparison_v1'
BASE = ROOT/'research/baseline_bootstrap_v2'
MAIN_RUN = ROOT/'runs/full_comparison_v1_repair1'
MAIN_SIGNATURE = 'd94204c5f5d74f2ae5c27ef3c98cc558ef89e5795be507a4f569660ab32a7b43'
SCOPES = ('development', 'heldout')
METHODS, BUDGETS = engine.METHODS, engine.BUDGETS
METHOD_CONFIG = dict(engine.CONFIG)
CONFIG = dict(version=1, dataset='LoCoMo10', dataset_sha256='79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4',
    development=['conv-43','conv-44'], development_questions=64,
    heldout=['conv-49','conv-26','conv-41','conv-47','conv-30','conv-50','conv-42','conv-48'],
    heldout_questions='all native categories 1--4; category5 excluded; no subsampling',
    methods=list(METHODS), budgets=list(BUDGETS),
    corpus='nonoverlapping packets of four native turns; sentence-level speaker/date/source metadata',
    candidates='same Qwen3-Embedding-0.6B top20 raw packets for every method; all-history retrieval',
    embedding_query_instruction=engine.CONFIG['dense_query_instruction'],
    paragraph_title='conversation identity; exact metadata repeated on each original sentence',
    selection='unchanged frozen MuSiQue ten methods including trained Beam2, predicted 1--3 hops and one-anchor guard',
    reader='Qwen3.5-9B', reader_variant='format_only', output_tokens=2048,
    decoding='greedy-thinking_false', retries=0, training=False,
    metric='category-aware native LoCoMo F1; failures zero; no MuSiQue scorer',
    primary='heldout question-macro F1 averaged over all four budgets',
    secondary='all budgets, categories, per-history and history-balanced macro; paired history bootstrap',
    uncertainty='8 independent heldout histories; 2000 paired history bootstrap samples, seed29',
    gate='technical completion and identity only; no performance threshold or tuning',
    scope='raw-history transfer; no generated-memory or full official SimpleMem replication')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if load(path) != value:
            raise ValueError('Refuse overwrite: '+str(path))
    else:
        write_json(path, value)


def jsonl(path, rows):
    path = Path(path)
    raw = ''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows).encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Refuse overwrite: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.building')
    temp.write_bytes(raw);temp.replace(path)


def seal(value):
    return dict(value,signature=fp(value))


def check_seal(value):
    if value['signature'] != fp({k:v for k,v in value.items() if k!='signature'}):
        raise ValueError('Invalid identity signature')


def scope_paths(scope):
    if scope not in SCOPES:
        raise ValueError('Unknown scope')
    return PLAN/scope, RUN/scope


def freeze():
    if (PLAN/'manifest.json').exists():
        return verify_code()
    split = load(BASE/'split.json')
    if split['development'] != CONFIG['development'] or split['heldout'] != CONFIG['heldout']:
        raise ValueError('Existing conversation split differs')
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'reader_memory').rglob('*.py') if '__pycache__' not in p.parts]
    names += ['scripts/locomo_comparison_v1_queue.py','tests/test_locomo_comparison_v1.py',
        'tools/locomo_comparison_v1_remote.py','tools/prepare_replacement_learning_v1.py',
        'research/locomo_comparison_v1/PROTOCOL.md',
        'research/path_retrieval_pilot_v1/sources/beam_retriever/retrieval/retriever_model.py',
        'third_party/simplemem_official/simplemem/core/answer_generator.py',
        'runs/path_retrieval_execution_v1/best.pt',
        'research/baseline_bootstrap_v2/split.json','research/baseline_bootstrap_v2/scorer_provenance.json',
        'research/baseline_bootstrap_v2/public/queries.json','research/baseline_bootstrap_v2/public/conversations.json']
    for model in ('Qwen3.5-9B','Qwen3-Embedding-0.6B','Qwen3-Reranker-4B','DeBERTa-v3-base',
                  'RECOMP-NQ-extractive','RECOMP-HotpotQA-extractive','llmlingua2-small'):
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'models'/model).iterdir()
                  if p.is_file() and (p.suffix in ('.json','.txt','.model','.jinja','.py') or p.name=='.verified.json')]
    value = seal(dict(config=CONFIG,parent_method_config=METHOD_CONFIG,main_signature=MAIN_SIGNATURE,runtime=runtime(),
        files={n:sha(ROOT/n) for n in sorted(set(names))},
        reader_marker_sha256=sha(ROOT/'models/Qwen3.5-9B/.verified.json'),
        heldout_contents_not_opened=True))
    immutable(PLAN/'manifest.json',value)
    return value['signature']


def verify_code():
    value = load(PLAN/'manifest.json');check_seal(value)
    if value['config'] != CONFIG or value['parent_method_config'] != METHOD_CONFIG:
        raise ValueError('Frozen algorithm/configuration changed')
    if value['runtime']!=runtime():raise ValueError('Frozen runtime changed')
    for n,digest in value['files'].items():
        if sha(ROOT/n) != digest:
            raise ValueError('Frozen dependency changed: '+n)
    return value['signature']


def runtime():
    return {name:importlib.metadata.version(name) for name in ('torch','transformers','llmlingua','sentencepiece','numpy','nltk')}


def packets(conversation):
    """Each selected sentence retains its own native temporal/speaker context."""
    turns=conversation['turns'];result=[];origin=[]
    for start in range(0,len(turns),4):
        pieces=[];spans=[];provenance=[];offset=0
        for turn in turns[start:start+4]:
            for span in engine.recomp.core.sentence_spans(turn['content']):
                text=span['text'].strip()
                if not text:
                    continue
                prefix=f"[{turn['source_id']}] [{turn['timestamp']}] {turn['speaker']}: "
                quoted=prefix+text
                if pieces:offset+=1
                spans.append(dict(start=offset,end=offset+len(quoted),text=quoted))
                pieces.append(quoted);offset+=len(quoted)
                provenance.append(dict(source_id=turn['source_id'],original_text=text))
        if not pieces:
            continue
        pid=str(len(result))
        result.append(dict(source_id=pid,title='Conversation '+conversation['conversation_id'],
            timestamp='per-sentence native timestamps',text='\n'.join(pieces),spans=spans))
        origin.append(dict(packet_id=pid,turn_ids=[t['source_id'] for t in turns[start:start+4]],sentences=provenance))
    if len(result)<20:
        raise ValueError('Fewer than20 real packets; no fabricated padding')
    return result,origin


def prepare(scope):
    code=verify_code();plan,run=scope_paths(scope)
    if (plan/'inputs.json').exists():
        verify_inputs(scope);return
    if scope=='development':
        conversations=load(BASE/'public/conversations.json')
        queries=load(BASE/'public/queries.json')
        refs=load(BASE/'private/references.json')
    else:
        gate=load(RUN/'development/technical_gate.json')
        if gate['code_signature']!=code or not gate['passed']:
            raise ValueError('Development technical gate has not passed')
        if sha(RUN/'development/report.json')!=gate['report_sha256']:
            raise ValueError('Development gate report changed')
        if sha(ROOT/'data/locomo/locomo10.json')!=CONFIG['dataset_sha256']:
            raise ValueError('Native dataset changed')
        samples=load(ROOT/'data/locomo/locomo10.json')
        by={s['sample_id']:s for s in samples};conversations=[];queries=[];refs=[]
        for cid in CONFIG['heldout']:
            sample=by[cid]
            conversations.append(dict(conversation_id=cid,turns=native.source_turns(sample)))
            for index,qa in enumerate(sample['qa']):
                if qa['category'] not in (1,2,3,4):continue
                qid=f'{cid}:qa{index}'
                queries.append(dict(question_id=qid,conversation_id=cid,question=qa['question']))
                refs.append(dict(question_id=qid,answer=str(qa['answer']),category=qa['category'],evidence=qa.get('evidence',[])))
    queries=[{k:q[k] for k in ('question_id','conversation_id','question')} for q in queries]
    if scope=='development' and len(queries)!=64:
        raise ValueError('Development question panel changed')
    if len({q['question_id'] for q in queries})!=len(queries):raise ValueError('Duplicate question')
    immutable(plan/'public/queries.json',queries)
    immutable(plan/'private/references.json',{r['question_id']:r for r in refs})
    for conv in conversations:
        values,origin=packets(conv);cid=conv['conversation_id']
        immutable(plan/'public/corpus'/(cid+'.json'),values)
        immutable(plan/'public/origin'/(cid+'.json'),origin)
    names=['public/queries.json','private/references.json']
    names += [str(p.relative_to(plan)) for part in ('corpus','origin') for p in sorted((plan/'public'/part).glob('*.json'))]
    value=seal(dict(code_signature=code,scope=scope,questions=len(queries),conversations=[c['conversation_id'] for c in conversations],
        files={n:sha(plan/n) for n in names},private_labels_used_for_selection=False))
    immutable(plan/'inputs.json',value)
    print('LOCOMO_INPUTS',scope,len(queries),flush=True)


def verify_inputs(scope,private=False):
    plan,_=scope_paths(scope);value=load(plan/'inputs.json');check_seal(value)
    if value['code_signature']!=verify_code():raise ValueError('Input code identity changed')
    for n,digest in value['files'].items():
        if not n.startswith('private/') or private:
            if sha(plan/n)!=digest:raise ValueError('Input changed: '+n)
    return value


def preselect(scope):
    inputs=verify_inputs(scope);plan,run=scope_paths(scope)
    if (plan/'selection_inputs.json').exists():
        verify_selection(scope);return
    import torch
    from transformers import AutoModel,AutoTokenizer
    engine.torch_setup()
    folder=ROOT/'models/Qwen3-Embedding-0.6B'
    tok=AutoTokenizer.from_pretrained(str(folder),local_files_only=True,padding_side='left',trust_remote_code=False)
    model=AutoModel.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    queries=load(plan/'public/queries.json');sources=[];provenance=[]
    def encode(texts):
        vectors=[];lengths=[];started=time.perf_counter()
        with torch.inference_mode():
            for first in range(0,len(texts),16):
                batch=tok(texts[first:first+16],padding=True,truncation=False,return_tensors='pt').to('cuda')
                if batch['input_ids'].shape[-1]>8192:raise ValueError('Coarse retrieval input exceeds8192; no clipping')
                hidden=model(**batch,use_cache=False).last_hidden_state
                if not bool(torch.all(batch['attention_mask'][:,-1]==1)):raise ValueError('Pooling mask mismatch')
                vectors.append(torch.nn.functional.normalize(hidden[:,-1].float(),dim=1).cpu())
                lengths.extend(batch['attention_mask'].sum(-1).cpu().tolist())
        return torch.cat(vectors),dict(seconds=time.perf_counter()-started,input_tokens=sum(lengths),forward_calls=(len(texts)+15)//16)
    for cid in inputs['conversations']:
        cache=run/'preselect'/(cid+'.json')
        corpus=load(plan/'public/corpus'/(cid+'.json'));qs=[q for q in queries if q['conversation_id']==cid]
        if cache.exists():
            value=load(cache)
            if value['input_signature']!=inputs['signature']:raise ValueError('Coarse cache changed')
        else:
            docs,dc=encode([p['title']+': '+p['text'] for p in corpus])
            qv,qc=encode([f"Instruct: {CONFIG['embedding_query_instruction']}\nQuery: {q['question']}" for q in qs])
            sims=(qv@docs.T).tolist();entries=[]
            for q,scores in zip(qs,sims,strict=True):
                ids=sorted(range(len(corpus)),key=lambda i:(-scores[i],i))[:20]
                entries.append(dict(question_id=q['question_id'],packet_ids=ids,scores=[scores[i] for i in ids]))
            value=dict(input_signature=inputs['signature'],entries=entries,costs=dict(corpus=dc,queries=qc))
            immutable(cache,value)
        by={q['question_id']:q for q in qs}
        if {v['question_id'] for v in value['entries']}!=set(by):raise ValueError('Incomplete coarse cache')
        for entry in value['entries']:
            q=by[entry['question_id']];ids=entry['packet_ids']
            if len(ids)!=20 or len(set(ids))!=20:raise ValueError('Invalid candidate selection')
            paras=[dict(corpus[index],source_id=str(j)) for j,index in enumerate(ids)]
            source=dict(question_id=q['question_id'],native_id=q['question_id'],mode='multi',question=q['question'],paragraphs=paras)
            engine.beam.validate_source(source);sources.append(source)
            provenance.append(dict(**entry,conversation_id=cid))
        print('LOCOMO_COARSE',scope,cid,len(qs),flush=True)
    order={q['question_id']:i for i,q in enumerate(queries)}
    sources.sort(key=lambda s:order[s['question_id']]);provenance.sort(key=lambda s:order[s['question_id']])
    jsonl(plan/'public/source_inputs.jsonl',sources);jsonl(plan/'public/pool_provenance.jsonl',provenance)
    immutable(plan/'preparation.json',dict(reader_marker_sha256=load(PLAN/'manifest.json')['reader_marker_sha256'],questions=len(sources)))
    names=['public/source_inputs.jsonl','public/pool_provenance.jsonl','preparation.json']
    value=seal(dict(code_signature=inputs['code_signature'],input_signature=inputs['signature'],questions=len(sources),
        files={n:sha(plan/n) for n in names},coarse_cache_files={str(p.relative_to(ROOT)):sha(p) for p in sorted((run/'preselect').glob('*.json'))}))
    immutable(plan/'selection_inputs.json',value)
    del model;gc.collect();torch.cuda.empty_cache()


def verify_selection(scope):
    inputs=verify_inputs(scope);plan,_=scope_paths(scope);value=load(plan/'selection_inputs.json');check_seal(value)
    if value['input_signature']!=inputs['signature']:raise ValueError('Selection source identity changed')
    for n,digest in value['files'].items():
        if sha(plan/n)!=digest:raise ValueError('Selected input changed: '+n)
    for n,digest in value['coarse_cache_files'].items():
        if sha(ROOT/n)!=digest:raise ValueError('Coarse scores changed: '+n)
    return value['signature']


@contextmanager
def bind(scope):
    """Process-local reuse only; never edits or rebinds the active main process."""
    plan,run=scope_paths(scope);n=load(plan/'inputs.json')['questions']
    original=engine.PLAN,engine.RUN,engine.CONFIG,engine.verify
    engine.PLAN,engine.RUN=plan,run
    engine.CONFIG=dict(original[2],questions=n,conditions=n*len(METHODS)*len(BUDGETS))
    engine.verify=lambda:verify_selection(scope)
    try:yield
    finally:engine.PLAN,engine.RUN,engine.CONFIG,engine.verify=original


def select_phase(scope,phase):
    with bind(scope):
        marker=engine.RUN/(phase+'_complete.json')
        if phase!='assemble' and marker.exists():
            sid=verify_selection(scope)
            if load(marker)['signature']!=sid:raise ValueError('Completed phase identity changed')
            if not all(engine.saved(phase,s,sid) for s in engine.sources()):raise ValueError('Completed phase cache missing')
            return
        if phase in ('recomp_sentence','recomp_hotpot_sentence'):engine.recomp_select(phase)
        elif phase=='assemble':
            if not (engine.RUN/'selection_complete.json').exists():engine.assemble()
            else:
                if sha(engine.RUN/'qa_plans.jsonl')!=load(engine.RUN/'selection_complete.json')['plans_sha256']:
                    raise ValueError('Committed QA plans changed')
        else:getattr(engine,phase)()


def audit_plans(plans,queries,counter,signature):
    expected={(q['question_id'],engine.arm(m,b)) for q in queries for m in METHODS for b in BUDGETS}
    seen=set();by={q['question_id']:q for q in queries}
    for p in plans:
        key=p['question_id'],p['arm']
        if key in seen or key not in expected:raise ValueError('Duplicate or unknown plan condition')
        seen.add(key)
        if p['signature']!=signature or p['question']!=by[p['question_id']]['question']:
            raise ValueError('Plan identity/query mismatch')
        if p['arm']!=engine.arm(p['method'],p['evidence_budget']):raise ValueError('Plan method/budget mismatch')
        if counter.count(p['evidence_text'])!=p['evidence_tokens'] or p['evidence_tokens']>p['evidence_budget']:
            raise ValueError('Plan token contract failed')
        if fp(p['evidence_text'])!=p['evidence_sha256']:raise ValueError('Plan evidence hash mismatch')
    if seen!=expected:raise ValueError('Missing plan conditions')
    return len(seen)


def audit(scope):
    from .tokenization import TokenCounter
    signature=verify_selection(scope);plan,run=scope_paths(scope)
    meta=load(run/'selection_complete.json')
    if sha(run/'qa_plans.jsonl')!=meta['plans_sha256']:raise ValueError('Committed plans changed')
    n=audit_plans(read_jsonl(run/'qa_plans.jsonl'),load(plan/'public/queries.json'),TokenCounter(ROOT/'models/Qwen3.5-9B'),signature)
    immutable(run/'plan_audit.json',dict(signature=signature,passed=True,conditions=n,plans_sha256=meta['plans_sha256']))
    print('LOCOMO_PLAN_AUDIT',scope,n,flush=True)


def preflight(scope):
    from .tokenization import TokenCounter
    from .evidence_acquisition_v2 import parse_answer
    if parse_answer('{"answer":"Berlin"}')!='Berlin':raise ValueError('Native parser contract failed')
    if native.native_f1('Berlin','Berlin',2)!=1. or native.native_f1('2001','2001; other',3)!=1.:
        raise ValueError('Native scorer contract failed')
    inputs=verify_inputs(scope);plan,run=scope_paths(scope)
    counter=TokenCounter(ROOT/'models/Qwen3.5-9B');tokenizer=engine.beam.tokenizer()
    queries=load(plan/'public/queries.json');counts={};smoke=0
    for cid in inputs['conversations']:
        corpus=load(plan/'public/corpus'/(cid+'.json'));counts[cid]=len(corpus)
        for p in corpus:
            if any(p['text'][s['start']:s['end']]!=s['text'] for s in p['spans']):raise ValueError('Corpus offset mismatch')
        q=next(q for q in queries if q['conversation_id']==cid)
        source=dict(question_id=q['question_id'],native_id=q['question_id'],mode='multi',question=q['question'],paragraphs=corpus[:20])
        engine.beam.validate_source(source)
        # Synthetic scores exercise serialization/budget plumbing, never saved as model results.
        scores={uid:-float(i) for i,(uid,_) in enumerate(engine.recomp.units(source))}
        for budget in BUDGETS:
            plans,_=engine.retained_plans(source,scores,['0','1'],counter,budget)
            for p in plans:
                if counter.count(p['evidence_text'])!=p['evidence_tokens'] or p['evidence_tokens']>budget:
                    raise ValueError('Packing smoke token mismatch')
                smoke+=1
    for q in queries:
        text=q['question'][:-1] if q['question'].endswith('?') else q['question']
        if not 0<len(tokenizer.encode(text,add_special_tokens=False))<256:raise ValueError('Question unsupported by frozen retriever')
    immutable(run/'preflight.json',dict(passed=True,input_signature=inputs['signature'],scope=scope,
        questions=len(queries),corpus_packet_counts=counts,synthetic_packing_conditions=smoke,
        exact_reader_tokenizer=True,new_model_forwards=0,private_references_read=False))
    print('LOCOMO_CPU_PREFLIGHT',scope,len(queries),smoke,flush=True)


def qa(scope,endpoint):
    with bind(scope):engine.qa(endpoint)


def score(scope):
    inputs=verify_inputs(scope,private=True);sid=verify_selection(scope);plan,run=scope_paths(scope)
    complete=load(run/'qa_complete.json');meta=load(run/'selection_complete.json')
    if complete['signature']!=sid or sha(run/'qa_plans.jsonl')!=meta['plans_sha256']:
        raise ValueError('QA completion identity changed')
    if (run/'report.json').exists():
        report=load(run/'report.json')
        if report['signature']!=sid:raise ValueError('Report identity changed')
        if scope=='development':
            immutable(run/'technical_gate.json',dict(passed=True,code_signature=inputs['code_signature'],
                data_signature=sid,conditions=report['conditions'],report_sha256=sha(run/'report.json'),performance_threshold=None))
        return report
    refs=load(plan/'private/references.json');queries=load(plan/'public/queries.json')
    clusters={q['question_id']:q['conversation_id'] for q in queries}
    db=sqlite3.connect((run/'qa/ledger.sqlite3').as_uri()+'?mode=ro',uri=True)
    preds={(q,a):json.loads(v) for q,a,v in db.execute('SELECT q,arm,value FROM predictions')};db.close()
    expected={(q['question_id'],engine.arm(m,b)) for q in queries for m in METHODS for b in BUDGETS}
    if set(preds)!=expected:raise ValueError('Incomplete answer matrix')
    rows=[]
    for p in read_jsonl(run/'qa_plans.jsonl'):
        q=p['question_id'];r=preds[q,p['arm']]
        if r['plan_sha256']!=fp(p) or r['signature']!=sid:raise ValueError('Answer/plan mismatch')
        value=native.native_f1(r['hypothesis'],refs[q]['answer'],refs[q]['category']) if r['status']=='ok' else 0.
        rows.append(dict(**r,category=refs[q]['category'],conversation_id=clusters[q],f1=value,evidence_tokens=p['evidence_tokens']))
    jsonl(run/'scored.jsonl',rows)
    values={(r['question_id'],r['method'],r['budget']):r['f1'] for r in rows}
    summaries=[];comparisons=[]
    for budget in BUDGETS:
        for group in ('overall',1,2,3,4,*inputs['conversations']):
            def subset(m):
                return {r['question_id']:r for r in rows if r['method']==m and r['budget']==budget and
                    (group=='overall' or r['category']==group or r['conversation_id']==group)}
            for m in METHODS:
                rs=list(subset(m).values())
                if not rs:continue
                summaries.append(dict(method=m,budget=budget,group=group,n=len(rs),f1_points=100*statistics.mean(r['f1'] for r in rs),
                    mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in rs),statuses=dict(Counter(r['status'] for r in rs))))
            if group=='overall':
                for m in METHODS:
                    if m!=engine.anchor.MAIN:
                        comparisons.append(dict(comparator=m,budget=budget,**engine.paired_stats(
                            {q:r['f1'] for q,r in subset(engine.anchor.MAIN).items()},
                            {q:r['f1'] for q,r in subset(m).items()},clusters)))
    macro=[]
    for m in METHODS:
        rs=[r for r in rows if r['method']==m]
        history=[statistics.mean(r['f1'] for r in rs if r['conversation_id']==cid) for cid in inputs['conversations']]
        macro.append(dict(method=m,four_budget_macro_f1_points=100*statistics.mean(r['f1'] for r in rs),
            history_balanced_macro_f1_points=100*statistics.mean(history)))
        if m!=engine.anchor.MAIN:
            perq=lambda method:{q['question_id']:statistics.mean(values[q['question_id'],method,b] for b in BUDGETS) for q in queries}
            comparisons.append(dict(comparator=m,budget='macro',**engine.paired_stats(perq(engine.anchor.MAIN),perq(m),clusters)))
    costs={}
    for phase in ('retrieve','dense','qwen','llmlingua','recomp_sentence','recomp_hotpot_sentence'):
        totals=Counter()
        for path in (run/phase).glob('*.json'):
            v=load(path);totals.update({k:x for k,x in v.get('costs',{}).items() if isinstance(x,(int,float))})
            if phase=='retrieve':totals['seconds']+=v['seconds']
        costs[phase]=dict(totals)
    coarse=Counter()
    for path in (run/'preselect').glob('*.json'):
        for item in load(path)['costs'].values():coarse.update(item)
    costs['coarse']=dict(coarse)
    reader=Counter()
    for call in read_jsonl(run/'qa/answer_calls.jsonl'):
        reader.update(calls=1,seconds=call['seconds'],**{k:v for k,v in call['result'].get('usage',{}).items() if k in ('prompt_tokens','completion_tokens')})
    costs['reader']=dict(reader)
    report=dict(signature=sid,code_signature=inputs['code_signature'],scope=scope,complete=True,questions=len(queries),
        conditions=len(rows),metric=CONFIG['metric'],scores=summaries,macro_scores=macro,comparisons=comparisons,costs=costs,
        labels_used_for_selection=False,new_training=False,transfer_scope=CONFIG['scope'])
    immutable(run/'report.json',report)
    if scope=='development':
        immutable(run/'technical_gate.json',dict(passed=True,code_signature=inputs['code_signature'],
            data_signature=sid,conditions=len(rows),report_sha256=sha(run/'report.json'),performance_threshold=None))
    print('LOCOMO_SCORED',scope,len(rows),flush=True)
    return report


def status():
    value={}
    for scope in SCOPES:
        plan,run=scope_paths(scope)
        value[scope]={n:load(run/n) for n in ('qa/status.json','qa_complete.json','technical_gate.json') if (run/n).exists()}
        value[scope]['inputs']=load(plan/'inputs.json') if (plan/'inputs.json').exists() else None
    print(json.dumps(value,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['freeze','prepare','preflight','preselect','retrieve','dense','qwen','llmlingua',
        'recomp_sentence','recomp_hotpot_sentence','assemble','audit','qa','score','verify','status'])
    p.add_argument('--scope',choices=SCOPES,default='development');p.add_argument('--endpoint',default='http://127.0.0.1:8039')
    a=p.parse_args()
    if a.phase=='freeze':print(freeze())
    elif a.phase=='verify':print(verify_code())
    elif a.phase=='prepare':prepare(a.scope)
    elif a.phase=='preselect':preselect(a.scope)
    elif a.phase=='preflight':preflight(a.scope)
    elif a.phase=='qa':qa(a.scope,a.endpoint)
    elif a.phase=='audit':audit(a.scope)
    elif a.phase=='score':score(a.scope)
    elif a.phase=='status':status()
    else:select_phase(a.scope,a.phase)


if __name__=='__main__':main()
