"""Frozen full MuSiQue dev comparison; no fitting or outcome-driven selection."""
import argparse
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import time

from . import anchor_guard_controls_v1 as anchor
from . import llmlingua_budget_v1 as lingua
from . import path_retrieval_pilot_v1 as beam
from . import reranker_budget_v1 as ranker
from .schema import fingerprint as fp, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/full_comparison_v1'
RUN = ROOT / 'runs/full_comparison_v1'
DATA = ROOT / 'data/musique_full_eval_v1/musique_ans_v1.0_dev.jsonl'
load, sha, immutable, recomp = anchor.load, anchor.sha, anchor.immutable, anchor.recomp
BUDGETS = (256, 512, 1024, 2048)
METHODS = ('dense_sentence', ranker.NEW_METHOD, 'recomp_sentence', 'recomp_hotpot_sentence',
    *lingua.NEW_METHODS, 'qwen_path_compress', anchor.PREVIOUS, anchor.INDEPENDENT, anchor.MAIN)
CONFIG = dict(version=1, dataset='MuSiQue-Ans v1.0 official public dev via pinned mirror', questions=2417,
    source_sha256='15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b',
    methods=list(METHODS), budgets=list(BUDGETS), conditions=96680, reader='Qwen3.5-9B',
    output_tokens=2048, reader_variant='format_only', decoding='greedy-thinking_false',
    trained_checkpoint_sha256='eef0b33fabbf69d998d3d339685cecfc2e7aafe61da4e72572c22d8396184bdd',
    dense_query_instruction='Given a web search query, retrieve relevant passages that answer the query',
    dense_batch=16, dense_precision='GPU BF16 SDPA; FP32 last-token L2/cosine',
    recomp_device='cpu', recomp_precision='FP32/eager', recomp_batch=32,
    recomp_max_length=512, recomp_truncation='explicit native model input clipping only, recorded',
    selection_before_qa=True, new_training=False, retries=0, bootstrap=2000, bootstrap_seed=29,
    primary_aggregate='four-budget macro; all individual budgets and 2/3/4 hops reported',
    selection_failure='zero score; retain full matrix; never omit question',
    source_independent_from_method_development=True, original_mechanism_proven=False, paper_ready=False)
CODE_FILES = ['reader_memory/full_comparison_v1.py', 'tools/prepare_full_comparison_v1.py',
    'tools/full_comparison_v1_remote.py', 'scripts/full_comparison_v1_queue.py',
    'tests/test_full_comparison_v1.py', 'tools/download_musique_dev_v1.py',
    'research/full_comparison_v1/PROTOCOL.md', 'research/full_comparison_v1/sources/metrics_answer.py']


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def arm(method, budget):
    if method not in METHODS or budget not in BUDGETS:
        raise ValueError('Unregistered main condition')
    return method+'__b'+str(budget)


def freeze_code():
    parent_signature = anchor.verify()
    lingua.model_files()
    model_paths = ['models/Qwen3.5-9B/.verified.json', 'models/Qwen3-Reranker-4B/artifact_manifest.json',
        'models/RECOMP-NQ-extractive/artifact_manifest.json', 'models/RECOMP-HotpotQA-extractive/artifact_manifest.json',
        'models/Qwen3-Embedding-0.6B/source_manifest.json', 'models/llmlingua2-small/source_manifest.json',
        'runs/path_retrieval_execution_v1/training_complete.json', 'runs/path_retrieval_execution_v1/best.pt']
    if sha(ROOT/model_paths[-1]) != CONFIG['trained_checkpoint_sha256']:
        raise ValueError('Frozen Beam checkpoint changed')
    value = dict(config=CONFIG, parent_signature=parent_signature,
        frozen_before_dev_preprocessing=True, files={n:sha(ROOT/n) for n in CODE_FILES+model_paths})
    value['signature'] = fp(value)
    immutable(PLAN/'code_manifest.json', value)
    print('FULL_COMPARISON_CODE_FROZEN', value['signature'], flush=True)


def verify_code():
    value = load(PLAN/'code_manifest.json')
    if value['config'] != CONFIG or value['signature'] != fp({k:v for k,v in value.items() if k!='signature'}):
        raise ValueError('Main experiment identity changed')
    for n, digest in value['files'].items():
        if sha(ROOT/n) != digest:
            raise ValueError('Frozen code/model changed: '+n)
    if anchor.verify() != value['parent_signature']:
        raise ValueError('Retained computational method changed')
    return value['signature']


def freeze_data():
    code = verify_code()
    if sha(DATA) != CONFIG['source_sha256']:
        raise ValueError('Public dev provenance changed')
    names = ['public/source_inputs.jsonl', 'private/references.json', 'preparation.json', 'overlap_audit.json']
    value = dict(code_signature=code, files={n:sha(PLAN/n) for n in names}, raw_sha256=sha(DATA))
    value['signature'] = fp(value)
    immutable(PLAN/'manifest.json', value)
    print('FULL_COMPARISON_DATA_FROZEN', value['signature'], flush=True)


def verify():
    value = load(PLAN/'manifest.json')
    if value['code_signature'] != verify_code() or value['signature'] != fp({k:v for k,v in value.items() if k!='signature'}):
        raise ValueError('Bound dataset identity changed')
    for n, digest in value['files'].items():
        if sha(PLAN/n) != digest:
            raise ValueError('Frozen evaluation data changed: '+n)
    return value['signature']


def sources():
    yield from read_jsonl(PLAN/'public/source_inputs.jsonl')


def selected_path(phase, source):
    return RUN/phase/(source['question_id']+'.json')


def saved(phase, source, sid):
    path = selected_path(phase, source)
    if not path.exists():
        return None
    value = load(path)
    if value['signature'] != sid or value['source_sha256'] != fp(source):
        raise ValueError('Existing committed selection differs')
    return value


def commit(phase, source, sid, **value):
    immutable(selected_path(phase, source), dict(signature=sid, source_sha256=fp(source), **value))


def progress(phase, count):
    write_json(RUN/(phase+'_status.json'), dict(completed=count, expected=CONFIG['questions'], time=time.time()))
    print('FULL_'+phase.upper(), count, '/', CONFIG['questions'], flush=True)


def torch_setup():
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(30)
    torch.cuda.manual_seed_all(30)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def finished(phase, sid):
    immutable(RUN/(phase+'_complete.json'), dict(signature=sid, complete=True, time=time.time()))


def preflight():
    sid = verify()
    rows = list(sources())
    if len(rows) != 2417 or len({s['question_id'] for s in rows}) != 2417:
        raise ValueError('Full public dev matrix incomplete')
    for s in rows:
        beam.validate_source(s)
        if s['mode'] != 'multi':
            raise ValueError('Native main questions only')
    if load(PLAN/'overlap_audit.json')['exact_training_question_overlaps']:
        raise ValueError('Exact training question duplicate: inspect before any inference')
    ranker.model_files()
    lingua.model_files()
    beam.check_models()
    # Check the extra embedding assets against their downloaded upstream file hashes.
    folder = ROOT/'models/Qwen3-Embedding-0.6B'
    listed = {v['Path']:v for v in load(folder/'source_manifest.json')['Data']['Files']}
    for name in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json'):
        if sha(folder/name) != listed[name]['Sha256']:
            raise ValueError('Dense model changed: '+name)
    immutable(RUN/'preflight.json', dict(signature=sid, passed=True, questions=len(rows),
        conditions=96680, no_dev_answer_feedback=True, original_main_questions_only=True))
    print('FULL_DEV_PREFLIGHT_OK', len(rows), flush=True)


def retrieve():
    sid = verify()
    torch = torch_setup()
    tok, model = beam.make_model('cpu')
    model.float().to('cuda')
    model.load_state_dict(torch.load(ROOT/'runs/path_retrieval_execution_v1/best.pt', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    for i, source in enumerate(sources(), 1):
        if saved('retrieve', source, sid):
            continue
        value = beam.infer(torch, tok, model, source)
        commit('retrieve', source, sid, **value)
        progress('retrieve', i)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    finished('retrieve', sid)


def rename(plan, method, budget):
    plan.update(method=method, arm=arm(method,budget), evidence_budget=budget)
    return plan


def fail_plan(source, method, budget, error):
    return dict(**{k:source[k] for k in ('question_id','native_id','mode','question')}, method=method,
        arm=arm(method,budget), evidence_budget=budget, reader_variant='format_only', evidence_text='',
        evidence_tokens=0, evidence_sha256=fp(''), selected_ids=[], selected_ranges={}, selected_spans=[],
        fully_preserved_sources=[], selection_error=error, private_inputs_read=False, diagnostic_only=False)


def packed(source, scores, method, counter):
    plans = []
    for budget in BUDGETS:
        try:
            p, _ = recomp.pack(source, scores, 'recomp_sentence', counter, budget)
        except ValueError as exc:
            if not empty_packing_error(exc):
                raise
            plans.append(fail_plan(source,method,budget,str(exc)))
            continue
        plans.append(rename(p, method, budget) if p['evidence_text'].strip() else
            fail_plan(source,method,budget,'No complete source sentence fits the evidence budget'))
    return plans


def empty_packing_error(exc):
    return str(exc) in ('Empty or over-budget selection; never truncate',
        'Empty or over-budget selection; never slice sentences')


def retained_plans(source,scores,seed_ids,counter,budget):
    """Call the same frozen packing functions without an unrelated pure-path side effect."""
    try:
        baseline=ranker.sentence_plan(source,scores,counter,budget)
    except ValueError as exc:
        if not empty_packing_error(exc):
            raise
        return [fail_plan(source,m,budget,str(exc)) for m in anchor.METHODS],[]
    if not baseline['evidence_text'].strip():
        return [fail_plan(source,m,budget,'No complete source sentence fits the evidence budget')
            for m in anchor.METHODS],[]
    independent=anchor.packing.independent_seeds(source,scores,len(seed_ids))
    try:
        candidate,pt=anchor.packing.seed_fill(source,scores,seed_ids,counter,anchor.PREVIOUS,baseline['evidence_tokens'])
        candidate.update(method=anchor.PREVIOUS,arm=arm(anchor.PREVIOUS,budget),evidence_budget=budget)
        main,mt=anchor.guarded_plan(source,scores,baseline,candidate,counter,budget,anchor.MAIN)
    except ValueError as exc:
        if not empty_packing_error(exc):
            raise
        candidate,main=[fail_plan(source,m,budget,str(exc)) for m in (anchor.PREVIOUS,anchor.MAIN)]
        pt=mt=dict(selection_error=str(exc))
    try:
        other,it=anchor.packing.seed_fill(source,scores,independent,counter,anchor.INDEPENDENT,baseline['evidence_tokens'])
        indep,gt=anchor.guarded_plan(source,scores,baseline,other,counter,budget,anchor.INDEPENDENT)
    except ValueError as exc:
        if not empty_packing_error(exc):
            raise
        indep=fail_plan(source,anchor.INDEPENDENT,budget,str(exc))
        it=gt=dict(selection_error=str(exc))
    return [baseline,candidate,main,indep],[dict(path_packing=pt,independent_packing=it,main=mt,independent=gt)]


def pure_path_plan(source,scores,seed_ids,counter,budget):
    try:
        plan,trace=anchor.parent.packing.compressed_path(source,scores,seed_ids,counter,budget)
        return rename(plan,'qwen_path_compress',budget),trace
    except ValueError as exc:
        if not empty_packing_error(exc):
            raise
        return fail_plan(source,'qwen_path_compress',budget,str(exc)),dict(selection_error=str(exc))


def dense():
    sid = verify()
    torch = torch_setup()
    from transformers import AutoModel, AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    folder = ROOT/'models/Qwen3-Embedding-0.6B'
    tok = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, padding_side='left', trust_remote_code=False)
    model = AutoModel.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    counter = ExactCounter()
    for i,s in enumerate(sources(),1):
        if saved('dense',s,sid):
            continue
        docs = ranker.documents(s)
        texts = [f"Instruct: {CONFIG['dense_query_instruction']}\nQuery: {s['question']}"]+[t for _,t in docs]
        vectors, lengths = [], []
        begin = time.perf_counter()
        with torch.inference_mode():
            for start in range(0,len(texts),16):
                batch = tok(texts[start:start+16],padding=True,truncation=False,return_tensors='pt').to('cuda')
                if batch['input_ids'].shape[-1]>8192:
                    raise ValueError('Dense input exceeds predeclared8192 limit')
                lengths.extend(batch['attention_mask'].sum(-1).cpu().tolist())
                hidden = model(**batch,use_cache=False).last_hidden_state
                if not bool(torch.all(batch['attention_mask'][:,-1]==1)):
                    raise ValueError('Last-active pooling contract failed')
                vectors.append(torch.nn.functional.normalize(hidden[:,-1].float(),dim=1).cpu())
        values = torch.cat(vectors)
        scores = dict(zip((u for u,_ in docs),(values[0]@values[1:].T).tolist(),strict=True))
        plans = packed(s,scores,'dense_sentence',counter)
        commit('dense',s,sid,scores=scores,plans=plans,costs=dict(seconds=time.perf_counter()-begin,
            input_tokens=sum(lengths),forward_calls=math.ceil(len(texts)/16)))
        progress('dense',i)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    finished('dense',sid)


def qwen():
    sid = verify()
    torch = torch_setup()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    tok = AutoTokenizer.from_pretrained(str(ranker.MODEL),local_files_only=True,trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(ranker.MODEL),local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    counter = ExactCounter()
    for i,s in enumerate(sources(),1):
        if saved('qwen',s,sid):
            continue
        scores, lengths = {}, []
        begin = time.perf_counter()
        with torch.inference_mode():
            for uid,text in ranker.documents(s):
                ids = ranker.rank_input_ids(tok,s['question'],text)
                scores[uid] = float(ranker.two_head_scores(torch,tok,model,ids)['fp32'])
                lengths.append(len(ids))
        if not all(math.isfinite(v) for v in scores.values()):
            raise ValueError('Nonfinite frozen Qwen score')
        seed_ids = [str(j) for j in saved('retrieve',s,sid)['paths'][0]]
        plans, traces = [], []
        for budget in BUDGETS:
            p, trace = retained_plans(s,scores,seed_ids,counter,budget)
            plans.extend(p)
            pure, pt = pure_path_plan(s,scores,seed_ids,counter,budget)
            plans.append(pure)
            traces.append(dict(budget=budget,guarded=trace,pure_path=pt))
        commit('qwen',s,sid,scores=scores,plans=plans,traces=traces,costs=dict(seconds=time.perf_counter()-begin,
            input_tokens=sum(lengths),forward_calls=len(lengths)))
        progress('qwen',i)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    finished('qwen',sid)


def llmlingua():
    sid = verify()
    torch = torch_setup()
    from llmlingua import PromptCompressor
    from tools.prepare_replacement_learning_v1 import ExactCounter
    compressor = PromptCompressor(model_name=str(lingua.MODEL),device_map='cuda',use_llmlingua2=True,
        llmlingua2_config=dict(max_batch_size=4),model_config=dict(local_files_only=True,
            trust_remote_code=False,attn_implementation='eager',torch_dtype=torch.float32))
    compressor.model_name = lingua.CONFIG['checkpoint']
    compressor.model.float().eval()
    memo = lingua.ForwardMemo(compressor.model,torch)
    compressor.model.forward = memo
    def compress(contexts,rate):
        return compressor.compress_prompt_llmlingua2(contexts,rate=rate,use_context_level_filter=False,
            use_token_level_filter=True,token_to_word='mean',force_tokens=['\n'],force_reserve_digit=False,
            drop_consecutive=False,chunk_end_tokens=['.','\n'])
    toy = ['Alpha lives in Paris. Beta works in Rome.\nA study started in 2001.']
    a,b = compress(toy,.6),compress(toy,.6)
    memo.cache.clear()
    if a!=b or a!=compress(toy,.6) or memo.hits<1:
        raise ValueError('Official compression cache compatibility failed')
    counter = ExactCounter()
    for i,s in enumerate(sources(),1):
        if saved('llmlingua',s,sid):
            continue
        q = saved('qwen',s,sid)
        targets = {p['evidence_budget']:p['evidence_tokens'] for p in q['plans'] if p['method']==ranker.NEW_METHOD}
        plans,traces = [],[]
        before = memo.calls,memo.tokens,memo.seconds
        begin = time.perf_counter()
        for budget in BUDGETS:
            for method in lingua.NEW_METHODS:
                if targets[budget]<=0:
                    plans.append(fail_plan(s,method,budget,'Frozen global actual-token target is empty'))
                    continue
                contexts,ids = lingua.contexts_for(s,q['scores'],method,targets[budget],counter)
                try:
                    text,trace = lingua.calibrate(contexts,targets[budget],counter,compress)
                except ValueError as exc:
                    if 'No legal nonempty native compression' not in str(exc):
                        raise
                    plans.append(fail_plan(s,method,budget,str(exc)))
                    traces.append(dict(method=method,budget=budget,selection_error=str(exc)))
                    continue
                plan = fail_plan(s,method,budget,None)
                plan.pop('selection_error')
                plan.update(evidence_text=text,evidence_tokens=counter.count(text),evidence_sha256=fp(text),
                    compressed_input_source_ids=ids,selected_ids=['compressed:'+pid for pid in ids])
                plans.append(plan)
                traces.append(dict(method=method,budget=budget,target_tokens=targets[budget],calibration=trace,
                    input_source_ids=ids,input_reader_tokens=counter.count('\n\n'.join(contexts)),
                    native_chunk_lengths=lingua.native_chunk_lengths(compressor,contexts)))
        commit('llmlingua',s,sid,plans=plans,traces=traces,costs=dict(seconds=time.perf_counter()-begin,
            forward_calls=memo.calls-before[0],input_tokens=memo.tokens-before[1],forward_seconds=memo.seconds-before[2]))
        memo.cache.clear()
        progress('llmlingua',i)
    compressor.model.forward = memo.forward
    del compressor,memo
    gc.collect()
    torch.cuda.empty_cache()
    finished('llmlingua',sid)


def recomp_select(method):
    sid = verify()
    torch = torch_setup()
    from transformers import AutoModel, AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    folder = ROOT/'models'/('RECOMP-NQ-extractive' if method=='recomp_sentence' else 'RECOMP-HotpotQA-extractive')
    tok = AutoTokenizer.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False)
    model = AutoModel.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='eager').to('cpu').eval()
    counter = ExactCounter()
    for i,s in enumerate(sources(),1):
        if saved(method,s,sid):
            continue
        docs = recomp.units(s)
        texts = [s['question']]+[t for _,t in docs]
        original = [tok.encode(t,add_special_tokens=True,truncation=False) for t in texts]
        encoded = [tok.encode(t,add_special_tokens=True,truncation=True,max_length=512) for t in texts]
        chunks = []
        begin = time.perf_counter()
        with torch.inference_mode():
            for start in range(0,len(encoded),32):
                batch = tok.pad(dict(input_ids=encoded[start:start+32]),padding=True,return_tensors='pt')
                hidden = model(**batch).last_hidden_state
                chunks.append(recomp.mean_pooling(torch,hidden,batch['attention_mask']).cpu())
        vectors = torch.cat(chunks)
        scores = {u:float(vectors[0]@vectors[j]) for j,(u,_) in enumerate(docs,1)}
        commit(method,s,sid,scores=scores,plans=packed(s,scores,method,counter),
            clipped_items=[dict(item=j,original_tokens=len(a),retained_tokens=len(b)) for j,(a,b) in enumerate(zip(original,encoded)) if len(a)>len(b)],
            costs=dict(seconds=time.perf_counter()-begin,input_tokens=sum(map(len,encoded)),forward_calls=math.ceil(len(encoded)/32)))
        progress(method,i)
    del model
    gc.collect()
    finished(method,sid)


def assemble():
    sid = verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter = ExactCounter()
    tok = AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    marker = load(PLAN/'preparation.json')['reader_marker_sha256']
    if (RUN/'qa_plans.jsonl').exists():
        raise ValueError('Existing plan commitment; do not reassemble')
    path = RUN/'qa_plans.building'
    keys,count,maximum,failed = set(),0,0,0
    with path.open('w',encoding='utf-8') as stream:
        for source in sources():
            plans = []
            for phase in ('dense','qwen','llmlingua','recomp_sentence','recomp_hotpot_sentence'):
                plans.extend(saved(phase,source,sid)['plans'])
            if len(plans)!=40 or {(p['method'],p['evidence_budget']) for p in plans}!={(m,b) for m in METHODS for b in BUDGETS}:
                raise ValueError('Incomplete per-question comparison')
            for p in sorted(plans,key=lambda v:(v['evidence_budget'],METHODS.index(v['method']))):
                n = counter.count(p['evidence_text'])
                if n!=p['evidence_tokens'] or n>p['evidence_budget'] or fp(p['evidence_text'])!=p['evidence_sha256']:
                    raise ValueError('Evidence token/hash contract failed')
                if p.get('selection_error'):
                    failed+=1
                    p['request_key']=None
                else:
                    if not p['evidence_text'].strip():
                        raise ValueError('Unrecorded empty context')
                    messages = recomp.messages_for(p)
                    p['request_key']=recomp.interface.parent.request_key(messages,marker)
                    keys.add(p['request_key'])
                    rendered = tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    maximum=max(maximum,counter.count(rendered))
                p['signature']=sid
                stream.write(json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n')
                count+=1
        stream.flush()
        os.fsync(stream.fileno())
    if count!=CONFIG['conditions'] or maximum+2048>32768:
        raise ValueError('Full plan count or reader capacity failed')
    path.replace(RUN/'qa_plans.jsonl')
    immutable(RUN/'selection_complete.json',dict(signature=sid,plans_sha256=sha(RUN/'qa_plans.jsonl'),
        conditions=count,unique_requests=len(keys),selection_failures=failed,maximum_prompt_tokens=maximum,
        before_any_reader_answer=True))
    print('FULL_SELECTION_COMMITTED',count,len(keys),flush=True)


def qa(endpoint):
    sid = verify()
    meta = load(RUN/'selection_complete.json')
    if meta['plans_sha256']!=sha(RUN/'qa_plans.jsonl'):
        raise ValueError('Committed plans changed')
    from .evidence_acquisition_v2 import parse_answer
    base = recomp.ledger.base
    folder = RUN/'qa'
    folder.mkdir(exist_ok=True)
    intent = folder/'inflight.json'
    if intent.exists():
        raise ValueError('Unknown reader commit; no automatic retry')
    db = sqlite3.connect(folder/'ledger.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS answers(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS predictions(q TEXT, arm TEXT, value TEXT NOT NULL, PRIMARY KEY(q,arm))')
    calls = db.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
    done = db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    begin = time.perf_counter()
    for p in read_jsonl(RUN/'qa_plans.jsonl'):
        prior = db.execute('SELECT value FROM predictions WHERE q=? AND arm=?',(p['question_id'],p['arm'])).fetchone()
        if prior:
            if json.loads(prior[0])['plan_sha256']!=fp(p):
                raise ValueError('Committed prediction plan changed')
            continue
        key = p['request_key']
        cached = db.execute('SELECT value FROM answers WHERE key=?',(key,)).fetchone() if key else None
        reused = bool(cached)
        if p.get('selection_error'):
            entry = dict(hypothesis='',status='selection_failed',error=p['selection_error'])
        elif cached:
            entry = json.loads(cached[0])
        else:
            messages = recomp.messages_for(p)
            if base.rpc(endpoint,'/tokenize',dict(text=p['evidence_text']))['tokens']!=p['evidence_tokens']:
                raise ValueError('Actual reader tokenizer mismatch')
            write_json(intent,dict(signature=sid,key=key,messages=messages,max_tokens=2048))
            started = time.perf_counter()
            result,answer,status,error = {},'','failed',None
            try:
                result=base.rpc(endpoint,'/complete',dict(messages=messages,max_tokens=2048))
                answer=parse_answer(base.require_complete(result))
                status='ok'
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
            entry=dict(hypothesis=answer,status=status,error=error)
            call=dict(signature=sid,key=key,messages=messages,max_tokens=2048,**entry,result=result,
                seconds=time.perf_counter()-started,source_condition=[p['question_id'],p['arm']])
            append(folder/'answer_calls.jsonl',call)
            db.execute('INSERT INTO answers VALUES (?,?)',(key,json.dumps(entry,ensure_ascii=False)))
            db.commit()
            calls+=1
            if not result:
                # Keep the unknown/infrastructure condition for inspection, never repeat it.
                raise RuntimeError('No inference result; original call recorded, stopped')
        prediction=dict(question_id=p['question_id'],arm=p['arm'],method=p['method'],budget=p['evidence_budget'],
            signature=sid,plan_sha256=fp(p),request_key=key,reused=reused,**entry)
        db.execute('INSERT INTO predictions VALUES (?,?,?)',(p['question_id'],p['arm'],json.dumps(prediction,ensure_ascii=False)))
        db.commit()
        append(folder/'predictions.jsonl',prediction)
        if intent.exists():
            intent.unlink()
        done+=1
        write_json(folder/'status.json',dict(status='running',completed=done,expected=CONFIG['conditions'],
            new_model_calls=calls,unique_requests=meta['unique_requests'],elapsed_seconds=time.perf_counter()-begin,time=time.time()))
        if done%10==0:
            print('FULL_QA',done,'/96680',calls,'calls',flush=True)
    if done!=CONFIG['conditions'] or calls!=meta['unique_requests']:
        raise ValueError('QA completion/call accounting failed')
    db.close()
    immutable(RUN/'qa_complete.json',dict(signature=sid,conditions=done,new_model_calls=calls))
    write_json(folder/'status.json',dict(status='completed',completed=done,expected=CONFIG['conditions'],new_model_calls=calls))


@lru_cache(maxsize=1)
def official_metric():
    source = PLAN/'sources/metrics_answer.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = {'normalize_answer','get_tokens','compute_exact','compute_f1','metric_max_over_ground_truths'}
    nodes = [n for n in tree.body if isinstance(n,ast.Import) or isinstance(n,ast.FunctionDef) and n.name in names]
    namespace = {}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    if not names<=set(namespace):
        raise ValueError('Official answer scoring source incompatible')
    return namespace


def answer_scores(prediction,reference):
    m = official_metric()
    answers = [reference['answer']]+reference['aliases']
    return max(m['compute_exact'](a,prediction) for a in answers), max(m['compute_f1'](a,prediction) for a in answers)


def source_clusters(refs):
    parent={q:q for q in refs}
    def find(q):
        while parent[q]!=q:
            parent[q]=parent[parent[q]]
            q=parent[q]
        return q
    owner={}
    for q,r in refs.items():
        for atom in r['atomic_ids']:
            if atom in owner:
                parent[find(q)]=find(owner[atom])
            owner[atom]=q
    return {q:find(q) for q in refs}


def paired_stats(first,second,clusters):
    import numpy as np
    ids=sorted(first)
    if set(ids)!=set(second):
        raise ValueError('Paired comparison coverage differs')
    delta=np.array([first[q]-second[q] for q in ids],dtype=np.float64)
    grouped=defaultdict(list)
    for q,d in zip(ids,delta):
        grouped[clusters[q]].append(d)
    values=list(grouped.values())
    sums=np.array([sum(v) for v in values])
    sizes=np.array([len(v) for v in values])
    rng=np.random.default_rng(29)
    samples=rng.integers(len(values),size=(2000,len(values)))
    boots=sums[samples].sum(1)/sizes[samples].sum(1)
    return dict(n=len(ids),clusters=len(values),delta_points=float(100*delta.mean()),
        ci95_points=[float(x) for x in 100*np.percentile(boots,[2.5,97.5])],
        wins=int((delta>1e-9).sum()),ties=int((abs(delta)<=1e-9).sum()),losses=int((delta< -1e-9).sum()))


def score():
    sid=verify()
    if not (RUN/'qa_complete.json').exists():
        raise ValueError('Full QA required before final scoring')
    refs=load(PLAN/'private/references.json')
    clusters=source_clusters(refs)
    db=sqlite3.connect(RUN/'qa/ledger.sqlite3')
    results={}
    for (raw,) in db.execute('SELECT value FROM predictions'):
        p=json.loads(raw)
        results[p['question_id'],p['arm']]=p
    if len(results)!=96680:
        raise ValueError('Full denominator incomplete')
    rows=[]
    dest=RUN/'scored.jsonl'
    if dest.exists():
        raise ValueError('Existing immutable scores')
    for p in read_jsonl(RUN/'qa_plans.jsonl'):
        pred=results[p['question_id'],p['arm']]
        if pred['signature']!=sid or pred['plan_sha256']!=fp(p):
            raise ValueError('Answer identity changed')
        em,f1=answer_scores(pred['hypothesis'],refs[p['question_id']]) if pred['status']=='ok' else (0.,0.)
        ref=refs[p['question_id']]
        touched=set(map(str,p['selected_ranges']))
        coverage=None if p['method'] in lingua.NEW_METHODS else set(map(str,ref['support_indices']))<=touched
        r=dict(**pred,em=em,f1=f1,evidence_tokens=p['evidence_tokens'],
            hops=ref['hops'],cluster=clusters[p['question_id']],support_sources_touched=coverage,
            training_atomic_overlap=ref['training_atomic_overlap'])
        rows.append(r)
    with dest.open('w',encoding='utf-8') as stream:
        for r in rows:
            stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    scores,comparisons=[],[]
    for budget in BUDGETS:
        for scope in ('overall',2,3,4,'no_training_atomic_overlap'):
            def by(method):
                return {r['question_id']:r for r in rows if r['method']==method and r['budget']==budget and
                    (scope=='overall' or r['hops']==scope or scope=='no_training_atomic_overlap' and not r['training_atomic_overlap'])}
            for method in METHODS:
                selected=list(by(method).values())
                scores.append(dict(method=method,budget=budget,scope=scope,n=len(selected),
                    em_points=100*statistics.mean(r['em'] for r in selected) if selected else None,
                    f1_points=100*statistics.mean(r['f1'] for r in selected) if selected else None,
                    mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in selected) if selected else None,
                    support_source_touch_rate=statistics.mean(r['support_sources_touched'] for r in selected)
                        if selected and method not in lingua.NEW_METHODS else None,
                    statuses=dict(Counter(r['status'] for r in selected))))
            for other in METHODS:
                if other!=anchor.MAIN:
                    if not by(other):
                        continue
                    comparisons.append(dict(method=anchor.MAIN,comparator=other,budget=budget,scope=scope,
                        **paired_stats({q:r['f1'] for q,r in by(anchor.MAIN).items()},
                            {q:r['f1'] for q,r in by(other).items()},clusters)))
    macro=[]
    for method in METHODS:
        group=[s for s in scores if s['method']==method and s['scope']=='overall']
        macro.append(dict(method=method,four_budget_macro_em_points=statistics.mean(s['em_points'] for s in group),
            four_budget_macro_f1_points=statistics.mean(s['f1_points'] for s in group)))
    for other in METHODS:
        if other==anchor.MAIN:
            continue
        values={}
        for method in (anchor.MAIN,other):
            perq=defaultdict(list)
            for r in rows:
                if r['method']==method:
                    perq[r['question_id']].append(r['f1'])
            values[method]={q:statistics.mean(v) for q,v in perq.items()}
        comparisons.append(dict(method=anchor.MAIN,comparator=other,budget='macro',scope='overall',
            **paired_stats(values[anchor.MAIN],values[other],clusters)))
    costs={}
    for phase in ('retrieve','dense','qwen','llmlingua','recomp_sentence','recomp_hotpot_sentence'):
        totals=Counter()
        for s in sources():
            value=saved(phase,s,sid)
            totals.update({k:v for k,v in value.get('costs',{}).items() if isinstance(v,(int,float))})
            if phase=='retrieve':
                totals['seconds']+=value['seconds']
        costs[phase]=dict(totals)
    calls=Counter()
    for c in read_jsonl(RUN/'qa/answer_calls.jsonl'):
        calls.update(seconds=c['seconds'],calls=1,
            prompt_tokens=c['result'].get('usage',{}).get('prompt_tokens',0),
            completion_tokens=c['result'].get('usage',{}).get('completion_tokens',0))
    immutable(RUN/'report.json',dict(signature=sid,complete=True,dataset=CONFIG['dataset'],questions=2417,
        scores=scores,macro_scores=macro,comparisons=comparisons,selection_costs=costs,reader_costs=dict(calls),
        metric='MuSiQue official answer EM/F1, original aliases; no stemming',
        original_mechanism_proven=False,paper_ready=False,new_training=False,automatic_followup=False))
    print('FULL_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze_code','freeze_data','verify','preflight','retrieve','dense','qwen',
        'llmlingua','recomp_sentence','recomp_hotpot_sentence','assemble','qa','score'])
    p.add_argument('--endpoint',default='http://127.0.0.1:8037')
    a=p.parse_args()
    if a.mode in ('recomp_sentence','recomp_hotpot_sentence'):
        recomp_select(a.mode)
    elif a.mode=='qa':
        qa(a.endpoint)
    else:
        value=globals()[a.mode]()
        if value:
            print(value,flush=True)
