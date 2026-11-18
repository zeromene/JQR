"""Pinned Qwen3-Reranker-4B over frozen raw/mixed candidate pools; common 2K QA."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from . import baseline_bootstrap_v2 as base
from . import evidence_acquisition_v1 as evidence
from .evidence_acquisition_v2 import parse_answer
from .schema import fingerprint,read_jsonl,write_json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/mainline_reranker_v1'
MODEL=ROOT/'models/Qwen3-Reranker-4B'
ARMS=('reranker_raw_2k','reranker_mixed_2k')
CONFIG=dict(arms=list(ARMS),revision='22e683669bc0f0bd69640a1354a6d0aebcfeede5',
    ranker='Qwen3-Reranker-4B',reader='Qwen3.5-9B',precision='bf16-sdpa',
    ranker_max_input=8192,ranker_batch=4,instruction='Given a web search query, retrieve relevant passages that answer the query',
    scoring='official README yes/no last-token logit softmax; logits_to_keep=1 is memory-only optimization',
    document='exact individually rendered common candidate, including time/speaker',
    selection='descending reranker score, ties original pool order; existing source-union whole-entry pack',
    evidence_budget=2048,answer_max_tokens=2048,decoding='greedy-thinking_false',
    candidate_pool='frozen evidence_acquisition_v2 raw/fusion shared pool; raw arm takes raw subset',
    scope='64 exposed dev questions; not independent confirmation',heldout_accessed=False)
PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def identity():
    base.verify_inputs()
    manifest=load(MODEL/'artifact_manifest.json')
    if manifest['revision']!=CONFIG['revision'] or not (MODEL/'.verified.json').exists():
        raise ValueError('Pinned verified ranker missing')
    public=load(base.PLAN/'public/queries.json')
    paths=['reader_memory/mainline_reranker_v1.py','reader_memory/mainline_service_v1.py',
        'scripts/mainline_reranker_queue_v1.py','research/mainline_baselines_v1/RERANKER_PROTOCOL.md',
        'reader_memory/evidence_acquisition_v1.py','reader_memory/evidence_acquisition_v2.py',
        'reader_memory/tokenization.py','reader_memory/backend.py']
    cached=[]
    for cid in dict.fromkeys(q['conversation_id'] for q in public):
        cached.extend(['runs/evidence_acquisition_v1/'+cid+'/candidate_records.json',
            'runs/evidence_acquisition_v1/'+cid+'/manifest.json',
            'runs/evidence_acquisition_v2/'+cid+'/plans.json',
            'runs/evidence_acquisition_v2/'+cid+'/manifest.json'])
    value=dict(config=CONFIG,code={p:base.sha(ROOT/p) for p in paths},
        public_sha256=base.sha(base.PLAN/'public/queries.json'),
        common_pool={p:base.sha(ROOT/p) for p in cached},
        ranker_manifest_sha256=base.sha(MODEL/'artifact_manifest.json'),
        reader_marker_sha256=base.sha(ROOT/'models/Qwen3.5-9B/.verified.json'))
    target=RUN/'identity.json'
    if target.exists() and load(target)!=value:
        raise ValueError('Frozen ranker identity changed; use new run version')
    write_json(target,value)
    return fingerprint(value)


def rerank():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from .tokenization import TokenCounter
    sid=identity()
    public=load(base.PLAN/'public/queries.json')
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),padding_side='left',local_files_only=True)
    counter=TokenCounter(ROOT/'models/Qwen3.5-9B')
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    false_id=tokenizer.convert_tokens_to_ids('no')
    true_id=tokenizer.convert_tokens_to_ids('yes')
    pre=tokenizer.encode(PREFIX,add_special_tokens=False)
    suf=tokenizer.encode(SUFFIX,add_special_tokens=False)
    for cid in dict.fromkeys(q['conversation_id'] for q in public):
        out=RUN/cid
        out.mkdir(parents=True,exist_ok=True)
        if (out/'manifest.json').exists():
            verify_plans(cid,sid)
            continue
        records=load(ROOT/'runs/evidence_acquisition_v1'/cid/'candidate_records.json')
        upstream=load(ROOT/'runs/evidence_acquisition_v2'/cid/'plans.json')
        source={p['question_id']:p for p in upstream if p['arm']=='fusion_2k'}
        source_manifest=load(ROOT/'runs/evidence_acquisition_v2'/cid/'manifest.json')
        if source_manifest['plan_sha256']!=base.sha(ROOT/'runs/evidence_acquisition_v2'/cid/'plans.json'):
            raise ValueError('Frozen pool manifest mismatch')
        plans=[]
        for q in [q for q in public if q['conversation_id']==cid]:
            cache=out/(q['question_id'].replace(':','_')+'.scores.json')
            pool=source[q['question_id']]['candidate_pool_ids']
            if len(set(pool))!=len(pool):
                raise ValueError('Duplicate pool ID')
            if cache.exists():
                scored=load(cache)
                if scored['signature']!=sid or scored['candidate_pool_ids']!=pool:
                    raise ValueError('Cached ranker score identity mismatch')
            else:
                intent=out/'ranker_inflight.json'
                if intent.exists():
                    raise ValueError('Uncommitted ranker inference; no automatic duplicate computation')
                write_json(intent,dict(signature=sid,question_id=q['question_id']))
                scores=[]
                lengths=[]
                start=time.perf_counter()
                for offset in range(0,len(pool),CONFIG['ranker_batch']):
                    keys=pool[offset:offset+CONFIG['ranker_batch']]
                    inputs=[]
                    for key in keys:
                        doc=evidence.render([key],records)
                        body=f"<Instruct>: {CONFIG['instruction']}\n<Query>: {q['question']}\n<Document>: {doc}"
                        ids=pre+tokenizer.encode(body,add_special_tokens=False)+suf
                        if len(ids)>8192:
                            raise ValueError('Ranker document exceeds8192; no silent truncation')
                        inputs.append(ids)
                        lengths.append(len(ids))
                    batch=tokenizer.pad(dict(input_ids=inputs),padding=True,return_tensors='pt').to('cuda')
                    with torch.inference_mode():
                        logits=model(**batch,logits_to_keep=1).logits[:,-1,:].float()
                        two=torch.stack([logits[:,false_id],logits[:,true_id]],dim=1)
                        scores.extend(torch.nn.functional.log_softmax(two,dim=1)[:,1].exp().tolist())
                    torch.cuda.synchronize()
                if len(scores)!=len(pool) or not all(math.isfinite(s) for s in scores):
                    raise ValueError('Missing/nonfinite ranking scores')
                scored=dict(signature=sid,question_id=q['question_id'],candidate_pool_ids=pool,
                    scores=scores,pair_input_tokens=lengths,forward_calls=math.ceil(len(pool)/CONFIG['ranker_batch']),
                    seconds=time.perf_counter()-start)
                write_json(cache,scored)
                intent.unlink()
            ordered=sorted(range(len(pool)),key=lambda i:(-scored['scores'][i],i))
            order=[pool[i] for i in ordered]
            for arm in ARMS:
                allowed=[key for key in order if arm!='reranker_raw_2k' or records[key]['kind']=='raw']
                selected=evidence.ordered_pack(allowed,records,counter,2048)
                contexts=evidence.materialize(selected,records)
                text=evidence.render(selected,records)
                plans.append(dict(**q,signature=sid,arm=arm,contexts=contexts,selected_ids=selected,
                    candidate_pool_ids=pool,evidence_tokens=counter.count(text),evidence_budget=2048))
            write_json(RUN/'status.json',dict(status='running',phase='rerank',question_id=q['question_id']))
            print('RERANKER_SCORE',q['question_id'],len(pool),flush=True)
        write_json(out/'plans.json',plans)
        write_json(out/'manifest.json',dict(signature=sid,plans_sha256=base.sha(out/'plans.json'),
            n=len(plans),private_inputs_read=False))
    write_json(RUN/'ranker_complete.json',dict(signature=sid,questions=len(public)))


def verify_plans(cid,sid):
    out=RUN/cid
    meta=load(out/'manifest.json')
    if meta['signature']!=sid or meta['plans_sha256']!=base.sha(out/'plans.json'):
        raise ValueError('Ranker QA plan integrity mismatch')
    plans=load(out/'plans.json')
    qs=[q for q in load(base.PLAN/'public/queries.json') if q['conversation_id']==cid]
    expected={(q['question_id'],a) for q in qs for a in ARMS}
    if len(plans)!=len(expected) or {(p['question_id'],p['arm']) for p in plans}!=expected:
        raise ValueError('Ranker plan coverage mismatch')
    if any(p['signature']!=sid or p['evidence_tokens']>2048 for p in plans):
        raise ValueError('Ranker plan signature/budget mismatch')
    return plans


def run(endpoint):
    sid=identity()
    plans=[p for cid in dict.fromkeys(q['conversation_id'] for q in load(base.PLAN/'public/queries.json'))
        for p in verify_plans(cid,sid)]
    old=list(read_jsonl(RUN/'predictions.jsonl')) if (RUN/'predictions.jsonl').exists() else []
    calls=list(read_jsonl(RUN/'answer_calls.jsonl')) if (RUN/'answer_calls.jsonl').exists() else []
    done={(r['question_id'],r['arm']) for r in old}
    if len(done)!=len(old) or len(calls)!=len(old) or any(r['signature']!=sid for r in old+calls):
        raise ValueError('Invalid, duplicate or uncommitted ranker QA log')
    intent=RUN/'answer_inflight.json'
    if intent.exists():
        r=load(intent)
        if (r['question_id'],r['arm']) not in done:
            raise ValueError('Uncommitted generation; no invisible retry')
        intent.unlink()
    for p in plans:
        key=(p['question_id'],p['arm'])
        if key in done:
            continue
        text=evidence.FORMAT(None,[SimpleNamespace(**e) for e in p['contexts']])
        if base.rpc(endpoint,'/tokenize',dict(text=text))['tokens']!=p['evidence_tokens']:
            raise ValueError('Reader tokenization mismatch')
        messages=[dict(role='system',content='You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format.'),
            dict(role='user',content=evidence.PROMPT(None,p['question'],text))]
        call=dict(signature=sid,question_id=p['question_id'],arm=p['arm'],messages=messages,max_tokens=2048)
        write_json(intent,call)
        result={}
        start=time.perf_counter()
        hypothesis,status,error='','failed',None
        try:
            result=base.rpc(endpoint,'/complete',dict(messages=messages,max_tokens=2048))
            hypothesis=parse_answer(base.require_complete(result))
            status='ok'
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
        base.append(RUN/'answer_calls.jsonl',dict(**call,result=result,status=status,error=error,
            seconds=time.perf_counter()-start))
        base.append(RUN/'predictions.jsonl',dict(**p,hypothesis=hypothesis,status=status,error=error))
        intent.unlink()
        done.add(key)
        write_json(RUN/'status.json',dict(status='running',phase='answer',completed=len(done),expected=len(plans)))
        print('RERANKER_QA',len(done),'/',len(plans),*key,status,flush=True)
        if not result:
            raise RuntimeError('No inference result; recorded failed call and stopped')
    write_json(RUN/'status.json',dict(status='completed',completed=len(done),expected=len(plans)))


def score():
    base.verify_inputs(private=True)
    sid=identity()
    rows=list(read_jsonl(RUN/'predictions.jsonl'))
    expected={(q['question_id'],a) for q in load(base.PLAN/'public/queries.json') for a in ARMS}
    if len(rows)!=len({(r['question_id'],r['arm']) for r in rows}) or any(r['signature']!=sid for r in rows):
        raise ValueError('Invalid scoring rows')
    refs={r['question_id']:r for r in load(base.PLAN/'private/references.json')}
    groups=defaultdict(list)
    scored=[]
    for r in rows:
        ref=refs[r['question_id']]
        f1=base.native_f1(r['hypothesis'],ref['answer'],ref['category'])
        scored.append(dict(**r,official_native_f1=f1))
        for scope in ['overall','category:'+str(r['category']),'conversation:'+r['conversation_id']]:
            groups[r['arm'],scope].append((f1,r))
    summary=[dict(arm=a,scope=s,n=len(v),native_f1=statistics.mean(x[0] for x in v),
        mean_evidence_tokens=statistics.mean(x[1]['evidence_tokens'] for x in v),
        status_counts=dict(Counter(x[1]['status'] for x in v))) for (a,s),v in sorted(groups.items())]
    ranked=[load(p) for p in RUN.glob('conv-*/*.scores.json')]
    costs=dict(pairs=sum(len(v['scores']) for v in ranked),forward_calls=sum(v['forward_calls'] for v in ranked),
        input_tokens=sum(sum(v['pair_input_tokens']) for v in ranked),seconds=sum(v['seconds'] for v in ranked))
    report=dict(signature=sid,scores=summary,ranker_costs=costs,
        complete={(r['question_id'],r['arm']) for r in rows}==expected,
        raw_and_mixed_share_ranking_computation=True,heldout_accessed=False,scope=CONFIG['scope'])
    write_json(RUN/'scored.json',scored)
    write_json(RUN/'report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['rerank','run','score'])
    p.add_argument('--endpoint',default='http://127.0.0.1:8017')
    args=p.parse_args()
    if args.mode=='run':
        run(args.endpoint)
    else:
        globals()[args.mode]()
