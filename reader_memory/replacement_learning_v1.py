"""Frozen QA rewards -> set preference learning -> anchored swaps -> complete dev QA."""
import argparse
from collections import Counter,defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

from . import baseline_bootstrap_v2 as base
from . import evidence_acquisition_v1 as evidence
from . import replacement_core_v1 as core
from . import replacement_ranker_v1 as ranker
from .evidence_acquisition_v2 import parse_answer
from .schema import fingerprint,read_jsonl,write_json

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'research/replacement_learning_v1'
RUN=ROOT/'runs/replacement_learning_v1'
MODEL=ROOT/'models/Qwen3-Reranker-4B'
ARMS=('learned_single_swap','learned_group_swap')
SYSTEM='You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format.'


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def freeze():
    paths=['reader_memory/replacement_core_v1.py','reader_memory/replacement_ranker_v1.py',
        'reader_memory/replacement_learning_v1.py','scripts/replacement_learning_queue_v1.py',
        'tools/prepare_replacement_learning_v1.py','tools/audit_replacement_learning_v1.py',
        'reader_memory/mainline_reranker_v1.py','reader_memory/mainline_service_v1.py',
        'reader_memory/evidence_acquisition_v1.py','reader_memory/evidence_acquisition_v2.py',
        'reader_memory/backend.py','reader_memory/baseline_bootstrap_v2.py','reader_memory/tokenization.py',
        'reader_memory/schema.py','research/replacement_learning_v1/PROTOCOL.md',
        'tests/test_replacement_learning_v1.py',
        'research/replacement_learning_v1/public/reward_inputs.json',
        'research/replacement_learning_v1/private/references.json',
        'research/replacement_learning_v1/private/lineage.json',
        'research/replacement_learning_v1/private/source_selection.json',
        'research/replacement_learning_v1/preparation.json',
        'research/replacement_learning_v1/model_artifact_reference.json',
        'research/replacement_learning_v1/execution_repair.json']
    value=dict(config=ranker.CONFIG,instruction=ranker.INSTRUCTION,
        files={p:base.sha(ROOT/p) for p in paths},heldout_accessed=False)
    value['signature']=fingerprint(value)
    path=PLAN/'manifest.json'
    if path.exists() and load(path)!=value:raise ValueError('Frozen method files changed')
    write_json(path,value)
    print('FROZEN_REPLACEMENT_LEARNING',value['signature'],flush=True)


def verify():
    value=load(PLAN/'manifest.json')
    if value['config']!=ranker.CONFIG or value['instruction']!=ranker.INSTRUCTION:
        raise ValueError('Frozen ranker configuration changed')
    if fingerprint({k:v for k,v in value.items() if k!='signature'})!=value['signature']:
        raise ValueError('Manifest signature mismatch')
    for path,digest in value['files'].items():
        if base.sha(ROOT/path)!=digest:raise ValueError('Changed frozen method input: '+path)
    if load(MODEL/'artifact_manifest.json')!=load(PLAN/'model_artifact_reference.json'):
        raise ValueError('Model revision or upstream file hashes differ from frozen reference')
    return value['signature']


def preflight():
    sid=verify()
    if not (MODEL/'.verified.json').exists():raise ValueError('Verified offline ranker marker missing')
    from transformers import AutoTokenizer
    from .tokenization import TokenCounter
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    counter=TokenCounter(ROOT/'models/Qwen3.5-9B')
    inputs=load(PLAN/'public/reward_inputs.json')
    seen=set();lengths=[]
    for row in inputs:
        key=(row['question_id'],row['variant'])
        if key in seen:raise ValueError('Repeated reward condition')
        seen.add(key)
        rendered=evidence.FORMAT(None,[SimpleNamespace(**c) for c in row['contexts']])
        if rendered!=row['evidence_text'] or counter.count(rendered)!=row['evidence_tokens'] or row['evidence_tokens']>2048:
            raise ValueError('Local/remote exact text/token mismatch')
        lengths.append(len(ranker.input_ids(tok,row['question'],rendered)))
    write_json(RUN/'preflight.json',dict(signature=sid,contexts=len(inputs),
        ranker_max_tokens=max(lengths),tokenization_verified=True,heldout_accessed=False))
    print('PREFLIGHT_OK',len(inputs),'max_ranker_tokens',max(lengths),flush=True)


def generate_rows(rows,folder,endpoint,sid):
    """Answer only; this path never reads references. Unknown commits are not retried."""
    folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'predictions.jsonl';calls=folder/'answer_calls.jsonl';intent=folder/'inflight.json'
    existing=list(read_jsonl(dest)) if dest.exists() else []
    oldcalls=list(read_jsonl(calls)) if calls.exists() else []
    key=lambda r:(r['question_id'],r.get('arm',r.get('variant')))
    done={key(r):r for r in existing}
    if len(done)!=len(existing) or len(existing)!=len(oldcalls) or any(r['signature']!=sid for r in existing+oldcalls):
        raise ValueError('Invalid or uncommitted answer ledger')
    for p,c in zip(existing,oldcalls):
        if key(p)!=key(c) or p['request_sha256']!=c['request_sha256']:
            raise ValueError('Answer and cost ledgers differ')
    if intent.exists():
        if key(load(intent)) not in done:raise ValueError('Uncommitted inference; no invisible retry')
        intent.unlink()
    for row in rows:
        request=fingerprint(row)
        if key(row) in done:
            if done[key(row)]['request_sha256']!=request:raise ValueError('Committed QA request changed')
            continue
        text=row['evidence_text']
        if base.rpc(endpoint,'/tokenize',dict(text=text))['tokens']!=row['evidence_tokens']:
            raise ValueError('Service tokenizer differs from frozen input')
        messages=[dict(role='system',content=SYSTEM),dict(role='user',content=evidence.PROMPT(None,row['question'],text))]
        write_json(intent,dict(**row,signature=sid,request_sha256=request))
        started=time.perf_counter();result={};answer='';status='failed';error=None
        try:
            result=base.rpc(endpoint,'/complete',dict(messages=messages,max_tokens=2048))
            answer=parse_answer(base.require_complete(result));status='ok'
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        identifier=dict(question_id=row['question_id'])
        identifier['arm' if 'arm' in row else 'variant']=row.get('arm',row.get('variant'))
        base.append(calls,dict(**identifier,signature=sid,request_sha256=request,messages=messages,
            max_tokens=2048,result=result,status=status,error=error,seconds=time.perf_counter()-started))
        pred=dict(**row,signature=sid,request_sha256=request,hypothesis=answer,status=status,error=error)
        base.append(dest,pred);intent.unlink();done[key(row)]=pred
        write_json(folder/'status.json',dict(status='running',completed=len(done),expected=len(rows)))
        print('REPLACEMENT_QA',folder.name,len(done),'/',len(rows),*key(row),status,flush=True)
        if not result:raise RuntimeError('No service result; recorded failure and stopped')
    if set(done)!={key(r) for r in rows}:raise ValueError('Unexpected QA conditions')
    write_json(folder/'status.json',dict(status='completed',completed=len(done),expected=len(rows)))


def rewards(endpoint):
    sid=verify()
    generate_rows(load(PLAN/'public/reward_inputs.json'),RUN/'rewards',endpoint,sid)


def score_rewards():
    sid=verify();rows=list(read_jsonl(RUN/'rewards/predictions.jsonl'))
    public=load(PLAN/'public/reward_inputs.json');refs=load(PLAN/'private/references.json')
    if len(rows)!=len(public) or {(r['question_id'],r['variant']) for r in rows}!={(r['question_id'],r['variant']) for r in public}:
        raise ValueError('Reward collection incomplete')
    targets=[]
    for row in rows:
        if row['signature']!=sid:raise ValueError('Reward signature differs')
        ref=refs[row['question_id']]
        answers=list(dict.fromkeys([ref['answer']]+ref.get('aliases',[])))
        reward=max(base.native_f1(row['hypothesis'],a,2) for a in answers) if row['status']=='ok' else 0.
        targets.append(dict(question_id=row['question_id'],variant=row['variant'],reward=reward,
            evidence_tokens=row['evidence_tokens'],split=row['split'],status=row['status']))
    write_json(RUN/'reward_targets.json',targets)
    print('REWARD_TARGETS',len(targets),'statuses',dict(Counter(r['status'] for r in targets)),flush=True)


def training_pairs(inputs,targets,split):
    by={(r['question_id'],r['variant']):r for r in inputs if r['split']==split}
    groups=defaultdict(list)
    for t in targets:
        if t['split']==split:groups[t['question_id']].append(t)
    pairs=[]
    for qid,values in groups.items():
        options=[]
        for a,b in itertools.combinations(values,2):
            if a['reward']==b['reward'] or not core.matched_cost(a['evidence_tokens'],b['evidence_tokens']):continue
            options.append((a,b))
        options.sort(key=lambda p:fingerprint([core.SEED,qid,p[0]['variant'],p[1]['variant']]))
        for a,b in options[:ranker.CONFIG['max_pairs_per_question']]:
            pairs.append((by[qid,a['variant']],by[qid,b['variant']],a['reward'],b['reward']))
    return pairs


def train():
    sid=verify()
    if (RUN/'training_started.json').exists():raise ValueError('Training already attempted; no hidden retraining')
    inputs=load(PLAN/'public/reward_inputs.json');targets=load(RUN/'reward_targets.json')
    pairs=training_pairs(inputs,targets,'train')
    if not pairs:raise ValueError('No non-tied actual-QA equal-cost preferences; cannot train')
    torch,tok,model,paths=ranker.load(MODEL,training=True)
    encoded={(r['question_id'],r['variant']):ranker.input_ids(tok,r['question'],r['evidence_text']) for r in inputs}
    parameters=[p for p in model.parameters() if p.requires_grad]
    write_json(RUN/'training_started.json',dict(signature=sid,pairs=len(pairs),adapter_paths=paths,
        trainable_parameters=sum(p.numel() for p in parameters),torch=torch.__version__,
        reward_targets_sha256=base.sha(RUN/'reward_targets.json')))
    optimizer=torch.optim.AdamW(parameters,lr=ranker.CONFIG['learning_rate'],weight_decay=ranker.CONFIG['weight_decay'])
    optimizer.zero_grad(set_to_none=True);updates=0;started=time.perf_counter()
    rng=random.Random(ranker.CONFIG['seed'])
    def score(row):return ranker.logit(torch,tok,model,encoded[row['question_id'],row['variant']])
    for epoch in range(ranker.CONFIG['epochs']):
        order=list(range(len(pairs)));rng.shuffle(order);losses=[]
        for start in range(0,len(order),ranker.CONFIG['accumulation']):
            batch=order[start:start+ranker.CONFIG['accumulation']]
            for index in batch:
                a,b,ra,rb=pairs[index];va,vb=score(a),score(b)
                pref=torch.tensor(1/(1+math.exp(-ranker.CONFIG['difference_temperature']*(ra-rb))),device='cuda')
                loss=torch.nn.functional.binary_cross_entropy_with_logits(va-vb,pref)
                loss=loss+ranker.CONFIG['regression_weight']*((va.sigmoid()-ra)**2+(vb.sigmoid()-rb)**2)/2
                if not torch.isfinite(loss):raise ValueError('Nonfinite learning loss')
                losses.append(float(loss.detach()));(loss/len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(parameters,ranker.CONFIG['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            base.append(RUN/'training_log.jsonl',dict(epoch=epoch+1,updates=updates,
                seen_pairs=min(start+len(batch),len(order)),mean_loss=statistics.mean(losses[-len(batch):]),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-started))
            if updates%10==0:print('REPLACEMENT_TRAIN',epoch+1,updates,len(pairs),flush=True)
        # Development is descriptive only; fixed last epoch and no threshold tuning.
        model.eval();dev=[]
        with torch.inference_mode():
            for row in [r for r in inputs if r['split']=='dev']:
                value=float(score(row).sigmoid())
                dev.append(dict(question_id=row['question_id'],variant=row['variant'],score=value))
        write_json(RUN/f'dev_epoch{epoch+1}.json',dev);model.train()
    checkpoint=RUN/'adapter.pt'
    torch.save(ranker.adapter_state(model),checkpoint)
    write_json(RUN/'training_complete.json',dict(signature=sid,updates=updates,pairs=len(pairs),
        checkpoint_sha256=base.sha(checkpoint),seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_selection='fixed last epoch'))


def select():
    sid=verify();base.verify_inputs()
    checkpoint=load(RUN/'training_complete.json')
    if checkpoint['signature']!=sid or checkpoint['checkpoint_sha256']!=base.sha(RUN/'adapter.pt'):
        raise ValueError('Learned checkpoint identity differs')
    torch,tok,model,_=ranker.load(MODEL)
    ranker.load_adapter(model,torch.load(RUN/'adapter.pt',map_location='cpu',weights_only=True))
    from .tokenization import TokenCounter
    counter=TokenCounter(ROOT/'models/Qwen3.5-9B')
    questions=load(base.PLAN/'public/queries.json');plans=[]
    for q in questions:
        target=RUN/'selected'/(q['question_id'].replace(':','_')+'.json')
        if target.exists():
            value=load(target)
            if value['signature']!=sid:raise ValueError('Selection signature mismatch')
            plans.extend(value['plans']);continue
        cid=q['conversation_id']
        records=load(ROOT/'runs/evidence_acquisition_v1'/cid/'candidate_records.json')
        source_manifest=load(ROOT/'runs/evidence_acquisition_v2'/cid/'manifest.json')
        if source_manifest['plan_sha256']!=base.sha(ROOT/'runs/evidence_acquisition_v2'/cid/'plans.json'):
            raise ValueError('Original fusion plan integrity mismatch')
        upstream=load(ROOT/'runs/evidence_acquisition_v2'/cid/'plans.json')
        source=next(r for r in upstream if r['question_id']==q['question_id'] and r['arm']=='fusion_2k')
        cached=load(ROOT/'runs/mainline_reranker_v1'/cid/(q['question_id'].replace(':','_')+'.scores.json'))
        if cached['signature']!=fingerprint(load(ROOT/'runs/mainline_reranker_v1/identity.json')):
            raise ValueError('Shared semantic ranking identity mismatch')
        pool=source['candidate_pool_ids']
        if pool!=cached['candidate_pool_ids']:raise ValueError('Semantic ranker candidate pool differs')
        scores=dict(zip(pool,cached['scores'],strict=True))
        alternatives={arm:core.proposal_sets(q['question'],source['selected_ids'],pool,records,scores,counter,
            allow_groups=arm=='learned_group_swap',max_sets=24,pairs=source['candidate_pairs']) for arm in ARMS}
        unique={p['evidence_sha256']:p for values in alternatives.values() for p in values}
        intent=RUN/'selection_inflight.json'
        if intent.exists():raise ValueError('Uncommitted selector forward; no hidden repeat')
        write_json(intent,dict(signature=sid,question_id=q['question_id']))
        started=time.perf_counter();values={};lengths=[]
        with torch.inference_mode():
            for digest,p in unique.items():
                ids=ranker.input_ids(tok,q['question'],p['evidence_text']);lengths.append(len(ids))
                values[digest]=float(ranker.logit(torch,tok,model,ids))
        if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite set ranking score')
        selected=[]
        for arm,proposals in alternatives.items():
            # Python max returns first on ties; anchor is always first.
            best=max(proposals,key=lambda p:values[p['evidence_sha256']])
            selected.append(dict(**q,arm=arm,**best,contexts=evidence.materialize(best['selected_ids'],records),
                candidate_pool_ids=pool,predicted_score=values[best['evidence_sha256']],
                anchor_score=values[proposals[0]['evidence_sha256']],scored_sets=len(proposals),
                private_inputs_read=False))
        value=dict(signature=sid,plans=selected,alternatives=alternatives,scores=values,
            costs=dict(forward_calls=len(unique),input_tokens=sum(lengths),seconds=time.perf_counter()-started),
            shared_ranker_cache_sha256=base.sha(ROOT/'runs/mainline_reranker_v1'/cid/(q['question_id'].replace(':','_')+'.scores.json')),
            anchor_plan_sha256=base.sha(ROOT/'runs/evidence_acquisition_v2'/cid/'plans.json'))
        write_json(target,value);intent.unlink();plans.extend(selected)
        print('REPLACEMENT_SELECT',q['question_id'],len(unique),flush=True)
    write_json(RUN/'qa_plans.json',plans)
    write_json(RUN/'selection_complete.json',dict(signature=sid,questions=len(questions),
        plans_sha256=base.sha(RUN/'qa_plans.json'),checkpoint_sha256=checkpoint['checkpoint_sha256'],
        heldout_accessed=False))


def qa(endpoint):
    sid=verify();meta=load(RUN/'selection_complete.json')
    if meta['plans_sha256']!=base.sha(RUN/'qa_plans.json'):raise ValueError('QA plans changed')
    generate_rows(load(RUN/'qa_plans.json'),RUN/'qa',endpoint,sid)


def score():
    sid=verify();base.verify_inputs(private=True)
    rows=list(read_jsonl(RUN/'qa/predictions.jsonl'))
    qs=load(base.PLAN/'public/queries.json');refs={r['question_id']:r for r in load(base.PLAN/'private/references.json')}
    if len(rows)!=len(qs)*len(ARMS) or {(r['question_id'],r['arm']) for r in rows}!={(q['question_id'],a) for q in qs for a in ARMS}:
        raise ValueError('Complete development QA required before reporting result')
    groups=defaultdict(list);scored=[]
    for row in rows:
        ref=refs[row['question_id']]
        f1=base.native_f1(row['hypothesis'],ref['answer'],ref['category']) if row['status']=='ok' else 0.
        if row['signature']!=sid:raise ValueError('QA signature mismatch')
        scored.append(dict(**row,official_native_f1=f1))
        for scope in ['overall','conversation:'+row['conversation_id'],'category:'+str(row['category'])]:
            groups[row['arm'],scope].append((f1,row))
    summary=[dict(arm=a,scope=s,n=len(v),native_f1=statistics.mean(f for f,_ in v),
        mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for _,r in v),
        status_counts=dict(Counter(r['status'] for _,r in v))) for (a,s),v in sorted(groups.items())]
    costs=[load(p)['costs'] for p in (RUN/'selected').glob('*.json')]
    write_json(RUN/'scored.json',scored)
    report=dict(signature=sid,complete=True,scores=summary,selector_costs={k:sum(v[k] for v in costs)
        for k in ['forward_calls','input_tokens','seconds']},shared_semantic_ranker_costs_excluded_from_these_totals=True,
        scope='64 repeatedly exposed development questions; no independent confirmation',heldout_accessed=False)
    write_json(RUN/'report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['freeze','preflight','rewards','score_rewards','train','select','qa','score'])
    parser.add_argument('--endpoint',default='http://127.0.0.1:8018')
    args=parser.parse_args()
    if args.mode in ('rewards','qa'):globals()[args.mode](args.endpoint)
    else:globals()[args.mode]()
