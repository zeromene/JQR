"""Aligned natural replacement feedback; fixed-architecture, equal-update training controls."""
import argparse
from collections import Counter, defaultdict
import itertools
import math
from pathlib import Path
import random
import statistics
import time

from . import replacement_stage2_common as common
from . import replacement_precision_v2 as precision
from . import replacement_controls_v3 as controls
from . import replacement_core_v1 as core
from . import replacement_ranker_v1 as ranker
from . import replacement_learning_v1 as parent
from . import evidence_acquisition_v1 as evidence
from .schema import fingerprint, read_jsonl, write_json

ROOT = common.ROOT
PLAN = ROOT / 'research/replacement_training_v2'
RUN = ROOT / 'runs/replacement_training_v2'
CONDITIONS = ('aligned_non_tie','aligned_with_neutral')
CONFIG = dict(seed=29, conditions=list(CONDITIONS), updates=304, accumulation=8,
    lr=1e-4, weight_decay=.01, clip=1., regression_weight=.25, difference_temperature=4.,
    architecture='unchanged pinned Qwen3-Reranker-4B, same 655360 LoRA parameters',
    initialization='original pretrained ranker, same initialization seed in both conditions',
    checkpoint='fixed update304; no checkpoint/threshold selection on LoCoMo',
    precision='unchanged BF16 head; LoRA FP32 as v1', budget=2048,
    natural_variants='anchor, semantic_pack, two highest teacher-scored single and group proposals',
    auxiliary='two compatible saved one-copy/joint-copy deletion contexts',
    mining='same core proposal implementation, semantic individual ranks, original frozen v1 set teacher',
    candidate_source='existing MuSiQue lexical raw14/extract28 pool; not natural dialogue compression',
    sampling='uniform questions; non-neutral 75% natural/25% auxiliary when available; neutral condition 25% natural ties when available',
    pair_cap_per_bucket=12, failure_training='exclude non-ok outputs from preferences; preserve failed evaluation denominator',
    heldout_accessed=False)


def prepare():
    from tools import prepare_replacement_learning_v1 as previous
    if parent.base.sha(previous.SOURCE)!=previous.SOURCE_SHA: raise ValueError('MuSiQue training file changed')
    allocation = common.load(parent.PLAN/'private/source_selection.json')
    wanted = set(allocation['train']+allocation['dev']); sources = {}
    with previous.SOURCE.open(encoding='utf-8') as stream:
        import json
        for line in stream:
            row = json.loads(line)
            if row['id'] in wanted: sources[row['id']] = row
    if len(sources)!=112 or len(allocation['train'])!=96 or len(allocation['dev'])!=16:
        raise ValueError('Unexpected legal source allocation')
    keys = {s:set().union(*(previous.source_keys(sources[k]) for k in allocation[s])) for s in ['train','dev']}
    if keys['train'] & keys['dev']: raise ValueError('Train/dev fact overlap')
    old = common.load(parent.PLAN/'public/reward_inputs.json')
    anchors = {r['question_id']:r for r in old if r['variant']=='fusion'}
    counter = previous.ExactCounter(); inputs = []
    for split in ['train','dev']:
        for source_id in allocation[split]:
            row = sources[source_id]; records,_ = previous.records_for(row)
            for mode in ['multi','single']:
                qid = source_id+':'+mode; anchor = anchors[qid]; question = anchor['question']
                lexical = core.lexical_scores(question,records,list(records))
                ordered = sorted(records,key=lambda k:(-lexical[k],k))
                raw = [k for k in ordered if records[k]['kind']=='raw'][:14]
                compact = [k for k in ordered if records[k]['kind']=='compact'][:28]
                pool = [k for pair in itertools.zip_longest(raw,compact) for k in pair if k]
                seed = evidence.ordered_pack(pool,records,counter,2048)
                if evidence.render(seed,records)!=anchor['evidence_text']:
                    raise ValueError('Legacy anchor did not reproduce: '+qid)
                inputs.append(dict(question_id=qid,question=question,split=split,mode=mode,
                    source_id=source_id,candidate_pool_ids=pool,seed_ids=seed,
                    records={k:records[k] for k in pool}))
    common.immutable_json(PLAN/'public/source_inputs.json',inputs)
    common.immutable_json(PLAN/'private/references.json',common.load(parent.PLAN/'private/references.json'))
    common.immutable_json(PLAN/'private/source_selection.json',allocation)
    common.immutable_json(PLAN/'preparation.json',dict(questions=len(inputs),sources=dict(train=96,dev=16),
        source_sha256=previous.SOURCE_SHA,train_dev_fact_overlap=0,heldout_accessed=False))
    print('PREPARED_ALIGNED_SOURCES',len(inputs),flush=True)


def freeze():
    paths = ['reader_memory/replacement_training_v2.py','reader_memory/replacement_stage2_common.py',
        'reader_memory/replacement_controls_v3.py','scripts/replacement_stage2_queue_v2.py',
        'tools/prepare_replacement_stage2_server.py',
        'tests/test_replacement_stage2.py','research/replacement_training_v2/PROTOCOL.md',
        'research/replacement_controls_v3/manifest.json',
        'runs/replacement_controls_v3/qa_plans.json',
        'runs/replacement_controls_v3/matching_audit.json',
        'research/replacement_training_v2/public/source_inputs.json',
        'research/replacement_training_v2/private/references.json',
        'research/replacement_training_v2/private/source_selection.json',
        'research/replacement_training_v2/preparation.json',
        'runs/replacement_learning_v1_repair1/rewards/predictions.jsonl',
        'runs/replacement_learning_v1_repair1/rewards/answer_calls.jsonl',
        'runs/replacement_precision_v2/qa/answer_calls.jsonl']
    paths += [str(p.relative_to(ROOT)) for p in sorted((controls.RUN/'selected').glob('*.json'))]
    sid = common.identity(PLAN,CONFIG,paths); print('FROZEN_ALIGNED_TRAINING',sid,flush=True)


def verify(): return common.verify(PLAN,CONFIG)


def mine():
    sid = verify()
    from tools.prepare_replacement_learning_v1 import ExactCounter
    inputs = common.load(PLAN/'public/source_inputs.json')
    old_inputs = common.load(parent.PLAN/'public/reward_inputs.json')
    auxiliary = defaultdict(list)
    for r in old_inputs:
        if r['variant'] in ['one_copy_remove','joint_copy_remove']: auxiliary[r['question_id']].append(r)
    counter = ExactCounter(); torch,tok,model,_ = ranker.load(parent.MODEL)
    state = torch.load(precision.PARENT_RUN/'adapter.pt',map_location='cpu',weights_only=True)
    ranker.load_adapter(model,state)
    restores = [(p,state[n].to(p.device)) for n,p in model.named_parameters() if n.endswith('.b')]
    reward_inputs = []; manifests = {}
    for source in inputs:
        qid = source['question_id']; target = RUN/'mined'/(qid.replace(':','_')+'.json')
        if target.exists():
            saved = common.load(target)
            if saved['signature']!=sid or saved['source_sha256']!=fingerprint(source): raise ValueError('Mined source changed')
        else:
            intent = RUN/'mining_inflight.json'
            if intent.exists(): raise ValueError('Unknown mining commit; inspect before retry')
            write_json(intent,dict(signature=sid,question_id=qid))
            records = source['records']; pool = source['candidate_pool_ids']; question = source['question']
            individual = {}; scores = {}; lengths = {}; started=time.perf_counter()
            with torch.inference_mode():
                for parameter,_ in restores: parameter.zero_()
                for key in pool:
                    ids=common.semantic_ids(tok,question,evidence.render([key],records))
                    individual[key]=float(ranker.logit(torch,tok,model,ids).sigmoid())
                for parameter, value in restores: parameter.copy_(value)
                single=core.proposal_sets(question,source['seed_ids'],pool,records,individual,counter,False,max_sets=24)
                group=core.proposal_sets(question,source['seed_ids'],pool,records,individual,counter,True,max_sets=24)
                union=precision.ordered_union(single,group)
                for p in union:
                    ids=ranker.input_ids(tok,question,p['evidence_text'])
                    scores[p['evidence_sha256']]=float(ranker.logit(torch,tok,model,ids)); lengths[p['evidence_sha256']]=len(ids)
                chosen=[('natural_anchor',single[0])]
                packs=[p for p in union if p['proposal']=='semantic_pack']
                if packs: chosen.append(('natural_semantic',packs[0]))
                for kind,label in [('singleton_swap','single'),('group_swap','group')]:
                    options=sorted((p for p in union if p['proposal']==kind),key=lambda p:-scores[p['evidence_sha256']])
                    chosen.extend((f'natural_{label}_{i+1}',p) for i,p in enumerate(options[:2]))
                rows=[]; seen=set()
                for arm,p in chosen:
                    if p['evidence_sha256'] in seen: continue
                    seen.add(p['evidence_sha256'])
                    rows.append(dict(question_id=qid,question=question,split=source['split'],mode=source['mode'],
                        arm=arm,**p,teacher_score=scores[p['evidence_sha256']],training_kind='natural'))
                for old in auxiliary[qid]:
                    text=old['evidence_text']; digest=fingerprint(text)
                    if digest in seen: continue
                    seen.add(digest)
                    ids=ranker.input_ids(tok,question,text)
                    score=float(ranker.logit(torch,tok,model,ids))
                    rows.append(dict(question_id=qid,question=question,split=source['split'],mode=source['mode'],
                        arm='aux_'+old['variant'],evidence_text=text,evidence_sha256=digest,
                        evidence_tokens=old['evidence_tokens'],teacher_score=score,training_kind='auxiliary'))
            saved=dict(signature=sid,source_sha256=fingerprint(source),rows=rows,individual_scores=individual,
                alternatives=union,teacher_scores=scores,costs=dict(semantic_forward_calls=len(pool),
                set_forward_calls=len(scores),seconds=time.perf_counter()-started))
            common.immutable_json(target,saved); intent.unlink()
        reward_inputs.extend(saved['rows']); manifests[str(target.relative_to(ROOT))]=parent.base.sha(target)
        print('MINED_NATURAL',qid,len(saved['rows']),len(reward_inputs),flush=True)
    if len({r['question_id'] for r in reward_inputs})!=224: raise ValueError('Missing source questions')
    common.immutable_json(RUN/'reward_inputs.json',reward_inputs)
    common.immutable_json(RUN/'mining_complete.json',dict(signature=sid,rows=len(reward_inputs),
        inputs_sha256=parent.base.sha(RUN/'reward_inputs.json'),mined=manifests,
        kinds=dict(Counter(r['training_kind'] for r in reward_inputs)),heldout_accessed=False))


def verify_mining():
    sid=verify(); meta=common.load(RUN/'mining_complete.json')
    if meta['signature']!=sid or meta['inputs_sha256']!=parent.base.sha(RUN/'reward_inputs.json'):
        raise ValueError('Mined feedback inputs changed')
    for path,digest in meta['mined'].items():
        if parent.base.sha(ROOT/path)!=digest: raise ValueError('Mined candidate artifact changed: '+path)
    return sid


def rewards(endpoint):
    sid=verify_mining()
    common.run_qa(RUN/'feedback',common.load(RUN/'reward_inputs.json'),sid,endpoint,
                  [controls.RUN/'qa/answer_calls.jsonl'])


def score_rewards():
    sid=verify_mining(); plans=common.load(RUN/'reward_inputs.json')
    rows=list(read_jsonl(RUN/'feedback/qa/predictions.jsonl'))
    expected={(p['question_id'],p['arm']):p for p in plans}
    if len(rows)!=len(expected) or {(r['question_id'],r['arm']) for r in rows}!=set(expected): raise ValueError('Feedback incomplete')
    refs=common.load(PLAN/'private/references.json'); targets=[]
    for row in rows:
        if row['signature']!=sid or row['plan_sha256']!=fingerprint(expected[row['question_id'],row['arm']]):
            raise ValueError('Feedback identity mismatch')
        ref=refs[row['question_id']]; aliases=list(dict.fromkeys([ref['answer']]+ref.get('aliases',[])))
        reward=max(parent.base.native_f1(row['hypothesis'],a,2) for a in aliases) if row['status']=='ok' else 0.
        targets.append(dict(question_id=row['question_id'],arm=row['arm'],split=row['split'],reward=reward,
            status=row['status'],evidence_tokens=row['evidence_tokens'],training_kind=row['training_kind']))
    common.immutable_json(RUN/'reward_targets.json',targets)
    pools=preference_buckets(plans,targets,'train')
    counts={key:sum(len(v[key]) for v in pools.values()) for key in ['natural','auxiliary','neutral']}
    natural_q=sum(bool(v['natural']) for v in pools.values())
    common.immutable_json(RUN/'feedback_audit.json',dict(signature=sid,conditions=len(targets),
        statuses=dict(Counter(r['status'] for r in targets)),training_buckets=counts,
        natural_non_tie_questions=natural_q,targets_sha256=parent.base.sha(RUN/'reward_targets.json'),
        ready=natural_q>=32 and counts['neutral']>0,heldout_accessed=False))
    print('FEEDBACK_AUDIT',counts,'natural_non_tie_questions',natural_q,flush=True)


def preference_buckets(inputs, targets, split):
    by={(r['question_id'],r['arm']):r for r in inputs if r['split']==split}; questions=defaultdict(list)
    for target in targets:
        if target['split']==split and target['status']=='ok': questions[target['question_id']].append(target)
    result={}
    for qid,values in sorted(questions.items()):
        buckets=dict(natural=[],auxiliary=[],neutral=[])
        for a,b in itertools.combinations(values,2):
            if not core.matched_cost(a['evidence_tokens'],b['evidence_tokens']): continue
            ra,rb=by[qid,a['arm']],by[qid,b['arm']]
            natural=ra['training_kind']==rb['training_kind']=='natural'
            if a['reward']==b['reward']:
                if not natural: continue
                kind='neutral'
            else: kind='natural' if natural else 'auxiliary'
            buckets[kind].append((ra,rb,a['reward'],b['reward']))
        for kind in buckets:
            def order(p):
                anchor=any(r['arm']=='natural_anchor' for r in p[:2])
                return (not anchor,fingerprint([CONFIG['seed'],qid,p[0]['arm'],p[1]['arm']]))
            buckets[kind]=sorted(buckets[kind],key=order)[:CONFIG['pair_cap_per_bucket']]
        result[qid]=buckets
    return result


def draw_schedule(pools,condition):
    if condition not in CONDITIONS: raise ValueError('Unknown training condition')
    rng=random.Random(CONFIG['seed']); with_neutral=condition=='aligned_with_neutral'
    eligible=[q for q,p in pools.items() if p['natural'] or p['auxiliary'] or (with_neutral and p['neutral'])]
    if not eligible: raise ValueError('No usable training preferences')
    rng.shuffle(eligible); schedule=[]; index=0
    for _ in range(CONFIG['updates']*CONFIG['accumulation']):
        if index==len(eligible): rng.shuffle(eligible);index=0
        qid=eligible[index];index+=1;p=pools[qid]
        if with_neutral and p['neutral'] and (rng.random()<.25 or not(p['natural'] or p['auxiliary'])): kind='neutral'
        elif p['natural'] and p['auxiliary']: kind='auxiliary' if rng.random()<.25 else 'natural'
        elif p['natural']: kind='natural'
        elif p['auxiliary']: kind='auxiliary'
        else: kind='neutral'
        schedule.append((kind,rng.choice(p[kind])))
    return schedule


def train():
    sid=verify_mining(); audit=common.load(RUN/'feedback_audit.json')
    if not audit['ready'] or audit['targets_sha256']!=parent.base.sha(RUN/'reward_targets.json'):
        raise ValueError('Feedback failed readiness/integrity gate; do not train on inadequate supervision')
    inputs=common.load(RUN/'reward_inputs.json'); targets=common.load(RUN/'reward_targets.json')
    pools=preference_buckets(inputs,targets,'train')
    for condition in CONDITIONS:
        out=RUN/'models'/condition
        if (out/'training_complete.json').exists():
            meta=common.load(out/'training_complete.json')
            if meta['signature']!=sid or meta['checkpoint_sha256']!=parent.base.sha(out/'adapter.pt'): raise ValueError('Completed checkpoint changed')
            continue
        if (out/'training_started.json').exists(): raise ValueError('Training already attempted; no hidden restart')
        torch,tok,model,_=ranker.load(parent.MODEL,training=True)
        encoded={(r['question_id'],r['arm']):ranker.input_ids(tok,r['question'],r['evidence_text']) for r in inputs}
        parameters=[p for p in model.parameters() if p.requires_grad]
        if sum(p.numel() for p in parameters)!=655360: raise ValueError('Unexpected trainable architecture')
        schedule=draw_schedule(pools,condition)
        common.immutable_json(out/'training_started.json',dict(signature=sid,condition=condition,
            targets_sha256=audit['targets_sha256'],updates=CONFIG['updates'],pairs=len(schedule),
            trainable_parameters=sum(p.numel() for p in parameters),
            schedule_sha256=fingerprint([[kind,p[0]['question_id'],p[0]['arm'],p[1]['arm']] for kind,p in schedule])))
        optimizer=torch.optim.AdamW(parameters,lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
        optimizer.zero_grad(set_to_none=True);started=time.perf_counter();draws=Counter();input_tokens=0
        for update in range(CONFIG['updates']):
            losses=[]
            for kind,pair in schedule[update*CONFIG['accumulation']:(update+1)*CONFIG['accumulation']]:
                a,b,ra,rb=pair;va=ranker.logit(torch,tok,model,encoded[a['question_id'],a['arm']])
                vb=ranker.logit(torch,tok,model,encoded[b['question_id'],b['arm']])
                pref=torch.tensor(1/(1+math.exp(-CONFIG['difference_temperature']*(ra-rb))),device='cuda')
                loss=torch.nn.functional.binary_cross_entropy_with_logits(va-vb,pref)
                loss=loss+CONFIG['regression_weight']*((va.sigmoid()-ra)**2+(vb.sigmoid()-rb)**2)/2
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                (loss/CONFIG['accumulation']).backward();losses.append(float(loss.detach()));draws[kind]+=1
                input_tokens+=len(encoded[a['question_id'],a['arm']])+len(encoded[b['question_id'],b['arm']])
            torch.nn.utils.clip_grad_norm_(parameters,CONFIG['clip'],error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            parent.base.append(out/'training_log.jsonl',dict(update=update+1,mean_loss=statistics.mean(losses),
                seconds=time.perf_counter()-started,draws=dict(draws),input_tokens=input_tokens))
            if (update+1)%10==0: print('ALIGNED_TRAIN',condition,update+1,'/',CONFIG['updates'],dict(draws),flush=True)
        torch.save(ranker.adapter_state(model),out/'adapter.pt')
        common.immutable_json(out/'training_complete.json',dict(signature=sid,condition=condition,
            checkpoint_sha256=parent.base.sha(out/'adapter.pt'),updates=CONFIG['updates'],draws=dict(draws),
            input_tokens=input_tokens,seconds=time.perf_counter()-started))
        del va,vb,loss,pref,optimizer,parameters,model;import gc;gc.collect();torch.cuda.empty_cache()


def evaluate_dev():
    sid=verify_mining();inputs=[r for r in common.load(RUN/'reward_inputs.json') if r['split']=='dev' and r['training_kind']=='natural']
    targets={(r['question_id'],r['arm']):r for r in common.load(RUN/'reward_targets.json')}
    reports=[]
    for condition in ['v1_teacher',*CONDITIONS]:
        path=RUN/'dev'/(condition+'.json')
        if path.exists():
            if common.load(path)['signature']!=sid: raise ValueError('Wrong dev identity')
            reports.append(common.load(path)['report']);continue
        if condition=='v1_teacher': scores={(r['question_id'],r['arm']):r['teacher_score'] for r in inputs}
        else:
            torch,tok,model,_=ranker.load(parent.MODEL)
            state=common.load(RUN/'models'/condition/'training_complete.json')
            if state['checkpoint_sha256']!=parent.base.sha(RUN/'models'/condition/'adapter.pt'): raise ValueError('Checkpoint changed')
            ranker.load_adapter(model,torch.load(RUN/'models'/condition/'adapter.pt',map_location='cpu',weights_only=True))
            with torch.inference_mode(): scores={(r['question_id'],r['arm']):common.teacher_score(torch,tok,model,r['question'],r['evidence_text']) for r in inputs}
            del model;import gc;gc.collect();torch.cuda.empty_cache()
        byq=defaultdict(list)
        for r in inputs: byq[r['question_id']].append(r)
        selected=[];anchor=[];directions=Counter()
        for qid,values in byq.items():
            chosen=max(values,key=lambda r:scores[qid,r['arm']]);selected.append(targets[qid,chosen['arm']]['reward'])
            anchor.append(targets[qid,'natural_anchor']['reward'])
            for a,b in itertools.combinations(values,2):
                ta,tb=targets[qid,a['arm']],targets[qid,b['arm']]
                if ta['status']!='ok' or tb['status']!='ok' or ta['reward']==tb['reward'] or not core.matched_cost(a['evidence_tokens'],b['evidence_tokens']): continue
                d=(scores[qid,a['arm']]-scores[qid,b['arm']])*(ta['reward']-tb['reward'])
                directions['correct' if d>0 else 'tie' if d==0 else 'wrong']+=1
        report=dict(condition=condition,questions=len(byq),selected_f1_points=100*statistics.mean(selected),
            anchor_f1_points=100*statistics.mean(anchor),pair_directions=dict(directions),
            scope='32 legal training-dev questions; only natural candidate contexts; fixed checkpoints')
        common.immutable_json(path,dict(signature=sid,report=report,
            scores=[dict(question_id=q,arm=a,score=v) for (q,a),v in scores.items()]))
        reports.append(report)
    common.immutable_json(RUN/'dev_report.json',dict(signature=sid,reports=reports,heldout_accessed=False))
    print('NATURAL_DEV_REPORT',reports,flush=True)


def select():
    sid=verify_mining();plans=[];questions={r['question_id']:r for r in common.load(parent.base.PLAN/'public/queries.json')}
    for condition in CONDITIONS:
        torch,tok,model,_=ranker.load(parent.MODEL)
        checkpoint=RUN/'models'/condition/'adapter.pt'
        if common.load(checkpoint.parent/'training_complete.json')['checkpoint_sha256']!=parent.base.sha(checkpoint): raise ValueError('Checkpoint changed')
        ranker.load_adapter(model,torch.load(checkpoint,map_location='cpu',weights_only=True))
        for path in sorted((precision.RUN/'sets').glob('*.json')):
            old=common.load(path);qid=old['plans'][0]['question_id'];q=questions[qid]
            target=RUN/'selected'/condition/path.name
            if target.exists():
                saved=common.load(target)
                if saved['signature']!=sid: raise ValueError('Selection identity changed')
            else:
                exact=common.load(controls.RUN/'selected'/path.name)
                alternatives=dict(union=old['alternatives']['union_fp32'],wide_single=old['alternatives']['wide_single_fp32'],
                    matched_single=exact['alternatives']['matched_single_bf16'],matched_union=exact['alternatives']['matched_union_bf16'])
                unique=precision.ordered_union(alternatives['union'],alternatives['wide_single'])
                intent=RUN/'selection_inflight.json'
                if intent.exists(): raise ValueError('Unknown selection commit')
                write_json(intent,dict(signature=sid,condition=condition,question_id=qid))
                with torch.inference_mode(): values={p['evidence_sha256']:common.teacher_score(torch,tok,model,q['question'],p['evidence_text']) for p in unique}
                chosen=[]
                for suffix,candidates in alternatives.items():
                    best=max(candidates,key=lambda p:values[p['evidence_sha256']])
                    chosen.append(dict(**q,arm=condition+'_'+suffix,**best,scored_sets=len(candidates),private_inputs_read=False))
                saved=dict(signature=sid,plans=chosen,scores=values,source_candidates_sha256=parent.base.sha(path))
                common.immutable_json(target,saved);intent.unlink()
            plans.extend(saved['plans']); print('ALIGNED_SELECT',condition,qid,flush=True)
        del model;import gc;gc.collect();torch.cuda.empty_cache()
    common.immutable_json(RUN/'deployment/qa_plans.json',plans)


def qa(endpoint):
    sid=verify_mining();common.run_qa(RUN/'deployment',common.load(RUN/'deployment/qa_plans.json'),sid,endpoint,
        [controls.RUN/'qa/answer_calls.jsonl',RUN/'feedback/qa/answer_calls.jsonl'])


def score():
    sid=verify_mining();rows=common.official_rows(common.load(RUN/'deployment/qa_plans.json'),list(read_jsonl(RUN/'deployment/qa/predictions.jsonl')))
    groups=defaultdict(list)
    for r in rows: groups[r['arm']].append(r)
    report=dict(signature=sid,complete=True,scope='64 exposed development questions; no independent confirmation',
        scores=[dict(arm=a,n=len(rs),native_f1_points=100*statistics.mean(r['official_native_f1'] for r in rs),
                     statuses=dict(Counter(r['status'] for r in rs))) for a,rs in sorted(groups.items())],
        controls_report=common.load(controls.RUN/'report.json'),dev_report=common.load(RUN/'dev_report.json'),heldout_accessed=False)
    common.immutable_json(RUN/'scored.json',rows);common.immutable_json(RUN/'report.json',report)
    print('ALIGNED_FINAL',report['scores'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','freeze','verify','mine','rewards','score_rewards','train','evaluate_dev','select','qa','score'])
    parser.add_argument('--endpoint',default='http://127.0.0.1:8018');args=parser.parse_args()
    if args.mode in ['qa','rewards']: globals()[args.mode](args.endpoint)
    else:
        result=globals()[args.mode]()
        if args.mode=='verify': print('VERIFIED_ALIGNED',result,flush=True)
