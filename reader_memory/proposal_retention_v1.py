"""Frozen-proposal postprocessing with an exact per-question token control."""
from collections import Counter
from pathlib import Path
import math
from . import length_repack_qa_v1 as u

ROOT=u.ROOT;BASE=ROOT/'research/proposal_retention_v1';RUN=ROOT/'runs/proposal_retention_v1'
KINDS=('role','direct')
ARMS=('retain_fill','greedy','knapsack','knapsack_matched')
CONFIG=dict(version='proposal-retention-v1',sources=96,per_dataset=48,budget=512,
    extractors=list(KINDS),arms=list(ARMS),logical_conditions=768,reader='Qwen3.5-9B',
    seed=30,precision='bfloat16',attention='sdpa',thinking=False,do_sample=False,
    max_input=8192,hotpot_max_output=128,musique_max_output=2048,
    training=False,new_proposer_calls=0,automatic_expansion=False,
    unit='original complete sentence, quote overlap expands to whole sentences',
    rendering='source-order units: title + colon-space + verbatim sentence + two newlines',
    utility='round(1e9 * sigmoid(frozen reranker logit)), minimum 1; not calibrated probability',
    primary='retain_fill minus knapsack_matched, separately by dataset and extractor',
    length_gate='all primary pairs exactly equal evidence AND full reader prompt token counts',
    minimum_distinct_primary_pairs_per_group=12, # Initial planning warning, retained in the preparation record.
    sparse_pair_handling='report all groups; the initial 12-pair planning warning is not a powered scientific stop test; see pre-inference protocol',
    planning_decision='role delta >=2 F1 on both datasets; direct delta >0 on both; fixed-cap controls no deficit worse than -1 F1 in any group',
    admission_mib=26000,reserve_mib=24000,gpu_cap_mib=24500,poll_seconds=.1,
    retry_delay_seconds=1,wait_timeout=None,wall_timeout=None,oom_retry_limit=None)


def units(source,scores,count):
    result=[]
    for p in source['paragraphs']:
        for i,sp in enumerate(p['spans']):
            uid=p['source_id']+':span:'+str(i)
            score=float(scores[uid]);assert math.isfinite(score)
            sentence=p['text'][sp['start']:sp['end']]
            assert sentence==sp['text'] if 'text' in sp else True
            text=p['title']+': '+sentence+'\n\n'
            value=max(1,round(1e9/(1+math.exp(-max(-100,min(100,score))))))
            result.append(dict(id=uid,source=p['source_id'],start=sp['start'],end=sp['end'],
                               text=text,cost=count(text),score=score,value=value))
    assert len(result)==len(scores) and {v['id'] for v in result}==set(scores)
    assert all(v['cost']>0 for v in result)
    return result


def proposal_core(source,items,parsed,kind):
    cs=parsed['candidates']
    if kind=='role':
        represented=set().union(*(set(c['roles']) for c in cs)) if cs else set()
        valid=parsed['status']=='valid' and bool(cs) and represented==set(range(len(parsed['roles'])))
    else:valid=parsed['schema_ok'] and parsed['usable']
    if not valid:return [],'unusable_or_incomplete_proposal'
    by={p['source_id']:p for p in source['paragraphs']};chosen=set()
    for c in cs:
        p=by[c['source']]
        assert p['text'][c['start']:c['end']]==c['text'],'Quote provenance mismatch'
        touched=[i for i,v in enumerate(items) if v['source']==c['source'] and v['start']<c['end'] and v['end']>c['start']]
        assert touched,'Quote has no source sentence'
        # Require every non-whitespace quoted character to be covered by a native span.
        for pos in range(c['start'],c['end']):
            assert p['text'][pos].isspace() or any(items[i]['start']<=pos<items[i]['end'] for i in touched)
        chosen.update(touched)
    return sorted(chosen),None


def greedy(items,cap,initial=()):
    selected=set(initial);used=sum(items[i]['cost'] for i in selected)
    assert used<=cap
    for i in sorted(range(len(items)),key=lambda j:(-items[j]['score'],j)):
        if i not in selected and used+items[i]['cost']<=cap:
            selected.add(i);used+=items[i]['cost']
    return tuple(sorted(selected))


def knapsack_table(items,cap):
    # Descending capacity enforces 0/1 selection; tuple state is immutable.
    dp=[None]*(cap+1);dp[0]=(0,())
    for i,item in enumerate(items):
        c=item['cost']
        for target in range(cap,c-1,-1):
            prev=dp[target-c]
            if prev is None:continue
            candidate=(prev[0]+item['value'],prev[1]+(i,))
            old=dp[target]
            if old is None or candidate[0]>old[0] or (candidate[0]==old[0] and candidate[1]<old[1]):
                dp[target]=candidate
    return dp


def make_plans(source,scores,parsed,kind,count):
    item=units(source,scores,count);cap=CONFIG['budget']
    core,reason=proposal_core(source,item,parsed,kind)
    core_tokens=sum(item[i]['cost'] for i in core)
    if core_tokens>cap:reason='complete_sentence_core_over_budget'
    retained=greedy(item,cap,core if reason is None else ())
    plain=greedy(item,cap);dp=knapsack_table(item,cap)
    best=max((v[0],-c,v[1]) for c,v in enumerate(dp) if v is not None)[2]
    target=sum(item[i]['cost'] for i in retained)
    assert dp[target] is not None # Retained itself is a feasible exact-length witness.
    matched=dp[target][1]
    selections=dict(retain_fill=retained,greedy=plain,knapsack=best,knapsack_matched=matched)
    plans=[]
    for arm,indices in selections.items():
        text=''.join(item[i]['text'] for i in indices);length=count(text)
        assert length==sum(item[i]['cost'] for i in indices)<=cap,'Nonadditive rendering or budget violation'
        plans.append(dict(question_id=source['question_id'],extractor=kind,arm=arm,budget=cap,
            evidence_text=text,evidence_tokens=length,selected_ids=[item[i]['id'] for i in indices],
            utility=sum(item[i]['value'] for i in indices),core_count=len(core),core_tokens=core_tokens,
            core_retained=sum(i in indices for i in core),proposal_fallback=reason,
            exact_target=target if arm=='knapsack_matched' else None,status='ready'))
    assert plans[0]['evidence_tokens']==plans[3]['evidence_tokens']
    assert plans[3]['utility']>=plans[0]['utility']
    assert plans[2]['utility']>=plans[1]['utility']
    return plans


def verify(models=False):
    m=u.load(BASE/'manifest.json');assert m['config']==CONFIG
    for n,h in m['files'].items():assert u.sha(ROOT/n)==h,n
    if models:
        for n,h in m['model_files'].items():assert u.sha(ROOT/n)==h,n
    return u.sha(BASE/'manifest.json')
