"""Gold-blind equal-budget proposals and exact source-span interventions."""
from copy import deepcopy
import itertools
import math
import re

from . import evidence_acquisition_v1 as evidence
from .schema import fingerprint

BUDGET=2048
SEED='replacement-learning-20261028-seed29'


def terms(text):
    stop=set('a an the of in on at by for to is was were are be what which who where when how did does do from and or that this as its it'.split())
    return set(re.findall(r'[a-z0-9]+',text.lower()))-stop


def lexical_scores(question,records,pool):
    q=terms(question)
    counts={t:sum(t in terms(records[k]['feature_text']) for k in pool) for t in q}
    return {k:sum(math.log(1+len(pool)/(1+counts[t])) for t in q&terms(records[k]['feature_text']))
        /max(1,len(terms(records[k]['feature_text'])))**0.15 for k in pool}


def fusion_order(pool,records):
    raw=[k for k in pool if records[k]['kind']=='raw']
    memory=[k for k in pool if records[k]['kind']!='raw']
    return [k for pair in itertools.zip_longest(raw,memory) for k in pair if k]


def matched_cost(a,b):
    return abs(a-b)<=max(64,0.05*max(a,b))


def swap_pack(seed,action,records,scores,counter,budget=BUDGET):
    """Remove low-ranked entries until an entire proposed action fits; score later."""
    action=list(dict.fromkeys(action))
    if not action or all(k in seed for k in action):return None
    keep=[k for k in seed if k not in action]
    removed=[]
    drop_order=sorted(keep,key=lambda k:(scores[k],-seed.index(k)))
    def assemble():
        at=min((seed.index(k) for k in removed),default=len(keep))
        at=min(at,len(keep))
        return keep[:at]+action+keep[at:]
    while counter.count(evidence.render(assemble(),records))>budget:
        if not drop_order:return None
        key=drop_order.pop(0);keep.remove(key);removed.append(key)
    selected=assemble()
    # Fill with original anchor entries only; no arbitrary alternating padding.
    for key in seed:
        if key not in selected and counter.count(evidence.render(selected+[key],records))<=budget:
            selected.append(key)
    if evidence.render(selected,records)==evidence.render(seed,records):return None
    return selected


def proposal_sets(question,seed,pool,records,scores,counter,allow_groups,max_sets=24,pairs=()):
    """All arguments are public. References/support labels never enter this function."""
    if not counter.exact:raise ValueError('Exact reader tokenizer required')
    if counter.count(evidence.render(seed,records))>BUDGET:raise ValueError('Anchor exceeds budget')
    if len(set(pool))!=len(pool) or not set(seed)<=set(pool) or set(scores)!=set(pool):
        raise ValueError('Invalid shared candidate pool or scores')
    rank={k:i for i,k in enumerate(pool)}
    ordered=sorted(pool,key=lambda k:(-scores[k],rank[k]))
    base_tokens=counter.count(evidence.render(seed,records))
    result=[];seen=set()
    def add(selected,action,kind):
        if selected is None:return
        text=evidence.render(selected,records);size=counter.count(text);key=fingerprint(text)
        if key in seen or size>BUDGET:return
        if kind!='anchor' and not matched_cost(size,base_tokens):return
        seen.add(key)
        result.append(dict(selected_ids=selected,evidence_text=text,evidence_tokens=size,
            proposal=kind,action=list(action),evidence_sha256=key))
    add(list(seed),[],'anchor')
    add(evidence.ordered_pack(ordered,records,counter,BUDGET),[],'semantic_pack')
    singles=[k for k in ordered if k not in seed]
    single_limit=max_sets-2 if not allow_groups else (max_sets-2)//2
    for k in singles:
        add(swap_pack(seed,[k],records,scores,counter),[k],'singleton_swap')
        if sum(r['proposal']=='singleton_swap' for r in result)>=single_limit:break
    if allow_groups:
        actions=[];known=set()
        for item in pairs:
            values=item['ids'] if isinstance(item,dict) else item
            if len(values)!=2 or not set(values)<=set(pool):continue
            key=tuple(sorted(values))
            if key not in known:actions.append(list(values));known.add(key)
        # Semantic ranking proposes plausible pairs; this does not certify dependency.
        for a,b in itertools.combinations(ordered[:16],2):
            if a in seed and b in seed:continue
            shared=terms(records[a]['feature_text'])&terms(records[b]['feature_text'])
            if not shared:continue
            key=tuple(sorted((a,b)))
            if key not in known:actions.append([a,b]);known.add(key)
        actions.sort(key=lambda a:(-sum(scores[k] for k in a),tuple(rank[k] for k in a)))
        for action in actions:
            add(swap_pack(seed,action,records,scores,counter),action,'group_swap')
            if len(result)>=max_sets:break
    return result[:max_sets]


def sentence_spans(text):
    # Source spans, not a claim of perfect semantic atomization.
    spans=[]
    for m in re.finditer(r'\S[\s\S]*?(?:[.!?](?=\s+[A-Z0-9"\[])|\Z)',text):
        value=m.group().strip()
        if value:spans.append(dict(start=m.start(),end=m.end(),text=value))
    return spans or [dict(start=0,end=len(text),text=text)]


def intervene(records,units,only_keys=None):
    """Erase confirmed exact source spans, retaining the rest of each raw paragraph."""
    changed=deepcopy(records);audit=[]
    for key,r in changed.items():
        if only_keys is not None and key not in only_keys:continue
        for unit in units:
            value=r.get('source_spans',{}).get(unit)
            if value is None:continue
            if r['kind']=='raw':
                for turn in r['turns']:
                    if value in turn['content']:
                        turn['content']=turn['content'].replace(value,'',1).strip()
                        audit.append(dict(entry_id=key,unit=unit,exact_text=value))
                r['turns']=[t for t in r['turns'] if t['content']]
                r['feature_text']=' '.join(t['content'] for t in r['turns'])
            else:
                text=r['entry']['lossless_restatement']
                if value not in text:raise ValueError('Invalid claimed extractive copy')
                r['entry']['lossless_restatement']=text.replace(value,'',1).strip()
                audit.append(dict(entry_id=key,unit=unit,exact_text=value))
                # A bare title is not an evidence item after its sentence is removed.
                if len(r['source_spans'])==1:r['entry']['lossless_restatement']=''
                r['feature_text']=r['entry']['lossless_restatement']
            del r['source_spans'][unit]
    empty={k for k,r in changed.items() if not r['feature_text'].strip()}
    return {k:r for k,r in changed.items() if k not in empty},audit
