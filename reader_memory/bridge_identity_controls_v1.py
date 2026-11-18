"""Exactly token-matched bridge-identity perturbations; synthetic diagnostics only."""
import argparse
from collections import Counter
from copy import deepcopy
from itertools import product
from pathlib import Path
import re
import statistics

from . import anchor_intervention_v1 as parent
from .schema import fingerprint, read_jsonl

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'research/bridge_identity_controls_v1'
RUN=ROOT/'runs/bridge_identity_controls_v1'
load,sha,immutable,recomp=parent.load,parent.sha,parent.immutable,parent.recomp
BASE,INDEPENDENT,FULL=parent.BASE,parent.INDEPENDENT,parent.FULL
BOTH,A_ONLY,B_ONLY='bridge_rename_both','bridge_rename_anchor','bridge_rename_path'
METHODS=(BASE,INDEPENDENT,FULL,BOTH,A_ONLY,B_ONLY)
NEW_METHODS=(BOTH,A_ONLY,B_ONLY)
PAIRS=((FULL,BASE),(FULL,INDEPENDENT),(FULL,BOTH),(BOTH,A_ONLY),(BOTH,B_ONLY))
BUDGETS=(256,)
STOP={'The','A','An','In','On','At','By','With','For','From','It','He','She','His','Her','They',
    'This','That','After','Before','However','As','And','Of','To','City','County','State','School',
    'United','States','University','National','New','North','South','East','West','March','April',
    'May','June','July','August','September','October','November','December','January','February'}
CONNECTORS={'of','the','and','de','van','von'}
NAME=re.compile(r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*(?:[ \t]+(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*|of|the|and|de|van|von)){0,5}")
WORDS=('Mira','Vale','Arden','Nola','Sable','Tovan','Vexa','Meridian','Neral','Qorven',
       'Alvoria','Taverton','Zelvoria','Mirevans','Nalverton','Teralosia',
       'Luminara','Vorden','Neralo','Veloria','Ruvana','Selvero','Norvellen','Talmeria')
ALIASES=tuple(dict.fromkeys([*WORDS,*(' '.join(x) for x in product(WORDS[:12],WORDS[12:]))]))[:256]
# Public lexical review before any new QA: these one-word descriptors/name fragments
# remain recorded perturbation controls, not purported entity-link witnesses.
LEXICAL_QC={'American','Federal','Western','Eastern','Northern','Southern','Marine','League','Empire',
    'York','Pacific','Atlantic','British','English','Royal','International','Regional','Central','General',
    'Military','Public','Private','Professional','Historical','Modern','Ancient','National','State','County',
    'District','Province','Territory','University','College','School','City','Town','Village'}
CONFIG=dict(version='bridge-identity-controls-v1',questions=128,sources=64,budgets=[256],
    methods=list(METHODS),max_conditions=768,max_new_reader_calls=384,
    reader='Qwen3.5-9B',reader_variant='format_only',output_tokens=2048,
    decoding='greedy-thinking_false',retries=0,
    group_definition='unchanged public source-block rule from anchor_intervention_v1',
    candidate='shared capitalized phrase in A and B content, absent from question and background content/title',
    candidate_order='exact source-title or title-prefix match, descending word count and length, normalized lexical tie',
    max_entity_candidates=4,aliases=list(ALIASES),
    alias_search='first public entity and fixed alias yielding exact full-context token equality in all three perturbations',
    interventions='A-only rename, B-only rename, consistent A+B rename; content and titles, all literal mentions in affected blocks',
    original_question_unchanged=True,all_other_material_unchanged=True,exact_same_evidence_tokens=True,
    no_eligible_pair='byte-identical no-op copies retained in full denominator and explicitly flagged',
    endpoint_overlap='reference overlap checked only after selection; flagged, never used to choose entity or alias',
    lexical_qc_words=sorted(LEXICAL_QC),
    lexical_qc='public pre-QA word review: single-word descriptors or ambiguous name fragments retained as perturbation controls, excluded from entity-link witness count',
    primary='public eligible, lexical QC clear, no endpoint overlap, all four calls ok, original and consistent rename exact, both broken variants nonexact',
    strict_secondary='both broken variants zero original-answer F1 under the primary criteria',
    outcome='original-reference answer retention F1 in synthetic stimuli; not new natural-task accuracy',
    semantic_bridge_certified=False,independent_confirmation=False,training=False,
    new_retrieval=False,new_encoder_calls=0,protected_final_data_accessed=False,
    paper_ready=False,automatic_followup=False)


def arm(method,budget=256):
    if method not in METHODS or budget!=256:raise ValueError('Unregistered identity control')
    return method+'__b256'


def pattern(name):
    return re.compile(r'(?<!\w)'+r'\s+'.join(re.escape(x) for x in name.split())+r'(?!\w)',re.IGNORECASE)


def records_for(source,full,groups):
    by={p['source_id']:p for p in source['paragraphs']}
    labels={u:label for label,key in (('a','a'),('b','b'),('background','background')) for u in groups[key]}
    records=[]
    for pid,ranges in full['selected_ranges'].items():
        kinds={labels[u] for u in full['selected_spans'] if u.split(':')[0]==pid}
        if len(kinds)!=1:return None
        p=by[pid]
        records.append(dict(source_id=pid,group=next(iter(kinds)),title=p['title'],timestamp=p['timestamp'],
            content='\n[...]\n'.join(p['text'][a:b] for a,b in ranges)))
    return records


def render(records):
    values={r['source_id']:dict(kind='raw',turns=[dict(source_id=r['source_id'],timestamp=r['timestamp'],
        speaker=r['title'],content=r['content'])]) for r in records}
    return recomp.evidence.render(list(values),values)


def entity_candidates(source,records):
    a='\n'.join(r['content'] for r in records if r['group']=='a')
    b='\n'.join(r['content'] for r in records if r['group']=='b')
    c='\n'.join(r['title']+'\n'+r['content'] for r in records if r['group']=='background')
    names={}
    titles={parent.literal_key(x) for r in records for x in (r['title'],re.split(r'[,()]',r['title'])[0])}
    for r in records:
        for text in (r['title'],r['content']):
            for m in NAME.finditer(text):
                words=m.group().split()
                for start in range(len(words)):
                    for end in range(start+1,min(len(words),start+6)+1):
                        part=words[start:end]
                        if part[0] in STOP or part[0].casefold() in CONNECTORS or part[-1].casefold() in CONNECTORS:continue
                        if len(part)==1 and (part[0] in STOP or len(part[0])<4):continue
                        name=' '.join(part);pat=pattern(name)
                        if not pat.search(a) or not pat.search(b) or pat.search(source['question']) or pat.search(c):continue
                        key=parent.literal_key(name)
                        names.setdefault(key,name)
    return sorted(names.values(),key=lambda name:(-int(parent.literal_key(name) in titles),
        -len(name.split()),-len(name),parent.literal_key(name)))


def renamed(records,entity,alias,groups):
    pat=pattern(entity);changed=deepcopy(records);edits=[]
    for r in changed:
        if r['group'] not in groups:continue
        for field in ('title','content'):
            before=r[field];after,n=pat.subn(alias,before)
            if n:
                r[field]=after
                edits.append(dict(source_id=r['source_id'],group=r['group'],field=field,occurrences=n,
                    before_sha256=fingerprint(before),after_sha256=fingerprint(after)))
    return render(changed),edits


def view_plan(full,text,method,trace,counter,edits):
    result=deepcopy(full)
    result.update(method=method,arm=arm(method),diagnostic_only=True,
        synthetic_counterfactual=trace['eligible'],offsets_reference_original=True,
        original_evidence_sha256=full['evidence_sha256'],evidence_text=text,
        evidence_tokens=counter.count(text),evidence_sha256=fingerprint(text),
        identity_intervention=dict(entity=trace.get('entity'),alias=trace.get('alias'),edits=edits))
    return result


def plans_for(source,scores,blind_path,full,baseline,independent,counter):
    if not counter.exact or any(p['evidence_budget']!=256 for p in (full,baseline,independent)):
        raise ValueError('Exact tokenizer and fixed primary budget required')
    groups=parent.groups_for(source,scores,blind_path,full,baseline)
    records=records_for(source,full,groups)
    if records is not None and render(records)!=full['evidence_text']:
        raise ValueError('Original presentation does not reproduce')
    trace=dict(eligible=False,reason='no_other_path_block',entity=None,alias=None,groups=groups,
        target_tokens=full['evidence_tokens'],considered_entities=[])
    chosen=None
    if groups['both_blocks_present']:
        if records is None:trace['reason']='mixed_source_block'
        else:
            candidates=entity_candidates(source,records)
            trace['considered_entities']=candidates[:4]
            trace['reason']='no_shared_public_entity' if not candidates else 'no_exact_token_alias'
            original='\n'.join([source['question'],*(p['title']+'\n'+p['text'] for p in source['paragraphs'])])
            for entity in candidates[:4]:
                for alias in ALIASES:
                    if pattern(alias).search(original):continue
                    rendered=[renamed(records,entity,alias,g) for g in ({'a','b'},{'a'},{'b'})]
                    if all(counter.count(text)==full['evidence_tokens'] for text,_ in rendered):
                        trace.update(eligible=True,reason='exact_matched_shared_entity',entity=entity,alias=alias)
                        chosen=rendered;break
                if chosen is not None:break
    if chosen is None:chosen=[(full['evidence_text'],[]) for _ in NEW_METHODS]
    new=[view_plan(full,text,method,trace,counter,edits) for method,(text,edits) in zip(NEW_METHODS,chosen)]
    if any(p['evidence_tokens']!=full['evidence_tokens'] for p in new):raise ValueError('Token match failed')
    trace['edits']={p['method']:p['identity_intervention']['edits'] for p in new}
    return [deepcopy(baseline),deepcopy(independent),deepcopy(full),*new],trace


def replay_cache():
    marker=load(PLAN/'preparation.json')['reader_marker_sha256'];cache={}
    for item in load(PLAN/'private/replay_cache.json'):
        recomp.interface.parent.check_cache_entry(item,marker)
        if item['key'] in cache:raise ValueError('Duplicate historical request')
        cache[item['key']]=item['entry']
    return cache


def freeze():
    prep=load(PLAN/'preparation.json')
    names=['reader_memory/bridge_identity_controls_v1.py','tools/prepare_bridge_identity_controls_v1.py',
        'scripts/bridge_identity_controls_v1_queue.py','tests/test_bridge_identity_controls_v1.py',
        'tools/status_bridge_identity_controls_v1.py','research/anchor_intervention_v1/manifest.json',*prep['server_source_files']]
    names += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file() and 'snapshots' not in p.relative_to(PLAN).parts
        and p.suffix!='.pyc' and p.name not in {'manifest.json','CURRENT_STATUS.md','submission_receipt.json','STATUS_LATEST.json','SNAPSHOT_LATEST.json'}]
    value=dict(config=CONFIG,parent_signature=parent.verify(),files={n:sha(ROOT/n) for n in sorted(set(names))})
    value['signature']=fingerprint(value);immutable(PLAN/'manifest.json',value)
    print('FROZEN_BRIDGE_IDENTITY',value['signature'],flush=True)


def verify():
    v=load(PLAN/'manifest.json')
    if v['config']!=CONFIG or v['signature']!=fingerprint({k:x for k,x in v.items() if k!='signature'}):raise ValueError('Identity protocol changed')
    if parent.verify()!=v['parent_signature']:raise ValueError('Frozen parent changed')
    for n,d in v['files'].items():
        if sha(ROOT/n)!=d:raise ValueError('Frozen input changed: '+n)
    return v['signature']


def preflight():
    sid=verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    prep=load(PLAN/'preparation.json');marker=prep['reader_marker_sha256']
    if sha(ROOT/'models/Qwen3.5-9B/.verified.json')!=marker:raise ValueError('Reader model changed')
    sources,blind=load(PLAN/'public/source_inputs.json'),load(PLAN/'public/blind_paths.json')
    plans=load(PLAN/'public/qa_plans.json');by={(p['question_id'],p['arm']):p for p in plans}
    expected={(s['question_id'],arm(m)) for s in sources for m in METHODS}
    if len(sources)!=128 or len({s['native_id'] for s in sources})!=64 or len(plans)!=768 or set(by)!=expected:raise ValueError('Incomplete fixed panel')
    counter=ExactCounter();traces=[]
    for index,s in enumerate(sources,1):
        q=s['question_id'];cached=load(ROOT/'runs/reranker_budget_v1/selected'/(q.replace(':','_')+'.json'))
        if cached['source_sha256']!=fingerprint(s) or blind[q]['source_sha256']!=fingerprint(s):raise ValueError('Source/path provenance changed')
        if blind[q]['predicted_hops']!=len(blind[q]['seed_ids']):raise ValueError('Blind path count changed')
        generated,trace=plans_for(s,cached['scores'],blind[q]['seed_ids'],by[q,arm(FULL)],by[q,arm(BASE)],by[q,arm(INDEPENDENT)],counter)
        if any(p!=by[q,p['arm']] for p in generated):raise ValueError('Server identity stimulus mismatch')
        traces.append(dict(question_id=q,**trace))
        print('IDENTITY_REPRODUCE',index,'/128',flush=True)
    if traces!=load(PLAN/'public/identity_traces.json'):raise ValueError('Public perturbation trace changed')
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    cache,keys,maximum=replay_cache(),set(),0
    for p in plans:
        if counter.count(p['evidence_text'])!=p['evidence_tokens'] or not 0<p['evidence_tokens']<=256 or fingerprint(p['evidence_text'])!=p['evidence_sha256']:raise ValueError('Evidence identity changed')
        if p['method'] in NEW_METHODS and p['evidence_tokens']!=by[p['question_id'],arm(FULL)]['evidence_tokens']:raise ValueError('Exact token control failed')
        messages=recomp.messages_for(p);key=recomp.interface.parent.request_key(messages,marker);keys.add(key)
        if p['method'] not in NEW_METHODS and key not in cache:raise ValueError('Original controls must be reused')
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        maximum=max(maximum,counter.count(text))
    new=len(keys-set(cache))
    if new>384 or new!=prep['new_reader_calls_upper_bound'] or maximum+2048>32768:raise ValueError('Inference cap or prompt capacity failed')
    immutable(RUN/'qa_plans.json',plans)
    immutable(RUN/'selection_complete.json',dict(signature=sid,plans_sha256=sha(RUN/'qa_plans.json'),conditions=768,uncached_requests=new))
    immutable(RUN/'preflight.json',dict(signature=sid,passed=True,questions=128,sources=64,conditions=768,
        unique_requests=len(keys),uncached_requests=new,full_prompt_max_tokens=maximum,server_selection_reproduced=True,
        exact_token_matching=True,eligible_questions=sum(t['eligible'] for t in traces),new_training=False,
        new_encoder_calls=0,protected_final_data_accessed=False))
    print('BRIDGE_IDENTITY_PREFLIGHT_OK',new,'uncached calls',flush=True)


def qa(endpoint):
    sid=verify();ledger=recomp.ledger
    old=ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for
    ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for=RUN,lambda:sid,replay_cache,recomp.messages_for
    try:ledger.qa(endpoint)
    finally:ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for=old


def summarize(rows):
    summaries,comparisons=[],[]
    for scope in ('overall','multi','single'):
        def by(method):return {r['question_id']:r for r in rows if r['method']==method and (scope=='overall' or r['mode']==scope)}
        for method in METHODS:
            group=list(by(method).values())
            summaries.append(dict(method=method,budget=256,scope=scope,n=len(group),f1_points=100*statistics.mean(r['local_native_f1'] for r in group),
                full_f1_count=sum(r['local_native_f1']==1 for r in group),mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in group),
                statuses=dict(Counter(r['status'] for r in group))))
        for a,b in PAIRS:comparisons.append(dict(method=a,comparator=b,budget=256,scope=scope,**recomp.interface.paired(by(a),by(b))))
    return summaries,comparisons


def endpoint_overlap(entity,alias,reference):
    if not entity:return False
    for answer in [reference['answer'],*reference['aliases']]:
        if not answer.strip():continue
        if any(pattern(name).search(answer) or pattern(answer).search(name) for name in (entity,alias)):return True
    return False


def mechanism_rows(rows,traces,refs):
    by={(r['question_id'],r['method']):r for r in rows};result=[]
    for trace in traces:
        q=trace['question_id'];original,both,a,b=[by[q,m] for m in (FULL,BOTH,A_ONLY,B_ONLY)]
        normal=all(r['status']=='ok' for r in (original,both,a,b));endpoint=endpoint_overlap(trace['entity'],trace['alias'],refs[q])
        lexical=trace['entity'] is not None and trace['entity'].casefold() in {x.casefold() for x in LEXICAL_QC}
        if not trace['eligible']:category='public_ineligible:'+trace['reason']
        elif lexical:category='lexical_perturbation_control'
        elif endpoint:category='endpoint_overlap_flag'
        elif not normal:category='unresolved_failure'
        elif original['local_native_f1']!=1:category='original_not_exact'
        elif both['local_native_f1']!=1:category='consistent_rename_not_exact'
        elif a['local_native_f1']<1 and b['local_native_f1']<1:category='identity_link_dependence_candidate'
        elif a['local_native_f1']<1 or b['local_native_f1']<1:category='one_sided_break_effect'
        else:category='neither_break_hurts'
        result.append(dict(question_id=q,native_id=original['native_id'],mode=original['mode'],public_eligible=trace['eligible'],
            endpoint_overlap_flag=endpoint,lexical_qc_flag=lexical,all_four_normal=normal,category=category,
            original_exact=original['local_native_f1']==1,consistent_exact=both['local_native_f1']==1,
            strict_joint_zero=category=='identity_link_dependence_candidate' and a['local_native_f1']==0 and b['local_native_f1']==0,
            original_answer_f1={m:by[q,m]['local_native_f1'] for m in (FULL,BOTH,A_ONLY,B_ONLY)},
            compatibility_gap=(original['local_native_f1']+both['local_native_f1']-a['local_native_f1']-b['local_native_f1'])/2,
            full_vs_global_delta=original['local_native_f1']-by[q,BASE]['local_native_f1'],
            full_vs_independent_delta=original['local_native_f1']-by[q,INDEPENDENT]['local_native_f1']))
    return result


def mechanism_summary(rows):
    result=[]
    for scope in ('overall','multi','single'):
        group=[r for r in rows if scope=='overall' or r['mode']==scope]
        eligible=[r for r in group if r['public_eligible'] and not r['lexical_qc_flag'] and not r['endpoint_overlap_flag'] and r['all_four_normal']]
        result.append(dict(scope=scope,n=len(group),public_eligible=sum(r['public_eligible'] for r in group),
            lexical_perturbation_controls=sum(r['public_eligible'] and r['lexical_qc_flag'] for r in group),
            eligible_endpoint_clear_all_normal=len(eligible),original_exact_among_eligible=sum(r['original_exact'] for r in eligible),
            original_and_consistent_exact=sum(r['original_exact'] and r['consistent_exact'] for r in eligible),
            categories=dict(Counter(r['category'] for r in group)),strict_joint_zero=sum(r['strict_joint_zero'] for r in group),
            mean_compatibility_gap_eligible=statistics.mean(r['compatibility_gap'] for r in eligible) if eligible else None,
            joint_candidate_among_global_gains=sum(r['category']=='identity_link_dependence_candidate' and r['full_vs_global_delta']>0 for r in group)))
    return result


def score():
    sid=verify();refs=load(PLAN/'private/references.json')
    rows=recomp.interface.parent.score_rows(load(RUN/'qa_plans.json'),list(read_jsonl(RUN/'qa/predictions.jsonl')),refs,sid)
    cache=replay_cache()
    for r in rows:
        if r['method'] not in NEW_METHODS:
            entry=cache[r['request_key']]
            if not r['reused'] or any(r[k]!=entry[k] for k in ('hypothesis','status','error')):raise ValueError('Historical output changed')
    summaries,comparisons=summarize(rows);mechanisms=mechanism_rows(rows,load(PLAN/'public/identity_traces.json'),refs)
    calls=list(read_jsonl(RUN/'qa/answer_calls.jsonl')) if (RUN/'qa/answer_calls.jsonl').exists() else []
    if len(calls)!=load(PLAN/'preparation.json')['new_reader_calls_upper_bound']:raise ValueError('Call accounting mismatch')
    immutable(RUN/'scored.json',rows);immutable(RUN/'mechanism_rows.json',mechanisms)
    immutable(RUN/'report.json',dict(signature=sid,complete=True,scores=summaries,comparisons=comparisons,
        mechanism_summary=mechanism_summary(mechanisms),new_reader_calls=len(calls),reused_conditions=sum(r['reused'] for r in rows),
        reader_cost={k:sum(c['result'].get('usage',{}).get(k,0) for c in calls) for k in ('prompt_tokens','completion_tokens')},
        new_reader_seconds=sum(c['seconds'] for c in calls),config=CONFIG,synthetic_diagnostic_only=True,
        paper_ready=False,independent_confirmation=False,automatic_followup=False))
    print('BRIDGE_IDENTITY_COMPLETE',mechanism_summary(mechanisms),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','verify','preflight','qa','score'])
    p.add_argument('--endpoint',default='http://127.0.0.1:8032');a=p.parse_args()
    if a.mode=='qa':qa(a.endpoint)
    else:
        result=globals()[a.mode]()
        if result:print(result,flush=True)
