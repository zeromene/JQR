"""Stage A: pinned RECOMP granularity controls; no learned relation/path claim."""
import argparse
from collections import Counter
from copy import deepcopy
import gc
import math
from pathlib import Path
import statistics
import time

from . import reader_interface_controls_v1 as interface
from . import replacement_core_v1 as core
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/relation_preserving_pilot_v1'
RUN = ROOT / 'runs/relation_preserving_pilot_v1'
MODEL = ROOT / 'models/RECOMP-NQ-extractive'
MODEL_REV = '46f98c1f57fa919c72d86ced16e3ece3403f9e9e'
NEW_ARMS = ('recomp_sentence', 'recomp_window', 'recomp_paragraph')
OLD_ARMS = ('static_fusion', 'semantic_mixed_batch4', 'raw20_batch4')
ARMS = OLD_ARMS + NEW_ARMS
CONFIG = dict(version='relation-preserving-pilot-v1-stage-a', sources=16, questions=32,
    arms=list(ARMS), evidence_budget=2048, reader='Qwen3.5-9B', output_tokens=2048,
    reader_variant='format_only', decoding='greedy-thinking_false', retries=0,
    compressor='fangyuan/nq_extractive_compressor', compressor_revision=MODEL_REV,
    compressor_precision='FP32, TF32 disabled', compressor_batch=32, compressor_device='cpu',
    pooling='official masked mean; unnormalized dot product; no query prefix',
    rank_input='original sentence only; titles retained in reader context for all new arms',
    order='identical all-sentence ranking; native paragraph/sentence index breaks ties',
    representations=['sentence', 'sentence plus one neighbor on each side', 'full paragraph'],
    packing='same greedy rank scan, exact union of source offsets, source first-acquisition order',
    truncation=False, max_conditions=192, max_new_reader_calls=96,
    scope='previously exposed original16 dev sources; feasibility only',
    stage='ordinary extraction and context restoration controls before relation/path implementation',
    training=False, learned_relation_selector=False, path_search=False,
    independent_confirmation=False, protected_final_data_accessed=False,
    automatic_followup=False)

base, ledger, evidence = interface.base, interface.ledger, interface.parent.evidence
load, immutable = interface.load, interface.immutable


def source_from_public(old):
    """Explicit allow-list; no references, hop counts or decompositions can enter."""
    paragraphs = []
    for raw_id in old['all_raw_ids']:
        r = old['records'][raw_id]
        if r['kind'] != 'raw' or len(r['turns']) != 1:
            raise ValueError('Expected one complete native paragraph')
        t = r['turns'][0]
        spans = core.sentence_spans(t['content'])
        for span in spans:
            while span['start'] < span['end'] and t['content'][span['start']].isspace():
                span['start'] += 1
            while span['end'] > span['start'] and t['content'][span['end']-1].isspace():
                span['end'] -= 1
            if t['content'][span['start']:span['end']] != span['text']:
                raise ValueError('Source offset mismatch')
        pid = t['source_id']
        if [r['source_spans'][pid+':span:'+str(i)] for i in range(len(spans))] != [s['text'] for s in spans]:
            raise ValueError('Sentence segmentation changed from shared source')
        covered = ''.join(s['text'] for s in spans)
        if ''.join(covered.split()) != ''.join(t['content'].split()):
            raise ValueError('Sentence segmentation lost source characters')
        paragraphs.append(dict(source_id=pid, title=t['speaker'], timestamp=t['timestamp'],
            text=t['content'], spans=spans))
    if len(paragraphs) != 20 or len({p['source_id'] for p in paragraphs}) != 20:
        raise ValueError('All20 original paragraphs required')
    return dict(**{k:old[k] for k in ('question_id','native_id','mode','question')}, paragraphs=paragraphs)


def units(source):
    return [(p['source_id']+':span:'+str(i), s['text'])
        for p in source['paragraphs'] for i,s in enumerate(p['spans'])]


def merge_ranges(text, ranges):
    merged = []
    for start,end in sorted(set(map(tuple,ranges))):
        if not 0 <= start < end <= len(text):
            raise ValueError('Invalid source range')
        if merged and (start <= merged[-1][1] or not text[merged[-1][1]:start].strip()):
            merged[-1][1] = max(end,merged[-1][1])
        else:
            merged.append([start,end])
    return merged


def render_ranges(source, selected):
    by = {p['source_id']:p for p in source['paragraphs']}
    records = {}
    for pid,ranges in selected.items():
        p = by[pid]
        ranges = merge_ranges(p['text'],ranges)
        # Only original slices are quoted. Explicit separators mark omitted text.
        text = '\n[...]\n'.join(p['text'][start:end] for start,end in ranges)
        records[pid] = dict(kind='raw', turns=[dict(source_id=pid, timestamp=p['timestamp'],
            speaker=p['title'],content=text)])
    return evidence.render(list(records), records)


def pack(source, scores, arm, counter, budget=2048):
    if arm not in NEW_ARMS or not counter.exact:
        raise ValueError('Unknown representation or inexact counter')
    unit_list = units(source)
    if set(scores) != {k for k,_ in unit_list} or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError('Incomplete or nonfinite all-sentence ranking')
    position = {k:i for i,(k,_) in enumerate(unit_list)}
    by = {p['source_id']:p for p in source['paragraphs']}
    selected, trace = {}, []
    for uid in sorted(scores,key=lambda k:(-scores[k],position[k])):
        pid,_,index = uid.split(':')
        index = int(index); p = by[pid]
        if arm == 'recomp_paragraph':
            addition = [0,len(p['text'])]
        else:
            radius = int(arm == 'recomp_window')
            lo,hi = max(0,index-radius),min(len(p['spans'])-1,index+radius)
            addition = [p['spans'][lo]['start'],p['spans'][hi]['end']]
        proposal = deepcopy(selected)
        proposal[pid] = merge_ranges(p['text'],proposal.get(pid,[])+[addition])
        if proposal == selected:
            continue
        text = render_ranges(source,proposal)
        n = counter.count(text)
        accepted = n <= budget
        trace.append(dict(seed=uid, score=scores[uid], addition=addition,
            proposed_tokens=n, accepted=accepted))
        if accepted:
            selected = proposal
    text = render_ranges(source,selected)
    if not text.strip() or counter.count(text)>budget:
        raise ValueError('Empty or over-budget selection; never truncate')
    kept = [uid for uid,_ in unit_list if any(a<=by[uid.split(':')[0]]['spans'][int(uid.split(':')[2])]['start']
        and b>=by[uid.split(':')[0]]['spans'][int(uid.split(':')[2])]['end']
        for a,b in selected.get(uid.split(':')[0],[]))]
    full = [pid for pid,ranges in selected.items() if
        not ''.join(by[pid]['text'][:ranges[0][0]].split()) and
        not ''.join(by[pid]['text'][ranges[-1][1]:].split()) and len(ranges)==1]
    plan = dict(**{k:source[k] for k in ('question_id','native_id','mode','question')},
        arm=arm,reader_variant='format_only',selected_ids=['quote:'+k for k in selected],
        selected_ranges=selected,selected_spans=kept,fully_preserved_sources=full,
        evidence_text=text,evidence_tokens=counter.count(text),evidence_sha256=fingerprint(text),
        private_inputs_read=False,diagnostic_only=False)
    return plan, trace


def messages_for(row):
    return interface.messages_for(dict(question=row['question'],evidence_text=row['evidence_text'],
        reader_variant='format_only'))


def model_files():
    manifest = load(MODEL/'artifact_manifest.json')
    if manifest['repository'] != CONFIG['compressor'] or manifest['revision'] != MODEL_REV or not manifest['upstream_hashes_verified']:
        raise ValueError('Unverified compressor provenance')
    for f in manifest['files']:
        p = MODEL/f['name']
        if p.stat().st_size != f['size'] or base.sha(p) != f['sha256']:
            raise ValueError('Compressor artifact mismatch: '+f['name'])
    return manifest


def freeze():
    inherited = load(interface.PLAN/'manifest.json')
    paths = [n for n in inherited['files'] if n.endswith('.py')]
    paths += ['reader_memory/relation_preserving_pilot_v1.py',
        'tools/prepare_relation_preserving_pilot_v1.py','tools/stage_recomp_controls_v1.py',
        'scripts/relation_preserving_pilot_v1_queue.py','tests/test_relation_preserving_pilot_v1.py',
        'models/RECOMP-NQ-extractive/artifact_manifest.json']
    paths += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*')
        if p.is_file() and p.name not in {'manifest.json','CURRENT_STATUS.md','submission_receipt.json'}
        and not p.name.endswith('.pyc')]
    value = dict(config=CONFIG,parent_signature=inherited['signature'],
        files={n:base.sha(ROOT/n) for n in sorted(set(paths))})
    model_files()
    value['signature'] = fingerprint(value)
    immutable(PLAN/'manifest.json',value)
    print('FROZEN_RELATION_PILOT',value['signature'],flush=True)


def verify():
    m = load(PLAN/'manifest.json')
    if m['config']!=CONFIG or m['signature']!=fingerprint({k:v for k,v in m.items() if k!='signature'}):
        raise ValueError('Scientific identity changed')
    for n,digest in m['files'].items():
        if base.sha(ROOT/n)!=digest:
            raise ValueError('Frozen input changed: '+n)
    return m['signature']


def replay_cache():
    marker = load(PLAN/'model_references.json')['reader_marker_sha256']
    cache = {}
    for c in load(PLAN/'private/replay_cache.json'):
        interface.parent.check_cache_entry(c,marker)
        if c['key'] in cache:
            raise ValueError('Duplicate historical request')
        cache[c['key']] = c['entry']
    return cache


def preflight():
    sid = verify(); model_files()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter = ExactCounter()
    marker = load(PLAN/'model_references.json')['reader_marker_sha256']
    if base.sha(ROOT/'models/Qwen3.5-9B/.verified.json') != marker:
        raise ValueError('Reader marker mismatch')
    sources = load(PLAN/'public/source_inputs.json')
    if len(sources)!=32 or len({s['native_id'] for s in sources})!=16:
        raise ValueError('Wrong development matrix')
    tok = AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    maximum,total = 0,0
    for s in sources:
        if set(s)!={'question_id','native_id','mode','question','paragraphs'} or len(s['paragraphs'])!=20:
            raise ValueError('Private input fields or candidate loss')
        lengths = [len(tok.encode(x,add_special_tokens=True,truncation=False)) for x in [s['question']]+[t for _,t in units(s)]]
        maximum=max(maximum,max(lengths));total+=len(lengths)-1
        if max(lengths)>512:
            raise ValueError('RECOMP encoder input over512; no silent truncation')
        for arm in NEW_ARMS:
            pack(s,{k:float(-i) for i,(k,_) in enumerate(units(s))},arm,counter)
    cache = replay_cache()
    old = load(PLAN/'public/baseline_plans.json')
    if len(old)!=96 or {(p['question_id'],p['arm']) for p in old}!={(s['question_id'],a) for s in sources for a in OLD_ARMS}:
        raise ValueError('Incomplete historical controls')
    for p in old:
        if counter.count(p['evidence_text'])!=p['evidence_tokens'] or p['evidence_tokens']>2048 or fingerprint(p['evidence_text'])!=p['evidence_sha256']:
            raise ValueError('Historical evidence changed')
        if interface.parent.request_key(messages_for(p),marker) not in cache:
            raise ValueError('Historical failure/success must be reused')
    immutable(RUN/'preflight.json',dict(signature=sid,passed=True,questions=32,sources=16,
        paragraphs_per_question=20,total_sentence_candidates=total,max_encoder_tokens=maximum,
        cached_baseline_conditions=96,protected_final_data_accessed=False))
    print('RELATION_PILOT_PREFLIGHT_OK',total,'sentences',maximum,'max encoder tokens',flush=True)


def mean_pooling(torch, hidden, mask):
    # carriex/recomp run_extractive_compressor.py, MIT; pinned source shipped in sources/.
    hidden = hidden.masked_fill(~mask[...,None].bool(),0.)
    return hidden.sum(dim=1)/mask.sum(dim=1)[...,None]


def select():
    sid=verify();model_files()
    import torch
    from transformers import AutoModel,AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    counter=ExactCounter()
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float32,attn_implementation='eager').to('cpu').eval()
    for i,s in enumerate(load(PLAN/'public/source_inputs.json')):
        dest=RUN/'selected'/(s['question_id'].replace(':','_')+'.json')
        if dest.exists():
            saved=load(dest)
            if saved['signature']!=sid or saved['source_sha256']!=fingerprint(s):
                raise ValueError('Committed selection changed')
            continue
        intent=RUN/'selection_inflight.json'
        if intent.exists():
            raise ValueError('Unknown selection commit; no automatic retry')
        write_json(intent,dict(signature=sid,question_id=s['question_id']))
        texts=[s['question']]+[t for _,t in units(s)]
        encoded=[tok.encode(t,add_special_tokens=True,truncation=False) for t in texts]
        if max(map(len,encoded))>512:
            raise ValueError('No encoder truncation permitted')
        started=time.perf_counter();chunks=[]
        with torch.inference_mode():
            for start in range(0,len(encoded),32):
                batch=tok.pad(dict(input_ids=encoded[start:start+32]),padding=True,return_tensors='pt')
                hidden=model(**batch).last_hidden_state
                chunks.append(mean_pooling(torch,hidden,batch['attention_mask']).cpu())
        em=torch.cat(chunks)
        # Same CPU dot product as author code. No normalization or calibration.
        values=[float(em[0]@em[j]) for j in range(1,len(texts))]
        scores=dict(zip([k for k,_ in units(s)],values,strict=True))
        plans,traces=[],{}
        for arm in NEW_ARMS:
            plan,trace=pack(s,scores,arm,counter);plans.append(plan);traces[arm]=trace
        value=dict(signature=sid,source_sha256=fingerprint(s),scores=scores,plans=plans,traces=traces,
            costs=dict(forward_calls=math.ceil(len(encoded)/32),encoder_items=len(encoded),
                input_tokens=sum(map(len,encoded)),seconds=time.perf_counter()-started),
            shared_scoring=True,private_inputs_read=False)
        immutable(dest,value);intent.unlink()
        write_json(RUN/'selection_status.json',dict(completed=i+1,expected=32,question_id=s['question_id']))
        print('RECOMP_SELECT',i+1,'/32',s['question_id'],len(scores),'sentences',flush=True)
    del model;gc.collect()


def assemble():
    sid=verify()
    from tools.prepare_replacement_learning_v1 import ExactCounter
    from transformers import AutoTokenizer
    counter=ExactCounter()
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    plans=load(PLAN/'public/baseline_plans.json')
    for s in load(PLAN/'public/source_inputs.json'):
        saved=load(RUN/'selected'/(s['question_id'].replace(':','_')+'.json'))
        if saved['signature']!=sid or saved['source_sha256']!=fingerprint(s):
            raise ValueError('Wrong selection provenance')
        plans+=saved['plans']
    positions={s['question_id']:i for i,s in enumerate(load(PLAN/'public/source_inputs.json'))}
    plans.sort(key=lambda p:(positions[p['question_id']],ARMS.index(p['arm'])))
    if len(plans)!=192 or {(p['question_id'],p['arm']) for p in plans}!={(q,a) for q in positions for a in ARMS}:
        raise ValueError('Incomplete condition matrix')
    keys=set();marker=load(PLAN/'model_references.json')['reader_marker_sha256']
    for p in plans:
        if counter.count(p['evidence_text'])!=p['evidence_tokens'] or p['evidence_tokens']>2048 or fingerprint(p['evidence_text'])!=p['evidence_sha256']:
            raise ValueError('Evidence identity changed')
        messages=messages_for(p)
        full=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if counter.count(full)+2048>32768:
            raise ValueError('Reader prompt overflow; never truncate')
        keys.add(interface.parent.request_key(messages,marker))
    new=len(keys-set(replay_cache()))
    if new>96:
        raise ValueError('Call budget exceeded')
    immutable(RUN/'qa_plans.json',plans)
    immutable(RUN/'selection_complete.json',dict(signature=sid,plans_sha256=base.sha(RUN/'qa_plans.json'),
        conditions=len(plans),unique_requests=len(keys),uncached_requests=new))
    print('ASSEMBLED_RELATION_PILOT',len(plans),'conditions',new,'new calls',flush=True)


def qa(endpoint):
    sid=verify()
    old=ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for
    ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for=RUN,lambda:sid,replay_cache,messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for=old


def score():
    sid=verify()
    plans=load(RUN/'qa_plans.json');refs=load(PLAN/'private/references.json')
    scored=interface.parent.score_rows(plans,list(read_jsonl(RUN/'qa/predictions.jsonl')),refs,sid)
    historical={(r['question_id'],r['arm']):r for r in load(PLAN/'private/baseline_outcomes.json')}
    for r in scored:
        if r['arm'] in OLD_ARMS:
            old=historical[r['question_id'],r['arm']]
            if not r['reused'] or any(old[k]!=r[k] for k in ('hypothesis','status','local_native_f1')):
                raise ValueError('Historical outcome changed')
    summaries=[]
    for arm in ARMS:
        for scope in ('overall','multi','single'):
            group=[r for r in scored if r['arm']==arm and (scope=='overall' or r['mode']==scope)]
            cover=[set(map(str,refs[r['question_id']]['support_indices'])) <= {u.split(':')[0] for u in r['selected_spans']} for r in group]
            full=[set(map(str,refs[r['question_id']]['support_indices'])) <= set(r['fully_preserved_sources']) for r in group]
            summaries.append(dict(arm=arm,scope=scope,n=len(group),
                f1_points=100*statistics.mean(r['local_native_f1'] for r in group),
                mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in group),
                statuses=dict(Counter(r['status'] for r in group)),
                all_support_sources_touched=sum(cover),all_support_sources_full=sum(full),
                coverage_is_source_proxy_not_relation_sufficiency=True))
    comparisons=[]
    pairs=[(a,b) for a in NEW_ARMS for b in ('raw20_batch4','semantic_mixed_batch4','static_fusion')]
    pairs += [('recomp_window','recomp_sentence'),('recomp_paragraph','recomp_sentence'),('recomp_window','recomp_paragraph')]
    for a,b in pairs:
        for scope in ('overall','multi','single'):
            by=lambda arm:{r['question_id']:r for r in scored if r['arm']==arm and (scope=='overall' or r['mode']==scope)}
            comparisons.append(dict(arm=a,comparator=b,scope=scope,**interface.paired(by(a),by(b))))
    calls=list(read_jsonl(RUN/'qa/answer_calls.jsonl')) if (RUN/'qa/answer_calls.jsonl').exists() else []
    if len(calls)>96:
        raise ValueError('Reader call budget exceeded')
    selections=[load(p) for p in (RUN/'selected').glob('*.json')]
    report=dict(signature=sid,complete=True,scores=summaries,comparisons=comparisons,
        new_reader_calls=len(calls),reused_conditions=sum(r['reused'] for r in scored),
        shared_encoder_costs={k:sum(s['costs'][k] for s in selections) for k in ('forward_calls','input_tokens','seconds')},
        standalone_encoder_cost_not_zero=True,
        reader_cost={k:sum(c['result'].get('usage',{}).get(k,0) for c in calls) for k in ('prompt_tokens','completion_tokens')},
        historical_control_costs_not_zero=True,config=CONFIG,paper_ready=False,
        relation_selector_implemented=False,path_search_implemented=False,automatic_followup=False)
    immutable(RUN/'scored.json',scored);immutable(RUN/'report.json',report)
    print('RELATION_PILOT_RESULTS',summaries,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['freeze','verify','preflight','select','assemble','qa','score'])
    parser.add_argument('--endpoint',default='http://127.0.0.1:8021')
    args=parser.parse_args()
    if args.mode=='qa':qa(args.endpoint)
    else:
        value=globals()[args.mode]()
        if value:print(value,flush=True)
