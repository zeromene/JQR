"""Bounded first-response manipulation check; no QA, training or reference values."""
from collections import Counter
import json
import time
from . import planning_mechanism_v1 as previous
parent=previous.parent
ROOT=parent.ROOT;BASE=ROOT/'research/planning_behavior_v2';PLAN=BASE;RUN=ROOT/'runs/planning_behavior_v2'
load,sha,fp,immutable,write_json=parent.load,parent.sha,parent.fp,parent.immutable,parent.write_json
MODES=('direct','rationale','plan')
CONFIG=dict(version='planning-behavior-v2',sources=24,early_sources=8,modes=list(MODES),seed=30,
    max_calls=72,generation_cap=2048,max_input=32768,repairs=0,qa=False,training=False,
    automatic_expansion=False,resource_caps_mib=dict(propose9=20500),resource_headroom_mib=dict(propose9=500))
MODE_RULES={
 'direct':'OUTPUT MODE: DIRECT. The need field is a reserved empty placeholder. Every need value MUST be exactly "" (zero characters). Do not put a summary, explanation, subquestion, or answer in need. You may reason about the required relations internally; output only the source evidence with empty need values.',
 'rationale':'OUTPUT MODE: RELEVANCE. Every need value MUST be a short nonempty declarative explanation of why the associated evidence is relevant to the original question. Do not write relation questions or question marks. Use at most 24 words per need.',
 'plan':'OUTPUT MODE: RELATION QUESTIONS. Every need value MUST be a short nonempty relation question naming a fact required to answer the original question. Each need must end with a question mark. Include the final requested fact and necessary intermediate links only. Use at most 24 words per need.'}


def shared_prompt():
    blocks=parent.role_prompt.PROMPT.split('\n')
    assert len(blocks)==6
    schema='Return one JSON object with exactly the key "roles", a list of at most 6 objects. Every role has exactly three keys: "need" (a string of 0 to 500 characters, whose content MUST follow OUTPUT MODE), "parents" (always the empty list [] in this experiment), and "evidence" (a list of at most 2 source quotes). A quote object has exactly "source" (the supplied source id as a string) and "quote" (one contiguous verbatim substring of that source text). The need field may be empty as required by OUTPUT MODE. Do not infer its format from its field name. No Markdown or surrounding explanation.'
    # The old task, question restrictions and extraction instructions remain verbatim.
    return '\n'.join([blocks[0],blocks[1],schema,blocks[3],blocks[4],blocks[5]])


def messages(s,mode):
    assert mode in MODES
    return [dict(role='system',content=MODE_RULES[mode]+'\n'+shared_prompt()),
        dict(role='user',content=json.dumps(parent.execution_payload(s),ensure_ascii=False,separators=(',',':'))+
            '\nEND OF SOURCE DATA. Output reminder: '+MODE_RULES[mode])]


def verify():
    m=load(BASE/'manifest.json')
    assert m['config']==CONFIG and m['signature']==fp({k:v for k,v in m.items() if k!='signature'})
    assert m['parent_signature']==previous.verify()
    for n,h in m['files'].items():assert sha(ROOT/n)==h,n
    return m['signature']


def sources():
    ss=load(BASE/'public/sources.json');assert len(ss)==len({s['question_id'] for s in ss})==24
    return [parent.execution_source(s) for s in ss]


def inspect(s,result):
    """One parser for all arms; note compliance is reported separately, never repaired."""
    bad=dict(schema_ok=False,notes=[],quote_count=0,matched_quotes=0,quote_errors=[],usable=False,roles=[],candidates=[])
    if result['finish_reason']!='stop':return dict(bad,error='output_limit')
    try:
        obj=json.loads(result['text'])
        if not isinstance(obj,dict) or set(obj)!={'roles'} or not isinstance(obj['roles'],list) or len(obj['roles'])>6:raise ValueError('roles_schema')
        by={p['source_id']:p for p in s['paragraphs']};notes=[];errors=[];cs=[];count=matched=0
        for i,r in enumerate(obj['roles']):
            if not isinstance(r,dict) or set(r)!={'need','parents','evidence'}:raise ValueError('role_fields')
            if not isinstance(r['need'],str) or len(r['need'])>500 or r['parents']!=[]:raise ValueError('need_or_parents')
            if not isinstance(r['evidence'],list) or len(r['evidence'])>2:raise ValueError('quote_count')
            notes.append(r['need'])
            for j,e in enumerate(r['evidence']):
                if not isinstance(e,dict) or set(e)!={'source','quote'} or not isinstance(e['source'],str) or not isinstance(e['quote'],str):raise ValueError('quote_fields')
                count+=1
                try:
                    if e['source'] not in by:raise ValueError('source_reference')
                    p=by[e['source']];a,z=parent.role.locate_quote(p['text'],e['quote']);matched+=1
                    cs.append(dict(source=e['source'],start=a,end=z,text=p['text'][a:z],role=i))
                except ValueError as exc:errors.append(dict(role=i,quote=j,error=str(exc)))
        return dict(schema_ok=True,notes=notes,quote_count=count,matched_quotes=matched,quote_errors=errors,
            usable=bool(matched),roles=obj['roles'],candidates=cs,error=None)
    except (TypeError,ValueError,KeyError) as exc:return dict(bad,error=str(exc))


def style_ok(parsed,mode):
    if not parsed['schema_ok'] or not parsed['usable'] or not parsed['notes']:return False
    ns=parsed['notes']
    if mode=='direct':return all(n=='' for n in ns)
    if mode=='rationale':return all(n.strip() and '?' not in n and len(n.split())<=24 for n in ns)
    return all(n.strip().endswith('?') and len(n.split())<=24 for n in ns)


def groups(rows):
    result={}
    for m in MODES:
        rr=[r for r in rows if r['mode']==m];assert rr
        count=sum(r['inspection']['quote_count'] for r in rr);matched=sum(r['inspection']['matched_quotes'] for r in rr)
        result[m]=dict(sources=len(rr),schema_ok=sum(r['inspection']['schema_ok'] for r in rr),
            usable=sum(r['inspection']['usable'] for r in rr),style_ok=sum(style_ok(r['inspection'],m) for r in rr),
            notes=sum(len(r['inspection']['notes']) for r in rr),nonempty_notes=sum(bool(n.strip()) for r in rr for n in r['inspection']['notes']),
            quote_count=count,matched_quotes=matched,quote_match_fraction=matched/count if count else 0,
            output_limit_sources=sum(r['result']['finish_reason']!='stop' for r in rr),
            total_output_tokens=sum(r['result']['usage']['completion_tokens'] for r in rr),
            total_seconds=sum(r['result']['usage']['elapsed_seconds'] for r in rr))
    return result


def technical_decision(rows,early=False):
    g=groups(rows);n=8 if early else 24;minimum=7 if early else 23
    assert len(rows)==n*3 and all(v['sources']==n for v in g.values())
    criteria={m:dict(schema=v['schema_ok']>=minimum,usable=v['usable']>=minimum,style=v['style_ok']>=minimum,
        quote_matching=True if early else v['quote_match_fraction']>=.9) for m,v in g.items()}
    passed=all(all(v.values()) for v in criteria.values())
    return dict(groups=g,criteria=criteria,passed=passed,decision=('continue_remaining_behavior_cases' if early else 'requires_semantic_review') if passed else 'stop_behavior_protocol',
        publication_ready=False,mechanism_confirmed=False,effectiveness_tested=False,automatic_expansion=False)


def preflight():
    from transformers import AutoTokenizer
    sid=verify()
    for n,h in load(BASE/'manifest.json')['model_files'].items():assert sha(ROOT/n)==h,n
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    lengths=[]
    for s in sources():
        for m in MODES:
            text=tok.apply_chat_template(messages(s,m),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert text.count(MODE_RULES[m])==2
            lengths.append(len(tok.encode(text,add_special_tokens=False)))
    assert len(lengths)==72 and max(lengths)<=CONFIG['max_input']
    immutable(RUN/'preflight.json',dict(signature=sid,prompts=72,max_prompt_tokens=max(lengths),mode_instructions_present=True,reference_values_read=False))


def propose():
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid=verify();torch_setup();backend=TransformersBackend(ROOT/'models/Qwen3.5-9B',max_input_tokens=CONFIG['max_input']);rows=[]
    for i,s in enumerate(sources(),1):
        q=s['question_id']
        for mode in sorted(MODES,key=lambda m:fp([30,q,m])):
            key=fp([q,mode]);path=RUN/'proposals'/(key+'.json');start=RUN/'requests'/(key+'.started.json')
            assert not path.exists() and not start.exists(),'Unknown request retry forbidden'
            msg=messages(s,mode);identity=dict(signature=sid,question_id=q,source_hash=fp(s),mode=mode,max_output=2048,prompt_hash=fp(msg))
            immutable(start,dict(identity,time=time.time()))
            try:result=generate(backend,msg,2048)
            except Exception as exc:
                immutable(RUN/'failures'/(key+'.json'),dict(identity,error=repr(exc)));raise
            row=dict(identity,result=result,inspection=inspect(s,result));immutable(path,row);rows.append(row)
            print('PROPOSE',len(rows),'/72',mode,'style_ok',style_ok(row['inspection'],mode),flush=True)
        if i==8:
            gate=technical_decision(rows,early=True);immutable(RUN/'early_gate.json',dict(signature=sid,**gate))
            if not gate['passed']:break
    immutable(RUN/'generation_complete.json',dict(signature=sid,calls=len(rows),early_stopped=len(rows)==24,
        expected_questions=[s['question_id'] for s in sources()[:len(rows)//3]],reference_values_read=False))


def summarize():
    sid=verify();done=load(RUN/'generation_complete.json');assert done['signature']==sid
    qids=done['expected_questions'];assert qids==[s['question_id'] for s in sources()[:len(qids)]] and len(qids) in (8,24)
    rows=[load(RUN/'proposals'/(fp([q,m])+'.json')) for q in qids for m in MODES]
    assert len(rows)==done['calls'] and all(r['signature']==sid for r in rows)
    gate=technical_decision(rows,early=done['early_stopped'])
    immutable(RUN/'summary.json',dict(signature=sid,complete=True,actual_calls=len(rows),sources=len(qids),
        early_stopped=done['early_stopped'],reference_values_read=False,**gate))
    print('BEHAVIOR_COMPLETE',gate['decision'],flush=True)


def phase(name):
    if name=='propose9':propose()
    else:globals()[name]()
