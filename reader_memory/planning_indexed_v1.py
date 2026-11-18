"""Shared sentence identifiers for three planning modes, with frozen granularity controls."""
from copy import deepcopy
import json
import math
import statistics
import time
from . import planning_effectiveness_supplement_v1 as previous

original = previous.original
behavior = original.behavior
parent = original.parent
ROOT = original.ROOT
BASE = ROOT / 'research/planning_indexed_v1'
INPUT_BASE = original.BASE
PLAN = BASE
RUN = ROOT / 'runs/planning_indexed_v1'
load, sha, fp, immutable, write_json = original.load, original.sha, original.fp, original.immutable, original.write_json
READERS, BUDGETS, MODES = original.READERS, original.BUDGETS, original.MODES
OLD_ARMS = original.OLD_ARMS
ARMS = MODES + tuple('quote_' + m for m in MODES) + tuple('quote_sentence_' + m for m in MODES) + OLD_ARMS
PRIMARY = (('plan', 'direct'), ('plan', 'rationale'))
PAIRS = PRIMARY + (('rationale', 'direct'), ('plan', 'role_common'), ('plan', 'original'), ('plan', 'budget_plain')) + tuple(
    pair for m in MODES for pair in [(m, 'quote_' + m), (m, 'quote_sentence_' + m), ('quote_sentence_' + m, 'quote_' + m)])
sources, pack, paired = original.sources, original.pack, original.paired
CONFIG = dict(version='planning-indexed-v1', sources=72, modes=list(MODES), arms=list(ARMS),
    readers=list(READERS), budgets=list(BUDGETS), max_proposer_calls=216, max_reader_conditions=3456,
    seed=30, bootstrap_replicates=10000, generation_cap=2048, proposer_max_input=32768,
    reader_max_output=128, reader_max_input=4096, repairs=0, training=False, automatic_expansion=False,
    whole_run_behavior_abort=False, primary_minimum=.02, reference_match_target=.98,
    resource_caps_mib=dict(propose9=20500, qa4=10500, qa9=19500),
    resource_headroom_mib=dict(propose9=500, qa4=1000, qa9=1000))


def verify():
    m = load(BASE / 'manifest.json')
    assert m['config'] == CONFIG and m['signature'] == fp({k: v for k, v in m.items() if k != 'signature'})
    assert m['parent_signature'] == previous.verify()
    for name, digest in m['files'].items():
        assert sha(ROOT / name) == digest, name
    return m['signature']


def catalog(s):
    result = {}
    for p in s['paragraphs']:
        for i, span in enumerate(p['spans']):
            assert p['text'][span['start']:span['end']] == span['text']
            key = p['source_id'] + ':span:' + str(i)
            assert key not in result
            result[key] = dict(source=p['source_id'], start=span['start'], end=span['end'], text=span['text'])
    return result


def payload(s):
    # Retain every native sentence and title; IDs come only from frozen source coordinates.
    parent.execution_payload(s)
    return dict(question=s['question'], sources=[dict(id=p['source_id'], title=p['title'],
        sentences=[dict(id=p['source_id'] + ':span:' + str(i), text=span['text']) for i, span in enumerate(p['spans'])])
        for p in s['paragraphs']])


def shared_prompt():
    blocks = parent.role_prompt.PROMPT.split('\n')
    assert len(blocks) == 6
    schema = 'Return one JSON object with exactly the key "roles", a list of at most 6 objects. Every role has exactly three keys: "need" (a string of 0 to 500 characters, whose content MUST follow OUTPUT MODE), "parents" (always the empty list [] in this experiment), and "evidence" (a list of at most 2 sentence-id strings copied exactly from the supplied SOURCES). The need field may be empty as required by OUTPUT MODE. Do not infer its format from its field name. No Markdown or surrounding explanation.'
    selection = 'For each relation select the shortest supplied original sentence that establishes it. Preserve the subject, predicate, object and any relevant date, direction, negation or condition. Keep complete personal names: initials and surnames are part of the identity. If a sentence uses a pronoun or incomplete name, include its antecedent in one additional sentence. Titles will be visible to the reader. A table value needs its field label and the correct entity. Do not confuse a workplace with a birthplace, a collaborator with an author, an event date with a person\'s death, or nearby countries and historical periods. A sentence must establish the relation, not merely contain related words.'
    identifiers = 'Use one best sentence id per role, or at most two when an antecedent or an alternative source is needed. At most 12 identifiers total. Copy identifiers exactly, including their source and span index. Never output source text, invented identifiers, or just a source id in evidence. The program will retrieve the original text. If a required relation is not established in the sources, keep its role with evidence:[]; never silently drop the relation or replace it with an easier one. If no useful evidence is identifiable, return {"roles":[]}.'
    return '\n'.join([blocks[0], blocks[1], schema, selection, identifiers, blocks[5]])


def messages(s, mode):
    assert mode in MODES
    return [dict(role='system', content=behavior.MODE_RULES[mode] + '\n' + shared_prompt()),
        dict(role='user', content=json.dumps(payload(s), ensure_ascii=False, separators=(',', ':')) +
             '\nEND OF SOURCE DATA. Output reminder: ' + behavior.MODE_RULES[mode])]


def inspect(s, result):
    bad = dict(schema_ok=False, notes=[], roles=[], reference_count=0, matched_references=0,
        reference_errors=[], usable=False, candidates=[])
    if result['finish_reason'] != 'stop':
        return dict(bad, error='output_limit')
    try:
        obj = json.loads(result['text'])
        if not isinstance(obj, dict) or set(obj) != {'roles'} or not isinstance(obj['roles'], list) or len(obj['roles']) > 6:
            raise ValueError('roles_schema')
        by_id = catalog(s)
        notes, errors, candidates = [], [], []
        count = matched = 0
        for i, role in enumerate(obj['roles']):
            if not isinstance(role, dict) or set(role) != {'need', 'parents', 'evidence'}:
                raise ValueError('role_fields')
            if not isinstance(role['need'], str) or len(role['need']) > 500 or role['parents'] != []:
                raise ValueError('need_or_parents')
            if not isinstance(role['evidence'], list) or len(role['evidence']) > 2:
                raise ValueError('reference_count')
            notes.append(role['need'])
            for j, identifier in enumerate(role['evidence']):
                if not isinstance(identifier, str):
                    raise ValueError('reference_type')
                count += 1
                if identifier not in by_id:
                    errors.append(dict(role=i, reference=j, identifier=identifier, error='unknown_sentence_identifier'))
                    continue
                matched += 1
                candidates.append(dict(by_id[identifier], role=i))
        return dict(schema_ok=True, notes=notes, roles=obj['roles'], reference_count=count, matched_references=matched,
            reference_errors=errors, usable=bool(candidates), candidates=candidates, error=None)
    except (TypeError, ValueError, KeyError) as exc:
        return dict(bad, error=str(exc))


def style_ok(parsed, mode):
    # Reference usability is deliberately separate from the mode manipulation check.
    notes = parsed['notes']
    if not parsed['schema_ok'] or not notes:
        return False
    if mode == 'direct':
        return all(n == '' for n in notes)
    if mode == 'rationale':
        return all(n.strip() and '?' not in n and len(n.split()) <= 24 for n in notes)
    return all(n.strip().endswith('?') and len(n.split()) <= 24 for n in notes)


def technical_report(rows, qids):
    assert len(qids) == len(set(qids)) == 72
    assert len(rows) == 216 and {(r['question_id'], r['mode']) for r in rows} == {(q, m) for q in qids for m in MODES}
    groups = {}
    for mode in MODES:
        rr = [r for r in rows if r['mode'] == mode]
        pp = [r['inspection'] for r in rr]
        count = sum(p['reference_count'] for p in pp)
        matched = sum(p['matched_references'] for p in pp)
        lost_last = sum(bool(p['roles']) and bool(p['roles'][-1]['evidence']) and
            not any(c['role'] == len(p['roles']) - 1 for c in p['candidates']) for p in pp)
        groups[mode] = dict(sources=72, schema_ok=sum(p['schema_ok'] for p in pp), usable=sum(p['usable'] for p in pp),
            style_ok=sum(style_ok(p, mode) for p in pp), reference_count=count, matched_references=matched,
            reference_match_fraction=matched / count if count else 0, last_role_references_all_rejected=lost_last,
            empty_last_role_sources=sum(bool(p['roles']) and not p['roles'][-1]['evidence'] for p in pp),
            output_limit_sources=sum(r['result']['finish_reason'] != 'stop' for r in rr),
            total_output_tokens=sum(r['result']['usage']['completion_tokens'] for r in rr),
            total_seconds=sum(r['result']['usage']['elapsed_seconds'] for r in rr))
    reliable = all(g['schema_ok'] >= 69 and g['style_ok'] >= 69 and g['reference_match_fraction'] >= .98 for g in groups.values())
    return dict(groups=groups, behavior_reliable=reliable, abort_qa=False, actual_calls=216, reference_values_read=False,
        costs={k: sum(r['result']['usage'].get(k, 0) for r in rows) for k in ('prompt_tokens', 'completion_tokens', 'elapsed_seconds')})


def expand_old_quotes(s, parsed):
    expanded = deepcopy(parsed)
    expanded['candidates'] = []
    for c in parsed['candidates']:
        for sentence in catalog(s).values():
            if sentence['source'] == c['source'] and sentence['start'] < c['end'] and sentence['end'] > c['start']:
                expanded['candidates'].append(dict(sentence, role=c['role']))
    return expanded


def validate_plan_matrix(plans, qids):
    assert len(qids) == len(set(qids)) == 72
    expected = {(q, b, a) for q in qids for b in BUDGETS for a in ARMS}
    assert len(plans) == len(expected) == 1728
    assert {(p['question_id'], p['budget'], p['arm']) for p in plans} == expected
    assert all(0 <= p['evidence_tokens'] <= p['budget'] for p in plans)


def preflight():
    from transformers import AutoTokenizer
    sid = verify()
    for name, digest in load(BASE / 'manifest.json')['model_files'].items():
        assert sha(ROOT / name) == digest, name
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/Qwen3.5-9B'), local_files_only=True, trust_remote_code=False)
    lengths = []
    for s in sources():
        for mode in MODES:
            text = tok.apply_chat_template(messages(s, mode), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert text.count(behavior.MODE_RULES[mode]) == 2
            lengths.append(len(tok.encode(text, add_special_tokens=False)))
    assert len(lengths) == 216 and max(lengths) <= 32768
    fixed = load(BASE / 'public/fixed_plans.json')
    assert len(fixed) == 1296
    counter = parent.CounterTokens()
    for p in fixed:
        assert counter.count(p['evidence_text']) == p['evidence_tokens'] <= p['budget']
    immutable(RUN / 'preflight.json', dict(signature=sid, prompts=216, maximum_prompt=max(lengths),
        source_catalog_hashes={s['question_id']: fp(catalog(s)) for s in sources()}, reference_values_read=False))


def propose():
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid = verify()
    torch_setup()
    backend = TransformersBackend(ROOT / 'models/Qwen3.5-9B', max_input_tokens=32768)
    rows = []
    ss = sources()
    for s in ss:
        q = s['question_id']
        for mode in sorted(MODES, key=lambda m: fp([30, q, m])):
            key = fp([q, mode])
            path = RUN / 'proposals' / (key + '.json')
            start = RUN / 'requests' / (key + '.started.json')
            assert not path.exists() and not start.exists(), 'Unknown request retry forbidden'
            msg = messages(s, mode)
            identity = dict(signature=sid, question_id=q, source_hash=fp(s), mode=mode, max_output=2048,
                prompt_hash=fp(msg), catalog_hash=fp(catalog(s)))
            immutable(start, dict(identity, time=time.time()))
            try:
                result = generate(backend, msg, 2048)
            except Exception as exc:
                immutable(RUN / 'failures' / (key + '.json'), dict(identity, error=repr(exc)))
                raise
            row = dict(identity, result=result, inspection=inspect(s, result))
            immutable(path, row)
            rows.append(row)
            print('PROPOSE', len(rows), '/216', mode, 'usable', row['inspection']['usable'], flush=True)
    immutable(RUN / 'proposal_report.json', dict(signature=sid, **technical_report(rows, [s['question_id'] for s in ss])))


def assemble():
    sid = verify()
    counter = parent.CounterTokens()
    cached = load(INPUT_BASE / 'public/inherited_inputs.json')
    plans = deepcopy(load(BASE / 'public/fixed_plans.json'))
    for s in sources():
        q = s['question_id']
        assert cached[q]['source_hash'] == fp(s)
        for mode in MODES:
            p = load(RUN / 'proposals' / (fp([q, mode]) + '.json'))
            assert p['signature'] == sid and p['source_hash'] == fp(s)
            for b in BUDGETS:
                plans.append(dict(pack(s, cached[q], p['inspection'], b, counter), arm=mode))
    validate_plan_matrix(plans, [s['question_id'] for s in sources()])
    immutable(RUN / 'plans.json', dict(signature=sid, selection_before_qa=True, plans=plans))


def decide(comparisons, per_reader, per_budget, report):
    values = {(v['arm'], v['control']): v['mean_delta_f1'] for v in comparisons}
    criteria = dict(two_points_over_direct=values['plan', 'direct'] >= .02,
        two_points_over_rationale=values['plan', 'rationale'] >= .02,
        each_reader_nonnegative=all(v['mean_delta_f1'] >= 0 for group in per_reader.values() for v in group.values()),
        each_budget_nonnegative=all(v['mean_delta_f1'] >= 0 for group in per_budget.values() for v in group.values()),
        close_to_existing_role=values['plan', 'role_common'] >= -.02)
    ratio = report['groups']['plan']['total_output_tokens'] / max(1, report['groups']['rationale']['total_output_tokens'])
    if not all(criteria.values()):
        decision = 'stop_expansion_of_indexed_planning_mechanism'
    elif not report['behavior_reliable']:
        decision = 'exploratory_effect_with_behavior_confound'
    elif ratio > 1.25:
        decision = 'exploratory_effect_requires_effort_matched_followup'
    else:
        decision = 'exploratory_effect_warrants_fresh_validation'
    return dict(criteria=criteria, decision=decision, plan_over_rationale_output_ratio=ratio,
        behavior_confounds_present=not report['behavior_reliable'], equal_realized_compute_established=False,
        mechanism_confirmed=False, publication_ready=False, automatic_expansion=False)


def phase(name):
    if name == 'propose9':
        propose()
    elif name in ('qa4', 'qa9'):
        qa('Qwen3.5-' + ('4B' if name == 'qa4' else '9B'))
    else:
        globals()[name]()


def qa(reader):
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid=verify();bundle=load(RUN/'plans.json');assert bundle['signature']==sid and bundle['selection_before_qa']
    assert (RUN/'assemble_complete.json').exists()
    questions={s['question_id']:s['question'] for s in sources()};validate_plan_matrix(bundle['plans'],questions)
    replay=load(BASE/'replay/reader_cache.json');torch_setup()
    backend=TransformersBackend(ROOT/'models'/reader,max_input_tokens=4096)
    for i,p in enumerate(sorted(bundle['plans'],key=lambda p:fp([30,p['question_id'],p['budget'],p['arm']])),1):
        q,b,a=p['question_id'],p['budget'],p['arm'];out=RUN/'responses'/reader/(fp([reader,q,b,a])+'.json');assert not out.exists()
        base=dict(signature=sid,reader=reader,question_id=q,budget=b,arm=a,plan_hash=fp(p))
        if p['status']!='ready':immutable(out,dict(base,status='selection_failed',text='',usage={}));continue
        assert len(backend.tokenizer.encode(p['evidence_text'],add_special_tokens=False))==p['evidence_tokens']
        msg=parent.role.old.common.messages(questions[q],p['evidence_text']);content=fp([reader,msg,128])
        prompt=backend.tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert len(backend.tokenizer.encode(prompt,add_special_tokens=False))<=4096
        key=fp([sid,content]);cache=RUN/'request_cache'/reader/(key+'.json');start=cache.with_suffix('.started.json')
        reused=cache.exists();origin='within_run' if reused else None
        if reused:
            saved=load(cache);assert saved['signature']==sid and saved['content_key']==content;result=saved['result']
        elif content in replay:
            e=replay[content];assert e['reader']==reader and e['messages']==msg and e['max_output']==128
            result=e['result'];reused=True;origin='frozen_prior_request'
            immutable(cache,dict(signature=sid,content_key=content,result=result,replayed_from=e['parent_signature']))
        else:
            assert not start.exists(),'Unknown-call replay forbidden'
            immutable(start,dict(signature=sid,content_key=content,time=time.time()));result=generate(backend,msg,128)
            immutable(cache,dict(signature=sid,content_key=content,result=result))
        immutable(out,dict(base,status='ok' if result['text'].strip() else 'empty',request_key=key,reused=reused,reuse_origin=origin,**result))
        print('QA',reader,i,'/1728',flush=True)



def score():
    sid=verify();gate=load(RUN/'proposal_report.json');assert gate['signature']==sid
    bundle=load(RUN/'plans.json');assert bundle['signature']==sid and bundle['selection_before_qa']
    plans=bundle['plans'];qids=[s['question_id'] for s in sources()];validate_plan_matrix(plans,qids);responses={}
    for r in READERS:
        for p in plans:
            q,b,a=p['question_id'],p['budget'],p['arm'];v=load(RUN/'responses'/r/(fp([r,q,b,a])+'.json'))
            assert v['signature']==sid and v['plan_hash']==fp(p) and v['status'] in ('ok','empty','selection_failed')
            assert (v['reader'],v['question_id'],v['budget'],v['arm'])==(r,q,b,a);responses[r,q,b,a]=v
    assert len(responses)==3456
    refs=load(INPUT_BASE/'private/references.json');assert set(refs)==set(qids);metric=parent.official_metrics();records=[]
    for r in READERS:
        for p in plans:
            q,b,a=p['question_id'],p['budget'],p['arm'];v=responses[r,q,b,a];gold=refs[q]['answer']
            records.append(dict(reader=r,question_id=q,budget=b,arm=a,f1=float(metric['f1_score'](v['text'],gold)[0]),
                em=float(metric['exact_match_score'](v['text'],gold)),evidence_tokens=p['evidence_tokens'],status=v['status'],
                reused=v.get('reused',False),usage=v['usage'],finish_reason=v.get('finish_reason')))
    ix={(v['reader'],v['question_id'],v['budget'],v['arm']):v for v in records}
    def values(a,c,rr=READERS,bb=BUDGETS):return [statistics.mean(ix[r,q,b,a]['f1']-ix[r,q,b,c]['f1'] for r in rr for b in bb) for q in qids]
    comparisons=[dict(arm=a,control=c,**paired(values(a,c))) for a,c in PAIRS]
    per_reader={r:{c:paired(values('plan',c,(r,))) for c in ('direct','rationale')} for r in READERS}
    per_budget={str(b):{c:paired(values('plan',c,bb=(b,))) for c in ('direct','rationale')} for b in BUDGETS}
    groups=[]
    for r in ('pooled',*READERS):
        for b in ('pooled',*BUDGETS):
            for a in ARMS:
                rs=[v for v in records if v['arm']==a and (r=='pooled' or v['reader']==r) and (b=='pooled' or v['budget']==b)]
                groups.append(dict(reader=r,budget=b,arm=a,f1_points=100*statistics.mean(v['f1'] for v in rs),
                    em_points=100*statistics.mean(v['em'] for v in rs),mean_tokens=statistics.mean(v['evidence_tokens'] for v in rs)))
    fresh=[v for v in records if not v['reused'] and v['status'] in ('ok','empty')]
    result=dict(signature=sid,new_proposer_calls=216,complete=True,conditions=len(records),sources=72,exposed_development=True,effectiveness_tested=True,
        comparisons=comparisons,groups=groups,per_reader=per_reader,per_budget=per_budget,records=records,proposal_report=gate,
        new_reader_calls=len(fresh),new_reader_costs={k:sum(v['usage'].get(k,0) for v in fresh) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
        inherited_costs=load(BASE/'replay/costs.json'),**decide(comparisons,per_reader,per_budget,gate))
    immutable(RUN/'summary.json',result);print('SCORE_COMPLETE',result['decision'],comparisons[:2],flush=True)

