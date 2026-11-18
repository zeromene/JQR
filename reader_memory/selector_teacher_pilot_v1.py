"""Bounded teacher diagnostics on allocated training sources; no student training."""
from collections import Counter
from copy import deepcopy
import importlib.util
import statistics
from . import hotpot_confirmation_v1 as parent

ROOT = parent.ROOT
BASE = ROOT / 'research/selector_teacher_pilot_v1'
PLAN = BASE
RUN = ROOT / 'runs/selector_teacher_pilot_v1'
load, sha, fp, immutable, write_json = parent.load, parent.sha, parent.fp, parent.immutable, parent.write_json
READERS, BUDGETS = parent.READERS, parent.BUDGETS
ARMS = ('original', 'ap3', 'role_common', 'budget_plain')
CONFIG = dict(parent.CONFIG, version='selector-teacher-pilot-v1', sources=64, arms=list(ARMS),
    initial_proposals=192, max_proposer_calls=384, max_reader_conditions=1024,
    training=False, source_role='First64 allocated training components; development only',
    resource_caps_mib=dict(retrieve=4000, rank=9000, propose9=20500, qa4=10500, qa9=19500),
    resource_headroom_mib=dict(retrieve=1000, rank=1000, propose9=500, qa4=1000, qa9=1000))


def verify():
    m = load(BASE / 'manifest.json')
    assert m['config'] == CONFIG and m['signature'] == fp({k: v for k, v in m.items() if k != 'signature'})
    assert parent.verify() == m['parent_signature']
    for n, v in m['files'].items(): assert sha(ROOT / n) == v, n
    return m['signature']


def sources():
    rows = load(BASE / 'public/sources.json'); assert len(rows) == len({r['question_id'] for r in rows}) == 64
    return [parent.execution_source(r) for r in rows]


def cases():
    rows = [(s, kind, b) for s in sources() for kind, b in (('role', None), ('budget_plain', 256), ('budget_plain', 512))]
    return sorted(rows, key=lambda x: fp([30, x[0]['question_id'], x[1], x[2]]))


def runtime():
    spec = importlib.util.spec_from_file_location('reader_memory._selector_teacher_runtime', ROOT / 'reader_memory/hotpot_confirmation_v1.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.BASE, m.PLAN, m.RUN, m.CONFIG = BASE, BASE, RUN, CONFIG
    m.sources, m.verify, m.cases, m.ARMS = sources, verify, cases, ARMS
    return m


def preflight():
    from transformers import AutoTokenizer
    from .full_comparison_v1 import beam, ranker
    sid = verify()
    for n, h in load(BASE / 'manifest.json')['model_files'].items(): assert sha(ROOT / n) == h, n
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/Qwen3.5-9B'), local_files_only=True, trust_remote_code=False)
    lengths = [len(tok.encode(tok.apply_chat_template(parent.messages(s, k, b), tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)) for s, k, b in cases()]
    assert max(lengths) <= CONFIG['proposer_max_input']
    bt = beam.tokenizer(); rt = AutoTokenizer.from_pretrained(str(ranker.MODEL), local_files_only=True, trust_remote_code=False)
    n = 0
    for s in sources():
        beam.codes(bt, s, 'cpu')
        for _, text in ranker.documents(s): ranker.rank_input_ids(rt, s['question'], text); n += 1
    immutable(RUN / 'preflight.json', dict(signature=sid, sources=64, prompts=len(lengths), maximum_prompt=max(lengths), rank_units=n, private_values_parsed=False))
    print('PREFLIGHT_OK', len(lengths), max(lengths), n, flush=True)


def make_plans(s, scores, retrieval, proposals, counter):
    plans = []
    for b in BUDGETS:
        baseline, ap3 = parent.short.short_plans(s, scores, [str(x) for x in retrieval['paths'][0]], counter, b, b)
        choices = [('ap3', deepcopy(ap3), dict(fallback=None))]
        if baseline.get('selection_error') or ap3.get('selection_error'):
            packed, trace = deepcopy(ap3), dict(fallback='baseline_selection_failure')
        else: packed, trace = parent.anchored.quote_plan(s, scores, baseline, ap3, proposals['role', None], counter, 'quote_anchor')
        choices.append(('original', packed, trace))
        for arm, key in (('role_common', ('role', None)), ('budget_plain', ('budget_plain', b))):
            packed, trace = parent.development.common_pack(s, scores, baseline, ap3, proposals[key], counter, b)
            choices.append((arm, packed, trace))
        for arm, packed, trace in choices:
            assert counter.count(packed['evidence_text']) == packed['evidence_tokens'] <= b
            plans.append(dict(question_id=s['question_id'], budget=b, arm=arm, evidence_text=packed['evidence_text'],
                evidence_tokens=packed['evidence_tokens'], selected_ranges=packed.get('selected_ranges', {}),
                selected_ids=packed.get('selected_spans', []), trace=trace, private_inputs_read=False,
                status='selection_failed' if packed.get('selection_error') else 'ready', reason=packed.get('selection_error')))
    return plans


def assemble():
    sid = verify(); plans = []; counter = parent.CounterTokens()
    for i, s in enumerate(sources(), 1):
        q = s['question_id']; ranked = load(RUN / 'ranking' / (q + '.json')); retrieval = load(RUN / 'retrieval' / (q + '.json'))
        ps = {(k, b): load(RUN / 'proposals' / (fp([q, k, b]) + '.json')) for k, b in (('role', None), ('budget_plain', 256), ('budget_plain', 512))}
        assert all(v['signature'] == sid and v['source_hash'] == fp(s) for v in [ranked, retrieval, *ps.values()])
        plans.extend(make_plans(s, ranked['scores'], retrieval, ps, counter))
        print('ASSEMBLE', i, '/64', flush=True)
    assert len(plans) == len({(p['question_id'], p['budget'], p['arm']) for p in plans}) == 512
    immutable(RUN / 'plans.json', dict(signature=sid, plans=plans, selection_before_qa=True))


def plan_ranges(s, plan):
    rr = deepcopy(plan.get('selected_ranges') or {})
    for uid in plan.get('selected_ids', []):
        pid, _, si = uid.split(':'); sp = s['paragraphs'][int(pid)]['spans'][int(si)]
        rr.setdefault(pid, []).append([sp['start'], sp['end']])
    return {pid: parent.merge_ranges(s['paragraphs'][int(pid)]['text'], ranges) for pid, ranges in rr.items()}


def coverage(s, plan, labels):
    # Diagnostic only: annotations are not exhaustive and string overlap is not entailment.
    rr = plan_ranges(s, plan); full = 0; fractions = []; docs = set()
    for label in labels:
        pid, a, z = label['source'], label['start'], label['end']; n = max(1, z - a)
        overlap = sum(max(0, min(z, end) - max(a, start)) for start, end in rr.get(pid, []))
        fractions.append(min(1., overlap / n)); full += any(start <= a and end >= z for start, end in rr.get(pid, []))
        if overlap: docs.add(pid)
    expected = {x['source'] for x in labels}
    return dict(annotated_sentences=len(labels), full_sentences=full,
        mean_annotation_character_coverage=statistics.mean(fractions) if fractions else None,
        touched_support_documents=len(docs), expected_support_documents=len(expected), all_support_documents_touched=docs == expected)


def score():
    sid = verify(); plans = load(RUN / 'plans.json')['plans']; assert len(plans) == 512
    rows = {(r, p['question_id'], p['budget'], p['arm']): load(RUN / 'responses' / r / (fp([r, p['question_id'], p['budget'], p['arm']]) + '.json')) for r in READERS for p in plans}
    assert len(rows) == 1024
    for r in READERS:
        for p in plans:
            row = rows[r, p['question_id'], p['budget'], p['arm']]
            assert row['signature'] == sid and row['plan_hash'] == fp(p) and row['status'] in ('ok', 'empty', 'selection_failed')
    refs = load(BASE / 'private/references.json'); byq = {s['question_id']: s for s in sources()}; assert set(refs) == set(byq)
    metric = parent.official_metrics(); records = []
    for r in READERS:
        for p in plans:
            q, b, a = p['question_id'], p['budget'], p['arm']; row = rows[r, q, b, a]; gold = refs[q]['answer']
            records.append(dict(reader=r, question_id=q, budget=b, arm=a, f1=float(metric['f1_score'](row['text'], gold)[0]),
                em=float(metric['exact_match_score'](row['text'], gold)), evidence_tokens=p['evidence_tokens'], status=row['status'],
                usage=row['usage'], reused=row.get('reused', False), finish_reason=row.get('finish_reason'), fallback=p['trace'].get('fallback')))
    ix = {(r['reader'], r['question_id'], r['budget'], r['arm']): r for r in records}
    from .discriminative_pilot_v1 import paired
    comparisons = []
    for a, c in (('original', 'budget_plain'), ('role_common', 'budget_plain'), ('original', 'ap3')):
        values = [statistics.mean(ix[r, q, b, a]['f1'] - ix[r, q, b, c]['f1'] for r in READERS for b in BUDGETS) for q in refs]
        comparisons.append(dict(arm=a, control=c, **paired(values)))
    groups = [dict(arm=a, f1_points=100 * statistics.mean(r['f1'] for r in records if r['arm'] == a),
                   mean_tokens=statistics.mean(p['evidence_tokens'] for p in plans if p['arm'] == a)) for a in ARMS]
    diagnostic = [dict(question_id=p['question_id'], arm=p['arm'], budget=p['budget'], **coverage(byq[p['question_id']], p, refs[p['question_id']]['support_sentences'])) for p in plans]
    gate = load(RUN / 'proposal_gate.json'); role_valid = gate['status_counts'].get('role:valid', 0) / 64
    delta = comparisons[1]['mean_delta_f1']
    stop = role_valid < .85 or delta < -.02
    fresh = [r for r in records if not r['reused'] and r['status'] in ('ok', 'empty')]
    result = dict(signature=sid, complete=True, development_only=True, records=records, groups=groups, comparisons=comparisons,
        annotation_diagnostics=diagnostic, proposal_gate=gate, role_valid_fraction=role_valid,
        decision='stop_teacher_for_review' if stop else 'eligible_for_semantic_label_review',
        student_training_authorized_by_this_result=False, semantic_reliability_established=False,
        new_reader_calls=len(fresh), reader_costs={k: sum(r['usage'].get(k, 0) for r in fresh) for k in ('prompt_tokens', 'completion_tokens', 'elapsed_seconds')},
        note='Pilot is allocated training material. Character coverage is diagnostic, not a semantic label; keep failures. Passing permits explicit semantic review and a separately frozen student design, not automatic training.')
    immutable(RUN / 'summary.json', result)
    print('SCORE_COMPLETE', result['decision'], comparisons, flush=True)


def phase(name):
    if name in ('preflight', 'assemble', 'score'): globals()[name]()
    elif name in ('retrieve', 'propose9'): getattr(runtime(), 'retrieve' if name == 'retrieve' else 'propose')()
    elif name == 'rank': runtime().runtime('relation_bundle_pilot_v1').rank()
    elif name in ('qa4', 'qa9'): runtime().phase(name)
    else: raise ValueError(name)
