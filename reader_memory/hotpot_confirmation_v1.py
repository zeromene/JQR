"""Frozen-source HotpotQA confirmation, with native context and official scoring."""
import ast
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import math
import re
import statistics
import string
import sys
import time
from . import budget_plain_control_v1 as development
from . import hotpot_public_source_v1 as public
from . import role_quote_v1 as role
from . import role_quote_v2 as role_prompt
from . import granularity_factorial_v1 as ordinary
from . import anchored_quote_v1 as anchored
from . import short_budget_controls_v1 as short
from .discriminative_pilot_v1 import CounterTokens
from .relation_preserving_pilot_v1 import merge_ranges, render_ranges

ROOT = development.ROOT
BASE = ROOT / 'research/hotpot_confirmation_v1'
PLAN = BASE
RUN = ROOT / 'runs/hotpot_confirmation_v1'
load, sha, fp, immutable, write_json = development.load, development.sha, development.fp, development.immutable, development.write_json
READERS = ('Qwen3.5-9B', 'Qwen3.5-4B')
BUDGETS = (256, 512)
ARMS = ('original', 'ap3', 'old_plain', 'role_common', 'budget_plain', 'exit_title')
CONFIG = dict(version='hotpot-confirmation-v1', sources=480, readers=list(READERS), budgets=list(BUDGETS),
    arms=list(ARMS), primary_control='budget_plain', proposer='Qwen3.5-9B', proposer_output=2048,
    proposer_max_input=32768, reader_output=128, seed=30, bootstrap_replicates=10000,
    max_proposer_calls=3840, max_reader_conditions=11520, initial_proposals=1920,
    training=False, automatic_expansion=False, unknown_request_retries=0,
    primary_minimum=.02, mechanism_minimum=.01,
    resource_caps_mib=dict(retrieve=4000, rank=9000, propose9=20500, exit=9000, qa4=10500, qa9=19500),
    resource_headroom_mib=dict(retrieve=1000, rank=1000, propose9=500, exit=1000, qa4=1000, qa9=1000))


def verify_allocation():
    a = load(BASE / 'allocation.json')
    payload = {k: v for k, v in a.items() if k != 'signature'}
    # The allocation author hashed Counter(int keys) before JSON serialization.
    # Restore that declared type for verification; never rewrite the frozen artifact.
    payload['selected_context_counts'] = {int(k): v for k, v in payload['selected_context_counts'].items()}
    assert fp(payload) == a['signature']
    for n, h in a['files'].items():
        assert sha(ROOT / n) == h, 'Allocation changed: ' + n
    assert len(a['selected']) == CONFIG['sources']
    assert len({r['component_id'] for r in a['selected']}) == CONFIG['sources']
    return a


def verify():
    m = load(BASE / 'manifest.json')
    assert m['config'] == CONFIG and fp({k: v for k, v in m.items() if k != 'signature'}) == m['signature']
    for n, h in m['files'].items():
        assert sha(ROOT / n) == h, 'Frozen file changed: ' + n
    assert load(BASE / 'allocation.json')['signature'] == m['allocation_signature']
    return m['signature']


def sources():
    ss = load(BASE / 'public/sources.json')
    assert len(ss) == CONFIG['sources'] and len({s['question_id'] for s in ss}) == len(ss)
    return [execution_source(s) for s in ss]


def execution_source(raw):
    public.validate_public(raw)
    s = deepcopy(raw)
    for p in s['paragraphs']:
        for sp in p['spans']:
            original = p['text'][sp['start']:sp['end']]
            assert original.startswith(sp['text']) and original.strip() == sp['text']
            # Six native sentence tails include whitespace excluded by span.text.
            # Correct only that coordinate; preserve all text, IDs and segmentation.
            sp['end'] = sp['start'] + len(sp['text'])
            assert p['text'][sp['start']:sp['end']] == sp['text']
    return s


def execution_payload(s):
    raw = deepcopy(s)
    for p in raw['paragraphs']:
        p['spans'] = public.sentence_spans(p['text'])
    assert s == execution_source(raw), 'Execution source differs from declared whitespace-only offset adapter'
    return dict(question=s['question'], sources=[dict(id=p['source_id'], title=p['title'], text=p['text']) for p in s['paragraphs']])


def messages(s, kind, budget=None):
    assert kind in ('role', 'plain', 'budget_plain')
    data = execution_payload(s)
    prompt = role_prompt.PROMPT if kind == 'role' else ordinary.PLAIN_PROMPT
    if kind == 'budget_plain':
        assert budget in BUDGETS
        data['evidence_budget_tokens'] = budget
        prompt += '\n' + development.BUDGET_INSTRUCTION
    else:
        assert budget is None
    return [dict(role='system', content=prompt), dict(role='user', content=json.dumps(data, ensure_ascii=False, separators=(',', ':')))]


def cases():
    rows = [(s, k, b) for s in sources() for k, b in (('role', None), ('plain', None), ('budget_plain', 256), ('budget_plain', 512))]
    return sorted(rows, key=lambda x: fp([CONFIG['seed'], x[0]['question_id'], x[1], x[2]]))


def parse_result(s, kind, result):
    if result['finish_reason'] != 'stop':
        return dict(status='invalid', roles=[], candidates=[], errors=['output_limit'])
    return (role.parse_proposal if kind == 'role' else ordinary.parse_plain)(s, result['text'])


def propose():
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid = verify(); torch_setup()
    backend = TransformersBackend(ROOT / 'models' / CONFIG['proposer'], max_input_tokens=CONFIG['proposer_max_input'])
    counts, costs, actual = Counter(), Counter(), 0
    for i, (s, kind, b) in enumerate(cases(), 1):
        cid = fp([s['question_id'], kind, b]); out = RUN / 'proposals' / (cid + '.json')
        assert not out.exists(), 'Existing final proposal'
        msg, attempts = messages(s, kind, b), []
        for attempt in range(2):
            p = RUN / 'proposal_attempts' / (cid + '.' + str(attempt) + '.json')
            started = p.with_suffix('.started.json')
            assert not p.exists() and not started.exists(), 'No unknown request retry'
            immutable(started, dict(signature=sid, source_hash=fp(s), prompt_hash=fp(msg), time=time.time()))
            try:
                result = generate(backend, msg, CONFIG['proposer_output'])
            except Exception as exc:
                immutable(p, dict(signature=sid, status='runtime_failure', error=repr(exc)))
                raise
            parsed = parse_result(s, kind, result)
            immutable(p, dict(signature=sid, source_hash=fp(s), question_id=s['question_id'], kind=kind, budget=b,
                              prompt_hash=fp(msg), result=result, **parsed))
            attempts.append(dict(attempt=attempt, prompt_hash=fp(msg), result=result, status=parsed['status'], errors=parsed['errors']))
            costs.update(result['usage'])
            if parsed['status'] != 'invalid':
                break
            if kind == 'role':
                repair = 'Repair your output once. Return the complete valid JSON object only. Validation errors: ' + json.dumps(parsed['errors']) + '. Use only exact quotes from the original sources; do not invent facts. If no supported evidence is identifiable, return {"roles":[]}.'
            else:
                repair = 'Repair your output once. Return the complete JSON object with the key "quotes" only. Validation errors: ' + json.dumps(parsed['errors']) + '. Copy only exact original evidence. If none is identifiable return {"quotes":[]}.'
            msg = messages(s, kind, b) + [dict(role='assistant', content=result['text']), dict(role='user', content=repair)]
        immutable(out, dict(signature=sid, source_hash=fp(s), question_id=s['question_id'], kind=kind, budget=b, attempts=attempts, **parsed))
        counts[kind + ':' + parsed['status']] += 1; actual += len(attempts)
        print('PROPOSE', i, '/', CONFIG['initial_proposals'], kind, b, parsed['status'], 'attempts', len(attempts), flush=True)
    assert actual <= CONFIG['max_proposer_calls']
    immutable(RUN / 'proposal_gate.json', dict(signature=sid, actual_calls=actual, costs=dict(costs), status_counts=dict(counts), references_parsed=False))


def effective_hops(raw, n):
    assert raw in (1, 2, 3) and 2 <= n <= 10
    return min(raw, n)


def retrieve():
    from .full_comparison_v1 import beam, torch_setup
    sid = verify(); torch = torch_setup(); tok, model = beam.make_model('cpu'); model.float().to('cuda')
    model.load_state_dict(torch.load(ROOT / 'runs/path_retrieval_execution_v1/best.pt', map_location='cpu', weights_only=True), strict=True); model.eval()
    for i, s in enumerate(sources(), 1):
        encoded, audit = beam.codes(tok, s, 'cuda'); start = time.perf_counter()
        costs, handle = beam.encoder_meter(model)
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits = beam.count_logits(torch, model, encoded['q_codes'][0]).float()
                raw = int(logits.argmax(dim=-1).item()) + 1; hops = effective_hops(raw, len(s['paragraphs']))
                result = model(**encoded, sf_idx=[[]], hop=hops)
        finally:
            handle.remove()
        torch.cuda.synchronize(); paths = result['current_preds']
        assert paths and all(len(p) == hops and len(set(p)) == len(p) and all(0 <= j < len(s['paragraphs']) for j in p) for p in paths)
        immutable(RUN / 'retrieval' / (s['question_id'] + '.json'), dict(signature=sid, source_hash=fp(s),
            raw_predicted_hops=raw, predicted_hops=hops, hop_logits=logits[0].cpu().tolist(), paths=paths,
            tokenization=audit, seconds=time.perf_counter() - start, costs=costs))
        print('RETRIEVE', i, '/', CONFIG['sources'], flush=True)


def runtime(module_name):
    spec = importlib.util.spec_from_file_location('reader_memory._hotpot_' + module_name, ROOT / 'reader_memory' / (module_name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.RUN, m.PLAN, m.BASE, m.sources, m.verify = RUN, PLAN, BASE, sources, verify
    m.CONFIG = dict(m.CONFIG, seed=CONFIG['seed'], reader_max_output=CONFIG['reader_output'])
    return m


def exit_ranges(s, candidates):
    """Official sentence indices align with our splitter; retain exact original offsets."""
    result = []
    for c in candidates:
        p = s['paragraphs'][c['document_index']]; sp = p['spans'][c['sentence_index']]
        assert c['sentence'] == sp['text'].strip(), 'EXIT sentence alignment drift'
        assert c['context'] == p['title'].strip() + '\n' + p['text'].strip()
        result.append((p['source_id'], sp['start'], sp['end']))
    assert len(result) == sum(len(p['spans']) for p in s['paragraphs']) and len(set(result)) == len(result)
    return result


def exit_primitives():
    spec = importlib.util.spec_from_file_location('_hotpot_exit_pure', ROOT / 'third_party/recent_exit/compressors/baselines/exit/core.py')
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return m


def exit_pack(s, candidates, scores, counter, budget):
    assert len(candidates) == len(scores) and all(math.isfinite(v) and 0 <= v <= 1 for v in scores)
    units = exit_ranges(s, candidates); by = {p['source_id']: p for p in s['paragraphs']}
    def render(ids):
        rr = {}
        for k in sorted(ids):
            pid, a, z = units[k]; rr.setdefault(pid, []).append([a, z])
        rr = {pid: merge_ranges(by[pid]['text'], ranges) for pid, ranges in rr.items()}
        text = render_ranges(s, rr) if rr else ''
        return dict(evidence_text=text, evidence_tokens=counter.count(text), selected_ranges=rr, selected_spans=[])
    native = [i for i, v in enumerate(scores) if v >= .5]
    packed = render(native); selected = native
    if packed['evidence_tokens'] > budget:
        selected = []
        for i in sorted(native, key=lambda k: (-scores[k], k)):
            if render(selected + [i])['evidence_tokens'] <= budget:
                selected.append(i)
        packed = render(selected)
    return packed, dict(threshold=.5, selected_indices=sorted(selected), native_indices=native,
        adaptation='title-preserving whole-sentence cap; descending probability admission then source order',
        empty_selection=not selected, fallback=None)


def exit_select():
    from dataclasses import asdict
    from experiments.recent_baselines_v1.engines import EXIT
    from .full_comparison_v1 import torch_setup
    sid = verify(); torch_setup(); engine = EXIT(RUN / 'unused_exit_cache')
    from compressors.baselines.exit.core import assemble_compression, build_compression_prompt
    for i, s in enumerate(sources(), 1):
        path = RUN / 'exit_scores' / (s['question_id'] + '.json'); started = path.with_suffix('.started.json')
        assert not path.exists() and not started.exists(), 'No unknown EXIT request retry'
        cs = engine.prepare_candidates([engine.document_type(p['source_id'], p['title'], p['text']) for p in s['paragraphs']], engine.splitter)
        candidates = [asdict(c) for c in cs]; exit_ranges(s, candidates)
        lengths = [len(x) for x in engine.worker.tokenizer([build_compression_prompt(s['question'], c.context, c.sentence) for c in cs], truncation=False)['input_ids']]
        immutable(started, dict(signature=sid, source_hash=fp(s), candidate_hash=fp(candidates), time=time.time()))
        begin = time.perf_counter()
        try:
            scores = engine.worker.score_sentences(s['question'], [c.context for c in cs], [c.sentence for c in cs])
            native = assemble_compression(cs, scores, .5)
        except Exception as exc:
            immutable(path, dict(signature=sid, status='runtime_failure', error=repr(exc))); raise
        immutable(path, dict(signature=sid, source_hash=fp(s), candidates=candidates, scores=list(scores),
            native_text=native.text, costs=dict(seconds=time.perf_counter() - begin, sentences=len(cs), batches=math.ceil(len(cs) / 4), input_tokens=sum(lengths)),
            precision='FP16, no quantization'))
        print('EXIT', i, '/', CONFIG['sources'], 'sentences', len(cs), flush=True)


def make_source_plans(s, ranking, retrieval, proposals, exit_row, counter):
    result = []; scores = ranking['scores']; seeds = [str(x) for x in retrieval['paths'][0]]
    for b in BUDGETS:
        baseline, ap3 = short.short_plans(s, scores, seeds, counter, b, b)
        choices = [('ap3', deepcopy(ap3), dict(fallback=None))]
        for arm, kind in (('original', 'role'), ('old_plain', 'plain')):
            if baseline.get('selection_error') or ap3.get('selection_error'):
                packed, trace = deepcopy(ap3), dict(fallback='baseline_selection_failure')
            else:
                packed, trace = anchored.quote_plan(s, scores, baseline, ap3, proposals[kind, None], counter, 'quote_anchor')
            choices.append((arm, packed, trace))
        for arm, key in (('role_common', ('role', None)), ('budget_plain', ('budget_plain', b))):
            packed, trace = development.common_pack(s, scores, baseline, ap3, proposals[key], counter, b)
            choices.append((arm, packed, trace))
        packed, trace = exit_pack(s, exit_row['candidates'], exit_row['scores'], counter, b)
        choices.append(('exit_title', packed, trace))
        for arm, packed, trace in choices:
            assert counter.count(packed['evidence_text']) == packed['evidence_tokens'] <= b
            result.append(dict(question_id=s['question_id'], budget=b, arm=arm, evidence_text=packed['evidence_text'],
                evidence_tokens=packed['evidence_tokens'], selected_ranges=packed.get('selected_ranges', {}),
                selected_ids=packed.get('selected_spans', []), trace=trace, private_inputs_read=False,
                status='selection_failed' if packed.get('selection_error') else 'ready', reason=packed.get('selection_error')))
    return result


def assemble():
    sid = verify(); counter = CounterTokens(); plans = []
    for i, s in enumerate(sources(), 1):
        q = s['question_id']; rows = [load(RUN / folder / (q + '.json')) for folder in ('ranking', 'retrieval', 'exit_scores')]
        ps = {(k, b): load(RUN / 'proposals' / (fp([q, k, b]) + '.json')) for k, b in (('role', None), ('plain', None), ('budget_plain', 256), ('budget_plain', 512))}
        assert all(r['signature'] == sid and r['source_hash'] == fp(s) for r in [*rows, *ps.values()])
        for key, p in ps.items():
            assert (p['kind'], p['budget']) == key and p['question_id'] == q
        plans.extend(make_source_plans(s, rows[0], rows[1], ps, rows[2], counter))
        print('ASSEMBLE', i, '/', CONFIG['sources'], flush=True)
    assert len(plans) == len({(p['question_id'], p['budget'], p['arm']) for p in plans}) == CONFIG['max_reader_conditions'] // 2
    immutable(RUN / 'plans.json', dict(signature=sid, selection_before_qa=True, plans=plans))


def official_metrics():
    p = BASE / 'sources/hotpot_evaluate_v1.py'
    assert sha(p) == 'd35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0'
    names = {'normalize_answer', 'exact_match_score', 'f1_score'}
    nodes = [n for n in ast.parse(p.read_text(encoding='utf8')).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    ns = dict(re=re, string=string, Counter=Counter)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), ns)
    return ns


def paired(values):
    import numpy as np
    x = np.array(values, dtype=float); assert len(x) == CONFIG['sources']
    rng = np.random.default_rng(CONFIG['seed'])
    bs = x[rng.integers(0, len(x), (CONFIG['bootstrap_replicates'], len(x)))].mean(axis=1)
    return dict(n=len(x), mean_delta_f1=float(x.mean()), ci95=[float(v) for v in np.quantile(bs, [.025, .975])],
                wins=int((x > 0).sum()), ties=int((x == 0).sum()), losses=int((x < 0).sum()))


def decide(primary, mechanism, per_reader, per_budget, recent):
    criteria = dict(primary_minimum=primary['mean_delta_f1'] >= .02, primary_positive_interval=primary['ci95'][0] > 0,
        each_reader_nonnegative=all(v['mean_delta_f1'] >= 0 for v in per_reader.values()),
        each_budget_nonnegative=all(v['mean_delta_f1'] >= 0 for v in per_budget.values()),
        mechanism_minimum=mechanism['mean_delta_f1'] >= .01, mechanism_positive_interval=mechanism['ci95'][0] > 0,
        recent_baseline_point_nonnegative=recent['mean_delta_f1'] >= 0)
    if all(criteria.values()):
        decision = 'eligible_for_offline_selector_design'
    elif primary['mean_delta_f1'] <= 0 or primary['ci95'][1] < .02:
        decision = 'stop_practical_futility'
    elif criteria['primary_minimum'] and criteria['primary_positive_interval'] and not (criteria['mechanism_minimum'] and criteria['mechanism_positive_interval']):
        decision = 'stop_mechanism_not_confirmed'
    else:
        decision = 'stop_unconfirmed_or_noncompetitive'
    return dict(criteria=criteria, decision=decision, publication_ready=False, automatic_extra_samples=False)


def score():
    sid = verify(); bundle = load(RUN / 'plans.json'); plans = bundle['plans']
    assert bundle['signature'] == sid and bundle['selection_before_qa']
    assert len(plans) == len({(p['question_id'], p['budget'], p['arm']) for p in plans}) == CONFIG['max_reader_conditions'] // 2
    responses = {(r, p['question_id'], p['budget'], p['arm']): load(RUN / 'responses' / r / (fp([r, p['question_id'], p['budget'], p['arm']]) + '.json')) for r in READERS for p in plans}
    assert len(responses) == CONFIG['max_reader_conditions']
    # Validate EVERY response before parsing a reference answer.
    for r in READERS:
        for p in plans:
            row = responses[r, p['question_id'], p['budget'], p['arm']]
            assert row['signature'] == sid and row['plan_hash'] == fp(p) and row['status'] in ('ok', 'empty', 'selection_failed')
    refs = load(BASE / 'private/references.json'); assert set(refs) == {s['question_id'] for s in sources()}
    metric = official_metrics(); records = []
    for r in READERS:
        for p in plans:
            row = responses[r, p['question_id'], p['budget'], p['arm']]; gold = refs[p['question_id']]['answer']
            em, f1 = float(metric['exact_match_score'](row['text'], gold)), float(metric['f1_score'](row['text'], gold)[0])
            records.append(dict(reader=r, question_id=p['question_id'], budget=p['budget'], arm=p['arm'], em=em, f1=f1,
                evidence_tokens=p['evidence_tokens'], status=row['status'], reused=row.get('reused', False), usage=row['usage'],
                finish_reason=row.get('finish_reason'), fallback=p['trace'].get('fallback')))
    ix = {(v['reader'], v['question_id'], v['budget'], v['arm']): v for v in records}; qids = list(refs)
    def contrast(a, c, rr=READERS, bb=BUDGETS):
        return paired([statistics.mean(ix[r, q, b, a]['f1'] - ix[r, q, b, c]['f1'] for r in rr for b in bb) for q in qids])
    primary = contrast('original', 'budget_plain'); mechanism = contrast('role_common', 'budget_plain')
    per_reader = {r: contrast('original', 'budget_plain', (r,)) for r in READERS}
    per_budget = {str(b): contrast('original', 'budget_plain', bb=(b,)) for b in BUDGETS}
    recent = contrast('original', 'exit_title'); comparisons = []
    groups = [dict(reader=r, budget=b, arm=a, f1_points=100 * statistics.mean(ix[rr, q, bb, a]['f1'] for rr in (READERS if r == 'pooled' else (r,)) for q in qids for bb in (BUDGETS if b == 'pooled' else (b,))),
                   em_points=100 * statistics.mean(ix[rr, q, bb, a]['em'] for rr in (READERS if r == 'pooled' else (r,)) for q in qids for bb in (BUDGETS if b == 'pooled' else (b,))),
                   mean_tokens=statistics.mean(ix[rr, q, bb, a]['evidence_tokens'] for rr in (READERS if r == 'pooled' else (r,)) for q in qids for bb in (BUDGETS if b == 'pooled' else (b,))))
              for r in (*READERS, 'pooled') for b in (*BUDGETS, 'pooled') for a in ARMS]
    for a, c in (('original', 'budget_plain'), ('role_common', 'budget_plain'), ('original', 'exit_title'), ('original', 'ap3'), ('original', 'old_plain'), ('original', 'role_common')):
        comparisons.append(dict(arm=a, control=c, **contrast(a, c)))
    fresh = [v for v in records if not v['reused'] and v['status'] in ('ok', 'empty')]
    summary = dict(signature=sid, complete=True, records=records, groups=groups, comparisons=comparisons,
        primary=primary, mechanism=mechanism, per_reader=per_reader, per_budget=per_budget, recent=recent,
        **decide(primary, mechanism, per_reader, per_budget, recent), new_reader_calls=len(fresh),
        reader_costs={k: sum(v['usage'].get(k, 0) for v in fresh) for k in ('prompt_tokens', 'completion_tokens', 'elapsed_seconds')},
        proposal_gate=load(RUN / 'proposal_gate.json'),
        caveats='480 source components, repeated reader/budget cells. Primary then mechanism fixed-sequence inference; other CIs descriptive. Same model family. EXIT title/budget adaptation with Hotpot train supervision. Equal caps, not equal lengths or end-to-end compute. Not a full-benchmark official test or publication certification.')
    immutable(RUN / 'summary.json', summary)
    print('SCORE_COMPLETE', summary['decision'], json.dumps(primary), flush=True)


def preflight():
    from dataclasses import asdict
    from transformers import AutoTokenizer
    from .full_comparison_v1 import beam, ranker
    sid = verify(); a = verify_allocation()
    for n, h in load(BASE / 'manifest.json')['model_files'].items():
        assert sha(ROOT / n) == h, 'Model changed: ' + n
    ss = sources(); assert {s['native_id'] for s in ss} == {v['native_id'] for v in a['selected']}
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models' / CONFIG['proposer']), local_files_only=True, trust_remote_code=False)
    lengths = [len(tok.encode(tok.apply_chat_template(messages(s, k, b), tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)) for s, k, b in cases()]
    assert max(lengths) <= CONFIG['proposer_max_input']
    bt = beam.tokenizer(); rt = AutoTokenizer.from_pretrained(str(ranker.MODEL), local_files_only=True, trust_remote_code=False)
    et = AutoTokenizer.from_pretrained(str(ROOT / 'models/gemma-2b-it'), local_files_only=True, trust_remote_code=False)
    official = exit_primitives(); n = 0; exit_max = 0
    for s in ss:
        beam.codes(bt, s, 'cpu')
        for _, text in ranker.documents(s):
            ranker.rank_input_ids(rt, s['question'], text); n += 1
        cs = official.prepare_candidates([official.EXITDocument(p['source_id'], p['title'], p['text']) for p in s['paragraphs']],
            lambda text: [sp['text'] for sp in public.sentence_spans(text)])
        exit_ranges(s, [asdict(c) for c in cs])
        el = [len(x) for x in et([official.build_compression_prompt(s['question'], c.context, c.sentence) for c in cs], truncation=False)['input_ids']]
        exit_max = max(exit_max, max(el))
    assert exit_max <= load(ROOT / 'models/gemma-2b-it/config.json')['max_position_embeddings']
    immutable(RUN / 'preflight.json', dict(signature=sid, model_bytes_verified=True, sources=len(ss),
        proposer_prompts=len(lengths), maximum_proposer_tokens=max(lengths), ranking_units=n,
        maximum_exit_tokens=exit_max, references_parsed=False))
    print('PREFLIGHT_OK', len(ss), max(lengths), n, flush=True)


def phase(name):
    if name == 'preflight': preflight()
    elif name == 'retrieve': retrieve()
    elif name == 'rank': runtime('relation_bundle_pilot_v1').rank()
    elif name == 'propose9': propose()
    elif name == 'exit': exit_select()
    elif name == 'assemble': assemble()
    elif name in ('qa4', 'qa9'):
        assert not (PLAN / 'private/reader_replay.json').exists(), 'External replay is not registered'
        runtime('role_quote_v1').qa('Qwen3.5-' + ('4B' if name == 'qa4' else '9B'))
    elif name == 'score': score()
    else: raise ValueError(name)
