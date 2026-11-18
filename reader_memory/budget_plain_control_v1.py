"""Bounded exposed-development test of a budget-aware ordinary extractor."""
from collections import Counter
from copy import deepcopy
import json
import statistics
import time
from . import granularity_factorial_v1 as parent
from .relation_preserving_pilot_v1 import merge_ranges, render_ranges

ROOT = parent.ROOT
BASE = ROOT / 'research/budget_plain_control_v1'
PLAN = BASE / 'development'
RUN = ROOT / 'runs/budget_plain_control_v1'
load, sha, fp, immutable, write_json = parent.load, parent.sha, parent.fp, parent.immutable, parent.write_json
READERS = parent.READERS
BUDGETS = (128, 256, 512)
ARMS = ('original', 'ap3', 'old_plain', 'role_common', 'plain_common', 'budget_plain')
ORDINARY = ('old_plain', 'plain_common', 'budget_plain')
CONFIG = dict(version='budget-plain-control-v1', sources=96, exposed=True, seed=30,
              readers=list(READERS), budgets=list(BUDGETS), arms=list(ARMS),
              proposer='Qwen3.5-9B', proposer_output=2048, proposer_max_input=32768,
              reader_output=128, max_proposer_calls=576, max_reader_conditions=3456,
              calls_per_source_budget='one proposal; one format repair only on invalid output',
              common_packing='same ranked exact quotes, same retained global anchor, exact nominal cap; no role gate',
              development_gate='original minus strongest ordinary >=.02; each reader and main budget >=0; role_common minus best common ordinary >=.01',
              no_ci_gate=True, external_inference_in_this_run=False, training=False,
              resource_caps_mib=dict(propose9=20500, qa4=10500, qa9=19500),
              resource_headroom_mib=dict(propose9=500, qa4=1000, qa9=1000))

BUDGET_INSTRUCTION = '''The public input includes evidence_budget_tokens, the maximum length of the final evidence shown to the reader under the Qwen tokenizer, including source titles and formatting. Select a useful evidence set specifically for this cap. Retain the facts needed for the final requested answer and the intermediate links. Use available space for relevant details or disambiguating context when needed, without filling it with unrelated background. Prioritize more useful quotes first if the complete set may not fit. The program will enforce the exact token cap without cutting a quote in half. The original question remains the target; do not output an answer.'''


def verify():
    m = load(BASE / 'manifest.json')
    assert m['config'] == CONFIG and fp({k: v for k, v in m.items() if k != 'signature'}) == m['signature']
    assert parent.verify() == m['parent_signature']
    for n, digest in m['files'].items():
        assert sha(ROOT / n) == digest, 'Frozen file changed: ' + n
    return m['signature']


def sources():
    return load(PLAN / 'public/sources.json')


def messages(s, budget):
    parent.parent.parent.base.old.common.validate_public(s)
    assert budget in BUDGETS
    data = dict(question=s['question'], evidence_budget_tokens=budget,
                sources=[dict(id=p['source_id'], title=p['title'], text=p['text']) for p in s['paragraphs']])
    return [dict(role='system', content=parent.PLAIN_PROMPT + '\n' + BUDGET_INSTRUCTION),
            dict(role='user', content=json.dumps(data, ensure_ascii=False, separators=(',', ':')))]


def parse_result(s, result):
    if result['finish_reason'] != 'stop':
        return dict(status='invalid', roles=[], candidates=[], errors=['output_limit'])
    return parent.parse_plain(s, result['text'])


def repair_message(s, budget, result, parsed):
    return messages(s, budget) + [dict(role='assistant', content=result['text']), dict(role='user', content=
        'Repair your output once. Return the complete JSON object with the key "quotes" only. Validation errors: '
        + json.dumps(parsed['errors']) + '. Copy only exact original evidence. If none is identifiable return {"quotes":[]}.')]


def runtime():
    m = parent.runtime()
    m.BASE, m.PLAN, m.RUN = BASE, PLAN, RUN
    m.ARMS, m.BUDGETS, m.verify, m.SHARED_CACHE = ARMS, BUDGETS, verify, RUN / 'request_cache'
    m.CONFIG = dict(m.CONFIG, version=CONFIG['version'], arms=list(ARMS), budgets=list(BUDGETS))
    return m


def preflight():
    from transformers import AutoTokenizer
    sid = verify()
    for n, digest in load(BASE / 'manifest.json')['model_files'].items():
        assert sha(ROOT / n) == digest, 'Model changed: ' + n
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models' / CONFIG['proposer']), local_files_only=True, trust_remote_code=False)
    lengths = [len(tok.encode(tok.apply_chat_template(messages(s, b), tokenize=False, add_generation_prompt=True,
                    enable_thinking=False), add_special_tokens=False)) for s in sources() for b in BUDGETS]
    assert len(lengths) == 288 and max(lengths) <= CONFIG['proposer_max_input']
    immutable(RUN / 'preflight.json', dict(signature=sid, model_bytes_verified=True, requests=288,
                                         maximum_proposer_tokens=max(lengths)))
    print('PREFLIGHT_OK', max(lengths), flush=True)


def propose():
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid = verify(); torch_setup()
    backend = TransformersBackend(ROOT / 'models' / CONFIG['proposer'], max_input_tokens=32768)
    counts, costs, actual_calls = Counter(), Counter(), 0
    cases = [(s, b) for s in sources() for b in BUDGETS]
    cases.sort(key=lambda x: fp([CONFIG['seed'], x[0]['question_id'], x[1]]))
    for i, (s, b) in enumerate(cases, 1):
        cid = fp([s['question_id'], b]); out = RUN / 'proposals' / (cid + '.json')
        assert not out.exists(), 'Existing final proposal'
        msg, attempts = messages(s, b), []
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
            parsed = parse_result(s, result)
            immutable(p, dict(signature=sid, source_hash=fp(s), question_id=s['question_id'], budget=b,
                              prompt_hash=fp(msg), result=result, **parsed))
            attempts.append(dict(attempt=attempt, prompt_hash=fp(msg), result=result, status=parsed['status'], errors=parsed['errors']))
            costs.update(result['usage'])
            if parsed['status'] != 'invalid':
                break
            msg = repair_message(s, b, result, parsed)
        immutable(out, dict(signature=sid, source_hash=fp(s), question_id=s['question_id'], budget=b, attempts=attempts, **parsed))
        counts[parsed['status']] += 1
        actual_calls += len(attempts)
        print('PROPOSE', i, '/ 288', b, parsed['status'], 'attempts', len(attempts), flush=True)
    immutable(RUN / 'proposal_gate.json', dict(signature=sid, status_counts=dict(counts), costs=dict(costs),
                                             references_read=False, actual_calls=actual_calls))


def common_pack(s, scores, baseline, ap3, proposal, counter, budget):
    """Only exact source intervals and ranking matter; ignore role metadata."""
    trace = dict(packing='common_ranked_quotes_with_anchor', target_tokens=budget, role_gate=False)
    if baseline.get('selection_error') or ap3.get('selection_error'):
        return deepcopy(ap3), dict(trace, fallback='baseline_selection_failure')
    candidates = proposal.get('candidates', [])
    if proposal.get('status') != 'valid' or not candidates:
        return deepcopy(ap3), dict(trace, fallback='no_valid_quotes')
    by = {p['source_id']: p for p in s['paragraphs']}
    order = parent.packing.sentence_order(s, scores); ranks = {uid: i for i, uid in enumerate(order)}
    units = {}
    for c in candidates:
        p = by[c['source']]
        assert 0 <= c['start'] < c['end'] <= len(p['text'])
        assert p['text'][c['start']:c['end']] == c['text'], 'Source quote drift'
        overlaps = [ranks[p['source_id'] + ':span:' + str(i)] for i, sp in enumerate(p['spans'])
                    if c['start'] < sp['end'] and c['end'] > sp['start']]
        assert overlaps
        key = (c['source'], c['start'], c['end'])
        units[key] = min(overlaps)
    anchor = next(uid for uid in order if uid in set(baseline['selected_spans']))
    pid, _, si = anchor.split(':'); span = by[pid]['spans'][int(si)]
    rr = {pid: [[span['start'], span['end']]]}

    def render(ranges):
        merged = {pid: merge_ranges(by[pid]['text'], rs) for pid, rs in ranges.items()}
        def rank(pid):
            return min(ranks[pid + ':span:' + str(i)] for i, sp in enumerate(by[pid]['spans'])
                       if any(a < sp['end'] and z > sp['start'] for a, z in merged[pid]))
        merged = {pid: merged[pid] for pid in sorted(merged, key=rank)}
        text = render_ranges(s, merged)
        return text, counter.count(text), merged

    assert render(rr)[1] <= budget, 'Existing retained anchor must fit'
    kept, skipped = [], []
    for key in sorted(units, key=lambda k: (units[k], k)):
        pid, a, z = key; trial = deepcopy(rr); trial.setdefault(pid, []).append([a, z])
        text, tokens, merged = render(trial)
        if tokens <= budget:
            rr = merged; kept.append(list(key))
        else:
            skipped.append(list(key))
    text, tokens, rr = render(rr)
    return dict(evidence_text=text, evidence_tokens=tokens, selected_ranges=rr, selected_spans=[]), dict(
        trace, fallback=None, anchor=anchor, accepted_quote_ranges=kept, skipped_quote_ranges=skipped,
        candidate_quotes=len(units), anchor_only=not kept)


def make_source_plans(s, cache, old, budget_proposals, counter):
    q = s['question_id']; scores = cache['ranking']['scores']; result = []
    seeds = [str(x) for x in cache['retrieval']['paths'][0]]
    for b in BUDGETS:
        baseline, ap3 = parent.short.short_plans(s, scores, seeds, counter, b, max(256, b))
        assert ap3['evidence_text'] == old[q, b, 'ap3']['evidence_text']
        for arm, origin in (('original', 'anchored'), ('ap3', 'ap3'), ('old_plain', 'plain_anchor')):
            p = deepcopy(old[q, b, origin]); p['arm'] = arm
            p['trace'] = dict(p['trace'], frozen_origin_arm=origin)
            result.append(p)
        for arm, pr in (('role_common', cache['proposal']), ('plain_common', cache['plain']), ('budget_plain', budget_proposals[b])):
            packed, trace = common_pack(s, scores, baseline, ap3, pr, counter, b)
            result.append(dict(question_id=q, budget=b, arm=arm, evidence_text=packed['evidence_text'],
                evidence_tokens=packed['evidence_tokens'], selected_ranges=packed.get('selected_ranges', {}),
                selected_ids=packed.get('selected_spans', []), trace=trace, private_inputs_read=False,
                status='selection_failed' if packed.get('selection_error') else 'ready', reason=packed.get('selection_error')))
    for p in result:
        assert counter.count(p['evidence_text']) == p['evidence_tokens'] <= p['budget']
    return result


def assemble():
    sid = verify(); cached = load(PLAN / 'public/cached_inputs.json')
    old = {(p['question_id'], p['budget'], p['arm']): p for p in load(PLAN / 'public/parent_plans.json')}
    counter = parent.parent.parent.base.old.common.CounterTokens(); plans = []
    for i, s in enumerate(sources(), 1):
        q = s['question_id']; proposals = {}
        for b in BUDGETS:
            p = load(RUN / 'proposals' / (fp([q, b]) + '.json'))
            assert p['signature'] == sid and p['source_hash'] == fp(s) and p['budget'] == b
            proposals[b] = p
        for k in ('proposal', 'plain', 'retrieval', 'ranking'):
            assert cached[q][k]['source_hash'] == fp(s)
        plans.extend(make_source_plans(s, cached[q], old, proposals, counter))
        print('ASSEMBLE', i, '/ 96', flush=True)
    assert len(plans) == 1728 and len({(p['question_id'], p['budget'], p['arm']) for p in plans}) == 1728
    immutable(RUN / 'plans.json', dict(signature=sid, selection_before_qa=True, plans=plans))


def summarize(records, refs):
    from .discriminative_pilot_v1 import paired
    ix = {(r['reader'], r['question_id'], r['budget'], r['arm']): r for r in records}
    assert len(ix) == len(records) == 3456
    def mean(arm, key, readers=READERS, budgets=(256, 512)):
        return statistics.mean(ix[r, q, b, arm][key] for r in readers for q in refs for b in budgets)
    # Choose one global ordinary method for later confirmation, never per-question oracle routing.
    strongest = max(ORDINARY, key=lambda a: (mean(a, 'f1'), -mean(a, 'evidence_tokens'), -ORDINARY.index(a)))
    common_best = max(('plain_common', 'budget_plain'), key=lambda a: (mean(a, 'f1'), -mean(a, 'evidence_tokens'), a == 'plain_common'))
    contrasts = [('original', a) for a in ORDINARY] + [('original', 'ap3'), ('role_common', 'plain_common'),
                ('role_common', 'budget_plain'), ('budget_plain', 'plain_common'), ('original', 'role_common')]
    groups, comparisons = [], []
    for reader in (*READERS, 'pooled'):
        rr = READERS if reader == 'pooled' else (reader,)
        for budget in (*BUDGETS, 'main_pooled'):
            bb = (256, 512) if budget == 'main_pooled' else (budget,)
            for arm in ARMS:
                groups.append(dict(reader=reader, budget=budget, arm=arm, f1_points=100 * mean(arm, 'f1', rr, bb),
                    em_points=100 * mean(arm, 'em', rr, bb), mean_tokens=mean(arm, 'evidence_tokens', rr, bb)))
            for a, c in contrasts:
                ds = [statistics.mean(ix[r, q, b, a]['f1'] - ix[r, q, b, c]['f1'] for r in rr for b in bb) for q in refs]
                comparisons.append(dict(reader=reader, budget=budget, arm=a, control=c, **paired(ds)))
    def contrast(a, c, r='pooled', b='main_pooled'):
        return next(x for x in comparisons if (x['arm'], x['control'], x['reader'], x['budget']) == (a, c, r, b))
    primary = contrast('original', strongest); mechanism = contrast('role_common', common_best)
    conditions = dict(system_gain=primary['mean_delta_f1'] >= .02,
        each_reader_nonnegative=all(contrast('original', strongest, r)['mean_delta_f1'] >= 0 for r in READERS),
        each_main_budget_nonnegative=all(contrast('original', strongest, b=b)['mean_delta_f1'] >= 0 for b in (256, 512)),
        selection_increment=mechanism['mean_delta_f1'] >= .01)
    return dict(groups=groups, comparisons=comparisons, strongest_ordinary=strongest, strongest_common_ordinary=common_best,
        primary=primary, selection_comparison=mechanism, conditions=conditions,
        development_passed=all(conditions.values()),
        decision='eligible_for_external_confirmation' if all(conditions.values()) else 'stop_before_external_confirmation')


def score():
    from .full_comparison_v1 import answer_scores
    sid = verify(); plans = load(RUN / 'plans.json')['plans']
    responses = {(r, p['question_id'], p['budget'], p['arm']): load(RUN / 'responses' / r /
                    (fp([r, p['question_id'], p['budget'], p['arm']]) + '.json')) for r in READERS for p in plans}
    assert len(responses) == 3456
    refs = load(PLAN / 'private/references.json'); records = []
    for reader in READERS:
        for p in plans:
            q, b, a = p['question_id'], p['budget'], p['arm']; r = responses[reader, q, b, a]
            assert r['signature'] == sid and r['plan_hash'] == fp(p)
            em, f1 = answer_scores(r['text'], refs[q])
            records.append(dict(reader=reader, question_id=q, budget=b, arm=a, em=em, f1=f1,
                evidence_tokens=p['evidence_tokens'], fallback=p['trace'].get('fallback'), status=r['status'],
                reused=r.get('reused', False), usage=r['usage']))
    fresh = [r for r in records if not r['reused'] and r['status'] in ('ok', 'empty')]
    result = summarize(records, refs)
    summary = dict(signature=sid, complete=True, conditions_count=3456, records=records, **result,
        new_reader_calls=len(fresh), reader_costs={k: sum(r['usage'].get(k, 0) for r in fresh)
        for k in ('prompt_tokens', 'completion_tokens', 'elapsed_seconds')},
        proposal_gate=load(RUN / 'proposal_gate.json'),
        caveats='Exposed development only. CIs descriptive. Global ordinary selection is development model selection. '
                'Three budget-specific proposals vs one reusable old proposal: equal per-deployment call ceilings, unequal total curve cost. '
                'Common cap is not exact actual length matching. No independent semantic gold, external inference or training.')
    immutable(RUN / 'summary.json', summary)
    print('SCORE_COMPLETE', summary['decision'], primary_short(summary), flush=True)


def primary_short(summary):
    return dict(primary=summary['primary'], selection=summary['selection_comparison'], conditions=summary['conditions'])


def phase(name):
    if name == 'preflight': preflight()
    elif name == 'propose9': propose()
    elif name == 'assemble': assemble()
    elif name in ('qa4', 'qa9'): runtime().qa('Qwen3.5-' + ('4B' if name == 'qa4' else '9B'))
    elif name == 'score': score()
    else: raise ValueError(name)
