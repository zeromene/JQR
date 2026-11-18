"""One registered family transfer on unchanged evidence; no selection or training."""
from collections import Counter
import json
import statistics
import time
from . import hotpot_confirmation_v1 as parent

ROOT = parent.ROOT
BASE = ROOT / 'research/hotpot_phi_transfer_v1'
RUN = ROOT / 'runs/hotpot_phi_transfer_v1'
load, sha, fp, immutable, write_json = parent.load, parent.sha, parent.fp, parent.immutable, parent.write_json
READER = 'Phi-4-mini-instruct'
BUDGETS, ARMS = parent.BUDGETS, parent.ARMS
CONFIG = dict(version='hotpot-phi-transfer-v1', sources=480, conditions=5760, reader=READER,
    budgets=list(BUDGETS), arms=list(ARMS), output=128, max_input=4096, seed=30,
    source_panel='Already scored with Qwen; new reader transfer, not new-source confirmation',
    resource_caps_mib=dict(qa_phi=9500), resource_headroom_mib=dict(qa_phi=1000),
    training=False, automatic_expansion=False, primary_minimum=.02)


def verify():
    m = load(BASE / 'manifest.json')
    assert m['config'] == CONFIG and m['signature'] == fp({k: v for k, v in m.items() if k != 'signature'})
    assert parent.verify() == m['parent_signature']
    for n, h in m['files'].items(): assert sha(ROOT / n) == h, n
    return m['signature']


def messages(p, source):
    return parent.role.old.common.messages(source['question'], p['evidence_text'])


def preflight():
    from transformers import AutoTokenizer
    sid = verify()
    for item in load(ROOT / 'models' / READER / 'runtime_manifest.json')['files']:
        p = ROOT / 'models' / READER / item['path']
        assert p.stat().st_size == item['size'] and sha(p) == item['sha256']
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models' / READER), local_files_only=True, trust_remote_code=False)
    byq = {s['question_id']: s for s in parent.sources()}; bundle = load(BASE / 'public/parent_plans.json')
    assert bundle['signature'] == parent.verify() and bundle['selection_before_qa']
    plans = []
    for p in bundle['plans']:
        msg = messages(p, byq[p['question_id']])
        prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        n = len(tok.encode(prompt, add_special_tokens=False)); assert n <= CONFIG['max_input']
        plans.append(dict(p, origin_plan_hash=fp(p), phi_evidence_tokens=len(tok.encode(p['evidence_text'], add_special_tokens=False)), phi_prompt_tokens=n))
    assert len(plans) == len({(p['question_id'], p['budget'], p['arm']) for p in plans}) == CONFIG['conditions']
    immutable(RUN / 'plans.json', dict(signature=sid, selection_before_qa=True, plans=plans))
    immutable(RUN / 'preflight.json', dict(signature=sid, model_bytes_verified=True, conditions=len(plans), maximum_prompt_tokens=max(p['phi_prompt_tokens'] for p in plans)))
    print('PREFLIGHT_OK', len(plans), flush=True)


def qa():
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    sid = verify(); bundle = load(RUN / 'plans.json'); assert bundle['signature'] == sid
    byq = {s['question_id']: s for s in parent.sources()}; torch_setup()
    backend = TransformersBackend(ROOT / 'models' / READER, max_input_tokens=CONFIG['max_input'])
    for i, p in enumerate(sorted(bundle['plans'], key=lambda p: fp([30, p['question_id'], p['budget'], p['arm']])), 1):
        out = RUN / 'responses' / (fp([READER, p['question_id'], p['budget'], p['arm']]) + '.json')
        assert not out.exists(), 'Existing answer; no silent rerun'
        base = dict(signature=sid, reader=READER, question_id=p['question_id'], budget=p['budget'], arm=p['arm'], plan_hash=fp(p))
        if p['status'] != 'ready':
            immutable(out, dict(base, status='selection_failed', text='', usage={})); continue
        assert len(backend.tokenizer.encode(p['evidence_text'], add_special_tokens=False)) == p['phi_evidence_tokens']
        msg = messages(p, byq[p['question_id']]); key = fp([sid, READER, msg, CONFIG['output']])
        cache = RUN / 'request_cache' / (key + '.json'); start = cache.with_suffix('.started.json'); reused = cache.exists()
        if reused:
            saved = load(cache); assert saved['signature'] == sid and saved['request_key'] == key
            result = saved['result']
        else:
            assert not start.exists(), 'No unknown request retry'
            immutable(start, dict(signature=sid, request_key=key, time=time.time()))
            try: result = generate(backend, msg, CONFIG['output'])
            except Exception as exc:
                immutable(out, dict(base, status='generation_failed', text='', usage={}, error=repr(exc))); raise
            immutable(cache, dict(signature=sid, request_key=key, result=result))
        immutable(out, dict(base, status='ok' if result['text'].strip() else 'empty', request_key=key, reused=reused, **result))
        print('QA_PHI', i, '/', CONFIG['conditions'], flush=True)


def decision(primary, per_budget):
    passed = primary['mean_delta_f1'] >= .02 and primary['ci95'][0] > 0 and all(v['mean_delta_f1'] >= 0 for v in per_budget.values())
    return 'family_transfer_supported' if passed else 'family_transfer_not_confirmed'


def score():
    sid = verify(); bundle = load(RUN / 'plans.json'); assert bundle['signature'] == sid
    plans = bundle['plans']; assert len(plans) == CONFIG['conditions']
    responses = {(p['question_id'], p['budget'], p['arm']): load(RUN / 'responses' / (fp([READER, p['question_id'], p['budget'], p['arm']]) + '.json')) for p in plans}
    assert len(responses) == CONFIG['conditions']
    for p in plans:
        r = responses[p['question_id'], p['budget'], p['arm']]
        assert r['signature'] == sid and r['plan_hash'] == fp(p) and r['status'] in ('ok', 'empty', 'selection_failed')
    refs = load(parent.BASE / 'private/references.json'); metric = parent.official_metrics(); records = []
    for p in plans:
        q, b, a = p['question_id'], p['budget'], p['arm']; r = responses[q, b, a]; gold = refs[q]['answer']
        records.append(dict(question_id=q, budget=b, arm=a, em=float(metric['exact_match_score'](r['text'], gold)),
            f1=float(metric['f1_score'](r['text'], gold)[0]), status=r['status'], usage=r['usage'], reused=r.get('reused', False),
            finish_reason=r.get('finish_reason'), qwen_tokens=p['evidence_tokens'], phi_tokens=p['phi_evidence_tokens']))
    ix = {(r['question_id'], r['budget'], r['arm']): r for r in records}
    def contrast(a, c, bb=BUDGETS):
        return parent.paired([statistics.mean(ix[q, b, a]['f1'] - ix[q, b, c]['f1'] for b in bb) for q in refs])
    primary = contrast('original', 'budget_plain'); per_budget = {str(b): contrast('original', 'budget_plain', (b,)) for b in BUDGETS}
    comparisons = [dict(arm=a, control=c, **contrast(a, c)) for a, c in (('original', 'budget_plain'), ('role_common', 'budget_plain'), ('original', 'exit_title'), ('original', 'ap3'), ('original', 'old_plain'))]
    groups = []
    for b in (*BUDGETS, 'pooled'):
        for a in ARMS:
            rr = [r for r in records if r['arm'] == a and (b == 'pooled' or r['budget'] == b)]
            groups.append(dict(budget=b, arm=a, f1_points=100 * statistics.mean(r['f1'] for r in rr), em_points=100 * statistics.mean(r['em'] for r in rr),
                               qwen_tokens=statistics.mean(r['qwen_tokens'] for r in rr), phi_tokens=statistics.mean(r['phi_tokens'] for r in rr)))
    fresh = [r for r in records if not r['reused'] and r['status'] in ('ok', 'empty')]
    result = dict(signature=sid, complete=True, conditions=len(records), records=records, groups=groups, primary=primary,
        per_budget=per_budget, comparisons=comparisons, decision=decision(primary, per_budget), new_reader_calls=len(fresh),
        reader_costs={k: sum(r['usage'].get(k, 0) for r in fresh) for k in ('prompt_tokens', 'completion_tokens', 'elapsed_seconds')},
        finish_reasons=dict(Counter(str(r['finish_reason']) for r in records)), publication_ready=False,
        caveats='Registered after Qwen results, before Phi outputs; same exposed sources and exact fixed evidence. Not new-source confirmation. Qwen caps and actual Phi token lengths differ. Secondary intervals descriptive. No retraining or repacking.')
    immutable(RUN / 'summary.json', result); print('SCORE_COMPLETE', result['decision'], json.dumps(primary), flush=True)


def phase(name):
    if name == 'qa_phi': qa()
    elif name in ('preflight', 'score'): globals()[name]()
    else: raise ValueError(name)
