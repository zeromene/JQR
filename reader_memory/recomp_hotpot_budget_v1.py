"""Pinned multi-hop-domain RECOMP baseline; frozen unfilled path retained."""
import argparse
from collections import Counter
import gc
import math
from pathlib import Path
import statistics
import time

from . import path_budget_curve_v1 as parent
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/recomp_hotpot_budget_v1'
RUN = ROOT / 'runs/recomp_hotpot_budget_v1'
MODEL = ROOT / 'models/RECOMP-HotpotQA-extractive'
REVISION = '25d11ad98ac802590c903af50d8735049679e8ef'
load, sha, immutable = parent.load, parent.sha, parent.immutable
recomp = parent.recomp
BUDGETS = (256, 512, 1024, 2048)
METHODS = ('recomp_sentence', 'beam_path_compress', 'recomp_hotpot_sentence')
CONFIG = dict(version='recomp-hotpot-budget-v1', questions=128, sources=64,
    budgets=list(BUDGETS), methods=list(METHODS), max_conditions=1536,
    max_new_reader_calls=512, model_repository='fangyuan/hotpotqa_extractive_compressor',
    model_revision=REVISION, encoder='CPU FP32/eager; batch32; masked mean; unnormalized dot product',
    reader='Qwen3.5-9B', reader_variant='format_only', output_tokens=2048,
    decoding='greedy-thinking_false', retries=0, new_training=False,
    primary='256 per-budget paired comparison on exposed development data',
    secondary='512; 1024 and2048 as higher-budget reference; no favorable-budget omission',
    role='stronger domain-relevant baseline check; no new path or compression mechanism',
    independent_confirmation=False, protected_final_data_accessed=False, automatic_followup=False)


def arm(method, budget):
    if method not in METHODS or budget not in BUDGETS:
        raise ValueError('Unregistered baseline condition')
    return method + '__b' + str(budget)


def hotpot_plan(source, scores, counter, budget):
    if budget not in BUDGETS:
        raise ValueError('Unregistered budget')
    parent.parent.sentence_order(source, scores)  # Exact unit coverage and finite scores.
    plan, _ = recomp.pack(source, scores, 'recomp_sentence', counter, budget)
    plan.update(method='recomp_hotpot_sentence', evidence_budget=budget,
                arm=arm('recomp_hotpot_sentence', budget))
    return plan


def model_files():
    manifest = load(MODEL / 'artifact_manifest.json')
    if manifest['revision'] != REVISION or manifest['repository'] != CONFIG['model_repository']:
        raise ValueError('Official multi-hop model identity differs')
    for item in manifest['files']:
        path = MODEL / item['name']
        if path.stat().st_size != item['size'] or sha(path) != item['sha256']:
            raise ValueError('Model artifact changed: ' + item['name'])


def freeze():
    previous = parent.verify()
    model_files()
    names = ['reader_memory/recomp_hotpot_budget_v1.py', 'tools/prepare_recomp_hotpot_budget_v1.py',
        'tools/stage_recomp_hotpot_budget_v1.py', 'scripts/recomp_hotpot_budget_v1_queue.py',
        'tests/test_recomp_hotpot_budget_v1.py', 'research/path_budget_curve_v1/manifest.json',
        'models/RECOMP-HotpotQA-extractive/artifact_manifest.json']
    names += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
              and p.name not in {'manifest.json', 'CURRENT_STATUS.md', 'submission_receipt.json'}
              and 'snapshots' not in p.relative_to(PLAN).parts and p.suffix != '.pyc']
    value = dict(config=CONFIG, parent_signature=previous,
                 files={name: sha(ROOT / name) for name in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_RECOMP_HOTPOT_BUDGET', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Hotpot baseline identity changed')
    if parent.verify() != value['parent_signature']:
        raise ValueError('Frozen parent changed')
    for name, digest in value['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return value['signature']


def replay_cache():
    marker = load(PLAN / 'preparation.json')['reader_marker_sha256']
    cache = {}
    for item in load(PLAN / 'private/replay_cache.json'):
        recomp.interface.parent.check_cache_entry(item, marker)
        if item['key'] in cache:
            raise ValueError('Duplicate historical request')
        cache[item['key']] = item['entry']
    return cache


def preflight():
    sid = verify()
    model_files()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    counter = ExactCounter()
    sources = load(PLAN / 'public/source_inputs.json')
    if len(sources) != 128 or len({s['native_id'] for s in sources}) != 64:
        raise ValueError('Development panel changed')
    maximum, units = 0, 0
    for source in sources:
        parent.parent.previous.parent.validate_source(source)
        encoded = [tok.encode(text, add_special_tokens=True, truncation=False)
                   for text in [source['question']] + [t for _, t in recomp.units(source)]]
        maximum = max(maximum, max(map(len, encoded)))
        units += len(encoded) - 1
        if maximum > 512:
            raise ValueError('No silent compressor truncation')
    cache = replay_cache()
    old = load(PLAN / 'public/baseline_plans.json')
    marker = load(PLAN / 'preparation.json')['reader_marker_sha256']
    if sha(ROOT / 'models/Qwen3.5-9B/.verified.json') != marker:
        raise ValueError('Reader identity changed')
    expected = {(s['question_id'], arm(m, b)) for s in sources for b in BUDGETS for m in METHODS[:2]}
    if len(old) != 1024 or {(p['question_id'], p['arm']) for p in old} != expected:
        raise ValueError('Incomplete cached control matrix')
    for plan in old:
        key = recomp.interface.parent.request_key(recomp.messages_for(plan), marker)
        if key not in cache or counter.count(plan['evidence_text']) != plan['evidence_tokens'] or plan['evidence_tokens'] > plan['evidence_budget']:
            raise ValueError('Historical request not reusable under this budget')
    immutable(RUN / 'preflight.json', dict(signature=sid, passed=True, questions=128, sources=64,
        max_encoder_tokens=maximum, sentence_candidates=units, cached_control_conditions=1024,
        protected_final_data_accessed=False, new_training=False))
    print('RECOMP_HOTPOT_PREFLIGHT_OK', maximum, units, flush=True)


def select():
    sid = verify()
    model_files()
    import torch
    from transformers import AutoModel, AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    counter = ExactCounter()
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation='eager').to('cpu').eval()
    for index, source in enumerate(load(PLAN / 'public/source_inputs.json'), 1):
        dest = RUN / 'selected' / (source['question_id'].replace(':', '_') + '.json')
        if dest.exists():
            raise ValueError('Existing selection; no automatic selection rerun')
        encoded = [tok.encode(text, add_special_tokens=True, truncation=False)
                   for text in [source['question']] + [t for _, t in recomp.units(source)]]
        if max(map(len, encoded)) > 512:
            raise ValueError('No compressor truncation')
        chunks, started = [], time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(encoded), 32):
                batch = tok.pad(dict(input_ids=encoded[start:start+32]), padding=True, return_tensors='pt')
                hidden = model(**batch).last_hidden_state
                chunks.append(recomp.mean_pooling(torch, hidden, batch['attention_mask']).cpu())
        embedding = torch.cat(chunks)
        scores = {uid: float(embedding[0] @ embedding[j])
                  for j, (uid, _) in enumerate(recomp.units(source), 1)}
        plans = [hotpot_plan(source, scores, counter, budget) for budget in BUDGETS]
        immutable(dest, dict(signature=sid, source_sha256=fingerprint(source), scores=scores, plans=plans,
            costs=dict(seconds=time.perf_counter()-started, forward_calls=math.ceil(len(encoded)/32),
                encoder_items=len(encoded), input_tokens=sum(map(len, encoded))), private_inputs_read=False))
        write_json(RUN / 'selection_status.json', dict(completed=index, expected=128))
        print('HOTPOT_SELECT', index, '/128', flush=True)
    del model
    gc.collect()


def assemble():
    sid = verify()
    from tools.prepare_replacement_learning_v1 import ExactCounter
    from transformers import AutoTokenizer
    counter = ExactCounter()
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/Qwen3.5-9B'), local_files_only=True, trust_remote_code=False)
    sources = load(PLAN / 'public/source_inputs.json')
    plans = load(PLAN / 'public/baseline_plans.json')
    for source in sources:
        value = load(RUN / 'selected' / (source['question_id'].replace(':', '_') + '.json'))
        if value['signature'] != sid or value['source_sha256'] != fingerprint(source):
            raise ValueError('New score provenance changed')
        plans += value['plans']
    positions = {s['question_id']: i for i, s in enumerate(sources)}
    plans.sort(key=lambda p: (positions[p['question_id']], p['evidence_budget'], METHODS.index(p['method'])))
    expected = {(q, arm(m, b)) for q in positions for b in BUDGETS for m in METHODS}
    if len(plans) != 1536 or {(p['question_id'], p['arm']) for p in plans} != expected:
        raise ValueError('Incomplete three-method four-budget matrix')
    marker = load(PLAN / 'preparation.json')['reader_marker_sha256']
    cache, keys, maximum = replay_cache(), set(), 0
    for plan in plans:
        if counter.count(plan['evidence_text']) != plan['evidence_tokens'] or plan['evidence_tokens'] > plan['evidence_budget'] or fingerprint(plan['evidence_text']) != plan['evidence_sha256']:
            raise ValueError('Evidence budget or identity changed')
        messages = recomp.messages_for(plan)
        keys.add(recomp.interface.parent.request_key(messages, marker))
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        maximum = max(maximum, counter.count(rendered))
    new = len(keys - set(cache))
    if new > 512 or maximum + 2048 > 32768:
        raise ValueError('Reader call or prompt capacity budget exceeded')
    immutable(RUN / 'qa_plans.json', plans)
    immutable(RUN / 'selection_complete.json', dict(signature=sid, plans_sha256=sha(RUN / 'qa_plans.json'),
        conditions=1536, uncached_requests=new, full_prompt_max_tokens=maximum))
    print('ASSEMBLED_HOTPOT_BUDGET', new, 'new calls', flush=True)


def qa(endpoint):
    sid = verify()
    ledger = recomp.ledger
    old = ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for
    ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = RUN, lambda: sid, replay_cache, recomp.messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = old


def score():
    sid = verify()
    refs = load(PLAN / 'private/references.json')
    rows = recomp.interface.parent.score_rows(load(RUN / 'qa_plans.json'),
        list(read_jsonl(RUN / 'qa/predictions.jsonl')), refs, sid)
    summaries, comparisons = [], []
    pairs = [('beam_path_compress', 'recomp_hotpot_sentence'), ('beam_path_compress', 'recomp_sentence'),
             ('recomp_hotpot_sentence', 'recomp_sentence')]
    for budget in BUDGETS:
        for scope in ('overall', 'multi', 'single'):
            def by(method):
                return {r['question_id']: r for r in rows if r['method'] == method and r['evidence_budget'] == budget
                        and (scope == 'overall' or r['mode'] == scope)}
            for method in METHODS:
                selected = list(by(method).values())
                summaries.append(dict(method=method, budget=budget, scope=scope, n=len(selected),
                    f1_points=100*statistics.mean(r['local_native_f1'] for r in selected),
                    full_f1_count=sum(r['local_native_f1'] == 1 for r in selected),
                    mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in selected),
                    statuses=dict(Counter(r['status'] for r in selected))))
            for a, b in pairs:
                comparisons.append(dict(method=a, comparator=b, budget=budget, scope=scope,
                                        **recomp.interface.paired(by(a), by(b))))
    cache = replay_cache()
    for row in rows:
        if row['method'] != 'recomp_hotpot_sentence':
            entry = cache[row['request_key']]
            if not row['reused'] or any(row[k] != entry[k] for k in ('hypothesis', 'status', 'error')):
                raise ValueError('Frozen control output changed')
    calls = list(read_jsonl(RUN / 'qa/answer_calls.jsonl')) if (RUN / 'qa/answer_calls.jsonl').exists() else []
    if len(calls) > 512:
        raise ValueError('New inference cap exceeded')
    immutable(RUN / 'scored.json', rows)
    immutable(RUN / 'report.json', dict(signature=sid, complete=True, scores=summaries,
        comparisons=comparisons, new_reader_calls=len(calls), reused_conditions=sum(r['reused'] for r in rows),
        reader_cost={k: sum(c['result'].get('usage', {}).get(k, 0) for c in calls)
                     for k in ('prompt_tokens', 'completion_tokens')},
        new_reader_seconds=sum(c['seconds'] for c in calls),
        compressor_costs=[load(p)['costs'] for p in sorted((RUN / 'selected').glob('*.json'))],
        config=CONFIG, paper_ready=False, independent_confirmation=False, automatic_followup=False))
    print('HOTPOT_BUDGET_COMPLETE', [s for s in summaries if s['scope'] == 'overall'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'verify', 'preflight', 'select', 'assemble', 'qa', 'score'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8027')
    args = parser.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    else:
        value = globals()[args.mode]()
        if value:
            print(value, flush=True)
