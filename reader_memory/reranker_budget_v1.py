"""A strong sentence reranker under unchanged four-budget reader contracts."""
import argparse
from collections import Counter
import gc
import math
from pathlib import Path
import statistics
import time

from . import recomp_hotpot_budget_v1 as parent
from . import mainline_reranker_v1 as semantic
from .replacement_precision_v2 import two_head_scores
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/reranker_budget_v1'
RUN = ROOT / 'runs/reranker_budget_v1'
MODEL = ROOT / 'models/Qwen3-Reranker-4B'
REVISION = '22e683669bc0f0bd69640a1354a6d0aebcfeede5'
NEW_METHOD = 'qwen_rerank_sentence'
BUDGETS = parent.BUDGETS
METHODS = parent.METHODS + (NEW_METHOD,)
load, sha, immutable, recomp = parent.load, parent.sha, parent.immutable, parent.recomp
CONFIG = dict(version='reranker-budget-v1', questions=128, sources=64,
    budgets=list(BUDGETS), methods=list(METHODS), max_conditions=2048,
    max_new_reader_calls=512, ranker_repository='Qwen/Qwen3-Reranker-4B',
    ranker_revision=REVISION, ranker_max_tokens=8192, ranker_batch=1,
    ranker_precision='BF16 SDPA backbone; same forward FP32 yes/no projection; TF32 disabled',
    ranker_score='FP32 yes-minus-no logit; retain BF16 logit differences as numerical audit',
    ranker_document='public source title + colon + original shared sentence; no IDs, gold or mode',
    ranker_instruction=semantic.CONFIG['instruction'],
    packing='identical original-sentence greedy packing and offset union as RECOMP',
    reader='Qwen3.5-9B', reader_variant='format_only', output_tokens=2048,
    decoding='greedy-thinking_false', retries=0, new_training=False,
    primary='256 path versus strong sentence ranking on exposed development data',
    secondary='512; report 1024 and2048 without omission', independent_confirmation=False,
    protected_final_data_accessed=False, automatic_followup=False)


def arm(method, budget):
    if method not in METHODS or budget not in BUDGETS:
        raise ValueError('Unregistered condition')
    return method + '__b' + str(budget)


def documents(source):
    """Only original titles and sentence text enter the unadapted ranker."""
    result = []
    for p in source['paragraphs']:
        for i, span in enumerate(p['spans']):
            if p['text'][span['start']:span['end']] != span['text']:
                raise ValueError('Shared sentence offset changed')
            result.append((p['source_id'] + ':span:' + str(i), p['title'] + ': ' + span['text']))
    if len({uid for uid, _ in result}) != len(result):
        raise ValueError('Duplicate source sentence')
    return result


def rank_input_ids(tok, question, document):
    body = f"<Instruct>: {CONFIG['ranker_instruction']}\n<Query>: {question}\n<Document>: {document}"
    ids = (tok.encode(semantic.PREFIX, add_special_tokens=False)
           + tok.encode(body, add_special_tokens=False)
           + tok.encode(semantic.SUFFIX, add_special_tokens=False))
    if len(ids) > CONFIG['ranker_max_tokens']:
        raise ValueError('Ranker input exceeds8192; no silent truncation')
    return ids


def sentence_plan(source, scores, counter, budget):
    arm(NEW_METHOD, budget)
    plan, _ = recomp.pack(source, scores, 'recomp_sentence', counter, budget)
    plan.update(method=NEW_METHOD, evidence_budget=budget, arm=arm(NEW_METHOD, budget))
    return plan


def model_files():
    manifest = load(MODEL / 'artifact_manifest.json')
    if (manifest['revision'] != REVISION or manifest['repository'] != CONFIG['ranker_repository']
            or not manifest['upstream_hashes_verified'] or not (MODEL / '.verified.json').exists()):
        raise ValueError('Pinned verified official ranker missing')
    for item in manifest['files']:
        path = MODEL / item['name']
        if path.stat().st_size != item['size'] or sha(path) != item['sha256']:
            raise ValueError('Ranker artifact changed: ' + item['name'])


def freeze():
    previous = parent.verify()
    model_files()
    names = ['reader_memory/reranker_budget_v1.py', 'tools/prepare_reranker_budget_v1.py',
        'scripts/reranker_budget_v1_queue.py', 'tests/test_reranker_budget_v1.py',
        'reader_memory/mainline_reranker_v1.py', 'reader_memory/replacement_precision_v2.py',
        'research/recomp_hotpot_budget_v1/manifest.json',
        'models/Qwen3-Reranker-4B/artifact_manifest.json', 'models/Qwen3-Reranker-4B/.verified.json']
    names += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
        and p.name not in {'manifest.json', 'CURRENT_STATUS.md', 'submission_receipt.json'}
        and 'snapshots' not in p.relative_to(PLAN).parts and p.suffix != '.pyc']
    value = dict(config=CONFIG, parent_signature=previous,
        files={name: sha(ROOT / name) for name in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_RERANKER_BUDGET', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Reranker baseline identity changed')
    if parent.verify() != value['parent_signature']:
        raise ValueError('Parent scientific identity changed')
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
    lengths = []
    for source in sources:
        parent.parent.parent.previous.parent.validate_source(source)
        lengths.extend(len(rank_input_ids(tok, source['question'], text)) for _, text in documents(source))
    plans = load(PLAN / 'public/baseline_plans.json')
    expected = {(s['question_id'], arm(m, b)) for s in sources for b in BUDGETS for m in parent.METHODS}
    if len(plans) != 1536 or {(p['question_id'], p['arm']) for p in plans} != expected:
        raise ValueError('Incomplete historical control matrix')
    marker = load(PLAN / 'preparation.json')['reader_marker_sha256']
    if sha(ROOT / 'models/Qwen3.5-9B/.verified.json') != marker:
        raise ValueError('Reader identity changed')
    cache = replay_cache()
    for plan in plans:
        key = recomp.interface.parent.request_key(recomp.messages_for(plan), marker)
        if (key not in cache or counter.count(plan['evidence_text']) != plan['evidence_tokens']
                or plan['evidence_tokens'] > plan['evidence_budget']):
            raise ValueError('Control not exactly reusable under budget')
    immutable(RUN / 'preflight.json', dict(signature=sid, passed=True, questions=128,
        sources=64, sentence_candidates=len(lengths), max_ranker_tokens=max(lengths),
        cached_control_conditions=1536, protected_final_data_accessed=False, new_training=False))
    print('RERANKER_BUDGET_PREFLIGHT_OK', max(lengths), len(lengths), flush=True)


def select():
    sid = verify()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    torch.set_num_threads(8)
    torch.manual_seed(30)
    torch.cuda.manual_seed_all(30)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    counter = ExactCounter()
    for index, source in enumerate(load(PLAN / 'public/source_inputs.json'), 1):
        target = RUN / 'selected' / (source['question_id'].replace(':', '_') + '.json')
        intent = RUN / 'selection_inflight.json'
        if target.exists() or intent.exists():
            raise ValueError('Existing or unknown selection; no automatic rerun')
        write_json(intent, dict(signature=sid, question_id=source['question_id']))
        scores, bf16, lengths = {}, {}, {}
        started = time.perf_counter()
        with torch.inference_mode():
            for uid, text in documents(source):
                ids = rank_input_ids(tok, source['question'], text)
                pair = two_head_scores(torch, tok, model, ids)
                scores[uid], bf16[uid] = float(pair['fp32']), float(pair['bf16'])
                lengths[uid] = len(ids)
        torch.cuda.synchronize()
        if not all(math.isfinite(x) for x in [*scores.values(), *bf16.values()]):
            raise ValueError('Nonfinite ranker output')
        elapsed = time.perf_counter() - started
        packing_started = time.perf_counter()
        plans = [sentence_plan(source, scores, counter, b) for b in BUDGETS]
        immutable(target, dict(signature=sid, source_sha256=fingerprint(source),
            scores=scores, bf16_scores=bf16, plans=plans, pair_input_tokens=lengths,
            precision_audit=dict(fp32_distinct=len(set(scores.values())), bf16_distinct=len(set(bf16.values())),
                maximum_logit_difference=max(abs(scores[u]-bf16[u]) for u in scores)),
            costs=dict(seconds=elapsed, packing_seconds=time.perf_counter()-packing_started,
                forward_calls=len(scores), input_tokens=sum(lengths.values())),
            private_inputs_read=False))
        intent.unlink()
        write_json(RUN / 'selection_status.json', dict(completed=index, expected=128))
        print('RERANKER_SELECT', index, '/128', len(scores), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


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
            raise ValueError('Ranker score provenance changed')
        plans += value['plans']
    positions = {s['question_id']: i for i, s in enumerate(sources)}
    plans.sort(key=lambda p: (positions[p['question_id']], p['evidence_budget'], METHODS.index(p['method'])))
    expected = {(q, arm(m, b)) for q in positions for b in BUDGETS for m in METHODS}
    if len(plans) != 2048 or {(p['question_id'], p['arm']) for p in plans} != expected:
        raise ValueError('Incomplete four-method matrix')
    marker = load(PLAN / 'preparation.json')['reader_marker_sha256']
    cache, keys, maximum = replay_cache(), set(), 0
    for plan in plans:
        if (counter.count(plan['evidence_text']) != plan['evidence_tokens']
                or plan['evidence_tokens'] > plan['evidence_budget']
                or fingerprint(plan['evidence_text']) != plan['evidence_sha256']):
            raise ValueError('Evidence budget or identity changed')
        messages = recomp.messages_for(plan)
        keys.add(recomp.interface.parent.request_key(messages, marker))
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        maximum = max(maximum, counter.count(rendered))
    new = len(keys - set(cache))
    if new > 512 or maximum + 2048 > 32768:
        raise ValueError('Reader call or prompt capacity cap exceeded')
    immutable(RUN / 'qa_plans.json', plans)
    immutable(RUN / 'selection_complete.json', dict(signature=sid, plans_sha256=sha(RUN / 'qa_plans.json'),
        conditions=2048, uncached_requests=new, full_prompt_max_tokens=maximum))
    print('ASSEMBLED_RERANKER_BUDGET', new, 'new calls', flush=True)


def qa(endpoint):
    sid = verify()
    ledger = recomp.ledger
    old = ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for
    ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = RUN, lambda: sid, replay_cache, recomp.messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = old


def summarize(rows):
    summaries, comparisons = [], []
    pairs = [('beam_path_compress', m) for m in METHODS if m != 'beam_path_compress']
    pairs.append((NEW_METHOD, 'recomp_hotpot_sentence'))
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
    return summaries, comparisons


def score():
    sid = verify()
    refs = load(PLAN / 'private/references.json')
    rows = recomp.interface.parent.score_rows(load(RUN / 'qa_plans.json'),
        list(read_jsonl(RUN / 'qa/predictions.jsonl')), refs, sid)
    summaries, comparisons = summarize(rows)
    cache = replay_cache()
    for row in rows:
        if row['method'] != NEW_METHOD:
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
        ranker_costs=[load(p)['costs'] for p in sorted((RUN / 'selected').glob('*.json'))],
        config=CONFIG, paper_ready=False, independent_confirmation=False, automatic_followup=False))
    print('RERANKER_BUDGET_COMPLETE', [s for s in summaries if s['scope'] == 'overall'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'verify', 'preflight', 'select', 'assemble', 'qa', 'score'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8028')
    args = parser.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    else:
        value = globals()[args.mode]()
        if value:
            print(value, flush=True)
