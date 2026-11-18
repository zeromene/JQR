"""One bounded end-to-end conditional reranking test on predicted evidence."""
import argparse
from collections import Counter
import gc
import math
from pathlib import Path
import statistics
import time

from . import anchor_guard_controls_v1 as parent
from . import reranker_budget_v1 as ranker
from .replacement_precision_v2 import two_head_scores
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/conditional_rerank_execution_v1'
RUN = ROOT / 'runs/conditional_rerank_execution_v1'
load, sha, immutable, recomp = parent.load, parent.sha, parent.immutable, parent.recomp
BASE, MAIN, INDEP = parent.BASE, parent.MAIN, parent.INDEPENDENT
NEW_PATH, NEW_INDEP = 'conditional_path_fill', 'conditional_independent_fill'
METHODS = (BASE, MAIN, INDEP, NEW_PATH, NEW_INDEP)
NEW_METHODS = (NEW_PATH, NEW_INDEP)
PAIRS = ((NEW_PATH, MAIN), (NEW_PATH, BASE), (NEW_PATH, NEW_INDEP), (NEW_PATH, INDEP),
    (NEW_INDEP, INDEP), (NEW_INDEP, MAIN), (MAIN, BASE))
CONFIG = dict(version=1, sources=64, questions=128, budget=256, methods=list(METHODS),
    conditions=640, maximum_reader_calls=256, prefix='all first blind path paragraphs except candidate source',
    separator='\n\nPreviously retrieved evidence:\n', empty_prefix='reuse frozen original question scores',
    scoring='shared FP32 conditional sentence scores; BF16 SDPA backbone; TF32 disabled; batch1',
    model_revision=ranker.REVISION, maximum_ranker_tokens=8192, seed=30,
    reader='Qwen3.5-9B', reader_variant='format_only', output_tokens=2048, retries=0,
    length_min_mean_ratio=.95, minimum_overall_gain=1., minimum_multi_gain=2.,
    maximum_single_loss=1.5625, minimum_positive_multi_sources=3,
    new_training=False, independent_confirmation=False, protected_data_accessed=False,
    automatic_followup=False)


def arm(method):
    if method not in METHODS:
        raise ValueError('Unregistered arm')
    return method + '__b256'


def scoring_query(question, paragraphs, path, candidate_source):
    """Whitelist only question and public text; candidate cannot condition itself."""
    by = {p['source_id']: p for p in paragraphs}
    if not path or len(path) != len(set(path)) or not set(path) <= set(by):
        raise ValueError('Invalid predicted path')
    context = [by[p]['title'] + ': ' + by[p]['text'] for p in path if p != candidate_source]
    return question + CONFIG['separator'] + '\n\n'.join(context) if context else question


def input_tasks(source, path):
    for uid, document in ranker.documents(source):
        yield uid, scoring_query(source['question'], source['paragraphs'], path, uid.split(':')[0]), document


def new_plans(source, original, conditional, path, baseline, counter):
    parent.packing.sentence_order(source, original)
    parent.packing.sentence_order(source, conditional)
    seeds = parent.packing.independent_seeds(source, original, len(path))
    plans, traces = [], []
    for name, old_name, ids in [(NEW_PATH, MAIN, path), (NEW_INDEP, INDEP, seeds)]:
        candidate, trace = parent.packing.seed_fill(source, conditional, ids, counter, old_name,
            baseline['evidence_tokens'])
        result, protection = parent.guarded_plan(source, original, baseline, candidate,
            counter, 256, old_name)
        result.update(method=name, arm=arm(name), evidence_budget=256)
        plans.append(result)
        traces.append(dict(method=name, seed_ids=ids, packing=trace, protection=protection))
    return plans, traces


def freeze():
    previous = parent.verify()
    ranker.model_files()
    names = ['reader_memory/conditional_rerank_execution_v1.py',
        'tools/prepare_conditional_rerank_execution_v1.py', 'tools/conditional_rerank_execution_v1_remote.py',
        'scripts/conditional_rerank_execution_v1_queue.py', 'tests/test_conditional_rerank_execution_v1.py',
        'reader_memory/reranker_budget_v1.py', 'reader_memory/replacement_precision_v2.py',
        'research/anchor_guard_controls_v1/manifest.json',
        'models/Qwen3-Reranker-4B/artifact_manifest.json', 'models/Qwen3-Reranker-4B/.verified.json']
    names += list(load(PLAN / 'preparation.json')['server_source_files'])
    names += ['research/conditional_rerank_execution_v1/' + n for n in [
        'PROTOCOL.md', 'preparation.json', 'public/source_inputs.json', 'public/blind_paths.json',
        'public/base_scores.json', 'public/historical_plans.json', 'private/references.json',
        'private/replay_cache.json']]
    value = dict(config=CONFIG, parent_signature=previous, files={n: sha(ROOT / n) for n in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_CONDITIONAL_EXECUTION', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Frozen identity changed')
    if parent.verify() != value['parent_signature']:
        raise ValueError('Parent changed')
    for name, digest in value['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return value['signature']


def preflight():
    sid = verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    sources = load(PLAN / 'public/source_inputs.json')
    blind = load(PLAN / 'public/blind_paths.json')
    originals = load(PLAN / 'public/base_scores.json')
    old = load(PLAN / 'public/historical_plans.json')
    if len(sources) != 128 or len({s['native_id'] for s in sources}) != 64 or len(old) != 384:
        raise ValueError('Panel changed')
    by = {(p['question_id'], p['method']): p for p in old}
    if len(by) != 384 or set(by) != {(s['question_id'], m) for s in sources for m in METHODS[:3]}:
        raise ValueError('Incomplete original matrix')
    tok = AutoTokenizer.from_pretrained(str(ranker.MODEL), local_files_only=True, trust_remote_code=False)
    counter = ExactCounter()
    maximum, calls, reused = 0, 0, 0
    for s in sources:
        q = s['question_id']
        record, path = originals[q], blind[q]['seed_ids']
        if record['source_sha256'] != fingerprint(s) or blind[q]['source_sha256'] != fingerprint(s):
            raise ValueError('Source/path/score provenance changed')
        if len(path) != blind[q]['predicted_hops']:
            raise ValueError('Invalid predicted count')
        generated, _ = parent.plans_for(s, record['scores'], path, counter, 256)
        for p in generated:
            if p['method'] in METHODS[:3] and p != by[q, p['method']]:
                raise ValueError('Historical plan did not reproduce')
        for _, query, document in input_tasks(s, path):
            ids = ranker.rank_input_ids(tok, query, document)
            maximum = max(maximum, len(ids))
            if query == s['question']:
                reused += 1
            else:
                calls += 1
    prep = load(PLAN / 'preparation.json')
    if calls != prep['ranker_forward_cap'] or reused != prep['original_score_reuse_count']:
        raise ValueError('Scoring cap changed')
    marker = prep['reader_marker_sha256']
    if sha(ROOT / 'models/Qwen3.5-9B/.verified.json') != marker:
        raise ValueError('Reader changed')
    immutable(RUN / 'preflight.json', dict(signature=sid, questions=128, sources=64,
        historical_conditions=384, ranker_forward_cap=calls, original_score_reuse_count=reused,
        ranker_maximum_input_tokens=maximum, passed=True))
    print('CONDITIONAL_PREFLIGHT_OK', calls, 'new ranker forwards', flush=True)


def select():
    sid = verify()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if (RUN / 'selection_inflight.json').exists() or (RUN / 'scoring_complete.json').exists():
        raise ValueError('Existing/unknown scoring; no automatic repetition')
    torch.set_num_threads(8)
    torch.manual_seed(30)
    torch.cuda.manual_seed_all(30)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(ranker.MODEL), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(ranker.MODEL), local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    load_seconds = time.perf_counter() - started
    sources = load(PLAN / 'public/source_inputs.json')
    paths, originals = load(PLAN / 'public/blind_paths.json'), load(PLAN / 'public/base_scores.json')
    total_calls, total_tokens, total_seconds = 0, 0, 0.
    for index, source in enumerate(sources, 1):
        q = source['question_id']
        target = RUN / 'selected' / (q.replace(':', '_') + '.json')
        if target.exists():
            raise ValueError('Existing partial selection; no automatic rerun')
        write_json(RUN / 'selection_inflight.json', dict(signature=sid, question_id=q))
        scores, bf16, lengths, hashes = {}, {}, {}, {}
        calls, tokens, reused = 0, 0, 0
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.inference_mode():
            for uid, query, document in input_tasks(source, paths[q]['seed_ids']):
                ids = ranker.rank_input_ids(tok, query, document)
                lengths[uid], hashes[uid] = len(ids), fingerprint(ids)
                if query == source['question']:
                    scores[uid] = originals[q]['scores'][uid]
                    bf16[uid] = originals[q]['bf16_scores'][uid]
                    reused += 1
                else:
                    pair = two_head_scores(torch, tok, model, ids)
                    scores[uid], bf16[uid] = float(pair['fp32']), float(pair['bf16'])
                    calls += 1
                    tokens += len(ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        if not all(math.isfinite(v) for v in [*scores.values(), *bf16.values()]):
            raise ValueError('Nonfinite score')
        immutable(target, dict(signature=sid, source_sha256=fingerprint(source), scores=scores,
            bf16_scores=bf16, input_tokens=lengths, input_ids_sha256=hashes,
            costs=dict(forward_calls=calls, input_tokens=tokens, original_score_reuse=reused,
                seconds=elapsed), private_inputs_read=False))
        (RUN / 'selection_inflight.json').unlink()
        total_calls += calls
        total_tokens += tokens
        total_seconds += elapsed
        write_json(RUN / 'selection_status.json', dict(signature=sid, completed=index, expected=128,
            forward_calls=total_calls, input_tokens=total_tokens, seconds=total_seconds))
        print('CONDITIONAL_SELECT', index, '/128', calls, 'forwards', flush=True)
    if total_calls != load(RUN / 'preflight.json')['ranker_forward_cap']:
        raise ValueError('Scoring count mismatch')
    immutable(RUN / 'scoring_complete.json', dict(signature=sid, forward_calls=total_calls,
        input_tokens=total_tokens, seconds=total_seconds, load_seconds=load_seconds,
        shared_between_new_arms=True, private_inputs_read=False))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def replay_cache():
    marker = load(PLAN / 'preparation.json')['reader_marker_sha256']
    result = {}
    for item in load(PLAN / 'private/replay_cache.json'):
        recomp.interface.parent.check_cache_entry(item, marker)
        if item['key'] in result:
            raise ValueError('Duplicate cached request')
        result[item['key']] = item['entry']
    return result


def assemble():
    sid = verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter = ExactCounter()
    sources = load(PLAN / 'public/source_inputs.json')
    paths, originals = load(PLAN / 'public/blind_paths.json'), load(PLAN / 'public/base_scores.json')
    plans = load(PLAN / 'public/historical_plans.json')
    baseline = {p['question_id']: p for p in plans if p['method'] == BASE}
    traces, lengths = [], []
    for s in sources:
        q = s['question_id']
        record = load(RUN / 'selected' / (q.replace(':', '_') + '.json'))
        if record['signature'] != sid or record['source_sha256'] != fingerprint(s):
            raise ValueError('Conditional score provenance changed')
        new, trace = new_plans(s, originals[q]['scores'], record['scores'], paths[q]['seed_ids'], baseline[q], counter)
        plans.extend(new)
        traces.append(dict(question_id=q, traces=trace))
    for p in plans:
        target = baseline[p['question_id']]['evidence_tokens']
        if counter.count(p['evidence_text']) != p['evidence_tokens'] or p['evidence_tokens'] > target:
            raise ValueError('Budget contract changed')
        if fingerprint(p['evidence_text']) != p['evidence_sha256']:
            raise ValueError('Evidence changed')
        lengths.append(dict(question_id=p['question_id'], method=p['method'], mode=p['mode'],
            target=target, tokens=p['evidence_tokens'], ratio=p['evidence_tokens'] / target))
    length_statistics = []
    for method in METHODS:
        for scope in ('overall', 'multi', 'single'):
            group = [r for r in lengths if r['method'] == method and (scope == 'overall' or r['mode'] == scope)]
            item = dict(method=method, scope=scope, n=len(group), mean_tokens=statistics.mean(r['tokens'] for r in group),
                mean_ratio=statistics.mean(r['ratio'] for r in group), min_ratio=min(r['ratio'] for r in group))
            length_statistics.append(item)
    immutable(RUN / 'selection_traces.json', traces)
    immutable(RUN / 'length_audit.json', dict(rows=lengths, summaries=length_statistics))
    if any(s['method'] in NEW_METHODS and s['mean_ratio'] < .95 for s in length_statistics):
        raise ValueError('Fixed-version actual-length gate failed; stop without repacking')
    # Cache is not consulted until all public selections and length checks finish.
    cache = replay_cache()
    marker = load(PLAN / 'preparation.json')['reader_marker_sha256']
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/Qwen3.5-9B'), local_files_only=True, trust_remote_code=False)
    keys, maximum = set(), 0
    for p in plans:
        messages = recomp.messages_for(p)
        key = recomp.interface.parent.request_key(messages, marker)
        keys.add(key)
        if p['method'] not in NEW_METHODS and key not in cache:
            raise ValueError('Historical output must be reused')
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        maximum = max(maximum, counter.count(prompt))
    new_calls = len(keys - set(cache))
    if len(plans) != 640 or len({(p['question_id'], p['arm']) for p in plans}) != 640 or new_calls > 256 or maximum + 2048 > 32768:
        raise ValueError('Fixed matrix, reader cap or prompt capacity changed')
    immutable(RUN / 'qa_plans.json', plans)
    immutable(RUN / 'selection_complete.json', dict(signature=sid, conditions=640,
        qa_plans_sha256=sha(RUN / 'qa_plans.json'), uncached_requests=new_calls,
        maximum_full_prompt_tokens=maximum, length_gate_passed=True, before_any_new_answer=True))
    print('CONDITIONAL_ASSEMBLY_OK', new_calls, 'new reader calls', flush=True)


def qa(endpoint):
    sid = verify()
    if not load(RUN / 'selection_complete.json')['length_gate_passed']:
        raise ValueError('Missing length gate')
    ledger = recomp.ledger
    old = ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for
    ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = RUN, lambda: sid, replay_cache, recomp.messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = old


def adoption_gate(rows):
    def by(method, scope='overall'):
        return {r['question_id']: r for r in rows if r['method'] == method and (scope == 'overall' or r['mode'] == scope)}
    def delta(a, b, scope='overall'):
        x, y = by(a, scope), by(b, scope)
        return 100 * statistics.mean(x[q]['local_native_f1'] - y[q]['local_native_f1'] for q in x)
    normal = [r['local_native_f1'] - by(MAIN)[q]['local_native_f1'] for q, r in by(NEW_PATH).items()
        if r['status'] == by(MAIN)[q]['status'] == 'ok']
    positive_sources = {r['native_id'] for q, r in by(NEW_PATH, 'multi').items()
        if r['local_native_f1'] > by(MAIN, 'multi')[q]['local_native_f1'] + 1e-9}
    conditions = dict(overall_gain=delta(NEW_PATH, MAIN) >= 1., multi_gain=delta(NEW_PATH, MAIN, 'multi') >= 2.,
        single_limit=delta(NEW_PATH, MAIN, 'single') >= -1.5625,
        positive_multi_sources=len(positive_sources) >= 3,
        normal_pairs_positive=bool(normal) and statistics.mean(normal) > 0,
        beats_global=delta(NEW_PATH, BASE) > 0, beats_conditional_independent=delta(NEW_PATH, NEW_INDEP) > 0)
    return dict(passed=all(conditions.values()), conditions=conditions,
        delta_overall=delta(NEW_PATH, MAIN), delta_multi=delta(NEW_PATH, MAIN, 'multi'),
        delta_single=delta(NEW_PATH, MAIN, 'single'), positive_multi_sources=len(positive_sources),
        normal_pair_delta_points=100 * statistics.mean(normal) if normal else None,
        delta_vs_conditional_independent=delta(NEW_PATH, NEW_INDEP), automatic_followup=False,
        failure_action='close_this_modification_without_further_diagnostics', paper_ready=False)


def score():
    sid = verify()
    rows = recomp.interface.parent.score_rows(load(RUN / 'qa_plans.json'),
        list(read_jsonl(RUN / 'qa/predictions.jsonl')), load(PLAN / 'private/references.json'), sid)
    summaries, comparisons = [], []
    for scope in ('overall', 'multi', 'single'):
        def by(method):
            return {r['question_id']: r for r in rows if r['method'] == method and (scope == 'overall' or r['mode'] == scope)}
        for method in METHODS:
            group = list(by(method).values())
            summaries.append(dict(method=method, scope=scope, n=len(group),
                f1_points=100 * statistics.mean(r['local_native_f1'] for r in group),
                mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in group),
                statuses=dict(Counter(r['status'] for r in group))))
        for a, b in PAIRS:
            comparisons.append(dict(method=a, comparator=b, scope=scope, **recomp.interface.paired(by(a), by(b))))
    cache = replay_cache()
    for r in rows:
        if r['method'] not in NEW_METHODS:
            entry = cache[r['request_key']]
            if not r['reused'] or any(r[k] != entry[k] for k in ('hypothesis', 'status', 'error')):
                raise ValueError('Historical control output changed')
    calls_path = RUN / 'qa/answer_calls.jsonl'
    calls = list(read_jsonl(calls_path)) if calls_path.exists() else []
    if len(calls) != load(RUN / 'selection_complete.json')['uncached_requests']:
        raise ValueError('Reader call accounting mismatch')
    immutable(RUN / 'scored.json', rows)
    value = dict(signature=sid, complete=True, scores=summaries, comparisons=comparisons,
        adoption_gate=adoption_gate(rows), new_reader_calls=len(calls), reused_conditions=sum(r['reused'] for r in rows),
        ranker_cost=load(RUN / 'scoring_complete.json'),
        reader_cost={k: sum(c['result'].get('usage', {}).get(k, 0) for c in calls) for k in ('prompt_tokens', 'completion_tokens')},
        reader_seconds=sum(c['seconds'] for c in calls), automatic_followup=False, paper_ready=False)
    immutable(RUN / 'report.json', value)
    print('CONDITIONAL_EXECUTION_COMPLETE', [s for s in summaries if s['scope'] == 'overall'], value['adoption_gate'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'verify', 'preflight', 'select', 'assemble', 'qa', 'score'])
    p.add_argument('--endpoint', default='http://127.0.0.1:8035')
    a = p.parse_args()
    result = qa(a.endpoint) if a.mode == 'qa' else globals()[a.mode]()
    if result:
        print(result, flush=True)
