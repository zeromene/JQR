"""Controlled joint-action/stopping diagnostic; only score() reads references."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import urllib.error

from . import baseline_bootstrap_v2 as baseline
from . import evidence_acquisition_v1 as upstream
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/evidence_acquisition_v2'
PLAN = ROOT/'research/evidence_acquisition_v2'
ARMS = ('raw_2k', 'fusion_2k', 'raw_matched', 'fusion_matched',
        'singleton_stop', 'singleton_fill', 'bundle_stop', 'bundle_fill')
CONFIG = {'version': 'evidence-acquisition-v2', 'reader': 'Qwen3.5-9B',
    'arms': list(ARMS), 'evidence_budget': 2048, 'answer_max_tokens': 2048,
    'answer_calls_per_arm': 1, 'decoding': 'bf16-sdpa-greedy-enable_thinking_false',
    'shared_pool': 'frozen v1 bundle candidate IDs, raw/compact orders and candidate pairs',
    'embedding': 'CPU Qwen3-Embedding-0.6B; absolute local ST; 1024d; no query prompt',
    'proxy_weights': {k: upstream.CONFIG[k] for k in ['bundle_query_coverage_weight',
        'bundle_relevance_weight', 'bundle_pair_weight', 'gain_cost_exponent']},
    'singleton_objective': 'same pair-aware proxy as bundle; only joint actions disabled',
    'fill': 'preserve seed and append effective whole entries in fixed alternating order to maximal fit',
    'matched': 'per-question cap = bundle_stop evidence tokens before any answers; log occupancy gaps',
    'learned_policy': False, 'semantic_relations_verified': False, 'heldout_accessed': False,
    'purpose': 'post-v1 development diagnosis, not independent confirmation'}
COMPARISONS = [('bundle_fill', 'bundle_stop'), ('singleton_fill', 'singleton_stop'),
    ('bundle_fill', 'singleton_fill'), ('bundle_stop', 'singleton_stop'),
    ('bundle_fill', 'fusion_2k'), ('bundle_stop', 'raw_matched'),
    ('bundle_stop', 'fusion_matched'), ('fusion_2k', 'raw_2k')]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity():
    baseline.verify_inputs()
    original = load(upstream.RUN/'identity.json')
    if original['config'] != upstream.CONFIG:
        raise ValueError('Upstream configuration changed')
    for name, digest in original['code'].items():
        if baseline.sha(ROOT/name) != digest:
            raise ValueError('Upstream frozen dependency changed: '+name)
    base_id = load(baseline.RUN/'identity.json')
    if original['baseline_identity'] != fingerprint(base_id):
        raise ValueError('Upstream baseline identity mismatch')
    for name, digest in base_id['code'].items():
        if baseline.sha(ROOT/name) != digest:
            raise ValueError('Frozen baseline code changed: '+name)
    conversations = load(baseline.PLAN/'public/conversations.json')
    cids = [c['conversation_id'] for c in conversations]
    if len(cids) != 2 or len(set(cids)) != 2:
        raise ValueError('Expected the two frozen development conversations')
    for cid in cids:
        upstream.verify_plan(cid, fingerprint(original))
    paths = ['reader_memory/evidence_acquisition_v2.py', 'scripts/evidence_acquisition_v2_queue.py',
             'research/evidence_acquisition_v2/PLAN.md', 'reader_memory/tokenization.py']
    value = {'config': CONFIG, 'upstream_identity': fingerprint(original),
        'sources': {f'{cid}/{name}': baseline.sha(upstream.RUN/cid/name)
                    for cid in cids for name in ['manifest.json', 'plans.json', 'candidate_records.json']},
        'public': {p: baseline.sha(baseline.PLAN/p) for p in ['public/queries.json', 'public/conversations.json']},
        'code': {p: baseline.sha(ROOT/p) for p in paths}}
    target = RUN/'identity.json'
    if target.exists():
        if load(target) != value:
            raise ValueError('V2 identity changed; use a new version')
        return fingerprint(value)
    write_json(target, value)
    return fingerprint(value)


def exact(counter):
    if not counter.exact:
        raise ValueError('Exact reader tokenizer required')


def append_fill(seed, order, records, counter, budget):
    """Append only; no clipping, replacement, rewriting or invisible duplicates."""
    exact(counter)
    selected, trace = list(seed), []
    text = upstream.render(selected, records)
    if counter.count(text) > budget:
        raise ValueError('Seed exceeds hard budget')
    while True:
        changed = False
        for key in order:
            if key in selected:
                continue
            trial = selected+[key]
            new_text = upstream.render(trial, records)
            tokens = counter.count(new_text)
            if new_text != text and tokens <= budget:
                selected, text, changed = trial, new_text, True
                trace.append({'phase': 'fill', 'action': [key], 'evidence_tokens': tokens})
        if not changed:
            break
    return selected, trace


def proxy_pack(order, records, query, pairs, df, total, counter, budget, joint):
    """The same objective for both arms; joint only changes available actions."""
    exact(counter)
    selected, trace = [], []
    actions = [(k,) for k in order]
    if joint:
        actions += [tuple(p['ids']) for p in pairs]
    while True:
        old_text = upstream.render(selected, records)
        old_tokens = counter.count(old_text)
        old_value = upstream.proxy_value(selected, records, query, pairs, df, total)
        best = None
        for action in actions:
            trial = selected+[k for k in action if k not in selected]
            if trial == selected:
                continue
            gain = upstream.proxy_value(trial, records, query, pairs, df, total)-old_value
            if gain <= 1e-12:
                continue
            text = upstream.render(trial, records)
            if text == old_text:
                continue
            tokens = counter.count(text)
            if tokens > budget:
                continue
            rank = gain/max(1, tokens-old_tokens)**CONFIG['proxy_weights']['gain_cost_exponent']
            if best is None or rank > best[0]+1e-12:
                best = rank, trial, action, gain, tokens
        if best is None:
            break
        _, selected, action, gain, tokens = best
        trace.append({'phase': 'proxy', 'action': list(action), 'proxy_gain': gain,
                      'evidence_tokens': tokens})
    return selected, trace


def question_plans(query, source, records, scores, counter, sid):
    exact(counter)
    if set(query) != {'question_id', 'conversation_id', 'question', 'category'}:
        raise ValueError('Only frozen public question fields are permitted')
    pool, pairs = source['candidate_pool_ids'], source['candidate_pairs']
    if len(set(pool)) != len(pool) or set(scores) != set(pool):
        raise ValueError('Shared candidate/feature coverage mismatch')
    if any(k not in records or not math.isfinite(scores[k]) for k in pool):
        raise ValueError('Missing or nonfinite candidate feature')
    for pair in pairs:
        a, z = pair['ids']
        if a == z or a not in pool or z not in pool or pair['semantic_relation_verified'] is not False or \
           records[a]['kind'] != 'raw' or records[z]['kind'] != 'raw' or \
           set(records[a]['source_ids']) & set(records[z]['source_ids']):
            raise ValueError('Invalid shared candidate pair')
    current = {k: {**r, 'similarity': scores.get(k, 0.)} for k, r in records.items()}
    raw = [k for k in pool if current[k]['kind'] == 'raw']
    compact = [k for k in pool if current[k]['kind'] == 'compact']
    fusion = [k for pair in itertools.zip_longest(raw, compact) for k in pair if k]
    all_raw = [k for k, r in records.items() if r['kind'] == 'raw']
    df = Counter(t for k in all_raw for t in upstream.terms(records[k]['feature_text']))
    budget = CONFIG['evidence_budget']
    single, st = proxy_pack(pool, current, query['question'], pairs, df, len(all_raw), counter, budget, False)
    bundle, bt = proxy_pack(pool, current, query['question'], pairs, df, len(all_raw), counter, budget, True)
    target = counter.count(upstream.render(bundle, current))
    choices = {'singleton_stop': (single, st, budget, []), 'bundle_stop': (bundle, bt, budget, [])}
    for name, seed, trace in [('singleton_fill', single, st), ('bundle_fill', bundle, bt)]:
        selected, ft = append_fill(seed, fusion, current, counter, budget)
        choices[name] = selected, trace+ft, budget, seed
    for name, order, cap in [('raw_2k', raw, budget), ('fusion_2k', fusion, budget),
                             ('raw_matched', raw, target), ('fusion_matched', fusion, target)]:
        selected, ft = append_fill([], order, current, counter, cap)
        choices[name] = selected, ft, cap, []
    pool_id = fingerprint({'ids': pool, 'pairs': pairs, 'similarities': scores})
    plans = []
    for arm in ARMS:
        selected, trace, cap, seed = choices[arm]
        contexts = upstream.materialize(selected, current)
        text = upstream.render(selected, current)
        tokens = counter.count(text)
        if tokens > cap or cap > budget:
            raise ValueError('Evidence exceeds frozen cap')
        plans.append({**query, 'arm': arm, 'signature': sid, 'selected_ids': selected,
            'contexts': contexts, 'evidence_tokens': tokens, 'evidence_sha256': fingerprint(text),
            'evidence_budget': cap, 'matched_target_tokens': target, 'seed_selected_ids': seed,
            'shared_pool_id': pool_id, 'candidate_pool_ids': pool, 'candidate_pairs': pairs,
            'selection_trace': trace, 'source_ids': sorted({s for k in selected for s in records[k]['source_ids']}),
            'status': 'ready' if contexts else 'no_context', 'semantic_dependencies_verified': False})
    return plans


def source_plans(cid):
    sid = fingerprint(load(upstream.RUN/'identity.json'))
    return {p['question_id']: p for p in upstream.verify_plan(cid, sid) if p['arm'] == 'bundle_proxy'}


def prepare(cid):
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sid = identity()
    out = RUN/cid
    if (out/'manifest.json').exists():
        verify_plan(cid, sid)
        print('ALREADY_PREPARED', cid, flush=True)
        return
    started = time.perf_counter()
    sources = source_plans(cid)
    queries = [q for q in load(baseline.PLAN/'public/queries.json') if q['conversation_id'] == cid]
    if set(sources) != {q['question_id'] for q in queries}:
        raise ValueError('Source question coverage mismatch')
    records = load(upstream.RUN/cid/'candidate_records.json')
    features_path = out/'features.json'
    if features_path.exists():
        features = load(features_path)
        if features['signature'] != sid:
            raise ValueError('Embedding cache signature mismatch')
    else:
        import numpy as np
        import torch
        torch.set_num_threads(8)
        sys.path.insert(0, str(baseline.OFFICIAL))
        from simplemem.core.utils.embedding import EmbeddingModel
        embedding = EmbeddingModel(str((ROOT/'models/Qwen3-Embedding-0.6B').resolve()))
        if str(embedding.model.device) != 'cpu' or embedding.dimension != 1024 or \
           embedding.model_type != 'sentence_transformer':
            raise ValueError('Unexpected embedding configuration')
        ids = list(records)
        print('ENCODING_CANDIDATES_CPU', cid, len(ids), flush=True)
        vectors = np.asarray(embedding.encode([records[k]['embedding_text'] for k in ids]))
        qvectors = embedding.encode([q['question'] for q in queries], is_query=False)
        positions = {k: i for i, k in enumerate(ids)}
        similarities = {}
        for q, vector in zip(queries, qvectors):
            sim = vectors @ np.asarray(vector)
            similarities[q['question_id']] = {k: float(sim[positions[k]])
                for k in sources[q['question_id']]['candidate_pool_ids']}
        features = {'signature': sid, 'similarities': similarities,
                    'embedding_device': 'cpu', 'embedding_dimension': 1024,
                    'embedding_wall_seconds': time.perf_counter()-started}
        write_json(features_path, features)
    from .tokenization import TokenCounter
    counter = TokenCounter(ROOT/'models/Qwen3.5-9B')
    plans = []
    for q in queries:
        plans.extend(question_plans(q, sources[q['question_id']], records,
                                   features['similarities'][q['question_id']], counter, sid))
        print('PREPARED_CONTROLLED', q['question_id'], len(ARMS), flush=True)
    write_json(out/'plans.json', plans)
    old_matches = sum(p['evidence_sha256'] == sources[p['question_id']]['evidence_sha256']
                      for p in plans if p['arm'] == 'bundle_stop')
    write_json(out/'manifest.json', {'signature': sid, 'conversation_id': cid,
        'plan_sha256': baseline.sha(out/'plans.json'), 'features_sha256': baseline.sha(features_path),
        'source_manifest_sha256': baseline.sha(upstream.RUN/cid/'manifest.json'),
        'questions': len(queries), 'plans': len(plans), 'cpu_wall_seconds': time.perf_counter()-started,
        'embedding_device': 'cpu', 'counter_exact': counter.exact, 'private_inputs_read': False,
        'bundle_stop_contexts_matching_v1': old_matches,
        'mean_evidence_tokens': {a: statistics.mean(p['evidence_tokens'] for p in plans if p['arm'] == a)
                                 for a in ARMS},
        'matched_mean_gap_tokens': {a: statistics.mean(p['matched_target_tokens']-p['evidence_tokens']
            for p in plans if p['arm'] == a) for a in ['raw_matched', 'fusion_matched']}})


def verify_plan(cid, sid):
    out = RUN/cid
    manifest = load(out/'manifest.json')
    if manifest['signature'] != sid or manifest['plan_sha256'] != baseline.sha(out/'plans.json') or \
       manifest['features_sha256'] != baseline.sha(out/'features.json') or \
       manifest['source_manifest_sha256'] != baseline.sha(upstream.RUN/cid/'manifest.json'):
        raise ValueError('Controlled plan/source/features changed')
    features = load(out/'features.json')
    if features['signature'] != sid:
        raise ValueError('Feature identity mismatch')
    plans = load(out/'plans.json')
    sources = source_plans(cid)
    queries = {q['question_id']: q for q in load(baseline.PLAN/'public/queries.json')
               if q['conversation_id'] == cid}
    if set(features['similarities']) != set(queries):
        raise ValueError('Feature question coverage mismatch')
    expected = {(q, a) for q in queries for a in ARMS}
    if len(plans) != len(expected) or {(p['question_id'], p['arm']) for p in plans} != expected:
        raise ValueError('Controlled action coverage mismatch')
    records = load(upstream.RUN/cid/'candidate_records.json')
    by_key = {(p['question_id'], p['arm']): p for p in plans}
    for p in plans:
        qid, arm = p['question_id'], p['arm']
        source = sources[qid]
        pool = source['candidate_pool_ids']
        score = features['similarities'][qid]
        pool_id = fingerprint({'ids': pool, 'pairs': source['candidate_pairs'], 'similarities': score})
        if p['signature'] != sid or any(p[k] != v for k, v in queries[qid].items()) or \
           p['candidate_pool_ids'] != pool or p['candidate_pairs'] != source['candidate_pairs'] or \
           p['shared_pool_id'] != pool_id or set(score) != set(pool):
            raise ValueError('Question/shared pool identity mismatch')
        selected = p['selected_ids']
        if len(set(selected)) != len(selected) or not set(selected) <= set(pool):
            raise ValueError('Selection outside shared pool or duplicate IDs')
        if arm.startswith('raw_') and any(records[k]['kind'] != 'raw' for k in selected):
            raise ValueError('Raw control contains compact evidence')
        target = by_key[qid, 'bundle_stop']['evidence_tokens']
        cap = target if arm.endswith('_matched') else CONFIG['evidence_budget']
        if p['matched_target_tokens'] != target or p['evidence_budget'] != cap or \
           not 0 <= p['evidence_tokens'] <= cap <= CONFIG['evidence_budget']:
            raise ValueError('Invalid matched target/hard budget')
        if p['contexts'] != upstream.materialize(selected, records):
            raise ValueError('Evidence differs from frozen source records')
        if p['source_ids'] != sorted({s for k in selected for s in records[k]['source_ids']}):
            raise ValueError('Quoted source IDs differ from selected records')
        text = upstream.render(selected, records)
        if p['evidence_sha256'] != fingerprint(text):
            raise ValueError('Serialized evidence changed')
        if arm.endswith('_fill'):
            seed = by_key[qid, arm.replace('_fill', '_stop')]['selected_ids']
            if p['seed_selected_ids'] != seed or selected[:len(seed)] != seed or \
               not text.startswith(upstream.render(seed, records)):
                raise ValueError('Fill changed or removed seed evidence')
    return plans


def all_plans(sid):
    return [p for c in load(baseline.PLAN/'public/conversations.json')
            for p in verify_plan(c['conversation_id'], sid)]


def parse_answer(text):
    sys.path.insert(0, str(baseline.OFFICIAL))
    from simplemem.core.utils.llm_client import LLMClient
    parsed = LLMClient.__new__(LLMClient).extract_json(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get('answer'), str) or not parsed['answer'].strip():
        raise ValueError('Answer JSON missing nonempty answer string')
    return parsed['answer']


def check_logs(rows, calls, plans, sid, require_committed=False):
    done = upstream.validate_predictions(rows, plans, sid)
    keys = [(c['question_id'], c['arm']) for c in calls]
    expected = {(p['question_id'], p['arm']) for p in plans}
    if len(set(keys)) != len(keys) or not set(keys) <= expected or any(c['signature'] != sid for c in calls):
        raise ValueError('Invalid call identity/duplicates/coverage')
    if set(keys)-done:
        raise ValueError('Uncommitted inference call; refusing automatic retry')
    if require_committed and set(keys) != {(r['question_id'], r['arm']) for r in rows if r['contexts']}:
        raise ValueError('Cost log differs from attempted answers')
    return done


def run(endpoint):
    sid = identity()
    plans = all_plans(sid)
    previous = list(read_jsonl(RUN/'predictions.jsonl')) if (RUN/'predictions.jsonl').exists() else []
    calls = list(read_jsonl(RUN/'calls.jsonl')) if (RUN/'calls.jsonl').exists() else []
    done = check_logs(previous, calls, plans, sid, require_committed=True)
    intent = RUN/'inflight.json'
    if intent.exists():
        pending = load(intent)
        if pending['signature'] != sid or (pending['question_id'], pending['arm']) not in done:
            raise ValueError('Uncommitted inference intent; refusing an invisible retry')
        intent.unlink()
    for p in plans:
        key = p['question_id'], p['arm']
        if key in done:
            continue
        write_json(RUN/'status.json', {'status': 'running', 'phase': 'answer', 'question_id': key[0],
            'arm': key[1], 'completed_predictions': len(done), 'expected_predictions': len(plans), 'signature': sid})
        text = upstream.FORMAT(None, [SimpleNamespace(**v) for v in p['contexts']])
        tokens = baseline.rpc(endpoint, '/tokenize', {'text': text})['tokens']
        if tokens != p['evidence_tokens'] or tokens > p['evidence_budget']:
            raise ValueError('Server tokenizer differs from frozen plan/budget')
        started = time.perf_counter()
        answer, status, error, result, transport = '', 'no_context', None, None, False
        if p['contexts']:
            messages = [{'role': 'system', 'content': 'You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format.'},
                        {'role': 'user', 'content': upstream.PROMPT(None, p['question'], text)}]
            call = {'signature': sid, 'question_id': key[0], 'arm': key[1], 'messages': messages,
                    'max_tokens': CONFIG['answer_max_tokens']}
            write_json(intent, {'signature': sid, 'question_id': key[0], 'arm': key[1],
                                'request_fingerprint': fingerprint(call)})
            try:
                result = baseline.rpc(endpoint, '/complete', {'messages': messages,
                    'max_tokens': CONFIG['answer_max_tokens']})
                answer = parse_answer(baseline.require_complete(result))
                status = 'ok'
            except Exception as exc:
                error, status = f'{type(exc).__name__}: {exc}', 'failed'
                transport = isinstance(exc, (OSError, urllib.error.URLError))
            call.update(result or {})
            call.update(status=status, error=error, wall_seconds=time.perf_counter()-started)
            baseline.append(RUN/'calls.jsonl', call)
        baseline.append(RUN/'predictions.jsonl', {**p, 'hypothesis': answer, 'status': status,
            'error': error, 'seconds': time.perf_counter()-started, 'generation_calls': int(bool(p['contexts']))})
        if intent.exists():
            intent.unlink()
        done.add(key)
        print('CONTROLLED_CHECKPOINT', len(done), '/', len(plans), *key, status, flush=True)
        if transport:
            raise RuntimeError('Inference service unavailable; failure recorded; remaining actions not attempted')
    rows = list(read_jsonl(RUN/'predictions.jsonl'))
    if upstream.validate_predictions(rows, plans, sid) != {(p['question_id'], p['arm']) for p in plans}:
        raise ValueError('Final answer coverage mismatch')
    failures = sum(r['status'] != 'ok' for r in rows)
    write_json(RUN/'status.json', {'status': 'completed_with_failures' if failures else 'completed',
        'signature': sid, 'completed_predictions': len(rows), 'expected_predictions': len(plans),
        'failed_or_empty_predictions': failures, 'heldout_accessed': False})


def score():
    baseline.verify_inputs(private=True)
    sid = identity()
    plans = all_plans(sid)
    rows = list(read_jsonl(RUN/'predictions.jsonl'))
    calls = list(read_jsonl(RUN/'calls.jsonl')) if (RUN/'calls.jsonl').exists() else []
    done = check_logs(rows, calls, plans, sid, require_committed=True)
    refs = {r['question_id']: r for r in load(baseline.PLAN/'private/references.json')}
    expected = {(q, a) for q in refs for a in ARMS}
    complete = done == expected
    scored, groups = [], defaultdict(list)
    for r in rows:
        if r['category'] != refs[r['question_id']]['category']:
            raise ValueError('Reference category mismatch')
        f1 = baseline.native_f1(r['hypothesis'], refs[r['question_id']]['answer'], r['category'])
        value = {**r, 'official_native_f1': f1, 'reference': refs[r['question_id']]['answer']}
        scored.append(value)
        for scope in ['overall', 'category:'+str(r['category']), 'conversation:'+r['conversation_id']]:
            groups[r['arm'], scope].append(value)
    scores = [{'arm': a, 'scope': scope, 'n': len(v),
        'native_f1': statistics.mean(r['official_native_f1'] for r in v),
        'mean_evidence_tokens': statistics.mean(r['evidence_tokens'] for r in v),
        'mean_cap_tokens': statistics.mean(r['evidence_budget'] for r in v),
        'statuses': dict(Counter(r['status'] for r in v))} for (a, scope), v in sorted(groups.items())]
    lookup = {(r['question_id'], r['arm']): r for r in scored}
    paired, interaction = {}, None
    if complete:
        for a, z in COMPARISONS:
            ds = [lookup[q, a]['official_native_f1']-lookup[q, z]['official_native_f1'] for q in refs]
            paired[a+'_minus_'+z] = {'n': len(ds), 'native_f1_delta': statistics.mean(ds),
                'wins': sum(d > 1e-9 for d in ds), 'ties': sum(abs(d) <= 1e-9 for d in ds),
                'losses': sum(d < -1e-9 for d in ds)}
        interaction = (paired['bundle_fill_minus_bundle_stop']['native_f1_delta']-
                       paired['singleton_fill_minus_singleton_stop']['native_f1_delta'])
    matching = {}
    for arm in ['raw_matched', 'fusion_matched']:
        rr = [r for r in rows if r['arm'] == arm]
        gaps = [r['matched_target_tokens']-r['evidence_tokens'] for r in rr]
        if gaps:
            matching[arm] = {'n': len(gaps), 'mean_target_minus_actual_tokens': statistics.mean(gaps),
                'max_gap_tokens': max(gaps), 'fraction_gap_le_64': sum(g <= 64 for g in gaps)/len(gaps),
                'fraction_gap_le_128': sum(g <= 128 for g in gaps)/len(gaps), 'exact_cost_match_claim': False}
    costs = {a: {'calls': sum(c['arm'] == a for c in calls),
        'failed_calls': sum(c['arm'] == a and c['status'] != 'ok' for c in calls),
        'wall_seconds': sum(c['wall_seconds'] for c in calls if c['arm'] == a),
        **{k: sum(c.get('usage', {}).get(k, 0) for c in calls if c['arm'] == a)
           for k in ['prompt_tokens', 'completion_tokens']}} for a in ARMS}
    write_json(RUN/'scored.json', scored)
    report = {'signature': sid, 'complete': complete, 'observed_predictions': len(rows),
        'expected_predictions': len(expected), 'all_answers_successful': complete and all(r['status'] == 'ok' for r in rows),
        'failed_or_empty_predictions': sum(r['status'] != 'ok' for r in rows), 'scores': scores,
        'paired_comparisons': paired, 'fill_effect_interaction': interaction, 'matched_occupancy': matching,
        'category_macro_f1': {a: statistics.mean(s['native_f1'] for s in scores
            if s['arm'] == a and s['scope'].startswith('category:')) for a in ARMS} if complete else None,
        'costs': costs, 'heldout_accessed': False, 'semantic_relations_verified': False,
        'scope': CONFIG['purpose']+'; 64 questions in two source histories; no significance claim',
        'cost_scope': 'Fresh answer calls; upstream writer/index reuse and CPU preparation separate; no end-to-end efficiency claim'}
    write_json(RUN/'report.json', report)
    print(json.dumps([s for s in scores if s['scope'] == 'overall']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'run', 'score'])
    parser.add_argument('--conversation-id', choices=['conv-43', 'conv-44'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8004')
    args = parser.parse_args()
    if args.mode in {'prepare', 'verify'} and not args.conversation_id:
        parser.error('--conversation-id is required')
    if args.mode == 'prepare':
        prepare(args.conversation_id)
    elif args.mode == 'verify':
        from .tokenization import TokenCounter
        counter = TokenCounter(ROOT/'models/Qwen3.5-9B')
        plans = verify_plan(args.conversation_id, identity())
        for p in plans:
            text = upstream.FORMAT(None, [SimpleNamespace(**e) for e in p['contexts']])
            if counter.count(text) != p['evidence_tokens']:
                raise ValueError('CPU serialized token count mismatch')
        print('VERIFIED_CONTROLLED_PLANS', len(plans), flush=True)
    elif args.mode == 'run':
        run(args.endpoint)
    else:
        score()


if __name__ == '__main__':
    main()
