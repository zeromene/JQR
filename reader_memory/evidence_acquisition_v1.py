"""Frozen 2K action pilot: raw, compact, fusion, source expansion, bundle proxy.

V1 tests the action space. Its conditional-coverage heuristic is NOT a learned or
calibrated information-value estimator. Topic/scope links are retrieval hypotheses,
not entailment, causality, temporal validity, or exact memory-to-source alignment.
Only score() reads private references. Preparation and inference are gold-blind.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import textwrap
import time
from types import SimpleNamespace

from .schema import fingerprint, read_jsonl, write_json
from . import baseline_bootstrap_v2 as baseline

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/evidence_acquisition_v1'
ARMS = ('raw_dense', 'compact_memory', 'static_fusion', 'source_expansion', 'bundle_proxy')
CONFIG = {'version': 'evidence-acquisition-v1', 'reader': 'Qwen3.5-9B',
    'evidence_budget': 2048, 'answer_max_tokens': 2048, 'answer_calls_per_arm': 1,
    'decoding': 'bf16-sdpa-greedy-enable_thinking_false',
    'arms': list(ARMS), 'raw_top_k': 20, 'compact_top_k': 25,
    'expansion_seed_memories': 6, 'expansion_raw_limit': 20,
    'bundle_raw_limit': 24, 'bundle_pair_limit': 16,
    'bundle_query_coverage_weight': 1.0, 'bundle_relevance_weight': .25,
    'bundle_pair_weight': .25, 'gain_cost_exponent': .7,
    'relation_contract': 'shared-topic-and-scope candidate; no entailment claim',
    'selector': 'untrained conditional-coverage proxy; action-space development pilot',
    'storage': 'exact original quotes plus existing compact views; no fact rewriting',
    'heldout_accessed': False, 'same_budget_not_equal_occupancy': True}
STOP = set(('a an the i me my we our you your he him his she her they their it its '
    'is am are was were be been being do does did have has had and or but if to of '
    'in on at for from with as by about this that these those what which when where '
    'who why how would could should can will just really very much also some any '
    'john tim hey hi good great thanks thank know like going get got said stated '
    'mentioned asked tell').split())
SCOPE = re.compile(r'\b(?:not|never|only|unless|except|instead|before|after|since|'
    r'currently|previously|now|changed|switched|moved|last|next|during|without)\b', re.I)


def terms(text):
    return set(re.findall(r'[a-z0-9]+', text.lower())) - STOP


def author_method(name):
    path = ROOT / 'third_party/simplemem_official/simplemem/core/answer_generator.py'
    text = path.read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'AnswerGenerator')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {}
    exec(compile('from __future__ import annotations\n' +
        textwrap.dedent(ast.get_source_segment(text, node)), str(path), 'exec'), env)
    return env[name]


FORMAT = author_method('_format_contexts')
PROMPT = author_method('_build_answer_prompt')


def memory_entry(entry_id, content, **metadata):
    result = dict(entry_id=entry_id, lossless_restatement=content, keywords=[],
        timestamp=None, location=None, persons=[], entities=[], topic=None)
    result.update(metadata)
    return result


def materialize(selected, records):
    """Source union: a turn is quoted once; exact compact duplicates are skipped."""
    result, sources, compact_texts = [], set(), set()
    for key in selected:
        r = records[key]
        if r['kind'] == 'raw':
            fresh = [t for t in r['turns'] if t['source_id'] not in sources]
            if not fresh:
                continue
            sources.update(t['source_id'] for t in fresh)
            text = '\n'.join(f"[{t['source_id']}] [{t['timestamp']}] {t['speaker']}: {t['content']}" for t in fresh)
            result.append(memory_entry(key, text))
        else:
            text = r['entry']['lossless_restatement'].strip()
            if text in compact_texts:
                continue
            compact_texts.add(text)
            result.append(r['entry'])
    return result


def render(selected, records):
    return FORMAT(None, [SimpleNamespace(**e) for e in materialize(selected, records)])


def fits(selected, records, counter, budget):
    if not counter.exact:
        raise ValueError('Exact reader tokenizer required')
    return counter.count(render(selected, records)) <= budget


def ordered_pack(order, records, counter, budget):
    selected = []
    for key in order:
        if key not in selected and fits(selected + [key], records, counter, budget):
            selected.append(key)
    return selected


def candidate_pairs(order, records, df, total):
    """Only literal quote packets. Shared words/cues suggest relevance, not truth."""
    pairs = []
    for a, b in itertools.combinations(order[:CONFIG['bundle_raw_limit']], 2):
        ra, rb = records[a], records[b]
        if set(ra['source_ids']) & set(rb['source_ids']):
            continue
        shared = {t for t in terms(ra['feature_text']) & terms(rb['feature_text'])
                  if len(t) >= 4 and df.get(t, total) <= .35 * total}
        if not shared or not (SCOPE.search(ra['feature_text']) or SCOPE.search(rb['feature_text'])):
            continue
        score = ra['similarity'] + rb['similarity']
        pairs.append((score, a, b, sorted(shared)))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    return [{'ids': [a, b], 'anchors': anchors,
             'relation': 'shared-topic-and-scope-candidate', 'semantic_relation_verified': False}
            for _, a, b, anchors in pairs[:CONFIG['bundle_pair_limit']]]


def proxy_value(selected, records, query, pairs, df, total):
    query_terms = terms(query)
    seen = set().union(*(terms(records[k]['feature_text']) for k in selected)) if selected else set()
    weights = {t: math.log(1 + total/(1 + df.get(t, 0))) for t in query_terms}
    coverage = sum(w for t, w in weights.items() if t in seen) / max(1e-9, sum(weights.values()))
    scores = sorted((max(0., records[k]['similarity']) for k in selected), reverse=True)[:4]
    relevance = sum(scores)/4
    # An explicit pair term makes the proxy non-additive. It is not measured QA gain.
    completed = sum(set(p['ids']) <= set(selected) for p in pairs)
    synergy = min(2, completed)/2
    return (CONFIG['bundle_query_coverage_weight']*coverage +
            CONFIG['bundle_relevance_weight']*relevance + CONFIG['bundle_pair_weight']*synergy)


def bundle_pack(order, records, query, pairs, df, total, counter, budget):
    if not counter.exact:
        raise ValueError('Exact reader tokenizer required')
    selected, trace = [], []
    actions = [(k,) for k in order] + [tuple(p['ids']) for p in pairs]
    while True:
        old_tokens = counter.count(render(selected, records))
        old_value = proxy_value(selected, records, query, pairs, df, total)
        best = None
        for action in actions:
            trial = selected + [k for k in action if k not in selected]
            if trial == selected:
                continue
            gain = proxy_value(trial, records, query, pairs, df, total) - old_value
            if gain <= 1e-12:
                continue
            trial_text = render(trial, records)
            if trial_text == render(selected, records):
                continue
            tokens = counter.count(trial_text)
            if tokens > budget:
                continue
            rank = gain / max(1, tokens-old_tokens)**CONFIG['gain_cost_exponent']
            if gain > 1e-12 and (best is None or rank > best[0] + 1e-12):
                best = rank, trial, action, gain, tokens
        if best is None:
            break
        _, selected, action, gain, tokens = best
        trace.append({'action': list(action), 'proxy_gain': gain, 'evidence_tokens': tokens})
    return selected, trace


def identity():
    baseline.verify_inputs()
    base_id = json.loads((baseline.RUN / 'identity.json').read_text(encoding='utf-8'))
    for name, digest in base_id['code'].items():
        if baseline.sha(ROOT/name) != digest:
            raise ValueError('Frozen upstream code changed: ' + name)
    paths = ['reader_memory/evidence_acquisition_v1.py', 'scripts/evidence_acquisition_v1_queue.py',
             'reader_memory/tokenization.py']
    value = {'config': CONFIG, 'baseline_identity': fingerprint(base_id),
        'public_inputs': {p: baseline.sha(baseline.PLAN / p) for p in
            ['public/queries.json', 'public/conversations.json']},
        'code': {p: baseline.sha(ROOT / p) for p in paths},
        'official_revision': baseline.check_sources()}
    target = RUN / 'identity.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != value:
        raise ValueError('Pilot identity changed; use a new version/directory')
    write_json(target, value)
    return fingerprint(value)


def prepare(conversation_id):
    """CPU-only. No dataset file, private reference, or model answer is read."""
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sid = identity()
    out = RUN / conversation_id
    if (out / 'manifest.json').exists():
        verify_plan(conversation_id, sid)
        print('ALREADY_PREPARED', conversation_id, flush=True)
        return
    started = time.perf_counter()
    memories_path = baseline.RUN / conversation_id / 'memories.json'
    memories = json.loads(memories_path.read_text(encoding='utf-8'))
    base_signature = fingerprint(json.loads((baseline.RUN/'identity.json').read_text(encoding='utf-8')))
    if memories['signature'] != base_signature:
        raise ValueError('Memory signature mismatch')
    conversations = json.loads((baseline.PLAN / 'public/conversations.json').read_text(encoding='utf-8'))
    conv = next(c for c in conversations if c['conversation_id'] == conversation_id)
    queries = [q for q in json.loads((baseline.PLAN/'public/queries.json').read_text(encoding='utf-8'))
               if q['conversation_id'] == conversation_id]
    from .tokenization import TokenCounter
    counter = TokenCounter(ROOT / 'models/Qwen3.5-9B')
    sys.path.insert(0, str(baseline.OFFICIAL))
    from simplemem.core.utils.embedding import EmbeddingModel
    import numpy as np
    import torch
    torch.set_num_threads(8)
    embedding = EmbeddingModel(str((ROOT/'models/Qwen3-Embedding-0.6B').resolve()))
    if str(embedding.model.device) != 'cpu' or embedding.dimension != 1024:
        raise ValueError('Preparation must use the pinned 1024-d CPU embedding')
    records, raw_ids, memory_ids = {}, [], []
    for start in range(0, len(conv['turns']), 3):
        turns = conv['turns'][start:start+4]
        key = f'{conversation_id}:raw{start:04d}'
        records[key] = {'kind': 'raw', 'turns': turns, 'source_ids': [t['source_id'] for t in turns],
            'feature_text': ' '.join(t['content'] for t in turns),
            'embedding_text': '\n'.join(f"[{t['timestamp']}] {t['speaker']}: {t['content']}" for t in turns)}
        raw_ids.append(key)
    windows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(
        (baseline.RUN/conversation_id/'windows').glob('*.json'))]
    origin = defaultdict(set)
    for w in windows:
        if w['signature'] != base_signature:
            raise ValueError('Writer window signature mismatch')
        for e in w['entries']:
            origin[e['entry_id']].update(w['source_ids'])
    for e in memories['entries']:
        key = e['entry_id']
        if key not in origin:
            raise ValueError('Memory has no candidate writer-window origin')
        if key in records:
            raise ValueError('Candidate ID collision')
        records[key] = {'kind': 'compact', 'entry': e, 'source_ids': [],
            'candidate_window_source_ids': sorted(origin[key]),
            'source_relation': 'window-candidate-only; not exact factual provenance',
            'feature_text': e['lossless_restatement'], 'embedding_text': e['lossless_restatement']}
        memory_ids.append(key)
    all_ids = raw_ids + memory_ids
    vectors = embedding.encode([records[k]['embedding_text'] for k in all_ids])
    qvectors = embedding.encode([q['question'] for q in queries], is_query=False)
    positions = {key: i for i, key in enumerate(all_ids)}
    df = Counter(t for k in raw_ids for t in terms(records[k]['feature_text']))
    total = len(raw_ids)
    plans = []
    for q, qvector in zip(queries, qvectors):
        similarities = np.asarray(vectors) @ np.asarray(qvector)
        current = {k: {**r, 'similarity': float(similarities[positions[k]])} for k, r in records.items()}
        sort = lambda ids: sorted(ids, key=lambda k: (-current[k]['similarity'], k))
        raw_order = sort(raw_ids)[:CONFIG['raw_top_k']]
        compact_order = sort(memory_ids)[:CONFIG['compact_top_k']]
        expansion_rank = {}
        for rank, key in enumerate(compact_order[:CONFIG['expansion_seed_memories']]):
            possible = set(current[key]['candidate_window_source_ids'])
            for raw in raw_ids:
                if set(current[raw]['source_ids']) & possible:
                    expansion_rank[raw] = min(rank, expansion_rank.get(raw, rank))
        expansion = sorted(expansion_rank, key=lambda k: (expansion_rank[k], -current[k]['similarity'], k))[:CONFIG['expansion_raw_limit']]
        raw_pool = sort(set(raw_order + expansion))
        common = raw_pool + compact_order
        pair_records = candidate_pairs(raw_pool, current, df, total)
        fusion = [k for pair in itertools.zip_longest(raw_order, compact_order) for k in pair if k]
        orders = {'raw_dense': raw_order, 'compact_memory': compact_order,
                  'static_fusion': fusion, 'source_expansion': list(dict.fromkeys(expansion+raw_order))}
        for arm in ARMS:
            trace = []
            if arm == 'bundle_proxy':
                selected, trace = bundle_pack(common, current, q['question'], pair_records,
                                              df, total, counter, CONFIG['evidence_budget'])
            else:
                selected = ordered_pack(orders[arm], current, counter, CONFIG['evidence_budget'])
            evidence = render(selected, current)
            contexts = materialize(selected, current)
            tokens = counter.count(evidence)
            if tokens > CONFIG['evidence_budget']:
                raise ValueError('Serialized evidence exceeds hard budget')
            plans.append({**q, 'arm': arm, 'signature': sid, 'selected_ids': selected,
                'contexts': contexts, 'evidence_tokens': tokens, 'evidence_sha256': fingerprint(evidence),
                'source_ids': sorted({s for k in selected for s in current[k]['source_ids']}),
                'candidate_pool_ids': common, 'candidate_pairs': pair_records,
                'proxy_trace': trace, 'status': 'ready' if contexts else 'no_context',
                'semantic_dependencies_verified': False})
        print('PREPARED_ACTIONS', q['question_id'], len(ARMS), flush=True)
    write_json(out/'plans.json', plans)
    write_json(out/'candidate_records.json', records)
    write_json(out/'manifest.json', {'signature': sid, 'conversation_id': conversation_id,
        'memory_sha256': baseline.sha(memories_path), 'plan_sha256': baseline.sha(out/'plans.json'),
        'candidate_sha256': baseline.sha(out/'candidate_records.json'),
        'questions': len(queries), 'plans': len(plans), 'cpu_wall_seconds': time.perf_counter()-started,
        'embedding_dimension': 1024, 'embedding_device': 'cpu', 'counter_exact': counter.exact,
        'feature_weights': CONFIG, 'private_inputs_read': False, 'gpu_used': False})


def verify_plan(cid, sid):
    out = RUN / cid
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if manifest['signature'] != sid or manifest['plan_sha256'] != baseline.sha(out/'plans.json') or \
       manifest['candidate_sha256'] != baseline.sha(out/'candidate_records.json') or \
       manifest['memory_sha256'] != baseline.sha(baseline.RUN/cid/'memories.json'):
        raise ValueError('Prepared plan/source changed')
    plans = json.loads((out/'plans.json').read_text(encoding='utf-8'))
    queries = [q for q in json.loads((baseline.PLAN/'public/queries.json').read_text(encoding='utf-8'))
               if q['conversation_id'] == cid]
    expected = {(q['question_id'], arm) for q in queries for arm in ARMS}
    if {(p['question_id'], p['arm']) for p in plans} != expected or len(plans) != len(expected):
        raise ValueError('Prepared action coverage mismatch')
    public = {q['question_id']: q for q in queries}
    records = json.loads((out/'candidate_records.json').read_text(encoding='utf-8'))
    for p in plans:
        if p['signature'] != sid or any(p[k] != public[p['question_id']][k] for k in public[p['question_id']]):
            raise ValueError('Prepared question/signature mismatch')
        text = FORMAT(None, [SimpleNamespace(**e) for e in p['contexts']])
        if p['contexts'] != materialize(p['selected_ids'], records):
            raise ValueError('Prepared evidence does not match source records')
        if fingerprint(text) != p['evidence_sha256'] or p['evidence_tokens'] > CONFIG['evidence_budget']:
            raise ValueError('Prepared evidence changed/exceeds budget')
    return plans


def validate_predictions(rows, plans, sid):
    lookup = {(p['question_id'], p['arm']): p for p in plans}
    seen = set()
    for row in rows:
        key = row['question_id'], row['arm']
        if key in seen or key not in lookup or row['signature'] != sid:
            raise ValueError('Invalid pilot prediction identity/coverage/duplicate')
        plan = lookup[key]
        if any(row[k] != v for k, v in plan.items() if k != 'status'):
            raise ValueError('Prediction differs from frozen action plan')
        if row['status'] not in {'ok', 'failed', 'no_context'} or not isinstance(row['hypothesis'], str):
            raise ValueError('Invalid prediction status/answer')
        if row['status'] != 'ok' and row['hypothesis']:
            raise ValueError('Failed answer must remain empty')
        seen.add(key)
    return seen


def run(endpoint):
    sid = identity()
    convs = json.loads((baseline.PLAN/'public/conversations.json').read_text(encoding='utf-8'))
    plans = [p for c in convs for p in verify_plan(c['conversation_id'], sid)]
    previous = list(read_jsonl(RUN/'predictions.jsonl')) if (RUN/'predictions.jsonl').exists() else []
    done = validate_predictions(previous, plans, sid)
    old_calls = list(read_jsonl(RUN/'calls.jsonl')) if (RUN/'calls.jsonl').exists() else []
    call_keys = [(c['question_id'], c['arm']) for c in old_calls]
    if any(c['signature'] != sid for c in old_calls) or len(set(call_keys)) != len(call_keys):
        raise ValueError('Invalid prior call identity/duplicates')
    if set(call_keys) - done:
        raise ValueError('Uncommitted inference call found; refusing an unlogged retry')
    sys.path.insert(0, str(baseline.OFFICIAL))
    from simplemem.core.utils.llm_client import LLMClient
    parser = LLMClient.__new__(LLMClient)
    for plan in plans:
        key = plan['question_id'], plan['arm']
        if key in done:
            continue
        write_json(RUN/'status.json', {'status': 'running', 'phase': 'answer',
            'question_id': key[0], 'arm': key[1], 'completed_predictions': len(done),
            'expected_predictions': len(plans), 'signature': sid})
        start = time.perf_counter()
        evidence = FORMAT(None, [SimpleNamespace(**e) for e in plan['contexts']])
        count = baseline.rpc(endpoint, '/tokenize', {'text': evidence})['tokens']
        if count != plan['evidence_tokens'] or count > CONFIG['evidence_budget']:
            raise ValueError('Inference tokenizer differs from frozen preparation/budget')
        messages = [{'role': 'system', 'content': 'You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format.'},
                    {'role': 'user', 'content': PROMPT(None, plan['question'], evidence)}]
        answer, status, error, result = '', 'no_context', None, None
        if plan['contexts']:
            call = {'signature': sid, 'question_id': key[0], 'arm': key[1], 'messages': messages,
                    'max_tokens': CONFIG['answer_max_tokens']}
            try:
                result = baseline.rpc(endpoint, '/complete', {'messages': messages, 'max_tokens': CONFIG['answer_max_tokens']})
                text = baseline.require_complete(result)
                parsed = parser.extract_json(text)
                if not isinstance(parsed, dict) or not isinstance(parsed.get('answer'), str) or not parsed['answer'].strip():
                    raise ValueError('Answer JSON missing nonempty answer string')
                answer, status = parsed['answer'], 'ok'
            except Exception as exc:
                status, error = 'failed', f'{type(exc).__name__}: {exc}'
            call.update(result or {})
            call.update(status=status, error=error, wall_seconds=time.perf_counter()-start)
            baseline.append(RUN/'calls.jsonl', call)
        baseline.append(RUN/'predictions.jsonl', {**plan, 'hypothesis': answer, 'status': status,
            'error': error, 'seconds': time.perf_counter()-start, 'generation_calls': int(result is not None) if not plan['contexts'] else 1})
        done.add(key)
        print('ACTION_CHECKPOINT', len(done), '/', len(plans), key[0], key[1], status, flush=True)
    if done != {(p['question_id'], p['arm']) for p in plans}:
        raise ValueError('Final pilot answer coverage mismatch')
    all_rows = list(read_jsonl(RUN/'predictions.jsonl'))
    failures = sum(r['status'] != 'ok' for r in all_rows)
    write_json(RUN/'status.json', {'status': 'completed_with_failures' if failures else 'completed',
        'failed_or_empty_predictions': failures, 'completed_predictions': len(done),
        'expected_predictions': len(plans), 'signature': sid, 'heldout_accessed': False})


def score():
    baseline.verify_inputs(private=True)
    sid = identity()
    refs = {r['question_id']: r for r in json.loads((baseline.PLAN/'private/references.json').read_text(encoding='utf-8'))}
    rows = list(read_jsonl(RUN/'predictions.jsonl'))
    convs = json.loads((baseline.PLAN/'public/conversations.json').read_text(encoding='utf-8'))
    plans = [p for c in convs for p in verify_plan(c['conversation_id'], sid)]
    validate_predictions(rows, plans, sid)
    expected = {(q, arm) for q in refs for arm in ARMS}
    if len({(r['question_id'], r['arm']) for r in rows}) != len(rows) or any(r['signature'] != sid for r in rows):
        raise ValueError('Invalid score identity/duplicates')
    groups = defaultdict(list)
    scored = []
    for r in rows:
        if r['category'] != refs[r['question_id']]['category']:
            raise ValueError('Score category differs from reference')
        f1 = baseline.native_f1(r['hypothesis'], refs[r['question_id']]['answer'], r['category'])
        value = {**r, 'reference': refs[r['question_id']]['answer'], 'official_native_f1': f1}
        scored.append(value)
        for scope in ['overall', 'category:'+str(r['category']), 'conversation:'+r['conversation_id']]:
            groups[r['arm'], scope].append(value)
    scores = [{'arm': arm, 'scope': scope, 'n': len(v),
        'native_f1': sum(r['official_native_f1'] for r in v)/len(v),
        'mean_evidence_tokens': sum(r['evidence_tokens'] for r in v)/len(v),
        'statuses': dict(Counter(r['status'] for r in v))} for (arm, scope), v in sorted(groups.items())]
    lookup = {(r['question_id'], r['arm']): r['official_native_f1'] for r in scored}
    complete = {(r['question_id'], r['arm']) for r in rows} == expected
    headroom = None
    if complete:
        best_fixed = max(sum(lookup[q, a] for q in refs)/len(refs) for a in ARMS)
        oracle = sum(max(lookup[q, a] for a in ARMS) for q in refs)/len(refs)
        headroom = {'post_hoc_best_action_native_f1': oracle, 'best_fixed_native_f1': best_fixed,
                    'difference': oracle-best_fixed, 'deployable_score': False}
    calls = list(read_jsonl(RUN/'calls.jsonl')) if (RUN/'calls.jsonl').exists() else []
    call_keys = [(c['question_id'], c['arm']) for c in calls]
    if any(c['signature'] != sid for c in calls) or len(set(call_keys)) != len(call_keys):
        raise ValueError('Mixed cost identity')
    expected_calls = {(r['question_id'], r['arm']) for r in rows if r['contexts']}
    if set(call_keys) != expected_calls:
        raise ValueError('Cost log does not match attempted answers')
    costs = {arm: {'calls': sum(c['arm'] == arm for c in calls),
        'failed_calls': sum(c['arm'] == arm and c['status'] != 'ok' for c in calls),
        **{k: sum(c.get('usage', {}).get(k, 0) for c in calls if c['arm'] == arm)
           for k in ['prompt_tokens', 'completion_tokens']}} for arm in ARMS}
    write_json(RUN/'scored.json', scored)
    write_json(RUN/'report.json', {'signature': sid, 'complete': complete,
        'all_answers_successful': complete and all(r['status'] == 'ok' for r in rows),
        'failed_or_empty_predictions': sum(r['status'] != 'ok' for r in rows),
        'expected_predictions': len(expected), 'observed_predictions': len(rows),
        'scores': scores, 'costs': costs, 'action_headroom_diagnostic': headroom,
        'category_macro_f1': {a: sum(s['native_f1'] for s in scores if s['arm'] == a and s['scope'].startswith('category:'))/4
                              for a in ARMS} if complete else None,
        'scope': 'two-conversation development action-space pilot; not a learned information-value policy',
        'upstream_costs': 'Reuse baseline writer and shared CPU embeddings; construction/storage costs must be reported separately; no end-to-end efficiency claim.',
        'causal_semantic_relations_verified': False, 'heldout_accessed': False})
    print(json.dumps([s for s in scores if s['scope'] == 'overall']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'score', 'verify'])
    parser.add_argument('--conversation-id')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8003')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.conversation_id)
    elif args.mode == 'run':
        run(args.endpoint)
    elif args.mode == 'score':
        score()
    else:
        sid = identity()
        plans = verify_plan(args.conversation_id, sid)
        print('VERIFIED_PREPARED_ACTIONS', len(plans), flush=True)


if __name__ == '__main__':
    main()
