"""Complete pinned SimpleMem versus raw dense RAG on frozen native LoCoMo dev QA.

Only `prepare` reads the source dataset. `run` consumes public conversations and
questions, and never opens the private answers. `score` is a separate CPU process.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
import urllib.request
import uuid

from .schema import fingerprint, read_jsonl, write_json
from .official_simplemem import check_sources, OFFICIAL

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/baseline_bootstrap_v1'
RUN = ROOT / 'runs/baseline_bootstrap_v1'
SEED = 'native-locomo-baseline-20261027-v1'
SCORER_NAMES = ['normalize_answer', 'exact_match_score', 'f1_score', 'f1']
CONFIG = {
    'experiment': 'baseline_bootstrap_v1', 'model': 'Qwen3.5-9B',
    'embedding': 'Qwen3-Embedding-0.6B', 'embedding_device': 'cpu',
    'embedding_adapter': 'official absolute-local-path standard ST; no query prompt',
    'window_size': 20, 'overlap_size': 2, 'parallel_write': False,
    'semantic_top_k': 25, 'keyword_top_k': 5, 'structured_top_k': 5,
    'planning': True, 'reflection': True, 'max_reflection_rounds': 2,
    'parallel_retrieval': False, 'raw_chunk_turns': 4, 'raw_chunk_overlap': 1,
    'raw_top_k': 20, 'writer_max_tokens': 8192, 'planner_max_tokens': 2048,
    'answer_max_tokens': 2048, 'max_input_tokens': 32768,
    'decoding': 'bf16-sdpa-greedy-enable_thinking_false',
    'answer_prompt': 'unchanged official AnswerGenerator for both methods',
    'evidence_budget': None, 'cost_matched': False,
    'category_5': 'excluded', 'purpose': 'two-conversation development baseline; not final test',
    'sampling': '32 per conversation; nominal 8/category; shortages filled round-robin categories1-4',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def source_turns(sample):
    conv = sample['conversation']
    turns = []
    keys = sorted((k for k, v in conv.items() if k.startswith('session_') and isinstance(v, list)),
        key=lambda k: int(k.split('_')[1]))
    for key in keys:
        date = conv[key + '_date_time']
        for turn in conv[key]:
            text = turn.get('text', '')
            if 'img_url' in turn and 'blip_caption' in turn:
                text = f"[Image: {turn['blip_caption']}]" + (f' {text}' if text else '')
            turns.append({'dialogue_id': len(turns) + 1, 'source_id': turn['dia_id'],
                'speaker': turn['speaker'], 'content': text, 'timestamp': date})
    return turns


def select_questions(samples):
    # Identity and category only; answer/evidence and observed results are not selection inputs.
    ordered = sorted(samples, key=lambda s: fingerprint([SEED, s['sample_id']]))
    dev = ordered[:2]
    queries, refs = [], []
    for sample in dev:
        pools = {}
        for category in range(1, 5):
            eligible = [(i, q) for i, q in enumerate(sample['qa']) if q['category'] == category]
            eligible.sort(key=lambda iq: fingerprint([SEED, sample['sample_id'], category, iq[0]]))
            pools[category] = eligible
        counts = {c: min(8, len(pool)) for c, pool in pools.items()}
        while sum(counts.values()) < 32:
            before = sum(counts.values())
            for category in range(1, 5):
                if counts[category] < len(pools[category]) and sum(counts.values()) < 32:
                    counts[category] += 1
            if before == sum(counts.values()):
                raise ValueError('Fewer than 32 eligible questions')
        for category in range(1, 5):
            for index, qa in pools[category][:counts[category]]:
                qid = f"{sample['sample_id']}:qa{index}"
                queries.append({'question_id': qid, 'conversation_id': sample['sample_id'],
                    'question': qa['question'], 'category': category})
                refs.append({'question_id': qid, 'answer': str(qa['answer']),
                    'evidence': qa.get('evidence', []), 'category': category})
    return dev, ordered[2:], queries, refs


def prepare():
    if (PLAN / 'manifest.json').exists():
        verify_inputs()
        print('ALREADY_FROZEN', flush=True)
        return
    dataset = ROOT / 'data/locomo/locomo10.json'
    samples = json.loads(dataset.read_text(encoding='utf-8'))
    dev, heldout, queries, refs = select_questions(samples)
    public = PLAN / 'public'
    private = PLAN / 'private'
    write_json(public / 'conversations.json', [{'conversation_id': s['sample_id'],
        'turns': source_turns(s)} for s in dev])
    write_json(public / 'queries.json', queries)
    write_json(private / 'references.json', refs)
    write_json(PLAN / 'split.json', {'seed': SEED,
        'selection': 'first two conversation identities sorted by sha256 canonical JSON [seed,id]',
        'development': [s['sample_id'] for s in dev],
        'heldout': [s['sample_id'] for s in heldout],
        'questions': CONFIG['sampling'] + '; within-category identity hash order',
        'development_unselected_questions': 'not used; same dev conversations, not independent heldout'})
    source = ROOT / 'research/sources/baseline_bootstrap_20261027/locomo_evaluation.py'
    expected = '8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd'
    if sha(source) != expected:
        raise ValueError('Official scoring source hash mismatch')
    text = source.read_text(encoding='utf-8')
    nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    subsets = [ast.get_source_segment(text, nodes[name]) for name in SCORER_NAMES]
    scorer = ('# Unmodified function subset from snap-research/locomo task_eval/evaluation.py\n'
        '# Revision 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376; CC BY-NC 4.0.\n'
        'import regex\nimport string\nimport numpy as np\nfrom collections import Counter\n'
        'from nltk.stem import PorterStemmer\nps = PorterStemmer()\n\n' + '\n\n'.join(subsets) + '\n')
    vendor = ROOT / 'reader_memory/_vendor/locomo_native_f1.py'
    vendor.write_text(scorer, encoding='utf-8')
    (vendor.parent / 'locomo_LICENSE.txt').write_bytes((ROOT / 'data/locomo/LICENSE.txt').read_bytes())
    write_json(PLAN / 'scorer_provenance.json', {'revision': '3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376',
        'source': str(source.relative_to(ROOT)), 'sha256': expected,
        'url': 'https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py',
        'functions': {name: {'start': nodes[name].lineno, 'end': nodes[name].end_lineno}
            for name in SCORER_NAMES}, 'subset_sha256': sha(vendor),
        'category_wrapper': 'category3 gold before first semicolon; category1 f1; category2/3/4 f1_score'})
    paths = ['split.json', 'scorer_provenance.json', 'public/conversations.json',
        'public/queries.json', 'private/references.json']
    write_json(PLAN / 'manifest.json', {'config': CONFIG, 'dataset_sha256': sha(dataset),
        'official_revision': check_sources(), 'query_count': len(queries),
        'files': {p: sha(PLAN / p) for p in paths}, 'created_unix': time.time()})
    print(json.dumps({'development': [s['sample_id'] for s in dev],
        'queries': len(queries), 'full_turns': [len(source_turns(s)) for s in dev]}), flush=True)


def verify_inputs(private=False):
    manifest = json.loads((PLAN / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if name.startswith('private/') and not private:
            continue
        if sha(PLAN / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    if manifest['config'] != CONFIG or manifest['official_revision'] != check_sources():
        raise ValueError('Frozen configuration/revision mismatch')
    provenance = json.loads((PLAN / 'scorer_provenance.json').read_text(encoding='utf-8'))
    if sha(ROOT / 'reader_memory/_vendor/locomo_native_f1.py') != provenance['subset_sha256']:
        raise ValueError('Scorer subset changed')
    return manifest


def rpc(endpoint, route, value):
    if urlparse(endpoint).hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('Only loopback inference is permitted')
    req = urllib.request.Request(endpoint.rstrip('/') + route,
        json.dumps(value).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def require_complete(result):
    if result.get('finish_reason') != 'stop' or not result.get('eos_seen'):
        raise ValueError('Generation truncated or did not terminate with EOS')
    if not isinstance(result.get('text'), str) or not result['text'].strip():
        raise ValueError('Empty model output')
    return result['text']


def bind_identity(output, manifest, endpoint):
    identity = {'manifest_sha256': sha(PLAN / 'manifest.json'), 'config': CONFIG,
        'official_revision': manifest['official_revision'],
        'code': {p: sha(ROOT / p) for p in ['reader_memory/baseline_bootstrap.py',
            'reader_memory/baseline_bootstrap_service.py', 'reader_memory/backend.py',
            'scripts/baseline_bootstrap_queue.py']}}
    path = output / 'identity.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != identity:
        raise ValueError('Run identity changed; use a new output directory')
    write_json(path, identity)
    return fingerprint(identity)


def run(endpoint):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    manifest = verify_inputs()
    RUN.mkdir(parents=True, exist_ok=True)
    signature = bind_identity(RUN, manifest, endpoint)
    sys.path.insert(0, str(OFFICIAL))
    from simplemem.core.utils.llm_client import LLMClient
    from simplemem.core.utils.embedding import EmbeddingModel
    from simplemem.core.models.memory_entry import Dialogue, MemoryEntry
    from simplemem.core.memory_builder import MemoryBuilder
    from simplemem.core.hybrid_retriever import HybridRetriever
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.database.vector_store import VectorStore
    from simplemem.core.settings import settings
    for key, value in {'OVERLAP_SIZE': 2, 'SEMANTIC_TOP_K': 25,
        'KEYWORD_TOP_K': 5, 'STRUCTURED_TOP_K': 5}.items():
        if getattr(settings, key) != value:
            raise ValueError('Unexpected official runtime setting: ' + key)

    active = {}
    search_events = []

    class Client(LLMClient):
        def __init__(self, stage, limit):
            self.stage, self.limit = stage, limit
            self.events = []
            self.last_json = None

        def chat_completion(self, messages, **kwargs):
            self.last_json = None
            event = {'signature': signature, 'call_id': str(uuid.uuid4()), **active,
                'stage': self.stage, 'messages': messages, 'max_tokens': self.limit,
                'requested_temperature': kwargs.get('temperature'), 'effective_decoding': 'greedy',
                'requested_response_format': kwargs.get('response_format'), 'started_unix': time.time()}
            started = time.perf_counter()
            try:
                result = rpc(endpoint, '/complete', {'messages': messages, 'max_tokens': self.limit})
                event.update(result)
                value = require_complete(result)
                event['status'] = 'complete'
                return value
            except Exception as exc:
                event.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                event['wall_seconds'] = time.perf_counter() - started
                append(RUN / 'calls.jsonl', event)
                self.events.append(event)

        def extract_json(self, text):
            event = {'signature': signature, **active, 'stage': self.stage,
                'call_id': self.events[-1]['call_id'] if self.events else None}
            try:
                value = super().extract_json(text)
                self.last_json = value
                event.update(status='parsed', parsed_type=type(value).__name__)
                return value
            except Exception as exc:
                event.update(status='failed', error=str(exc))
                raise
            finally:
                append(RUN / 'parses.jsonl', event)

    class AuditedBackend:
        def __init__(self, backend):
            self.wrapped = backend

        def __getattr__(self, name):
            target = getattr(self.wrapped, name)
            if name not in {'semantic_search', 'keyword_search', 'structured_search'}:
                return target

            def search(*args, **kwargs):
                started = time.perf_counter()
                event = {'signature': signature, **active, 'channel': name,
                    'arguments': args if name != 'semantic_search' else ['embedding vector'],
                    'kwargs': kwargs}
                try:
                    results = target(*args, **kwargs)
                    event.update(status='ok', entry_ids=[r.entry_id for r in results])
                    return results
                except Exception as exc:
                    event.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    event['seconds'] = time.perf_counter() - started
                    append(RUN / 'searches.jsonl', event)
                    search_events.append(event)
            return search

    def store_at(path):
        store = VectorStore(str(path), embedding)
        store.backend = AuditedBackend(store.backend)
        store.clear()
        return store

    embedding = EmbeddingModel(str((ROOT / 'models' / CONFIG['embedding']).resolve()))
    if embedding.dimension != 1024 or embedding.model_type != 'sentence_transformer':
        raise ValueError('Unexpected embedding configuration')
    write_json(RUN / 'embedding_runtime.json', {'dimension': embedding.dimension,
        'model_type': embedding.model_type, 'supports_query_prompt': embedding.supports_query_prompt,
        'prompts_available': getattr(embedding.model, 'prompts', {}), 'device': str(embedding.model.device)})
    conversations = json.loads((PLAN / 'public/conversations.json').read_text(encoding='utf-8'))
    questions = json.loads((PLAN / 'public/queries.json').read_text(encoding='utf-8'))
    predictions = list(read_jsonl(RUN / 'predictions.jsonl')) if (RUN / 'predictions.jsonl').exists() else []
    if any(row['signature'] != signature for row in predictions):
        raise ValueError('Mixed run signatures')
    done = {(r['question_id'], r['method']) for r in predictions}
    if len(done) != len(predictions):
        raise ValueError('Duplicate predictions')
    for conv in conversations:
        cid = conv['conversation_id']
        active.clear()
        active.update(conversation_id=cid, method='simplemem', phase='write')
        croot = RUN / cid
        croot.mkdir(exist_ok=True)
        store = store_at(croot / 'simplemem_store')
        writer = Client('write', CONFIG['writer_max_tokens'])

        class CachedBuilder(MemoryBuilder):
            window_index = 0

            def _generate_memory_entries(self, dialogues):
                index = self.window_index
                self.window_index += 1
                active['window'] = index
                key = fingerprint({'signature': signature,
                    'dialogues': [d.model_dump() for d in dialogues],
                    'previous': [e.model_dump() for e in self.previous_entries]})
                cache = croot / 'windows' / f'{index:04d}.json'
                if cache.exists():
                    value = json.loads(cache.read_text(encoding='utf-8'))
                    if value['key'] != key:
                        raise ValueError('Window cache identity mismatch')
                    return [MemoryEntry(**e) for e in value['entries']]
                begin = len(writer.events)
                entries = super()._generate_memory_entries(dialogues)
                calls = writer.events[begin:]
                value = {'key': key, 'signature': signature, 'window': index,
                    'dialogue_ids': [d.dialogue_id for d in dialogues],
                    'source_ids': [conv['turns'][d.dialogue_id - 1]['source_id'] for d in dialogues],
                    'entries': [e.model_dump() for e in entries],
                    'calls': [c['call_id'] for c in calls], 'empty_window': not entries,
                    'all_calls_failed': bool(calls) and all(c['status'] == 'failed' for c in calls)}
                write_json(cache, value)
                print(f'WRITE_CHECKPOINT {cid} window={index} entries={len(entries)} calls={len(calls)}', flush=True)
                return entries

        write_json(RUN / 'status.json', {'status': 'running', 'phase': 'write', 'conversation_id': cid,
            'signature': signature, 'completed_predictions': len(done), 'expected_predictions': 128})
        builder = CachedBuilder(writer, store, window_size=20, enable_parallel_processing=False)
        builder.add_dialogues([Dialogue(**{k: v for k, v in t.items() if k != 'source_id'})
            for t in conv['turns']])
        builder.process_remaining()
        entries = store.get_all_entries()
        if not entries:
            raise ValueError('All official memory windows empty; inspect writer calls')
        # Tantivy indexes are immutable snapshots. Refresh after all inserts, then verify.
        backend = store.backend.wrapped
        backend._fts_initialized = False
        backend._init_fts_index()
        if not backend._fts_initialized:
            raise ValueError('Official lexical index unavailable')
        write_json(croot / 'memories.json', {'signature': signature,
            'entries': [e.model_dump() for e in entries], 'turns': len(conv['turns']),
            'lexical_index_refreshed_after_write': True})
        active.clear()
        active.update(conversation_id=cid, method='raw_dense', phase='index')
        raw_store = store_at(croot / 'raw_store')
        raw_entries = []
        for start in range(0, len(conv['turns']), 3):
            turns = conv['turns'][start:start + 4]
            content = '\n'.join(f"[{t['timestamp']}] {t['speaker']}: {t['content']}" for t in turns)
            raw_entries.append(MemoryEntry(entry_id=f'{cid}:turn{start}', lossless_restatement=content))
        raw_store.add_entries(raw_entries)
        write_json(croot / 'raw_chunks.json', [{'entry': e.model_dump(),
            'source_ids': [t['source_id'] for t in conv['turns'][i*3:i*3+4]]}
            for i, e in enumerate(raw_entries)])
        for q in [q for q in questions if q['conversation_id'] == cid]:
            for method in ['raw_dense', 'simplemem']:
                if (q['question_id'], method) in done:
                    continue
                active.clear()
                active.update(conversation_id=cid, question_id=q['question_id'], method=method, phase='query')
                started = time.perf_counter()
                write_json(RUN / 'status.json', {'status': 'running', **active, 'signature': signature,
                    'completed_predictions': len(done), 'expected_predictions': 128})
                planner = Client('plan-retrieve-reflect', CONFIG['planner_max_tokens'])
                answerer = Client('answer', CONFIG['answer_max_tokens'])
                search_start = len(search_events)
                contexts, hypothesis, error = [], '', None
                try:
                    if method == 'simplemem':
                        retriever = HybridRetriever(planner, store, semantic_top_k=25, keyword_top_k=5,
                            structured_top_k=5, enable_planning=True, enable_reflection=True,
                            max_reflection_rounds=2, enable_parallel_retrieval=False)
                        contexts = retriever.retrieve(q['question'])
                    else:
                        contexts = raw_store.semantic_search(q['question'], top_k=20)
                    generator = AnswerGenerator(answerer)
                    evidence = generator._format_contexts(contexts)
                    tokens = rpc(endpoint, '/tokenize', {'text': evidence})['tokens']
                    hypothesis = generator.generate_answer(q['question'], contexts)
                    parsed = answerer.last_json
                    valid = (isinstance(parsed, dict) and isinstance(parsed.get('answer'), str)
                        and bool(parsed['answer'].strip()) and parsed['answer'] == hypothesis
                        and bool(answerer.events) and answerer.events[-1]['status'] == 'complete')
                    if not contexts:
                        status = 'no_context'
                    elif valid:
                        status = 'ok'
                    else:
                        status = 'answer_fallback_or_failure'
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    tokens, status = None, 'failed'
                row = {'signature': signature, **q, 'method': method, 'hypothesis': hypothesis,
                    'status': status, 'error': error, 'evidence_tokens': tokens,
                    'contexts': [e.model_dump() for e in contexts],
                    'query_call_ids': [c['call_id'] for c in planner.events + answerer.events],
                    'degraded_calls': sum(c['status'] != 'complete' for c in planner.events + answerer.events),
                    'retrieval_errors': sum(e['status'] == 'failed' for e in search_events[search_start:]),
                    'seconds': time.perf_counter() - started}
                append(RUN / 'predictions.jsonl', row)
                done.add((q['question_id'], method))
                print(f"ANSWER_CHECKPOINT {len(done)}/128 {q['question_id']} {method} {status}", flush=True)
    expected = {(q['question_id'], method) for q in questions for method in ['simplemem', 'raw_dense']}
    if done != expected:
        raise ValueError('Prediction coverage mismatch')
    write_json(RUN / 'status.json', {'status': 'completed', 'signature': signature,
        'completed_predictions': len(done), 'expected_predictions': len(expected), 'heldout_accessed': False})


def native_f1(prediction, answer, category):
    from ._vendor.locomo_native_f1 import f1, f1_score
    if category == 3:
        answer = answer.split(';')[0].strip()
    if category == 1:
        return float(f1(prediction, answer))
    if category in [2, 3, 4]:
        return float(f1_score(prediction, answer))
    raise ValueError('Category outside frozen protocol')


def score():
    verify_inputs(private=True)
    refs = {r['question_id']: r for r in json.loads((PLAN / 'private/references.json').read_text(encoding='utf-8'))}
    rows = list(read_jsonl(RUN / 'predictions.jsonl'))
    identity = json.loads((RUN / 'identity.json').read_text(encoding='utf-8'))
    signature = fingerprint(identity)
    if any(r['signature'] != signature for r in rows):
        raise ValueError('Score run signature mismatch')
    if len({(r['question_id'], r['method']) for r in rows}) != len(rows):
        raise ValueError('Duplicate score rows')
    scored, groups = [], defaultdict(list)
    for row in rows:
        ref = refs[row['question_id']]
        text = row['hypothesis'] if isinstance(row['hypothesis'], str) else ''
        f1 = native_f1(text, ref['answer'], ref['category'])
        value = {**row, 'reference': ref['answer'], 'official_native_f1': f1,
            'successful_format': row['status'] == 'ok'}
        scored.append(value)
        for scope in ['overall', 'category:' + str(row['category']), 'conversation:' + row['conversation_id']]:
            groups[(row['method'], scope)].append(value)
    report = []
    for (method, scope), values in sorted(groups.items()):
        report.append({'method': method, 'scope': scope, 'n': len(values),
            'native_f1': sum(v['official_native_f1'] for v in values)/len(values),
            'status_counts': dict(Counter(v['status'] for v in values)),
            'degraded_calls': sum(v['degraded_calls'] for v in values),
            'retrieval_errors': sum(v['retrieval_errors'] for v in values),
            'mean_evidence_tokens': sum(v['evidence_tokens'] or 0 for v in values)/len(values),
            'query_wall_seconds': sum(v['seconds'] for v in values)})
    costs = defaultdict(lambda: {'calls': 0, 'failed_calls': 0, 'prompt_tokens': 0,
        'completion_tokens': 0, 'generation_seconds': 0, 'wall_seconds': 0})
    if (RUN / 'calls.jsonl').exists():
        for call in read_jsonl(RUN / 'calls.jsonl'):
            if call['signature'] != signature:
                raise ValueError('Cost identity mismatch')
            key = f"{call['method']}:{call['stage']}"
            cost = costs[key]
            cost['calls'] += 1
            cost['failed_calls'] += call['status'] != 'complete'
            cost['wall_seconds'] += call['wall_seconds']
            usage = call.get('usage', {})
            for field in ['prompt_tokens', 'completion_tokens']:
                cost[field] += usage.get(field, 0)
            cost['generation_seconds'] += usage.get('elapsed_seconds', 0)
    write_json(RUN / 'scored.json', scored)
    write_json(RUN / 'report.json', {'signature': signature, 'expected_predictions': len(refs)*2,
        'observed_predictions': len(rows), 'complete': len(rows) == len(refs)*2,
        'scope': 'two-conversation development; categories 1-4; not an official full benchmark replication',
        'primary': 'unmodified official category-aware F1 including returned fallback outputs and failures',
        'scores': report, 'costs': dict(costs),
        'cost_note': 'All logged calls including retries/interrupted windows; CPU embeddings separate in queue wall time',
        'independent_conversations': 2, 'significance_test': None, 'cost_matched': False})
    print(json.dumps([r for r in report if r['scope'] == 'overall']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'score'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8002')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'run':
        run(args.endpoint)
    else:
        score()


if __name__ == '__main__':
    main()
