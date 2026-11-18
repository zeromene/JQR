"""Pinned official SimpleMem text core with explicit offline model adaptation.

This is a separate system baseline, not the reader-conditioned selection method.
Run using .venv_baselines. Oracle pilot output is always diagnostic-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

from .backend import HTTPBackend
from .datasets import load_questions
from .experiment import answer_messages
from .schema import fingerprint, history_hash, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / 'third_party/simplemem_official'


def check_sources():
    provenance = json.loads((OFFICIAL / 'PROVENANCE.json').read_text(encoding='utf-8'))
    for item in provenance['files']:
        content = (OFFICIAL / item['path']).read_bytes()
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('Official source changed: ' + item['path'])
    return provenance['revision']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'pilot'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--embedding-path', default='models/Qwen3-Embedding-0.6B')
    parser.add_argument('--reader-path', default='models/Qwen3.5-4B')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--writer', default='Qwen3.5-4B')
    parser.add_argument('--reader', default='Qwen3.5-4B')
    parser.add_argument('--budget', type=int, default=2048)
    parser.add_argument('--answer-mode', choices=['shared', 'official'], default='shared')
    parser.add_argument('--limit', type=int, default=2)
    args = parser.parse_args()
    if args.budget <= 0:
        raise ValueError('Budget must be positive')
    parsed = urlparse(args.endpoint)
    if parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('This offline adapter requires a loopback model endpoint')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    sys.path.insert(0, str(OFFICIAL))
    revision = check_sources()
    from simplemem.core.utils.embedding import EmbeddingModel
    from simplemem.core.database.vector_store import VectorStore
    from simplemem.core.models.memory_entry import Dialogue, MemoryEntry
    from simplemem.core.memory_builder import MemoryBuilder
    from simplemem.core.hybrid_retriever import HybridRetriever
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.utils.llm_client import LLMClient

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embedding = EmbeddingModel(str(Path(args.embedding_path).resolve()))
    if args.mode == 'check':
        store = VectorStore(str(output / 'vector_check'), embedding)
        store.clear()
        store.add_entries([MemoryEntry(entry_id='smoke-city',
            lossless_restatement='Mira lives in Hangzhou, previously Shanghai.',
            keywords=['Mira', 'Hangzhou'], persons=['Mira'], location='Hangzhou')])
        checks = {'semantic': store.semantic_search('Where does Mira live?', 1),
            'lexical': store.keyword_search(['Hangzhou'], 1),
            'structured': store.structured_search(persons=['Mira'], top_k=1)}
        if not all(any(entry.entry_id == 'smoke-city' for entry in values) for values in checks.values()):
            raise ValueError('Official retrieval path failed')
        write_json(output / 'status.json', {'status': 'passed', 'revision': revision,
            'checks': list(checks), 'real_cpu_embedding': True, 'llm_tested': False,
            'note': 'Environment/retrieval check only; no benchmark score.'})
        print('SIMPLEMEM_OFFICIAL_RETRIEVAL_CHECK_PASSED', flush=True)
        return

    data = ROOT / 'research/stage1/pilot_oracle.json'
    signature = {'revision': revision, 'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
        'writer': args.writer, 'reader': args.reader, 'budget': args.budget,
        'window_size': 8, 'overlap_size': 2, 'embedding': str(Path(args.embedding_path).resolve()),
        'answer_mode': args.answer_mode, 'writer_max_tokens': 4096, 'planner_max_tokens': 1024,
        'answer_max_tokens': 384, 'decoding': 'greedy-no-thinking',
        'oracle_diagnostic_only': True, 'limit': args.limit,
        'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config = output / 'config.json'
    if config.exists() and json.loads(config.read_text(encoding='utf-8')) != signature:
        raise ValueError('Baseline configuration changed; use another output directory')
    write_json(config, signature)
    sid = fingerprint(signature)
    predictions = output / 'predictions.jsonl'
    previous = list(read_jsonl(predictions)) if predictions.exists() else []
    if any(row['signature'] != sid for row in previous):
        raise ValueError('Mixed baseline run signatures')
    done = {row['question_id'] for row in previous}
    from .tokenization import TokenCounter
    counter = TokenCounter(args.reader_path)

    class LoggedClient(LLMClient):
        def __init__(self, model, stage, max_tokens, question_id):
            self.backend = HTTPBackend(args.endpoint, model, timeout=1200)
            self.stage, self.max_tokens, self.question_id = stage, max_tokens, question_id

        def chat_completion(self, messages, **kwargs):
            started = time.perf_counter()
            result = self.backend.complete(messages, max_tokens=self.max_tokens)
            with (output / 'calls.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'question_id': self.question_id, 'stage': self.stage,
                    'messages': messages, 'text': result['text'], 'usage': result['usage'],
                    'requested_temperature': kwargs.get('temperature'), 'effective_decoding': 'greedy',
                    'seconds': time.perf_counter() - started}, ensure_ascii=False) + '\n')
            return result['text']

    for question in load_questions('longmemeval', data, args.limit or None):
        if question.question_id in done:
            continue
        started = time.perf_counter()
        qroot = output / question.question_id
        qroot.mkdir(exist_ok=True)
        store = VectorStore(str(qroot / 'vector_store'), embedding)
        memories = qroot / 'memories.json'
        writer = LoggedClient(args.writer, 'write', 4096, question.question_id)
        if memories.exists():
            saved = json.loads(memories.read_text(encoding='utf-8'))
            if saved['history_hash'] != history_hash(question.turns) or saved['signature'] != sid:
                raise ValueError('Baseline memory provenance mismatch')
            store.clear()
            store.add_entries([MemoryEntry(**entry) for entry in saved['entries']])
        else:
            store.clear()
            builder = MemoryBuilder(writer, store, window_size=8, enable_parallel_processing=False)
            builder.add_dialogues([Dialogue(dialogue_id=i + 1, speaker=turn.role,
                content=turn.content, timestamp=turn.date) for i, turn in enumerate(question.turns)])
            builder.process_remaining()
            entries = store.get_all_entries()
            if not entries:
                raise ValueError('Official writer produced no memories; inspect calls and log')
            write_json(memories, {'history_hash': history_hash(question.turns), 'signature': sid,
                'entries': [entry.model_dump() for entry in entries]})
        planner = LoggedClient(args.reader, 'plan-retrieve-reflect', 1024, question.question_id)
        retriever = HybridRetriever(planner, store, enable_parallel_retrieval=False)
        contexts = retriever.retrieve(question.question)
        formatter = AnswerGenerator(planner)
        selected = []
        for entry in contexts:
            if counter.count(formatter._format_contexts(selected + [entry])) <= args.budget:
                selected.append(entry)
        evidence = formatter._format_contexts(selected)
        answer_client = LoggedClient(args.reader, 'answer', 384, question.question_id)
        if args.answer_mode == 'official':
            hypothesis = AnswerGenerator(answer_client).generate_answer(question.question, selected)
        else:
            hypothesis = answer_client.chat_completion(answer_messages(question.question, question.date, evidence))
        row = {'signature': sid, 'question_id': question.question_id,
            'question_type': question.question_type, 'reader': args.reader,
            'method': 'SimpleMem-official-core-local-adaptation', 'hypothesis': hypothesis,
            'oracle_diagnostic_only': True, 'score_status': 'not-judged',
            'budget': args.budget, 'evidence_tokens': counter.count(evidence),
            'selected_entries': [entry.model_dump() for entry in selected],
            'retrieved_entries': len(contexts), 'answer_mode': args.answer_mode,
            'seconds': time.perf_counter() - started}
        with predictions.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print('SIMPLEMEM_PILOT_ANSWER', question.question_id, flush=True)
    write_json(output / 'status.json', {'status': 'completed',
        'rows': len(list(read_jsonl(predictions))), 'score_status': 'not-judged',
        'oracle_diagnostic_only': True})


if __name__ == '__main__':
    main()
