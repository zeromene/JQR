"""Full official SimpleMem cached retrieval, native order, strict common 2K QA.

prepare/run never load private references. Only score loads existing dev answers.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from . import baseline_bootstrap_v2 as base
from . import evidence_acquisition_v1 as evidence
from .evidence_acquisition_v2 import parse_answer
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/mainline_simplemem_v1'
CONFIG = dict(method='SimpleMem-full-retrieval-2k', reader='Qwen3.5-9B',
    budget=2048, answer_max_tokens=2048, decoding='bf16-sdpa-greedy-thinking_false',
    selection='native cached retriever order; whole entry first-fit; exact text dedup',
    query_prompt='unmodified SimpleMem AnswerGenerator',
    reuse='complete frozen official planner/hybrid/reflection retrieval; fresh QA',
    final_test_accessed=False)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity():
    base.verify_inputs()
    paths = ['reader_memory/mainline_simplemem_v1.py', 'reader_memory/mainline_service_v1.py',
        'scripts/mainline_baseline_queue_v1.py', 'research/mainline_baselines_v1/SIMPLEMEM_PROTOCOL.md',
        'reader_memory/evidence_acquisition_v1.py', 'reader_memory/evidence_acquisition_v2.py',
        'reader_memory/tokenization.py', 'reader_memory/backend.py']
    value = dict(config=CONFIG, code={p: base.sha(ROOT/p) for p in paths},
        public_inputs={p: base.sha(base.PLAN/p) for p in
            ['public/queries.json', 'public/conversations.json']},
        cached_retrieval_sha256=base.sha(base.RUN/'predictions.jsonl'),
        cached_identity_sha256=base.sha(base.RUN/'identity.json'),
        official_revision=base.check_sources())
    target = RUN/'identity.json'
    if target.exists() and load(target) != value:
        raise ValueError('Frozen identity changed; create new version')
    write_json(target, value)
    return fingerprint(value)


def prepare():
    from .tokenization import TokenCounter
    sid = identity()
    if (RUN/'plans.json').exists():
        verify(sid)
        return
    counter = TokenCounter(ROOT/'models/Qwen3.5-9B')
    queries = load(base.PLAN/'public/queries.json')
    cached = list(read_jsonl(base.RUN/'predictions.jsonl'))
    cached_sid = fingerprint(load(base.RUN/'identity.json'))
    selected = [r for r in cached if r['method'] == 'simplemem']
    if len(selected) != len(queries) or len({r['question_id'] for r in selected}) != len(queries):
        raise ValueError('Incomplete/duplicate upstream retrieval')
    lookup = {r['question_id']: r for r in selected}
    plans = []
    for q in queries:
        row = lookup[q['question_id']]
        if row['signature'] != cached_sid or any(row[k] != q[k] for k in q):
            raise ValueError('Cached query identity mismatch')
        if row['retrieval_errors'] or row['degraded_calls'] or row['status'] != 'ok':
            raise ValueError('Cached complete retrieval has a degradation; explicit review required')
        chosen, skipped, seen = [], [], set()
        for entry in row['contexts']:
            text = entry['lossless_restatement'].strip()
            if text in seen:
                skipped.append(dict(entry_id=entry['entry_id'], reason='exact_duplicate'))
                continue
            seen.add(text)
            trial = chosen + [entry]
            tokens = counter.count(evidence.FORMAT(None, [SimpleNamespace(**e) for e in trial]))
            if tokens <= CONFIG['budget']:
                chosen = trial
            else:
                skipped.append(dict(entry_id=entry['entry_id'], reason='budget'))
        serialized = evidence.FORMAT(None, [SimpleNamespace(**e) for e in chosen])
        plans.append(dict(**q, signature=sid, method=CONFIG['method'], contexts=chosen,
            evidence_tokens=counter.count(serialized), evidence_budget=CONFIG['budget'],
            skipped=skipped, retrieved_entries=len(row['contexts']),
            reused_query_call_ids=row['query_call_ids'],
            upstream_retrieval_cache='runs/baseline_bootstrap_v2/predictions.jsonl'))
    write_json(RUN/'plans.json', plans)
    write_json(RUN/'manifest.json', dict(signature=sid, plans_sha256=base.sha(RUN/'plans.json'),
        n=len(plans), mean_evidence_tokens=statistics.mean(p['evidence_tokens'] for p in plans),
        private_inputs_read=False, heldout_accessed=False))
    print('PREPARED_FULL_SIMPLEMEM_2K', len(plans), flush=True)


def verify(sid):
    manifest = load(RUN/'manifest.json')
    if manifest['signature'] != sid or manifest['plans_sha256'] != base.sha(RUN/'plans.json'):
        raise ValueError('Plan integrity mismatch')
    plans = load(RUN/'plans.json')
    qs = load(base.PLAN/'public/queries.json')
    if len(plans) != len(qs) or {p['question_id'] for p in plans} != {q['question_id'] for q in qs}:
        raise ValueError('Plan coverage mismatch')
    if any(p['signature'] != sid or p['evidence_tokens'] > CONFIG['budget'] for p in plans):
        raise ValueError('Plan signature/budget mismatch')
    return plans


def run(endpoint):
    sid = identity()
    plans = verify(sid)
    previous = list(read_jsonl(RUN/'predictions.jsonl')) if (RUN/'predictions.jsonl').exists() else []
    calls = list(read_jsonl(RUN/'calls.jsonl')) if (RUN/'calls.jsonl').exists() else []
    if len(previous) != len({r['question_id'] for r in previous}) or any(r['signature'] != sid for r in previous+calls):
        raise ValueError('Invalid/duplicate run identity')
    done = {r['question_id'] for r in previous}
    if len(calls) != len({r['question_id'] for r in calls}) or {r['question_id'] for r in calls} != {
        r['question_id'] for r in previous if r['contexts']}:
        raise ValueError('Uncommitted or duplicate generation; no automatic retry')
    intent = RUN/'inflight.json'
    if intent.exists():
        if load(intent)['question_id'] not in done:
            raise ValueError('Uncommitted generation intent; explicit review required')
        intent.unlink()
    for p in plans:
        if p['question_id'] in done:
            continue
        write_json(RUN/'status.json', dict(status='running', phase='answer',
            completed=len(done), expected=len(plans), question_id=p['question_id']))
        text = evidence.FORMAT(None, [SimpleNamespace(**e) for e in p['contexts']])
        if base.rpc(endpoint, '/tokenize', dict(text=text))['tokens'] != p['evidence_tokens']:
            raise ValueError('Tokenizer differs from prepared evidence')
        started = time.perf_counter()
        answer, status, error, result = '', 'no_context', None, {}
        if p['contexts']:
            messages = [dict(role='system', content='You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format.'),
                dict(role='user', content=evidence.PROMPT(None, p['question'], text))]
            call = dict(signature=sid, question_id=p['question_id'], messages=messages,
                max_tokens=CONFIG['answer_max_tokens'])
            write_json(intent, dict(signature=sid, question_id=p['question_id'], request_hash=fingerprint(call)))
            try:
                result = base.rpc(endpoint, '/complete', dict(messages=messages, max_tokens=CONFIG['answer_max_tokens']))
                answer = parse_answer(base.require_complete(result))
                status = 'ok'
            except Exception as exc:
                status, error = 'failed', f'{type(exc).__name__}: {exc}'
            base.append(RUN/'calls.jsonl', dict(**call, **result, status=status, error=error,
                wall_seconds=time.perf_counter()-started))
        base.append(RUN/'predictions.jsonl', dict(**p, hypothesis=answer, status=status,
            error=error, wall_seconds=time.perf_counter()-started))
        if intent.exists():
            intent.unlink()
        done.add(p['question_id'])
        print('SIMPLEMEM_2K_QA', len(done), '/', len(plans), p['question_id'], status, flush=True)
        if status == 'failed' and not result:
            raise RuntimeError('No inference result; failure recorded, stopping remaining calls')
    write_json(RUN/'status.json', dict(status='completed', completed=len(done),
        expected=len(plans), heldout_accessed=False))


def score():
    base.verify_inputs(private=True)
    sid = identity()
    plans = verify(sid)
    refs = {r['question_id']: r for r in load(base.PLAN/'private/references.json')}
    rows = list(read_jsonl(RUN/'predictions.jsonl'))
    if len(rows) != len({r['question_id'] for r in rows}) or any(r['signature'] != sid for r in rows):
        raise ValueError('Invalid scoring rows')
    groups, scored = defaultdict(list), []
    for row in rows:
        ref = refs[row['question_id']]
        f1 = base.native_f1(row['hypothesis'], ref['answer'], ref['category'])
        scored.append(dict(**row, official_native_f1=f1))
        for scope in ['overall', 'category:'+str(row['category']), 'conversation:'+row['conversation_id']]:
            groups[scope].append((f1, row))
    summary = [dict(scope=k, n=len(v), native_f1=statistics.mean(x[0] for x in v),
        mean_evidence_tokens=statistics.mean(x[1]['evidence_tokens'] for x in v),
        status_counts=dict(Counter(x[1]['status'] for x in v))) for k,v in sorted(groups.items())]
    calls = list(read_jsonl(RUN/'calls.jsonl'))
    report = dict(signature=sid, method=CONFIG['method'], scores=summary,
        complete={r['question_id'] for r in rows} == {p['question_id'] for p in plans},
        fresh_answer_calls=len(calls), generation_tokens={k: sum(c.get('usage', {}).get(k,0) for c in calls)
            for k in ['prompt_tokens','completion_tokens']},
        reused_retrieval_costs='original baseline_bootstrap_v2/calls.jsonl retained; not zero-cost retrieval',
        comparison_controls=dict(raw_2k=45.69, fusion_2k=51.24,
            provenance='research/evidence_acquisition_v2/RESULTS_20261027.md; frozen existing results'),
        scope='64 exposed development questions; two histories; not final confirmation', heldout_accessed=False)
    write_json(RUN/'scored.json', scored)
    write_json(RUN/'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare','run','score'])
    p.add_argument('--endpoint', default='http://127.0.0.1:8016')
    a = p.parse_args()
    if a.mode == 'run':
        run(a.endpoint)
    else:
        globals()[a.mode]()


if __name__ == '__main__':
    main()
