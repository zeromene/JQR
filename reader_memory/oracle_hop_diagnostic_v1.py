"""One fixed CPU-matched oracle-hop diagnostic; shares only existing reader RPC."""
import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time

from . import full_comparison_v1 as main
from . import full_comparison_v1_repair1 as original
from .schema import fingerprint as fp, read_jsonl, write_json

ROOT = main.ROOT
PLAN = ROOT / 'research/oracle_hop_diagnostic_v1'
RUN = ROOT / 'runs/oracle_hop_diagnostic_v1'
ARMS = ('cpu_predicted_hops', 'cpu_oracle_hops')
ENDPOINT = 'http://127.0.0.1:8037'


def verify():
    manifest = main.load(PLAN / 'manifest.json')
    if fp({k: v for k, v in manifest.items() if k != 'signature'}) != manifest['signature']:
        raise ValueError('Diagnostic manifest changed')
    for name, digest in manifest['files'].items():
        if main.sha(ROOT / name) != digest:
            raise ValueError('Diagnostic dependency changed: ' + name)
    return manifest


def stamped(plan, method, marker):
    p = deepcopy(plan)
    p.update(method=method, arm=method + '__b256', evidence_budget=256,
             diagnostic_only=True, oracle_hops_used=method == ARMS[1])
    p['request_key'] = main.recomp.interface.parent.request_key(main.recomp.messages_for(p), marker)
    return p


def infer_cpu(torch, tok, model, source, hops):
    encoded, tokenization = main.beam.codes(tok, source, 'cpu')
    begin = time.perf_counter()
    with torch.inference_mode():
        result = model(**encoded, sf_idx=[[]], hop=hops)
    paths = result['current_preds']
    if not paths or any(len(p) != hops or len(set(p)) != hops or
                        any(not 0 <= j < len(source['paragraphs']) for j in p) for p in paths):
        raise ValueError('Invalid diagnostic path')
    return dict(hops=hops, paths=paths, seconds=time.perf_counter() - begin,
                precision='CPU FP32, matched in both diagnostic arms', tokenization=tokenization)


def select():
    manifest = verify()
    if (RUN / 'selection_complete.json').exists():
        raise ValueError('Already selected; no automatic recomputation')
    with original.bind():
        if original.verify() != manifest['parent_signature']:
            raise ValueError('Frozen main identity differs')
    import torch
    from tools.prepare_replacement_learning_v1 import ExactCounter
    torch.set_num_threads(24)
    torch.manual_seed(30)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok, model = main.beam.make_model('cpu')
    model.float().eval()
    checkpoint = ROOT / 'runs/path_retrieval_execution_v1/best.pt'
    if main.sha(checkpoint) != main.CONFIG['trained_checkpoint_sha256']:
        raise ValueError('Checkpoint differs')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    counter = ExactCounter()
    marker = manifest['reader_marker_sha256']
    prepared, traces = [], []
    rows = list(read_jsonl(PLAN / 'inputs.jsonl'))
    assert len(rows) == 369
    selection_started = time.monotonic()
    for index, row in enumerate(rows, 1):
        source = row['source']
        original.native_validate(source)
        old = row['original_main_plan']
        if counter.count(old['evidence_text']) != old['evidence_tokens']:
            raise ValueError('Original context/tokenizer differs')
        trace = dict(question_id=source['question_id'], predicted_hops=row['predicted_hops'],
                     true_hops=row['true_hops'], original_path=row['original_path'])
        if row['predicted_hops'] == row['true_hops']:
            selected = [stamped(old, method, marker) for method in ARMS]
            trace.update(unchanged_hop_reused=True, same_path_as_original=True,
                         input_pair_identical=True)
        else:
            predictions = [infer_cpu(torch, tok, model, source, h)
                           for h in (row['predicted_hops'], row['true_hops'])]
            selected = []
            for method, value in zip(ARMS, predictions):
                plans, packing = main.retained_plans(source, row['scores'],
                    [str(p) for p in value['paths'][0]], counter, 256)
                p = next(p for p in plans if p['method'] == main.anchor.MAIN)
                selected.append(stamped(p, method, marker))
                value['packing'] = packing
            trace.update(predicted=predictions[0], oracle=predictions[1],
                         same_path_as_original=predictions[0]['paths'][0] == row['original_path'],
                         cpu_control_context_matches_original=selected[0]['evidence_text'] == old['evidence_text'],
                         input_pair_identical=selected[0]['request_key'] == selected[1]['request_key'])
        for p in selected:
            if p['evidence_tokens'] > 256:
                raise ValueError('Diagnostic budget exceeded')
        prepared.extend(selected)
        traces.append(trace)
        changed_count = sum(not t.get('unchanged_hop_reused', False) for t in traces)
        write_json(RUN / 'selection_status.json', dict(completed=index, expected=len(rows),
            changed_hop_processed=changed_count, elapsed_seconds=time.monotonic() - selection_started,
            last_pair_inference_seconds=sum(trace[k]['seconds'] for k in ('predicted', 'oracle'))
                if not trace.get('unchanged_hop_reused') else 0, time=time.time()))
        print('ORACLE_SELECT', index, '/', len(rows), flush=True)
    del model
    gc.collect()
    with (RUN / 'qa_plans.jsonl').open('x', encoding='utf-8') as stream:
        for p in prepared:
            stream.write(json.dumps(p, ensure_ascii=False, separators=(',', ':')) + '\n')
    main.immutable(RUN / 'selection_traces.json', traces)
    keys = {p['request_key'] for p in prepared}
    with sqlite3.connect('file:' + str(original.RUN / 'qa/ledger.sqlite3') + '?mode=ro', uri=True) as db:
        reusable = sum(db.execute('SELECT 1 FROM answers WHERE key=?', (key,)).fetchone() is not None for key in keys)
    main.immutable(RUN / 'selection_complete.json', dict(signature=manifest['signature'],
        conditions=len(prepared), questions=len(rows), unique_requests=len(keys),
        existing_frozen_requests=reusable, maximum_new_reader_calls=len(keys) - reusable,
        plans_sha256=main.sha(RUN / 'qa_plans.jsonl'), oracle_uses_support_ids=False,
        oracle_uses_answers=False, new_gpu_models=0, selected_before_diagnostic_qa=True))


def qa_chunk(seconds=180):
    """Caller owns a stopped-main lease; timeout/unknown calls are never retried."""
    if (RUN / 'inflight.json').exists():
        raise ValueError('Unknown diagnostic call; refuse automatic retry')
    manifest = main.load(PLAN / 'manifest.json')
    committed = main.load(RUN / 'selection_complete.json')
    if main.sha(RUN / 'qa_plans.jsonl') != committed['plans_sha256']:
        raise ValueError('Committed diagnostic plans changed')
    plans = list(read_jsonl(RUN / 'qa_plans.jsonl'))
    predictions = list(read_jsonl(RUN / 'predictions.jsonl')) if (RUN / 'predictions.jsonl').exists() else []
    done = {(p['question_id'], p['arm']) for p in predictions}
    if len(done) != len(predictions):
        raise ValueError('Duplicate diagnostic condition')
    own = {c['key']: c for c in read_jsonl(RUN / 'answer_calls.jsonl')} if (RUN / 'answer_calls.jsonl').exists() else {}
    from .evidence_acquisition_v2 import parse_answer
    rpc = main.recomp.ledger.base.rpc
    require_complete = main.recomp.ledger.base.require_complete
    begin = time.monotonic()
    with sqlite3.connect('file:' + str(original.RUN / 'qa/ledger.sqlite3') + '?mode=ro', uri=True) as db:
        for p in plans:
            identity = (p['question_id'], p['arm'])
            if identity in done:
                continue
            key = p['request_key']
            prior = db.execute('SELECT value FROM answers WHERE key=?', (key,)).fetchone()
            reused = 'main_frozen_cache' if prior else 'diagnostic_cache' if key in own else None
            if p.get('selection_error'):
                entry = dict(hypothesis='', status='selection_failed', error=p['selection_error'])
                reused = 'selection_failure'
            elif prior:
                entry = json.loads(prior[0])
            elif key in own:
                entry = {k: own[key][k] for k in ('hypothesis', 'status', 'error')}
            else:
                if len(own) >= 380:
                    raise ValueError('Diagnostic reader-call bound reached')
                messages = main.recomp.messages_for(p)
                if rpc(ENDPOINT, '/tokenize', dict(text=p['evidence_text']))['tokens'] != p['evidence_tokens']:
                    raise ValueError('Reader tokenizer mismatch')
                write_json(RUN / 'inflight.json', dict(key=key, signature=manifest['signature'],
                    question_id=p['question_id'], arm=p['arm'], messages=messages, max_tokens=2048))
                started = time.perf_counter()
                result, answer, status, error = {}, '', 'failed', None
                try:
                    result = rpc(ENDPOINT, '/complete', dict(messages=messages, max_tokens=2048))
                    answer = parse_answer(require_complete(result))
                    status = 'ok'
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                call = dict(signature=manifest['signature'], key=key, messages=messages,
                    hypothesis=answer, status=status, error=error, result=result, max_tokens=2048,
                    seconds=time.perf_counter() - started, source_condition=list(identity))
                main.append(RUN / 'answer_calls.jsonl', call)
                if not result:
                    raise RuntimeError('No inference result; retain unknown call, never retry')
                own[key] = call
                entry = dict(hypothesis=answer, status=status, error=error)
            prediction = dict(question_id=p['question_id'], arm=p['arm'], method=p['method'],
                budget=256, signature=manifest['signature'], plan_sha256=fp(p), request_key=key,
                reused_from=reused, diagnostic_only=True, **entry)
            main.append(RUN / 'predictions.jsonl', prediction)
            done.add(identity)
            if (RUN / 'inflight.json').exists():
                (RUN / 'inflight.json').unlink()
            write_json(RUN / 'qa_status.json', dict(completed=len(done), expected=len(plans),
                new_model_calls=len(own), time=time.time(), reader=ENDPOINT))
            print('ORACLE_QA', len(done), '/', len(plans), 'new_calls', len(own), flush=True)
            if time.monotonic() - begin >= seconds:
                return False
    main.immutable(RUN / 'qa_complete.json', dict(conditions=len(done), new_model_calls=len(own)))
    return True


def score():
    if not (RUN / 'qa_complete.json').exists():
        raise ValueError('Diagnostic matrix incomplete')
    refs = main.load(PLAN / 'references.json')
    meta = main.load(PLAN / 'manifest.json')
    records = list(read_jsonl(RUN / 'predictions.jsonl'))
    originals = main.load(PLAN / 'original_predictions.json')
    with original.bind():
        scored = {(r['question_id'], r['method']): main.answer_scores(r['hypothesis'], refs[r['question_id']])
                  if r['status'] == 'ok' else (0., 0.) for r in records}
        for q, methods in originals.items():
            for method, r in methods.items():
                scored[q, method] = main.answer_scores(r['hypothesis'], refs[q]) if r['status'] == 'ok' else (0., 0.)
    inp = list(read_jsonl(PLAN / 'inputs.jsonl'))
    low = {r['source']['question_id'] for r in inp if r['predicted_hops'] != r['true_hops']}
    groups = {}
    methods = (*ARMS, 'anchor_path_fill', 'qwen_rerank_sentence')
    for name, qs in [('all_369', set(refs)), ('underestimated_190', low), ('unchanged_179', set(refs) - low)]:
        groups[name] = [dict(method=m, n=len(qs), em=100 * statistics.mean(scored[q, m][0] for q in qs),
                            f1=100 * statistics.mean(scored[q, m][1] for q in qs)) for m in methods]
    clusters = main.source_clusters(refs)
    comparisons = {}
    for name, qs in [('all_369', set(refs)), ('underestimated_190', low)]:
        comparisons[name] = main.paired_stats({q: scored[q, ARMS[1]][1] for q in qs},
            {q: scored[q, ARMS[0]][1] for q in qs}, clusters)
    traces = main.load(RUN / 'selection_traces.json')
    changed = [t for t in traces if not t.get('unchanged_hop_reused')]
    structural = dict(cpu_control_same_original_path=sum(t['same_path_as_original'] for t in changed),
        cpu_control_same_original_context=sum(t['cpu_control_context_matches_original'] for t in changed),
        input_pairs_identical=sum(t['input_pair_identical'] for t in changed), changed_hop_questions=len(changed))
    for arm, label in [('predicted', 'cpu_predicted'), ('oracle', 'cpu_oracle')]:
        structural[label + '_top1_support_complete'] = sum(set(refs[t['question_id']]['support_indices']) <=
            set(t[arm]['paths'][0]) for t in changed)
    plans = list(read_jsonl(RUN / 'qa_plans.jsonl'))
    for arm in ARMS:
        arm_plans = [p for p in plans if p['method'] == arm and p['question_id'] in low]
        structural[arm + '_context_has_all_support_sources'] = sum(
            {str(i) for i in refs[p['question_id']]['support_indices']} <= set(p.get('selected_ranges', {}))
            for p in arm_plans)
        structural[arm + '_mean_evidence_tokens'] = statistics.mean(p['evidence_tokens'] for p in arm_plans)
    report = dict(signature=meta['signature'], complete=True, diagnostic_only=True, budget=256,
        primary='CPU-matched oracle versus predicted hops, same frozen retriever weights and scores',
        groups=groups, comparisons=comparisons, structural=structural,
        selection=main.load(RUN / 'selection_complete.json'), qa=main.load(RUN / 'qa_complete.json'),
        main_method_changed=False, new_training=False, paper_ready=False,
        caveats=['Previously examined nonrandom 369-question prefix; diagnostic, not independent confirmation.',
                 'CPU FP32 paired retrieval controls differ from the original GPU BF16 execution; both are reported.',
                 'Only true hop count is granted, never support identity or answers; not a deployable baseline.',
                 'Source coverage is not evidence sufficiency; failed outputs retain zero scores.'])
    main.immutable(RUN / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['select', 'score'])
    a = p.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if a.mode == 'select':
        select()
    else:
        score()
