"""One controlled replay of v1 training, excluding failed-feedback preferences."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import time

from . import baseline_bootstrap_v2 as base
from . import replacement_learning_v1 as parent
from . import replacement_ranker_v1 as ranker
from . import replacement_precision_v2 as precision
from . import replacement_controls_v3 as controls
from . import replacement_stage2_common as common
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/clean_feedback_control_v1'
RUN = ROOT / 'runs/clean_feedback_control_v1'
ARMS = ('clean_full_union', 'clean_wide_single', 'clean_matched_single', 'clean_matched_union')
CONFIG = dict(version='clean-feedback-control-v1', arms=list(ARMS), ranker_config=ranker.CONFIG,
    seed=29, initialization='original pretrained ranker, identical to v1',
    change='replace failed-feedback training slots with normally completed preferences only',
    replay='same original slot order and batch boundaries; same-question replacement where available',
    checkpoint='fixed last epoch', score_precision='unchanged BF16 head, FP32 yes-no subtraction',
    candidate_search='exact existing precision-v2 and controls-v3 alternatives, no expansion',
    budget=2048, reader='Qwen3.5-9B', answer_max_tokens=2048,
    scope='clean training integrity control; exposed development only', heldout_accessed=False)
ORIGINAL_TRAINING_PAIRS = parent.training_pairs


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pair_key(pair):
    a, b, ra, rb = pair
    return [a['question_id'], a['variant'], b['variant'], ra, rb]


def clean_replay_pairs(pairs, statuses):
    valid_indices = [i for i, p in enumerate(pairs) if all(
        statuses[r['question_id'], r['variant']] == 'ok' for r in p[:2])]
    if not valid_indices:
        raise ValueError('No normal completed preferences; cannot train')
    byq = defaultdict(list)
    for i in valid_indices:
        byq[pairs[i][0]['question_id']].append(i)
    valid = set(valid_indices)
    indices, replacements = [], []
    for i, pair in enumerate(pairs):
        if i in valid:
            indices.append(i)
            continue
        qid = pair[0]['question_id']
        candidates = byq.get(qid, valid_indices)
        chosen = min(candidates, key=lambda j: fingerprint(
            ['clean-feedback-control-20261029', i, pair_key(pairs[j])]))
        indices.append(chosen)
        replacements.append(dict(original_slot=i, replacement_pair_index=chosen,
            original_question=qid, replacement_question=pairs[chosen][0]['question_id'],
            same_question=qid == pairs[chosen][0]['question_id']))
    replay = [pairs[i] for i in indices]
    audit = dict(original_pair_count=len(pairs), unique_normal_pairs=len(valid_indices),
        removed_failed_pair_count=len(replacements), replay_slots=len(replay),
        original_questions=len({p[0]['question_id'] for p in pairs}),
        clean_questions=len({p[0]['question_id'] for p in replay}),
        same_question_replacements=sum(r['same_question'] for r in replacements),
        global_replacements=sum(not r['same_question'] for r in replacements),
        replay_pair_indices=indices, replacements=replacements,
        original_pairs_sha256=fingerprint([pair_key(p) for p in pairs]),
        clean_replay_sha256=fingerprint([pair_key(p) for p in replay]))
    return replay, audit


def slot_schedule(n):
    rng = random.Random(ranker.CONFIG['seed'])
    batches = []
    for epoch in range(ranker.CONFIG['epochs']):
        order = list(range(n))
        rng.shuffle(order)
        for start in range(0, n, ranker.CONFIG['accumulation']):
            batches.append(dict(epoch=epoch + 1,
                slots=order[start:start + ranker.CONFIG['accumulation']]))
    return batches


def prepare():
    if (PLAN / 'manifest.json').exists():
        raise ValueError('Control already frozen')
    old = Path(load(ROOT / 'research/replacement_learning_v1/preserved/LATEST.json')['local_directory'])
    stage2 = Path(load(ROOT / 'research/replacement_training_v2/SNAPSHOT_LATEST.json')['directory'])
    prec = Path(load(ROOT / 'research/replacement_precision_v2/SNAPSHOT_LATEST.json')['directory'])
    inputs = load(parent.PLAN / 'public/reward_inputs.json')
    targets = load(old / 'runs/replacement_learning_v1_repair1/reward_targets.json')
    statuses = {(r['question_id'], r['variant']): r['status'] for r in targets}
    original = ORIGINAL_TRAINING_PAIRS(inputs, targets, 'train')
    replay, audit = clean_replay_pairs(original, statuses)
    batches = slot_schedule(len(replay))
    audit.update(updates=len(batches), draws=sum(len(b['slots']) for b in batches),
                 last_batch_size_per_epoch=[len(b['slots']) for i, b in enumerate(batches)
                    if i + 1 == len(batches) or batches[i + 1]['epoch'] != b['epoch']],
                 failed_training_preferences=0, heldout_accessed=False)
    common.immutable_json(PLAN / 'private/source_reward_targets.json', targets)
    common.immutable_json(PLAN / 'training_plan.json', audit)
    common.immutable_json(PLAN / 'training_schedule.json', batches)
    all_dev = load(stage2 / 'runs/replacement_training_v2/reward_inputs.json')
    dev = [r for r in all_dev if r['split'] == 'dev' and r['training_kind'] == 'natural']
    keys = {(r['question_id'], r['arm']) for r in dev}
    dev_targets = [r for r in load(stage2 / 'runs/replacement_training_v2/reward_targets.json')
                   if (r['question_id'], r['arm']) in keys]
    common.immutable_json(PLAN / 'public/dev_inputs.json', dev)
    common.immutable_json(PLAN / 'private/dev_targets.json', dev_targets)
    sources = {}
    for name in ['research/replacement_learning_v1/manifest.json',
                 'research/replacement_learning_v1/public/reward_inputs.json',
                 'research/replacement_learning_v1/private/source_selection.json',
                 'research/replacement_learning_v1/model_artifact_reference.json']:
        sources[name] = base.sha(ROOT / name)
    for directory, prefix in [(prec, 'runs/replacement_precision_v2/sets'),
                              (stage2, 'runs/replacement_controls_v3/selected')]:
        files = sorted((directory / prefix).glob('*.json'))
        if len(files) != 64:
            raise ValueError('Expected complete frozen 64-question candidate inventory')
        sources.update({p.relative_to(directory).as_posix(): base.sha(p) for p in files})
    common.immutable_json(PLAN / 'source_artifacts.json', sources)
    print('PREPARED_CLEAN_CONTROL', json.dumps({k: v for k, v in audit.items()
          if k not in ['replay_pair_indices', 'replacements']}, ensure_ascii=False), flush=True)


def freeze():
    paths = ['reader_memory/clean_feedback_control_v1.py', 'scripts/clean_feedback_control_v1_queue.py',
        'tests/test_clean_feedback_control_v1.py', 'research/clean_feedback_control_v1/PROTOCOL.md',
        'reader_memory/replacement_ranker_v1.py', 'reader_memory/replacement_learning_v1.py',
        'reader_memory/replacement_precision_v2.py', 'reader_memory/replacement_controls_v3.py',
        'reader_memory/replacement_stage2_common.py', 'reader_memory/replacement_stage2_cache_repair.py',
        'reader_memory/replacement_core_v1.py', 'reader_memory/mainline_service_v1.py',
        'reader_memory/baseline_bootstrap_v2.py', 'reader_memory/evidence_acquisition_v1.py',
        'reader_memory/evidence_acquisition_v2.py', 'reader_memory/backend.py', 'reader_memory/schema.py']
    paths += ['research/clean_feedback_control_v1/' + p for p in ['training_plan.json',
        'training_schedule.json', 'source_artifacts.json', 'private/source_reward_targets.json',
        'public/dev_inputs.json', 'private/dev_targets.json']]
    value = dict(config=CONFIG, files={p: base.sha(ROOT / p) for p in paths},
                 source_artifacts=load(PLAN / 'source_artifacts.json'))
    value['signature'] = fingerprint(value)
    common.immutable_json(PLAN / 'manifest.json', value)
    print('FROZEN_CLEAN_CONTROL', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or fingerprint({k: v for k, v in value.items() if k != 'signature'}) != value['signature']:
        raise ValueError('Control identity changed')
    for name, digest in {**value['files'], **value['source_artifacts']}.items():
        if base.sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return value['signature']


def preflight():
    sid = verify()
    parent.verify()
    precision.verify()
    common.verify(controls.PLAN, controls.CONFIG)
    inputs = load(parent.PLAN / 'public/reward_inputs.json')
    targets = load(PLAN / 'private/source_reward_targets.json')
    pairs = ORIGINAL_TRAINING_PAIRS(inputs, targets, 'train')
    statuses = {(r['question_id'], r['variant']): r['status'] for r in targets}
    _, audit = clean_replay_pairs(pairs, statuses)
    if any(audit[k] != load(PLAN / 'training_plan.json')[k] for k in audit):
        raise ValueError('Clean training replay changed')
    if slot_schedule(len(pairs)) != load(PLAN / 'training_schedule.json'):
        raise ValueError('Original training batch schedule changed')
    cache = common.cached_answers(cache_files())
    common.immutable_json(RUN / 'preflight.json', dict(signature=sid, failed_preferences=0,
        cache_entries=len(cache), training_plan=load(PLAN / 'training_plan.json'), heldout_accessed=False))
    print('CLEAN_CONTROL_PREFLIGHT_OK', len(pairs), 'slots', flush=True)


def train():
    sid = verify()
    targets = load(PLAN / 'private/source_reward_targets.json')
    common.immutable_json(RUN / 'reward_targets.json', targets)
    inputs = load(parent.PLAN / 'public/reward_inputs.json')
    statuses = {(r['question_id'], r['variant']): r['status'] for r in targets}
    replay, audit = clean_replay_pairs(ORIGINAL_TRAINING_PAIRS(inputs, targets, 'train'), statuses)
    if audit['clean_replay_sha256'] != load(PLAN / 'training_plan.json')['clean_replay_sha256']:
        raise ValueError('Training replay differs from frozen plan')
    # Reuse the exact original optimizer, loss, epochs, seed and batch-boundary implementation.
    old_run, old_verify, old_pairs = parent.RUN, parent.verify, parent.training_pairs
    parent.RUN, parent.verify = RUN, lambda: sid
    def clean_pairs(actual_inputs, actual_targets, split):
        if split != 'train' or actual_targets != targets or actual_inputs != inputs:
            raise ValueError('Unexpected original trainer input')
        return replay
    parent.training_pairs = clean_pairs
    try:
        parent.train()
    finally:
        parent.RUN, parent.verify, parent.training_pairs = old_run, old_verify, old_pairs
    complete = load(RUN / 'training_complete.json')
    if complete['updates'] != load(PLAN / 'training_plan.json')['updates']:
        raise ValueError('Optimizer steps differ from control plan')
    common.immutable_json(RUN / 'clean_training_audit.json', dict(signature=sid,
        clean_replay_sha256=audit['clean_replay_sha256'], failed_preferences=0,
        draws=load(PLAN / 'training_plan.json')['draws'], updates=complete['updates'],
        unique_normal_pairs=audit['unique_normal_pairs']))


def load_clean_model():
    sid = verify()
    complete = load(RUN / 'training_complete.json')
    if complete['signature'] != sid or complete['checkpoint_sha256'] != base.sha(RUN / 'adapter.pt'):
        raise ValueError('Clean checkpoint identity changed')
    torch, tokenizer, model, _ = ranker.load(parent.MODEL)
    ranker.load_adapter(model, torch.load(RUN / 'adapter.pt', map_location='cpu', weights_only=True))
    return torch, tokenizer, model


def evaluate_dev():
    sid = verify()
    inputs = load(PLAN / 'public/dev_inputs.json')
    targets = {(r['question_id'], r['arm']): r for r in load(PLAN / 'private/dev_targets.json')}
    torch, tokenizer, model = load_clean_model()
    scores = {}
    with torch.inference_mode():
        for row in inputs:
            ids = ranker.input_ids(tokenizer, row['question'], row['evidence_text'])
            scores[row['question_id'], row['arm']] = float(ranker.logit(torch, tokenizer, model, ids))
    byq = defaultdict(list)
    for row in inputs:
        byq[row['question_id']].append(row)
    rows = []
    for qid, values in byq.items():
        selected = max(values, key=lambda r: scores[qid, r['arm']])
        teacher = max(values, key=lambda r: r['teacher_score'])
        a, b = targets[qid, selected['arm']], targets[qid, teacher['arm']]
        rows.append(dict(question_id=qid, selected_arm=selected['arm'], teacher_arm=teacher['arm'],
            selected_status=a['status'], teacher_status=b['status'],
            selected_f1=a['reward'] if a['status'] == 'ok' else None,
            teacher_f1=b['reward'] if b['status'] == 'ok' else None))
    valid = [r for r in rows if r['selected_f1'] is not None and r['teacher_f1'] is not None]
    deltas = [r['selected_f1'] - r['teacher_f1'] for r in valid]
    report = dict(signature=sid, questions=len(rows), paired_valid=len(valid),
        paired_delta_points=100 * statistics.mean(deltas) if deltas else None,
        wins=sum(d > 1e-9 for d in deltas), ties=sum(abs(d) <= 1e-9 for d in deltas),
        losses=sum(d < -1e-9 for d in deltas), rows=rows,
        checkpoint_selection='none; fixed last epoch', scope='previously exposed 32-question internal dev')
    common.immutable_json(RUN / 'natural_dev_report.json', report)
    print('CLEAN_NATURAL_DEV', {k: v for k, v in report.items() if k != 'rows'}, flush=True)


def select():
    sid = verify()
    torch, tokenizer, model = load_clean_model()
    questions = {r['question_id']: r for r in load(base.PLAN / 'public/queries.json')}
    plans = []
    for path in sorted((precision.RUN / 'sets').glob('*.json')):
        old = load(path)
        qid = old['plans'][0]['question_id']
        q = questions[qid]
        target = RUN / 'selected' / path.name
        if target.exists():
            saved = load(target)
            if saved['signature'] != sid:
                raise ValueError('Selection identity differs')
            plans.extend(saved['plans'])
            continue
        matched = load(controls.RUN / 'selected' / path.name)
        alternatives = dict(clean_full_union=old['alternatives']['union_fp32'],
            clean_wide_single=old['alternatives']['wide_single_fp32'],
            clean_matched_single=matched['alternatives']['matched_single_bf16'],
            clean_matched_union=matched['alternatives']['matched_union_bf16'])
        unique = {}
        for candidates in alternatives.values():
            for row in candidates:
                unique.setdefault(row['evidence_sha256'], row)
        intent = RUN / 'selection_inflight.json'
        if intent.exists():
            raise ValueError('Unknown selector commit; no hidden repeat')
        write_json(intent, dict(signature=sid, question_id=qid))
        scores, lengths, durations = {}, {}, {}
        with torch.inference_mode():
            for digest, row in unique.items():
                ids = ranker.input_ids(tokenizer, q['question'], row['evidence_text'])
                started = time.perf_counter()
                scores[digest] = float(ranker.logit(torch, tokenizer, model, ids))
                durations[digest] = time.perf_counter() - started
                lengths[digest] = len(ids)
        if not all(math.isfinite(v) for v in scores.values()):
            raise ValueError('Nonfinite score')
        chosen = []
        for arm, candidates in alternatives.items():
            best = max(candidates, key=lambda r: scores[r['evidence_sha256']])
            chosen.append(dict(**q, arm=arm, **best, private_inputs_read=False,
                predicted_score=scores[best['evidence_sha256']], scored_sets=len(candidates)))
        saved = dict(signature=sid, plans=chosen, scores=scores,
            costs=dict(forward_calls=len(unique), input_tokens=sum(lengths.values()), seconds=sum(durations.values())),
            independent_arm_costs={arm: dict(forward_calls=len(candidates),
                input_tokens=sum(lengths[r['evidence_sha256']] for r in candidates),
                seconds=sum(durations[r['evidence_sha256']] for r in candidates))
                for arm, candidates in alternatives.items()})
        common.immutable_json(target, saved)
        intent.unlink()
        plans.extend(chosen)
        print('CLEAN_SELECT', qid, len(unique), flush=True)
    if len(plans) != 64 * len(ARMS):
        raise ValueError('Frozen 64-question coverage differs')
    common.immutable_json(RUN / 'deployment/qa_plans.json', plans)


def cache_files():
    return [ROOT / p for p in ['runs/replacement_controls_v3/qa/answer_calls.jsonl',
        'runs/replacement_training_v2/feedback/qa/answer_calls.jsonl',
        'runs/replacement_training_v2/deployment/qa/answer_calls.jsonl']]


def qa(endpoint):
    from .replacement_stage2_cache_repair import known_failure_rows
    sid = verify()
    run = RUN / 'deployment'
    plans = load(run / 'qa_plans.json')
    cache = common.cached_answers(cache_files())
    dest = run / 'qa/predictions.jsonl'
    committed = list(read_jsonl(dest)) if dest.exists() else []
    for row in known_failure_rows(plans, committed, cache, sid):
        base.append(dest, row)
    common.run_qa(run, plans, sid, endpoint, cache_files())


def score():
    sid = verify()
    predictions = list(read_jsonl(RUN / 'deployment/qa/predictions.jsonl'))
    scored = common.official_rows(load(RUN / 'deployment/qa_plans.json'), predictions)
    groups = defaultdict(dict)
    for row in scored:
        groups[row['arm']][row['question_id']] = row
    old_sources = [('old_full_union', precision.RUN / 'scored.json', 'union_bf16'),
        ('old_wide_single', controls.RUN / 'scored.json', 'wide_bf16'),
        ('old_matched_single', controls.RUN / 'scored.json', 'matched_single_bf16'),
        ('old_matched_union', controls.RUN / 'scored.json', 'matched_union_bf16'),
        ('static_fusion', ROOT / 'runs/evidence_acquisition_v2/scored.json', 'fusion_2k'),
        ('semantic_mixed', precision.RUN / 'scored.json', 'reranker_mixed_bf16_serial')]
    for name, path, arm in old_sources:
        groups[name] = {r['question_id']: r for r in load(path) if r['arm'] == arm}
    comparisons = []
    pairs = [(a, b) for a, b in zip(ARMS, ['old_full_union', 'old_wide_single',
                                           'old_matched_single', 'old_matched_union'])]
    pairs += [('clean_full_union', b) for b in ['static_fusion', 'semantic_mixed', 'clean_wide_single']]
    pairs.append(('clean_matched_union', 'clean_matched_single'))
    for a, b in pairs:
        if set(groups[a]) != set(groups[b]) or len(groups[a]) != 64:
            raise ValueError('Comparison coverage differs: ' + a + ':' + b)
        differences = {q: groups[a][q]['official_native_f1'] - groups[b][q]['official_native_f1'] for q in groups[a]}
        histories = defaultdict(list)
        for q, delta in differences.items():
            histories[groups[a][q]['conversation_id']].append(delta)
        comparisons.append(dict(arm=a, comparator=b, delta_points=100 * statistics.mean(differences.values()),
            wins=sum(d > 1e-9 for d in differences.values()), ties=sum(abs(d) <= 1e-9 for d in differences.values()),
            losses=sum(d < -1e-9 for d in differences.values()),
            history_delta_points={h: 100 * statistics.mean(v) for h, v in histories.items()}))
    costs = [load(p) for p in (RUN / 'selected').glob('*.json')]
    report = dict(signature=sid, complete=True, primary_arm='clean_full_union',
        scores=[dict(arm=a, n=len(groups[a]), f1_points=100 * statistics.mean(
            r['official_native_f1'] for r in groups[a].values()),
            statuses=dict(Counter(r['status'] for r in groups[a].values()))) for a in groups],
        comparisons=comparisons, training_plan=load(PLAN / 'training_plan.json'),
        selector_actual_costs={k: sum(r['costs'][k] for r in costs) for k in ['forward_calls', 'input_tokens', 'seconds']},
        per_arm_selector_costs={a: {k: sum(r['independent_arm_costs'][a][k] for r in costs)
            for k in ['forward_calls', 'input_tokens', 'seconds']} for a in ARMS},
        natural_dev=load(RUN / 'natural_dev_report.json'), checkpoint_selection='fixed last epoch',
        heldout_accessed=False, paper_ready=False, scope=CONFIG['scope'])
    common.immutable_json(RUN / 'scored.json', scored)
    common.immutable_json(RUN / 'report.json', report)
    print('CLEAN_CONTROL_FINAL', report['scores'], flush=True)


def status():
    value = {}
    for name, relative in [('queue', 'queue_status.json'), ('process', 'queue_process.json'),
        ('training_started', 'training_started.json'), ('training_complete', 'training_complete.json'),
        ('natural_dev', 'natural_dev_report.json'), ('deployment', 'deployment/qa/status.json')]:
        path = RUN / relative
        value[name] = load(path) if path.exists() else None
    log = RUN / 'training_log.jsonl'
    value['training_last'] = list(read_jsonl(log))[-1] if log.exists() and log.stat().st_size else None
    value['selected_questions'] = len(list((RUN / 'selected').glob('*.json')))
    value['report_available'] = (RUN / 'report.json').exists()
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'freeze', 'verify', 'preflight', 'train',
        'evaluate_dev', 'select', 'qa', 'score', 'status'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8018')
    args = parser.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    elif args.mode == 'verify':
        print('VERIFIED_CLEAN_CONTROL', verify(), flush=True)
    else:
        globals()[args.mode]()
