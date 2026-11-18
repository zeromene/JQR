"""Bounded training-data pilot: literal cross-representation copies vs matched edits.

This pilot collects QA feedback only. It neither trains a ranker nor opens the
reserved validation sources. Privileged annotations stay in data preparation.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from . import baseline_bootstrap_v2 as base
from . import evidence_acquisition_v1 as evidence
from . import replacement_precision_v2 as qa_ledger
from . import replacement_stage2_common as common
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/copy_supervision_pilot_v1'
RUN = ROOT / 'runs/copy_supervision_pilot_v1'
ARMS = ('full', 'delete_raw', 'delete_memory', 'delete_group', 'delete_copies',
        'replace_raw', 'replace_memory', 'replace_group', 'replace_copies')
CONFIG = dict(version='copy-supervision-pilot-v1', seed='copy-unit-20261029-v1',
    sources=12, source_hops={'3': 2, '2': 10}, questions=24, conditions=216,
    reader='Qwen3.5-9B', output_tokens=2048, evidence_cap=2048,
    full_context_cap=1920, target_sentence_tokens=[12, 120],
    max_replacement_delta_tokens=8, group_removed_relative_tolerance=.10,
    group_removed_absolute_tolerance=8,
    decoding='bf16-sdpa-greedy-enable_thinking_false',
    gate=dict(normal_conditions_min=206, usable_full_questions_min=16,
              usable_full_multi_min=6, usable_full_sources_min=8,
              useful_shared_questions_min=8, useful_shared_sources_min=6,
              useful_shared_multi_sources_min=3, preference_min_f1_gap=.25),
    source_selection='hash, hop quota, structural eligibility; no model feedback',
    ordinary_group='same raw edit plus different compact sentence closest in cost; no utility filtering',
    single_training_policy='raw and memory edits balanced with equal total presentations, fixed after pilot',
    training=False, automatic_training=False, heldout_accessed=False,
    scope='new MuSiQue train sources; gold-staged support; literal extractive copies; data qualification only')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze():
    names = ['reader_memory/copy_supervision_pilot_v1.py',
             'tools/prepare_copy_supervision_pilot_v1.py',
             'tools/screen_copy_supervision_sources_v1.py',
             'scripts/copy_supervision_pilot_v1_queue.py',
             'tests/test_copy_supervision_pilot_v1.py',
             'tools/prepare_replacement_learning_v1.py',
             'reader_memory/fixed_background_diagnostic_v1.py',
             'reader_memory/replacement_core_v1.py',
             'reader_memory/replacement_ranker_v1.py',
             'reader_memory/replacement_precision_v2.py',
             'reader_memory/replacement_learning_v1.py',
             'reader_memory/replacement_stage2_common.py',
             'reader_memory/evidence_acquisition_v1.py',
             'reader_memory/evidence_acquisition_v2.py',
             'reader_memory/mainline_service_v1.py', 'reader_memory/backend.py',
             'reader_memory/baseline_bootstrap_v2.py', 'reader_memory/schema.py',
             'reader_memory/_vendor/locomo_native_f1.py',
             'third_party/simplemem_official/simplemem/core/answer_generator.py',
             'third_party/simplemem_official/simplemem/core/utils/llm_client.py',
             'models/Qwen3.5-9B/tokenizer.json', 'models/Qwen3.5-9B/source_manifest.json',
             'models/Qwen3-Reranker-4B/artifact_manifest.json']
    names += ['research/copy_supervision_pilot_v1/' + name for name in
              ['PROTOCOL.md', 'preparation.json', 'public/qa_plans.json',
               'private/references.json', 'private/interventions.json',
               'private/source_review.json', 'private/source_selection.json',
               'private/prior_exposure.json', 'private/semantic_screening.json',
               'private/expanded_screening.json']]
    names += [p.relative_to(ROOT).as_posix() for p in
              sorted((PLAN / 'preparation_attempts/01_structural_only').rglob('*.json'))]
    manifest = dict(config=CONFIG, files={n: base.sha(ROOT / n) for n in names})
    manifest['signature'] = fingerprint(manifest)
    common.immutable_json(PLAN / 'manifest.json', manifest)
    print('FROZEN_COPY_SUPERVISION_PILOT', manifest['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint(
            {k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Pilot identity changed')
    for name, digest in value['files'].items():
        if base.sha(ROOT / name) != digest:
            raise ValueError('Frozen pilot dependency changed: ' + name)
    return value['signature']


def preflight():
    from tools.prepare_replacement_learning_v1 import ExactCounter
    sid = verify()
    counter = ExactCounter()
    plans = load(PLAN / 'public/qa_plans.json')
    refs = load(PLAN / 'private/references.json')
    audits = load(PLAN / 'private/interventions.json')
    prep = load(PLAN / 'preparation.json')
    if prep['source_hops'] != CONFIG['source_hops'] or prep['prior_exposed_sources'] < 749:
        raise ValueError('Source allocation/configuration mismatch')
    by = defaultdict(list)
    for p in plans:
        by[p['question_id']].append(p)
        if p['evidence_sha256'] != fingerprint(p['evidence_text']):
            raise ValueError('Evidence hash mismatch')
        if counter.count(p['evidence_text']) != p['evidence_tokens'] or p['evidence_tokens'] > 2048:
            raise ValueError('Exact budget mismatch')
        if evidence.FORMAT(None, [__import__('types').SimpleNamespace(**c) for c in p['contexts']]) != p['evidence_text']:
            raise ValueError('Context formatter mismatch')
        if set(p) != {'question_id', 'native_id', 'mode', 'split', 'question', 'arm',
                       'contexts', 'evidence_text', 'evidence_tokens', 'evidence_sha256'}:
            raise ValueError('Unexpected public fields (possible annotation leak)')
    if len(plans) != CONFIG['conditions'] or len(by) != CONFIG['questions'] or set(by) != set(refs):
        raise ValueError('Wrong pilot coverage')
    if len(audits) != len(plans):
        raise ValueError('Incomplete intervention audit')
    for q, rows in by.items():
        if Counter(r['arm'] for r in rows) != Counter(ARMS):
            raise ValueError('Repeated or missing conditions: ' + q)
        full = next(r for r in rows if r['arm'] == 'full')
        if any(abs(r['evidence_tokens'] - full['evidence_tokens']) > CONFIG['max_replacement_delta_tokens']
               for r in rows if r['arm'].startswith('replace_')):
            raise ValueError('Replacement cost mismatch')
    if any(not a['unchanged_background_verified'] or not a['slot_order_verified'] for a in audits):
        raise ValueError('Background/slot audit failed')
    if not (ROOT / 'models/Qwen3.5-9B/.verified.json').exists():
        raise ValueError('Offline reader verification marker missing')
    write_json(RUN / 'preflight.json', dict(signature=sid, passed=True, conditions=len(plans),
        sources=12, training=False, heldout_accessed=False))
    print('PILOT_PREFLIGHT_OK', len(plans), flush=True)


def feedback(endpoint):
    # Reuse the established single-request transaction ledger without legacy-cache
    # loading: all questions are new, and repeated requests within this run reuse
    # the ledger's own exact-request cache. Existing outcomes are never re-asked.
    sid = verify()
    plans = load(PLAN / 'public/qa_plans.json')
    common.immutable_json(RUN / 'qa_plans.json', plans)
    common.immutable_json(RUN / 'selection_complete.json', dict(signature=sid,
        plans_sha256=base.sha(RUN / 'qa_plans.json')))
    old = qa_ledger.RUN, qa_ledger.verify, qa_ledger.saved_cache
    qa_ledger.RUN, qa_ledger.verify, qa_ledger.saved_cache = RUN, lambda: sid, lambda: {}
    try:
        qa_ledger.qa(endpoint)
    finally:
        qa_ledger.RUN, qa_ledger.verify, qa_ledger.saved_cache = old


def qualify(scored):
    """Predeclared data gate, never a gate on the desired deletion effect's sign."""
    gate = CONFIG['gate']
    by = defaultdict(dict)
    for row in scored:
        by[row['question_id']][row['arm']] = row
    usable, shared, multi_sources = [], [], set()
    pair_counts = Counter()
    for q, rows in by.items():
        # A normal shared full context must actually be answerable by this reader.
        full = rows['full']
        if full['status'] != 'ok' or full['local_native_f1'] < .5:
            continue
        usable.append(full)
        # One common eligibility set for all three prospective training arms.
        family = [rows[a] for a in ('full', 'replace_raw', 'replace_memory',
                                    'replace_group', 'replace_copies')]
        if any(r['status'] != 'ok' for r in family):
            continue
        gaps = {a: abs(full['local_native_f1'] - rows[a]['local_native_f1']) for a in
                ('replace_raw', 'replace_memory', 'replace_group', 'replace_copies')}
        for a, gap in gaps.items():
            pair_counts[a] += gap >= gate['preference_min_f1_gap']
        if max(gaps.values()) >= gate['preference_min_f1_gap']:
            shared.append(full)
            if full['mode'] == 'multi':
                multi_sources.add(full['native_id'])
    counts = dict(normal_conditions=sum(r['status'] == 'ok' for r in scored),
        usable_full_questions=len(usable), usable_full_multi=sum(r['mode'] == 'multi' for r in usable),
        usable_full_sources=len({r['native_id'] for r in usable}),
        useful_shared_questions=len(shared), useful_shared_sources=len({r['native_id'] for r in shared}),
        useful_shared_multi_sources=len(multi_sources))
    checks = {k: counts[k] >= gate[k + '_min'] for k in counts}
    return dict(passed=all(checks.values()), counts=counts, checks=checks,
                normal_preference_pairs_by_action=dict(pair_counts),
                meaning='Data usable for fixing a bounded three-arm training protocol, not mechanism success',
                automatic_training=False)


def score():
    sid = verify()
    plans = load(PLAN / 'public/qa_plans.json')
    predictions = list(read_jsonl(RUN / 'qa/predictions.jsonl'))
    expected = {(p['question_id'], p['arm']) for p in plans}
    if len(predictions) != len(expected) or {(p['question_id'], p['arm']) for p in predictions} != expected:
        raise ValueError('Complete, unique feedback coverage required')
    refs = load(PLAN / 'private/references.json')
    scored = []
    for p in predictions:
        if p['signature'] != sid:
            raise ValueError('Wrong feedback identity')
        plan = next(x for x in plans if (x['question_id'], x['arm']) == (p['question_id'], p['arm']))
        if p['plan_sha256'] != fingerprint(plan) or any(p[k] != plan[k] for k in plan):
            raise ValueError('Feedback changed a frozen context or question')
        ref = refs[p['question_id']]
        answers = list(dict.fromkeys([ref['answer']] + ref['aliases']))
        f1 = max(base.native_f1(p['hypothesis'], a, 2) for a in answers) if p['status'] == 'ok' else 0.
        scored.append(dict(**p, local_native_f1=f1))
    groups = defaultdict(list)
    for p in scored:
        for scope in ('overall', 'mode:' + p['mode']):
            groups[p['arm'], scope].append(p)
    summaries = [dict(arm=a, scope=s, n=len(rows),
        f1_points=100 * statistics.mean(r['local_native_f1'] for r in rows),
        status_counts=dict(Counter(r['status'] for r in rows))) for (a, s), rows in sorted(groups.items())]
    normal = defaultdict(dict)
    for p in scored:
        normal[p['question_id']][p['arm']] = p
    contrasts = []
    for a, b in [('delete_copies', 'delete_raw'), ('delete_copies', 'delete_group'),
                 ('replace_copies', 'replace_raw'), ('replace_copies', 'replace_group')]:
        values = [rows[a]['local_native_f1'] - rows[b]['local_native_f1'] for rows in normal.values()
                  if rows[a]['status'] == rows[b]['status'] == 'ok']
        contrasts.append(dict(arm=a, comparator=b, normal_pairs=len(values),
            delta_f1_points=100 * statistics.mean(values) if values else None,
            interpretation='Descriptive feedback, not training or independent mechanism validation'))
    calls = list(read_jsonl(RUN / 'qa/answer_calls.jsonl'))
    qualification = qualify(scored)
    report = dict(signature=sid, complete=True, scores=summaries, contrasts=contrasts,
        qualification=qualification, new_reader_calls=len(calls),
        reused_conditions=sum(p['reused'] for p in scored),
        call_seconds=sum(c.get('seconds', 0) for c in calls),
        prompt_tokens=sum(c.get('result', {}).get('usage', {}).get('prompt_tokens', 0) for c in calls),
        output_tokens=sum(c.get('result', {}).get('usage', {}).get('completion_tokens', 0) for c in calls),
        training=False, independent_validation=False, heldout_accessed=False,
        scope=CONFIG['scope'], next_action='Fix training size/update budget once' if qualification['passed']
            else 'Stop this data route; inspect failures and withdraw unsupported assumptions')
    write_json(RUN / 'scored.json', scored)
    write_json(RUN / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'preflight', 'feedback', 'score', 'verify'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8018')
    args = parser.parse_args()
    if args.mode == 'feedback':
        feedback(args.endpoint)
    else:
        result = globals()[args.mode]()
        if result:
            print(result)
