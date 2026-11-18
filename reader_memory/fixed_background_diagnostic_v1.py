"""Bounded, fixed-background deletion diagnostic on existing MuSiQue dev sources."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import itertools
import json
from pathlib import Path
import re
from types import SimpleNamespace

from . import baseline_bootstrap_v2 as base
from . import evidence_acquisition_v1 as evidence
from . import replacement_stage2_common as common
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/fixed_background_diagnostic_v1'
RUN = ROOT / 'runs/fixed_background_diagnostic_v1'
DESIGN = ROOT / 'research/replacement_training_v2/diagnosis_20261029/next_diagnostic_design.json'
OLD = ROOT / 'research/replacement_learning_v1'
CONFIG = dict(version='fixed-background-diagnostic-v1', max_conditions=160,
              reader='Qwen3.5-9B', output_tokens=2048, evidence_cap=2048,
              decoding='bf16-sdpa-greedy-enable_thinking_false', fill_after_delete=False,
              full_success_minimum=8, strict_dependency_minimum=4, heldout_accessed=False,
              control_selection='closest character length to A, then fixed hash; no answer feedback',
              semantic_copy_completeness_claimed=False, automatic_training=False)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def erase_exact(records, texts, only_keys=None):
    """Delete all literal copies in record bodies; preserve unrelated records and order."""
    changed = deepcopy(records)
    audit = []
    for key, record in list(changed.items()):
        if only_keys is not None and key not in only_keys:
            continue
        touched = False
        for text in dict.fromkeys(texts):
            count = 0
            if record['kind'] == 'raw':
                for turn in record['turns']:
                    n = turn['content'].count(text)
                    if n:
                        turn['content'] = turn['content'].replace(text, '').strip()
                        count += n
            else:
                content = record['entry']['lossless_restatement']
                count = content.count(text)
                if count:
                    record['entry']['lossless_restatement'] = content.replace(text, '').strip()
            if not count:
                continue
            touched = True
            audit.append(dict(entry_id=key, exact_text=text, occurrences=count))
            for uid, value in list(record.get('source_spans', {}).items()):
                if text in value:
                    remaining = value.replace(text, '').strip()
                    if remaining:
                        record['source_spans'][uid] = remaining
                    else:
                        del record['source_spans'][uid]
        if not touched:
            continue
        if record['kind'] == 'raw':
            record['turns'] = [t for t in record['turns'] if t['content']]
            record['feature_text'] = ' '.join(t['content'] for t in record['turns'])
        else:
            # Extractive entries whose only sentence disappeared must not leave a bare title.
            if not record.get('source_spans'):
                record['entry']['lossless_restatement'] = ''
            record['feature_text'] = record['entry']['lossless_restatement']
        if not record['feature_text'].strip():
            del changed[key]
    return changed, audit


def prepare():
    from tools import prepare_replacement_learning_v1 as original
    if (PLAN / 'manifest.json').exists():
        raise ValueError('Prepared diagnostic is frozen')
    design = load(DESIGN)
    if fingerprint({k: v for k, v in design.items() if k != 'signature'}) != design['signature']:
        raise ValueError('Design signature mismatch')
    protocol = DESIGN.parent / 'NEXT_DIAGNOSTIC_PROTOCOL.md'
    if base.sha(protocol) != design['protocol_sha256'] or base.sha(original.SOURCE) != original.SOURCE_SHA:
        raise ValueError('Frozen protocol or training source changed')
    allocation = load(OLD / 'private/source_selection.json')
    if set(design['sources']) != set(allocation['dev']):
        raise ValueError('Source boundary changed')
    sources = {}
    with original.SOURCE.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['id'] in set(design['sources']):
                sources[row['id']] = row
    if set(sources) != set(design['sources']):
        raise ValueError('Missing authorized dev sources')
    lineage = {r['question_id']: r for r in load(OLD / 'private/lineage.json')}
    previous = load(OLD / 'public/reward_inputs.json')
    references = load(OLD / 'private/references.json')
    counter = original.ExactCounter()
    plans, audits, reviews = [], [], []
    for qid in design['question_ids']:
        native, mode = qid.rsplit(':', 1)
        source, old = sources[native], lineage[qid]
        trace = {t['variant']: t for t in old['trace']}
        records, units = original.records_for(source)
        a_units = sorted({a['unit'] for a in trace['joint_copy_remove']['interventions']})
        b_units = sorted({a['unit'] for a in trace.get('other_hop_joint_remove', {}).get('interventions', [])})
        if len(a_units) != 1 or (mode == 'multi' and len(b_units) != 1):
            raise ValueError('Original intervention target is ambiguous: ' + qid)
        a, b = a_units[0], b_units[0] if b_units else None
        old_rows = [p for p in previous if p['question_id'] == qid]
        question = old_rows[0]['question']
        # Reproduce the original support pack, including cases deduplicated under fusion.
        scores = original.core.lexical_scores(question, records, list(records))
        ranked = sorted(records, key=lambda k: (-scores[k], k))
        raw = [k for k in ranked if records[k]['kind'] == 'raw'][:14]
        compact = [k for k in ranked if records[k]['kind'] != 'raw'][:28]
        pool = [k for pair in itertools.zip_longest(raw, compact) for k in pair if k]
        needed = [d['paragraph_support_idx'] for d in source['question_decomposition']]
        if mode == 'single':
            needed = needed[:1]
        order = list(dict.fromkeys(['raw:' + str(i) for i in needed] + ['extract:' + a] + pool))
        full_ids = evidence.ordered_pack(order, records, counter, 2048)
        full_text = evidence.render(full_ids, records)
        matches = [p for p in old_rows if p['evidence_text'] == full_text]
        if not matches:
            raise ValueError('Reconstructed complete context differs from all frozen contexts: ' + qid)
        match = matches[0]
        if trace[match['variant']]['selected_ids'] != full_ids:
            raise ValueError('Full record order differs from original lineage')
        selected_records = {k: records[k] for k in full_ids}
        cases = [('full', [], None), ('noA_raw', [a], {'raw:' + str(units[a]['paragraph_idx'])}),
                 ('noA_joint', [a], None)]
        if b:
            cases += [('noB_joint', [b], None), ('noAB_joint', [a, b], None)]
        nonsupport = {p['idx'] for p in source['paragraphs'] if not p['is_supporting']}
        control_units = [u for u, value in units.items() if value['paragraph_idx'] in nonsupport
                         and any(value['text'] in r['feature_text'] for r in selected_records.values())]
        control = min(control_units, key=lambda u: (abs(len(units[u]['text']) - len(units[a]['text'])),
                      fingerprint([design['signature'], qid, u]))) if control_units else None
        if control:
            cases.append(('delete_nonsupport_control', [control], None))
        ref = references[qid]
        answers = list(dict.fromkeys([ref['answer']] + ref.get('aliases', [])))
        checks = []
        for arm, removed, only in cases:
            texts = [units[u]['text'] for u in removed]
            edited, changes = erase_exact(selected_records, texts, only)
            selected = [k for k in full_ids if k in edited]
            if not set(selected) <= set(full_ids) or selected != [k for k in full_ids if k in selected]:
                raise ValueError('Intervention added or reordered records')
            changed_ids = {v['entry_id'] for v in changes}
            if any(edited[k] != selected_records[k] for k in selected if k not in changed_ids):
                raise ValueError('Background record changed')
            text = evidence.render(selected, edited)
            size = counter.count(text)
            if not text.strip() or size > 2048:
                raise ValueError('Invalid rendered context budget')
            remaining = [t for t in texts if t in text]
            if only is None and remaining:
                raise ValueError('A claimed exact copy remains in the rendered context')
            plans.append(dict(question_id=qid, native_id=native, mode=mode, split='dev', question=question,
                              arm=arm, selected_ids=selected, contexts=evidence.materialize(selected, edited),
                              evidence_text=text, evidence_tokens=size, evidence_sha256=fingerprint(text)))
            audits.append(dict(question_id=qid, arm=arm, source_full_variant=match['variant'],
                               full_selected_ids=full_ids, removed_unit_ids=removed,
                               interventions=changes, exact_copy_remains=bool(remaining),
                               filler_added=False, unchanged_background_verified=True,
                               tokens_removed=counter.count(full_text) - size))
            title_hits = sorted({t['speaker'] for r in edited.values() if r['kind'] == 'raw'
                                 for t in r['turns'] if any(v.lower() in t['speaker'].lower() for v in answers)})
            answer_sentences = [u['text'] for u in units.values() if u['text'] in text
                                and any(v.lower() in u['text'].lower() for v in answers)]
            checks.append(dict(arm=arm, title_answer_hits=title_hits,
                               remaining_answer_sentences=list(dict.fromkeys(answer_sentences))))
        reviews.append(dict(question_id=qid, question=question, reference=ref, A=units[a],
                            B=units[b] if b else None, control=units[control] if control else None,
                            source_full_variant=match['variant'], checks=checks,
                            semantic_alternative_evidence='not ruled out; no insufficiency label forced',
                            review_role='assistant source audit, not independent human gold'))
    core_count = sum(p['arm'] != 'delete_nonsupport_control' for p in plans)
    if core_count != 128 or len(plans) > CONFIG['max_conditions']:
        raise ValueError('Frozen condition cap changed')
    common.immutable_json(PLAN / 'public/qa_plans.json', plans)
    common.immutable_json(PLAN / 'private/references.json', {q: references[q] for q in design['question_ids']})
    common.immutable_json(PLAN / 'private/interventions.json', audits)
    common.immutable_json(PLAN / 'private/source_review.json', reviews)
    common.immutable_json(PLAN / 'preparation.json', dict(questions=32, sources=16, core_conditions=core_count,
        total_conditions=len(plans), arms=dict(Counter(p['arm'] for p in plans)), heldout_accessed=False,
        full_variants=dict(Counter(r['source_full_variant'] for r in reviews)),
        exact_tokenization=True, background_checks_passed=True, no_answers_used_to_select_sources=True))
    print('PREPARED_FIXED_BACKGROUND', len(plans), 'conditions', flush=True)


def freeze():
    paths = ['reader_memory/fixed_background_diagnostic_v1.py', 'scripts/fixed_background_diagnostic_v1_queue.py',
             'tests/test_fixed_background_diagnostic_v1.py', 'tools/prepare_replacement_learning_v1.py',
             'reader_memory/mainline_service_v1.py', 'reader_memory/backend.py', 'reader_memory/schema.py',
             'reader_memory/evidence_acquisition_v1.py', 'reader_memory/evidence_acquisition_v2.py',
             'reader_memory/replacement_precision_v2.py', 'reader_memory/replacement_learning_v1.py',
             'reader_memory/replacement_stage2_common.py', 'reader_memory/replacement_stage2_cache_repair.py',
             'reader_memory/replacement_core_v1.py', 'reader_memory/baseline_bootstrap_v2.py',
             'third_party/simplemem_official/simplemem/core/answer_generator.py',
             'research/fixed_background_diagnostic_v1/IMPLEMENTATION_NOTES.md',
             'research/fixed_background_diagnostic_v1/SOURCE_AUDIT.md']
    paths += [p.relative_to(ROOT).as_posix() for p in [DESIGN, DESIGN.parent / 'NEXT_DIAGNOSTIC_PROTOCOL.md']]
    paths += ['research/fixed_background_diagnostic_v1/' + p for p in
              ['public/qa_plans.json', 'private/references.json', 'private/interventions.json',
               'private/source_review.json', 'preparation.json']]
    value = dict(config=CONFIG, design_signature=load(DESIGN)['signature'],
                 files={p: base.sha(ROOT / p) for p in paths})
    value['signature'] = fingerprint(value)
    common.immutable_json(PLAN / 'manifest.json', value)
    print('FROZEN_FIXED_BACKGROUND', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or fingerprint({k: v for k, v in value.items() if k != 'signature'}) != value['signature']:
        raise ValueError('Diagnostic identity changed')
    for path, digest in value['files'].items():
        if base.sha(ROOT / path) != digest:
            raise ValueError('Frozen diagnostic dependency changed: ' + path)
    return value['signature']


def preflight():
    from tools.prepare_replacement_learning_v1 import ExactCounter
    sid = verify()
    counter = ExactCounter()
    plans = load(PLAN / 'public/qa_plans.json')
    seen = set()
    for row in plans:
        key = row['question_id'], row['arm']
        if key in seen:
            raise ValueError('Duplicate condition')
        seen.add(key)
        rendered = evidence.FORMAT(None, [SimpleNamespace(**c) for c in row['contexts']])
        if rendered != row['evidence_text'] or fingerprint(rendered) != row['evidence_sha256']:
            raise ValueError('Context rendering differs')
        if counter.count(rendered) != row['evidence_tokens'] or row['evidence_tokens'] > 2048:
            raise ValueError('Exact tokenizer differs')
    write_json(RUN / 'preflight.json', dict(signature=sid, conditions=len(plans),
               tokenization_verified=True, heldout_accessed=False))
    print('FIXED_BACKGROUND_PREFLIGHT_OK', len(plans), flush=True)


def cache_files():
    return [ROOT / p for p in ['runs/replacement_controls_v3/qa/answer_calls.jsonl',
        'runs/replacement_training_v2/feedback/qa/answer_calls.jsonl',
        'runs/replacement_training_v2/deployment/qa/answer_calls.jsonl']]


def qa(endpoint):
    from .replacement_stage2_cache_repair import known_failure_rows
    sid = verify()
    plans = load(PLAN / 'public/qa_plans.json')
    cache = common.cached_answers(cache_files())
    dest = RUN / 'qa/predictions.jsonl'
    committed = list(read_jsonl(dest)) if dest.exists() else []
    # A known failed request is missing feedback, never an implicit retry or negative label.
    for row in known_failure_rows(plans, committed, cache, sid):
        base.append(dest, row)
    common.run_qa(RUN, plans, sid, endpoint, cache_files())


def diagnostic_gate(full_correct, strict_count):
    if full_correct < CONFIG['full_success_minimum']:
        return 'stop_dependency_training_check_materials_or_reader'
    if strict_count < CONFIG['strict_dependency_minimum']:
        return 'stop_dependency_training_on_this_small_sample'
    return 'review_alternative_evidence_and_controls_before_any_training'


def score():
    sid = verify()
    plans = load(PLAN / 'public/qa_plans.json')
    predictions = list(read_jsonl(RUN / 'qa/predictions.jsonl'))
    refs = load(PLAN / 'private/references.json')
    expected = {(p['question_id'], p['arm']) for p in plans}
    if len(predictions) != len(expected) or {(p['question_id'], p['arm']) for p in predictions} != expected:
        raise ValueError('Complete condition coverage required')
    scored, byq, groups = [], defaultdict(dict), defaultdict(list)
    for row in predictions:
        if row['signature'] != sid:
            raise ValueError('Prediction identity mismatch')
        ref = refs[row['question_id']]
        answers = list(dict.fromkeys([ref['answer']] + ref.get('aliases', [])))
        value = max(base.native_f1(row['hypothesis'], a, 2) for a in answers) if row['status'] == 'ok' else None
        result = dict(**row, native_f1=value)
        scored.append(result)
        byq[row['question_id']][row['arm']] = result
        groups[row['mode'], row['arm']].append(result)
    summary = []
    for (mode, arm), rows in sorted(groups.items()):
        valid = [r for r in rows if r['native_f1'] is not None]
        summary.append(dict(mode=mode, arm=arm, conditions=len(rows), valid=len(valid),
                           failed=len(rows) - len(valid), fully_correct=sum(r['native_f1'] == 1 for r in valid),
                           mean_f1_valid_points=100 * sum(r['native_f1'] for r in valid) / len(valid) if valid else None,
                           mean_evidence_tokens=sum(r['evidence_tokens'] for r in rows) / len(rows)))
    multi = {q: rows for q, rows in byq.items() if q.endswith(':multi')}
    required = ['full', 'noA_joint', 'noB_joint', 'noAB_joint']
    strict, contrasts = [], []
    full_correct = sum(rows['full']['native_f1'] == 1 for rows in multi.values())
    for q, rows in multi.items():
        values = [rows[a]['native_f1'] for a in required]
        if all(v is not None for v in values):
            strict_case = values == [1., 0., 0., 0.]
            contrasts.append(dict(question_id=q, rewards=values,
                                  four_context_contrast=values[0] - values[1] - values[2] + values[3],
                                  strict_pattern=strict_case, semantic_dependency_confirmed=False))
            if strict_case:
                strict.append(q.rsplit(':', 1)[0])
    report = dict(signature=sid, complete=True, scope='Existing internal dev; descriptive dependency diagnostic',
                  conditions=len(scored), statuses=dict(Counter(r['status'] for r in scored)), scores=summary,
                  multi_full_correct=full_correct, multi_sources=16,
                  multi_full_failed=sum(rows['full']['native_f1'] is None for rows in multi.values()),
                  complete_four_condition_sources=len(contrasts), strict_dependency_sources=sorted(set(strict)),
                  strict_dependency_count=len(set(strict)), contrasts=contrasts,
                  decision=diagnostic_gate(full_correct, len(set(strict))),
                  failed_feedback_is_missing=True, same_budget_claimed=False, heldout_accessed=False,
                  automatic_training=False, human_gold_claimed=False,
                  review_required='Titles, alternative evidence, direct controls and answer wording must be checked')
    common.immutable_json(RUN / 'scored.json', scored)
    common.immutable_json(RUN / 'report.json', report)
    print('FIXED_BACKGROUND_RESULT', json.dumps(report, ensure_ascii=False), flush=True)


def status():
    value = dict(plan=load(PLAN / 'preparation.json') if (PLAN / 'preparation.json').exists() else None)
    for key, path in [('queue', RUN / 'queue_status.json'), ('process', RUN / 'queue_process.json'),
                      ('qa', RUN / 'qa/status.json'), ('report', RUN / 'report.json')]:
        value[key] = load(path) if path.exists() else None
    predictions = list(read_jsonl(RUN / 'qa/predictions.jsonl')) if (RUN / 'qa/predictions.jsonl').exists() else []
    calls = list(read_jsonl(RUN / 'qa/answer_calls.jsonl')) if (RUN / 'qa/answer_calls.jsonl').exists() else []
    value['committed_conditions'] = len(predictions)
    value['new_reader_calls'] = len(calls)
    value['cached_conditions'] = sum(r['reused'] for r in predictions)
    value['inflight'] = (RUN / 'qa/inflight.json').exists()
    if value['process']:
        import os
        try:
            os.kill(value['process']['pid'], 0)
            value['process_alive'] = True
        except ProcessLookupError:
            value['process_alive'] = False
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'freeze', 'verify', 'preflight', 'qa', 'score', 'status'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8018')
    args = parser.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    elif args.mode == 'verify':
        print('VERIFIED_FIXED_BACKGROUND', verify(), flush=True)
    else:
        globals()[args.mode]()
