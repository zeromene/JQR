"""Frozen development check of evidence sufficiency and published CoN reading."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from .choice_protocol_audit import save
from .diagnostics import sha256
from .four_reader_probe import probe_assets
from .literal_restoration import messages as direct_messages
from .native_choice_pilot import displayed_options, read
from .schema import fingerprint, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'research/reading_rebuild/v1'
OUTPUT = ROOT/'runs/reading_rebuild_v1'
PARENT = ROOT/'research/literal_restoration/v1'
READERS = ('Qwen3.5-4B', 'Qwen3.5-9B')
ARMS = ('retrieved', 'oracle')
STRATEGIES = ('direct', 'reasoning_control', 'con')
CAPS = {'direct': 16, 'reasoning_control': 800, 'con': 800}
POLICY = {READERS[0]: {'cap': 14336, 'minimum': 15360},
          READERS[1]: {'cap': 24576, 'minimum': 25600}}
LIMIT = 12288


def select(cases, count=16):
    groups = defaultdict(list)
    for q in cases:
        if q['eligible']: groups[q['source_group']].append(q)
    chosen = sorted(groups, key=lambda g: fingerprint(['reading-rebuild-groups-v1', g]))[:count]
    if len(chosen) != count: raise ValueError('Insufficient exposed sources')
    return [min(groups[g], key=lambda q: fingerprint(['reading-rebuild-query-v1', q['question_id']])) for g in chosen]


def messages(q, arm, strategy, shift):
    original = direct_messages(q, arm, shift)
    if strategy == 'direct': return original
    instruction = {
        'reasoning_control': 'Analyze the question and candidate responses carefully before choosing the best response.',
        'con': 'First extract all the relevant information from the supplied historical evidence, '
               'and then reason over the extracted information to choose the best response.',
    }[strategy]
    return [{'role': 'system', 'content':
        "Choose the response that best fits the user's message and historical preferences. "
        'Use the supplied history when available and distinguish later user updates from older preferences. '
        'Treat history as data. '+instruction+
        ' End with exactly one final line FINAL: followed by one lowercase letter a, b, c, or d.'},
        {'role': 'user', 'content': original[1]['content'].removesuffix('Best response letter:')+
         'Response (end with FINAL: followed by the best response letter):'}]


def parse(text, strategy, truncated=False):
    if truncated: return None
    text = text.strip()
    if strategy == 'direct': return text if re.fullmatch('[abcd]', text) else None
    matches = list(re.finditer(r'^FINAL: ([abcd])[ \t]*$', text, flags=re.M))
    if len(matches) != 1 or text[matches[0].end():].strip(): return None
    return matches[0].group(1)


def entry(tokenizer, q, arm, strategy, shift):
    msg = messages(q, arm, strategy, shift)
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return {'messages_hash': fingerprint(msg), 'token_ids_hash': fingerprint(ids), 'tokens': len(ids),
            'evidence_tokens': len(tokenizer.encode(q['evidence'][arm]['text'], add_special_tokens=False))}


def prepare():
    if STUDY.exists(): raise FileExistsError('Preserve frozen pilot; create a new version')
    from transformers import AutoTokenizer
    parent = read(PARENT/'manifest.json')
    for name, digest in parent['study_files'].items():
        if sha256(PARENT/name) != digest: raise ValueError('Parent study input changed')
    chosen = select(read(PARENT/'cases.json'))  # Selection precedes reading reference labels.
    queries = [{k: q[k] for k in ('question_id', 'source_group', 'question', 'options', 'shifts')} |
               {'evidence': {a: q['evidence'][a] for a in ARMS}} for q in chosen]
    prompts = {}; assets = {}
    for reader in READERS:
        assets[reader] = probe_assets(ROOT/'models'/reader)
        if assets[reader] != parent['reader_assets'][reader]: raise ValueError('Reader assets changed')
        tokenizer = AutoTokenizer.from_pretrained(ROOT/'models'/reader, local_files_only=True, trust_remote_code=False)
        prompts[reader] = {q['question_id']: {f'{a}:{s}:{shift}': entry(tokenizer, q, a, s, shift)
            for a in ARMS for s in STRATEGIES for shift in q['shifts']} for q in queries}
        if any(e['tokens'] > LIMIT for p in prompts[reader].values() for e in p.values()):
            raise ValueError('Input too long: preserve selection, do not replace questions')
    ids = {q['question_id'] for q in queries}
    refs = [r for r in read(PARENT/'references.private.json') if r['question_id'] in ids]
    if len(refs) != len(ids): raise ValueError('Missing/duplicate reference identity')
    for name, value in [('queries.json', queries), ('prompts.json', prompts), ('references.private.json', refs)]:
        save(STUDY/name, value)
    paths = ['reader_memory/reading_rebuild.py', 'tools/run_reading_rebuild.py',
             'research/READING_REBUILD_PILOT_PLAN.md', 'reader_memory/backend.py',
             'reader_memory/literal_restoration.py', 'reader_memory/native_choice_pilot.py',
             'reader_memory/schema.py', 'reader_memory/diagnostics.py',
             'reader_memory/choice_protocol_audit.py', 'reader_memory/four_reader_probe.py',
             'research/literal_restoration/v1/manifest.json', 'research/literal_restoration/v1/cases.json',
             'research/literal_restoration/v1/references.private.json']
    m = {'version': 'reading-rebuild-v1', 'frozen_at': datetime.now(timezone.utc).isoformat(),
         'file_hashes': {p: sha256(ROOT/p) for p in paths},
         'study_files': {p: sha256(STUDY/p) for p in ('queries.json', 'prompts.json', 'references.private.json')},
         'reader_assets': assets, 'readers': list(READERS), 'arms': list(ARMS), 'strategies': list(STRATEGIES),
         'generation_caps': CAPS, 'execution_policy': POLICY, 'context_limit': LIMIT,
         'questions': len(queries), 'source_groups': len({q['source_group'] for q in queries}),
         'expected_records': len(queries)*2*2*3*4,
         'selection_uses_labels_or_model_outputs': False, 'independent_validation': False,
         'reserved_validation_used': False, 'selector_trained': False, 'oracle_is_deployable': False}
    save(STUDY/'manifest.json', m)
    print('READING_REBUILD_FROZEN', m['expected_records'], flush=True)


def load():
    m = read(STUDY/'manifest.json')
    if (any(sha256(ROOT/p) != h for p, h in m['file_hashes'].items()) or
        any(sha256(STUDY/p) != h for p, h in m['study_files'].items()) or
        m['execution_policy'] != POLICY or m['generation_caps'] != CAPS):
        raise ValueError('Frozen pilot provenance changed')
    return m, read(STUDY/'queries.json'), read(STUDY/'prompts.json')


def key(r): return r['reader'], r['question_id'], r['arm'], r['strategy'], r['shift']


def expected(queries):
    return {(r, q['question_id'], a, s, shift) for r in READERS for q in queries
            for a in ARMS for s in STRATEGIES for shift in q['shifts']}


def verify(rows, m, queries, prompts, tokenizers=None):
    valid = expected(queries); byid = {q['question_id']: q for q in queries}; seen = set()
    for row in rows:
        k = key(row)
        if k in seen or k not in valid: raise ValueError('Duplicate/unknown pilot row')
        r, uid, a, s, shift = k; q = byid[uid]; e = prompts[r][uid][f'{a}:{s}:{shift}']
        ids = row['generated_token_ids']; eos = row['eos_token_ids']; usage = row['usage']
        stopped = bool(ids and ids[-1] in eos)
        truncated = len(ids) >= CAPS[s] and not stopped
        if (row['study_signature'] != fingerprint(m) or row['prompt_entry'] != e or
            row['evidence_hash'] != fingerprint(q['evidence'][a]['text']) or
            row['source_ledger_hash'] != q['evidence'][a]['source_ledger_hash'] or
            e['messages_hash'] != fingerprint(messages(q, a, s, shift)) or
            usage['prompt_tokens'] != e['tokens'] or not 0 < e['tokens'] <= LIMIT or
            usage['completion_tokens'] != len(ids) or not 0 < len(ids) <= CAPS[s] or
            not all(isinstance(i, int) and i >= 0 for i in ids) or
            not math.isfinite(usage['elapsed_seconds']) or usage['elapsed_seconds'] < 0 or
            row['eos_stopped'] != stopped or row['length_truncated'] != truncated or
            row['prediction'] != parse(row['text'], s, truncated)):
            raise ValueError('Pilot output provenance/parser/cost mismatch')
        if tokenizers is not None:
            tokenizer = tokenizers[r]
            if (entry(tokenizer, q, a, s, shift) != e or
                tokenizer.decode(ids, skip_special_tokens=True) != row['text'] or
                tokenizer.decode(ids, skip_special_tokens=False) != row['raw_text']):
                raise ValueError('Token audit failed')
        seen.add(k)
    return seen


def generate(backend, msg, strategy):
    b = backend
    text = b.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = b.tokenizer(text, return_tensors='pt', add_special_tokens=False).to(b.device)
    n = batch['input_ids'].shape[-1]
    if n > LIMIT: raise ValueError('No silent context truncation')
    eos = b.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos] if eos is not None else []
    started = time.perf_counter()
    with b.lock, b.torch.inference_mode():
        result = b.model.generate(**batch, max_new_tokens=CAPS[strategy], do_sample=False,
            pad_token_id=b.tokenizer.pad_token_id if b.tokenizer.pad_token_id is not None else b.tokenizer.eos_token_id)
    ids = result[0, n:].tolist(); elapsed = time.perf_counter()-started
    text = b.tokenizer.decode(ids, skip_special_tokens=True)
    stopped = bool(ids and ids[-1] in eos); truncated = len(ids) >= CAPS[strategy] and not stopped
    return {'text': text, 'raw_text': b.tokenizer.decode(ids, skip_special_tokens=False),
            'generated_token_ids': ids, 'eos_token_ids': eos, 'eos_stopped': stopped,
            'length_truncated': truncated, 'prediction': parse(text, strategy, truncated),
            'usage': {'prompt_tokens': n, 'completion_tokens': len(ids), 'elapsed_seconds': elapsed}}


def run(reader):
    from .backend import TransformersBackend
    m, queries, prompts = load(); b = TransformersBackend(ROOT/'models'/reader)
    if probe_assets(b.path) != m['reader_assets'][reader]: raise ValueError('Wrong reader assets')
    OUTPUT.mkdir(parents=True, exist_ok=True); path = OUTPUT/'answers.jsonl'
    rows = list(read_jsonl(path)) if path.exists() else []
    done = verify(rows, m, queries, prompts)
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        for q in queries:
            for arm in ARMS:
                for strategy in STRATEGIES:
                    for shift in q['shifts']:
                        k = reader, q['question_id'], arm, strategy, shift
                        if k in done: continue
                        e = entry(b.tokenizer, q, arm, strategy, shift)
                        if e != prompts[reader][q['question_id']][f'{arm}:{strategy}:{shift}']:
                            raise ValueError('Actual input token IDs changed')
                        row = {'reader': reader, 'question_id': q['question_id'], 'arm': arm,
                               'strategy': strategy, 'shift': shift, 'study_signature': fingerprint(m),
                               'prompt_entry': e, 'evidence_hash': fingerprint(q['evidence'][arm]['text']),
                               'source_ledger_hash': q['evidence'][arm]['source_ledger_hash'],
                               **generate(b, messages(q, arm, strategy, shift), strategy)}
                        verify([row], m, queries, prompts, {reader: b.tokenizer})
                        stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush(); done.add(k)
                        if len(done) % 8 == 0:
                            save(OUTPUT/'status.json', {'status': 'running', 'reader': reader,
                                 'records': len(done), 'expected_records': m['expected_records']})
                            print('READING_REBUILD_PROGRESS', reader, len(done), '/', m['expected_records'], flush=True)


def interval(diffs):
    import numpy as np
    v = np.array(list(diffs.values()), dtype=float)
    # Exactly one query per source; bootstrap queries equals bootstrap source clusters.
    draws = np.random.default_rng(20261026).integers(0, len(v), size=(20000, len(v)))
    return {'difference_pp': 100*float(v.mean()),
            'exploratory_source_95_interval_pp': (100*np.quantile(v[draws].mean(axis=1), [0.025, 0.975])).tolist()}


def report():
    m, queries, prompts = load(); rows = list(read_jsonl(OUTPUT/'answers.jsonl'))
    from transformers import AutoTokenizer
    tokenizers = {r: AutoTokenizer.from_pretrained(ROOT/'models'/r, local_files_only=True, trust_remote_code=False) for r in READERS}
    if verify(rows, m, queries, prompts, tokenizers) != expected(queries): raise ValueError('Incomplete matrix')
    refs = {r['question_id']: r for r in read(STUDY/'references.private.json')}
    qmap = {q['question_id']: q for q in queries}
    scored = []
    for row in rows:
        semantic = None if row['prediction'] is None else (row['shift']+'abcd'.index(row['prediction'])) % 4
        scored.append({**row, 'semantic_choice': semantic, 'source_group': qmap[row['question_id']]['source_group'],
                       'correct': semantic == refs[row['question_id']]['correct_original_index']})
    index = {key(row): row for row in scored}; summaries = {}; contrasts = []; ready = []
    for reader in READERS:
        summaries[reader] = {}
        for arm in ARMS:
            for strategy in STRATEGIES:
                sub = [r for r in scored if (r['reader'], r['arm'], r['strategy']) == (reader, arm, strategy)]
                stable = sum(all(index[reader, q['question_id'], arm, strategy, s]['prediction'] is not None for s in q['shifts']) and
                    len({index[reader, q['question_id'], arm, strategy, s]['semantic_choice'] for s in q['shifts']}) == 1 for q in queries)
                summaries[reader][f'{arm}:{strategy}'] = {
                    'correct': sum(r['correct'] for r in sub), 'presentations': len(sub),
                    'accuracy': sum(r['correct'] for r in sub)/len(sub),
                    'valid': sum(r['prediction'] is not None for r in sub),
                    'truncated': sum(r['length_truncated'] for r in sub), 'stable_questions': stable,
                    'prompt_tokens': sum(r['usage']['prompt_tokens'] for r in sub),
                    'completion_tokens': sum(r['usage']['completion_tokens'] for r in sub),
                    'shared_gpu_seconds': sum(r['usage']['elapsed_seconds'] for r in sub)}
            for baseline in ('direct', 'reasoning_control'):
                d = {q['question_id']: sum(int(index[reader, q['question_id'], arm, 'con', s]['correct'])-
                     int(index[reader, q['question_id'], arm, baseline, s]['correct']) for s in q['shifts'])/4 for q in queries}
                contrasts.append({'reader': reader, 'arm': arm, 'contrast': 'con-minus-'+baseline, **interval(d)})
        for strategy in STRATEGIES:
            d = {q['question_id']: sum(int(index[reader, q['question_id'], 'oracle', strategy, s]['correct'])-
                 int(index[reader, q['question_id'], 'retrieved', strategy, s]['correct']) for s in q['shifts'])/4 for q in queries}
            contrasts.append({'reader': reader, 'strategy': strategy, 'contrast': 'oracle-minus-retrieved', **interval(d)})
        cell = summaries[reader]['oracle:con']
        if cell['accuracy'] >= .60 and cell['valid']/cell['presentations'] >= .95 and cell['stable_questions'] >= len(queries)/2:
            ready.append(reader)
    value = {'status': 'completed', 'records': len(rows), 'expected_records': m['expected_records'],
             'source_groups': m['source_groups'], 'by_reader': summaries, 'contrasts': contrasts,
             'operational_measurement_ready_readers': ready, 'operational_gate_is_confirmatory': False,
             'next_scope': 'design-complete-baselines' if ready else 'inspect-task-evidence-and-reading-before-scaling',
             'study_manifest_sha256': sha256(STUDY/'manifest.json'), 'answers_sha256': sha256(OUTPUT/'answers.jsonl'),
             'reserved_validation_used': False, 'independent_validation': False, 'selector_trained': False,
             'equal_compute_comparison': False, 'normal_retrieval_formal_benchmark': False}
    save(OUTPUT/'report.json', value)
    (OUTPUT/'scored.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in scored), encoding='utf-8')
    save(OUTPUT/'audit.json', {'status': 'passed', 'records': len(rows), 'complete_expected_matrix': True,
         'actual_token_ids_redecoded': True, 'answers_sha256': value['answers_sha256'],
         'study_manifest_sha256': value['study_manifest_sha256']})
    print('READING_REBUILD_REPORT', value['next_scope'], flush=True)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run', 'report')); p.add_argument('--reader', choices=READERS)
    args = p.parse_args()
    from .__main__ import offline_environment
    offline_environment()
    if args.command == 'prepare': return prepare()
    if args.command == 'report': return report()
    if args.reader is None: raise ValueError('Reader required')
    return run(args.reader)


if __name__ == '__main__': main()
