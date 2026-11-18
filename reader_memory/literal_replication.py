"""Prospectively fixed format replication on fresh development source groups."""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .choice_protocol_audit import save
from .diagnostics import sha256
from .four_reader_probe import READERS, SharedCounter, pack, probe_assets
from .independent_pack import normalize
from .literal_evidence import canonical, characters, decode, render
from .literal_restoration import LIMIT, POLICY, messages, prompt_entry, whole
from .native_choice_pilot import classify, read, score_row
from .personamem2 import REVISION, history_path, list_field, source_rows
from .schema import Turn, fingerprint, read_jsonl
from .shared_retrieval_audit import TASK, document, query_content, rank_scores
from .token_windows import token_windows

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'research/literal_replication/v1'
OUTPUT = ROOT / 'runs/literal_replication_v1'
PLAN = ROOT / 'research/LITERAL_REPLICATION_PLAN.md'
ALLOCATION = STUDY / 'allocation.json'
SPLIT = ROOT / 'research/personamem_v2/v1/split.json'
PRIOR_ALLOCATION = ROOT / 'runs/research_viability_v1/source_allocation.json'
METADATA = ROOT / 'research/sources/expanded_data_20261026/ImplicitPersona.metadata.json'
ARMS = ('restored', 'structured')
STAGE_ORDER = ('Qwen3.5-4B', 'Qwen3.5-2B', 'Phi-4-mini-instruct', 'Qwen3.5-9B')
SEED = 20261027


def select(rows, split, prior):
    excluded = set(split['probe_groups']) | {g['source_group'] for g in prior['source_allocations']}
    forbidden_paths = {q['history_path'] for g in prior['source_allocations'] for q in g['queries']}
    forbidden_pids = {p for g in excluded for p in split['persona_ids_by_group'][g]}
    groups = defaultdict(dict); duplicates = defaultdict(set)
    for index, row in enumerate(rows):
        try:
            pid = str(int(row['persona_id'])); group = fingerprint(normalize(row['short_persona']))
            answers = [row['correct_answer'], *list_field(row['incorrect_answers'])]
            name = history_path(row['chat_history_32k_link'])
            if len(answers) != 4 or len({normalize(x) for x in answers}) != 4 or not all(x.strip() for x in answers):
                continue
            if not row['user_query'].strip() or not row['short_persona'].strip(): continue
            uid = 'pm2-' + fingerprint([REVISION, pid, normalize(row['user_query'])])[:24]
            duplicates[uid].add(fingerprint([name, answers]))
            if group not in split['train_groups'] or group in excluded or pid in forbidden_pids or name in forbidden_paths:
                continue
            groups[group][uid] = {'question_id': uid, 'row_index': index, 'persona_id': pid,
                'source_group': group, 'history_path': name, 'question': row['user_query'],
                'actual_query_fingerprint': fingerprint(normalize(query_content(row['user_query']))),
                'options': sorted(answers, key=lambda x: fingerprint(['pm2-options-v1', uid, x])), 'shifts': [0, 1, 2, 3]}
        except (KeyError, ValueError, TypeError, SyntaxError):
            continue
    candidates = {}
    for group, values in groups.items():
        unique = {}; values = sorted(values.values(), key=lambda q: fingerprint(['literal-replication-queries-v1', q['question_id']]))
        for q in values:
            if len(duplicates[q['question_id']]) == 1: unique.setdefault(q['actual_query_fingerprint'], q)
        identities = defaultdict(list)
        for q in unique.values(): identities[(q['persona_id'],q['history_path'])].append(q)
        eligible_ids = [identity for identity, qs in identities.items() if len(qs) >= 2]
        if eligible_ids:
            identity = min(eligible_ids, key=lambda x: fingerprint(['literal-replication-history-v1',group,x]))
            candidates[group] = identities[identity][:2]
    chosen = sorted(candidates, key=lambda g: fingerprint(['literal-replication-groups-v1', g]))[:64]
    if len(chosen) != 64: raise ValueError('Insufficient fresh development groups')
    return [q for g in chosen for q in candidates[g]]


def allocate():
    if ALLOCATION.exists(): raise FileExistsError('Preserve fixed replication allocation')
    _, rows = source_rows(); split = read(SPLIT); prior = read(PRIOR_ALLOCATION)
    if prior['existing_split_sha256'] != sha256(SPLIT): raise ValueError('Prior split changed')
    queries = select(rows, split, prior)
    paths = [PLAN, SPLIT, PRIOR_ALLOCATION, ROOT/'reader_memory/literal_replication.py', METADATA,
             ROOT/'data/personamem_v2/source_manifest.json', ROOT/'research/literal_restoration/v1/manifest.json']
    value = {'version': 'literal-replication-allocation-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'file_hashes': {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}, 'revision': REVISION,
        'queries': queries, 'intended_groups': 64, 'intended_questions': 128,
        'source_selection_uses_model_results': False, 'source_selection_uses_snippet_alignment': False,
        'confirmation_or_reserved_sources_selected': False}
    save(ALLOCATION, value); print('REPLICATION_ALLOCATED', 64, 128, flush=True)


def allocation():
    a = read(ALLOCATION)
    if any(sha256(ROOT/p) != h for p, h in a['file_hashes'].items()):
        raise ValueError('Replication allocation dependencies changed')
    _, rows = source_rows()
    if select(rows, read(SPLIT), read(PRIOR_ALLOCATION)) != a['queries']:
        raise ValueError('Allocation does not reproduce')
    return a


def sources():
    a = allocation(); source = read(STUDY/'source_manifest.json')
    names = {q['history_path'] for q in a['queries']}
    if source['allocation_sha256'] != sha256(ALLOCATION) or set(source['files']) != names or source['revision'] != REVISION:
        raise ValueError('Source download identity changed')
    result = {}; audit = []
    for q in a['queries']:
        group = q['source_group']
        if group in result: continue
        path = STUDY/'sources'/q['history_path']; entry = source['files'][q['history_path']]
        if sha256(path) != entry['sha256'] or path.stat().st_size != entry['size']:
            raise ValueError('New source bytes changed')
        value = read(path)
        if str(value['metadata']['persona_id']) != q['persona_id']: raise ValueError('Wrong new persona')
        msgs = value['chat_history']
        if any(m['role'] not in {'user', 'assistant', 'system'} or not isinstance(m['content'], str) for m in msgs):
            raise ValueError('Unknown source schema')
        turns = [Turn(f"p{q['persona_id']}:m{i:04d}:t0000", m['role'], m['content'], '')
                 for i, m in enumerate(msgs) if m['role'] != 'system']
        result[group] = turns
        audit.append({'source_group': group, 'history_path': q['history_path'], 'history_sha256': sha256(path),
            'history_body_fingerprint': fingerprint([(t.role, t.content) for t in turns]),
            'retained_turns': len(turns), 'removed_system_messages': sum(m['role'] == 'system' for m in msgs)})
    return result, audit


def overlap(histories, audits):
    # The established audit reads only the 32 exposed v2 histories and exposed
    # v1 prefixes. Its path allowlist keeps reserved v1 prefixes inaccessible.
    from .representation_development import load_sources, overlap_audit
    from .four_reader_probe import load as load_exposed
    from .personamem2 import load_history
    exposed = {}; old_bodies = set()
    for q in load_exposed()[1]:
        if q['history_path'] not in exposed: exposed[q['history_path']] = load_history(q)[0]
    old_bodies = {fingerprint([(t.role, t.content) for t in ts]) for ts in exposed.values()}
    duplicate_bodies = {g for g, ts in histories.items() if fingerprint([(t.role,t.content) for t in ts]) in old_bodies}
    safe = {g: (ts, '') for g, ts in histories.items() if g not in duplicate_bodies}
    base = overlap_audit(safe, [a for a in audits if a['source_group'] in safe])
    base['rows'].extend({'source_group': g, 'exact_long_user_message_overlap_with_exposed_v2': 0,
                        'exact_long_user_message_overlap_with_exposed_v1': 0,
                        'exposed_v2_body_duplicate': True} for g in sorted(duplicate_bodies))
    old, _ = load_sources()
    prior_bodies = {fingerprint([(t.role, t.content) for t in ts]) for ts, _ in old.values()}
    prior_long = {fingerprint(normalize(t.content)) for ts, _ in old.values() for t in ts
                  if t.role == 'user' and len(normalize(t.content)) >= 200}
    bodies = defaultdict(set); longs = defaultdict(set)
    for g, ts in histories.items():
        bodies[fingerprint([(t.role, t.content) for t in ts])].add(g)
        for t in ts:
            if t.role == 'user' and len(normalize(t.content)) >= 200: longs[fingerprint(normalize(t.content))].add(g)
    excluded = {}; checks = []
    for row in base['rows']:
        g = row['source_group']; ts = histories[g]
        body = fingerprint([(t.role, t.content) for t in ts])
        ls = {fingerprint(normalize(t.content)) for t in ts if t.role == 'user' and len(normalize(t.content)) >= 200}
        duplicate_new = len(bodies[body]) > 1 or any(len(longs[h]) > 1 for h in ls)
        duplicate_prior = body in prior_bodies or bool(ls & prior_long)
        old_long = row['exact_long_user_message_overlap_with_exposed_v2'] + row['exact_long_user_message_overlap_with_exposed_v1']
        if duplicate_new or duplicate_prior or old_long or g in duplicate_bodies:
            excluded[g] = 'exact-source-or-long-user-message-overlap'
        checks.append({**row, 'representation_dev_long_user_overlap': len(ls & prior_long),
            'representation_dev_body_duplicate': body in prior_bodies, 'same_batch_source_overlap': duplicate_new,
            'excluded': g in excluded})
    return {'rows': checks, 'excluded_groups': excluded, 'reserved_bodies_read': False,
            'exact_text_only_not_subject_identity_guarantee': True}


def retrieval(counters, encoder):
    if (STUDY/'retrieval_manifest.json').exists(): raise FileExistsError('Preserve frozen retrieval')
    started = time.perf_counter(); a = allocation(); histories, audits = sources(); checks = overlap(histories, audits)
    common = SharedCounter(counters); queries = a['queries']
    texts = [query_content(q['question']) for q in queries]
    vectors, cache = encoder.encode([f'Instruct: {TASK}\nQuery: {t}' for t in texts], OUTPUT/'embedding_cache')
    caches = [cache]; groups_by_source = {}; records = []
    for i, q in enumerate(queries):
        g = q['source_group']; ts = histories[g]
        if g not in groups_by_source:
            windows = token_windows(ts, common, limit=384)
            vs, cache = encoder.encode([document(w) for w in windows], OUTPUT/'embedding_cache')
            groups_by_source[g] = windows, vs; caches.append(cache)
        windows, vs = groups_by_source[g]; scores = (vectors[i] @ vs.T).tolist(); order = rank_scores(scores)
        packed, chosen = pack([(windows[j], scores[j]) for j in order[:24]], common, 2048)
        records.append({'question_id': q['question_id'], 'source_group': g,
            'query_content_hash': fingerprint(texts[i]), 'windows': [w.to_dict() for w in windows],
            'scores': scores, 'order': order, 'selected_chunk_ids': packed['raw']['chunk_ids'],
            'packed_raw': packed['raw'], 'spans': [s for w in chosen for s in w.metadata['spans']]})
        print('REPLICATION_RETRIEVAL', i + 1, '/', len(queries), flush=True)
    for name, value in [('retrieval.json', records), ('source_audit.json', audits), ('overlap.json', checks)]:
        save(STUDY/name, value)
    m = {'allocation_sha256': sha256(ALLOCATION), 'embedding_assets': encoder.assets, 'embedding_caches': caches,
         'files': {n: sha256(STUDY/n) for n in ('retrieval.json', 'source_audit.json', 'overlap.json')},
         'preparation_wall_seconds_including_embedding': time.perf_counter()-started,
         'oracle_annotation_used_for_retrieval': False, 'options_used_for_retrieval': False}
    save(STUDY/'retrieval_manifest.json', m)


def build(counters):
    a = allocation(); histories, _ = sources(); _, rows = source_rows()
    retrieval = {r['question_id']: r for r in read(STUDY/'retrieval.json')}
    excluded = read(STUDY/'overlap.json')['excluded_groups']; cases = []; refs = []; prompts = {r: {} for r in READERS}
    for source in a['queries']:
        q = dict(source); q.update(eligible=False, exclusion_reason=None, evidence=None)
        native = rows[q['row_index']]; ts = histories[q['source_group']]
        if native['user_query'] != q['question']: raise ValueError('Question source identity changed')
        refs.append({'question_id': q['question_id'], 'correct_original_index': q['options'].index(native['correct_answer'])})
        reason = excluded.get(q['source_group'])
        clean = normalize(query_content(q['question']))
        if any(normalize(t.content) in {clean, normalize(q['question'])} for t in ts if t.role == 'user'):
            reason = reason or 'current-query-present-in-history'
        try: snippet = json.loads(native['related_conversation_snippet'])
        except json.JSONDecodeError: snippet = ast.literal_eval(native['related_conversation_snippet'])
        if not isinstance(snippet, list) or not snippet or any(s['role'] not in {'user','assistant'} or not isinstance(s['content'], str) for s in snippet):
            raise ValueError('Official snippet schema changed')
        matches = [[t for t in ts if t.role == s['role'] and normalize(t.content) == normalize(s['content'])] for s in snippet]
        q['alignment'] = {'annotation_hash': fingerprint(snippet), 'matched_source_ids': [[t.source_id for t in v] for v in matches],
                          'literal_annotation_matches': [len(v) == 1 and v[0].content == s['content'] for v, s in zip(matches, snippet)]}
        if not all(len(v) == 1 for v in matches): reason = reason or 'official-snippet-not-uniquely-aligned'
        if reason:
            q['exclusion_reason'] = reason; cases.append(q); continue
        root = canonical(ts, retrieval[q['question_id']]['spans'])
        restored = canonical(ts, root + [whole(v[0]) for v in matches])
        q['evidence'] = {}
        for arm in ARMS:
            view = 'raw' if arm == 'restored' else 'structured'; text = render(restored, view)
            if decode(text, view) != restored: raise ValueError('Views are not reversible')
            q['evidence'][arm] = {'source_ledger': restored, 'source_ledger_hash': fingerprint(restored), 'text': text,
                'tokens_by_reader': {r: c.count(text) for r, c in counters.items()}}
        q['restoration'] = {'retrieved_source_characters': characters(root), 'restored_source_characters': characters(restored)}
        entries = {r: {f'{arm}:{s}': prompt_entry(c, q, arm, s) for arm in ARMS for s in q['shifts']} for r, c in counters.items()}
        if max(e['tokens'] for es in entries.values() for e in es.values()) > LIMIT:
            q['exclusion_reason'] = 'complete-prompt-exceeds-common-context-limit'
        else:
            q['eligible'] = True
            for r in READERS: prompts[r][q['question_id']] = entries[r]
        cases.append(q)
    return cases, prompts, refs


def prepare(counters):
    if (STUDY/'manifest.json').exists(): raise FileExistsError('Preserve replication input freeze')
    assets = {r: probe_assets(ROOT/'models'/r) for r in READERS}
    if assets != read(ROOT/'research/literal_restoration/v1/manifest.json')['reader_assets']:
        raise ValueError('Parent reader assets changed')
    cases, prompts, refs = build(counters); eligible = [q for q in cases if q['eligible']]
    if len(eligible) < 96 or len({q['source_group'] for q in eligible}) < 48:
        save(OUTPUT/'data_readiness_failure.json', {'eligible_questions': len(eligible), 'cases': cases})
        raise ValueError('Insufficient eligible sources/questions; do not run readers or replace samples')
    files = ['reader_memory/literal_replication.py', 'tools/prepare_literal_replication_sources.py',
        'tools/audit_literal_replication.py', 'tools/run_literal_replication.py', 'research/LITERAL_REPLICATION_PLAN.md',
        'reader_memory/four_reader_probe.py', 'reader_memory/shared_retrieval_audit.py', 'reader_memory/token_windows.py',
        'reader_memory/backend.py', 'reader_memory/schema.py', 'reader_memory/tokenization.py', 'reader_memory/native_choice_pilot.py',
        'reader_memory/literal_evidence.py', 'reader_memory/literal_restoration.py', 'reader_memory/representation_development.py',
        'reader_memory/personamem2.py', 'reader_memory/independent_pack.py', 'reader_memory/selection.py',
        'reader_memory/diagnostics.py', 'reader_memory/choice_protocol_audit.py', 'reader_memory/retrieval.py',
        'reader_memory/source_records.py', 'tools/audit_literal_restoration.py', 'tools/prepare_personamem2.py',
        'research/literal_restoration/v1/manifest.json']
    for name, value in [('cases.json', cases), ('prompts.json', prompts), ('references.private.json', refs)]: save(STUDY/name, value)
    inputs = ['allocation.json', 'source_manifest.json', 'retrieval_manifest.json', 'retrieval.json',
              'source_audit.json', 'overlap.json', 'cases.json', 'prompts.json', 'references.private.json']
    m = {'version': 'literal-replication-v1', 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'file_hashes': {p: sha256(ROOT/p) for p in files}, 'study_files': {p: sha256(STUDY/p) for p in inputs},
        'reader_assets': assets, 'execution_policy': POLICY,
        'intended_questions': 128, 'eligible_questions': len(eligible), 'source_groups': len({q['source_group'] for q in eligible}),
        'expected_records': len(eligible)*len(READERS)*len(ARMS)*4,
        'primary_metrics': ['Qwen3.5-4B-format-gain', 'Qwen3.5-4B-minus-other-reader-mean-gain'],
        'primary_family_interval_quantiles': [0.0125, 0.9875], 'bootstrap_seed': SEED,
        'oracle_diagnostic': True, 'normal_retrieval_benchmark': False, 'equal_budget_comparison': False,
        'reserved_validation_used': False, 'confirmation_sources_read': False, 'selector_trained': False,
        'reader_capability_hypothesis_tested': False}
    save(STUDY/'manifest.json', m); print('REPLICATION_FROZEN', len(eligible), m['expected_records'], flush=True)


def load():
    m = read(STUDY/'manifest.json')
    if any(sha256(ROOT/p) != h for p, h in m['file_hashes'].items()) or any(sha256(STUDY/p) != h for p, h in m['study_files'].items()):
        raise ValueError('Frozen replication code/inputs changed')
    if m['execution_policy'] != POLICY or m['bootstrap_seed'] != SEED: raise ValueError('Policy changed')
    from .literal_restoration import load as parent_load
    parent_load(); sources()
    return m, read(STUDY/'cases.json'), read(STUDY/'prompts.json')


def key(row):
    return row['reader'], row['question_id'], row['arm'], row['shift']


def expected(cases):
    return {(r, q['question_id'], a, s) for q in cases if q['eligible'] for r in READERS for a in ARMS for s in q['shifts']}


def verify(rows, m, cases, prompts):
    valid = expected(cases); byid = {q['question_id']: q for q in cases}; seen = {}
    for row in rows:
        k = key(row)
        if k not in valid or k in seen: raise ValueError('Duplicate/unknown replication answer')
        r, uid, arm, shift = k; item = byid[uid]['evidence'][arm]; entry = prompts[r][uid][f'{arm}:{shift}']
        scores = row['restricted_label_logits']; usage = row['usage']
        if (row['study_signature'] != fingerprint(m) or row['prompt_entry'] != entry or row['evidence_hash'] != fingerprint(item['text'])
            or row['source_ledger_hash'] != item['source_ledger_hash'] or row['evidence_tokens'] != item['tokens_by_reader'][r]
            or usage['prompt_tokens'] != entry['tokens'] or not 0 < usage['prompt_tokens'] <= LIMIT
            or not math.isfinite(usage['elapsed_seconds']) or usage['elapsed_seconds'] < 0
            or len(scores) != 4 or not all(math.isfinite(v) for v in scores)
            or row['prediction'] != 'abcd'[max(range(4), key=lambda i: scores[i])]):
            raise ValueError('Replication answer provenance/cost changed')
        if row['prediction_origin'] == 'cache-identical-prompt':
            previous = seen.get(tuple(row['cached_from_key']))
            if previous is None or previous['reader'] != r or previous['prompt_entry'] != entry or previous['restricted_label_logits'] != scores or usage['elapsed_seconds'] != 0:
                raise ValueError('Invalid exact-prompt cache')
        elif row['prediction_origin'] != 'model-forward' or row['cached_from_key'] is not None:
            raise ValueError('Unknown answer origin')
        seen[k] = row
    return set(seen)


def run(backend, counter):
    m, cases, prompts = load(); reader = backend.model_name
    if reader not in READERS or not counter.exact or probe_assets(backend.path) != m['reader_assets'][reader]: raise ValueError('Wrong reader')
    path = OUTPUT/'answers.jsonl'; rows = list(read_jsonl(path)) if path.exists() else []
    done = verify(rows, m, cases, prompts)
    cache = {(r['reader'], r['prompt_entry']['messages_hash'], r['prompt_entry']['token_ids_hash']): r for r in rows}
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        for q in cases:
            if not q['eligible']: continue
            for arm in ARMS:
                item = q['evidence'][arm]
                if counter.count(item['text']) != item['tokens_by_reader'][reader]: raise ValueError('Evidence cost changed')
                for shift in q['shifts']:
                    k = reader, q['question_id'], arm, shift
                    if k in done: continue
                    entry = prompt_entry(counter, q, arm, shift)
                    if entry != prompts[reader][q['question_id']][f'{arm}:{shift}']: raise ValueError('Prompt IDs changed')
                    ck = reader, entry['messages_hash'], entry['token_ids_hash']; prev = cache.get(ck)
                    result = ({'prediction': prev['prediction'], 'restricted_label_logits': prev['restricted_label_logits'],
                               'usage': {**prev['usage'], 'elapsed_seconds': 0.0}} if prev else classify(backend, messages(q, arm, shift)))
                    row = {'reader': reader, 'question_id': q['question_id'], 'arm': arm, 'shift': shift,
                        'study_signature': fingerprint(m), 'prompt_entry': entry, 'evidence_hash': fingerprint(item['text']),
                        'source_ledger_hash': item['source_ledger_hash'], 'evidence_tokens': item['tokens_by_reader'][reader],
                        'prediction_origin': 'cache-identical-prompt' if prev else 'model-forward',
                        'cached_from_key': list(key(prev)) if prev else None, **result}
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush(); done.add(k); cache[ck] = row
                    if len(done) % 64 == 0:
                        save(OUTPUT/'status.json', {'reader': reader, 'records': len(done), 'expected_records': m['expected_records']})
                        print('REPLICATION_PROGRESS', reader, len(done), '/', m['expected_records'], flush=True)


def cluster_interval(diffs, groups, *, family=False):
    import numpy as np
    names = sorted(set(groups.values())); ids = list(diffs)
    nums = np.array([sum(diffs[u] for u in ids if groups[u] == g) for g in names])
    counts = np.array([sum(groups[u] == g for u in ids) for g in names])
    draws = np.random.default_rng(SEED).integers(0, len(names), size=(20000, len(names)))
    samples = nums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    value = {'difference_pp': 100*sum(diffs.values())/len(diffs),
             'cluster_95_interval_pp': (100*np.quantile(samples, [0.025, 0.975])).tolist(),
             'source_equal_weight_difference_pp': 100*float((nums/counts).mean())}
    if family: value['primary_family_97_5_interval_pp'] = (100*np.quantile(samples, [0.0125, 0.9875])).tolist()
    return value


def decision(primary):
    gain, interaction = primary
    if gain['difference_pp'] >= 2 and all(p['primary_family_97_5_interval_pp'][0] > 0 for p in primary):
        return 'replicated-format-signal-design-independent-capability-study'
    if gain['primary_family_97_5_interval_pp'][0] > 0:
        return 'format-benefit-reader-specific-practical-gate-not-met'
    if gain['primary_family_97_5_interval_pp'][1] < 2:
        return 'practical-4b-format-gain-not-supported-stop-current-format-line'
    return 'insufficient-evidence-no-format-sweep-or-selector-training'


def report():
    m, cases, prompts = load(); rows = list(read_jsonl(OUTPUT/'answers.jsonl'))
    if verify(rows, m, cases, prompts) != expected(cases): raise ValueError('Complete answer matrix required')
    qs = [q for q in cases if q['eligible']]; qmap = {q['question_id']: q for q in qs}
    refs = {r['question_id']: r for r in read(STUDY/'references.private.json')}
    scored = [{**r, 'source_group': qmap[r['question_id']]['source_group'],
        'correct': score_row(r, qmap[r['question_id']], refs[r['question_id']]),
        'semantic_choice': (r['shift']+'abcd'.index(r['prediction'])) % 4} for r in rows]
    index = {key(r): r for r in scored}; groups = {q['question_id']: q['source_group'] for q in qs}
    deltas = {}; summaries = {}; stability = {}
    for reader in READERS:
        summaries[reader] = {}; stability[reader] = {}
        for arm in ARMS:
            sub = [r for r in scored if r['reader'] == reader and r['arm'] == arm]
            summaries[reader][arm] = {'correct': sum(r['correct'] for r in sub), 'presentations': len(sub),
                'accuracy': sum(r['correct'] for r in sub)/len(sub),
                'mean_evidence_tokens': sum(r['evidence_tokens'] for r in sub)/len(sub),
                'mean_prompt_tokens': sum(r['usage']['prompt_tokens'] for r in sub)/len(sub),
                'semantic_choice_varies_across_orders': sum(len({index[(reader,q['question_id'],arm,s)]['semantic_choice'] for s in q['shifts']}) > 1 for q in qs)}
        d = {q['question_id']: sum(index[(reader,q['question_id'],'structured',s)]['correct']-index[(reader,q['question_id'],'restored',s)]['correct'] for s in q['shifts'])/4 for q in qs}
        deltas[reader] = d; summaries[reader]['format_contrast'] = cluster_interval(d, groups)
        totals = {g: sum(v for u,v in d.items() if groups[u] == g) for g in set(groups.values())}
        loo = [100*sum(v for u,v in d.items() if groups[u] != g)/sum(groups[u] != g for u in d) for g in totals]
        stability[reader] = {'question_gain_loss_tie': {k: sum((v > 0 if k == 'gain' else v < 0 if k == 'loss' else v == 0) for v in d.values()) for k in ('gain','loss','tie')},
            'source_gain_loss_tie': {k: sum((v > 0 if k == 'gain' else v < 0 if k == 'loss' else v == 0) for v in totals.values()) for k in ('gain','loss','tie')},
            'position_effect_pp': {str(s): 100*sum(index[(reader,q['question_id'],'structured',s)]['correct']-index[(reader,q['question_id'],'restored',s)]['correct'] for q in qs)/len(qs) for s in range(4)},
            'leave_one_source_out_point_effect_range_pp': [min(loo),max(loo)], 'by_question': d,
            'by_source_total_question_gain': totals, 'leave_one_source_out_is_new_validation': False}
    gain = deltas['Qwen3.5-4B']
    interaction = {u: gain[u]-sum(deltas[r][u] for r in READERS if r != 'Qwen3.5-4B')/3 for u in gain}
    primary = [{'metric': name, **cluster_interval(d, groups, family=True)} for name,d in zip(m['primary_metrics'], (gain, interaction))]
    pairwise = [{'reader': r, **cluster_interval({u: gain[u]-deltas[r][u] for u in gain}, groups)} for r in READERS if r != 'Qwen3.5-4B']
    value = {'status': 'completed-new-development-source-format-replication', 'records': len(rows),
        'eligible_questions': len(qs), 'source_groups': len(set(groups.values())), 'intended_questions': 128,
        'excluded_questions': [{'question_id': q['question_id'], 'source_group': q['source_group'], 'reason': q['exclusion_reason']} for q in cases if not q['eligible']],
        'by_reader': summaries, 'primary_metrics': primary, 'secondary_4b_pairwise_interactions': pairwise,
        'primary_gate_decision': decision(primary), 'position_and_source_stability': stability,
        'actual_model_forward_calls': sum(r['prediction_origin'] == 'model-forward' for r in rows),
        'identical_prompt_cache_reuses': sum(r['prediction_origin'] != 'model-forward' for r in rows),
        'forward_seconds_shared_gpu': sum(r['usage']['elapsed_seconds'] for r in rows),
        'actual_forward_prompt_tokens': sum(r['usage']['prompt_tokens'] for r in rows if r['prediction_origin'] == 'model-forward'),
        'study_manifest_sha256': sha256(STUDY/'manifest.json'), 'answers_sha256': sha256(OUTPUT/'answers.jsonl'),
        'oracle_diagnostic': True, 'equal_budget_comparison': False, 'normal_retrieval_benchmark': False,
        'reader_capability_hypothesis_tested': False, 'reserved_validation_used': False, 'confirmation_sources_read': False,
        'selector_trained': False, 'timing_isolated_from_other_gpu_workloads': False}
    (OUTPUT/'scored.jsonl').write_bytes(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in scored).encode('utf-8'))
    save(OUTPUT/'report.json', value); print('REPLICATION_REPORT', value['primary_gate_decision'], flush=True)
    return value


def render_report():
    m, _, _ = load(); r = read(OUTPUT/'report.json'); audit = read(OUTPUT/'input_integrity.json')
    if audit['status'] != 'passed' or audit['answers_sha256'] != r['answers_sha256'] or r['answers_sha256'] != sha256(OUTPUT/'answers.jsonl'):
        raise ValueError('Audited complete outputs required')
    lines = ['# 同信息格式收益的新开发来源复核', '',
        f"意向 64 来源/128 题；合格 {r['source_groups']} 来源/{r['eligible_questions']} 题，{r['records']} 次呈现。新来源来自同一公开训练集，不是封存确认或跨数据集验证。", '',
        '使用官方来源注释进行补回，只比较同一账本的原文和 JSON；不是正常检索或等预算主成绩。', '',
        '| 读者 | 原文正确率 | JSON 正确率 | 差值，百分点 | 描述性 95% 来源区间 |',
        '| --- | ---: | ---: | ---: | --- |']
    for reader in READERS:
        b = r['by_reader'][reader]; c = b['format_contrast']; lo, hi = c['cluster_95_interval_pp']
        lines.append(f"| {reader} | {100*b['restored']['accuracy']:.2f}% | {100*b['structured']['accuracy']:.2f}% | {c['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', '| 预先指定共同主要指标 | 差值，百分点 | Bonferroni 校正各 97.5% 来源区间 |', '| --- | ---: | --- |']
    for p in r['primary_metrics']:
        lo, hi = p['primary_family_97_5_interval_pp']; lines.append(f"| {p['metric']} | {p['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', '冻结判定：`'+r['primary_gate_decision']+'`。只有 4B 点估计至少 +2 pp 且两个共同主要指标的校正区间下界均大于零才通过。', '',
        '所有区间按完整来源组配对 bootstrap，20,000 次。选项位置不是独立样本。读者差异不能自动归因为独立能力画像；未训练选择器或预算分配。', '',
        '全部排除、实际 token 成本、逐来源/逐题/各位置差值、位置敏感性和去一来源范围见 `../runs/literal_replication_v1/report.json`。资源记录与准备阶段耗时在同目录；共享 GPU 耗时不作独占速度比较。', '',
        '48 个原确认来源、159 保留组、旧四个验证 persona 与 LME 440 题继续封存。协议见 [LITERAL_REPLICATION_PLAN.md](LITERAL_REPLICATION_PLAN.md)。', '']
    (ROOT/'research/LITERAL_REPLICATION_REPORT.md').write_bytes('\n'.join(lines).encode('utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=('allocate','retrieval','prepare','run','report','render'))
    p.add_argument('--reader', choices=READERS); args = p.parse_args()
    if args.command == 'allocate': return allocate()
    if args.command == 'report': return report()
    if args.command == 'render': return render_report()
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    if args.command in {'retrieval','prepare'}:
        counters = {r: TokenCounter(ROOT/'models'/r) for r in READERS}
        if args.command == 'prepare': return prepare(counters)
        from .shared_retrieval_audit import DenseEncoder
        return retrieval(counters, DenseEncoder())
    if args.reader is None: raise ValueError('Reader required')
    from .backend import TransformersBackend
    return run(TransformersBackend(ROOT/'models'/args.reader), TokenCounter(ROOT/'models'/args.reader))


if __name__ == '__main__': main()
