"""Audit the pinned public text train split and reserve complete persona groups."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath

from .diagnostics import sha256
from .independent_pack import normalize
from .schema import Turn, fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/personamem_v2'
STUDY = ROOT/'research/personamem_v2/v1'
REVISION = 'ed956dea41521fc4499acbc63f966e0fd3c053ba'


def list_field(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        result = ast.literal_eval(value)
    if not isinstance(result, list) or any(not isinstance(x, str) or not x.strip() for x in result):
        raise ValueError('Expected a list of answer strings')
    return result


def history_path(value):
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or path.parts[:2] != ('data', 'chat_history_32k') or path.suffix != '.json':
        raise ValueError('Expected the explicitly referenced text-only 32k history')
    return path.as_posix()


def source_rows():
    source = json.loads((DATA/'source_manifest.json').read_text(encoding='utf-8'))
    name = 'benchmark/text/train.csv'
    if source['revision'] != REVISION or source['files'][name]['sha256'] != sha256(DATA/name):
        raise ValueError('Unverified public train source')
    csv.field_size_limit(16 << 20)
    with (DATA/name).open(encoding='utf-8', newline='') as stream:
        return source, list(csv.DictReader(stream))


def plan(study=STUDY, probe_personas=32):
    study = Path(study)
    if study.exists():
        raise FileExistsError('Preserve frozen source split')
    source, rows = source_rows()
    groups = defaultdict(list)
    valid, invalid = [], Counter()
    duplicate_keys = defaultdict(list)
    for index, row in enumerate(rows):
        try:
            pid = str(int(row['persona_id']))
            answers = [row['correct_answer'], *list_field(row['incorrect_answers'])]
            if len(answers) != 4 or len({normalize(x) for x in answers}) != 4 or not all(x.strip() for x in answers):
                raise ValueError('Four distinct nonempty answers required')
            path = history_path(row['chat_history_32k_link'])
            if not row['user_query'].strip() or not row['short_persona'].strip():
                raise ValueError('Missing query/persona description')
            # Duplicate descriptions share one split group even if IDs differ.
            group = fingerprint(normalize(row['short_persona']))
            record = {'row_index': index, 'persona_id': pid, 'source_group': group,
                'history_path': path, 'question': row['user_query'], 'answers': answers,
                'question_id': 'pm2-'+fingerprint([REVISION,pid,normalize(row['user_query'])])[:24]}
            groups[group].append(record)
            duplicate_keys[record['question_id']].append(record)
            valid.append(record)
        except (ValueError, SyntaxError, KeyError, TypeError) as exc:
            invalid[str(exc)] += 1
    ambiguous = {qid for qid, records in duplicate_keys.items()
                 if len({fingerprint([r['history_path'],r['answers']]) for r in records}) > 1}
    owner_groups = defaultdict(set)
    for r in valid:
        owner_groups[r['persona_id']].add(r['source_group'])
    if any(len(g)>1 for g in owner_groups.values()):
        raise ValueError('A persona ID has multiple descriptions; union those groups before splitting')
    ordered = sorted(groups, key=lambda g: fingerprint(['pm2-persona-split-v1',g]))
    cut = int(len(ordered)*.8)
    train, reserved = ordered[:cut], ordered[cut:]
    available = {}
    for group in train:
        unique = {r['question_id']: r for r in groups[group] if r['question_id'] not in ambiguous}
        if len(unique) >= 2:
            available[group] = sorted(unique.values(), key=lambda r: fingerprint(['pm2-probe-v1',r['question_id']]))
    chosen_groups = sorted(available, key=lambda g: fingerprint(['pm2-probe-groups-v1',g]))[:probe_personas]
    if len(chosen_groups) != probe_personas:
        raise ValueError('Insufficient independent training persona groups')
    queries, references = [], []
    for group in chosen_groups:
        for r in available[group][:2]:
            answers = sorted(r['answers'], key=lambda text: fingerprint(['pm2-options-v1',r['question_id'],text]))
            queries.append({k:r[k] for k in ('question_id','row_index','persona_id','history_path','question','source_group')}
                           | {'options': answers, 'shifts':[0,1,2,3]})
            references.append({'question_id':r['question_id'],'correct_original_index':answers.index(r['answers'][0])})
    seen_queries = set()
    for path in ('research/native_choice/v1/queries.json','research/window_utility/v1/queries.json'):
        seen_queries.update(normalize(q['question']) for q in json.loads((ROOT/path).read_text(encoding='utf-8')))
    audit = {'version':'personamem2-train-input-audit-v1', 'created_at':datetime.now(timezone.utc).isoformat(),
        'repository':source['repository'],'revision':REVISION,'license':source['license'],
        'train_csv_sha256':sha256(DATA/'benchmark/text/train.csv'),
        'rows':len(rows),'valid_rows':len(valid),'invalid_rows_by_reason':dict(invalid),
        'distinct_persona_ids':len({r['persona_id'] for r in valid}),
        'distinct_normalized_persona_descriptions':len(groups),
        'distinct_history_paths':len({r['history_path'] for r in valid}),
        'duplicate_question_rows':len(valid)-len(duplicate_keys),'ambiguous_query_identity_count':len(ambiguous),
        'scenarios':dict(Counter(r['conversation_scenario'] for r in rows)),
        'preference_updates':dict(Counter(r['updated'] for r in rows)),
        'preference_owners':dict(Counter(r['who'] for r in rows)),
        'training_groups':len(train),'reserved_internal_validation_groups':len(reserved),
        'probe_groups':len(chosen_groups),'probe_questions':len(queries),
        'probe_queries_matching_exposed_pm1_queries':sum(normalize(q['question']) in seen_queries for q in queries),
        'split_policy':'Complete normalized persona-description groups, deterministic 80/20; only training groups enter the probe.',
        'selection_policy':'32 training groups, two distinct queries each, fixed hashes; no score/key-based sampling.',
        'official_val_or_benchmark_content_loaded':False,
        'official_split_disjointness_verified':False,
        'official_split_documentation_conflict':'README claims disjoint persona IDs; pinned public code shuffles remaining rows for train/val and can split the final benchmark persona. Actual official split overlap has not been measured.',
        'synthetic_personas_not_real_human_users':True,
        'cross_dataset_persona_identity_verified':False,
        'input_allowlist':['current user query','four official response strings','referenced history role/content and original message order'],
        'oracle_fields_excluded':['short_persona','expanded_persona','preference','prev_pref','related_conversation_snippet','updated','who','pref_type','topic_query','topic_preference','sensitive_info'],
        'history_audit_pending':True}
    split = {'version':'pm2-internal-persona-split-v1','csv_sha256':audit['train_csv_sha256'],
        'train_groups':train,'reserved_internal_validation_groups':reserved,
        'probe_groups':chosen_groups,
        'persona_ids_by_group':{g:sorted({r['persona_id'] for r in v},key=int) for g,v in groups.items()}}
    if set(train)&set(reserved):
        raise ValueError('Persona split overlaps')
    study.mkdir(parents=True)
    for name, value in (('data_audit.json',audit),('split.json',split),('probe_inputs.json',queries),('references.private.json',references)):
        write_json(study/name,value)
    print('PM2_DATA_AUDIT',json.dumps({k:audit[k] for k in ('rows','distinct_persona_ids','training_groups','reserved_internal_validation_groups','probe_questions')}),flush=True)
    return audit


def load_history(query):
    path = DATA/history_path(query['history_path'])
    source = json.loads((DATA/'source_manifest.json').read_text(encoding='utf-8'))
    if source['files'][query['history_path']]['sha256'] != sha256(path):
        raise ValueError('History publisher checksum changed')
    value = json.loads(path.read_text(encoding='utf-8'))
    if str(value['metadata']['persona_id']) != query['persona_id']:
        raise ValueError('History belongs to another persona')
    messages = value['chat_history']
    if any(not isinstance(m.get('content'),str) or m['role'] not in {'system','user','assistant'} for m in messages):
        raise ValueError('Unexpected multimodal or unknown history content')
    # The native task appends a new query after this complete history. The system
    # message contains an oracle persona profile and must never be a memory.
    turns = [Turn(f"p{query['persona_id']}:m{i:04d}:t0000",m['role'],m['content'],'')
             for i,m in enumerate(messages) if m['role'] in {'user','assistant'}]
    if any(normalize(t.content)==normalize(query['question']) for t in turns if t.role=='user'):
        raise ValueError('Current query already exists in supplied history')
    return turns, {'history_sha256':sha256(path),'messages':len(messages),'retained_turns':len(turns),
                   'removed_system_messages':sum(m['role']=='system' for m in messages)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan',))
    args=parser.parse_args()
    return plan()


if __name__=='__main__':
    main()
