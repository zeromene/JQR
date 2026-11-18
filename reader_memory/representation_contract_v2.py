"""Separately frozen evidence-contract retest on the already exposed dev units."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .choice_protocol_audit import save
from .diagnostics import sha256
from .four_reader_probe import READERS, probe_assets
from .native_choice_pilot import read
from . import representation_development as prior
from .schema import fingerprint, read_jsonl
from .source_records import render_records, reviewer_messages, source_text
from .source_records_v2 import catalog, parse_writer, writer_messages

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'research/representation_development/v2'
OUTPUT = ROOT / 'runs/representation_development_v2'
WRITER, REVIEWER = prior.WRITER, prior.REVIEWER
WRITER_CAP, REVIEWER_CAP = 1024, 768
POLICY = {'writer': {'allocation_cap_mib': 12288, 'minimum_free_mib': 13312},
          'review': {'allocation_cap_mib': 21504, 'minimum_free_mib': 22528}}


def prepare(counter):
    if STUDY.exists():
        raise FileExistsError('Preserve frozen v2 inputs')
    pm, units, _ = prior.load()
    if not counter.exact or read(prior.OUTPUT / 'input_integrity.json')['status'] != 'passed':
        raise ValueError('Verified v1 evidence and exact tokenizer required')
    ids = read(prior.STUDY / 'review_unit_ids.json')
    catalogs = {u['unit_id']: catalog(u) for u in units}
    prompts = {u['unit_id']: prior.prompt_entry(counter, writer_messages(u)) for u in units}
    if len(units) != 64 or len(ids) != 32 or any(not v for v in catalogs.values()):
        raise ValueError('Invalid reused sample or unreferenceable unit')
    STUDY.mkdir(parents=True)
    for name, value in (('units.json', units), ('catalogs.json', catalogs),
                        ('writer_prompts.json', prompts), ('review_unit_ids.json', ids)):
        save(STUDY / name, value)
    paths = ('research/REPRESENTATION_CONTRACT_V2_PLAN.md', 'reader_memory/source_records_v2.py',
        'reader_memory/representation_contract_v2.py', 'tools/run_representation_contract_v2.py',
        'tools/audit_representation_contract_v2.py', 'reader_memory/source_records.py',
        'reader_memory/representation_development.py', 'reader_memory/backend.py', 'reader_memory/schema.py',
        'reader_memory/tokenization.py', 'reader_memory/four_reader_probe.py',
        'research/representation_development/v1/manifest.json',
        'runs/representation_development_v1/writer.jsonl', 'runs/representation_development_v1/reviewer.jsonl',
        'runs/representation_development_v1/input_integrity.json')
    manifest = {'version': 'representation-contract-v2', 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'file_hashes': {p: sha256(ROOT / p) for p in paths},
        'input_hashes': {p: sha256(STUDY / p) for p in ('units.json', 'catalogs.json', 'writer_prompts.json', 'review_unit_ids.json')},
        'prior_study_signature': fingerprint(pm), 'writer': WRITER, 'reviewer': REVIEWER,
        'writer_assets': probe_assets(ROOT / 'models' / WRITER), 'reviewer_assets': probe_assets(ROOT / 'models' / REVIEWER),
        'reader_tokenizer_assets': {r: probe_assets(ROOT / 'models' / r) for r in READERS},
        'writer_cap': WRITER_CAP, 'reviewer_cap': REVIEWER_CAP, 'execution_policy': POLICY,
        'source_groups': 16, 'writer_units': 64, 'review_units': 32,
        'reuses_exposed_development_units': True, 'new_independent_validation': False,
        'confirmation_sources_read': False, 'reserved_validation_used': False}
    save(STUDY / 'manifest.json', manifest)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print('CONTRACT_V2_FROZEN', 64, 32, flush=True)


def load():
    m = read(STUDY / 'manifest.json')
    pm, old_units, _ = prior.load()
    if (fingerprint(pm) != m['prior_study_signature']
            or any(sha256(ROOT / p) != h for p, h in m['file_hashes'].items())
            or any(sha256(STUDY / p) != h for p, h in m['input_hashes'].items())
            or m['execution_policy'] != POLICY):
        raise ValueError('Frozen v2 dependency/input changed')
    units = read(STUDY / 'units.json')
    if units != old_units or read(STUDY / 'review_unit_ids.json') != read(prior.STUDY / 'review_unit_ids.json'):
        raise ValueError('V2 changed development units or review selection')
    return m, units, read(STUDY / 'writer_prompts.json')


def decode_writer(response, unit):
    if response['usage']['completion_tokens'] >= WRITER_CAP:
        return {'status': 'output-cap', 'derived': None, 'error': 'writer output cap reached'}
    try:
        return {'status': 'accepted', 'derived': parse_writer(response['text'], unit), 'error': None}
    except (ValueError, KeyError, TypeError) as exc:
        return {'status': 'mechanical-rejection', 'derived': None, 'error': repr(exc)}


def check_usage(response, entry, cap):
    usage = response['usage']
    if (usage['prompt_tokens'] != entry['tokens'] or not 0 <= usage['completion_tokens'] <= cap
            or not math.isfinite(usage['elapsed_seconds']) or usage['elapsed_seconds'] < 0):
        raise ValueError('Prompt/cost provenance mismatch')


def verify_writer(rows, m, units, prompts):
    byid = {u['unit_id']: u for u in units}; seen = set()
    for row in rows:
        uid = row['unit_id']
        if uid not in byid or uid in seen:
            raise ValueError('Duplicate or unknown writer unit')
        unit = byid[uid]; entry = prompts[uid]
        check_usage(row['response'], entry, WRITER_CAP)
        expected = {'unit_id': uid, 'study_signature': fingerprint(m), 'unit_hash': fingerprint(unit),
            'prompt_entry': entry, 'response': row['response'], **decode_writer(row['response'], unit)}
        if row != expected or entry['messages_hash'] != fingerprint(writer_messages(unit)):
            raise ValueError('Writer derivation changed')
        seen.add(uid)
    return seen


def review_inputs(counter):
    m, units, prompts = load(); rows = list(read_jsonl(OUTPUT / 'writer.jsonl'))
    if len(verify_writer(rows, m, units, prompts)) != 64:
        raise ValueError('Complete writer collection required')
    byid = {u['unit_id']: u for u in units}; written = {r['unit_id']: r for r in rows}
    entries = {uid: (prior.prompt_entry(counter, reviewer_messages(byid[uid], written[uid]['derived']['records']))
                    if written[uid]['status'] == 'accepted' else None) for uid in read(STUDY / 'review_unit_ids.json')}
    assets = probe_assets(ROOT / 'models' / REVIEWER)
    if assets != m['reviewer_assets']:
        raise ValueError('Reviewer assets changed')
    return {'study_signature': fingerprint(m), 'writer_sha256': sha256(OUTPUT / 'writer.jsonl'),
            'reviewer_assets': assets, 'entries': entries}


def freeze_review(counter):
    value = review_inputs(counter); path = OUTPUT / 'review_inputs.json'
    if path.exists() and read(path) != value:
        raise ValueError('Preserve frozen reviewer inputs')
    if not path.exists():
        save(path, value)
    print('CONTRACT_V2_REVIEW_INPUTS_FROZEN', 32, flush=True)


def verify_review(rows, m, units, written, inputs):
    byid = {u['unit_id']: u for u in units}; seen = set()
    if (inputs['study_signature'] != fingerprint(m) or inputs['writer_sha256'] != sha256(OUTPUT / 'writer.jsonl')
            or set(inputs['entries']) != set(read(STUDY / 'review_unit_ids.json'))):
        raise ValueError('Wrong review collection')
    for row in rows:
        uid = row['unit_id']
        if uid not in inputs['entries'] or uid in seen:
            raise ValueError('Duplicate/unknown reviewer unit')
        base = {'unit_id': uid, 'study_signature': fingerprint(m), 'writer_row_hash': fingerprint(written[uid]),
            'review_inputs_sha256': sha256(OUTPUT / 'review_inputs.json'), 'prompt_entry': inputs['entries'][uid]}
        if written[uid]['status'] != 'accepted':
            expected = {**base, 'response': None, 'status': 'skipped-writer-unavailable', 'review': None,
                        'error': written[uid]['error']}
        else:
            entry = inputs['entries'][uid]; response = row['response']
            check_usage(response, entry, REVIEWER_CAP)
            if entry['messages_hash'] != fingerprint(reviewer_messages(byid[uid], written[uid]['derived']['records'])):
                raise ValueError('Reviewer message changed')
            expected = {**base, 'response': response, **prior.decode_review(response, written[uid]['derived']['records'])}
        if row != expected:
            raise ValueError('Reviewer derivation changed')
        seen.add(uid)
    return seen


def run_model(stage, backend, counter):
    m, units, prompts = load(); byid = {u['unit_id']: u for u in units}
    model = WRITER if stage == 'writer' else REVIEWER
    if backend.model_name != model or probe_assets(backend.path) != m['writer_assets' if stage == 'writer' else 'reviewer_assets']:
        raise ValueError('Wrong model/assets')
    path = OUTPUT / ('writer.jsonl' if stage == 'writer' else 'reviewer.jsonl')
    rows = list(read_jsonl(path)) if path.exists() else []
    if stage == 'writer':
        done = verify_writer(rows, m, units, prompts); ids = list(byid)
    else:
        raw = list(read_jsonl(OUTPUT / 'writer.jsonl'))
        if len(verify_writer(raw, m, units, prompts)) != 64:
            raise ValueError('Incomplete writer')
        written = {r['unit_id']: r for r in raw}; inputs = read(OUTPUT / 'review_inputs.json')
        done = verify_review(rows, m, units, written, inputs); ids = read(STUDY / 'review_unit_ids.json')
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        for uid in ids:
            if uid in done:
                continue
            unit = byid[uid]
            if stage == 'writer':
                messages = writer_messages(unit); entry = prior.prompt_entry(counter, messages)
                if entry != prompts[uid]:
                    raise ValueError('Actual writer token IDs changed')
                response = backend.complete(messages, max_tokens=WRITER_CAP)
                row = {'unit_id': uid, 'study_signature': fingerprint(m), 'unit_hash': fingerprint(unit),
                       'prompt_entry': entry, 'response': response, **decode_writer(response, unit)}
                verify_writer([row], m, units, prompts)
            else:
                entry = inputs['entries'][uid]
                row = {'unit_id': uid, 'study_signature': fingerprint(m), 'writer_row_hash': fingerprint(written[uid]),
                       'review_inputs_sha256': sha256(OUTPUT / 'review_inputs.json'), 'prompt_entry': entry}
                if written[uid]['status'] != 'accepted':
                    row.update(response=None, status='skipped-writer-unavailable', review=None, error=written[uid]['error'])
                else:
                    records = written[uid]['derived']['records']; messages = reviewer_messages(unit, records)
                    if prior.prompt_entry(counter, messages) != entry:
                        raise ValueError('Actual reviewer token IDs changed')
                    response = backend.complete(messages, max_tokens=REVIEWER_CAP)
                    row.update(response=response, **prior.decode_review(response, records))
                verify_review([row], m, units, written, inputs)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush(); done.add(uid)
            save(OUTPUT / 'status.json', {'stage': stage, 'units': len(done), 'expected_units': len(ids)})
            print('CONTRACT_V2', stage, len(done), '/', len(ids), row['status'], flush=True)


def report(counters):
    m, units, prompts = load(); writer = list(read_jsonl(OUTPUT / 'writer.jsonl'))
    if (set(counters) != set(READERS) or not all(c.exact for c in counters.values())
            or {r: probe_assets(ROOT / 'models' / r) for r in READERS} != m['reader_tokenizer_assets']):
        raise ValueError('Exact, frozen reader tokenizers required')
    if len(verify_writer(writer, m, units, prompts)) != 64:
        raise ValueError('Incomplete writer')
    written = {r['unit_id']: r for r in writer}; reviewer = list(read_jsonl(OUTPUT / 'reviewer.jsonl'))
    if len(verify_review(reviewer, m, units, written, read(OUTPUT / 'review_inputs.json'))) != 32:
        raise ValueError('Incomplete reviewer')
    valid = [r for r in writer if r['status'] == 'accepted']; byid = {u['unit_id']: u for u in units}
    records = [v for r in valid for v in r['derived']['records']]
    judged = [v for r in reviewer if r['status'] == 'reviewed' for v in r['review']['reviews']]
    lengths = []
    for row in valid:
        uid = row['unit_id']; rs = row['derived']['records']
        texts = {'raw': source_text(byid[uid]['spans']), 'table': render_records(rs, 'table'), 'compact': render_records(rs, 'compact')}
        lengths.append({'unit_id': uid, 'records': len(rs),
            'tokens_by_reader': {r: {a: c.count(t) for a, t in texts.items()} for r, c in counters.items()}})
    human = []
    for uid in read(STUDY / 'review_unit_ids.json'):
        row = written[uid]
        human.append({'case_id': 'blind-' + fingerprint(['source-record-human-v2', uid])[:20],
            'source_group': byid[uid]['source_group'], 'source_unit': byid[uid]['spans'],
            'candidate_records': row['derived']['records'] if row['status'] == 'accepted' else [],
            'writer_available': row['status'] == 'accepted', 'all_claims_supported': None,
            'attribution_correct': None, 'conditions_and_state_preserved': None, 'important_omissions': None, 'notes': None})
    human.sort(key=lambda c: c['case_id'])
    value = {'status': 'completed-contract-v2-development-retest', 'source_groups': 16, 'writer_units': 64, 'review_units': 32,
        'writer_status_counts': dict(Counter(r['status'] for r in writer)),
        'writer_error_counts': dict(Counter(r['error'] for r in writer if r['status'] != 'accepted')),
        'accepted_empty_units': sum(not r['derived']['records'] for r in valid),
        'accepted_nonempty_units': sum(bool(r['derived']['records']) for r in valid), 'records': len(records),
        'record_kind_counts': dict(Counter(r['kind'] for r in records)),
        'literal_citations_checked': sum(len(r['evidence']) for r in records),
        'review_status_counts': dict(Counter(r['status'] for r in reviewer)),
        'ai_record_verdict_counts': dict(Counter(v['verdict'] for v in judged)),
        'ai_issue_counts': dict(Counter(i for v in judged for i in v['issues'])),
        'ai_units_with_omissions': sum(bool(r['review']['omissions']) for r in reviewer if r['status'] == 'reviewed'),
        'ai_omission_notes': sum(len(r['review']['omissions']) for r in reviewer if r['status'] == 'reviewed'),
        'actual_writer_calls': 64, 'actual_reviewer_calls': sum(r['response'] is not None for r in reviewer),
        'writer_seconds': sum(r['response']['usage']['elapsed_seconds'] for r in writer),
        'reviewer_seconds': sum(r['response']['usage']['elapsed_seconds'] for r in reviewer if r['response'] is not None),
        'writer_completion_tokens': sum(r['response']['usage']['completion_tokens'] for r in writer),
        'reviewer_completion_tokens': sum(r['response']['usage']['completion_tokens'] for r in reviewer if r['response'] is not None),
        'writer_prompt_tokens': sum(r['response']['usage']['prompt_tokens'] for r in writer),
        'reviewer_prompt_tokens': sum(r['response']['usage']['prompt_tokens'] for r in reviewer if r['response'] is not None),
        'mean_tokens_on_accepted_units': {r: {a: sum(v['tokens_by_reader'][r][a] for v in lengths) / len(lengths) if lengths else None
            for a in ('raw', 'table', 'compact')} for r in counters},
        'reuses_exposed_development_units': True, 'new_independent_validation': False,
        'semantic_precision_measured_against_humans': False, 'semantic_recall_measured': False,
        'lossless_compression_established': False, 'ai_review_is_human_gold': False,
        'task_accuracy_measured': False, 'selector_trained': False,
        'confirmation_sources_read': False, 'reserved_validation_used': False,
        'study_manifest_sha256': sha256(STUDY / 'manifest.json'), 'writer_sha256': sha256(OUTPUT / 'writer.jsonl'),
        'reviewer_sha256': sha256(OUTPUT / 'reviewer.jsonl')}
    for name, v in (('lengths.json', lengths), ('human_review.blank.json', human), ('report.json', value)):
        save(OUTPUT / name, v)
    print('CONTRACT_V2_REPORT', json.dumps(value, ensure_ascii=False), flush=True)
    return value


def render():
    r = read(OUTPUT / 'report.json'); a = read(OUTPUT / 'input_integrity.json'); load()
    if a['status'] != 'passed' or any(r[k] != sha256(OUTPUT / f) or a[k] != r[k]
        for f, k in (('writer.jsonl', 'writer_sha256'), ('reviewer.jsonl', 'reviewer_sha256'))):
        raise ValueError('Complete audited outputs required')
    lines = ['# 来源引用契约修复 v2：开发重测', '',
        '本轮复用已曝光的 16 来源、64 个完整单元与原定 32 个预审 ID，不是独立验证。v1 输出和协议保持原样。', '',
        '模型选择本地证据编号，由程序还原逐字引用；完整原文始终可见。来源展示、引用方式和主体字段契约同时改变，'
        '不能把前后差异全部归因于 ID 修复。引用正确不等于语义正确。', '',
        f"writer：`{r['writer_status_counts']}`；非空通过 {r['accepted_nonempty_units']}/64，空提取 {r['accepted_empty_units']}/64。"
        f"共 {r['records']} 条记录、{r['literal_citations_checked']} 条逐字引用。", '',
        f"机械错误：`{r['writer_error_counts']}`。失败保留，不重试或修补原始结果。", '',
        f"32 单元预审状态：`{r['review_status_counts']}`；实际模型调用 {r['actual_reviewer_calls']} 次。"
        f"AI 逐记录判断：`{r['ai_record_verdict_counts']}`；问题：`{r['ai_issue_counts']}`；"
        f"{r['ai_units_with_omissions']} 单元有遗漏提示，共 {r['ai_omission_notes']} 条。", '',
        'AI 预审不是人工 gold；空提取、未审和失败均不算语义支持，不用 AI 赞同率估计精度。', '',
        '| 记录类别 | 数量 |', '| --- | ---: |']
    lines += [f'| {k} | {v} |' for k, v in r['record_kind_counts'].items()]
    lines += ['', '| tokenizer | 原文平均 token | 命题表 | 紧凑表达 |', '| --- | ---: | ---: | ---: |']
    for reader, v in r['mean_tokens_on_accepted_units'].items():
        lines.append('| ' + reader + ' | ' + ' | '.join('—' if v[k] is None else f'{v[k]:.1f}' for k in ('raw', 'table', 'compact')) + ' |')
    lines += ['', '长度只统计机械通过单元，包含空提取。最多两条记录不保证覆盖，原文另行全保留；'
        '短表示不等于有效或无损压缩，也不能据此声称总存储减少。', '',
        f"真实 writer/reviewer 调用 {r['actual_writer_calls']}/{r['actual_reviewer_calls']}，生成 token "
        f"{r['writer_completion_tokens']}/{r['reviewer_completion_tokens']}，推理累计 "
        f"{r['writer_seconds']:.1f}/{r['reviewer_seconds']:.1f} 秒。共享 GPU 的耗时不是独占速度；加载和审计另计。", '',
        f"独立审计通过：{a['writer_rows_checked']} 条 writer、{a['reviewer_rows_checked']} 条 reviewer，"
        f"{a['literal_output_citations_checked']} 条实际引用及完整提示/token ID 均已核对。", '',
        '盲审空白包为 `../runs/representation_development_v2/human_review.blank.json`，不含 AI verdict。'
        '先核查非空候选的具体归属、条件、状态错误及遗漏，再判断是否进入读者可用性诊断。', '',
        '本轮未测人工语义精度/召回、答题收益、能力条件收益或联合预算；未训练选择器。'
        '48 个确认来源正文及原有保留集继续封存。', '',
        '协议见 [REPRESENTATION_CONTRACT_V2_PLAN.md](REPRESENTATION_CONTRACT_V2_PLAN.md)，'
        '冻结输入见 `representation_development/v2/`，原始输出、资源记录和审计见 `../runs/representation_development_v2/`。', '']
    (ROOT / 'research/REPRESENTATION_CONTRACT_V2_REPORT.md').write_bytes('\n'.join(lines).encode('utf-8'))
    print('CONTRACT_V2_RENDERED', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'writer', 'freeze-review', 'review', 'report', 'render'))
    command = p.parse_args().command
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    if command == 'prepare': return prepare(TokenCounter(ROOT / 'models' / WRITER))
    if command == 'freeze-review': return freeze_review(TokenCounter(ROOT / 'models' / REVIEWER))
    if command == 'report': return report({r: TokenCounter(ROOT / 'models' / r) for r in READERS})
    if command == 'render': return render()
    from .backend import TransformersBackend
    model = WRITER if command == 'writer' else REVIEWER
    run_model(command, TransformersBackend(ROOT / 'models' / model), TokenCounter(ROOT / 'models' / model))


if __name__ == '__main__':
    main()
