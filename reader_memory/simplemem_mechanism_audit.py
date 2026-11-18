"""CPU-only development audit, separate from the frozen baseline and its scores.

Gold evidence is used solely to build an explicitly private review packet. Window
membership is a candidate source relation, never proof of semantic preservation.
No model inference, graph construction, heldout data, or baseline writes occur.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .schema import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('simplemem', 'raw_dense')
LABELS = ('no_material_error', 'write_loss', 'retrieval_omission', 'reader_failure',
          'source_ambiguity', 'reference_or_metric_issue', 'mixed', 'unknown')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_id(value, available):
    """Diagnostic alias only; original frozen evidence and scores stay untouched."""
    if value in available:
        return {'original': value, 'resolved': value, 'resolution': 'exact'}
    match = re.fullmatch(r'D:(\d+):(\d+)', value)
    alias = f'D{match[1]}:{match[2]}' if match else None
    if alias in available:
        return {'original': value, 'resolved': alias,
                'resolution': 'diagnostic_alias_extra_colon'}
    return {'original': value, 'resolved': None, 'resolution': 'unresolved'}


def read_stream(path, inputs):
    """Ignore an unterminated live append, but fail on malformed completed rows."""
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b'\n') + 1
    complete = data[:end]
    inputs[str(path)] = {'sha256_complete_prefix': digest(complete),
                         'complete_bytes': len(complete),
                         'uncommitted_tail_bytes': len(data) - end}
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


def reflection_result(call):
    systems = [m.get('content', '') for m in call.get('messages', [])
               if m.get('role') == 'system']
    if not any('information completeness evaluator' in s for s in systems):
        return None
    # Read exact JSON only; do not silently repair or invent planner judgments.
    text = call.get('text', '')
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and value.get('assessment') in ('complete', 'incomplete'):
            return {'call_id': call['call_id'], 'assessment': value['assessment'],
                    'coverage_percentage': value.get('coverage_percentage'),
                    'reasoning': value.get('reasoning'),
                    'missing_info_types': value.get('missing_info_types')}
    return {'call_id': call['call_id'], 'assessment': 'unparsed'}


def source_coverage(ids, evidence):
    resolved = [e['resolved'] for e in evidence if e['resolved']]
    hit = sorted(set(ids) & set(resolved))
    return {'resolved_evidence_ids': resolved, 'hit_ids': hit,
            'all_resolved_ids_hit': bool(resolved) and set(resolved) <= set(ids),
            'all_annotated_ids_resolved_and_hit': bool(evidence) and
                all(e['resolved'] is not None and e['resolved'] in ids for e in evidence),
            'has_unresolved_annotation': any(e['resolved'] is None for e in evidence),
            'interpretation': 'source membership only; not semantic sufficiency'}


def build(root=ROOT, run=None, output=None):
    root = Path(root).resolve()
    run = Path(run or root / 'runs/baseline_bootstrap_v2').resolve()
    output = Path(output or root / 'runs/simplemem_mechanism_v1').resolve()
    if output == run or run in output.parents:
        raise ValueError('Audit output must be outside the frozen baseline directory')
    plan = root / 'research/baseline_bootstrap_v2'
    inputs = {}

    def load(path, default=None):
        if not path.exists() and default is not None:
            return default
        data = path.read_bytes()
        inputs[str(path)] = {'sha256': digest(data), 'bytes': len(data)}
        return json.loads(data)

    manifest = load(plan / 'manifest.json')
    for name, expected in manifest['files'].items():
        if digest((plan / name).read_bytes()) != expected:
            raise ValueError('Frozen input mismatch: ' + name)
    identity = load(run / 'identity.json')
    signature = fingerprint(identity)
    if identity['manifest_sha256'] != digest((plan / 'manifest.json').read_bytes()):
        raise ValueError('Run manifest mismatch')
    queries = load(plan / 'public/queries.json')
    conversations = load(plan / 'public/conversations.json')
    refs = {r['question_id']: r for r in load(plan / 'private/references.json')}
    if len(refs) != len(queries) or {q['question_id'] for q in queries} != set(refs):
        raise ValueError('Public/private question identity mismatch')
    predictions = read_stream(run / 'predictions.jsonl', inputs)
    calls = read_stream(run / 'calls.jsonl', inputs)
    parses = read_stream(run / 'parses.jsonl', inputs)
    searches = read_stream(run / 'searches.jsonl', inputs)
    for row in predictions + calls + parses + searches:
        if row.get('signature') != signature:
            raise ValueError('Mixed run signatures')
    expected = {(q['question_id'], method) for q in queries for method in METHODS}
    pred = {(r['question_id'], r['method']): r for r in predictions}
    if len(pred) != len(predictions) or not set(pred) <= expected:
        raise ValueError('Duplicate or out-of-protocol prediction')
    scored = load(run / 'scored.json', [])
    scored_map = {(r['question_id'], r['method']): r for r in scored}
    if len(scored_map) != len(scored) or not set(scored_map) <= set(pred):
        raise ValueError('Score rows do not match observed predictions')
    for key, row in scored_map.items():
        if row.get('signature') != signature or row['hypothesis'] != pred[key]['hypothesis']:
            raise ValueError('Stale or incompatible score row')
    calls_by_q = defaultdict(list)
    for call in calls:
        if 'question_id' in call:
            calls_by_q[(call['question_id'], call['method'])].append(call)
    windows, entry_windows, raw_sources = {}, {}, {}
    corpus = []
    for conv in conversations:
        cid = conv['conversation_id']
        by_id = {t['source_id']: t for t in conv['turns']}
        if len(by_id) != len(conv['turns']):
            raise ValueError('Duplicate source IDs')
        current = [load(p) for p in sorted((run / cid / 'windows').glob('*.json'))]
        windows[cid] = current
        origins = defaultdict(list)
        for w in current:
            if w.get('signature') != signature or not set(w['source_ids']) <= set(by_id):
                raise ValueError('Invalid window source/signature')
            for entry in w['entries']:
                origins[entry['entry_id']].append(w['window'])
        entry_windows[cid] = origins
        raw = load(run / cid / 'raw_chunks.json', [])
        raw_sources[cid] = {r['entry']['entry_id']: r['source_ids'] for r in raw}
        texts = Counter(e['lossless_restatement'].strip() for w in current for e in w['entries'])
        seen_sources = {s for w in current for s in w['source_ids']}
        corpus.append({'conversation_id': cid, 'turns': len(by_id), 'windows': len(current),
            'covered_turns': len(seen_sources), 'entries': sum(len(w['entries']) for w in current),
            'empty_windows': sum(not w['entries'] for w in current),
            'all_calls_failed_windows': sum(w.get('all_calls_failed', False) for w in current),
            'exact_duplicate_text_extras': sum(n - 1 for n in texts.values()),
            'source_text_utf8_bytes': sum(len(t['content'].encode()) for t in conv['turns'])})
    cases = []
    for q in queries:
        qid, cid = q['question_id'], q['conversation_id']
        conv = next(c for c in conversations if c['conversation_id'] == cid)
        turns = conv['turns']
        by_id = {t['source_id']: t for t in turns}
        evidence = [source_id(s, by_id) for s in refs[qid]['evidence']]
        resolved = {e['resolved'] for e in evidence if e['resolved']}
        source_neighbors = set()
        for index, turn in enumerate(turns):
            if turn['source_id'] in resolved:
                source_neighbors.update(range(max(0, index - 1), min(len(turns), index + 2)))
        candidate = [w for w in windows[cid] if set(w['source_ids']) & resolved]
        results = {}
        for method in METHODS:
            key = qid, method
            row = pred.get(key)
            if row is None:
                results[method] = {'status': 'pending'}
                continue
            context_ids = [e['entry_id'] for e in row['contexts']]
            reflection = [v for c in calls_by_q[key] if (v := reflection_result(c))]
            details = {**row, 'official_native_f1': scored_map.get(key, {}).get('official_native_f1'),
                       'reflection_assessments': reflection}
            if method == 'raw_dense':
                ids = {s for e in context_ids for s in raw_sources[cid].get(e, [])}
                details['raw_source_coverage'] = source_coverage(ids, evidence)
            else:
                possible = {wi for e in context_ids for wi in entry_windows[cid].get(e, [])}
                ids = {s for w in windows[cid] if w['window'] in possible for s in w['source_ids']}
                details['candidate_window_coverage'] = source_coverage(ids, evidence)
                details['candidate_window_coverage']['interpretation'] = (
                    'upper-bound candidate window membership; not entry-level provenance or evidence recall')
                details['unmapped_context_ids'] = [e for e in context_ids if e not in entry_windows[cid]]
            results[method] = details
        cases.append({**q, 'reference': refs[qid]['answer'], 'diagnostic_only': True,
            'evidence_resolution': evidence,
            'gold_source_turns': [t for t in turns if t['source_id'] in resolved],
            'source_neighbors_radius_1': [turns[i] for i in sorted(source_neighbors)],
            'candidate_source_windows': [{'window': w['window'], 'source_ids': w['source_ids'],
                                          'entries': w['entries']} for w in candidate],
            'predictions': results,
            'semantic_review': {'label': 'unknown', 'reviewer': None, 'human_gold': False,
                'reason': 'Requires explicit semantic review; no attribution from lexical/source overlap.'}})
    costs = defaultdict(lambda: {'calls': 0, 'failed_calls': 0, 'prompt_tokens': 0,
        'completion_tokens': 0, 'wall_seconds': 0, 'missing_usage_calls': 0})
    for c in calls:
        cost = costs[c['method'] + ':' + c['stage']]
        cost['calls'] += 1
        cost['failed_calls'] += c['status'] != 'complete'
        cost['wall_seconds'] += c.get('wall_seconds', 0)
        cost['missing_usage_calls'] += not bool(c.get('usage'))
        for field in ('prompt_tokens', 'completion_tokens'):
            cost[field] += c.get('usage', {}).get(field, 0)
    groups = defaultdict(list)
    for key, row in scored_map.items():
        for scope in ('overall', 'category:' + str(row['category']),
                      'conversation:' + row['conversation_id']):
            groups[(row['method'], scope)].append(row['official_native_f1'])
    scores = [{'method': m, 'scope': scope, 'n': len(values),
               'native_f1': sum(values) / len(values)} for (m, scope), values in sorted(groups.items())]
    paired = [q for q in queries if all((q['question_id'], m) in scored_map for m in METHODS)]
    paired_delta = [scored_map[(q['question_id'], 'simplemem')]['official_native_f1'] -
                    scored_map[(q['question_id'], 'raw_dense')]['official_native_f1'] for q in paired]
    summary = {'version': 'simplemem-mechanism-audit-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature,
        'baseline_status': load(run / 'status.json', {}),
        'queue_status': load(run / 'queue_status.json', {}),
        'expected_predictions': len(expected), 'observed_predictions': len(pred),
        'complete_predictions': set(pred) == expected, 'observed_scores': len(scored_map),
        'cases': len(cases), 'semantic_reviewed_cases': 0, 'human_gold_cases': 0,
        'corpus': corpus, 'scores': scores, 'costs': dict(costs),
        'paired_scored_questions': len(paired),
        'paired_native_f1_delta': sum(paired_delta)/len(paired_delta) if paired_delta else None,
        'paired_wins_ties_losses': {'wins': sum(d > 0 for d in paired_delta),
            'ties': sum(d == 0 for d in paired_delta), 'losses': sum(d < 0 for d in paired_delta)},
        'generation_failures': sum(c['status'] != 'complete' for c in calls),
        'parse_failures': sum(p['status'] == 'failed' for p in parses),
        'retrieval_failures': sum(s['status'] == 'failed' for s in searches),
        'truncated_calls': sum(c.get('finish_reason') == 'length' for c in calls),
        'evidence_resolution_counts': dict(Counter(e['resolution'] for c in cases for e in c['evidence_resolution'])),
        'semantic_label_contract': list(LABELS),
        'limitations': ['Two development conversations; no SOTA or significance claim.',
            'Gold evidence is diagnostic only, never a feature for deployed retrieval.',
            'Window membership is not evidence recall or semantic preservation.',
            'F1 is not accuracy; incomplete answers may receive partial credit.',
            'No semantic failure attribution has been automated or validated.',
            'Baseline systems are not token/compute matched; costs are observational.',
            'Missing model usage is reported, not treated as a known zero cost.'],
        'inputs': inputs, 'audit_code_sha256': digest(Path(__file__).read_bytes())}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'audit.json', summary)
    for name, rows in [('cases.private.jsonl', cases), ('review_queue.private.jsonl', [
            {'question_id': c['question_id'], 'conversation_id': c['conversation_id'],
             'label': 'unknown', 'reviewer': None, 'human_gold': False,
             'source_sufficient': 'unknown', 'memory_semantics_preserved': 'unknown',
             'retrieved_evidence_sufficient': 'unknown', 'supporting_entry_ids': [],
             'method_labels': {method: 'unknown' for method in METHODS},
             'supporting_source_ids': [], 'rationale': ''} for c in cases])]:
        target = output / name
        # Queue template is regenerated separately; completed reviews belong in reviews.ai.jsonl.
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        temporary.replace(target)
    lines = ['# SimpleMem 机制诊断准备与基线核对', '',
        '这是开发诊断，不是新方法实验；private 文件包含答案和证据，不能接入部署选择器。', '',
        f"快照时间（UTC）：{summary['created_utc']}。问答 {len(pred)}/{len(expected)}，已评分 {len(scored_map)}。",
        f'全题诊断包：{len(cases)} 题。语义归因尚未完成，不从词面或窗口重合推断原因。', '',
        '| 会话 | 原文轮数 | 已写窗口 | 覆盖轮数 | 记忆条数 | 空窗口 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {c['conversation_id']} | {c['turns']} | {c['windows']} | {c['covered_turns']} | {c['entries']} | {c['empty_windows']} |" for c in corpus]
    lines += ['', f"生成失败 {summary['generation_failures']}；解析失败 {summary['parse_failures']}；检索失败 {summary['retrieval_failures']}；长度截断 {summary['truncated_calls']}。", '',
        '先逐题核查原始来源与记忆语义，再判断检索遗漏和读取错误。允许混合原因与无法判断。',
        '原文精确来源覆盖与压缩记忆候选窗口覆盖分开；后者只是来源范围上界。',
        '官方 F1 只读取冻结基线独立评分产物，未评分题不记零，也不计算虚假全量均值。', '',
        '| 方法 | 范围 | 已评分数 | F1（0–100） |', '| --- | --- | ---: | ---: |']
    lines += [f"| {s['method']} | {s['scope']} | {s['n']} | {100*s['native_f1']:.2f} |" for s in scores]
    lines += ['', '各阶段调用、输入/输出 token、失败调用与输入文件哈希见 audit.json。',
              '画像、依赖构建和同预算方法尚未训练或运行。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build(args.root, args.run, args.output)
    print(json.dumps({k: result[k] for k in ('created_utc', 'observed_predictions',
        'expected_predictions', 'observed_scores', 'cases', 'complete_predictions')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
