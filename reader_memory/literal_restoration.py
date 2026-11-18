"""Frozen literal-source restoration diagnostic on already exposed dev tasks."""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path

from .choice_protocol_audit import save
from .diagnostics import sha256
from .four_reader_probe import READERS, probe_assets, load as load_original
from .independent_pack import normalize
from .literal_evidence import canonical, characters, decode, includes, render
from .native_choice_pilot import classify, messages as native_messages, read, score_row
from .personamem2 import load_history, source_rows
from . import representation_development as writing_v1
from . import representation_contract_v2 as writing_v2
from .retrieval_answer_probe import load as load_answer_parent
from .schema import fingerprint, read_jsonl
from .shared_retrieval_audit import load as load_retrieval

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'research/literal_restoration/v1'
OUTPUT = ROOT / 'runs/literal_restoration_v1'
ARMS = ('retrieved', 'restored', 'structured', 'oracle')
CONTRASTS = (('structured', 'restored'), ('restored', 'retrieved'),
             ('oracle', 'retrieved'), ('oracle', 'restored'))
LIMIT = 12288
POLICY = {r: {'allocation_cap_mib': cap, 'minimum_free_mib': cap + 1024} for r, cap in
          zip(READERS, (8192, 14336, 24576, 14336))}


def messages(q, arm, shift):
    # Preserve the established answer instructions/options. Ledger construction
    # itself never accepts a question, an option, or a correct answer.
    adapter = {k: q[k] for k in ('question', 'options')}
    adapter['evidence'] = {f'{arm}:0': {'text': q['evidence'][arm]['text']}}
    return native_messages(adapter, arm, 0, shift)


def prompt_entry(counter, q, arm, shift):
    msg = messages(q, arm, shift)
    text = counter.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = counter.tokenizer.encode(text, add_special_tokens=False)
    return {'messages_hash': fingerprint(msg), 'token_ids_hash': fingerprint(ids), 'tokens': len(ids)}


def whole(turn):
    return {'source_id': turn.source_id, 'start': 0, 'end': len(turn.content),
            'role': turn.role, 'date': turn.date, 'text': turn.content}


def unit_baseline(counters):
    writing_v2.load(); _, units, _ = writing_v1.load()
    histories, _ = writing_v1.load_sources(); result = []
    for unit in units:
        ledger = canonical(histories[unit['source_group']][0], unit['spans'])
        views = {v: render(ledger, v) for v in ('raw', 'structured')}
        if any(decode(text, v) != ledger for v, text in views.items()):
            raise ValueError('Unit view does not roundtrip')
        result.append({'unit_id': unit['unit_id'], 'source_group': unit['source_group'],
            'source_ledger': ledger, 'source_ledger_hash': fingerprint(ledger), 'views': views,
            'tokens_by_reader': {r: {v: c.count(t) for v, t in views.items()} for r, c in counters.items()}})
    if len(result) != 64:
        raise ValueError('All 64 exposed units required')
    return result


def build(counters):
    _, original = load_original(); _, queries, _ = load_answer_parent(); _, retrieved = load_retrieval()
    old = {q['question_id']: q for q in original}; parents = {q['question_id']: q for q in retrieved}
    _, native = source_rows(); histories = {}; cases = []; prompts = {r: {} for r in READERS}
    for parent in queries:
        uid = parent['question_id']; source = old[uid]; row = native[source['row_index']]
        if row['user_query'] != source['question'] or str(int(row['persona_id'])) != source['persona_id']:
            raise ValueError('Official snippet source identity mismatch')
        if source['history_path'] not in histories:
            histories[source['history_path']] = load_history(source)[0]
        turns = histories[source['history_path']]
        try:
            snippet = json.loads(row['related_conversation_snippet'])
        except json.JSONDecodeError:
            snippet = ast.literal_eval(row['related_conversation_snippet'])
        if not isinstance(snippet, list) or not snippet or any(s['role'] not in {'user', 'assistant'} or not isinstance(s['content'], str) for s in snippet):
            raise ValueError('Unexpected official snippet schema')
        matched = [[t for t in turns if t.role == s['role'] and normalize(t.content) == normalize(s['content'])] for s in snippet]
        q = {k: parent[k] for k in ('question_id', 'source_group', 'persona_id', 'history_path', 'question', 'options', 'shifts')}
        q['alignment'] = {'annotation_hash': fingerprint(snippet),
            'matched_source_ids': [[t.source_id for t in ts] for ts in matched],
            'method': 'unique-role-normalized-text-match-original-history-payload',
            'literal_annotation_matches': [len(ts) == 1 and ts[0].content == s['content'] for ts, s in zip(matched, snippet)]}
        q.update(eligible=False, exclusion_reason=None, evidence=None)
        if not all(len(ts) == 1 for ts in matched):
            q['exclusion_reason'] = 'official-snippet-not-uniquely-aligned'
            cases.append(q); continue
        native_spans = parents[uid]['branches']['dense:2048']['spans']
        root = canonical(turns, native_spans)
        oracle = canonical(turns, [whole(ts[0]) for ts in matched])
        restored = canonical(turns, root + oracle)
        if not includes(restored, root) or not includes(restored, oracle):
            raise ValueError('Restoration lost source ranges')
        q['evidence'] = {}
        for arm, spans in (('retrieved', root), ('restored', restored), ('structured', restored), ('oracle', oracle)):
            view = 'structured' if arm == 'structured' else 'raw'
            text = render(spans, view)
            if decode(text, view) != spans:
                raise ValueError('Task view does not roundtrip')
            q['evidence'][arm] = {'source_ledger': spans, 'source_ledger_hash': fingerprint(spans),
                'text': text, 'tokens_by_reader': {r: c.count(text) for r, c in counters.items()}}
        q['restoration'] = {'inherited_dense_2k_spans_hash': fingerprint(native_spans),
            'retrieved_source_characters': characters(root), 'restored_source_characters': characters(restored),
            'added_source_characters': characters(restored) - characters(root),
            'official_source_fully_present_in_retrieved': includes(root, oracle)}
        entries = {r: {f'{a}:{s}': prompt_entry(c, q, a, s) for a in ARMS for s in q['shifts']}
                   for r, c in counters.items()}
        if max(e['tokens'] for reader in entries.values() for e in reader.values()) > LIMIT:
            q['exclusion_reason'] = 'complete-prompt-exceeds-common-context-limit'
        else:
            q['eligible'] = True
            for r in READERS:
                prompts[r][uid] = entries[r]
        cases.append(q)
    if len(cases) != 64 or len({q['source_group'] for q in cases}) != 32:
        raise ValueError('Must reuse all 64 exposed tasks from 32 sources')
    return cases, prompts, unit_baseline(counters)


def prepare(counters):
    if STUDY.exists():
        raise FileExistsError('Preserve frozen literal-restoration study')
    if set(counters) != set(READERS) or not all(c.exact for c in counters.values()):
        raise ValueError('Every exact reader tokenizer required')
    parent, _, _ = load_answer_parent()
    if {r: probe_assets(ROOT / 'models' / r) for r in READERS} != parent['reader_assets']:
        raise ValueError('Parent reader assets changed')
    cases, prompts, units = build(counters)
    eligible = [q for q in cases if q['eligible']]
    if not eligible:
        raise ValueError('No eligible tasks')
    refs = read(ROOT / 'research/retrieval_answer_probe/v1/references.private.json')
    STUDY.mkdir(parents=True)
    for name, value in (('cases.json', cases), ('prompts.json', prompts), ('unit_baseline.json', units), ('references.private.json', refs)):
        save(STUDY / name, value)
    files = ('research/LITERAL_RESTORATION_PLAN.md', 'reader_memory/literal_evidence.py',
        'reader_memory/literal_restoration.py', 'tools/audit_literal_restoration.py', 'tools/run_literal_restoration.py',
        'reader_memory/backend.py', 'reader_memory/native_choice_pilot.py', 'reader_memory/schema.py',
        'reader_memory/tokenization.py', 'reader_memory/personamem2.py', 'reader_memory/independent_pack.py',
        'reader_memory/four_reader_probe.py', 'reader_memory/shared_retrieval_audit.py', 'reader_memory/retrieval_answer_probe.py',
        'reader_memory/representation_contract_v2.py', 'reader_memory/representation_development.py',
        'research/four_reader_probe/v1/manifest.json', 'research/shared_retrieval_audit/v1/manifest.json',
        'research/retrieval_answer_probe/v1/manifest.json', 'research/representation_development/v2/manifest.json',
        'runs/representation_development_v2/writer.jsonl', 'runs/representation_development_v2/reviewer.jsonl',
        'data/personamem_v2/source_manifest.json')
    m = {'version': 'literal-restoration-v1', 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'file_hashes': {p: sha256(ROOT / p) for p in files},
        'study_files': {p: sha256(STUDY / p) for p in ('cases.json', 'prompts.json', 'unit_baseline.json', 'references.private.json')},
        'reader_assets': parent['reader_assets'], 'readers': list(READERS), 'arms': list(ARMS), 'execution_policy': POLICY,
        'context_limit': LIMIT, 'intended_questions': 64, 'eligible_questions': len(eligible),
        'source_groups': len({q['source_group'] for q in eligible}), 'unit_baseline_units': 64,
        'expected_records': len(eligible) * len(READERS) * len(ARMS) * 4,
        'oracle_annotations_used_for_restoration': True, 'normal_retrieval_benchmark': False,
        'independent_validation': False, 'confirmation_sources_read': False, 'reserved_validation_used': False,
        'selector_trained': False, 'reader_capability_hypothesis_tested': False}
    save(STUDY / 'manifest.json', m); OUTPUT.mkdir(parents=True, exist_ok=True)
    print('LITERAL_FROZEN', len(eligible), m['expected_records'], flush=True)


def load():
    m = read(STUDY / 'manifest.json')
    if (any(sha256(ROOT / p) != h for p, h in m['file_hashes'].items())
            or any(sha256(STUDY / p) != h for p, h in m['study_files'].items()) or m['execution_policy'] != POLICY):
        raise ValueError('Frozen literal restoration changed')
    load_answer_parent(); writing_v2.load()
    return m, read(STUDY / 'cases.json'), read(STUDY / 'prompts.json')


def key(row):
    return row['reader'], row['question_id'], row['arm'], row['shift']


def expected(cases):
    return {(r, q['question_id'], a, s) for q in cases if q['eligible'] for r in READERS for a in ARMS for s in q['shifts']}


def verify(rows, m, cases, prompts):
    valid = expected(cases); byid = {q['question_id']: q for q in cases}; seen = {}
    for row in rows:
        k = key(row)
        if k not in valid or k in seen:
            raise ValueError('Duplicate/unknown literal answer')
        r, uid, arm, shift = k; q = byid[uid]; item = q['evidence'][arm]; entry = prompts[r][uid][f'{arm}:{shift}']
        scores = row['restricted_label_logits']; usage = row['usage']
        if (row['study_signature'] != fingerprint(m) or row['prompt_entry'] != entry
                or row['evidence_hash'] != fingerprint(item['text']) or row['source_ledger_hash'] != item['source_ledger_hash']
                or row['evidence_tokens'] != item['tokens_by_reader'][r] or usage['prompt_tokens'] != entry['tokens']
                or not 0 < usage['prompt_tokens'] <= LIMIT or not math.isfinite(usage['elapsed_seconds'])
                or usage['elapsed_seconds'] < 0 or len(scores) != 4 or not all(math.isfinite(v) for v in scores)
                or row['prediction'] != 'abcd'[max(range(4), key=lambda i: scores[i])]):
            raise ValueError('Literal answer provenance/cost changed')
        origin = row['prediction_origin']
        if origin == 'cache-identical-prompt':
            previous = seen.get(tuple(row['cached_from_key']))
            if (previous is None or previous['reader'] != r or previous['prompt_entry'] != entry
                    or previous['restricted_label_logits'] != scores or usage['elapsed_seconds'] != 0):
                raise ValueError('Invalid exact prompt cache')
        elif origin != 'model-forward' or row['cached_from_key'] is not None:
            raise ValueError('Unknown prediction origin')
        seen[k] = row
    return set(seen)


def run(backend, counter):
    m, cases, prompts = load(); reader = backend.model_name
    if reader not in READERS or not counter.exact or probe_assets(backend.path) != m['reader_assets'][reader]:
        raise ValueError('Wrong frozen reader')
    path = OUTPUT / 'answers.jsonl'; rows = list(read_jsonl(path)) if path.exists() else []
    done = verify(rows, m, cases, prompts)
    cache = {(r['reader'], r['prompt_entry']['messages_hash'], r['prompt_entry']['token_ids_hash']): r for r in rows}
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        for q in cases:
            if not q['eligible']: continue
            uid = q['question_id']
            for arm in ARMS:
                item = q['evidence'][arm]
                if counter.count(item['text']) != item['tokens_by_reader'][reader]:
                    raise ValueError('Actual evidence cost changed')
                for shift in q['shifts']:
                    k = reader, uid, arm, shift
                    if k in done: continue
                    entry = prompt_entry(counter, q, arm, shift)
                    if entry != prompts[reader][uid][f'{arm}:{shift}']:
                        raise ValueError('Actual prompt IDs changed')
                    ck = reader, entry['messages_hash'], entry['token_ids_hash']; previous = cache.get(ck)
                    result = ({'prediction': previous['prediction'], 'restricted_label_logits': previous['restricted_label_logits'],
                        'usage': {**previous['usage'], 'elapsed_seconds': 0.0}} if previous else classify(backend, messages(q, arm, shift)))
                    row = {'reader': reader, 'question_id': uid, 'arm': arm, 'shift': shift,
                        'study_signature': fingerprint(m), 'prompt_entry': entry, 'evidence_hash': fingerprint(item['text']),
                        'source_ledger_hash': item['source_ledger_hash'], 'evidence_tokens': item['tokens_by_reader'][reader],
                        'prediction_origin': 'cache-identical-prompt' if previous else 'model-forward',
                        'cached_from_key': list(key(previous)) if previous else None, **result}
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
                    done.add(k); cache[ck] = row
                    if len(done) % 64 == 0:
                        save(OUTPUT / 'status.json', {'reader': reader, 'records': len(done), 'expected_records': m['expected_records']})
                        print('LITERAL_PROGRESS', reader, len(done), '/', m['expected_records'], flush=True)


def report():
    import numpy as np
    m, cases, prompts = load(); rows = list(read_jsonl(OUTPUT / 'answers.jsonl'))
    if verify(rows, m, cases, prompts) != expected(cases):
        raise ValueError('Complete literal answer matrix required')
    eligible = [q for q in cases if q['eligible']]; qmap = {q['question_id']: q for q in eligible}
    refs = {r['question_id']: r for r in read(STUDY / 'references.private.json')}
    scored = [{**r, 'source_group': qmap[r['question_id']]['source_group'],
        'correct': score_row(r, qmap[r['question_id']], refs[r['question_id']]),
        'semantic_choice': (r['shift'] + 'abcd'.index(r['prediction'])) % 4} for r in rows]
    index = {key(r): r for r in scored}; summaries = {}; transitions = []
    for reader in READERS:
        summaries[reader] = {}
        for arm in ARMS:
            sub = [r for r in scored if r['reader'] == reader and r['arm'] == arm]
            byq = [[index[(reader, q['question_id'], arm, s)] for s in q['shifts']] for q in eligible]
            summaries[reader][arm] = {'presentations': len(sub), 'correct': sum(r['correct'] for r in sub),
                'accuracy': sum(r['correct'] for r in sub) / len(sub),
                'mean_evidence_tokens': sum(r['evidence_tokens'] for r in sub) / len(sub),
                'mean_prompt_tokens': sum(r['usage']['prompt_tokens'] for r in sub) / len(sub),
                'all_four_orders_correct_questions': sum(all(r['correct'] for r in group) for group in byq),
                'all_four_orders_wrong_questions': sum(not any(r['correct'] for r in group) for group in byq),
                'semantic_choice_varies_across_orders': sum(len({r['semantic_choice'] for r in group}) > 1 for group in byq)}
        for q in eligible:
            uid = q['question_id']
            outcomes = {a: [index[(reader, uid, a, s)]['correct'] for s in q['shifts']] for a in ARMS}
            transitions.append({'reader': reader, 'question_id': uid, 'source_group': q['source_group'],
                'correct_by_order': outcomes, 'added_source_characters': q['restoration']['added_source_characters'],
                'interpretation_is_descriptive_not_internal_causal_diagnosis': True})
    groups = sorted({q['source_group'] for q in eligible})
    counts = np.array([sum(q['source_group'] == g for q in eligible) for g in groups])
    draws = np.random.default_rng(20261026).integers(0, len(groups), size=(20000, len(groups)))
    def interval(diffs):
        nums = np.array([sum(diffs[q['question_id']] for q in eligible if q['source_group'] == g) for g in groups])
        samples = nums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        return {'difference_pp': 100 * sum(diffs.values()) / len(eligible),
                'exploratory_cluster_95_interval_pp': (100 * np.quantile(samples, [0.025, 0.975])).tolist()}
    deltas = {}; contrasts = []
    for reader in READERS:
        for candidate, baseline in CONTRASTS:
            d = {q['question_id']: sum(index[(reader, q['question_id'], candidate, s)]['correct'] -
                index[(reader, q['question_id'], baseline, s)]['correct'] for s in q['shifts']) / 4 for q in eligible}
            deltas[(reader, candidate, baseline)] = d
            contrasts.append({'reader': reader, 'candidate': candidate, 'baseline': baseline,
                'primary_same_information_format_contrast': candidate == 'structured', **interval(d)})
    interactions = []
    for a, b in combinations(READERS, 2):
        for candidate, baseline in CONTRASTS:
            d = {uid: value - deltas[(b, candidate, baseline)][uid] for uid, value in deltas[(a, candidate, baseline)].items()}
            interactions.append({'reader_a': a, 'reader_b': b, 'candidate': candidate, 'baseline': baseline, **interval(d)})
    value = {'status': 'completed-literal-restoration-development-diagnostic',
        'intended_questions': 64, 'eligible_questions': len(eligible), 'source_groups': len(groups),
        'excluded_questions': [{'question_id': q['question_id'], 'reason': q['exclusion_reason']} for q in cases if not q['eligible']],
        'records': len(rows), 'actual_model_forward_calls': sum(r['prediction_origin'] == 'model-forward' for r in rows),
        'identical_prompt_cache_reuses': sum(r['prediction_origin'] != 'model-forward' for r in rows),
        'forward_seconds_shared_gpu': sum(r['usage']['elapsed_seconds'] for r in rows),
        'actual_forward_prompt_tokens': sum(r['usage']['prompt_tokens'] for r in rows if r['prediction_origin'] == 'model-forward'),
        'unit_baseline_units': 64, 'unit_baseline_reversible_views': 128, 'by_reader': summaries,
        'contrasts': contrasts, 'reader_difference_in_differences': interactions,
        'official_source_already_fully_in_retrieved_questions': sum(q['restoration']['official_source_fully_present_in_retrieved'] for q in eligible),
        'oracle_diagnostic': True, 'normal_retrieval_benchmark': False, 'independent_validation': False,
        'semantic_sufficiency_established': False, 'reader_internal_failure_identified': False,
        'reader_capability_hypothesis_tested': False, 'selector_trained': False,
        'confirmation_sources_read': False, 'reserved_validation_used': False,
        'study_manifest_sha256': sha256(STUDY / 'manifest.json'), 'answers_sha256': sha256(OUTPUT / 'answers.jsonl')}
    save(OUTPUT / 'report.json', value); save(OUTPUT / 'paired_interventions.json', transitions)
    (OUTPUT / 'scored.jsonl').write_bytes(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in scored).encode('utf-8'))
    print('LITERAL_REPORT', len(rows), value['actual_model_forward_calls'], flush=True)
    return value


def render_report():
    load(); r = read(OUTPUT / 'report.json'); a = read(OUTPUT / 'input_integrity.json')
    if a['status'] != 'passed' or a['answers_sha256'] != r['answers_sha256'] or r['answers_sha256'] != sha256(OUTPUT / 'answers.jsonl'):
        raise ValueError('Audited outputs required')
    lines = ['# 逐字证据表示与来源补回诊断 v1', '',
        f"16 来源的 64 个单元均完成两种视图反解核对，共 128 份；没有生成新的语义 claim。任务诊断另用已曝光的 32 来源、64 题，"
        f"实际合格 {r['eligible_questions']} 题、{r['source_groups']} 来源，排除 {len(r['excluded_questions'])} 题。不是独立验证。", '',
        '**restored 和 oracle 使用官方相关片段注释，只是 oracle 诊断。新增内容、长度与上下文共同改变，不能称为等预算正常检索成绩。**', '',
        'retrieved 继承旧密集 2K 选择的字符范围但重新按原历史顺序序列化；restored 补回标注对应的完整实际来源回合；'
        'structured 与 restored 的来源账本逐字符相同，仅采用 JSON 视图；oracle 只含标注来源。', '',
        '| 读者 | retrieved | restored | structured | oracle |', '| --- | ---: | ---: | ---: | ---: |']
    for reader, values in r['by_reader'].items():
        lines.append('| ' + reader + ' | ' + ' | '.join(f"{values[arm]['correct']}/{values[arm]['presentations']}" for arm in ARMS) + ' |')
    lines += ['', '每题四个循环位置，不是独立题或全部 24 排列。正确选项采用原官方身份，没有 AI judge。', '',
        '| 读者 | 比较 | 差值（百分点） | 来源聚类探索性 95% 区间 |', '| --- | --- | ---: | --- |']
    for c in r['contrasts']:
        lo, hi = c['exploratory_cluster_95_interval_pp']
        lines.append(f"| {c['reader']} | {c['candidate']}−{c['baseline']} | {c['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', '主比较只有同信息 structured−restored 的四读者差异，其余是辅助诊断。全部为已曝光开发结果，区间未作多重校正，不能作确认性显著性判断。', '',
        '| 读者 | retrieved 平均 token | restored | structured | oracle |', '| --- | ---: | ---: | ---: | ---: |']
    for reader, values in r['by_reader'].items():
        lines.append('| ' + reader + ' | ' + ' | '.join(f"{values[arm]['mean_evidence_tokens']:.1f}" for arm in ARMS) + ' |')
    lines += ['', f"共 {r['records']} 条记录，真实前向 {r['actual_model_forward_calls']} 次，完全相同提示缓存复用 "
        f"{r['identical_prompt_cache_reuses']} 次。真实前向输入 {r['actual_forward_prompt_tokens']} token，累计 "
        f"{r['forward_seconds_shared_gpu']:.1f} 秒。模型加载、CPU 预检、审计和等待另计，阶段资源见 runs 中各 reader_resources.json。", '',
        '逐字可逆性只保证所选字符、角色和元数据没有被生成式改写，不保证标注片段足以回答或记忆语义完整。'
        '补回仍答错不能直接归因于内部能力；位置、干扰、题目及标注问题仍可能影响。'
        '读者间差之差不是独立能力向量有效性的证据，未训练选择器或预算算法。', '',
        f"独立审计核对 {a['complete_prompt_token_id_checks']} 份实际完整提示、{a['answer_rows_checked']} 条答案及官方评分。"
        '32 来源的区间不代表未见来源泛化。48 个确认来源正文、159 保留组、旧四个验证 persona 与 LME 440 题继续封存。', '',
        '排除原因、全位置稳定性、来源补回逐题转移和读者差之差分别保存在 `../runs/literal_restoration_v1/` 的 '
        'report.json、paired_interventions.json 与 scored.jsonl；冻结协议见 [LITERAL_RESTORATION_PLAN.md](LITERAL_RESTORATION_PLAN.md)。', '']
    (ROOT / 'research/LITERAL_RESTORATION_REPORT.md').write_bytes('\n'.join(lines).encode('utf-8'))
    print('LITERAL_RENDERED', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run', 'report', 'render'))
    p.add_argument('--reader', choices=READERS)
    args = p.parse_args()
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    if args.command == 'prepare': return prepare({r: TokenCounter(ROOT / 'models' / r) for r in READERS})
    if args.command == 'report': return report()
    if args.command == 'render': return render_report()
    if args.reader is None: raise ValueError('Reader required')
    from .backend import TransformersBackend
    run(TransformersBackend(ROOT / 'models' / args.reader), TokenCounter(ROOT / 'models' / args.reader))


if __name__ == '__main__':
    main()
