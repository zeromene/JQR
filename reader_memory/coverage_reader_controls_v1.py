"""Fixed old-dev coverage controls and annotated-support reader diagnostics."""
import argparse
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import random
import statistics
import time

from . import baseline_bootstrap_v2 as base
from . import evidence_acquisition_v1 as evidence
from . import replacement_precision_v2 as ledger
from . import replacement_ranker_v1 as ranker
from . import replacement_stage2_common as common
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/coverage_reader_controls_v1'
RUN = ROOT / 'runs/coverage_reader_controls_v1'
ARMS = ('static_fusion', 'semantic_mixed_batch4', 'raw14_batch4', 'raw20_batch4', 'support_only')
CONFIG = dict(version='coverage-reader-controls-v1', sources=16, questions=32,
    arms=list(ARMS), evidence_budget=2048, output_tokens=2048, ranker_batch=4,
    ranker='Qwen3-Reranker-4B', reader='Qwen3.5-9B',
    ranker_scoring='unadapted BF16, official yes/no softmax, original semantic instruction',
    raw_comparison='score all20 once in native paragraph order; raw14 uses identical shared scores and tie order',
    mixed_comparison='original raw14/extract28 pool and original batch4 semantics; no weaker serial substitute',
    support_order='original paragraph order; membership only from official support annotations',
    support_overflow='stop preparation before freeze; never silently truncate supporting text',
    decoding='greedy-thinking_false', training=False, retries=0,
    max_conditions=160, max_new_reader_calls=128, bootstrap_samples=2000, bootstrap_seed=29,
    resource_cues=dict(support_multi_f1_min=70, support_gap_over_raw20_min=5,
        raw20_multi_gain_over_raw14_min=3, raw20_multi_winning_sources_min=2),
    scope='previously exposed original16 internal-dev sources; feasibility diagnosis, not independent confirmation',
    protected_final_data_accessed=False)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable(path, value):
    common.immutable_json(path, value)


def request_key(messages, marker):
    return fingerprint(dict(messages=messages, max_tokens=2048, reader='Qwen3.5-9B',
        decoding='greedy-thinking_false', reader_marker_sha256=marker))


def public_source(native, question, mode, counter):
    # Annotation fields never affect this function's records, membership, or packing.
    from tools.prepare_replacement_learning_v1 import records_for
    from . import replacement_core_v1 as core
    records, _ = records_for(native)
    lexical = core.lexical_scores(question, records, list(records))
    ordered = sorted(records, key=lambda k: (-lexical[k], k))
    raw = [k for k in ordered if records[k]['kind'] == 'raw'][:14]
    compact = [k for k in ordered if records[k]['kind'] != 'raw'][:28]
    import itertools
    mixed = [k for pair in itertools.zip_longest(raw, compact) for k in pair if k]
    all_raw = ['raw:' + str(p['idx']) for p in native['paragraphs']]
    if len(all_raw) != 20 or len(set(all_raw)) != 20:
        raise ValueError('Expected the original20 complete paragraphs')
    return dict(question_id=native['id'] + ':' + mode, native_id=native['id'], mode=mode,
        question=question, records={k: records[k] for k in dict.fromkeys(all_raw + compact)},
        mixed_pool_ids=mixed, all_raw_ids=all_raw,
        raw14_ids=[k for k in all_raw if k in raw],
        seed_ids=evidence.ordered_pack(mixed, records, counter, 2048))


def make_plan(source, arm, selected, counter, diagnostic=False):
    text = evidence.render(selected, source['records'])
    tokens = counter.count(text)
    if not text.strip() or tokens > 2048:
        raise ValueError('Empty or over-budget evidence; no truncation')
    return dict(**{k: source[k] for k in ('question_id', 'native_id', 'mode', 'question')},
        arm=arm, selected_ids=list(selected), evidence_text=text, evidence_sha256=fingerprint(text),
        evidence_tokens=tokens, diagnostic_only=diagnostic, private_inputs_read=diagnostic)


def raw_order(source, scores, subset):
    if set(scores) != set(source['all_raw_ids']) or not set(subset) <= set(scores):
        raise ValueError('Raw controls must share the complete20 score map')
    position = {k: i for i, k in enumerate(source['all_raw_ids'])}
    return sorted(subset, key=lambda k: (-scores[k], position[k]))


def freeze():
    paths = ['reader_memory/coverage_reader_controls_v1.py',
        'tools/prepare_coverage_reader_controls_v1.py', 'scripts/coverage_reader_controls_v1_queue.py',
        'tests/test_coverage_reader_controls_v1.py',
        'reader_memory/replacement_precision_v2.py', 'reader_memory/replacement_ranker_v1.py',
        'reader_memory/replacement_stage2_common.py', 'reader_memory/replacement_core_v1.py',
        'reader_memory/replacement_learning_v1.py', 'reader_memory/mainline_reranker_v1.py',
        'reader_memory/mainline_service_v1.py', 'reader_memory/backend.py',
        'reader_memory/evidence_acquisition_v1.py', 'reader_memory/evidence_acquisition_v2.py',
        'reader_memory/baseline_bootstrap_v2.py', 'reader_memory/schema.py',
        'reader_memory/_vendor/locomo_native_f1.py', 'tools/prepare_replacement_learning_v1.py',
        'third_party/simplemem_official/simplemem/core/answer_generator.py',
        'third_party/simplemem_official/simplemem/core/utils/llm_client.py']
    paths += ['research/coverage_reader_controls_v1/' + n for n in
        ['PROTOCOL.md', 'public/source_inputs.json', 'public/static_plans.json',
         'private/support_plans.json', 'private/references.json', 'private/replay_cache.json',
         'preparation.json', 'model_references.json', 'verification_markers.json']]
    value = dict(config=CONFIG, files={n: base.sha(ROOT / n) for n in paths})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_COVERAGE_READER_CONTROLS', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Scientific identity changed')
    for name, digest in value['files'].items():
        if base.sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    # Compare metadata content, not a platform-dependent JSON serialization.
    for name, reference in load(PLAN / 'model_references.json').items():
        if load(ROOT / name) != reference:
            raise ValueError('Pinned model metadata changed: ' + name)
    return value['signature']


def check_cache_entry(item, marker):
    if request_key(item['messages'], marker) != item['key'] or item['max_tokens'] != 2048:
        raise ValueError('Historical request identity mismatch')
    entry = item['entry']
    if entry['status'] not in ('ok', 'failed') or not entry['result']:
        raise ValueError('Incomplete historical transaction')
    if entry['status'] == 'ok' and ledger.parse_answer(base.require_complete(entry['result'])) != entry['hypothesis']:
        raise ValueError('Historical answer/parser mismatch')
    if entry['status'] != 'ok' and entry['hypothesis']:
        raise ValueError('Failed answer must stay empty')


def replay_cache():
    marker = load(PLAN / 'verification_markers.json')['Qwen3.5-9B']
    cache = {}
    for item in load(PLAN / 'private/replay_cache.json'):
        check_cache_entry(item, marker)
        if item['key'] in cache:
            raise ValueError('Duplicate frozen cache key')
        cache[item['key']] = item['entry']
    return cache


def preflight():
    sid = verify()
    from tokenizers import Tokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter = ExactCounter()
    tokenizer = Tokenizer.from_file(str(ROOT / 'models/Qwen3-Reranker-4B/tokenizer.json'))
    class Adapter:
        def encode(self, text, add_special_tokens=False):
            return tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
    sources = load(PLAN / 'public/source_inputs.json')
    legal = set(load(ROOT / 'research/replacement_learning_v1/private/source_selection.json')['dev'])
    if len(sources) != 32 or {s['native_id'] for s in sources} != legal:
        raise ValueError('Only original16 exposed dev sources are allowed')
    if {s['question_id'] for s in sources} != {n + ':' + mode for n in legal for mode in ('multi', 'single')}:
        raise ValueError('Question coverage changed')
    maximum = 0
    for source in sources:
        if set(source) != {'question_id','native_id','mode','question','records','mixed_pool_ids','all_raw_ids','raw14_ids','seed_ids'}:
            raise ValueError('Private fields in retrieval inputs')
        if len(source['all_raw_ids']) != 20 or len(source['raw14_ids']) != 14:
            raise ValueError('Raw candidate coverage changed')
        for k in source['records']:
            maximum = max(maximum, len(common.semantic_ids(Adapter(), source['question'], evidence.render([k], source['records']))))
    plans = load(PLAN / 'public/static_plans.json') + load(PLAN / 'private/support_plans.json')
    cache = replay_cache()
    marker = load(PLAN / 'verification_markers.json')['Qwen3.5-9B']
    for row in plans:
        if row['evidence_sha256'] != fingerprint(row['evidence_text']) or counter.count(row['evidence_text']) != row['evidence_tokens'] or row['evidence_tokens'] > 2048:
            raise ValueError('Evidence identity or budget mismatch')
        if row['arm'] == 'static_fusion' and request_key(ledger.messages_for(row), marker) not in cache:
            raise ValueError('Original static answer must be reused, including failures')
    for model, digest in load(PLAN / 'verification_markers.json').items():
        if base.sha(ROOT / 'models' / model / '.verified.json') != digest:
            raise ValueError('Offline verified model marker changed: ' + model)
    for name, digest in load(PLAN / 'preparation.json')['source_files'].items():
        if base.sha(ROOT / name) != digest:
            raise ValueError('Original source artifact changed: ' + name)
    for name, receipt in load(PLAN / 'preparation.json')['source_serialization_equivalence'].items():
        if base.sha(ROOT / name) != receipt['server_sha256'] or fingerprint(load(ROOT / name)) != receipt['canonical_json_sha256']:
            raise ValueError('Previously verified source content differs: ' + name)
    immutable(RUN / 'preflight.json', dict(signature=sid, passed=True, questions=32,
        max_ranker_tokens=maximum, support_max_tokens=max(p['evidence_tokens'] for p in plans if p['diagnostic_only']),
        protected_final_data_accessed=False, new_training=False))
    print('COVERAGE_READER_PREFLIGHT_OK', maximum, flush=True)


def semantic_scores(torch, tokenizer, model, source, pool):
    yes, no = tokenizer.convert_tokens_to_ids('yes'), tokenizer.convert_tokens_to_ids('no')
    scores, lengths = {}, []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(pool), 4):
            keys = pool[offset:offset + 4]
            encoded = [common.semantic_ids(tokenizer, source['question'], evidence.render([k], source['records'])) for k in keys]
            lengths.extend(map(len, encoded))
            batch = tokenizer.pad(dict(input_ids=encoded), padding=True, return_tensors='pt').to('cuda')
            logits = model(**batch, logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
            two = torch.stack([logits[:, no], logits[:, yes]], dim=1)
            values = torch.nn.functional.log_softmax(two, dim=1)[:, 1].exp().tolist()
            scores.update(zip(keys, values, strict=True))
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError('Nonfinite semantic score')
    return scores, dict(forward_calls=math.ceil(len(pool)/4), documents=len(pool),
        input_tokens=sum(lengths), seconds=time.perf_counter() - started)


def select():
    sid = verify()
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter = ExactCounter()
    torch, tokenizer, model, _ = ranker.load(ROOT / 'models/Qwen3-Reranker-4B')
    # No trained checkpoint is loaded. Added LoRA branches are identically zero.
    if any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if n.endswith('.b')):
        raise ValueError('Expected unadapted semantic ranker')
    sources = load(PLAN / 'public/source_inputs.json')
    for i, source in enumerate(sources):
        qid = source['question_id']
        dest = RUN / 'selected' / (qid.replace(':', '_') + '.json')
        if dest.exists():
            saved = load(dest)
            if saved['signature'] != sid or saved['source_sha256'] != fingerprint(source):
                raise ValueError('Committed selection changed')
            continue
        intent = RUN / 'selection_inflight.json'
        if intent.exists():
            raise ValueError('Uncommitted selection; no silent retry')
        write_json(intent, dict(signature=sid, question_id=qid))
        mixed, mixed_cost = semantic_scores(torch, tokenizer, model, source, source['mixed_pool_ids'])
        raw, raw_cost = semantic_scores(torch, tokenizer, model, source, source['all_raw_ids'])
        pos = {k: i for i, k in enumerate(source['mixed_pool_ids'])}
        mixed_order = sorted(mixed, key=lambda k: (-mixed[k], pos[k]))
        orders = dict(semantic_mixed_batch4=mixed_order,
            raw14_batch4=raw_order(source, raw, source['raw14_ids']),
            raw20_batch4=raw_order(source, raw, source['all_raw_ids']))
        plans = [make_plan(source, a, evidence.ordered_pack(order, source['records'], counter, 2048), counter)
                 for a, order in orders.items()]
        immutable(dest, dict(signature=sid, source_sha256=fingerprint(source), plans=plans,
            mixed_scores=mixed, shared_raw_scores=raw, mixed_cost=mixed_cost, raw20_cost=raw_cost,
            raw14_standalone_forward_calls=math.ceil(14/4), raw14_shared_scores=True,
            private_inputs_read=False))
        intent.unlink()
        write_json(RUN / 'selection_status.json', dict(completed=i+1, expected=32))
        print('COVERAGE_SELECT', i+1, '/32', qid, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def assemble():
    sid = verify()
    sources = load(PLAN / 'public/source_inputs.json')
    statics = {p['question_id']: p for p in load(PLAN / 'public/static_plans.json')}
    support = {p['question_id']: p for p in load(PLAN / 'private/support_plans.json')}
    plans = []
    for s in sources:
        q = s['question_id']
        saved = load(RUN / 'selected' / (q.replace(':', '_') + '.json'))
        if saved['signature'] != sid or saved['source_sha256'] != fingerprint(s):
            raise ValueError('Selection identity changed')
        plans.extend([statics[q], *saved['plans'], support[q]])
    expected = {(s['question_id'], a) for s in sources for a in ARMS}
    if len(plans) != 160 or {(p['question_id'], p['arm']) for p in plans} != expected:
        raise ValueError('Incomplete fixed comparison')
    cache = replay_cache()
    marker = load(PLAN / 'verification_markers.json')['Qwen3.5-9B']
    unique = {request_key(ledger.messages_for(p), marker) for p in plans}
    uncached = unique - set(cache)
    if len(uncached) > 128:
        raise ValueError('Reader call budget exceeded')
    immutable(RUN / 'qa_plans.json', plans)
    immutable(RUN / 'selection_complete.json', dict(signature=sid, plans_sha256=base.sha(RUN / 'qa_plans.json')))
    immutable(RUN / 'selection_provenance.json', dict(signature=sid, conditions=160,
        unique_requests=len(unique), uncached_requests=len(uncached), support_is_diagnostic_only=True,
        original_internal_dev_only=True, retraining=False))


def qa(endpoint):
    sid = verify()
    if load(RUN / 'selection_complete.json')['plans_sha256'] != base.sha(RUN / 'qa_plans.json'):
        raise ValueError('Committed QA plans changed')
    old = ledger.RUN, ledger.verify, ledger.saved_cache
    ledger.RUN, ledger.verify, ledger.saved_cache = RUN, lambda: sid, replay_cache
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN, ledger.verify, ledger.saved_cache = old


def score_rows(plans, predictions, refs, signature):
    expected = {(r['question_id'], r['arm']): r for r in plans}
    if len(predictions) != len(plans) or {(r['question_id'], r['arm']) for r in predictions} != set(expected):
        raise ValueError('Complete unique QA required')
    out = []
    for row in predictions:
        p = expected[row['question_id'], row['arm']]
        if row['signature'] != signature or row['plan_sha256'] != fingerprint(p) or any(row[k] != p[k] for k in p):
            raise ValueError('Committed QA identity changed')
        ref = refs[row['question_id']]
        f1 = max(base.native_f1(row['hypothesis'], a, 2) for a in [ref['answer']] + ref['aliases']) if row['status'] == 'ok' else 0.
        out.append(dict(**row, local_native_f1=f1))
    return out


def score():
    sid = verify()
    refs = load(PLAN / 'private/references.json')
    plans = load(RUN / 'qa_plans.json')
    scored = score_rows(plans, list(read_jsonl(RUN / 'qa/predictions.jsonl')), refs, sid)
    by = {a: {r['question_id']: r for r in scored if r['arm'] == a} for a in ARMS}
    summaries = []
    for a in ARMS:
        for mode in ('overall', 'multi', 'single'):
            group = [r for r in by[a].values() if mode == 'overall' or r['mode'] == mode]
            support_cover = []
            for r in group:
                seen = {int(k.split(':')[1]) for k in r['selected_ids'] if k.startswith('raw:')}
                support_cover.append(set(refs[r['question_id']]['support_indices']) <= seen)
            summaries.append(dict(arm=a, scope=mode, n=len(group), diagnostic_only=a=='support_only',
                f1_points=100*statistics.mean(r['local_native_f1'] for r in group),
                mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in group),
                full_support_raw_count=sum(support_cover), statuses=dict(Counter(r['status'] for r in group))))
    comparisons = []
    for a, b in [('raw20_batch4','raw14_batch4'), ('raw20_batch4','semantic_mixed_batch4'),
                 ('raw20_batch4','static_fusion'), ('support_only','raw20_batch4'),
                 ('support_only','semantic_mixed_batch4')]:
        ds, normals, sources = [], [], defaultdict(list)
        for q, row in by[a].items():
            other = by[b][q]
            delta = row['local_native_f1'] - other['local_native_f1']
            ds.append(delta)
            sources[row['native_id']].append(delta)
            if row['status'] == other['status'] == 'ok':
                normals.append(delta)
        rng = random.Random(29)
        keys = sorted(sources)
        boots = sorted(100*statistics.mean(v for _ in keys for v in sources[rng.choice(keys)]) for _ in range(2000))
        comparisons.append(dict(arm=a, comparator=b, diagnostic_only=a=='support_only',
            delta_points=100*statistics.mean(ds), normal_pairs=len(normals),
            normal_pair_delta_points=100*statistics.mean(normals) if normals else None,
            wins=sum(d>1e-9 for d in ds), ties=sum(abs(d)<=1e-9 for d in ds), losses=sum(d < -1e-9 for d in ds),
            winning_sources=sum(statistics.mean(v)>0 for v in sources.values()),
            losing_sources=sum(statistics.mean(v)<0 for v in sources.values()),
            source_bootstrap_95_points=[boots[49], boots[1949]]))
    calls = list(read_jsonl(RUN / 'qa/answer_calls.jsonl')) if (RUN / 'qa/answer_calls.jsonl').exists() else []
    if len(calls) > 128:
        raise ValueError('Actual calls exceeded frozen limit')
    selected = [load(p) for p in (RUN / 'selected').glob('*.json')]
    summary_by = {(r['arm'],r['scope']):r for r in summaries}
    support_multi = summary_by['support_only','multi']['f1_points']
    raw_multi = summary_by['raw20_batch4','multi']['f1_points']
    raw14_multi = summary_by['raw14_batch4','multi']['f1_points']
    multi_wins = sum(r['mode']=='multi' and r['local_native_f1']>by['raw14_batch4'][q]['local_native_f1']
        for q,r in by['raw20_batch4'].items())
    resource_signals = dict(support_multi_f1=support_multi, support_gap_over_raw20=support_multi-raw_multi,
        raw20_multi_gain_over_raw14=raw_multi-raw14_multi, raw20_multi_winning_sources=multi_wins,
        support_reader_space_signal=support_multi>=70 and support_multi-raw_multi>=5,
        candidate_coverage_signal=raw_multi-raw14_multi>=3 and multi_wins>=2,
        independent_confirmation=False, automatic_launch=False)
    value = dict(signature=sid, complete=True, scores=summaries, comparisons=comparisons,
        resource_signals=resource_signals,
        new_reader_calls=len(calls), reused_conditions=sum(r['reused'] for r in scored),
        reader_cost={k:sum(c.get('result',{}).get('usage',{}).get(k,0) for c in calls) for k in ('prompt_tokens','completion_tokens')},
        semantic_costs={kind:{k:sum(s[kind][k] for s in selected) for k in ('forward_calls','input_tokens','seconds')}
            for kind in ('mixed_cost','raw20_cost')},
        raw14_scoring_reused_from_raw20=True, standalone_raw14_cost_not_zero=True,
        scope=CONFIG['scope'], protected_final_data_accessed=False, training=False, paper_ready=False,
        support_is_diagnostic_not_a_method_or_strict_upper_bound=True,
        automatic_followup=False, next_action='Review coverage/reader diagnosis; no automatic training or relation-search launch')
    write_json(RUN / 'scored.json', scored)
    write_json(RUN / 'report.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze','verify','preflight','select','assemble','qa','score'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8019')
    args = parser.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    else:
        result = globals()[args.mode]()
        if result:
            print(result, flush=True)
