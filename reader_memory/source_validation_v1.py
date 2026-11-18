"""One fixed-model validation on new, fact-disjoint MuSiQue train sources."""
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
from . import clean_feedback_control_v1 as clean
from . import evidence_acquisition_v1 as evidence
from . import mainline_reranker_v1 as semantic
from . import replacement_core_v1 as core
from . import replacement_precision_v2 as precision
from . import replacement_ranker_v1 as ranker
from . import replacement_stage2_common as common
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/source_validation_v1'
RUN = ROOT / 'runs/source_validation_v1'
ARMS = ('static_fusion', 'semantic_mixed', 'clean_full_union', 'clean_wide_single')
CONFIG = dict(version='source-validation-v1', sources=32, questions=64, reserve_sources=32,
    source_hops={'2': 16, '3': 8, '4': 8}, seed='source-validation-20261029-v1',
    arms=list(ARMS), primary='clean_full_union', budget=2048, output_tokens=2048,
    reader='Qwen3.5-9B', reader_decoding='greedy-thinking_false',
    single_cap=24, group_cap=24, ranker_batch=4, set_head='unchanged BF16',
    checkpoint='7e0514d7dbcb5fbdfbd01fe96417eab6e1247cfbafc59233555f5b533e22282c',
    scope='source-disjoint in-domain development validation, extractive memories; not a benchmark-native score',
    training=False, heldout_accessed=False, bootstrap_samples=2000, bootstrap_seed=29)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def public_input(row, question, mode, counter):
    """No answers or support annotations affect candidate construction."""
    from tools.prepare_replacement_learning_v1 import records_for
    records, _ = records_for(row)
    scores = core.lexical_scores(question, records, list(records))
    ordered = sorted(records, key=lambda key: (-scores[key], key))
    raw = [k for k in ordered if records[k]['kind'] == 'raw'][:14]
    compact = [k for k in ordered if records[k]['kind'] != 'raw'][:28]
    import itertools
    pool = [k for pair in itertools.zip_longest(raw, compact) for k in pair if k]
    seed = evidence.ordered_pack(pool, records, counter, CONFIG['budget'])
    return dict(question_id=row['id'] + ':' + mode, native_id=row['id'], mode=mode,
        question=question, records={k: records[k] for k in pool},
        candidate_pool_ids=pool, seed_ids=seed)


def alternatives(source, individual, counter):
    args = (source['question'], source['seed_ids'], source['candidate_pool_ids'],
            source['records'], individual, counter)
    single = core.proposal_sets(*args, False, max_sets=CONFIG['single_cap'])
    group = core.proposal_sets(*args, True, max_sets=CONFIG['group_cap'])
    union = precision.ordered_union(single, group)
    wide = core.proposal_sets(*args, False, max_sets=len(union))
    original = {p['evidence_sha256'] for p in single}
    if not original <= {p['evidence_sha256'] for p in wide}:
        raise ValueError('Wide search lost an original single proposal')
    if not original <= {p['evidence_sha256'] for p in union}:
        raise ValueError('Union lost an original single proposal')
    return dict(clean_full_union=union, clean_wide_single=wide)


def freeze():
    paths = ['reader_memory/source_validation_v1.py', 'scripts/source_validation_v1_queue.py',
        'tools/prepare_source_validation_v1.py', 'tests/test_source_validation_v1.py',
        'tools/prepare_replacement_learning_v1.py', 'reader_memory/replacement_ranker_v1.py',
        'reader_memory/replacement_core_v1.py', 'reader_memory/replacement_precision_v2.py',
        'reader_memory/replacement_stage2_common.py', 'reader_memory/mainline_reranker_v1.py',
        'reader_memory/mainline_service_v1.py', 'reader_memory/evidence_acquisition_v1.py',
        'reader_memory/evidence_acquisition_v2.py', 'reader_memory/baseline_bootstrap_v2.py',
        'reader_memory/backend.py', 'reader_memory/schema.py', 'reader_memory/clean_feedback_control_v1.py',
        'reader_memory/_vendor/locomo_native_f1.py',
        'third_party/simplemem_official/simplemem/core/answer_generator.py']
    paths += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
              and p.suffix in ('.md', '.json') and p.name != 'manifest.json' and 'snapshots' not in p.parts]
    value = dict(config=CONFIG, files={p: base.sha(ROOT / p) for p in sorted(paths)},
        checkpoint_reference=load(PLAN / 'checkpoint_reference.json'))
    value['signature'] = fingerprint(value)
    common.immutable_json(PLAN / 'manifest.json', value)
    print('FROZEN_SOURCE_VALIDATION', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or fingerprint({k: v for k, v in value.items() if k != 'signature'}) != value['signature']:
        raise ValueError('Validation identity changed')
    for name, digest in value['files'].items():
        if base.sha(ROOT / name) != digest:
            raise ValueError('Frozen validation input changed: ' + name)
    return value['signature']


def preflight():
    sid = verify()
    clean.verify()
    complete = load(clean.RUN / 'training_complete.json')
    if complete['checkpoint_sha256'] != CONFIG['checkpoint'] or base.sha(clean.RUN / 'adapter.pt') != CONFIG['checkpoint']:
        raise ValueError('Clean checkpoint changed')
    if complete != load(PLAN / 'checkpoint_reference.json'):
        raise ValueError('Checkpoint provenance changed')
    for model in ('Qwen3-Reranker-4B', 'Qwen3.5-9B'):
        if not (ROOT / 'models' / model / '.verified.json').is_file():
            raise ValueError('Offline model verification marker missing')
    for name, value in load(PLAN / 'model_references.json').items():
        if load(ROOT / name) != value:
            raise ValueError('Reader/ranker model artifact changed: ' + name)
    sources = load(PLAN / 'public/source_inputs.json')
    if len(sources) != CONFIG['questions'] or len({s['native_id'] for s in sources}) != CONFIG['sources']:
        raise ValueError('Validation coverage mismatch')
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter = ExactCounter()
    for source in sources:
        if len(set(source['candidate_pool_ids'])) != len(source['candidate_pool_ids']):
            raise ValueError('Duplicate candidate')
        if set(source['records']) != set(source['candidate_pool_ids']):
            raise ValueError('Candidate records differ from public pool')
        text = evidence.render(source['seed_ids'], source['records'])
        if not text.strip() or counter.count(text) > CONFIG['budget']:
            raise ValueError('Invalid static anchor')
    common.immutable_json(RUN / 'preflight.json', dict(signature=sid, questions=len(sources),
        checkpoint=CONFIG['checkpoint'], source_audit=load(PLAN / 'preparation.json'),
        training=False, heldout_accessed=False))
    print('SOURCE_VALIDATION_PREFLIGHT_OK', len(sources), flush=True)


def semantic_scores(torch, tok, model, source):
    pool, records = source['candidate_pool_ids'], source['records']
    yes, no = tok.convert_tokens_to_ids('yes'), tok.convert_tokens_to_ids('no')
    scores, lengths = {}, []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(pool), CONFIG['ranker_batch']):
            keys = pool[offset:offset + CONFIG['ranker_batch']]
            encoded = [common.semantic_ids(tok, source['question'], evidence.render([k], records)) for k in keys]
            lengths.extend(map(len, encoded))
            batch = tok.pad(dict(input_ids=encoded), padding=True, return_tensors='pt').to('cuda')
            logits = model(**batch, logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
            two = torch.stack([logits[:, no], logits[:, yes]], dim=1)
            values = torch.nn.functional.log_softmax(two, dim=1)[:, 1].exp().tolist()
            scores.update(zip(keys, values, strict=True))
    if len(scores) != len(pool) or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError('Invalid semantic scores')
    return scores, dict(forward_calls=math.ceil(len(pool) / CONFIG['ranker_batch']),
        documents=len(pool), input_tokens=sum(lengths), seconds=time.perf_counter() - started)


def select():
    sid = verify()
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter = ExactCounter()
    torch, tok, model, _ = ranker.load(clean.parent.MODEL)
    if base.sha(clean.RUN / 'adapter.pt') != CONFIG['checkpoint']:
        raise ValueError('Wrong checkpoint')
    state = torch.load(clean.RUN / 'adapter.pt', map_location='cpu', weights_only=True)
    ranker.load_adapter(model, state)
    restores = [(p, state[n].to(p.device)) for n, p in model.named_parameters() if n.endswith('.b')]
    plans = []
    for source in load(PLAN / 'public/source_inputs.json'):
        qid = source['question_id']
        dest = RUN / 'selected' / (qid.replace(':', '_') + '.json')
        if dest.exists():
            saved = load(dest)
            if saved['signature'] != sid or saved['source_sha256'] != fingerprint(source):
                raise ValueError('Committed selection identity changed')
        else:
            intent = RUN / 'selection_inflight.json'
            if intent.exists():
                raise ValueError('Uncommitted selector call; inspect instead of silently repeating')
            write_json(intent, dict(signature=sid, question_id=qid))
            with torch.no_grad():
                for parameter, _ in restores:
                    parameter.zero_()
            individual, semantic_cost = semantic_scores(torch, tok, model, source)
            with torch.no_grad():
                for parameter, value in restores:
                    parameter.copy_(value)
            records, pool = source['records'], source['candidate_pool_ids']
            order = sorted(pool, key=lambda k: (-individual[k], pool.index(k)))
            semantic_ids = evidence.ordered_pack(order, records, counter, CONFIG['budget'])
            candidates = alternatives(source, individual, counter)
            unique = precision.ordered_union(*candidates.values())
            scores, tokens, seconds = {}, {}, {}
            with torch.inference_mode():
                for candidate in unique:
                    key = candidate['evidence_sha256']
                    ids = ranker.input_ids(tok, source['question'], candidate['evidence_text'])
                    started = time.perf_counter()
                    scores[key] = float(ranker.logit(torch, tok, model, ids))
                    seconds[key] = time.perf_counter() - started
                    tokens[key] = len(ids)
            if not all(math.isfinite(v) for v in scores.values()):
                raise ValueError('Nonfinite set score')
            meta = {k: source[k] for k in ('question_id', 'native_id', 'mode', 'question')}
            chosen = []
            for arm, ids in [('static_fusion', source['seed_ids']), ('semantic_mixed', semantic_ids)]:
                text = evidence.render(ids, records)
                chosen.append(dict(**meta, arm=arm, selected_ids=ids, evidence_text=text,
                    evidence_tokens=counter.count(text), evidence_sha256=fingerprint(text),
                    proposal=arm, private_inputs_read=False))
            for arm, options in candidates.items():
                best = max(options, key=lambda p: scores[p['evidence_sha256']])
                chosen.append(dict(**meta, arm=arm, **best,
                    predicted_score=scores[best['evidence_sha256']], private_inputs_read=False))
            saved = dict(signature=sid, source_sha256=fingerprint(source), plans=chosen,
                candidate_pool_ids=pool, individual_scores=individual, scores=scores,
                alternative_counts={a: len(v) for a, v in candidates.items()},
                semantic_cost=semantic_cost,
                set_cost=dict(forward_calls=len(unique), input_tokens=sum(tokens.values()), seconds=sum(seconds.values())),
                independent_set_costs={a: dict(forward_calls=len(v),
                    input_tokens=sum(tokens[p['evidence_sha256']] for p in v),
                    seconds=sum(seconds[p['evidence_sha256']] for p in v)) for a, v in candidates.items()},
                private_inputs_read=False)
            common.immutable_json(dest, saved)
            intent.unlink()
        plans.extend(saved['plans'])
        write_json(RUN / 'selection_status.json', dict(signature=sid,
            completed_questions=len(plans) // len(ARMS), expected_questions=CONFIG['questions']))
        print('SOURCE_VALIDATION_SELECT', len(plans) // len(ARMS), '/', CONFIG['questions'], qid, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    expected = {(s['question_id'], arm) for s in load(PLAN / 'public/source_inputs.json') for arm in ARMS}
    if len(plans) != len(expected) or {(p['question_id'], p['arm']) for p in plans} != expected:
        raise ValueError('Incomplete selection coverage')
    common.immutable_json(RUN / 'qa_plans.json', plans)
    common.immutable_json(RUN / 'selection_complete.json', dict(signature=sid,
        plans_sha256=base.sha(RUN / 'qa_plans.json'), questions=CONFIG['questions'],
        private_inputs_read=False, checkpoint_sha256=CONFIG['checkpoint']))


def qa(endpoint):
    sid = verify()
    plans = load(RUN / 'qa_plans.json')
    if load(RUN / 'selection_complete.json')['plans_sha256'] != base.sha(RUN / 'qa_plans.json'):
        raise ValueError('Selection plans changed')
    common.run_qa(RUN, plans, sid, endpoint,
                  [clean.RUN / 'deployment/qa/answer_calls.jsonl'])


def score_rows(plans, predictions, references):
    expected = {(r['question_id'], r['arm']) for r in plans}
    if len(predictions) != len(expected) or {(r['question_id'], r['arm']) for r in predictions} != expected:
        raise ValueError('Incomplete or duplicate QA coverage')
    scored = []
    for row in predictions:
        ref = references[row['question_id']]
        answers = list(dict.fromkeys([ref['answer']] + ref.get('aliases', [])))
        f1 = max(base.native_f1(row['hypothesis'], answer, 2) for answer in answers) if row['status'] == 'ok' else 0.
        scored.append(dict(**row, local_native_f1=f1))
    return scored


def source_interval(deltas):
    """Bootstrap native sources, keeping their direct/multihop questions together."""
    ids = sorted(deltas)
    rng = random.Random(CONFIG['bootstrap_seed'])
    samples = []
    for _ in range(CONFIG['bootstrap_samples']):
        values = [v for _ in ids for v in deltas[rng.choice(ids)]]
        samples.append(100 * statistics.mean(values))
    samples.sort()
    return [samples[int(.025 * (len(samples) - 1))], samples[int(.975 * (len(samples) - 1))]]


def score():
    sid = verify()
    plans = load(RUN / 'qa_plans.json')
    predictions = list(read_jsonl(RUN / 'qa/predictions.jsonl'))
    if any(p['signature'] != sid for p in predictions):
        raise ValueError('Wrong QA signature')
    references = load(PLAN / 'private/references.json')
    sources = {s['question_id']: s for s in load(PLAN / 'public/source_inputs.json')}
    scored = score_rows(plans, predictions, references)
    for row in scored:
        source = sources[row['question_id']]
        support = {str(i) for i in references[row['question_id']]['paragraph_support_indices']}
        def covers(ids, raw_only=False):
            seen = {str(i) for key in ids for i in source['records'][key]['source_ids']
                    if not raw_only or source['records'][key]['kind'] == 'raw'}
            return support <= seen
        row['support_provenance_proxy'] = covers(row['selected_ids'])
        row['full_raw_support_proxy'] = covers(row['selected_ids'], raw_only=True)
        row['pool_support_provenance_proxy'] = covers(source['candidate_pool_ids'])
        row['pool_full_raw_support_proxy'] = covers(source['candidate_pool_ids'], raw_only=True)
    groups = defaultdict(list)
    for row in scored:
        for scope in ('overall', 'mode:' + row['mode']):
            groups[row['arm'], scope].append(row)
    summaries = [dict(arm=a, scope=s, n=len(v),
        f1_points=100 * statistics.mean(r['local_native_f1'] for r in v),
        status_counts=dict(Counter(r['status'] for r in v)),
        mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in v),
        provenance_proxy_rates={k: statistics.mean(r[k] for r in v) for k in
            ('support_provenance_proxy', 'full_raw_support_proxy',
             'pool_support_provenance_proxy', 'pool_full_raw_support_proxy')})
        for (a, s), v in sorted(groups.items())]
    by = {a: {r['question_id']: r for r in scored if r['arm'] == a} for a in ARMS}
    comparisons = []
    for b in ('static_fusion', 'semantic_mixed', 'clean_wide_single'):
        deltas, valid = defaultdict(list), []
        all_deltas = []
        for q, a in by['clean_full_union'].items():
            other = by[b][q]
            delta = a['local_native_f1'] - other['local_native_f1']
            all_deltas.append(delta)
            deltas[a['native_id']].append(delta)
            if a['status'] == other['status'] == 'ok':
                valid.append(delta)
        comparisons.append(dict(arm='clean_full_union', comparator=b,
            n=len(all_deltas), delta_points=100 * statistics.mean(all_deltas),
            wins=sum(d > 1e-9 for d in all_deltas), ties=sum(abs(d) <= 1e-9 for d in all_deltas),
            losses=sum(d < -1e-9 for d in all_deltas), source_bootstrap_95_points=source_interval(deltas),
            normal_pair_n=len(valid), normal_pair_delta_points=100 * statistics.mean(valid) if valid else None))
    costs = [load(p) for p in (RUN / 'selected').glob('*.json')]
    report = dict(signature=sid, complete=True, primary_arm=CONFIG['primary'],
        scores=summaries, comparisons=comparisons,
        actual_costs={name: {k: sum(c[name][k] for c in costs) for k in ('forward_calls', 'input_tokens', 'seconds')}
                      for name in ('semantic_cost', 'set_cost')},
        per_arm_set_costs={a: {k: sum(c['independent_set_costs'][a][k] for c in costs)
                             for k in ('forward_calls', 'input_tokens', 'seconds')}
                          for a in ('clean_full_union', 'clean_wide_single')},
        new_reader_calls=len(list(read_jsonl(RUN / 'qa/answer_calls.jsonl'))),
        reused_conditions=sum(p['reused'] for p in predictions),
        training=False, heldout_accessed=False, paper_ready=False,
        scope=CONFIG['scope'], mechanism_contribution_isolated=False,
        provenance_proxy_limitation='Paragraph provenance is not complete semantic evidence; full raw support is conservative',
        analysis_rule='report all fixed arms; no checkpoint, threshold or search selection after results')
    common.immutable_json(RUN / 'scored.json', scored)
    common.immutable_json(RUN / 'report.json', report)
    print('SOURCE_VALIDATION_FINAL', json.dumps(report, ensure_ascii=False), flush=True)


def status():
    value = {}
    for name in ('queue_status.json', 'queue_process.json', 'selection_status.json', 'qa/status.json'):
        p = RUN / name
        value[name] = load(p) if p.exists() else None
    value['report_available'] = (RUN / 'report.json').exists()
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'verify', 'preflight', 'select', 'qa', 'score', 'status'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8018')
    args = parser.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    else:
        result = globals()[args.mode]()
        if args.mode == 'verify':
            print('VERIFIED_SOURCE_VALIDATION', result, flush=True)
