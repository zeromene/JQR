"""One fixed three-arm supervision comparison followed by untouched-source QA."""
import argparse
from collections import Counter, defaultdict
import gc
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

from . import baseline_bootstrap_v2 as base
from . import evidence_acquisition_v1 as evidence
from . import replacement_core_v1 as core
from . import replacement_precision_v2 as ledger
from . import replacement_ranker_v1 as ranker
from . import replacement_stage2_common as common
from . import source_validation_v1 as source_pipeline
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/copy_supervision_training_v1'
RUN = ROOT / 'runs/copy_supervision_training_v1'
TRAIN_ARMS = ('ordinary_single', 'ordinary_group', 'joint_copies')
BASELINES = ('static_fusion', 'semantic_mixed', 'original_clean')
ARMS = BASELINES + TRAIN_ARMS
CONFIG = dict(version='copy-supervision-training-v1', seed=29, train_arms=list(TRAIN_ARMS),
    evaluation_arms=list(ARMS), primary='joint_copies', updates_per_arm=304,
    accumulation=8, shared_natural_pairs_per_update=6, new_pairs_per_update=2,
    total_presentations_per_arm=2432, shared_presentations_per_arm=1824,
    new_presentations_per_arm=608, shared_non_tie_presentations=1368,
    shared_neutral_presentations=456, single_raw_presentations=304,
    single_memory_presentations=304, common_max_pairs_per_question_per_bucket=12,
    shared_single_calibration_presentations=304, differential_presentations=304,
    total_common_presentations=2128,
    initialization='pretrained Qwen3-Reranker-4B, identical random LoRA seed29; no warm start',
    ranker_config=ranker.CONFIG, checkpoint='last update304 only; no dev/checkpoint selection',
    reader='Qwen3.5-9B', output_tokens=2048, evidence_cap=2048,
    validation_sources=32, validation_questions=64, max_evaluation_conditions=384,
    candidate_pool='unchanged lexical raw14/extract28; full single/group union, cap24 each',
    score_precision='unchanged BF16 output head; FP32 yes-no subtraction',
    baseline_reranker_batch=4, bootstrap_samples=2000, bootstrap_seed=29,
    original_clean_checkpoint='7e0514d7dbcb5fbdfbd01fe96417eab6e1247cfbafc59233555f5b533e22282c',
    shared_natural_tie_policy='retain ties; 25% of shared presentations, fixed before validation',
    new_tie_policy='retain every normal shared-eligible question regardless of utility gap/sign',
    progress_gate=dict(min_winning_sources_per_comparator=2,
        direct_noninferiority_points=-3.125, require_positive_normal_pair_delta=True),
    protected_final_benchmarks_accessed=False, reserved_sources_consumed_once=True,
    scope='one fixed source-disjoint in-domain mechanism development acceptance; not final benchmark or natural-memory transfer')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pair_id(a, b):
    return fingerprint([a, b])


def build_schedule(pools, intervention_pairs):
    """Exactly identical common slots and question order across all arms."""
    rng = random.Random(CONFIG['seed'])
    cycles = {}
    def draw(name, values):
        if name not in cycles or not cycles[name]:
            order = list(values)
            rng.shuffle(order)
            cycles[name] = order
        return cycles[name].pop()
    schedule = []
    for update in range(CONFIG['updates_per_arm']):
        slots = []
        neutral_count = 1 if update % 2 == 0 else 2
        for kind, count in [('non_tie', 6-neutral_count), ('neutral', neutral_count)]:
            for _ in range(count):
                qid = draw('questions:' + kind, sorted(pools[kind]))
                pid = draw('pairs:' + kind + ':' + qid, pools[kind][qid])
                slots.append(dict(kind='shared', pair_id=pid, bucket=kind))
        common_qid = draw('questions:new_calibration', sorted(intervention_pairs))
        common_action = 'replace_raw' if update % 2 == 0 else 'replace_memory'
        slots.append(dict(kind='single_calibration', question_id=common_qid,
            single_action=common_action,pair_id=intervention_pairs[common_qid][common_action]))
        qid = draw('questions:new_contrast', sorted(intervention_pairs))
        action = 'replace_memory' if update % 2 == 0 else 'replace_raw'
        ids = intervention_pairs[qid]
        slots.append(dict(kind='intervention', question_id=qid, single_action=action,
            pair_ids=dict(ordinary_single=ids[action], ordinary_group=ids['replace_group'],
                          joint_copies=ids['replace_copies'])))
        rng.shuffle(slots)
        schedule.append(dict(update=update+1, slots=slots))
    return schedule


def validate_schedule(schedule, pairs):
    if len(schedule) != 304:
        raise ValueError('Wrong update budget')
    counts = Counter()
    for index, batch in enumerate(schedule):
        if batch['update'] != index+1 or len(batch['slots']) != 8:
            raise ValueError('Wrong accumulation boundary')
        if sum(s['kind']=='shared' for s in batch['slots']) != 6:
            raise ValueError('Shared/new dose differs')
        for slot in batch['slots']:
            counts[slot['kind']] += 1
            if slot['kind'] in ('shared','single_calibration'):
                if slot['pair_id'] not in pairs:
                    raise ValueError('Missing shared pair')
                if slot['kind']=='shared':
                    counts[slot['bucket']] += 1
                else:
                    counts[slot['single_action']] += 1
            else:
                counts[slot['single_action']] += 1
                if set(slot['pair_ids']) != set(TRAIN_ARMS) or any(p not in pairs for p in slot['pair_ids'].values()):
                    raise ValueError('Unmatched arm coverage')
    expected = dict(shared=1824, single_calibration=304, intervention=304, non_tie=1368, neutral=456,
                    replace_raw=304, replace_memory=304)
    if dict(counts) != expected:
        raise ValueError('Training presentation budget changed: ' + str(counts))
    return dict(counts)


def freeze():
    names = ['reader_memory/copy_supervision_training_v1.py',
        'tools/prepare_copy_supervision_training_v1.py', 'scripts/copy_supervision_training_v1_queue.py',
        'tests/test_copy_supervision_training_v1.py', 'reader_memory/replacement_ranker_v1.py',
        'reader_memory/replacement_core_v1.py', 'reader_memory/replacement_precision_v2.py',
        'reader_memory/replacement_learning_v1.py', 'reader_memory/replacement_stage2_common.py',
        'reader_memory/source_validation_v1.py', 'reader_memory/clean_feedback_control_v1.py',
        'reader_memory/mainline_reranker_v1.py', 'reader_memory/mainline_service_v1.py',
        'reader_memory/baseline_bootstrap_v2.py', 'reader_memory/evidence_acquisition_v1.py',
        'reader_memory/evidence_acquisition_v2.py', 'reader_memory/backend.py', 'reader_memory/schema.py',
        'tools/prepare_replacement_learning_v1.py', 'reader_memory/_vendor/locomo_native_f1.py',
        'third_party/simplemem_official/simplemem/core/answer_generator.py',
        'third_party/simplemem_official/simplemem/core/utils/llm_client.py']
    names += ['research/copy_supervision_training_v1/' + n for n in
        ['PROTOCOL.md', 'public/training_inputs.json', 'private/targets.json', 'private/pairs.json',
         'training_schedule.json', 'preparation.json', 'source_provenance.json',
         'private/source_selection.json', 'model_references.json']]
    value = dict(config=CONFIG, files={n: base.sha(ROOT/n) for n in names})
    value['signature'] = fingerprint(value)
    common.immutable_json(PLAN/'manifest.json', value)
    print('FROZEN_COPY_SUPERVISION_TRAINING', value['signature'], flush=True)


def verify():
    saved = load(PLAN/'manifest.json')
    if saved['config'] != CONFIG or saved['signature'] != fingerprint({k:v for k,v in saved.items() if k!='signature'}):
        raise ValueError('Training scientific identity changed')
    for name, digest in saved['files'].items():
        if base.sha(ROOT/name) != digest:
            raise ValueError('Frozen training input changed: ' + name)
    for name, reference in load(PLAN/'model_references.json').items():
        if load(ROOT/name) != reference:
            raise ValueError('Pinned model artifact content changed: ' + name)
    return saved['signature']


def preflight():
    sid = verify()
    from tokenizers import Tokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    tok = Tokenizer.from_file(str(ROOT/'models/Qwen3-Reranker-4B/tokenizer.json'))
    class Adapter:
        def encode(self, text, add_special_tokens=False):
            return tok.encode(text, add_special_tokens=add_special_tokens).ids
    counter = ExactCounter()
    rows = load(PLAN/'public/training_inputs.json')
    targets = load(PLAN/'private/targets.json')
    max_ranker = 0
    for r in rows:
        if r['evidence_sha256'] != fingerprint(r['evidence_text']) or counter.count(r['evidence_text']) != r['evidence_tokens']:
            raise ValueError('Training text/token mismatch')
        if r['evidence_tokens'] > 2048 or targets[r['entry_id']]['status'] != 'ok':
            raise ValueError('Invalid training budget or failed-feedback example')
        max_ranker = max(max_ranker, len(ranker.input_ids(Adapter(), r['question'], r['evidence_text'])))
    counts = validate_schedule(load(PLAN/'training_schedule.json'), load(PLAN/'private/pairs.json'))
    for model in ['Qwen3-Reranker-4B', 'Qwen3.5-9B']:
        if not (ROOT/'models'/model/'.verified.json').exists():
            raise ValueError('Missing offline model verification marker')
    ranker_reference = load(PLAN/'model_references.json')['models/Qwen3-Reranker-4B/artifact_manifest.json']
    for item in ranker_reference['files']:
        if item['size'] <= 50000000 and base.sha(ROOT/'models/Qwen3-Reranker-4B'/item['name']) != item['sha256']:
            raise ValueError('Ranker configuration/tokenizer differs from upstream: '+item['name'])
    if base.sha(ROOT/'runs/clean_feedback_control_v1/adapter.pt') != CONFIG['original_clean_checkpoint']:
        raise ValueError('Original clean baseline checkpoint missing or changed')
    common.immutable_json(RUN/'preflight.json', dict(signature=sid, passed=True,
        training_contexts=len(rows), max_ranker_tokens=max_ranker, schedule_counts=counts,
        model_markers={m:base.sha(ROOT/'models'/m/'.verified.json') for m in ['Qwen3-Reranker-4B','Qwen3.5-9B']},
        reserved_contexts_materialized=False, protected_final_benchmarks_accessed=False))
    print('COPY_TRAIN_PREFLIGHT_OK', len(rows), flush=True)


def state_digest(state):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(arm):
    if arm not in TRAIN_ARMS:
        raise ValueError('Unknown training arm')
    sid = verify()
    folder = RUN/'models'/arm
    if (folder/'training_complete.json').exists():
        check_checkpoint(arm)
        print('TRAINING_ALREADY_COMPLETED', arm, flush=True)
        return
    if (folder/'training_started.json').exists():
        raise ValueError('Training already attempted; inspect interrupted run, no automatic retraining')
    common.immutable_json(folder/'training_started.json', dict(signature=sid, arm=arm,
        updates=304, presentations=2432, initialization=CONFIG['initialization']))
    torch, tok, model, paths = ranker.load(ROOT/'models/Qwen3-Reranker-4B', training=True)
    initial_digest = state_digest(ranker.adapter_state(model))
    common.immutable_json(RUN/'initial_adapter_identity.json', dict(signature=sid, tensor_sha256=initial_digest,
        adapter_paths=paths, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    rows = {r['entry_id']:r for r in load(PLAN/'public/training_inputs.json')}
    targets = load(PLAN/'private/targets.json')
    pairs = load(PLAN/'private/pairs.json')
    schedule = load(PLAN/'training_schedule.json')
    encoded = {key:ranker.input_ids(tok,r['question'],r['evidence_text']) for key,r in rows.items()}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=ranker.CONFIG['learning_rate'], weight_decay=ranker.CONFIG['weight_decay'])
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    tokens = 0
    for batch in schedule:
        losses = []
        for slot in batch['slots']:
            pid = slot['pair_id'] if slot['kind'] in ('shared','single_calibration') else slot['pair_ids'][arm]
            pair = pairs[pid]
            a, b = pair['a'], pair['b']
            ra, rb = targets[a]['reward'], targets[b]['reward']
            va = ranker.logit(torch, tok, model, encoded[a])
            vb = ranker.logit(torch, tok, model, encoded[b])
            preference = torch.tensor(1/(1+math.exp(-4*(ra-rb))), device='cuda')
            loss = torch.nn.functional.binary_cross_entropy_with_logits(va-vb, preference)
            loss = loss + .25*((va.sigmoid()-ra)**2+(vb.sigmoid()-rb)**2)/2
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            losses.append(float(loss.detach()))
            (loss/len(batch['slots'])).backward()
            tokens += len(encoded[a])+len(encoded[b])
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        base.append(folder/'training_log.jsonl', dict(signature=sid, arm=arm, updates=batch['update'],
            presentations=8*batch['update'], mean_loss=statistics.mean(losses),
            input_tokens=tokens, seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        write_json(folder/'status.json', dict(status='training', updates=batch['update'], expected_updates=304))
        if batch['update'] == 1 or batch['update']%10 == 0:
            print('COPY_TRAIN', arm, batch['update'], '/304', flush=True)
    checkpoint = folder/'adapter.pt'
    torch.save(ranker.adapter_state(model), checkpoint)
    common.immutable_json(folder/'training_complete.json', dict(signature=sid, arm=arm,
        updates=304, presentations=2432, initial_adapter_tensor_sha256=initial_digest,
        checkpoint_sha256=base.sha(checkpoint), input_tokens=tokens,
        seconds=time.perf_counter()-started, checkpoint_selection='last update304',
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    write_json(folder/'status.json', dict(status='completed', updates=304, expected_updates=304))


def check_checkpoint(arm):
    path = RUN/'models'/arm
    saved = load(path/'training_complete.json')
    if saved['signature'] != verify() or saved['updates'] != 304 or saved['presentations'] != 2432:
        raise ValueError('Training budget or identity differs')
    if base.sha(path/'adapter.pt') != saved['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if saved['initial_adapter_tensor_sha256'] != load(RUN/'initial_adapter_identity.json')['tensor_sha256']:
        raise ValueError('Arms had different initializations')
    return path/'adapter.pt'


def prepare_validation():
    sid = verify()
    for arm in TRAIN_ARMS:
        check_checkpoint(arm)
    selection = load(PLAN/'private/source_selection.json')
    reserved = selection['reserved_validation']
    if (RUN/'validation/manifest.json').exists():
        verify_validation()
        return
    from tools.prepare_replacement_learning_v1 import SOURCE, SOURCE_SHA, ExactCounter
    if base.sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Original dataset changed')
    native_rows = {}
    with SOURCE.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['id'] in set(reserved):
                native_rows[row['id']] = row
    if set(native_rows) != set(reserved):
        raise ValueError('Missing validation source')
    counter = ExactCounter()
    public, references = [], {}
    for native in reserved:
        row = native_rows[native]
        for mode in ['multi','single']:
            question = row['question'] if mode=='multi' else row['question_decomposition'][0]['question']
            if mode=='single' and '#' in question:
                raise ValueError('Unbound native direct question')
            source = source_pipeline.public_input(row, question, mode, counter)
            public.append(source)
            references[source['question_id']] = dict(native_id=native, mode=mode,
                answer=row['answer'] if mode=='multi' else row['question_decomposition'][0]['answer'],
                aliases=row.get('answer_aliases',[]) if mode=='multi' else [])
    common.immutable_json(RUN/'validation/public_sources.json', public)
    common.immutable_json(RUN/'validation/private_references.json', references)
    value = dict(signature=sid, reserved_sources=reserved, source_sha256=SOURCE_SHA,
        source_count=32, question_count=64, annotations_used_for_candidate_selection=False,
        files={n:base.sha(RUN/'validation'/n) for n in ['public_sources.json','private_references.json']},
        purpose='one fixed mechanism acceptance; never reuse as training or retune after results')
    value['receipt_signature'] = fingerprint(value)
    common.immutable_json(RUN/'validation/manifest.json', value)
    print('PREPARED_RESERVED_VALIDATION', len(public), flush=True)


def verify_validation():
    saved = load(RUN/'validation/manifest.json')
    if saved['signature'] != verify() or saved['receipt_signature'] != fingerprint({k:v for k,v in saved.items() if k!='receipt_signature'}):
        raise ValueError('Validation identity changed')
    if saved['reserved_sources'] != load(PLAN/'private/source_selection.json')['reserved_validation']:
        raise ValueError('Validation source changed')
    for name, digest in saved['files'].items():
        if base.sha(RUN/'validation'/name) != digest:
            raise ValueError('Validation artifact changed')
    return saved['receipt_signature']


def candidates():
    sid = verify()
    validation_sid = verify_validation()
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter = ExactCounter()
    torch, tok, model, _ = ranker.load(ROOT/'models/Qwen3-Reranker-4B')
    for source in load(RUN/'validation/public_sources.json'):
        qid = source['question_id']
        dest = RUN/'candidates'/(qid.replace(':','_')+'.json')
        if dest.exists():
            saved = load(dest)
            if saved['signature'] != sid or saved['source_sha256'] != fingerprint(source):
                raise ValueError('Candidate identity changed')
            continue
        intent = RUN/'candidate_inflight.json'
        if intent.exists():
            raise ValueError('Uncommitted candidate call; no automatic repetition')
        write_json(intent, dict(signature=sid, question_id=qid))
        individual, cost = source_pipeline.semantic_scores(torch,tok,model,source)
        records, pool = source['records'], source['candidate_pool_ids']
        order = sorted(pool,key=lambda k:(-individual[k],pool.index(k)))
        semantic_ids = evidence.ordered_pack(order,records,counter,2048)
        options = source_pipeline.alternatives(source,individual,counter)['clean_full_union']
        baselines = []
        meta = {k:source[k] for k in ['question_id','native_id','mode','question']}
        for name, ids in [('static_fusion',source['seed_ids']),('semantic_mixed',semantic_ids)]:
            text = evidence.render(ids,records)
            baselines.append(dict(**meta,arm=name,selected_ids=ids,evidence_text=text,
                evidence_sha256=fingerprint(text),evidence_tokens=counter.count(text),
                private_inputs_read=False))
        common.immutable_json(dest,dict(signature=sid,validation_signature=validation_sid,
            source_sha256=fingerprint(source), options=options, baselines=baselines,
            semantic_cost=cost, private_inputs_read=False))
        intent.unlink()
        print('COPY_CANDIDATES',qid,len(options),flush=True)
    files = list((RUN/'candidates').glob('*.json'))
    if len(files) != 64:
        raise ValueError('Incomplete candidate coverage')
    common.immutable_json(RUN/'candidate_manifest.json',dict(signature=sid,
        validation_signature=validation_sid,files={p.name:base.sha(p) for p in sorted(files)}))


def select(arm):
    if arm not in ('original_clean',)+TRAIN_ARMS:
        raise ValueError('Unknown set-scoring arm')
    sid = verify()
    verify_validation()
    cm = load(RUN/'candidate_manifest.json')
    if cm['signature'] != sid:
        raise ValueError('Wrong candidate manifest')
    checkpoint = ROOT/'runs/clean_feedback_control_v1/adapter.pt' if arm=='original_clean' else check_checkpoint(arm)
    if arm=='original_clean' and base.sha(checkpoint) != CONFIG['original_clean_checkpoint']:
        raise ValueError('Original clean baseline checkpoint changed')
    torch,tok,model,_ = ranker.load(ROOT/'models/Qwen3-Reranker-4B')
    ranker.load_adapter(model,torch.load(checkpoint,map_location='cpu',weights_only=True))
    sources = {s['question_id']:s for s in load(RUN/'validation/public_sources.json')}
    for name,digest in cm['files'].items():
        path = RUN/'candidates'/name
        if base.sha(path) != digest:
            raise ValueError('Common candidate changed')
        saved = load(path)
        qid = saved['baselines'][0]['question_id']
        dest = RUN/'selected'/arm/name
        if dest.exists():
            value = load(dest)
            if value['signature'] != sid or value['checkpoint_sha256'] != base.sha(checkpoint):
                raise ValueError('Saved selection identity changed')
            continue
        intent = RUN/('selection_inflight_'+arm+'.json')
        if intent.exists():
            raise ValueError('Uncommitted scoring call; inspect before repetition')
        write_json(intent,dict(signature=sid,question_id=qid,arm=arm))
        scores, lengths = {}, []
        started = time.perf_counter()
        with torch.inference_mode():
            for option in saved['options']:
                ids = ranker.input_ids(tok,sources[qid]['question'],option['evidence_text'])
                value = float(ranker.logit(torch,tok,model,ids))
                if not math.isfinite(value):
                    raise ValueError('Nonfinite selector score')
                scores[option['evidence_sha256']] = value
                lengths.append(len(ids))
        # First option is the static anchor; existing option order breaks ties.
        best = max(saved['options'],key=lambda p:scores[p['evidence_sha256']])
        meta = {k:sources[qid][k] for k in ['question_id','native_id','mode','question']}
        plan = dict(**meta,**best,arm=arm,private_inputs_read=False)
        common.immutable_json(dest,dict(signature=sid,checkpoint_sha256=base.sha(checkpoint),
            plan=plan,scores=scores,winning_ties=sum(v==max(scores.values()) for v in scores.values()),
            cost=dict(forward_calls=len(scores),input_tokens=sum(lengths),seconds=time.perf_counter()-started)))
        intent.unlink()
        print('COPY_SELECT',arm,qid,flush=True)


def assemble():
    sid = verify()
    validation_sid = verify_validation()
    for arm in TRAIN_ARMS:
        check_checkpoint(arm)
    plans = []
    sources = load(RUN/'validation/public_sources.json')
    for source in sources:
        name = source['question_id'].replace(':','_')+'.json'
        plans.extend(load(RUN/'candidates'/name)['baselines'])
        for arm in ('original_clean',)+TRAIN_ARMS:
            saved = load(RUN/'selected'/arm/name)
            if saved['signature'] != sid:
                raise ValueError('Selection signature changed')
            plans.append(saved['plan'])
    expected = {(s['question_id'],a) for s in sources for a in ARMS}
    if len(plans)!=384 or {(p['question_id'],p['arm']) for p in plans}!=expected:
        raise ValueError('Incomplete evaluation coverage')
    common.immutable_json(RUN/'qa_plans.json',plans)
    common.immutable_json(RUN/'selection_complete.json',dict(signature=sid,plans_sha256=base.sha(RUN/'qa_plans.json')))
    common.immutable_json(RUN/'selection_provenance.json',dict(signature=sid,validation_signature=validation_sid,
        conditions=384,candidate_manifest_sha256=base.sha(RUN/'candidate_manifest.json'),
        checkpoints={a:base.sha(check_checkpoint(a)) for a in TRAIN_ARMS},
        original_clean_checkpoint=CONFIG['original_clean_checkpoint']))


def qa(endpoint):
    sid = verify()
    verify_validation()
    if load(RUN/'selection_complete.json')['plans_sha256'] != base.sha(RUN/'qa_plans.json'):
        raise ValueError('Evaluation plans changed')
    old = ledger.RUN,ledger.verify,ledger.saved_cache
    ledger.RUN,ledger.verify,ledger.saved_cache = RUN,lambda:sid,lambda:{}
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN,ledger.verify,ledger.saved_cache = old


def interval(by_source):
    rng = random.Random(CONFIG['bootstrap_seed'])
    keys = sorted(by_source)
    values = []
    for _ in range(CONFIG['bootstrap_samples']):
        sample = [v for _ in keys for v in by_source[rng.choice(keys)]]
        values.append(100*statistics.mean(sample))
    values.sort()
    return [values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]]


def score():
    sid = verify()
    verify_validation()
    refs = load(RUN/'validation/private_references.json')
    plans = load(RUN/'qa_plans.json')
    rows = list(read_jsonl(RUN/'qa/predictions.jsonl'))
    expected = {(p['question_id'],p['arm']):p for p in plans}
    if len(rows)!=384 or {(r['question_id'],r['arm']) for r in rows}!=set(expected):
        raise ValueError('Complete unique QA required')
    scored = []
    for row in rows:
        plan = expected[row['question_id'],row['arm']]
        if row['signature']!=sid or row['plan_sha256']!=fingerprint(plan) or any(row[k]!=plan[k] for k in plan):
            raise ValueError('Committed QA plan changed')
        ref = refs[row['question_id']]
        f1 = max(base.native_f1(row['hypothesis'],a,2) for a in [ref['answer']]+ref['aliases']) if row['status']=='ok' else 0.
        scored.append(dict(**row,local_native_f1=f1))
    by = {a:{r['question_id']:r for r in scored if r['arm']==a} for a in ARMS}
    summaries = []
    for arm in ARMS:
        for mode in ['overall','multi','single']:
            group = [r for r in by[arm].values() if mode=='overall' or r['mode']==mode]
            summaries.append(dict(arm=arm,scope=mode,n=len(group),f1_points=100*statistics.mean(r['local_native_f1'] for r in group),
                mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in group),
                statuses=dict(Counter(r['status'] for r in group))))
    comparisons = []
    for comparator in [a for a in ARMS if a!='joint_copies']:
        delta,normal,direct = [],[],[]
        sources = defaultdict(list)
        for q,a in by['joint_copies'].items():
            b = by[comparator][q]
            d = a['local_native_f1']-b['local_native_f1']
            delta.append(d)
            sources[a['native_id']].append(d)
            if a['status']==b['status']=='ok':
                normal.append(d)
            if a['mode']=='single':
                direct.append(d)
        comparisons.append(dict(arm='joint_copies',comparator=comparator,delta_points=100*statistics.mean(delta),
            normal_pairs=len(normal),normal_pair_delta_points=100*statistics.mean(normal) if normal else None,
            direct_delta_points=100*statistics.mean(direct),source_bootstrap_95_points=interval(sources),
            winning_sources=sum(statistics.mean(v)>0 for v in sources.values()),
            losing_sources=sum(statistics.mean(v)<0 for v in sources.values()),
            wins=sum(d>1e-9 for d in delta),ties=sum(abs(d)<=1e-9 for d in delta),losses=sum(d<-1e-9 for d in delta)))
    checks = {}
    for c in comparisons:
        checks[c['comparator']] = (c['delta_points']>0 and (c['normal_pair_delta_points'] or 0)>0
            and c['winning_sources']>=2 and c['direct_delta_points']>=-3.125-1e-9)
    calls = list(read_jsonl(RUN/'qa/answer_calls.jsonl'))
    costs = {}
    for arm in ('original_clean',)+TRAIN_ARMS:
        selected = [load(p) for p in (RUN/'selected'/arm).glob('*.json')]
        costs[arm] = dict(forward_calls=sum(s['cost']['forward_calls'] for s in selected),
            input_tokens=sum(s['cost']['input_tokens'] for s in selected),
            seconds=sum(s['cost']['seconds'] for s in selected),winning_ties=sum(s['winning_ties']>1 for s in selected))
    candidate_rows = [load(p) for p in (RUN/'candidates').glob('*.json')]
    report = dict(signature=sid,complete=True,scores=summaries,comparisons=comparisons,
        progress_checks=checks,provisional_progress_signal=all(checks.values()),
        checkpoint_selection=False,hyperparameter_search=False,validation_sources_consumed=32,
        reader_calls=len(calls),reused_conditions=sum(r['reused'] for r in rows),
        reader_cost={k:sum(c.get('result',{}).get('usage',{}).get(k,0) for c in calls) for k in ['prompt_tokens','completion_tokens']},
        semantic_cost={k:sum(r['semantic_cost'][k] for r in candidate_rows) for k in ['forward_calls','input_tokens','seconds']},
        selector_costs=costs,training_costs={a:load(RUN/'models'/a/'training_complete.json') for a in TRAIN_ARMS},
        protected_final_benchmarks_accessed=False,paper_ready=False,scope=CONFIG['scope'],
        next_action='Consider natural-memory transfer and larger fixed confirmation' if all(checks.values())
            else 'Stop this fixed version; report unsupported mechanism/competitiveness without tuning on these sources')
    write_json(RUN/'scored.json',scored)
    write_json(RUN/'report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['freeze','verify','preflight','train','prepare_validation','candidates','select','assemble','qa','score'])
    parser.add_argument('--arm')
    parser.add_argument('--endpoint',default='http://127.0.0.1:8018')
    args = parser.parse_args()
    if args.mode in ['train','select']:
        globals()[args.mode](args.arm)
    elif args.mode=='qa':
        qa(args.endpoint)
    else:
        result = globals()[args.mode]()
        if result:
            print(result)
