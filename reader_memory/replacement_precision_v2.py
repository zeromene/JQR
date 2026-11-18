"""Isolated v2 numerical/candidate controls over preserved v1 inputs; no retraining."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from . import baseline_bootstrap_v2 as base
from . import evidence_acquisition_v1 as evidence
from . import mainline_reranker_v1 as semantic
from . import replacement_core_v1 as core
from . import replacement_learning_v1 as parent
from . import replacement_ranker_v1 as ranker
from .evidence_acquisition_v2 import parse_answer
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/replacement_precision_v2'
RUN = ROOT / 'runs/replacement_precision_v2'
PARENT_RUN = ROOT / 'runs/replacement_learning_v1_repair1'
ARMS = ('single_fp32', 'group_fp32', 'union_fp32', 'wide_single_fp32', 'union_bf16',
        'reranker_mixed_fp32', 'reranker_mixed_bf16_serial')
CONFIG = dict(arms=list(ARMS), budget=2048, reader='Qwen3.5-9B', reader_max_tokens=2048,
              score='same backbone forward; yes/no lm_head rows computed in FP32 before output rounding',
              tie='original single proposals first, then group-only; anchor first',
              nested='v1 single union v1 group',
              wide_single='only single swaps; proposal cap equals nested candidate count; actual costs reported',
              baseline='original unadapted weights, serial paired BF16/FP32 control, identical mixed pool',
              candidate_generation='frozen v1 semantic scores; only baseline arms re-rank individual documents',
              scope='engineering development controls on 64 exposed questions; no independent confirmation',
              heldout_accessed=False, retraining=False, batching=1)


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def ordered_union(single, group):
    result = {}
    for p in list(single) + list(group): result.setdefault(p['evidence_sha256'], p)
    return list(result.values())


def two_head_scores(torch, tokenizer, model, ids):
    """Capture the SAME head input for BF16 and FP32; preserve gradients for future training."""
    if model.config.model_type != 'qwen3' or model.__class__.__name__ != 'Qwen3ForCausalLM':
        raise ValueError('Unvalidated output-head architecture')
    yes = tokenizer.convert_tokens_to_ids('yes'); no = tokenizer.convert_tokens_to_ids('no')
    if yes == no or yes == tokenizer.unk_token_id or no == tokenizer.unk_token_id:
        raise ValueError('Invalid yes/no tokens')
    captured = {}
    def hook(module, args, output):
        if module.__class__.__name__ != 'Linear' or module.weight.ndim != 2:
            raise ValueError('Output head is not the validated linear map')
        hidden = args[0][:, -1, :]
        weight = module.weight[[yes, no]].float()
        bias = None if module.bias is None else module.bias[[yes, no]].float()
        projected = torch.nn.functional.linear(hidden.float(), weight, bias)
        captured['fp32'] = (projected[:, 0] - projected[:, 1])[0]
        captured['bf16'] = (output[:, -1, yes].float() - output[:, -1, no].float())[0]
    handle = model.lm_head.register_forward_hook(hook)
    try:
        value = torch.tensor([ids], device=model.device, dtype=torch.long)
        model(input_ids=value, attention_mask=torch.ones_like(value), logits_to_keep=1, use_cache=False)
    finally: handle.remove()
    if set(captured) != {'fp32', 'bf16'}: raise ValueError('Output-head hook was not called exactly as expected')
    return captured


def build_identity():
    signature = parent.verify(); base.verify_inputs()
    if load(PARENT_RUN / 'queue_status.json')['status'] != 'completed': raise ValueError('Parent not completed')
    checkpoint = load(PARENT_RUN / 'training_complete.json')['checkpoint_sha256']
    if base.sha(PARENT_RUN / 'adapter.pt') != checkpoint: raise ValueError('Parent checkpoint changed')
    receipts = sorted((parent.PLAN / 'preserved').glob('*.receipt.json'))
    if not receipts: raise ValueError('Preserve v1 before starting v2')
    receipt = load(receipts[-1])
    if receipt['scientific_signature'] != signature or receipt['checkpoint_sha256'] != checkpoint:
        raise ValueError('Preserved version differs from parent')
    paths = ['reader_memory/replacement_precision_v2.py', 'scripts/replacement_precision_queue_v2.py',
             'tests/test_replacement_precision_v2.py', 'research/replacement_precision_v2/PROTOCOL.md',
             'runs/replacement_learning_v1_repair1/qa/predictions.jsonl',
             'runs/replacement_learning_v1_repair1/qa/answer_calls.jsonl',
             'runs/replacement_learning_v1_repair1/scored.json', 'runs/evidence_acquisition_v2/scored.json',
             'runs/mainline_reranker_v1_reparsed/scored.json', 'runs/mainline_reranker_v1/answer_calls.jsonl']
    paths += [str(p.relative_to(ROOT)) for p in sorted((PARENT_RUN / 'selected').glob('*.json'))]
    for cid in ['conv-43', 'conv-44']:
        paths += [f'runs/evidence_acquisition_v1/{cid}/candidate_records.json',
                  f'runs/evidence_acquisition_v2/{cid}/plans.json']
        paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'runs/mainline_reranker_v1' / cid).glob('*.scores.json'))]
    return dict(config=CONFIG, parent_signature=signature, parent_checkpoint_sha256=checkpoint,
                preserved_archive_sha256=receipt['sha256'], files={p:base.sha(ROOT / p) for p in paths})


def freeze():
    value = build_identity(); value['signature'] = fingerprint(value)
    path = PLAN / 'manifest.json'
    if path.exists() and load(path) != value: raise ValueError('v2 already frozen with different inputs')
    write_json(path, value)
    print('FROZEN_PRECISION_V2', value['signature'], flush=True)


def verify():
    expected = load(PLAN / 'manifest.json'); value = build_identity(); value['signature'] = fingerprint(value)
    if value != expected: raise ValueError('Frozen v2 identity changed')
    return value['signature']


def alternatives_for(q, saved, records, counter):
    single = saved['alternatives']['learned_single_swap']; group = saved['alternatives']['learned_group_swap']
    union = ordered_union(single, group)
    pool = saved['plans'][0]['candidate_pool_ids']
    cached = load(ROOT / 'runs/mainline_reranker_v1' / q['conversation_id'] /
                  (q['question_id'].replace(':', '_') + '.scores.json'))
    if pool != cached['candidate_pool_ids']: raise ValueError('Candidate pool changed')
    scores = dict(zip(pool, cached['scores'], strict=True))
    wide = core.proposal_sets(q['question'], single[0]['selected_ids'], pool, records, scores, counter,
                             False, max_sets=len(union))
    if not {p['evidence_sha256'] for p in single} <= {p['evidence_sha256'] for p in union}:
        raise ValueError('Nested arm lost a single proposal')
    if not {p['evidence_sha256'] for p in single} <= {p['evidence_sha256'] for p in wide}:
        raise ValueError('Wide single arm lost its original proposals')
    return dict(single_fp32=single, group_fp32=group, union_fp32=union,
                wide_single_fp32=wide, union_bf16=union)


def select():
    sid = verify()
    from .tokenization import TokenCounter
    counter = TokenCounter(ROOT / 'models/Qwen3.5-9B')
    questions = load(base.PLAN / 'public/queries.json')
    torch, tok, model, _ = ranker.load(parent.MODEL)
    ranker.load_adapter(model, torch.load(PARENT_RUN / 'adapter.pt', map_location='cpu', weights_only=True))
    all_plans = []; all_ties = defaultdict(list); legacy_errors = []; costs = []; score_values = defaultdict(list)
    for q in questions:
        qid = q['question_id']; slug = qid.replace(':', '_')
        target = RUN / 'sets' / (slug + '.json')
        if target.exists():
            value = load(target)
            if value['signature'] != sid: raise ValueError('Different selection identity')
        else:
            saved = load(PARENT_RUN / 'selected' / (slug + '.json'))
            records = load(ROOT / 'runs/evidence_acquisition_v1' / q['conversation_id'] / 'candidate_records.json')
            alternatives = alternatives_for(q, saved, records, counter)
            unique = ordered_union(alternatives['union_fp32'], alternatives['wide_single_fp32'])
            intent = RUN / 'selection_inflight.json'
            if intent.exists(): raise ValueError('Unknown selector commit; inspect before rerun')
            write_json(intent, dict(signature=sid, question_id=qid, phase='sets'))
            started = time.perf_counter(); scores = {}; lengths = {}
            with torch.inference_mode():
                for p in unique:
                    ids = ranker.input_ids(tok, q['question'], p['evidence_text'])
                    pair = two_head_scores(torch, tok, model, ids)
                    scores[p['evidence_sha256']] = {k:float(v) for k,v in pair.items()}
                    lengths[p['evidence_sha256']] = len(ids)
            errors = [abs(scores[k]['bf16'] - v) for k,v in saved['scores'].items()]
            if max(errors, default=0.) > 1e-6:
                write_json(RUN / 'legacy_reproduction_failure.json', dict(question_id=qid, maximum=max(errors)))
                raise ValueError('Original BF16 scores did not reproduce; stop isolated precision comparison')
            plans = []
            for arm, proposed in alternatives.items():
                precision = 'bf16' if arm == 'union_bf16' else 'fp32'
                best = max(proposed, key=lambda p:scores[p['evidence_sha256']][precision])
                top = scores[best['evidence_sha256']][precision]
                plans.append(dict(**q, arm=arm, **best, scored_sets=len(proposed),
                                  score_input_tokens=sum(lengths[p['evidence_sha256']] for p in proposed),
                                  predicted_score=top, top_ties=sum(scores[p['evidence_sha256']][precision] == top for p in proposed),
                                  private_inputs_read=False))
            value = dict(signature=sid, plans=plans, scores=scores, alternatives=alternatives,
                         legacy_max_error=max(errors, default=0.),
                         costs=dict(forward_calls=len(unique), input_tokens=sum(lengths.values()),
                                    seconds=time.perf_counter()-started))
            write_json(target, value); intent.unlink()
        all_plans.extend(value['plans']); legacy_errors.append(value['legacy_max_error']); costs.append(value['costs'])
        for pair in value['scores'].values():
            for precision, score_value in pair.items(): score_values[precision].append(score_value)
        for p in value['plans']: all_ties[p['arm']].append(p['top_ties'])
        print('V2_SET_SCORE', qid, len(value['scores']), flush=True)
    # Apply the identical output-head control to the strong unadapted mixed-pool reranker.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('.b'): parameter.zero_()
    for q in questions:
        qid = q['question_id']; slug = qid.replace(':', '_'); target = RUN / 'baseline' / (slug + '.json')
        if target.exists():
            value = load(target)
            if value['signature'] != sid: raise ValueError('Different baseline identity')
        else:
            saved = load(PARENT_RUN / 'selected' / (slug + '.json'))
            pool = saved['plans'][0]['candidate_pool_ids']
            records = load(ROOT / 'runs/evidence_acquisition_v1' / q['conversation_id'] / 'candidate_records.json')
            intent = RUN / 'selection_inflight.json'
            if intent.exists(): raise ValueError('Unknown baseline commit')
            write_json(intent, dict(signature=sid, question_id=qid, phase='baseline'))
            started = time.perf_counter(); scores = {}; lengths = {}
            with torch.inference_mode():
                for key in pool:
                    body = f"<Instruct>: {semantic.CONFIG['instruction']}\n<Query>: {q['question']}\n<Document>: {evidence.render([key], records)}"
                    ids = tok.encode(semantic.PREFIX, add_special_tokens=False) + tok.encode(body, add_special_tokens=False) + tok.encode(semantic.SUFFIX, add_special_tokens=False)
                    if len(ids) > 8192: raise ValueError('Baseline input too long; no truncation')
                    pair = two_head_scores(torch, tok, model, ids)
                    scores[key] = {k:float(v) for k,v in pair.items()}; lengths[key] = len(ids)
            plans = []
            for precision, arm in [('bf16', 'reranker_mixed_bf16_serial'), ('fp32', 'reranker_mixed_fp32')]:
                order = sorted(pool, key=lambda key:(-scores[key][precision], pool.index(key)))
                selected = evidence.ordered_pack(order, records, counter, 2048)
                text = evidence.render(selected, records)
                plans.append(dict(**q, arm=arm, selected_ids=selected, evidence_text=text, evidence_sha256=fingerprint(text),
                                  evidence_tokens=counter.count(text), private_inputs_read=False))
            value = dict(signature=sid, plans=plans, scores=scores,
                         costs=dict(forward_calls=len(pool), input_tokens=sum(lengths.values()), seconds=time.perf_counter()-started))
            write_json(target, value); intent.unlink()
        all_plans.extend(value['plans']); costs.append(value['costs'])
        print('V2_BASELINE_SCORE', qid, len(value['scores']), flush=True)
    if {(p['question_id'],p['arm']) for p in all_plans} != {(q['question_id'],a) for q in questions for a in ARMS}:
        raise ValueError('Incomplete v2 plans')
    write_json(RUN / 'qa_plans.json', all_plans)
    write_json(RUN / 'selection_complete.json', dict(signature=sid, plans_sha256=base.sha(RUN / 'qa_plans.json'),
        legacy_max_error=max(legacy_errors), ties={a:sum(v>1 for v in values) for a,values in all_ties.items()},
        score_precision={p:dict(values=len(v), unique=len(set(v))) for p,v in score_values.items()},
        costs={k:sum(c[k] for c in costs) for k in ['forward_calls','input_tokens','seconds']},
        model_retrained=False, heldout_accessed=False))


def messages_for(row):
    return [dict(role='system', content=parent.SYSTEM),
            dict(role='user', content=evidence.PROMPT(None, row['question'], row['evidence_text']))]


def request_key(messages, max_tokens=2048):
    return fingerprint(dict(messages=messages, max_tokens=max_tokens, reader='Qwen3.5-9B',
                            decoding='greedy-thinking_false', reader_marker_sha256=base.sha(ROOT / 'models/Qwen3.5-9B/.verified.json')))


def saved_cache():
    cache = {}
    sources = [(PARENT_RUN / 'qa/predictions.jsonl', PARENT_RUN / 'qa/answer_calls.jsonl'),
               (ROOT / 'runs/mainline_reranker_v1_reparsed/scored.json', ROOT / 'runs/mainline_reranker_v1/answer_calls.jsonl')]
    for predictions, calls in sources:
        rows = load(predictions) if predictions.suffix == '.json' else list(read_jsonl(predictions))
        ledger = {(r['question_id'],r['arm']):r for r in read_jsonl(calls)}
        for r in rows:
            c = ledger[r['question_id'], r['arm']]
            text = r.get('evidence_text') or evidence.FORMAT(None, [SimpleNamespace(**v) for v in r['contexts']])
            messages = messages_for(dict(question=r['question'], evidence_text=text))
            if messages != c['messages'] or c['max_tokens'] != 2048: raise ValueError('Cache request mismatch')
            if r['status'] == 'ok' and parse_answer(base.require_complete(c['result'])) != r['hypothesis']:
                raise ValueError('Cache answer does not reproduce official parser')
            key = request_key(messages)
            entry = dict(hypothesis=r['hypothesis'], status=r['status'], error=r.get('error'), result=c['result'],
                         source=str(predictions.relative_to(ROOT)), source_condition=[r['question_id'],r['arm']])
            if key in cache and (cache[key]['hypothesis'],cache[key]['status']) != (entry['hypothesis'],entry['status']):
                raise ValueError('Identical cached requests have inconsistent outcomes')
            cache.setdefault(key, entry)
    return cache


def qa(endpoint):
    sid = verify(); meta = load(RUN / 'selection_complete.json')
    if meta['plans_sha256'] != base.sha(RUN / 'qa_plans.json'): raise ValueError('QA plans changed')
    plans = load(RUN / 'qa_plans.json'); cache = saved_cache()
    dest = RUN / 'qa/predictions.jsonl'; done = {}
    if dest.exists():
        for r in read_jsonl(dest):
            key = (r['question_id'],r['arm'])
            if key in done or r['signature'] != sid: raise ValueError('Invalid QA ledger')
            done[key] = r
    for path in (RUN / 'qa/cache').glob('*.json'):
        r = load(path)
        if r['signature'] != sid or r['key'] != path.stem: raise ValueError('Invalid local QA cache')
        cache[r['key']] = r['entry']
    intent = RUN / 'qa/inflight.json'
    if intent.exists(): raise ValueError('Unknown QA commit; no automatic retry')
    for row in plans:
        condition = row['question_id'],row['arm']
        if condition in done:
            if done[condition]['plan_sha256'] != fingerprint(row): raise ValueError('Committed QA plan changed')
            continue
        messages = messages_for(row); key = request_key(messages); reused = key in cache
        if not reused:
            if base.rpc(endpoint, '/tokenize', dict(text=row['evidence_text']))['tokens'] != row['evidence_tokens']:
                raise ValueError('Reader tokenizer mismatch')
            write_json(intent, dict(signature=sid, key=key, messages=messages, max_tokens=2048))
            result = {}; answer = ''; status = 'failed'; error = None; started = time.perf_counter()
            try:
                result = base.rpc(endpoint, '/complete', dict(messages=messages, max_tokens=2048))
                answer = parse_answer(base.require_complete(result)); status = 'ok'
            except Exception as exc: error = f'{type(exc).__name__}: {exc}'
            entry = dict(hypothesis=answer, status=status, error=error, result=result,
                         source='new_v2_call', source_condition=list(condition))
            base.append(RUN / 'qa/answer_calls.jsonl', dict(signature=sid, key=key, messages=messages,
                        max_tokens=2048, **entry, seconds=time.perf_counter()-started))
            write_json(RUN / 'qa/cache' / (key + '.json'), dict(signature=sid, key=key, entry=entry))
            cache[key] = entry
        entry = cache[key]
        pred = dict(**row, signature=sid, plan_sha256=fingerprint(row), request_key=key,
                    hypothesis=entry['hypothesis'], status=entry['status'], error=entry['error'],
                    reused=reused, cache_source=entry['source'], cache_condition=entry['source_condition'])
        base.append(dest, pred); done[condition] = pred
        if intent.exists(): intent.unlink()
        write_json(RUN / 'qa/status.json', dict(status='running', completed=len(done), expected=len(plans),
                    new_model_calls=len(list(read_jsonl(RUN / 'qa/answer_calls.jsonl'))) if (RUN / 'qa/answer_calls.jsonl').exists() else 0))
        print('V2_QA', len(done), '/', len(plans), *condition, 'cache' if reused else 'new', entry['status'], flush=True)
        if not entry['result']: raise RuntimeError('No inference result; failed condition recorded; stopped')
    if set(done) != {(r['question_id'],r['arm']) for r in plans}: raise ValueError('Unexpected QA conditions')
    write_json(RUN / 'qa/status.json', dict(status='completed', completed=len(done), expected=len(plans)))


def score():
    sid = verify(); base.verify_inputs(private=True)
    rows = list(read_jsonl(RUN / 'qa/predictions.jsonl')); plans = load(RUN / 'qa_plans.json')
    if len(rows) != len(plans) or {(r['question_id'],r['arm']) for r in rows} != {(r['question_id'],r['arm']) for r in plans}:
        raise ValueError('Complete QA required')
    refs = {r['question_id']:r for r in load(base.PLAN / 'private/references.json')}
    groups = defaultdict(list); scored = []
    for row in rows:
        if row['signature'] != sid: raise ValueError('Wrong QA identity')
        ref = refs[row['question_id']]
        f1 = base.native_f1(row['hypothesis'], ref['answer'], ref['category']) if row['status'] == 'ok' else 0.
        scored.append(dict(**row, official_native_f1=f1))
        for scope in ['overall','conversation:'+row['conversation_id'],'category:'+str(row['category'])]:
            groups[row['arm'],scope].append((f1,row))
    summaries = [dict(arm=a, scope=s, n=len(v), native_f1=statistics.mean(f for f,_ in v),
        mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for _,r in v),
        status_counts=dict(Counter(r['status'] for _,r in v))) for (a,s),v in sorted(groups.items())]
    by_arm = {a:{r['question_id']:r['official_native_f1'] for r in scored if r['arm']==a} for a in ARMS}
    old = load(PARENT_RUN / 'scored.json')
    by_arm['v1_group'] = {r['question_id']:r['official_native_f1'] for r in old if r['arm']=='learned_group_swap'}
    by_arm['fusion'] = {r['question_id']:r['official_native_f1'] for r in load(ROOT / 'runs/evidence_acquisition_v2/scored.json') if r['arm']=='fusion_2k'}
    comparisons = []
    for a,b in [('union_fp32','single_fp32'), ('union_fp32','wide_single_fp32'),
                ('group_fp32','single_fp32'), ('union_fp32','union_bf16'),
                ('reranker_mixed_fp32','reranker_mixed_bf16_serial'),
                ('union_fp32','reranker_mixed_fp32'), ('union_fp32','v1_group'), ('union_fp32','fusion')]:
        if set(by_arm[a]) != set(by_arm[b]): raise ValueError('Comparison coverage differs')
        delta = [by_arm[a][q]-by_arm[b][q] for q in by_arm[a]]
        comparisons.append(dict(arm=a, comparator=b, delta_points=100*statistics.mean(delta),
                                wins=sum(d>1e-9 for d in delta), ties=sum(abs(d)<=1e-9 for d in delta),
                                losses=sum(d<-1e-9 for d in delta)))
    calls = list(read_jsonl(RUN / 'qa/answer_calls.jsonl')) if (RUN / 'qa/answer_calls.jsonl').exists() else []
    write_json(RUN / 'scored.json', scored)
    write_json(RUN / 'report.json', dict(signature=sid, complete=True, scores=summaries,
        comparisons=comparisons, new_reader_calls=len(calls), reused_conditions=sum(r['reused'] for r in rows),
        selection=load(RUN / 'selection_complete.json'), scope=CONFIG['scope'], heldout_accessed=False))
    print(json.dumps([s for s in summaries if s['scope']=='overall']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze','verify','select','qa','score'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8018'); args = parser.parse_args()
    if args.mode == 'qa': qa(args.endpoint)
    else:
        result = globals()[args.mode]()
        if args.mode == 'verify': print('VERIFIED', result, flush=True)
