"""Complete two native LLMLingua-2 adaptations without changing the retained method."""
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import math
from pathlib import Path
import statistics
import time

from . import anchor_guard_controls_v1 as parent
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/llmlingua_budget_v1'
RUN = ROOT / 'runs/llmlingua_budget_v1'
MODEL = ROOT / 'models/llmlingua2-small'
load, sha, immutable, recomp = parent.load, parent.sha, parent.immutable, parent.recomp
BUDGETS = (256, 512, 1024, 2048)
NATIVE, RETRIEVED = 'llmlingua2_all20', 'qwen4x_llmlingua2'
OLD_METHODS = (parent.BASE, parent.PREVIOUS, parent.MAIN, parent.INDEPENDENT,
    'recomp_sentence', 'recomp_hotpot_sentence')
NEW_METHODS = (NATIVE, RETRIEVED)
METHODS = OLD_METHODS + NEW_METHODS
CONFIG = dict(version=1, questions=128, sources=64, budgets=list(BUDGETS),
    methods=list(METHODS), conditions=4096, historical_conditions=3072,
    maximum_reader_calls=1024, library='llmlingua', library_version='0.2.2',
    checkpoint='microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank',
    weight_sha256='22b9ecde52fec5c97e8c54a293be768727df95a81c6c8dccb03f262a50c58324',
    compressor_precision='GPU0 FP32/eager; TF32 disabled; batch4; native init seed42',
    retrieval_input_multiple=4, calibration_min_rate=.02, calibration_max_rate=.99,
    calibration_steps=8, maximum_api_calls_per_condition=10,
    force_tokens=['\n'], reader='Qwen3.5-9B', reader_variant='format_only',
    output_tokens=2048, retries=0, primary_budget=256, secondary_budget=512,
    new_training=False, method_modified=False, independent_confirmation=False,
    protected_data_accessed=False, automatic_followup=False)


def arm(method, budget):
    if method not in METHODS or budget not in BUDGETS:
        raise ValueError('Unregistered condition')
    return method + '__b' + str(budget)


def model_files():
    manifest = load(MODEL / 'source_manifest.json')
    wanted = ['config.json', 'model.safetensors', 'special_tokens_map.json',
        'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'README.md']
    listed = {v['Path']: v for v in manifest['Data']['Files']}
    if listed['model.safetensors']['Sha256'] != CONFIG['weight_sha256']:
        raise ValueError('Unexpected LLMLingua checkpoint')
    for name in wanted:
        item = listed[name]
        if (MODEL / name).stat().st_size != item['Size'] or sha(MODEL / name) != item['Sha256']:
            raise ValueError('Offline model changed: ' + name)
    if importlib.metadata.version('llmlingua') != CONFIG['library_version']:
        raise ValueError('Compression library version changed')
    base = Path(importlib.util.find_spec('llmlingua').origin).parent.parent
    runtime = load(PLAN / 'runtime_inspection.json')
    if not runtime['wheel_python_files_match']:
        raise ValueError('Inspected installation differs from pinned wheel')
    for name, digest in runtime['python_files'].items():
        if sha(base / name) != digest:
            raise ValueError('Installed library changed: ' + name)


def contexts_for(source, scores, method, target, counter):
    by = {p['source_id']: p for p in source['paragraphs']}
    order = list(by) if method == NATIVE else parent.packing.independent_seeds(source, scores, len(by))
    contexts, selected = [], []
    for pid in order:
        para = by[pid]
        contexts.append(recomp.render_ranges(source, {pid: [[0, len(para['text'])]]}))
        selected.append(pid)
        if method == RETRIEVED and counter.count('\n\n'.join(contexts)) >= 4 * target:
            break
    if method not in NEW_METHODS:
        raise ValueError('Unknown LLMLingua input pipeline')
    return contexts, selected


def calibrate(contexts, target, counter, compress):
    """Calibrate only exact reader length; never inspect quality or references."""
    full = '\n\n'.join(contexts)
    original_tokens = counter.count(full)
    if original_tokens <= target:
        return full, [dict(identity=True, input_tokens=original_tokens, output_tokens=original_tokens)]
    lo, hi = .02, .99
    traces, candidates = [], []
    def attempt(rate):
        result = compress(contexts, rate)
        text = result['compressed_prompt']
        n = counter.count(text)
        traces.append(dict(rate=rate, output_tokens=n, evidence_sha256=fingerprint(text),
            native_origin_tokens=result['origin_tokens'], native_compressed_tokens=result['compressed_tokens']))
        if text.strip() and n <= target:
            candidates.append((n, -len(traces), text))
        return n
    attempt(lo)
    rate = max(lo, min(hi, target / original_tokens))
    n = attempt(rate)
    if n <= target:
        lo = rate
    else:
        hi = rate
    for _ in range(CONFIG['calibration_steps']):
        rate = (lo + hi) / 2
        n = attempt(rate)
        if n <= target:
            lo = rate
        else:
            hi = rate
    if not candidates:
        raise ValueError('No legal nonempty native compression; stop without changing calibration')
    return max(candidates)[2], traces


def native_chunk_lengths(compressor, contexts):
    """Mirror the pinned API's force-token mapping before inspecting Dataset input."""
    token_map = {t: compressor.added_tokens[i] for i, t in enumerate(CONFIG['force_tokens'])
        if len(compressor.tokenizer.tokenize(t)) != 1}
    ends = {'.', '\n'} | {token_map[t] for t in ('.', '\n') if t in token_map}
    lengths = []
    for context in contexts:
        for original, replacement in token_map.items():
            context = context.replace(original, replacement)
        for chunk in compressor._PromptCompressor__chunk_context(context, chunk_end_tokens=ends):
            # TokenClfDataset uses tokenize plus CLS/SEP, then clips to max_seq_len.
            lengths.append(len(compressor.tokenizer.tokenize(chunk)) + 2)
    return lengths


class ForwardMemo:
    """Cache only a deterministic encoder's exact tensor input; keep native postprocessing."""
    def __init__(self, model, torch):
        self.model, self.torch, self.forward = model, torch, model.forward
        self.cache, self.calls, self.hits, self.tokens, self.seconds = {}, 0, 0, 0, 0.
        self.batch_lengths = []
    def __call__(self, *args, **kwargs):
        if args or set(kwargs) != {'input_ids', 'attention_mask'} or self.model.training:
            raise ValueError('Unsupported encoder memo contract')
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        key = fingerprint([ids.detach().cpu().tolist(), mask.detach().cpu().tolist()])
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.torch.cuda.synchronize()
        begin = time.perf_counter()
        value = self.forward(**kwargs)
        self.torch.cuda.synchronize()
        self.seconds += time.perf_counter() - begin
        self.calls += 1
        self.tokens += int(mask.sum())
        self.batch_lengths.append(dict(batch=int(ids.shape[0]), sequence=int(ids.shape[1]), active=int(mask.sum())))
        self.cache[key] = value
        return value


def freeze():
    previous = parent.verify()
    model_files()
    for name, digest in load(PLAN / 'preparation.json')['server_source_files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Server score source differs from completed snapshot: ' + name)
    names = ['reader_memory/llmlingua_budget_v1.py', 'tools/prepare_llmlingua_budget_v1.py',
        'tools/llmlingua_budget_v1_remote.py', 'scripts/llmlingua_budget_v1_queue.py',
        'tests/test_llmlingua_budget_v1.py', 'research/anchor_guard_controls_v1/manifest.json',
        'models/llmlingua2-small/source_manifest.json']
    names += list(load(PLAN / 'preparation.json')['server_source_files'])
    names += ['research/llmlingua_budget_v1/' + n for n in ['PROTOCOL.md', 'runtime_inspection.json',
        'preparation.json', 'public/source_inputs.json', 'public/base_scores.json',
        'public/baseline_plans.json', 'private/replay_cache.json', 'private/references.json']]
    value = dict(config=CONFIG, parent_signature=previous, files={n: sha(ROOT / n) for n in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_LLMLINGUA_BUDGET', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Frozen baseline identity changed')
    if parent.verify() != value['parent_signature']:
        raise ValueError('Retained method changed')
    for name, digest in value['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return value['signature']


def preflight():
    sid = verify()
    sources = load(PLAN / 'public/source_inputs.json')
    plans = load(PLAN / 'public/baseline_plans.json')
    if len(sources) != 128 or len({s['native_id'] for s in sources}) != 64:
        raise ValueError('Panel changed')
    expected = {(s['question_id'], arm(m, b)) for s in sources for m in OLD_METHODS for b in BUDGETS}
    if len(plans) != 3072 or {(x['question_id'], x['arm']) for x in plans} != expected:
        raise ValueError('Historical matrix incomplete')
    if sha(ROOT / 'models/Qwen3.5-9B/.verified.json') != load(PLAN / 'preparation.json')['reader_marker_sha256']:
        raise ValueError('Reader identity changed')
    # Expose the legacy ledger's required metadata key before loading any GPU model.
    if "meta['plans_sha256']" not in inspect.getsource(recomp.ledger.qa):
        raise ValueError('Unexpected legacy QA interface')
    immutable(RUN / 'preflight.json', dict(signature=sid, passed=True, questions=128,
        historical_conditions=3072, new_conditions_cap=1024, expected_metadata_key='plans_sha256'))
    print('LLMLINGUA_PREFLIGHT_OK', flush=True)


def select():
    sid = verify()
    if (RUN / 'selection_inflight.json').exists() or (RUN / 'scoring_complete.json').exists():
        raise ValueError('Existing/unknown compression; no automatic repetition')
    import torch
    from llmlingua import PromptCompressor
    from tools.prepare_replacement_learning_v1 import ExactCounter
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    compressor = PromptCompressor(model_name=str(MODEL), device_map='cuda', use_llmlingua2=True,
        llmlingua2_config=dict(max_batch_size=4), model_config=dict(local_files_only=True,
            trust_remote_code=False, attn_implementation='eager', torch_dtype=torch.float32))
    compressor.model_name = CONFIG['checkpoint']
    compressor.model.float().eval()
    if compressor.model.config.model_type != 'bert' or any(v.dtype != torch.float32 for v in compressor.model.parameters()):
        raise ValueError('Unexpected official BERT model or precision')
    memo = ForwardMemo(compressor.model, torch)
    compressor.model.forward = memo
    api_calls = 0
    def compress(contexts, rate):
        nonlocal api_calls
        api_calls += 1
        return compressor.compress_prompt_llmlingua2(contexts, rate=rate,
            use_context_level_filter=False, use_token_level_filter=True, token_to_word='mean',
            force_tokens=['\n'], force_reserve_digit=False, drop_consecutive=False,
            chunk_end_tokens=['.', '\n'])
    # Verify cache equivalence with synthetic public text only, before real material.
    toy = ['Alpha study began in 2001. Bob lives in Paris. Carol works in Rome.\nThe team met on Monday.']
    first = compress(toy, .6)
    second = compress(toy, .6)
    memo.cache.clear()
    third = compress(toy, .6)
    if first != second or first != third or memo.hits < 1:
        raise ValueError('Native API memoization did not reproduce; stop')
    immutable(RUN / 'compatibility.json', dict(signature=sid, passed=True,
        cached_result_identical=True, fresh_result_identical=True, synthetic_only=True,
        actual_forward_calls=memo.calls, actual_input_tokens=memo.tokens))
    memo.cache.clear()
    start_calls, start_tokens, start_seconds, start_api = memo.calls, memo.tokens, memo.seconds, api_calls
    counter = ExactCounter()
    scores = load(PLAN / 'public/base_scores.json')
    base_plans = load(PLAN / 'public/baseline_plans.json')
    targets = {(x['question_id'], x['evidence_budget']): x for x in base_plans if x['method'] == parent.BASE}
    sources = load(PLAN / 'public/source_inputs.json')
    for index, source in enumerate(sources, 1):
        q = source['question_id']
        target_file = RUN / 'selected' / (q.replace(':', '_') + '.json')
        if target_file.exists():
            raise ValueError('Existing partial selection; no automatic rerun')
        write_json(RUN / 'selection_inflight.json', dict(signature=sid, question_id=q))
        selected, traces = [], []
        begin = time.perf_counter()
        calls_before, tokens_before, api_before = memo.calls, memo.tokens, api_calls
        for budget in BUDGETS:
            target = targets[q, budget]['evidence_tokens']
            for method in NEW_METHODS:
                contexts, ids = contexts_for(source, scores[q], method, target, counter)
                text, trace = calibrate(contexts, target, counter, compress)
                selected.append(dict(**{k: source[k] for k in ('question_id', 'native_id', 'mode', 'question')},
                    method=method, arm=arm(method, budget), evidence_budget=budget, reader_variant='format_only',
                    evidence_text=text, evidence_tokens=counter.count(text), evidence_sha256=fingerprint(text),
                    selected_ids=['compressed:' + x for x in ids], selected_ranges={}, selected_spans=[],
                    fully_preserved_sources=[], compressed_input_source_ids=ids,
                    private_inputs_read=False, diagnostic_only=False))
                # Record potential Dataset length clipping inherent to the official package.
                chunk_lengths = native_chunk_lengths(compressor, contexts)
                traces.append(dict(method=method, budget=budget, target_tokens=target,
                    input_source_ids=ids, input_sha256=fingerprint(contexts),
                    input_reader_tokens=counter.count('\n\n'.join(contexts)), calibration=trace,
                    native_chunk_lengths=chunk_lengths, native_chunks_exceeding_512=sum(n > 512 for n in chunk_lengths)))
        immutable(target_file, dict(signature=sid, source_sha256=fingerprint(source), plans=selected, traces=traces,
            costs=dict(actual_forward_calls=memo.calls-calls_before, actual_input_tokens=memo.tokens-tokens_before,
                api_calls=api_calls-api_before, seconds=time.perf_counter()-begin)))
        (RUN / 'selection_inflight.json').unlink()
        memo.cache.clear()
        write_json(RUN / 'selection_status.json', dict(signature=sid, completed=index, expected=128,
            actual_forward_calls=memo.calls-start_calls, actual_input_tokens=memo.tokens-start_tokens,
            seconds=memo.seconds-start_seconds))
        print('LLMLINGUA_SELECT', index, '/128', flush=True)
    immutable(RUN / 'scoring_complete.json', dict(signature=sid, actual_forward_calls=memo.calls-start_calls,
        actual_input_tokens=memo.tokens-start_tokens, forward_seconds=memo.seconds-start_seconds,
        api_calls=api_calls-start_api, cached_forward_hits=memo.hits,
        native_batch_lengths=memo.batch_lengths, compatibility_included_in_batch_lengths=True,
        shared_encoder_cache=True, upstream_qwen_cost_is_not_zero=True))
    compressor.model.forward = memo.forward
    memo.cache.clear()
    del compressor, memo
    gc.collect()
    torch.cuda.empty_cache()


def replay_cache():
    marker = load(PLAN / 'preparation.json')['reader_marker_sha256']
    cache = {}
    for item in load(PLAN / 'private/replay_cache.json'):
        recomp.interface.parent.check_cache_entry(item, marker)
        if item['key'] in cache:
            raise ValueError('Duplicate cached request')
        cache[item['key']] = item['entry']
    return cache


def assemble():
    sid = verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    plans = load(PLAN / 'public/baseline_plans.json')
    for source in load(PLAN / 'public/source_inputs.json'):
        record = load(RUN / 'selected' / (source['question_id'].replace(':', '_') + '.json'))
        if record['signature'] != sid or record['source_sha256'] != fingerprint(source):
            raise ValueError('Compression provenance changed')
        plans.extend(record['plans'])
    expected = {(s['question_id'], arm(m,b)) for s in load(PLAN / 'public/source_inputs.json') for m in METHODS for b in BUDGETS}
    if len(plans) != 4096 or {(x['question_id'],x['arm']) for x in plans} != expected:
        raise ValueError('Incomplete full matrix')
    targets = {(x['question_id'],x['evidence_budget']):x['evidence_tokens'] for x in plans if x['method']==parent.BASE}
    counter, lengths = ExactCounter(), []
    for x in plans:
        n = counter.count(x['evidence_text'])
        if n != x['evidence_tokens'] or n > x['evidence_budget'] or fingerprint(x['evidence_text']) != x['evidence_sha256']:
            raise ValueError('Evidence identity or budget changed')
        target = targets[x['question_id'],x['evidence_budget']]
        if x['method'] in NEW_METHODS and (not x['evidence_text'].strip() or n > target):
            raise ValueError('New baseline exceeds per-question actual target')
        lengths.append(dict(question_id=x['question_id'],method=x['method'],mode=x['mode'],
            budget=x['evidence_budget'],tokens=n,target=target,ratio=n/target))
    # Private cache consulted only after complete public selections and length validation.
    cache = replay_cache()
    marker = load(PLAN/'preparation.json')['reader_marker_sha256']
    tok = AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    keys, maximum = set(), 0
    for x in plans:
        messages = recomp.messages_for(x)
        key = recomp.interface.parent.request_key(messages,marker)
        keys.add(key)
        if x['method'] not in NEW_METHODS and key not in cache:
            raise ValueError('Historical output not reusable')
        prompt = tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        maximum=max(maximum,counter.count(prompt))
    new = len(keys-set(cache))
    if new>1024 or maximum+2048>32768:
        raise ValueError('Reader call cap or prompt capacity exceeded')
    immutable(RUN/'qa_plans.json',plans)
    immutable(RUN/'length_audit.json',lengths)
    immutable(RUN/'selection_complete.json',dict(signature=sid,plans_sha256=sha(RUN/'qa_plans.json'),
        conditions=4096,uncached_requests=new,maximum_full_prompt_tokens=maximum,before_any_new_answer=True))
    print('LLMLINGUA_ASSEMBLY_OK',new,'new reader calls',flush=True)


def qa(endpoint):
    sid=verify()
    ledger=recomp.ledger
    old=ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for
    ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for=RUN,lambda:sid,replay_cache,recomp.messages_for
    try:ledger.qa(endpoint)
    finally:ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for=old


def score():
    sid=verify()
    rows=recomp.interface.parent.score_rows(load(RUN/'qa_plans.json'),list(read_jsonl(RUN/'qa/predictions.jsonl')),
        load(PLAN/'private/references.json'),sid)
    summaries,comparisons=[],[]
    for budget in BUDGETS:
        for scope in ('overall','multi','single'):
            def by(method):
                return {r['question_id']:r for r in rows if r['method']==method and r['evidence_budget']==budget
                    and (scope=='overall' or r['mode']==scope)}
            for method in METHODS:
                group=list(by(method).values())
                summaries.append(dict(method=method,budget=budget,scope=scope,n=len(group),
                    f1_points=100*statistics.mean(r['local_native_f1'] for r in group),
                    mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in group),
                    statuses=dict(Counter(r['status'] for r in group))))
            for other in METHODS:
                if other!=parent.MAIN:
                    comparisons.append(dict(method=parent.MAIN,comparator=other,budget=budget,scope=scope,
                        **recomp.interface.paired(by(parent.MAIN),by(other))))
    cache=replay_cache()
    for r in rows:
        if r['method'] not in NEW_METHODS:
            old=cache[r['request_key']]
            if not r['reused'] or any(r[k]!=old[k] for k in ('hypothesis','status','error')):
                raise ValueError('Historical results changed')
    calls_path=RUN/'qa/answer_calls.jsonl'
    calls=list(read_jsonl(calls_path)) if calls_path.exists() else []
    if len(calls)!=load(RUN/'selection_complete.json')['uncached_requests']:
        raise ValueError('New-call accounting changed')
    immutable(RUN/'scored.json',rows)
    report=dict(signature=sid,complete=True,scores=summaries,comparisons=comparisons,
        new_reader_calls=len(calls),reused_conditions=sum(r['reused'] for r in rows),
        encoder_cost=load(RUN/'scoring_complete.json'),reader_seconds=sum(c['seconds'] for c in calls),
        reader_cost={k:sum(c['result'].get('usage',{}).get(k,0) for c in calls) for k in ('prompt_tokens','completion_tokens')},
        method_modified=False,paper_ready=False,independent_confirmation=False,automatic_followup=False)
    immutable(RUN/'report.json',report)
    print('LLMLINGUA_COMPLETE',[s for s in summaries if s['scope']=='overall'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','verify','preflight','select','assemble','qa','score'])
    p.add_argument('--endpoint',default='http://127.0.0.1:8036')
    a=p.parse_args()
    result=qa(a.endpoint) if a.mode=='qa' else globals()[a.mode]()
    if result:print(result,flush=True)
