"""One finite training-only capability probe; no reader calls or model training."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .schema import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/conditional_rerank_probe_v1'
RUN = ROOT / 'runs/conditional_rerank_probe_v1'
TRAIN = ROOT / 'research/path_retrieval_pilot_v1'
MODES = ('question', 'conditioned')
CONFIG = dict(version=1, sample={2: 24, 3: 8}, minimum_pairs=24,
    minimum_accuracy=0.8, minimum_net_correct=4, maximum_forwards=128,
    prompt_separator='\n\nPreviously retrieved evidence:\n', seed=30,
    negative_word_ratio=[0.75, 1.25], bm25_k1=1.2, bm25_b=0.75,
    model_revision='22e683669bc0f0bd69640a1354a6d0aebcfeede5',
    precision='BF16 SDPA backbone, same-forward FP32 head, no TF32',
    ranker_max_tokens=8192, batch=1, ties='incorrect',
    training=False, reader_calls=0, protected_data_accessed=False,
    automatic_followup=False)
# JSON object keys must survive serialization identically.
CONFIG['sample'] = {str(k): v for k, v in CONFIG['sample'].items()}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def immutable(path, value):
    if path.exists():
        if load(path) != value:
            raise ValueError('Refuse changed existing artifact: ' + str(path))
    else:
        write_json(path, value)


def words(text):
    return re.findall(r'\w+', text.lower())


def document(p):
    return p['title'] + ': ' + p['text']


def bm25_scores(question, paragraphs):
    docs = [words(document(p)) for p in paragraphs]
    query = set(words(question))
    avg = sum(map(len, docs)) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    result = []
    for d in docs:
        tf = Counter(d)
        result.append(sum(math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
            * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
            for t in query if tf[t]))
    return result


def build(inputs, references):
    # Restrict to the already assigned training file; never consult other splits.
    pool = [s for s in inputs if s['mode'] == 'multi']
    sampled = []
    for hops, count in CONFIG['sample'].items():
        group = [s for s in pool if len(references[s['question_id']]['support_indices']) == int(hops)]
        group.sort(key=lambda s: hashlib.sha256(('conditional-rerank-v1|' + s['native_id']).encode()).hexdigest())
        if len(group) < count:
            raise ValueError('Insufficient fixed training stratum')
        sampled.extend(group[:count])
    cases, labels, excluded = [], {}, []
    for s in sampled:
        qid = s['question_id']
        chain = references[qid]['support_indices']
        paragraphs = s['paragraphs']
        if len(chain) != len(set(chain)) or not all(0 <= i < len(paragraphs) for i in chain):
            raise ValueError('Invalid ordered support')
        positive = chain[-1]
        length = len(words(document(paragraphs[positive])))
        eligible = [i for i, p in enumerate(paragraphs) if i not in chain
            and 0.75 * length <= len(words(document(p))) <= 1.25 * length
            and document(p) != document(paragraphs[positive])]
        if not eligible:
            excluded.append(dict(question_id=qid, reason='no_non_support_within_word_ratio', positive_words=length))
            continue
        lexical = bm25_scores(s['question'], paragraphs)
        negative = min(eligible, key=lambda i: (-lexical[i], paragraphs[i]['source_id']))
        order = sorted([positive, negative], key=lambda i: fingerprint([qid, document(paragraphs[i])]))
        candidates = [dict(candidate_id=paragraphs[i]['source_id'], document=document(paragraphs[i])) for i in order]
        evidence = [document(paragraphs[i]) for i in chain[:-1]]
        cases.append(dict(question_id=qid, question=s['question'], evidence=evidence, candidates=candidates))
        labels[qid] = dict(positive_id=paragraphs[positive]['source_id'],
            negative_id=paragraphs[negative]['source_id'], support_indices=chain,
            hop_count=len(chain), positive_words=length,
            negative_words=len(words(document(paragraphs[negative]))),
            negative_bm25=lexical[negative], eligible_negatives=len(eligible))
    return cases, labels, dict(sampled_ids=[s['question_id'] for s in sampled],
        usable=len(cases), excluded=excluded, no_refill=True)


def prepare():
    cases, labels, info = build(load(TRAIN / 'public/train_inputs.json'), load(TRAIN / 'private/train_references.json'))
    immutable(PLAN / 'public/cases.json', cases)
    immutable(PLAN / 'private/labels.json', labels)
    immutable(PLAN / 'preparation.json', info)
    print(json.dumps(info, ensure_ascii=False, indent=2))


def query(case, mode):
    if mode == 'question':
        return case['question']
    if mode == 'conditioned':
        return case['question'] + CONFIG['prompt_separator'] + '\n\n'.join(case['evidence'])
    raise ValueError('Unknown mode')


def freeze():
    from .reranker_budget_v1 import model_files
    model_files()
    names = ['reader_memory/conditional_rerank_probe_v1.py', 'scripts/conditional_rerank_probe_v1_queue.py',
        'tools/conditional_rerank_probe_v1_remote.py', 'tests/test_conditional_rerank_probe_v1.py',
        'reader_memory/reranker_budget_v1.py', 'reader_memory/replacement_precision_v2.py',
        'reader_memory/mainline_reranker_v1.py', 'reader_memory/schema.py',
        'research/conditional_rerank_probe_v1/PROTOCOL.md',
        'research/conditional_rerank_probe_v1/preparation.json',
        'research/conditional_rerank_probe_v1/public/cases.json',
        'research/conditional_rerank_probe_v1/private/labels.json',
        'research/path_retrieval_pilot_v1/public/train_inputs.json',
        'research/path_retrieval_pilot_v1/private/train_references.json',
        'models/Qwen3-Reranker-4B/artifact_manifest.json', 'models/Qwen3-Reranker-4B/.verified.json']
    value = dict(config=CONFIG, files={n: sha(ROOT / n) for n in names})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Frozen identity changed')
    for name, digest in value['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Input changed: ' + name)
    return value['signature']


def preflight():
    sid = verify()
    from transformers import AutoTokenizer
    from .reranker_budget_v1 import MODEL, rank_input_ids
    cases, labels, info = build(load(TRAIN / 'public/train_inputs.json'), load(TRAIN / 'private/train_references.json'))
    if (cases != load(PLAN / 'public/cases.json') or labels != load(PLAN / 'private/labels.json')
            or info != load(PLAN / 'preparation.json')):
        raise ValueError('Training construction did not reproduce')
    if len(cases) < CONFIG['minimum_pairs'] or len(cases) * 4 > CONFIG['maximum_forwards']:
        raise ValueError('Fixed construction resource gate failed; stop without resampling')
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    lengths = [len(rank_input_ids(tok, query(c, mode), d['document'])) for c in cases
        for mode in MODES for d in c['candidates']]
    immutable(RUN / 'preflight.json', dict(signature=sid, cases=len(cases),
        forwards=len(lengths), maximum_input_tokens=max(lengths), total_input_tokens=sum(lengths)))
    print('PREFLIGHT_OK', len(cases), len(lengths), flush=True)


def run():
    sid = verify()
    if not (RUN / 'preflight.json').is_file():
        raise ValueError('Preflight missing')
    if (RUN / 'scores.json').exists() or (RUN / 'inflight.json').exists():
        raise ValueError('Existing/unknown scores; no automatic repetition')
    from .reranker_budget_v1 import MODEL, rank_input_ids
    from .replacement_precision_v2 import two_head_scores
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)
    torch.manual_seed(30)
    torch.cuda.manual_seed_all(30)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    load_seconds = time.perf_counter() - started
    costs = dict(forward_calls=0, input_tokens=0, seconds=0.)
    rows = []
    # No label or reference file is opened by this scoring loop.
    for c in load(PLAN / 'public/cases.json'):
        for mode in MODES:
            for d in c['candidates']:
                ids = rank_input_ids(tok, query(c, mode), d['document'])
                write_json(RUN / 'inflight.json', dict(signature=sid, question_id=c['question_id'],
                    mode=mode, candidate_id=d['candidate_id']))
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.inference_mode():
                    pair = {k: float(v) for k, v in two_head_scores(torch, tok, model, ids).items()}
                torch.cuda.synchronize()
                seconds = time.perf_counter() - begin
                if not all(math.isfinite(v) for v in pair.values()):
                    raise ValueError('Nonfinite ranker score')
                rows.append(dict(question_id=c['question_id'], mode=mode, candidate_id=d['candidate_id'],
                    scores=pair, input_ids_sha256=fingerprint(ids), input_tokens=len(ids), seconds=seconds))
                costs['forward_calls'] += 1
                costs['input_tokens'] += len(ids)
                costs['seconds'] += seconds
                write_json(RUN / 'progress.json', dict(signature=sid, completed=costs['forward_calls'],
                    expected=load(RUN / 'preflight.json')['forwards']))
        print('PAIR_COMPLETE', len(rows) // 4, flush=True)
    immutable(RUN / 'scores.json', dict(signature=sid, rows=rows, costs=costs, load_seconds=load_seconds,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), private_labels_in_forward=False))
    (RUN / 'inflight.json').unlink()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def aggregate(rows, labels):
    paired = []
    for qid, ref in labels.items():
        margins = {}
        for mode in MODES:
            found = [r for r in rows if r['question_id'] == qid and r['mode'] == mode]
            if len(found) != 2 or {r['candidate_id'] for r in found} != {ref['positive_id'], ref['negative_id']}:
                raise ValueError('Missing/duplicate/unexpected paired scores')
            values = {r['candidate_id']: r['scores']['fp32'] for r in found}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Nonfinite saved score')
            margins[mode] = values[ref['positive_id']] - values[ref['negative_id']]
        paired.append(dict(question_id=qid, hop_count=ref['hop_count'], margins=margins,
            correct={m: margins[m] > 0 for m in MODES}))
    if len(rows) != len(labels) * 4:
        raise ValueError('Extra scores')
    def stats(items):
        correct = {m: sum(p['correct'][m] for p in items) for m in MODES}
        return dict(n=len(items), correct=correct,
            accuracy={m: correct[m] / len(items) if items else None for m in MODES},
            net_correct=correct['conditioned'] - correct['question'],
            wins=sum(p['correct']['conditioned'] and not p['correct']['question'] for p in items),
            losses=sum(p['correct']['question'] and not p['correct']['conditioned'] for p in items),
            ties=sum(p['correct']['question'] == p['correct']['conditioned'] for p in items),
            score_ties={m: sum(p['margins'][m] == 0 for p in items) for m in MODES})
    overall = stats(paired)
    passed = (len(paired) >= CONFIG['minimum_pairs']
        and overall['accuracy']['conditioned'] >= CONFIG['minimum_accuracy']
        and overall['net_correct'] >= CONFIG['minimum_net_correct'])
    return dict(overall=overall, by_hops={str(h): stats([p for p in paired if p['hop_count'] == h]) for h in (2, 3)},
        passed_resource_gate=passed, decision='eligible_for_discussion_only' if passed else 'close_semantic_background_branch',
        paired=paired, automatic_followup=False, main_task_claim=False, independent_confirmation=False)


def summarize():
    sid = verify()
    saved = load(RUN / 'scores.json')
    if saved['signature'] != sid:
        raise ValueError('Score identity mismatch')
    value = aggregate(saved['rows'], load(PLAN / 'private/labels.json'))
    value.update(signature=sid, costs=saved['costs'], scores_sha256=sha(RUN / 'scores.json'))
    immutable(RUN / 'summary.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'paired'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'freeze', 'preflight', 'run', 'summarize'])
    globals()[parser.parse_args().mode]()
