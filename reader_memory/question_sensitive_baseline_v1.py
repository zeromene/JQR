"""Frozen BGE contrast preflight. Inference sees only questions and native text."""
from pathlib import Path
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/question_sensitive_v1'
RUN = ROOT / 'runs/question_sensitive_baseline_v1'
CONFIG = dict(version='question-sensitive-baseline-v1', model='bge-reranker-v2-m3',
              dtype='float16', batch_size=4, maximum_tokens=8192, budgets=[256, 512],
              query_views=['a', 'b', 'paraphrase_a'], packing='score descending, native output order, whole native sentences',
              selection_threshold=None, truncation=False, training=False)


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write_new(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def units(source):
    return [dict(id=f'{p}:{s}', paragraph=p, sentence=s, title=title, text=text)
            for p, (title, ss) in enumerate(zip(source['context']['title'], source['context']['sentences']))
            for s, text in enumerate(ss) if text.strip()]


def render(us, selected):
    pieces = []
    previous = None
    for i in sorted(selected):
        u = us[i]
        if u['paragraph'] != previous:
            pieces.append(f'[{u["paragraph"]}] {u["title"]}')
            previous = u['paragraph']
        pieces.append(f'[{u["id"]}] {u["text"].strip()}')
    return '\n'.join(pieces)


def pack(us, scores, budget, tokenizer):
    selected = []
    for i in sorted(range(len(us)), key=lambda i: (-scores[i], i)):
        if len(tokenizer.encode(render(us, selected + [i]), add_special_tokens=False).ids) <= budget:
            selected.append(i)
    text = render(us, selected)
    return dict(ids=[us[i]['id'] for i in sorted(selected)], text=text,
                tokens=len(tokenizer.encode(text, add_special_tokens=False).ids))


def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tokenizers import Tokenizer
    manifest = load(BASE / 'baseline_manifest.json')
    assert manifest['config'] == CONFIG
    for name, expected in manifest['inference_files'].items():
        assert sha(ROOT / name) == expected, name
    # No candidate review or native answer file is read in this process.
    sources = load(BASE / 'public/contrast_inputs.json')
    torch.set_num_threads(4)
    torch.manual_seed(30)
    torch.cuda.set_per_process_memory_fraction(0.16)
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models' / CONFIG['model']), local_files_only=True)
    reader_tok = Tokenizer.from_file(str(ROOT / 'models/Qwen3.5-9B/tokenizer.json'))
    reader_tok.no_truncation(); reader_tok.no_padding()
    model = AutoModelForSequenceClassification.from_pretrained(str(ROOT / 'models' / CONFIG['model']),
        local_files_only=True, torch_dtype=torch.float16).cuda().eval()
    RUN.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    total_pairs = 0
    maximum_tokens = 0
    for k, source in enumerate(sources):
        us = units(source)
        for view in CONFIG['query_views']:
            dest = RUN / 'scores' / f'{source["index"]:02d}_{view}.json'
            if dest.exists():
                raise RuntimeError('Already scored; refuse duplicate baseline run')
            question = source['questions'][view]
            pairs = [(question, u['title'] + '\n' + u['text']) for u in us]
            scores = []
            lengths = [len(x) for x in tok(pairs, padding=False, truncation=False)['input_ids']]
            assert max(lengths) <= CONFIG['maximum_tokens'], (source['index'], max(lengths))
            maximum_tokens = max(maximum_tokens, max(lengths))
            begin = time.perf_counter()
            with torch.inference_mode():
                for j in range(0, len(pairs), CONFIG['batch_size']):
                    inputs = tok(pairs[j:j+CONFIG['batch_size']], return_tensors='pt',
                                 padding=True, truncation=False).to('cuda')
                    logits = model(**inputs).logits.view(-1).float().cpu().tolist()
                    scores.extend(logits)
            assert len(scores) == len(us) and all(torch.isfinite(torch.tensor(scores)))
            plans = {str(b): pack(us, scores, b, reader_tok) for b in CONFIG['budgets']}
            write_new(dest, dict(index=source['index'], native_id=source['native_id'], view=view,
                question=question, sentence_ids=[u['id'] for u in us], scores=scores,
                seconds=time.perf_counter()-begin, plans=plans, maximum_pair_tokens=max(lengths)))
            total_pairs += len(pairs)
        print(json.dumps(dict(completed_sources=k+1, total_sources=len(sources),
            elapsed_seconds=round(time.perf_counter()-start, 2))), flush=True)
    write_new(RUN / 'completion.json', dict(status='completed', sources=len(sources),
        questions=3*len(sources), scored_question_sentence_pairs=total_pairs,
        maximum_pair_tokens=maximum_tokens, elapsed_seconds=time.perf_counter()-start,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        model_class=type(model).__name__, parameters=sum(p.numel() for p in model.parameters()),
        answer_or_review_labels_read=False, new_qa_reader_calls=0, new_training_updates=0))


if __name__ == '__main__':
    run()
