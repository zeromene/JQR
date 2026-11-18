"""Bounded, label-free scoring of frozen natural parent questions."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import importlib.metadata as md
import json
import math
import os
import subprocess
import time
import traceback

from .question_sensitive_baseline_v1 import load, sha, write_new, units

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/native_chain_bge_v1'
RUN = ROOT / 'runs/native_chain_bge_v1'
CONFIG = dict(model='bge-reranker-v2-m3', dtype='float16', batch_size=4,
              maximum_tokens=8192, truncation=False, allocator_cap_mib=4096,
              headroom_mib=768, wait_seconds=7200, training=False, qa=False)


def state(status, **extra):
    data = dict(status=status, time=datetime.now(timezone.utc).isoformat(), **extra)
    temp = RUN / 'status.tmp'
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temp, RUN / 'status.json')
    print(json.dumps(data), flush=True)


def free_memory():
    return int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip())


def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    start = time.perf_counter()
    RUN.mkdir(parents=True, exist_ok=True)
    assert not list((RUN/'scores').glob('*.json')), 'Refuse re-inference after partial results'
    manifest = load(BASE/'manifest.json')
    assert manifest['config'] == CONFIG
    for name, expected in manifest['files'].items():
        assert sha(ROOT/name) == expected, ('Hash mismatch', name)
    sources = load(BASE/'public/inputs.json')
    assert len(sources) == 64
    assert all(set(s) == {'index', 'question', 'context'} for s in sources)
    assert [s['index'] for s in sources] == list(range(64))
    torch.set_num_threads(4)
    torch.manual_seed(30)
    tok = AutoTokenizer.from_pretrained(str(ROOT/'models'/CONFIG['model']), local_files_only=True)
    # All inputs checked before model allocation, rather than discovering an
    # oversized candidate halfway through a completed panel.
    cached = []
    for source in sources:
        us = units(source)
        pairs = [(source['question'], u['title']+'\n'+u['text']) for u in us]
        lengths = [len(ids) for ids in tok(pairs, padding=False, truncation=False)['input_ids']]
        assert max(lengths) <= CONFIG['maximum_tokens'], (source['index'], max(lengths))
        cached.append((us, pairs, lengths))
    write_new(RUN/'preflight.json', dict(sources=64, pairs=sum(len(x[0]) for x in cached),
        maximum_tokens=max(max(x[2]) for x in cached), lengths_by_source=[x[2] for x in cached],
        labels_read=False, truncation=False, manifest_sha256=sha(BASE/'manifest.json')))
    with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
        deadline = time.monotonic()+CONFIG['wait_seconds']
        while True:
            acquired = False
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            free = free_memory()
            if acquired and free >= CONFIG['allocator_cap_mib']+CONFIG['headroom_mib']:
                break
            if acquired:
                fcntl.flock(lock, fcntl.LOCK_UN)
            assert time.monotonic() < deadline, 'GPU admission timed out'
            state('waiting_for_memory', free_mib=free, lock_available=acquired)
            time.sleep(20)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(CONFIG['allocator_cap_mib']*2**20/total)
        state('loading', free_mib_at_admission=free)
        model = AutoModelForSequenceClassification.from_pretrained(str(ROOT/'models'/CONFIG['model']),
            local_files_only=True, torch_dtype=torch.float16).cuda().eval()
        inference_start = time.perf_counter()
        for source, (us, pairs, lengths) in zip(sources, cached):
            begin = time.perf_counter()
            scores = []
            with torch.inference_mode():
                for i in range(0, len(pairs), CONFIG['batch_size']):
                    inputs = tok(pairs[i:i+CONFIG['batch_size']], return_tensors='pt', padding=True,
                                 truncation=False).to('cuda')
                    scores.extend(model(**inputs).logits.view(-1).float().cpu().tolist())
            assert len(scores) == len(us) and all(math.isfinite(s) for s in scores)
            write_new(RUN/'scores'/f'{source["index"]:02d}.json', dict(index=source['index'],
                sentence_ids=[u['id'] for u in us], scores=scores, maximum_pair_tokens=max(lengths),
                seconds=time.perf_counter()-begin))
            state('running', sources_completed=source['index']+1, total_sources=64,
                  inference_seconds=time.perf_counter()-inference_start)
        done = dict(status='completed', sources=64, pairs=sum(len(x[0]) for x in cached),
            time=datetime.now(timezone.utc).isoformat(), manifest_sha256=sha(BASE/'manifest.json'),
            total_seconds=time.perf_counter()-start, inference_seconds=time.perf_counter()-inference_start,
            maximum_pair_tokens=max(max(x[2]) for x in cached), free_mib_at_admission=free,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            model_class=type(model).__name__, parameters=sum(p.numel() for p in model.parameters()),
            gpu=torch.cuda.get_device_name(0), versions={n:md.version(n) for n in ['torch','transformers','tokenizers']},
            private_files_read=False, training_updates=0, qa_calls=0)
        write_new(RUN/'completion.json', done)
        state('completed', sources_completed=64)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        RUN.mkdir(parents=True, exist_ok=True)
        state('failed', error=repr(exc), traceback=traceback.format_exc())
        raise
