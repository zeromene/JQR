"""BF16 CPU weight offload, GPU computation, for the complete 9B comparison."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import threading
import time
import traceback

from . import length_repack_qa_v1 as parent

ROOT = parent.ROOT
BASE = ROOT / 'research/length_repack_offload_v1'
RUN = ROOT / 'runs/length_repack_offload_v1'
CONFIG = dict(reader='Qwen3.5-9B', gpu_cap_mib=10500, resident_weights_mib=7800,
              admission_mib=11700, cpu_available_mib=32000, max_output=128,
              smoke_count=8, seed=30, wait_seconds=1800)


def state(status, **fields):
    value = dict(status=status, time=datetime.now(timezone.utc).isoformat(), **fields)
    temp = RUN / 'status.tmp'
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, RUN / 'status.json')
    print(json.dumps(value), flush=True)


def verify():
    parent.verify(models=True)
    manifest = parent.load(BASE / 'manifest.json')
    assert manifest['config'] == CONFIG
    assert parent.sha(parent.BASE / 'manifest.json') == manifest['parent_manifest_sha256']
    for name, digest in manifest['files'].items():
        assert parent.sha(ROOT / name) == digest, name
    for name, digest in parent.load(parent.BASE / 'manifest.json')['frozen_local'].items():
        # These parent artifacts are local only, unless present on the server.
        if (parent.BASE / name).exists():
            assert parent.sha(parent.BASE / name) == digest, name
    return parent.sha(BASE / 'manifest.json'), manifest


class OffloadBackend:
    def __init__(self):
        import torch
        from accelerate import dispatch_model
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.torch = torch
        self.device = 'cuda:0'
        self.lock = threading.Lock()
        path = ROOT / 'models' / CONFIG['reader']
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        # Same full checkpoint/class/dtype/attention as parent, before placement.
        model = AutoModelForImageTextToText.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa').eval()
        assert all(p.device.type == 'cpu' for p in model.parameters())
        assert all(p.dtype == torch.bfloat16 for p in model.parameters())
        language = model.model.language_model
        def size(module):
            return sum(p.numel() * p.element_size() for p in module.parameters())
        mapping = {'model.visual': 'cpu', 'lm_head': 0}
        resident = size(model.lm_head)
        for name, module in language.named_children():
            if name != 'layers':
                mapping['model.language_model.' + name] = 0
                resident += size(module)
        gpu_layers = []
        for i, layer in enumerate(language.layers):
            name = f'model.language_model.layers.{i}'
            if resident + size(layer) <= CONFIG['resident_weights_mib'] * 2**20:
                mapping[name] = 0
                resident += size(layer)
                gpu_layers.append(i)
            else:
                mapping[name] = 'cpu'
        assert 0 < len(gpu_layers) < len(language.layers)
        for name, _ in model.named_parameters():
            assert any(name == m or name.startswith(m + '.') for m in mapping), name
        # Whole-layer preloading covers direct access to conv/norm weights in
        # Qwen's gated-delta layers, which bypasses child.forward hooks.
        preload = sorted({type(layer).__name__ for layer in language.layers})
        self.model = dispatch_model(model, device_map=mapping, main_device=0,
                                    preload_module_classes=preload).eval()
        self.placement = dict(device_map=mapping, gpu_layers=gpu_layers,
            resident_weights_mib=resident / 2**20, preload_classes=preload,
            cpu_is_weight_storage=True, all_text_compute_device='cuda:0',
            quantization=False, precision='bfloat16', attention='sdpa')


def run():
    import fcntl
    import resource
    from .mainline_service_v1 import generate
    RUN.mkdir(parents=True, exist_ok=True)
    assert not list((RUN / 'outputs').glob('*.json'))
    assert not list((RUN / 'started').glob('*.json'))
    signature, manifest = verify()
    reader = CONFIG['reader']
    requests = parent.load(parent.BASE / 'public/requests.json')
    replay = parent.load(parent.BASE / 'public/replay.json')
    keys = sorted(k for k, r in requests.items() if r['reader'] == reader)
    assert len(keys) == 835 and len(manifest['smoke_keys']) == 8
    assert set(manifest['smoke_keys']) <= set(keys) & set(replay)
    old = ROOT / 'runs/length_repack_v1'
    assert not list((old / 'started' / reader).glob('*.json'))
    for receipt in ['submission.json', 'active_worker.json']:
        v = parent.load(old / receipt)
        p = Path('/proc') / str(v['pid'])
        assert not p.exists() or (p / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z', 'Parent still active'
    state('verifying_parent', requests=len(keys))
    old_signature = manifest['parent_manifest_sha256']
    for k, req in requests.items():
        if req['reader'] != 'Qwen3.5-4B':
            continue
        v = parent.load(old / 'outputs' / 'Qwen3.5-4B' / (k + '.json'))
        assert v['signature'] == old_signature and v['key'] == k
    cpu = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    assert cpu['MemAvailable'] / 1024 >= CONFIG['cpu_available_mib']
    deadline = time.monotonic() + CONFIG['wait_seconds']
    with (ROOT / 'runs/gpu_spare_admission.lock').open('a') as lock:
        while True:
            acquired = False
            free = parent.free_memory()
            if free >= CONFIG['admission_mib']:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    pass
            if acquired:
                free = parent.free_memory()
                if free >= CONFIG['admission_mib']:
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
            state('waiting_for_memory', free_mib=free, required_mib=CONFIG['admission_mib'])
            if time.monotonic() >= deadline:
                raise TimeoutError('Offload admission timeout; preserve parent outputs')
            time.sleep(20)
        import torch
        torch.set_num_threads(8)
        torch.manual_seed(CONFIG['seed'])
        torch.cuda.manual_seed_all(CONFIG['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_cap_mib'] * 2**20 / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        state('loading', free_mib_at_admission=free, gpu_cap_mib=CONFIG['gpu_cap_mib'])
        backend = OffloadBackend()
        parent.immutable(RUN / 'placement.json', backend.placement)
        for key in keys:
            req = requests[key]
            parent.validate_request(key, req)
            prompt = backend.tokenizer.apply_chat_template(req['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            length = len(backend.tokenizer.encode(prompt, add_special_tokens=False))
            assert length <= 4096
            if key in replay:
                assert length == replay[key]['result']['usage']['prompt_tokens']
        state('smoke_running', expected=8, resident_weights_mib=backend.placement['resident_weights_mib'])
        ordered = manifest['smoke_keys'] + [k for k in keys if k not in manifest['smoke_keys']]
        finished = 0
        comparisons = []
        smoke_seconds = []
        try:
            for key in ordered:
                req = requests[key]
                parent.immutable(RUN / 'started' / (key + '.json'),
                    dict(signature=signature, key=key, reader=reader, time=datetime.now(timezone.utc).isoformat()))
                result = generate(backend, req['messages'], req['max_output'])
                assert 0 < result['usage']['completion_tokens'] <= 128
                parent.immutable(RUN / 'outputs' / (key + '.json'),
                    dict(signature=signature, key=key, reader=reader, origin='new_bf16_weight_offload_inference', result=result))
                finished += 1
                if key in replay:
                    comparisons.append(dict(key=key,
                        token_identical=result['completion_token_ids'] == replay[key]['result']['completion_token_ids'],
                        text_identical=result['text'] == replay[key]['result']['text']))
                if finished <= 8:
                    smoke_seconds.append(result['usage']['elapsed_seconds'])
                    state('smoke_running', completed=finished, expected=8,
                          last_seconds=smoke_seconds[-1], peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
                if finished == 8:
                    parent.immutable(RUN / 'smoke.json', dict(signature=signature, comparisons=comparisons,
                        seconds=smoke_seconds, all_token_identical=all(c['token_identical'] for c in comparisons),
                        estimated_remaining_seconds=sum(smoke_seconds) / 8 * (len(keys) - 8),
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                        decision='run_all_835_under_one_backend_without_historical_9b_reuse'))
                if finished % 10 == 0 or finished == len(keys):
                    state('running', new_completed=finished, new_expected=len(keys),
                          phase_seconds=time.perf_counter() - started,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
            parent.immutable(RUN / 'historical_consistency.json', dict(comparisons=comparisons,
                checked=len(comparisons), identical_tokens=sum(c['token_identical'] for c in comparisons),
                identical_text=sum(c['text_identical'] for c in comparisons),
                note='Diagnostic only; every 9B outcome uses the new backend even if historical outputs differ.'))
            parent.immutable(RUN / 'completion.json', dict(status='completed', signature=signature,
                new_9b_calls=finished, inherited_4b_requests=835, time=datetime.now(timezone.utc).isoformat(),
                private_references_read=False))
            state('completed', new_completed=finished, new_expected=len(keys))
        finally:
            parent.immutable(RUN / 'resources.json', dict(reader=reader,
                status='completed' if finished == len(keys) else 'failed', new_calls=finished,
                phase_seconds=time.perf_counter() - started,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                cpu_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                free_mib_at_admission=free, placement=backend.placement))


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        RUN.mkdir(parents=True, exist_ok=True)
        state('failed', error=repr(error), traceback=traceback.format_exc())
        raise
