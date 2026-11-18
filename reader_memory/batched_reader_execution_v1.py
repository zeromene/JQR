"""Execution helper for padded greedy batches; no change to prompts or token limits."""
import time


def generate_batch(backend, requests):
    if not requests or len(requests) > 4:
        raise ValueError('Reader execution batches must contain 1..4 requests')
    limits = {int(r['max_tokens']) for r in requests}
    if len(limits) != 1 or not 1 <= next(iter(limits)) <= 16000:
        raise ValueError('Mixed or invalid output limits')
    prompts = [backend.tokenizer.apply_chat_template(r['messages'], tokenize=False,
               add_generation_prompt=True, enable_thinking=False) for r in requests]
    backend.tokenizer.padding_side = 'left'
    batch = backend.tokenizer(prompts, padding=True, return_tensors='pt',
                              add_special_tokens=False).to(backend.device)
    lengths = batch['attention_mask'].sum(-1).tolist()
    width = batch['input_ids'].shape[-1]
    if width > 32768:
        raise ValueError('Reader prompt exceeds original input limit; no truncation')
    started = time.perf_counter()
    with backend.lock, backend.torch.inference_mode():
        output = backend.model.generate(**batch, max_new_tokens=next(iter(limits)), do_sample=False,
            pad_token_id=backend.tokenizer.pad_token_id if backend.tokenizer.pad_token_id is not None
            else backend.tokenizer.eos_token_id)
    eos = backend.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    results = []
    for row, length in zip(output, lengths, strict=True):
        ids = row[width:].tolist()
        first = next((i for i, token in enumerate(ids) if token in eos), None)
        if first is not None:
            ids = ids[:first + 1]
        raw = backend.tokenizer.decode(ids, skip_special_tokens=True)
        stopped = bool(ids and ids[-1] in eos)
        results.append(dict(text=raw.split('</think>', 1)[-1].strip(), raw_text=raw,
            completion_token_ids=ids, finish_reason='stop' if stopped else 'length', eos_seen=stopped,
            usage=dict(prompt_tokens=int(length), completion_tokens=len(ids),
                       batch_elapsed_seconds=time.perf_counter()-started,
                       batch_size=len(requests))))
    return results
