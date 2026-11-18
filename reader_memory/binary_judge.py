"""A prefix-constrained yes/no decoder; semantic reliability still needs validation."""
from __future__ import annotations

import time


def allowed_next(prefix, sequences, eos_ids):
    allowed = set()
    prefix = tuple(prefix)
    for sequence in sequences:
        sequence = tuple(sequence)
        if sequence[:len(prefix)] == prefix:
            allowed.update(eos_ids if len(prefix) == len(sequence) else [sequence[len(prefix)]])
    if not allowed:
        raise ValueError("Decoder left the binary label grammar")
    return sorted(allowed)


class BinaryJudge:
    def __init__(self, backend):
        self.backend = backend
        self.model_name = backend.model_name
        self.sequences = [backend.tokenizer.encode(word, add_special_tokens=False) for word in ("yes", "no")]
        if any(not seq for seq in self.sequences) or self.sequences[0] == self.sequences[1]:
            raise ValueError("Tokenizer cannot distinguish binary labels")
        eos = backend.model.generation_config.eos_token_id
        self.eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        if not self.eos_ids or any(e is None for e in self.eos_ids):
            raise ValueError("Missing EOS token")

    def complete(self, messages, max_tokens=32):
        b = self.backend
        prompt = b.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = b.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(b.device)
        length = batch["input_ids"].shape[-1]
        if length > b.max_input_tokens:
            raise ValueError("Judge prompt too long; refusing silent truncation")
        start = time.perf_counter()
        limit = max(map(len, self.sequences)) + 1
        with b.lock, b.torch.inference_mode():
            output = b.model.generate(**batch, max_new_tokens=limit, do_sample=False,
                                     pad_token_id=b.tokenizer.pad_token_id or self.eos_ids[0],
                                     prefix_allowed_tokens_fn=lambda batch_id, ids: allowed_next(ids[length:].tolist(), self.sequences, self.eos_ids))
        generated = output[0, length:].tolist()
        text = b.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if text not in {"yes", "no"} or not generated or generated[-1] not in self.eos_ids:
            raise ValueError("Binary judge did not finish a valid label")
        return {"text": text, "usage": {"prompt_tokens": length, "completion_tokens": len(generated),
                "elapsed_seconds": round(time.perf_counter()-start, 4), "eos_terminated": True,
                "decoding": "prefix-constrained-yes-no"}}
