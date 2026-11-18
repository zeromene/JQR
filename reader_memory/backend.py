from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import urllib.request


class HTTPBackend:
    def __init__(self, endpoint, model, timeout=600):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages, max_tokens=512):
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False}).encode()
        headers = {"Content-Type": "application/json"}
        if os.environ.get("READER_MEMORY_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["READER_MEMORY_API_KEY"]
        request = urllib.request.Request(self.endpoint + "/chat/completions", body, headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        content = result["choices"][0]["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model returned empty/non-text content")
        return {"text": content, "usage": result.get("usage", {})}


class TransformersBackend:
    def __init__(self, model_path, device="cuda", max_input_tokens=12288):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        self.torch = torch
        self.path = str(Path(model_path).resolve())
        self.model_name = Path(self.path).name
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        configuration = AutoConfig.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        model_class = AutoModelForImageTextToText if configuration.model_type == "qwen3_5" else AutoModelForCausalLM
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = model_class.from_pretrained(self.path, local_files_only=True, trust_remote_code=False,
                                               dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.lock = threading.Lock()

    def complete(self, messages, max_tokens=512):
        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_length = batch["input_ids"].shape[-1]
        if input_length > self.max_input_tokens:
            raise ValueError(f"Input length {input_length} exceeds configured limit {self.max_input_tokens}; no silent truncation")
        start = time.perf_counter()
        with self.lock, self.torch.inference_mode():
            output = self.model.generate(**batch, max_new_tokens=max_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        new_tokens = output[0, input_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Some checkpoints still emit think markers even when thinking is disabled.
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return {"text": text.strip(), "usage": {"prompt_tokens": input_length, "completion_tokens": len(new_tokens),
                                                  "elapsed_seconds": round(time.perf_counter() - start, 4)}}


class FixtureBackend:
    """Deterministic fixture, explicitly distinct from model inference."""
    def complete(self, messages, max_tokens=512):
        text = messages[-1]["content"]
        if "JSON array" in text:
            return {"text": '[{"lossless_restatement":"The user moved to Hangzhou in 2026 and previously lived in Shanghai."}]', "usage": {}}
        return {"text": "The user currently lives in Hangzhou and previously lived in Shanghai.", "usage": {}}
