"""Isolated offline model service for the native-QA baseline experiment."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time

from .backend import TransformersBackend


def generate(backend, messages, max_tokens):
    if not 1 <= max_tokens <= 8192:
        raise ValueError('max_tokens must be in [1, 8192]')
    prompt = backend.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = backend.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(backend.device)
    input_length = batch['input_ids'].shape[-1]
    if input_length > backend.max_input_tokens:
        raise ValueError(f'Input {input_length} exceeds {backend.max_input_tokens}; no truncation')
    started = time.perf_counter()
    with backend.lock, backend.torch.inference_mode():
        output = backend.model.generate(**batch, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=backend.tokenizer.pad_token_id if backend.tokenizer.pad_token_id is not None
            else backend.tokenizer.eos_token_id)
    ids = output[0, input_length:].tolist()
    eos = backend.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    stopped = bool(ids and ids[-1] in eos)
    raw = backend.tokenizer.decode(ids, skip_special_tokens=True)
    cleaned = raw.split('</think>', 1)[-1].strip()
    return {'text': cleaned, 'raw_text': raw, 'completion_token_ids': ids,
        'finish_reason': 'stop' if stopped else 'length', 'eos_seen': stopped,
        'usage': {'prompt_tokens': input_length, 'completion_tokens': len(ids),
            'elapsed_seconds': time.perf_counter() - started}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--port', type=int, default=8002)
    args = parser.parse_args()
    backend = TransformersBackend(args.model_path, max_input_tokens=32768)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self.respond(200, {'status': 'ready', 'model': backend.model_name,
                    'service': 'native-baseline-v1', 'max_input_tokens': 32768,
                    'max_output_tokens': 8192})
            else:
                self.respond(404, {'error': 'unknown route'})

        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4_000_000:
                    raise ValueError('invalid request size')
                value = json.loads(self.rfile.read(size))
                if self.path == '/tokenize':
                    self.respond(200, {'tokens': len(backend.tokenizer.encode(
                        value['text'], add_special_tokens=False))})
                elif self.path == '/complete':
                    self.respond(200, generate(backend, value['messages'], int(value['max_tokens'])))
                else:
                    self.respond(404, {'error': 'unknown route'})
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {'error': f'{type(exc).__name__}: {exc}'})
            except Exception as exc:
                self.respond(500, {'error': f'{type(exc).__name__}: {exc}'})

    print(f'READY native-baseline-v1 port={args.port} model={backend.model_name}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
