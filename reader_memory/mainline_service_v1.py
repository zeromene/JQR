"""Offline common reader/writer service; original baseline service is unchanged."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time

from .backend import TransformersBackend


def generate(backend, messages, max_tokens):
    if not 1 <= max_tokens <= 16000:
        raise ValueError('max_tokens outside [1,16000]')
    prompt = backend.tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    batch = backend.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(backend.device)
    length = batch['input_ids'].shape[-1]
    if length > 32768:
        raise ValueError(f'Input length {length} exceeds 32768; no truncation')
    started = time.perf_counter()
    with backend.lock, backend.torch.inference_mode():
        output = backend.model.generate(**batch, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=backend.tokenizer.pad_token_id if backend.tokenizer.pad_token_id is not None
                else backend.tokenizer.eos_token_id)
    ids = output[0, length:].tolist()
    eos = backend.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    stopped = bool(ids and ids[-1] in eos)
    raw = backend.tokenizer.decode(ids, skip_special_tokens=True)
    return dict(text=raw.split('</think>', 1)[-1].strip(), raw_text=raw,
        completion_token_ids=ids, finish_reason='stop' if stopped else 'length', eos_seen=stopped,
        usage=dict(prompt_tokens=length, completion_tokens=len(ids),
            elapsed_seconds=time.perf_counter()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--port', type=int, default=8016)
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
            self.respond(200 if self.path == '/health' else 404,
                dict(service='mainline-native-v1', model=backend.model_name,
                    max_input_tokens=32768, max_output_tokens=16000))

        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4_000_000:
                    raise ValueError('Invalid request size')
                value = json.loads(self.rfile.read(size))
                if self.path == '/tokenize':
                    self.respond(200, dict(tokens=len(backend.tokenizer.encode(
                        value['text'], add_special_tokens=False))))
                elif self.path == '/complete':
                    self.respond(200, generate(backend, value['messages'], int(value['max_tokens'])))
                else:
                    self.respond(404, dict(error='Unknown route'))
            except Exception as exc:
                self.respond(500, dict(error=f'{type(exc).__name__}: {exc}'))

    print('READY mainline-native-v1', args.port, flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
