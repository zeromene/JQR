from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time


def serve(backend, host="127.0.0.1", port=8000):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ready", "model": backend.model_name, "offline": True})
            elif self.path == "/v1/models":
                self.respond(200, {"object": "list", "data": [{"id": backend.model_name, "object": "model"}]})
            else:
                self.respond(404, {"error": "unknown route"})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2_000_000:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                if self.path == "/v1/tokenize":
                    self.respond(200, {"tokens": len(backend.tokenizer.encode(request["text"], add_special_tokens=False))})
                    return
                if self.path != "/v1/chat/completions":
                    self.respond(404, {"error": "unknown route"})
                    return
                if request.get("stream"):
                    raise ValueError("This minimal offline server supports non-streaming requests only")
                if request.get("model", backend.model_name) != backend.model_name:
                    raise ValueError("Requested model is not loaded")
                result = backend.complete(request["messages"], int(request.get("max_tokens", 512)))
                self.respond(200, {"id": f"local-{time.time_ns()}", "object": "chat.completion", "model": backend.model_name,
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]}, "finish_reason": "stop"}], "usage": result["usage"]})
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                self.respond(500, {"error": type(exc).__name__ + ": " + str(exc)})
    print(f"READY http://{host}:{port} model={backend.model_name}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
