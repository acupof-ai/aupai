"""v42 runtime HTTP server on port 8732 (does NOT touch the running 8731).

Endpoints:
  GET  /            -> web UI
  GET  /healthz     -> liveness
  GET  /metrics     -> Engram hit/miss, RSS, queue depth
  POST /api/chat    -> {"messages":[...]} -> full JSON completion
  POST /v1/chat/completions -> OpenAI-compatible, SSE stream when stream=true
"""
from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx

from .generate import generate, IM_END
from .service import RuntimeService

PORT = 8732
DEFAULT_MAX_TOKENS = 512

# Built once at startup.
_svc: RuntimeService | None = None
_start_time = time.time()
_requests = 0


def _svc_get() -> RuntimeService:
    global _svc
    if _svc is None:
        _svc = RuntimeService(bits=8, prewarm=True)
    return _svc


def _chatml(messages) -> str:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    out.append("<|im_start|>assistant\n")
    return "\n".join(out)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open("v41f/mlx/runtime/web/index.html", "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(200, "<h1>v42 runtime</h1>", "text/html")
        elif self.path == "/healthz":
            self._send(200, {"status": "ok", "uptime_s": time.time() - _start_time,
                             "requests": _requests})
        elif self.path == "/metrics":
            svc = _svc_get()
            st = svc.engram.stats
            self._send(200, {
                "engram_hit_rate": st.hit_rate,
                "engram_ssd_lookups": st.ssd.lookups,
                "engram_ssd_hits": st.ssd.hits,
                "engram_mem_hits": st.mem_hits,
                "rss_gb": svc.rss_gb(),
                "requests": _requests,
                "uptime_s": time.time() - _start_time,
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global _requests
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})

        if self.path not in ("/api/chat", "/v1/chat/completions"):
            return self._send(404, {"error": "not found"})

        messages = body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]
        prompt = _chatml(messages)
        max_tokens = int(body.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(body.get("temperature", 0.0))
        stream = bool(body.get("stream", False))
        _requests += 1

        svc = _svc_get()
        if stream and self.path == "/v1/chat/completions":
            return self._stream(svc, prompt, max_tokens, temperature, body)
        res = generate(svc.model, svc.tokenizer, prompt, max_tokens=max_tokens,
                       temperature=temperature)
        self._send(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "text": res.text,
            "tokens": res.tokens,
            "usage": {"prompt_tokens": res.n_prompt, "completion_tokens": res.n_gen},
            "timing": {"prefill_s": res.t_prefill_s, "decode_s": res.t_decode_s,
                       "decode_tps": res.decode_tps},
        })

    def _stream(self, svc, prompt, max_tokens, temperature, body):
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
        ids = list(svc.tokenizer.encode(prompt).ids)
        sampler = make_sampler(temp=temperature) if temperature > 0 else None
        cache = svc.model.make_cache()
        step = generate_step(mx.array(ids, mx.int32), svc.model, max_tokens=max_tokens,
                             sampler=sampler, prompt_cache=cache, prefill_step_size=512)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            for tok, _lp in step:
                t = int(tok)
                if t == IM_END:
                    break
                chunk = {"choices": [{"delta": {"content": svc.tokenizer.decode([t])}}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except BrokenPipeError:
            pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    print(f"[server] prewarming model on :{args.port} ...", flush=True)
    _svc_get()
    print(f"[server] ready on :{args.port} rss={_svc.rss_gb():.2f}GB "
          f"build={_svc.build_s:.1f}s prewarm={_svc.prewarm_s:.1f}s", flush=True)
    srv = HTTPServer(("0.0.0.0", args.port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
