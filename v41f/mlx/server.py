"""Local HTTP server for the v42 MLX backend.

Serves a chat web page and streams model output over SSE (Server-Sent Events).
Single shared model, one generation at a time (a lock serializes requests).

Run:
    .venv/bin/python -m v41f.mlx.server --host 127.0.0.1 --port 8731

Routes:
    GET  /            chat web page
    GET  /healthz     {"status":"ok"}
    POST /api/chat    SSE stream; body {"messages":[{"role","content"}],
                      "temperature":0.7,"top_p":0.95,"max_tokens":1024}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "/Users/bytedance/code/aupai")
import mlx.core as mx

WEB = os.path.join(os.path.dirname(__file__), "web", "index.html")

# Globals: model, streaming weights, tokenizer, and a serialization lock.
MODEL = None
SW = None
LM_MODEL = None
TOK = None
LOCK = threading.Lock()


def render_chatml(messages):
    sys_msg = "You are a helpful assistant. 用中文短句回答。"
    out = f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role not in ("user", "assistant", "system"):
            role = "user"
        out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    return out


def forward_last(ids):
    from v41f.mlx.forward_streaming import forward_streaming
    lg = forward_streaming(MODEL, SW, mx.array([ids], dtype=mx.int32))
    mx.eval(lg)
    mx.clear_cache()
    return lg[0, -1]


def sample_token(logits, temperature, top_p):
    if temperature <= 0.0:
        return int(mx.argmax(logits))
    logits = logits / max(float(temperature), 1e-5)
    probs = mx.softmax(logits)
    if top_p and 0.0 < top_p < 1.0:
        probs = top_p_probs(probs, top_p)
    return int(mx.random.categorical(probs[None]))


def top_p_probs(probs, p):
    # Sort descending; keep the smallest prefix whose cumulative mass reaches p;
    # restore original vocab order via the inverse permutation.
    sorted_idx = mx.argsort(probs)[::-1]
    sorted_p = probs[sorted_idx]
    cum = mx.cumsum(sorted_p)
    idx = mx.arange(sorted_p.shape[0])
    keep = mx.where(idx == 0, True, cum <= p)
    filtered = mx.where(keep, sorted_p, 0.0)
    filtered = filtered / mx.sum(filtered)
    inv = mx.argsort(sorted_idx)
    return filtered[inv]


def stream(messages, temperature, top_p, max_tokens):
    """Yield SSE events using mlx-lm's mature async generation loop."""
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    chat = render_chatml(messages)
    ids = TOK.encode(chat).ids
    im_end = TOK.token_to_id("<|im_end|>")
    sampler = make_sampler(temp=temperature, top_p=top_p)
    out_ids, emitted, steps = [], "", 0

    for nxt, _ in generate_step(
            mx.array(ids, dtype=mx.int32), LM_MODEL,
            max_tokens=max_tokens, sampler=sampler):
        nxt = int(nxt)
        if nxt == im_end:
            yield {"done": True, "stop": "im_end", "tokens": steps}
            return
        out_ids.append(nxt)
        steps += 1
        full = TOK.decode(out_ids)
        if "\ufffd" not in full and full.startswith(emitted):
            piece = full[len(emitted):]
            emitted = full
            if piece:
                yield {"delta": piece}

    full = TOK.decode(out_ids)
    if "\ufffd" not in full and full.startswith(emitted):
        piece = full[len(emitted):]
        if piece:
            yield {"delta": piece}
    yield {"done": True, "stop": "max_tokens", "tokens": steps}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/healthz"):
            self._send(200, "application/json",
                       json.dumps({"status": "ok"}).encode())
        elif self.path in ("/", "/index.html"):
            try:
                body = open(WEB, "rb").read()
                self._send(200, "text/html; charset=utf-8", body)
            except FileNotFoundError:
                self._send(404, "text/plain", b"web page missing")
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if not self.path.startswith("/api/chat"):
            self._send(404, "text/plain", b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, "application/json", b'{"error":"bad request"}')
            return
        messages = payload.get("messages", [])
        if not messages:
            self._send(400, "application/json", b'{"error":"no messages"}')
            return
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.95))
        max_tokens = min(int(payload.get("max_tokens", 1024)), 2048)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        acquired = LOCK.acquire(timeout=120)
        try:
            if not acquired:
                self._sse({"error": "backend busy"})
                return
            for ev in stream(messages, temperature, top_p, max_tokens):
                self._sse(ev)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            try:
                self._sse({"error": str(e)})
            except Exception:
                return
        finally:
            if acquired:
                LOCK.release()

    def _sse(self, obj):
        self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()


def main():
    global MODEL, SW, LM_MODEL, TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8731)
    args = ap.parse_args()

    print("Loading model ...", flush=True)
    from v41f.mlx.cli import build_sft_model
    MODEL, SW, TOK = build_sft_model()
    from v41f.mlx.forward_cached import forward_cached
    from v41f.mlx.mlx_lm_adapter import V42MLXLMModel
    LM_MODEL = V42MLXLMModel(MODEL, SW, forward_cached)
    print(f"Model loaded. layers={MODEL.cfg.n_layers}", flush=True)
    if os.environ.get("V42_MOE_BACKEND", "q8") != "bf16":
        from v41f.mlx.q8_moe import get_qstore
        print("Prewarming fused Q8 experts ...", flush=True)
        get_qstore().prewarm(MODEL.cfg.n_layers)
        print("Fused Q8 experts ready.", flush=True)

    httpd = HTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving chat at {url}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
