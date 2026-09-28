#!/usr/bin/env python3
# restartable: read-only; loads a checkpoint and prints continuations, writes nothing.
"""Local continuation REPL for a BASE checkpoint (no ChatML: the pretraining corpus has none).

    python3 scripts/base_generate.py --ckpt ~/models/aupai_v41_ced_0926/v41_ced_0926_final_weights_bf16.pt \
        --tokenizer ~/models/aupai_v41_ced_0926/tokenizer.json
    python3 scripts/base_generate.py --ckpt ... --tokenizer ... --prompt "def fib(n):\\n    '''Return the n-th Fibonacci number.'''\\n"

Prompts that match pretraining: a Python signature + docstring, "Question: ...\\nAnswer:",
"问：...\\n答：". A literal \\n in the typed prompt becomes a newline. Empty line quits.
--serve 8765 serves the same generator as a web page at http://127.0.0.1:8765 (streams the text).
On a GPU box: --device cuda --serve 8766; on the laptop: --serve 8765 --relay 8766 shows the same page
and runs every request on the pod's GPU through ~/bin/pod.
Each step re-runs the whole sequence (no KV cache), so speed falls as the text grows.
"""
import argparse
import base64
import http.server
import json
import os
import subprocess
import sys
import threading
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.loader import IM_END, format_prompt, load_checkpoint, load_tokenizer  # noqa: E402

HOLD = 16
STOPS = ["\n\n\n", "\nQuestion:", "\n问：", "\ndef ", "\nclass ", "\nif __name__"]


def pick_device(want):
    if want != "auto":
        return want
    return "cpu"  # measured 2026-09-28 on M4 Pro: cpu 7.8 tok/s vs mps 4.5 (small per-token kernels)


def stream(model, tok, prompt, device, max_new, temp, stops, stop_ids=(1,)):
    """Yield the continuation in pieces; a piece that could still grow into a stop is held back."""
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], device=device)
    new, printed, text = [], "", ""
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(x[:, -model.cfg.seq:])[0][:, -1].float()
            if temp <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temp, -1), 1)
            if nxt.item() in stop_ids:
                break
            new.append(nxt.item())
            x = torch.cat([x, nxt.view(1, 1)], 1)
            text = tok.decode(new)
            cut = [text.find(s) for s in stops if s in text]
            if cut:
                text = text[:min(cut)]
                break
            safe = text[:max(len(printed), len(text) - HOLD)]
            yield safe[len(printed):]
            printed = safe
    yield text[len(printed):]
    stream.n_tokens = len(new)


def prep(p, tok, chat=False):
    """-> (prompt, text stops, stop token ids). Chat wraps the question in ChatML and stops on
    <|im_end|>, the token an instruction-tuned checkpoint ends its answer with."""
    if chat:
        return format_prompt(p), [], (1, tok.token_to_id(IM_END))
    if p.lstrip().startswith(("def ", "from ", "import ")):
        # HumanEval scores the rstrip-nl arm; a trailing blank line reads as end-of-function
        return p.rstrip("\n"), STOPS, (1,)
    return p, STOPS[:3], (1,)


def generate(model, tok, prompt, device, max_new, temp, stops, stop_ids):
    t0 = time.time()
    for piece in stream(model, tok, prompt, device, max_new, temp, stops, stop_ids):
        sys.stdout.write(piece)
        sys.stdout.flush()
    dt = time.time() - t0
    print(f"\n[{stream.n_tokens} tokens, {dt:.1f}s, {stream.n_tokens / max(dt, 1e-9):.1f} tok/s]")


def relay(body, port):
    """Forward one request to a --serve running in the pod container, through the pod exec
    channel (tn has no port forwarding). Killing the local wrapper does not stop the remote
    curl: an abandoned request finishes on the GPU and is discarded."""
    cmd = f"echo {base64.b64encode(body).decode()} | base64 -d | curl -4 -sN --data-binary @- http://127.0.0.1:{port}/"
    p = subprocess.Popen([os.path.expanduser("~/bin/pod"), cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while chunk := p.stdout.read1(4096):
            yield chunk
    finally:
        p.kill()


def serve(model, tok, device, port, relay_port=0):
    lock = threading.Lock()  # ponytail: one model, one request at a time

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if relay_port:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                try:
                    for chunk in relay(body, relay_port):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            req = json.loads(body)
            prompt, stops, stop_ids = prep(req["prompt"], tok, bool(req.get("chat")))
            max_new = max(1, min(int(req.get("max_new", 256)), 1024))
            temp = float(req.get("temp", 0.0))
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            with lock:
                t0 = time.time()
                try:
                    for piece in stream(model, tok, prompt, device, max_new, temp, stops, stop_ids):
                        if piece:
                            self.wfile.write(piece.encode())
                            self.wfile.flush()
                    dt = time.time() - t0
                    self.wfile.write(f"\n\u0000{stream.n_tokens} tokens, {dt:.1f}s".encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the browser stopped reading; free the model for the next request

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"serving http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>aupai base</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--card:#fff;--acc:#2563eb;--gen:#0f766e}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#e7e5e4;--mut:#a8a29e;--line:#292524;--card:#1c1917;--acc:#60a5fa;--gen:#5eead4}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:880px;margin:0 auto;padding:20px 16px}h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
textarea,pre{width:100%;font:13.5px/1.5 ui-monospace,Menlo,monospace;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:12px}
textarea{min-height:180px;resize:vertical}pre{white-space:pre-wrap;word-break:break-word;min-height:120px;margin:12px 0 4px}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.go{background:var(--acc);border-color:var(--acc);color:#fff}button:disabled{opacity:.5;cursor:default}
label{color:var(--mut);font-size:13px}input{width:70px;font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
.g{color:var(--gen)}.stat{color:var(--mut);font-size:12px}
</style></head><body><main>
<h1>aupai 基座模型 · 续写</h1>
<div class="sub">这是预训练基座，只会接着写，不会对话。按下面的格式开头效果最好。</div>
<div class="row" id="presets"></div>
<textarea id="p"></textarea>
<div class="row">
<button class="go" id="go">生成 (⌘↵)</button><button id="stop" disabled>停止</button>
<label>温度 <input id="t" type="number" value="0" step="0.1" min="0" max="2"></label>
<label>最多 token <input id="m" type="number" value="256" step="32" min="16" max="1024"></label>
<label><input id="c" type="checkbox" style="width:auto"> 对话模式（SFT 过的模型）</label>
</div>
<pre id="o"></pre><div class="stat" id="s"></div>
</main><script>
const P={
"代码":"def is_palindrome(s: str) -> bool:\\n    \\"\\"\\"Return True if s reads the same forwards and backwards, ignoring case and non-letters.\\n    >>> is_palindrome('A man, a plan, a canal: Panama')\\n    True\\n    \\"\\"\\"\\n",
"数学":"Question: A shop sells pens at 3 dollars each. Tom buys 7 pens and pays with a 50 dollar bill. How much change does he get?\\nAnswer:",
"中文问答":"问：为什么天空是蓝色的？\\n答：",
"英文续写":"The key idea behind binary search is",
"对话":"为什么天空是蓝色的？"};
const $=id=>document.getElementById(id);let ctl=null;
for(const k in P){const b=document.createElement("button");b.textContent=k;b.onclick=()=>{$("p").value=P[k];$("c").checked=k==="对话"};$("presets").append(b)}
$("p").value=P["代码"];
async function go(){
 const prompt=$("p").value;if(!prompt.trim())return;ctl=new AbortController();
 $("go").disabled=true;$("stop").disabled=false;$("s").textContent="生成中…";
 const o=$("o");o.textContent="";const g=document.createElement("span");g.className="g";o.append($("c").checked?"问："+prompt+"\n\n答：":prompt,g);
 try{const r=await fetch("/",{method:"POST",body:JSON.stringify({prompt,chat:$("c").checked,temp:+$("t").value,max_new:+$("m").value}),signal:ctl.signal});
  const rd=r.body.getReader(),dec=new TextDecoder();let buf="";
  for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
   const i=buf.indexOf("\\n\\u0000");g.textContent=i<0?buf:buf.slice(0,i);if(i>=0)$("s").textContent=buf.slice(i+2)}
 }catch(e){$("s").textContent=e.name==="AbortError"?"已停止":"出错："+e}
 $("go").disabled=false;$("stop").disabled=true;
}
$("go").onclick=go;$("stop").onclick=()=>ctl&&ctl.abort();
$("p").addEventListener("keydown",e=>{if(e.key==="Enter"&&(e.metaKey||e.ctrlKey))go()});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--tokenizer")
    ap.add_argument("--device", default="auto", help="auto (= cpu) | mps | cpu | cuda")
    ap.add_argument("--prompt", default=None, help="one prompt, then exit")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--chat", action="store_true", help="wrap each prompt in ChatML (instruction-tuned checkpoints)")
    ap.add_argument("--serve", type=int, default=0, metavar="PORT", help="serve a web page on 127.0.0.1:PORT")
    ap.add_argument("--relay", type=int, default=0, metavar="POD_PORT",
                    help="with --serve: load nothing locally, forward each request to the pod's --serve on POD_PORT")
    args = ap.parse_args()
    if args.relay:
        return serve(None, None, None, args.serve, relay_port=args.relay)
    if not (args.ckpt and args.tokenizer):
        ap.error("--ckpt and --tokenizer are required unless --relay")
    device = pick_device(args.device)
    model, cfg = load_checkpoint(os.path.expanduser(args.ckpt), device="cpu", dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16, low_mem=True)
    held = []
    if device == "mps":
        # MPS has no float64. MoEFFN pins its load counters (h_load, h_sums) to float64 in its own
        # _apply, so they are detached for the move and put back as float32 on the device; they are
        # training-side statistics and their precision does not touch the logits.
        for m in model.modules():
            for k, b in list(m._buffers.items()):
                if b is not None and b.dtype == torch.float64:
                    held.append((m, k, b.float()))
                    m._buffers[k] = None
    model = model.to(device).eval()
    for m, k, b in held:
        m._buffers[k] = b.to(device)
    model.cfg = cfg
    tok = load_tokenizer(os.path.expanduser(args.tokenizer), cfg)
    if args.serve:
        return serve(model, tok, device, args.serve)
    print(f"loaded on {device}; greedy" if args.temp <= 0 else f"loaded on {device}; temp {args.temp}")
    prompts = [args.prompt] if args.prompt is not None else None
    while True:
        p = prompts.pop(0) if prompts else (None if prompts is not None else input("\nprompt> "))
        if not p:
            return
        p, stops, stop_ids = prep(p.replace("\\n", "\n"), tok, args.chat)
        print(p, end="")
        generate(model, tok, p, device, args.max_new, args.temp, stops, stop_ids)


if __name__ == "__main__":
    main()
