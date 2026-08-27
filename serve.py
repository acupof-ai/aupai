#!/usr/bin/env python3
"""Standalone CPU inference server — no fla dependency.
Usage: python serve.py [--port 8080]"""

import argparse
import os
from types import SimpleNamespace

import torch
from flask import Flask, Response, jsonify, request
from tokenizers import Tokenizer

from sampling import top_p_sample
from train import (
    HybridLM,  # noqa: E402  (real architecture; the old inline GDN copy could not load current ckpts)
)

# ── load model ──

ROOT = os.path.dirname(os.path.abspath(__file__))
device = "cpu"
# prefer SFT checkpoint, fall back to pretrained
_ckpt_path = os.path.join(ROOT, "ckpt_sft.pt")
if not os.path.exists(_ckpt_path):
    _ckpt_path = os.path.join(ROOT, "ckpt.pt")
ck = torch.load(_ckpt_path, map_location=device, weights_only=False)
cfg = SimpleNamespace(**ck["cfg"])
model = HybridLM(cfg).to(device)
model.load_state_dict(ck["model"])
model.eval()
tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
# ensure ChatML + think special tokens exist
for t in ["<|im_start|>", "<|im_end|>", "<|think|>", "<|/think|>"]:
    if tok.token_to_id(t) is None:
        tok.add_special_tokens([t])
print(f"model loaded, vocab={tok.get_vocab_size()}, seq={cfg.seq}", flush=True)


def generate(prompt, max_new=200, temp=0.8, top_p=0.95, rep_penalty=1.2):
    eos = tok.token_to_id("<|im_end|>")
    prompt_ids = tok.encode(prompt).ids
    x = torch.tensor([prompt_ids], device=device)
    seen = {}  # token -> count, for repetition penalty
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(x[:, -cfg.seq :])[0][:, -1] / temp
        # mask padding tokens (never trained, avoid random high logits)
        logits[:, tok.get_vocab_size() :] = float("-inf")
        # repetition penalty: penalize tokens already generated
        if rep_penalty > 1.0:
            for tid, cnt in seen.items():
                if cnt > 0 and logits[0, tid] > 0:
                    logits[0, tid] /= rep_penalty
                elif cnt > 0:
                    logits[0, tid] *= rep_penalty
        nxt = top_p_sample(logits, top_p)
        x = torch.cat([x, nxt], dim=1)
        tid = nxt.item()
        seen[tid] = seen.get(tid, 0) + 1
        if tid == eos:
            break
    generated = x[0].tolist()[len(prompt_ids) :]
    return tok.decode(generated, skip_special_tokens=True).strip()


def format_history(history):
    parts = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    text = "".join(parts)
    if len(text) > 800:
        text = text[-800:]
    return text


# ── flask ──

app = Flask(__name__)


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    history = data.get("history", [])
    temp = float(data.get("temperature", 0.8))
    prompt = format_history(history)
    reply = generate(prompt, temp=temp)
    return jsonify({"reply": reply})


HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aupai Chat</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#f5f5f5;height:100vh;display:flex;flex-direction:column}
#chat{flex:1;overflow-y:auto;padding:16px;max-width:800px;margin:0 auto;width:100%}
.msg{margin-bottom:12px;display:flex;gap:8px}
.msg.user{flex-direction:row-reverse}
.bubble{max-width:75%;padding:10px 14px;border-radius:12px;line-height:1.6;white-space:pre-wrap;word-break:break-word}
.user .bubble{background:#007aff;color:#fff}
.assistant .bubble{background:#fff;color:#333;border:1px solid #e0e0e0}
#input-bar{display:flex;gap:8px;padding:12px 16px;background:#fff;border-top:1px solid #e0e0e0;max-width:800px;margin:0 auto;width:100%}
#input{flex:1;padding:10px 14px;border:1px solid #ddd;border-radius:8px;font-size:15px;resize:none;height:44px;font-family:inherit}
#send{padding:0 20px;background:#007aff;color:#fff;border:none;border-radius:8px;font-size:15px;cursor:pointer;white-space:nowrap}
#send:disabled{opacity:.5}
#temp{width:70px;padding:8px;border:1px solid #ddd;border-radius:8px;font-size:13px;text-align:center}
</style>
</head>
<body>
<div id="chat"></div>
<div id="input-bar">
  <input id="temp" type="number" value="0.8" step="0.1" min="0.1" max="2" title="temperature">
  <textarea id="input" placeholder="问点什么..." rows="1"></textarea>
  <button id="send" onclick="send()">发送</button>
</div>
<script>
const history = [];
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  div.appendChild(b);
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return b;
}
async function send() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addMsg('user', text);
  history.push({role:'user', content:text});
  sendBtn.disabled = true;
  const bubble = addMsg('assistant', '...');
  try {
    const r = await fetch('/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({history, temperature: parseFloat(document.getElementById('temp').value)})
    });
    const d = await r.json();
    bubble.textContent = d.reply || '(空回复)';
    history.push({role:'assistant', content:d.reply});
  } catch(e) {
    bubble.textContent = '错误: ' + e.message;
  }
  sendBtn.disabled = false;
  chatEl.scrollTop = chatEl.scrollHeight;
}
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=False)
