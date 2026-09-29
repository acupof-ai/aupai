#!/usr/bin/env python3
# restartable: read-only scoring; parity/heval20 sections are small and cheap to re-run,
# heval164 flushes one judged row per problem to data/eval/ced_cache_gate_humaneval164.jsonl.
"""Acceptance gate for the CED incremental KV cache (scripts/ced_cache.py), run on CPU.

Four sections, each usable on its own:
  parity    exact-cache logits vs a whole-sequence forward, at prefill and at every
            greedy decode step, over HumanEval prompts; bounded-vs-exact KL/top-1.
  heval20   HumanEval first 20: greedy TEXT byte-identical exact-cache vs recompute,
            and both judged through eval/humaneval_gen's scorer.
  heval164  HumanEval 164: independent greedy under bounded and exact, pass@1 each
            side and the per-step logits KL/top-1 gap on the exact greedy path.
  speed     prompt 30 / gen 120 tok/s for off (recompute), bounded, exact.

Cardless by construction: refuses unless CUDA_VISIBLE_DEVICES is set EMPTY.

    CUDA_VISIBLE_DEVICES= taskset -c 100-113 python3 scripts/ced_cache_gate.py \
        --ckpt ckpt_v41_ced_0926.pt --tokenizer data/tokenizer.json --section parity
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HE_DATA = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")


def _load_humaneval_scorer():
    """eval/humaneval_gen.py registers a SIGALRM handler at import; its judge/truncate
    are the scored HumanEval semantics, reuse them rather than a second copy."""
    p = os.path.join(ROOT, "eval", "humaneval_gen.py")
    spec = importlib.util.spec_from_file_location("humaneval_gen", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(ckpt, tok_path, threads, dtype="bf16"):
    from scripts.ced_cache import CEDCache  # noqa: F401  (import check)
    from scripts.loader import load_checkpoint, load_tokenizer
    torch.set_num_threads(threads)
    dt = torch.float32 if dtype == "fp32" else torch.bfloat16
    model, cfg = load_checkpoint(ckpt, device="cpu", dtype=dt, low_mem=True)
    model = model.eval()
    model.cfg = cfg
    tok = load_tokenizer(tok_path, cfg)
    return model, cfg, tok


@torch.no_grad()
def full_last(model, ids):
    """Whole-sequence recompute reference: last-position logits, cfg.seq window."""
    x = torch.tensor([ids], device="cpu")
    return model(x[:, -model.cfg.seq:])[0][0, -1].float()


@torch.no_grad()
def gen_cached(model, tok, ids, max_new, mode, stop_check=None):
    """Greedy with the incremental cache. Returns (token_ids, logits_per_step)."""
    from scripts.ced_cache import CEDCache
    eng = CEDCache(model, mode=mode)
    lg = eng.prefill(torch.tensor(ids, device="cpu")).float()
    out, steps = [], []
    for _ in range(max_new):
        nxt = int(lg.argmax().item())
        if nxt == 1:
            break
        out.append(nxt)
        if stop_check is not None and len(out) % 16 == 0 and stop_check(tok.decode(out)):
            break
        lg = eng.step(nxt).float()
        steps.append(lg)
    return out, steps


@torch.no_grad()
def gen_recompute(model, tok, ids, max_new, stop_check=None):
    """Original base_generate path: re-run the whole sequence every token."""
    x = torch.tensor([ids], device="cpu")
    out = []
    for _ in range(max_new):
        lg = model(x[:, -model.cfg.seq:])[0][0, -1].float()
        nxt = int(lg.argmax().item())
        if nxt == 1:
            break
        out.append(nxt)
        if stop_check is not None and len(out) % 16 == 0 and stop_check(tok.decode(out)):
            break
        x = torch.cat([x, torch.tensor([[nxt]])], 1)
    return out


def kl_pq(p, q):
    return float((p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum().item())


def section_parity(model, cfg, tok, probs, n_steps):
    """exact cache == full forward at prefill and every step; bounded KL/top-1 vs exact."""
    rows = []
    for p in probs:
        ids = tok.encode(p["prompt"].rstrip("\n")).ids
        cur = list(ids)
        from scripts.ced_cache import CEDCache
        ex, bd = CEDCache(model, "exact"), CEDCache(model, "bounded")
        lg_ex = ex.prefill(torch.tensor(cur)).float()
        lg_bd = bd.prefill(torch.tensor(cur)).float()
        ref = full_last(model, cur)
        pre_md = float((lg_ex - ref).abs().max())
        pre_agree = lg_ex.argmax().item() == ref.argmax().item()
        step_md, step_agree, kls, top1s = [], [], [], []
        kls.append(kl_pq(torch.softmax(lg_ex, -1), torch.softmax(lg_bd, -1)))
        top1s.append(lg_ex.argmax().item() == lg_bd.argmax().item())
        for _ in range(n_steps):
            nxt = int(lg_ex.argmax().item())
            if nxt == 1:
                break
            cur.append(nxt)
            lg_ex = ex.step(nxt).float()
            lg_bd = bd.step(nxt).float()
            ref = full_last(model, cur)
            step_md.append(float((lg_ex - ref).abs().max()))
            step_agree.append(lg_ex.argmax().item() == ref.argmax().item())
            kls.append(kl_pq(torch.softmax(lg_ex, -1), torch.softmax(lg_bd, -1)))
            top1s.append(lg_ex.argmax().item() == lg_bd.argmax().item())
        import statistics
        rows.append(dict(task=p["task_id"], P=len(ids), steps=len(step_md),
                         prefill_md=pre_md, prefill_agree=pre_agree,
                         step_md_max=max(step_md or [0.0]), step_md_mean=statistics.mean(step_md or [0.0]),
                         step_argmax=sum(step_agree), step_n=len(step_agree),
                         kl_max=max(kls), kl_mean=statistics.mean(kls),
                         bounded_top1=sum(top1s), top1_n=len(top1s)))
        print(json.dumps(rows[-1]), flush=True)
    md_all = max(r["step_md_max"] for r in rows)
    tok_all = all(r["prefill_agree"] and r["step_argmax"] == r["step_n"] for r in rows)
    bd_agree = sum(r["bounded_top1"] for r in rows)
    bd_n = sum(r["top1_n"] for r in rows)
    print(f"PARITY exact-vs-full: max step logit |d| = {md_all:.5f}; greedy argmax exact "
          f"{'AGREES at every step' if tok_all else 'DISAGREES'}")
    print(f"PARITY bounded-vs-exact: top-1 agree {bd_agree}/{bd_n} = "
          f"{100*bd_agree/max(bd_n,1):.2f}%; mean/max per-row KL "
          f"{sum(r['kl_mean'] for r in rows)/len(rows):.5f} / {max(r['kl_max'] for r in rows):.5f}")
    out = os.path.join(ROOT, "data", "eval", "ced_cache_gate_parity.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", out)
    return md_all, tok_all


def _heval_gens(model, tok, probs, max_new, modes):
    """Greedy text per mode through HumanEval stop/truncate semantics."""
    he = _load_humaneval_scorer()

    def stop_for(p):
        def _stop(s):
            return he.hits_stop(s, p["entry_point"])
        return _stop

    results = {m: [] for m in modes}
    for k, p in enumerate(probs):
        ids = tok.encode(p["prompt"].rstrip("\n")).ids
        stop = stop_for(p)
        for m in modes:
            if m == "off":
                toks = gen_recompute(model, tok, ids, max_new, stop_check=stop)
            else:
                toks, _ = gen_cached(model, tok, ids, max_new, m, stop_check=stop)
            results[m].append(he.truncate(tok.decode(toks), p["entry_point"]))
        print(f"  {k+1}/{len(probs)} {p['task_id']} lens "
              + " ".join(f"{m}={len(results[m][-1])}" for m in modes), flush=True)
    return results, he


def section_heval20(model, cfg, tok, probs, max_new):
    p20 = probs[:20]
    res, he = _heval_gens(model, tok, p20, max_new, ("exact", "off"))
    ndiff = sum(a != b for a, b in zip(res["exact"], res["off"], strict=True))
    pe = [he.judge(p, c, p["prompt"].rstrip("\n"))
          for p, c in zip(p20, res["exact"], strict=True)]
    po = [he.judge(p, c, p["prompt"].rstrip("\n"))
          for p, c in zip(p20, res["off"], strict=True)]
    print(f"HEVAL20 exact-vs-recompute text identical: {20-ndiff}/20; "
          f"pass exact={sum(pe)}/20 recompute={sum(po)}/20")
    if ndiff:
        for i, (a, b) in enumerate(zip(res["exact"], res["off"], strict=True)):
            if a != b:
                print(f"  DIFF {p20[i]['task_id']}: exact {len(a)} chars vs off {len(b)}")
    return ndiff


def section_heval164(model, cfg, tok, probs, max_new):
    # Streaming: exact and bounded greedy per problem, judge immediately, flush one row so
    # an interrupt hours in loses at most the current problem (parity text is not saved,
    # only pass/text-same -- the per-problem bytes live in humaneval_gen's own preds flow).
    he = _load_humaneval_scorer()
    out = os.path.join(ROOT, "data", "eval", "ced_cache_gate_humaneval164.jsonl")
    ne = nb = nsame = 0
    with open(out, "w") as f:
        for k, p in enumerate(probs, 1):
            ids = tok.encode(p["prompt"].rstrip("\n")).ids

            def stop(s, ep=p["entry_point"]):
                return he.hits_stop(s, ep)

            te, _ = gen_cached(model, tok, ids, max_new, "exact", stop_check=stop)
            tb, _ = gen_cached(model, tok, ids, max_new, "bounded", stop_check=stop)
            ce = he.truncate(tok.decode(te), p["entry_point"])
            cb = he.truncate(tok.decode(tb), p["entry_point"])
            oe = he.judge(p, ce, p["prompt"].rstrip("\n"))
            ob = he.judge(p, cb, p["prompt"].rstrip("\n"))
            ne += int(oe)
            nb += int(ob)
            nsame += int(ce == cb)
            f.write(json.dumps(dict(task_id=p["task_id"], exact=oe, bounded=ob,
                                    text_same=ce == cb)) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(probs):
                print(f"  {k}/164 pass exact={ne} bounded={nb} same-text={nsame}", flush=True)
    print(f"HEVAL164 pass@1 exact={ne}/164 bounded={nb}/164; greedy text identical {nsame}/164")
    print("wrote", out)
    return ne, nb, nsame


def section_speed(model, cfg, tok, prompt_text, gen_n):
    ids0 = tok.encode(prompt_text).ids[:30]
    if len(ids0) < 30:
        ids0 = (ids0 * (30 // len(ids0) + 1))[:30]
    for mode in ("off", "bounded", "exact"):
        ids = list(ids0)
        t0 = time.time()
        if mode == "off":
            toks = gen_recompute(model, tok, ids, gen_n)
        else:
            toks, _ = gen_cached(model, tok, ids, gen_n, mode)
        dt = time.time() - t0
        print(f"SPEED {mode}: {len(toks)} tokens in {dt:.1f}s = {len(toks)/max(dt,1e-9):.2f} tok/s "
              f"(prompt {len(ids0)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--section", required=True,
                    choices=("parity", "heval20", "heval164", "speed", "short", "all"))
    ap.add_argument("--max_new", type=int, default=280)
    ap.add_argument("--parity_steps", type=int, default=48)
    ap.add_argument("--parity_n", type=int, default=8)
    ap.add_argument("--speed_prompt", default=None)
    ap.add_argument("--speed_gen", type=int, default=120)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        sys.exit("REFUSING: this gate is cardless CPU; set CUDA_VISIBLE_DEVICES empty")
    with open(HE_DATA, encoding="utf-8") as fh:
        probs = [json.loads(l) for l in fh if l.strip()]
    model, cfg, tok = load(args.ckpt, args.tokenizer, args.threads, args.dtype)

    todo = (["parity", "heval20", "speed"] if args.section == "short"
            else ["parity", "heval20", "heval164", "speed"] if args.section == "all"
            else [args.section])
    if "parity" in todo:
        section_parity(model, cfg, tok, probs[:args.parity_n], args.parity_steps)
    if "heval20" in todo:
        section_heval20(model, cfg, tok, probs, args.max_new)
    if "heval164" in todo:
        section_heval164(model, cfg, tok, probs, args.max_new)
    if "speed" in todo:
        pr = args.speed_prompt or probs[0]["prompt"]
        section_speed(model, cfg, tok, pr, args.speed_gen)


if __name__ == "__main__":
    main()
