#!/usr/bin/env python3
"""CPU smoke test for periodic training checkpoints.

The script loads one checkpoint on CPU, checks finite weights and Engram
configuration, scores fixed short texts, and scores one fixed canary. It writes
one JSONL row per checkpoint. It never uses CUDA. It never restarts training.

Usage:
    CUDA_VISIBLE_DEVICES="" python3 scripts/cpu_ckpt_smoke.py --name s3a_seg1 --once
    CUDA_VISIBLE_DEVICES="" python3 scripts/cpu_ckpt_smoke.py --name s3a_seg1 --watch
"""
import argparse
import gc
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENGRAM = "1,5,9,13,17,21"
FIXED_TEXTS = [
    "The system stores data in blocks. Each block has a unique identifier. The reader requests one block by identifier.",
    "A procedure has three steps. The operator completes step one, step two, and then step three in sequence.",
    "The cache holds a key and a value. The lookup returns the value when the key matches.",
    "If the measurement is outside the valid range, the monitor sets an alarm and writes a log entry.",
]
CANARY_CODE = "7319"
CANARY_WRONG = ["1234", "2856", "9041"]
TARGET_MODULE_RE = re.compile(r"^(layers|engrams)\.(\d+)$")


def set_cpu_env(threads):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)
    return torch


def checkpoint_step(path):
    m = re.search(r"\.step(\d+)$", path)
    return int(m.group(1)) if m else None


def ledger_path(name):
    return ROOT / "runs" / f"cpu_ckpt_smoke_{name}.jsonl"


def done_checkpoints(name):
    out = set()
    path = ledger_path(name)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("status") == "ok" and row.get("ckpt"):
            out.add(row["ckpt"])
    return out


def discover(name):
    out = []
    for p in glob.glob(str(ROOT / f"ckpt_{name}.pt.step*")):
        step = checkpoint_step(p)
        if step is not None:
            out.append((step, p))
    return sorted(out)


def encode_text(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def force_cpu_fp32(model):
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            tensor.data = tensor.data.float()


def finite_stats(torch, model):
    finite = total = 0
    worst_abs = 0.0
    for tensor in model.parameters():
        t = tensor.detach()
        if not t.is_floating_point():
            continue
        mask = torch.isfinite(t)
        finite += int(mask.sum())
        total += int(mask.numel())
        if finite:
            worst_abs = max(worst_abs, float(t[mask].abs().max().cpu()))
    return {
        "finite_param_fraction": round(finite / total, 6) if total else None,
        "max_abs_param": round(worst_abs, 3),
    }


def target_modules(model):
    rows = []
    for name, module in model.named_modules():
        match = TARGET_MODULE_RE.fullmatch(name)
        if match:
            rows.append((int(match.group(2)), name, module))
    return sorted(rows, key=lambda x: (x[0], x[1]))


def flatten_tensors(torch, obj):
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (tuple, list)):
        out = []
        for item in obj:
            out.extend(flatten_tensors(torch, item))
        return out
    if isinstance(obj, dict):
        out = []
        for item in obj.values():
            out.extend(flatten_tensors(torch, item))
        return out
    return []


def persistent_buffers(module):
    out = []
    seen = set()
    for _name, submodule in module.named_modules():
        non_persistent = getattr(submodule, "_non_persistent_buffers_set", set())
        for buffer_name, buffer in submodule._buffers.items():
            if buffer is None or buffer_name in non_persistent:
                continue
            key = (buffer.data_ptr(), tuple(buffer.shape))
            if key in seen:
                continue
            seen.add(key)
            out.append(buffer)
    return out


def module_state_stats(torch, module):
    finite = total = 0
    square_sum = 0.0
    max_abs = 0.0
    float_tensors = 0
    all_tensors = 0
    tensors = list(module.parameters()) + persistent_buffers(module)
    for tensor in tensors:
        all_tensors += 1
        if not tensor.is_floating_point():
            continue
        float_tensors += 1
        value = tensor.detach().float()
        mask = torch.isfinite(value)
        finite += int(mask.sum())
        total += int(mask.numel())
        good = value[mask]
        if good.numel():
            square_sum += float(good.square().sum())
            max_abs = max(max_abs, float(good.abs().max()))
    return {
        "finite_fraction": round(finite / total, 6) if total else None,
        "norm": round(square_sum ** 0.5, 3),
        "max_abs": round(max_abs, 3),
        "float_tensors": float_tensors,
        "all_tensors": all_tensors,
    }


def tensor_collection_stats(torch, tensors):
    finite = total = 0
    abs_sum = 0.0
    max_abs = 0.0
    float_tensors = 0
    for tensor in tensors:
        if not tensor.is_floating_point():
            continue
        float_tensors += 1
        value = tensor.detach().float()
        mask = torch.isfinite(value)
        finite += int(mask.sum())
        total += int(mask.numel())
        good = value[mask]
        abs_sum += float(good.abs().sum())
        if good.numel():
            max_abs = max(max_abs, float(good.abs().max()))
    return {
        "finite_fraction": round(finite / total, 6) if total else None,
        "mean_abs": round(abs_sum / finite, 6) if finite else None,
        "max_abs": round(max_abs, 3),
        "tensor_count": float_tensors,
    }


@torch.no_grad()
def collect_activations(torch, model, tok, max_len=96):
    modules = target_modules(model)
    records = {}

    def hook(name):
        def _hook(_module, _inputs, output):
            records[name] = tensor_collection_stats(
                torch, flatten_tensors(torch, output)
            )
        return _hook

    handles = [module.register_forward_hook(hook(name)) for _, name, module in modules]
    try:
        ids = encode_text(tok, FIXED_TEXTS[0])[:max_len]
        x = torch.tensor([ids], dtype=torch.long)
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    return records


def layer_rows(torch, model, activations):
    out = []
    for _idx, name, module in target_modules(model):
        out.append({
            "module": name,
            "state": module_state_stats(torch, module),
            "activation": activations.get(name),
        })
    return out


def layer_errors(rows):
    errors = []
    for row in rows:
        name = row["module"]
        state = row["state"]
        activation = row["activation"]
        if state.get("float_tensors", 0) == 0:
            errors.append(f"{name} has no floating state")
        if state.get("finite_fraction") != 1.0:
            errors.append(f"{name} non-finite state")
        if activation is None:
            errors.append(f"{name} missing activation")
            continue
        if activation.get("tensor_count", 0) == 0:
            errors.append(f"{name} has no floating activation")
        if activation.get("finite_fraction") != 1.0:
            errors.append(f"{name} non-finite activation")
        if activation.get("mean_abs") in (None, 0):
            errors.append(f"{name} zero activation")
    return errors


@torch.no_grad()
def score_texts(torch, model, tok, max_len=128):
    rows = []
    for i, text in enumerate(FIXED_TEXTS):
        ids = encode_text(tok, text)[:max_len]
        x = torch.tensor([ids], dtype=torch.long)
        logits, _ = model(x)
        targets = x[:, 1:]
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        rows.append({
            "id": i,
            "tokens": len(ids),
            "mean_nll": round(float(nll.mean()), 6),
        })
    return rows


@torch.no_grad()
def score_canary(torch, model, tok):
    filler = " The procedure writes one log line after each operation." * 3
    prompt = (
        "Read the fact. The access code is " + CANARY_CODE + "." + filler
        + " What is the access code? Answer with only the four digits."
    )
    options = [" " + CANARY_CODE] + [" " + x for x in CANARY_WRONG]
    p_ids = encode_text(tok, prompt)
    scores = []
    for option in options:
        o_ids = encode_text(tok, option)
        ids = p_ids + o_ids
        x = torch.tensor([ids], dtype=torch.long)
        logits, _ = model(x)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        total = 0.0
        for pos, token in enumerate(o_ids, start=len(p_ids) - 1):
            total += float(log_probs[0, pos, token])
        scores.append(round(total, 6))
    pred = int(max(range(len(scores)), key=lambda i: scores[i]))
    return {
        "scores": scores,
        "pred": pred,
        "correct": pred == 0,
        "margin": round(scores[0] - max(scores[1:]), 6),
    }


def validate_config(raw_ckpt, path_step):
    cfg = raw_ckpt.get("cfg") or {}
    errors = []
    if raw_ckpt.get("step") != path_step:
        errors.append(f"checkpoint step {raw_ckpt.get('step')} != file step {path_step}")
    if str(cfg.get("v42_engram")) != EXPECTED_ENGRAM:
        errors.append(f"v42_engram={cfg.get('v42_engram')} expected {EXPECTED_ENGRAM}")
    if cfg.get("arch") != "v42":
        errors.append(f"arch={cfg.get('arch')} expected v42")
    return cfg, errors


def run_one(path, step, threads):
    torch = set_cpu_env(threads)
    t0 = time.time()
    row = {
        "ckpt": os.path.basename(path),
        "step": step,
        "device": "cpu",
        "threads": threads,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    model = None
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        cfg, cfg_errors = validate_config(raw, step)
        row["cfg"] = {
            "arch": cfg.get("arch"),
            "v42_engram": cfg.get("v42_engram"),
            "vocab": cfg.get("vocab"),
            "vocab_real": cfg.get("vocab_real"),
        }
        del raw
        if cfg_errors:
            row.update(status="fail", errors=cfg_errors)
            return row

        sys.path.insert(0, str(ROOT))
        from scripts.loader import load_checkpoint
        from tokenizers import Tokenizer

        model, loaded_cfg = load_checkpoint(path, device="cpu", claim=False)
        force_cpu_fp32(model)
        tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer.json"))
        row["finite"] = finite_stats(torch, model)
        if row["finite"]["finite_param_fraction"] != 1.0:
            row.update(status="fail", errors=["non-finite parameter found"])
            return row
        activations = collect_activations(torch, model, tok)
        row["layers"] = layer_rows(torch, model, activations)
        layer_failures = layer_errors(row["layers"])
        if layer_failures:
            row.update(status="fail", errors=layer_failures)
            return row
        text_rows = score_texts(torch, model, tok)
        canary = score_canary(torch, model, tok)
        row["fixed_text"] = text_rows
        row["canary"] = canary
        row["mean_nll"] = round(sum(x["mean_nll"] for x in text_rows) / len(text_rows), 6)
        # The canary is a signal, not a hard stop during early pretraining.
        # Hard failures use configuration and finite-weight checks.
        row.update(status="ok")
        return row
    except Exception as exc:
        row.update(status="fail", errors=[
            f"{type(exc).__name__}: {exc}",
            traceback.format_exc()[-3000:],
        ])
        return row
    finally:
        if model is not None:
            del model
        gc.collect()
        row["secs"] = round(time.time() - t0, 2)


def append_row(name, row):
    path = ledger_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_pending(name, threads):
    done = done_checkpoints(name)
    pending = [(s, p) for s, p in discover(name) if os.path.basename(p) not in done]
    if not pending:
        return 0
    for step, path in pending:
        row = run_one(path, step, threads)
        append_row(name, row)
        print(json.dumps(row, sort_keys=True))
    return len(pending)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    if args.once or not args.watch:
        run_pending(args.name, args.threads)
        return
    while True:
        run_pending(args.name, args.threads)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
