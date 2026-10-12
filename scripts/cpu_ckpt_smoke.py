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
DOMAIN_TEXTS = [
    ("code", "def add(a, b):\n    return a + b\n\nprint(add(2, 3))"),
    ("math", "A box contains 12 blocks. An operator removes 4 blocks. 8 blocks remain."),
    ("technical", "Install the part. Tighten the screws. Check the part for movement."),
    ("chinese", "系统将数据写入块。每个块有唯一编号。读取器按编号请求数据。"),
    ("dialogue", "Question: What is a log entry? Answer: A log entry records one event and its time."),
    ("prose", "The meeting starts at nine. The operator checks the agenda and opens the related document."),
]
CHAT_PROMPTS = [
    "Question: How do you check a service for errors? Answer with one short sentence.",
    "Question: What does a valid procedure include? Answer with one short sentence.",
]
FIXED_MC_QUESTIONS = [
    {"domain": "commonsense", "question": "What item opens a locked door?",
     "options": ["shoe", "key", "cup", "book"], "label": 1},
    {"domain": "science", "question": "At what temperature does water freeze at sea level?",
     "options": ["25 degrees Celsius", "50 degrees Celsius", "0 degrees Celsius", "100 degrees Celsius"], "label": 2},
    {"domain": "math", "question": "What is 7 multiplied by 8?",
     "options": ["48", "54", "64", "56"], "label": 3},
    {"domain": "code", "question": "Which keyword starts a function definition in Python?",
     "options": ["def", "func", "fn", "function"], "label": 0},
    {"domain": "chinese", "question": "用什么唯一标识一个数据块？",
     "options": ["颜色", "编号", "温度", "长度"], "label": 1},
    {"domain": "procedure", "question": "What must an operator do before the next step starts?",
     "options": ["remove the manual", "delete the log", "complete the current step", "disable the alarm"], "label": 2},
    {"domain": "logic", "question": "If all blocks have an identifier, and item X is a block, what does item X have?",
     "options": ["a wheel", "a battery", "a window", "an identifier"], "label": 3},
    {"domain": "safety", "question": "Which condition must trigger an alarm?",
     "options": ["measurement outside valid range", "normal log entry", "completed step", "matching key"], "label": 0},
]
GEN_MAX_NEW = 32
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
def score_domain_texts(torch, model, tok, max_len=96):
    rows = []
    for domain, text in DOMAIN_TEXTS:
        ids = encode_text(tok, text)[:max_len]
        x = torch.tensor([ids], dtype=torch.long)
        logits, _ = model(x)
        targets = x[:, 1:]
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        rows.append({
            "domain": domain,
            "tokens": len(ids),
            "mean_nll": round(float(nll.mean()), 6),
        })
    return rows


def repetition_stats(ids, text):
    count = len(ids)
    unique_fraction = round(len(set(ids)) / count, 6) if count else None
    max_run = 0
    run = 0
    previous = None
    for token in ids:
        run = run + 1 if token == previous else 1
        max_run = max(max_run, run)
        previous = token
    grams = [tuple(ids[i:i + 4]) for i in range(max(0, count - 3))]
    duplicate_4gram_fraction = None
    if grams:
        duplicate_4gram_fraction = round(
            (len(grams) - len(set(grams))) / len(grams), 6
        )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    duplicate_line_fraction = None
    if lines:
        duplicate_line_fraction = round(
            (len(lines) - len(set(lines))) / len(lines), 6
        )
    return {
        "tokens": count,
        "unique_token_fraction": unique_fraction,
        "max_token_run": max_run,
        "duplicate_4gram_fraction": duplicate_4gram_fraction,
        "duplicate_line_fraction": duplicate_line_fraction,
    }


@torch.no_grad()
def generate_chats(torch, model, tok, max_new=GEN_MAX_NEW):
    rows = []
    for prompt_id, prompt in enumerate(CHAT_PROMPTS):
        prompt_ids = encode_text(tok, prompt)
        generated = []
        stop_reason = "max_new"
        for _step in range(max_new):
            x = torch.tensor([prompt_ids + generated], dtype=torch.long)
            logits, _ = model(x)
            next_logits = logits[0, -1]
            if not torch.isfinite(next_logits).all():
                raise ValueError(f"non-finite chat logits at prompt {prompt_id}")
            next_id = int(next_logits.argmax())
            if next_id == 1:
                stop_reason = "eos"
                break
            generated.append(next_id)
        text = tok.decode(generated)
        stats = repetition_stats(generated, text)
        stats.update({
            "id": prompt_id,
            "stop_reason": stop_reason,
            "text": text,
        })
        rows.append(stats)
    return rows


def extended_warnings(chat_rows, domain_rows):
    warnings = []
    if chat_rows and all(row.get("tokens", 0) == 0 for row in chat_rows):
        warnings.append("all chat outputs are empty")
    if chat_rows and all(row.get("stop_reason") == "max_new" for row in chat_rows):
        warnings.append("all chat outputs reach max length")
    for row in chat_rows:
        prefix = f"chat{row.get('id')}"
        if row.get("tokens", 0) == 0:
            warnings.append(f"{prefix} output is empty")
        if row.get("unique_token_fraction") is not None and row["unique_token_fraction"] < 0.25:
            warnings.append(f"{prefix} token diversity is low")
        if row.get("max_token_run", 0) >= 8:
            warnings.append(f"{prefix} has a long repeated token run")
        if row.get("duplicate_4gram_fraction") is not None and row["duplicate_4gram_fraction"] >= 0.5:
            warnings.append(f"{prefix} has high repeated 4-gram rate")
    for row in domain_rows:
        if row.get("mean_nll") is not None and row["mean_nll"] > 8:
            warnings.append(f"domain {row['domain']} has high NLL")
    return warnings


@torch.no_grad()
def score_fixed_mc(torch, model, tok):
    letters = ["A", "B", "C", "D"]
    letter_tokens = [encode_text(tok, " " + letter)[-1] for letter in letters]
    rows = []
    for item in FIXED_MC_QUESTIONS:
        prompt = "Question: " + item["question"] + "\n"
        for letter, option in zip(letters, item["options"]):
            prompt += f"{letter}. {option}\n"
        prompt += "Answer:"
        ids = encode_text(tok, prompt)
        x = torch.tensor([ids], dtype=torch.long)
        logits, _ = model(x)
        next_log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
        if not torch.isfinite(next_log_probs).all():
            raise ValueError(f"non-finite MC logits for {item['domain']}")
        scores = [round(float(next_log_probs[token]), 6) for token in letter_tokens]
        pred = int(max(range(len(scores)), key=lambda idx: scores[idx]))
        correct = pred == item["label"]
        rows.append({
            "domain": item["domain"],
            "pred": pred,
            "label": item["label"],
            "correct": correct,
            "scores": scores,
            "margin": round(scores[item["label"]] - max(
                score for idx, score in enumerate(scores) if idx != item["label"]
            ), 6),
        })
    correct_count = sum(1 for row in rows if row["correct"])
    summary = {
        "correct": correct_count,
        "total": len(rows),
        "accuracy": round(100 * correct_count / len(rows), 2) if rows else None,
        "items": rows,
    }
    return summary


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
        domain_rows = score_domain_texts(torch, model, tok)
        chat_rows = generate_chats(torch, model, tok)
        fixed_mc = score_fixed_mc(torch, model, tok)
        canary = score_canary(torch, model, tok)
        row["fixed_text"] = text_rows
        row["domains"] = domain_rows
        row["chat"] = chat_rows
        row["fixed_mc"] = fixed_mc
        row["canary"] = canary
        row["mean_nll"] = round(sum(x["mean_nll"] for x in text_rows) / len(text_rows), 6)
        row["multi_domain_mean_nll"] = round(
            sum(x["mean_nll"] for x in domain_rows) / len(domain_rows), 6
        )
        warnings = extended_warnings(chat_rows, domain_rows)
        if fixed_mc.get("correct", 0) <= 2:
            warnings.append("fixed MC accuracy is at or below chance")
        if warnings:
            row["warnings"] = warnings
        # Canary and chat outputs are signals, not hard stops during early pretraining.
        # Hard failures use configuration and finite-state checks.
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
