#!/usr/bin/env python3
"""Shared contract for the E0-E6 evaluation ladder.

Owner: the main controller (see docs/standards/eval_ladder_v1_1009.md section 4).
Stage modules MUST NOT modify this file; they import it.

Everything a stage needs to be quotable later lives here: the fixed output paths,
the JSONL/summary/seal writers, the stop-reason rule (token id, never a decoded
string), the fail-closed `require`, and the loader wrapper that routes through
scripts/loader so the vocabulary identity check and the card claim both happen.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LADDER = ROOT / "runs" / "ladder"
TOKENIZER = ROOT / "data" / "tokenizer.json"

EXIT_OK = 0
EXIT_CRASH = 1
EXIT_REFUSE = 2
EXIT_GATE = 3
STAGES = ("e0", "e1", "e2", "e3", "e4", "e5", "e6")


class Refuse(Exception):
    """A precondition is not met. The stage prints REFUSE: and exits EXIT_REFUSE."""


def require(cond, msg):
    if not cond:
        raise Refuse(msg)


def ckpt_tag(ckpt):
    name = pathlib.Path(str(ckpt)).name
    for suffix in (".pt", ".safetensors"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def stage_paths(ckpt, stage):
    require(stage in STAGES, f"unknown stage {stage!r}, expected one of {STAGES}")
    d = LADDER / ckpt_tag(ckpt) / stage
    d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "items": d / "items.jsonl",
        "summary": d / "summary.json",
        "log": d / "log.txt",
        "seal": d / "seal.json",
    }


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_items(paths, rows):
    """One JSON object per line, no trailing blank line, keys sorted for diffing."""
    with open(paths["items"], "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def write_summary(paths, payload):
    payload = dict(payload)
    payload.setdefault("stage", paths["dir"].name)
    payload["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return payload


def seal(paths, **fields):
    """Provenance for the run that produced this stage's files.

    Reads what it can from the environment rather than trusting the caller:
    the command line, the pid, CUDA_VISIBLE_DEVICES.
    """
    rec = {
        "cmd": " ".join(sys.argv),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "host": os.uname().nodename,
    }
    rec.update(fields)
    with open(paths["seal"], "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return rec


def stop_reason(last_token_id, eos_id, im_end_id, n_new, max_new):
    """Why generation stopped, decided by TOKEN ID.

    The handover doc P2: a missing special symbol decoded from text is not evidence
    the model failed to stop. `n_new >= max_new` is the cap, and it is checked last
    so a token that is both the cap-th and a terminator reads as the terminator.
    """
    if last_token_id is not None and im_end_id is not None and last_token_id == im_end_id:
        return "im_end"
    if last_token_id is not None and eos_id is not None and last_token_id == eos_id:
        return "eos"
    if n_new >= max_new:
        return "max_new"
    return "other"


def load_for_eval(ckpt, device="cpu", dtype=None, claim=True):
    """(model, tok, meta) through scripts/loader: vocabulary identity + card claim.

    `claim` is passed straight through, so a caller that already holds the card
    under another name can say so instead of double-claiming.

    dtype defaults to bfloat16 on cuda and to the loader's default on cpu. Not a
    convenience: the CSA2 local branch runs flash-attn, whose `_flash_attn_fwd`
    asserts fp16/bf16/fp8, so an fp32 load dies inside v41f/docpack.py with an
    AssertionError that names the kernel rather than the missing dtype
    (measured 2026-10-09, E0's first run).

    The device index is checked against CUDA_VISIBLE_DEVICES before the load. With
    `CUDA_VISIBLE_DEVICES=5` the visible device is renumbered 0, so `--device cuda:5`
    makes torch.load raise "Attempting to deserialize object on CUDA device 5 but
    torch.cuda.device_count() is 1" from inside the deserializer -- an error that
    names the checkpoint rather than the flag (measured 2026-10-09, E5's first live
    run). The rule this enforces is AGENTS.md's: name the card with
    CUDA_VISIBLE_DEVICES, pass `cuda:0` to the code.
    """
    sys.path.insert(0, str(ROOT))
    require(pathlib.Path(ckpt).exists(), f"checkpoint {ckpt} does not exist")
    import torch
    from scripts.loader import load_checkpoint, load_tokenizer, vocab_fingerprint

    dev = str(device)
    if dev.startswith("cuda"):
        visible = [c for c in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if c.strip()]
        idx = dev.split(":", 1)[1] if ":" in dev else "0"
        if visible and idx.isdigit() and int(idx) >= len(visible):
            raise Refuse(
                f"--device {dev} with CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}: "
                f"only {len(visible)} device(s) are visible, so they are renumbered 0.."
                f"{len(visible) - 1} -- pass --device cuda:0 and name the physical card with "
                f"CUDA_VISIBLE_DEVICES (AGENTS.md: CUDA_VISIBLE_DEVICES, not cuda:N)")
    if dtype is None and dev.startswith("cuda"):
        dtype = torch.bfloat16

    model, cfg = load_checkpoint(str(ckpt), device=device, dtype=dtype, claim=claim)
    tok = load_tokenizer(str(TOKENIZER), cfg)
    head = read_ckpt_header(ckpt)
    step = head["step"] if head["step"] is not None else getattr(cfg, "step", None)
    kind = head["kind"] or getattr(cfg, "kind", None)
    meta = {
        "ckpt": str(ckpt),
        "ckpt_sha256": sha256_file(ckpt),
        "step": int(step) if step is not None else None,
        "step_source": "checkpoint" if head["step"] is not None else "cfg",
        "kind": kind,
        "vocab_id": vocab_fingerprint(tok),
        "vocab_real": int(tok.get_vocab_size()),
        "eos_id": int(tok.token_to_id("<|endoftext|>") or 1),
        "im_end_id": tok.token_to_id("<|im_end|>"),
        "im_start_id": tok.token_to_id("<|im_start|>"),
        "cfg": cfg,
    }
    return model, tok, meta


def read_ckpt_header(path):
    """step / kind / vocab_id, read from the checkpoint ITSELF.

    `load_checkpoint` returns the built cfg, and cfg carries no `step` for this
    checkpoint family (ckpt_v42s2.pt has step at the top level). Reading the cfg
    alone recorded step=0 in every generation-stage summary until codex reported it
    (eval/ladder_common.py:167, 2026-10-09). The read is mmap, so it maps the file
    instead of materialising it; the tensor views are dropped on return.
    """
    import torch

    ck = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    cfg = ck.get("cfg") or {}
    if not isinstance(cfg, dict):
        # vars(), not getattr(cfg, "__dict__", {}): harness check getattr_cfg_names_exist
        # reads a getattr whose name Cfg does not carry as a silent default that looks like
        # a measurement (eval/ladder_common.py:197 was one of its two hits, 2026-10-09).
        try:
            cfg = vars(cfg)
        except TypeError:
            cfg = {}
    step = None
    for src in (ck, cfg):
        for key in ("step", "total_steps"):
            if isinstance(src, dict) and src.get(key) is not None:
                step = int(src[key])
                break
        if step is not None:
            break
    out = {"step": step, "kind": cfg.get("kind"), "vocab_id": ck.get("vocab_id")}
    del ck
    return out


def verify_coverage(rows, expected_ids, label="rows"):
    """Exact ID set AND count. A merge that only counts lines hides a missing task.

    Returns (ok, detail). Never raises: the caller decides whether a coverage
    failure is a REFUSE (bad inputs) or a GATE failure (bad result).

    Ids are normalised to str for every sort. A row without an id is a missing id,
    not a crash: `sorted({None, "a"})` raises TypeError, so a malformed row used to
    kill the check instead of failing it (codex, 2026-10-09).
    """
    got = ["" if r.get("id") is None else str(r.get("id")) for r in rows]
    expected = [str(i) for i in expected_ids]
    dupes = sorted({i for i in got if got.count(i) > 1}) if len(got) != len(set(got)) else []
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    detail = {
        "label": label,
        "n_rows": len(got),
        "n_expected": len(expected),
        "blank_ids": got.count(""),
        "duplicates": dupes[:20],
        "missing": missing[:20],
        "extra": extra[:20],
    }
    ok = (not dupes and not missing and not extra
          and len(got) == len(expected) and detail["blank_ids"] == 0)
    return ok, detail


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_stage(stage, fn, ckpt=None):
    """Dispatch wrapper: uniform exit codes and a log file beside the artifacts.

    fn(paths) -> summary dict. Exceptions map to the contract's exit codes:
    Refuse -> 2, anything else -> 1 with the traceback in log.txt.
    """
    import traceback

    paths = stage_paths(ckpt, stage) if ckpt else None
    if paths is not None:
        logf = open(paths["log"], "a", encoding="utf-8", buffering=1)
    else:
        logf = sys.stdout

    class _Tee:
        def write(self, s):
            logf.write(s)
            if logf is not sys.stdout:
                sys.stdout.write(s)

        def flush(self):
            logf.flush()
            sys.stdout.flush()

    tee = _Tee()
    started = time.time()
    try:
        summary = fn(paths, tee) or {}
        if paths is not None:
            summary.setdefault("elapsed_s", round(time.time() - started, 1))
            summary.setdefault("exit_code", EXIT_OK)
            write_summary(paths, summary)
        return summary.get("exit_code", EXIT_OK)
    except Refuse as exc:
        tee.write(f"REFUSE: {exc}\n")
        if paths is not None:
            write_summary(paths, {"refused": str(exc), "exit_code": EXIT_REFUSE,
                                  "elapsed_s": round(time.time() - started, 1)})
        return EXIT_REFUSE
    except SystemExit as exc:
        # scripts/loader.py exits the process when CUDA_VISIBLE_DEVICES is unset: it
        # cannot name the card it is about to take, so it refuses. SystemExit is a
        # BaseException, so it sailed past `except Exception` and the caller's exit
        # code became 1 with no summary at all (codex, eval/ladder_common.py:162,
        # 2026-10-09; observed live on E0's first run). A refusal is exit 2.
        code = exc.code if isinstance(exc.code, int) else 1
        msg = (f"a library refused: sys.exit({code}) -- most often scripts/loader.py "
               f"refusing because CUDA_VISIBLE_DEVICES is unset, so it cannot name the card "
               f"it is about to claim")
        tee.write(f"REFUSE: {msg}\n")
        if paths is not None:
            write_summary(paths, {"refused": msg, "sys_exit": code,
                                  "exit_code": EXIT_REFUSE,
                                  "elapsed_s": round(time.time() - started, 1)})
        return EXIT_REFUSE
    except Exception:
        traceback.print_exc(file=logf)
        if paths is not None:
            write_summary(paths, {"crashed": traceback.format_exc()[-2000:],
                                  "exit_code": EXIT_CRASH})
        return EXIT_CRASH
    finally:
        if logf is not sys.stdout:
            logf.close()
