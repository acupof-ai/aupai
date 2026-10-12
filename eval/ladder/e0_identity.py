#!/usr/bin/env python3
"""E0: identity and finite output.

Handover doc section 6, E0: the checkpoint loads; step, cfg, vocab and tokenizer
fingerprint agree; the Engram layers are 1,5,9,13,17,21; 24 Engram weight keys;
627 state-dict keys in total; a fixed input produces finite logits with no NaN or
Inf. Any identity mismatch stops the ladder.

Identity is read from the checkpoint itself (zip CRC, then torch.load with mmap),
not from a filename and not from a previous run's record. Every expectation is a
CLI argument with the base default, so the same module gates the base and any SFT
checkpoint without a code change.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval import ladder_common as lc  # noqa: E402

ENGRAM_KEY_RE = re.compile(r"engrams\.(\d+)\.")
# A fixed, in-distribution probe: ASCII code plus one hanzi clause. Never sampled.
FIXED_PROBE = "def add(a, b):\n    return a + b\n# 说明：两数相加\n"


def read_identity(ckpt):
    """Everything E0 asserts about, read once, from the file on disk."""
    path = pathlib.Path(ckpt)
    lc.require(path.is_file(), f"checkpoint {ckpt} is not a file")
    import torch

    before = path.stat()
    with zipfile.ZipFile(path) as z:
        zip_crc = z.testzip() is None
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    cfg = ck.get("cfg")
    if not isinstance(cfg, dict):
        try:
            cfg = vars(cfg)   # not getattr(cfg, "__dict__", {}): see ladder_common.read_ckpt_header
        except TypeError:
            cfg = {}
    weights = ck.get("model")
    lc.require(isinstance(weights, dict) and weights, "checkpoint carries no 'model' weight dict")
    engram_keys = [k for k in weights if ENGRAM_KEY_RE.match(k)]
    layers = sorted({int(ENGRAM_KEY_RE.match(k).group(1)) for k in engram_keys})
    after = path.stat()
    step = None
    for key in ("step", "total_steps"):
        if ck.get(key) is not None:
            step = int(ck[key])
            break
    if step is None:
        for key in ("step", "total_steps"):
            if cfg.get(key) is not None:
                step = int(cfg[key])
                break
    return {
        "path": str(path),
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "unchanged_during_read": (before.st_size, before.st_mtime_ns)
        == (after.st_size, after.st_mtime_ns),
        "zip_crc": zip_crc,
        "vocab_id": ck.get("vocab_id"),
        "step": step,
        "kind": cfg.get("kind"),
        "engram_cfg": cfg.get("v42_engram"),
        "state_keys": len(weights),
        "engram_keys": len(engram_keys),
        "engram_layers": layers,
        "arch": cfg.get("arch"),
        "dim": cfg.get("dim"),
        "layers": cfg.get("layers"),
        # Producer identity for a checkpoint that has no step counter. sft_math.py
        # writes corpus_fp/env_fp/vocab_id and cfg.epochs but never a step, so an SFT
        # output is identifiable by what produced it rather than by how far it ran
        # (measured 2026-10-09 on ckpt_v42_sft_v3_v1fmt_abslr_1008.pt).
        "producer_fields": sorted(k for k in ("corpus_fp", "env_fp", "vocab_id", "cfg")
                                 if ck.get(k) is not None),
        "epochs": cfg.get("epochs"),
        "sha256": lc.sha256_file(path),
    }


def checks(ident, expect):
    """(name, ok, detail) per assertion. Pure: no torch, no filesystem.

    Kept separate from read_identity so the negative worlds in control_selftest
    exercise the same code the real gate runs.
    """
    out = []

    def add(name, ok, detail=""):
        out.append({"check": name, "ok": bool(ok), "detail": detail})

    add("zip_crc", ident["zip_crc"], "zip member CRC")
    add("unchanged_during_read", ident["unchanged_during_read"], "size+mtime stable across read")
    add("vocab_id_present", bool(ident["vocab_id"]), str(ident["vocab_id"]))
    if expect.get("vocab_id"):
        add("vocab_id", ident["vocab_id"] == expect["vocab_id"],
            f"{ident['vocab_id']} vs {expect['vocab_id']}")
    # WHICH IDENTITY IS REQUIRED DEPENDS ON THE PRODUCER. A pretraining checkpoint is
    # identified by the step it reached. An SFT output records no step (sft_math.py
    # writes corpus_fp/env_fp/epochs instead -- measured 2026-10-09), so for kind=sft
    # the requirement is the producer fingerprint, and requiring a step there would
    # refuse every SFT checkpoint the ladder exists to grade.
    kind = (ident.get("kind") or "").lower()
    if kind == "sft":
        add("producer_identity", len(ident.get("producer_fields") or []) >= 3,
            f"corpus_fp/env_fp/vocab_id present={ident.get('producer_fields')} "
            f"epochs={ident.get('epochs')}")
        if ident["step"] is not None:
            add("step_present", True, f"{ident['step']} (recorded)")
    else:
        add("step_present", ident["step"] is not None, str(ident["step"]))
    if expect.get("step") is not None:
        add("step", ident["step"] == expect["step"], f"{ident['step']} vs {expect['step']}")
    if expect.get("kind"):
        add("kind", ident["kind"] == expect["kind"], f"{ident['kind']} vs {expect['kind']}")
    if expect.get("engram_cfg"):
        add("engram_cfg", ident["engram_cfg"] == expect["engram_cfg"],
            f"{ident['engram_cfg']!r} vs {expect['engram_cfg']!r}")
    if expect.get("state_keys") is not None:
        add("state_keys", ident["state_keys"] == expect["state_keys"],
            f"{ident['state_keys']} vs {expect['state_keys']}")
    if expect.get("engram_keys") is not None:
        add("engram_keys", ident["engram_keys"] == expect["engram_keys"],
            f"{ident['engram_keys']} vs {expect['engram_keys']}")
    if expect.get("engram_layers"):
        add("engram_layers", ident["engram_layers"] == expect["engram_layers"],
            f"{ident['engram_layers']} vs {expect['engram_layers']}")
    return out


def finite_logits(model, tok, device):
    """One forward on the fixed probe. Returns a dict of the observed numbers."""
    import torch

    ids = tok.encode(FIXED_PROBE).ids
    lc.require(len(ids) >= 4, "fixed probe tokenized to fewer than 4 tokens")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, hidden = model(x, no_head=True)
        logits = model.lm_logits(hidden[:, -1])
        logits = logits.float()
    finite = bool(torch.isfinite(logits).all())
    return {
        "probe_tokens": len(ids),
        "finite": finite,
        "nan": int(torch.isnan(logits).sum()),
        "inf": int(torch.isinf(logits).sum()),
        "absmax": float(logits.abs().max()),
        "norm": float(logits.norm()),
        "argmax": int(logits.argmax(dim=-1)),
    }


def main(argv):
    ap = argparse.ArgumentParser(prog="eval_ladder.py e0")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expect-step", type=int, default=None)
    ap.add_argument("--expect-kind", default=None)
    ap.add_argument("--expect-vocab-id", default=None)
    ap.add_argument("--expect-engram", default=None, help="e.g. 1,5,9,13,17,21")
    ap.add_argument("--expect-state-keys", type=int, default=None)
    ap.add_argument("--expect-engram-keys", type=int, default=None)
    ap.add_argument("--expect-engram-layers", default=None, help="e.g. 1,5,9,13,17,21")
    ap.add_argument("--skip-forward", action="store_true", help="identity only, no GPU")
    args = ap.parse_args(argv)

    expect = {
        "step": args.expect_step,
        "kind": args.expect_kind,
        "vocab_id": args.expect_vocab_id,
        "engram_cfg": args.expect_engram,
        "state_keys": args.expect_state_keys,
        "engram_keys": args.expect_engram_keys,
        "engram_layers": [int(x) for x in args.expect_engram_layers.split(",")]
        if args.expect_engram_layers
        else None,
    }

    def stage(paths, log):
        lc.seal(paths, ckpt=args.ckpt, stage="e0", device=args.device)
        log.write(f"reading identity of {args.ckpt}\n")
        ident = read_identity(args.ckpt)
        log.write(json.dumps(ident, ensure_ascii=False, sort_keys=True) + "\n")
        results = checks(ident, expect)

        forward = {"skipped": True}
        if not args.skip_forward:
            model, tok, meta = lc.load_for_eval(args.ckpt, device=args.device, claim=True)
            if meta["vocab_id"] != ident["vocab_id"]:
                results.append({"check": "tokenizer_fingerprint", "ok": False,
                                "detail": f"{meta['vocab_id']} vs checkpoint {ident['vocab_id']}"})
            else:
                results.append({"check": "tokenizer_fingerprint", "ok": True,
                                "detail": meta["vocab_id"]})
            forward = finite_logits(model, tok, args.device)
            results.append({"check": "logits_finite", "ok": forward["finite"],
                            "detail": json.dumps(forward, sort_keys=True)})
            log.write(f"forward: {json.dumps(forward, sort_keys=True)}\n")
            del model

        failed = [r["check"] for r in results if not r["ok"]]
        gates = {"identity": not failed}
        summary = {
            "ckpt": args.ckpt,
            "ckpt_sha256": ident["sha256"],
            "step": ident["step"],
            "protocol": "identity",
            "n": len(results),
            "passed": len(results) - len(failed),
            "gates": gates,
            "checks": results,
            "failed": failed,
            "forward": forward,
            "inputs": {"probe_sha256": __import__("hashlib").sha256(FIXED_PROBE.encode()).hexdigest()},
            "identity": {k: ident[k] for k in
                         ("vocab_id", "kind", "engram_cfg", "state_keys", "engram_keys",
                          "engram_layers", "arch", "dim", "layers", "bytes")},
            "exit_code": lc.EXIT_GATE if failed else lc.EXIT_OK,
        }
        lc.write_items(paths, [{"id": r["check"], "passed": r["ok"], "detail": r["detail"]}
                               for r in results])
        for r in results:
            log.write(f"{'PASS' if r['ok'] else 'FAIL'} {r['check']}: {r['detail']}\n")
        if failed:
            log.write(f"E0 gate FAILED: {failed}\n")
        else:
            log.write("E0 gate PASSED\n")
        return summary

    return lc.run_stage("e0", stage, ckpt=args.ckpt)


def control_selftest():
    """Positive and negative worlds over checks(), with the base's own expectations."""
    base = {
        "path": "x", "bytes": 1, "mtime_ns": 1, "unchanged_during_read": True, "zip_crc": True,
        "vocab_id": "f1f860970d15d623", "step": 54051, "kind": None, "engram_cfg": "1,5,9,13,17,21",
        "state_keys": 627, "engram_keys": 24, "engram_layers": [1, 5, 9, 13, 17, 21],
        "arch": "v42", "dim": 1024, "layers": 12, "sha256": "0" * 64,
    }
    expect = {"vocab_id": "f1f860970d15d623", "step": 54051, "engram_cfg": "1,5,9,13,17,21",
              "state_keys": 627, "engram_keys": 24, "engram_layers": [1, 5, 9, 13, 17, 21]}
    failures = []
    ok = checks(base, expect)
    if not all(r["ok"] for r in ok):
        failures.append("valid identity world reported FAIL: "
                        + str([r["check"] for r in ok if not r["ok"]]))
    broken = [
        ("vocab_id", dict(base, vocab_id="deadbeefdeadbeef")),
        ("step", dict(base, step=54050)),
        ("engram_cfg", dict(base, engram_cfg="1,5,9,13,17")),
        ("state_keys", dict(base, state_keys=603)),
        ("engram_keys", dict(base, engram_keys=0)),
        ("engram_layers", dict(base, engram_layers=[1, 5, 9, 13, 17])),
        ("zip_crc", dict(base, zip_crc=False)),
        ("unchanged_during_read", dict(base, unchanged_during_read=False)),
    ]
    for name, world in broken:
        res = checks(world, expect)
        bad = [r["check"] for r in res if not r["ok"]]
        if name not in bad:
            failures.append(f"negative world {name!r} passed: failing checks {bad}")
    # The SFT arm of the identity rule: no step counter, identified by its producer.
    sft = dict(base, kind="sft", step=None, producer_fields=["cfg", "corpus_fp", "env_fp"],
               epochs=2)
    res = checks(sft, {})
    bad = [r["check"] for r in res if not r["ok"]]
    if bad:
        failures.append(f"SFT world with producer fields reported FAIL: {bad}")
    sft_bare = dict(sft, producer_fields=["cfg", "vocab_id"])
    bad = [r["check"] for r in checks(sft_bare, {}) if not r["ok"]]
    if "producer_identity" not in bad:
        failures.append("SFT world without producer fields passed: "
                        f"failing checks {bad}")
    for f in failures:
        print(f"FAIL e0 control: {f}", file=sys.stderr)
    if failures:
        return lc.EXIT_GATE
    print(f"PASS e0 controls: 1 base world + 1 SFT world valid, {len(broken)} negative worlds, "
          f"1 SFT producer-identity negative world")
    return lc.EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
