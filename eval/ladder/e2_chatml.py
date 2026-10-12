#!/usr/bin/env python3
"""E2: ChatML format and stop behaviour. SFT checkpoints only.

Handover doc section 6, E2, plus its section 5 P2 ("a missing special symbol must not
be judged as a failure to stop"): ChatML prefix correct; the model does not emit the
user turn; it does not repeat the system or user text; it stops with `<|im_end|>`;
empty output is detectable; a repeat loop is detectable; the stop reason is decided by
TOKEN ID, never by decoding the text and searching for a marker.

Subject. An SFT checkpoint ONLY. A base checkpoint REFUSEs here (exit 2) instead of
being scored: the ChatML prefix occurs 0 times in the pretraining corpus, so running
this on a base checkpoint measures the response to an unseen prefix, which is the
error AGENTS.md records (facts/base_eval.json#be.l1_fewshot_p324: +38.2pt of pure
format effect). `cfg.kind` is the discriminator, read from the checkpoint, and the
kind is recorded in the summary either way.

Owner: the main controller (unassigned in the contract's section 4 and filled here so
the E0-E6 entry is complete). Controls cover one positive world and six negative
worlds, each landing on its own judgment.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval import ladder_common as lc  # noqa: E402

DEFAULT_MAX_NEW = 256
PROMPTS = ROOT / "data" / "eval" / "ladder" / "e2_prompts.jsonl"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
# A token-level repeat rule, not a text one: the longest repeated n-gram's share of the
# generated tokens. Text rules miss a loop that decodes differently at every boundary.
REPEAT_NGRAM = 12
REPEAT_SHARE_MAX = 0.30
# The stop check is a token-id check with a bounded grace: a model that emits a few
# trailing tokens after <|im_end|> still stopped, but anything past this wrote a new turn.
STOP_SLACK_TOKENS = 0


def load_prompt_file(path):
    """Read the prompt set. Both shapes the two roles use, same as E1.

    Shape 1: this stage's operational file, one JSON object per line with
    {"id","question"} (optionally "system"). Shape 2: the auditor's cross-check set,
    {"prompts": [...]} as a single JSON object.
    """
    path = pathlib.Path(path)
    lc.require(path.is_file(), f"prompt file {path} is missing")
    raw = path.read_text(encoding="utf-8").strip()
    items = []
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            entries = obj.get("prompts")
            lc.require(isinstance(entries, list), f"{path}: 'prompts' must be a list")
            for i, entry in enumerate(entries):
                if isinstance(entry, str):
                    items.append({"id": f"p{i + 1:02d}", "question": entry})
                    continue
                lc.require(isinstance(entry, dict), f"{path}:prompts[{i}] is not an object")
                q = entry.get("question") or entry.get("prompt")
                lc.require(isinstance(q, str) and q.strip(), f"{path}:prompts[{i}] has no question")
                items.append({"id": str(entry.get("id") or f"p{i + 1:02d}"),
                              "question": q, "system": entry.get("system")})
        else:
            for line in raw.splitlines():
                if line.strip():
                    items.append(json.loads(line))
    else:
        for line in raw.splitlines():
            if line.strip():
                items.append(json.loads(line))
    lc.require(items, f"prompt file {path} holds no prompts")
    ids = [it["id"] for it in items]
    lc.require(len(ids) == len(set(ids)), f"prompt file {path} has duplicate ids")
    for it in items:
        lc.require(isinstance(it.get("question"), str) and it["question"].strip(),
                   f"{path}:{it.get('id')} question must be a non-empty string")
    return items


def build_prompt(tok, item):
    """The ChatML prefix, from the loader (single source of truth), as token ids."""
    from scripts.loader import format_prompt

    text = format_prompt(item["question"], item.get("system"))
    return text, tok.encode(text).ids


def repeat_share(ids, n=REPEAT_NGRAM):
    """Share of generated tokens covered by the most repeated n-gram."""
    if len(ids) < n:
        return 0.0
    grams = {}
    for i in range(len(ids) - n + 1):
        grams[tuple(ids[i:i + n])] = grams.get(tuple(ids[i:i + n]), 0) + 1
    top = max(grams.values())
    return (top * n) / len(ids) if top > 1 else 0.0


def judge(tok, prompt_text, prompt_ids, gen_ids, max_new, cfg):
    """Every E2 judgment, each named so a control can be aimed at one of them."""
    ids = list(gen_ids)
    im_end_id = tok.token_to_id(IM_END)
    im_start_id = tok.token_to_id(IM_START)
    eos_id = tok.token_to_id("<|endoftext|>") or 1
    decoded = tok.decode(ids)
    stop = lc.stop_reason(ids[-1] if ids else None, eos_id, im_end_id, len(ids), max_new)

    checks = {}
    checks["prefix_ok"] = prompt_text.endswith(f"{IM_START}assistant\n")
    checks["nonempty"] = len(ids) > 0
    checks["stopped"] = stop in ("im_end", "eos")
    checks["no_extra_turn"] = im_start_id not in ids
    echo_probe = ""
    if f"{IM_START}user\n" in prompt_text:
        echo_probe = prompt_text.split(f"{IM_START}user\n")[-1].split(IM_END)[0][:32]
    checks["no_echo"] = (echo_probe not in decoded) if echo_probe else True
    checks["no_repeat"] = repeat_share(ids) <= REPEAT_SHARE_MAX
    failed = [k for k, v in checks.items() if not v]
    return {
        "id": None,
        "prompt": prompt_text,
        "response_raw": decoded,
        "parsed": None,
        "passed": not failed,
        "stop_reason": stop,
        "n_tokens": len(ids),
        "token_ids": ids,
        "fail_class": failed,
        "meta": {"max_new": max_new, "repeat_share": round(repeat_share(ids), 4),
                 "im_end_id": im_end_id, "eos_id": eos_id, "cfg": cfg},
    }


def control_worlds():
    """(name, expect_pass, world) over judge(), with ids standing in for tokens.

    The tokenizer is faked at the two methods judge() uses, so the worlds are exact:
    a control that fails for the wrong reason cannot pass here.
    """
    im_start_id, im_end_id, eos_id = 32760, 32764, 1
    echo_id = 77  # decodes to the user question, so an echo world can be built exactly

    class Tok:
        def token_to_id(self, s):
            return {IM_START: im_start_id, IM_END: im_end_id, "<|endoftext|>": eos_id}[s]

        def decode(self, ids):
            names = {im_start_id: IM_START, im_end_id: IM_END, eos_id: "<eos>",
                     echo_id: "Q"}
            return "".join(names.get(i, "x") for i in ids)

    tok = Tok()
    prompt_text = f"{IM_START}system\nS{IM_END}\n{IM_START}user\nQ{IM_END}\n{IM_START}assistant\n"
    prompt_ids = [1, 2, 3]
    good = [{"id": "ctrl:stop_respected", "expect_pass": True,
             "gen": [11, 12, im_end_id], "why": "two-token answer then im_end"}]
    bad = [
        ("ctrl:no_stop", [11, 12, 13], "hits the cap without a terminator"),
        ("ctrl:extra_turn", [11, im_end_id, im_start_id, 20], "writes a new turn after stopping"),
        ("ctrl:repeat_loop", [7] * (REPEAT_NGRAM * 4), "one n-gram repeated for the whole output"),
        ("ctrl:empty", [], "no tokens at all"),
        ("ctrl:echo_user", [echo_id, im_end_id], "repeats the user turn back"),
        ("ctrl:bad_prefix", [11, im_end_id], "the prompt is not a ChatML assistant turn"),
    ]
    worlds = good + [{"id": n, "expect_pass": False, "gen": g, "why": w} for n, g, w in bad]
    out = []
    for w in worlds:
        text = "user\nQ" if w["id"] == "ctrl:bad_prefix" else prompt_text
        res = judge(tok, text, prompt_ids, w["gen"], 3, {})
        out.append({"id": w["id"], "expect_pass": w["expect_pass"], "why": w["why"],
                    "ok": res["passed"] == w["expect_pass"], "failed": res["fail_class"],
                    "echo_world": res["response_raw"]})
    return out


def control_selftest():
    """CPU only: the worlds above, plus one negative per judgment."""
    failures = []
    worlds = control_worlds()
    for w in worlds:
        if not w["ok"]:
            failures.append(f"{w['id']}: expected pass={w['expect_pass']}, failed={w['failed']}")
    # Aim one world at each judgment: a judgment that can never fail is not a judgment.
    aimed = {k: [] for k in ("prefix_ok", "nonempty", "stopped", "no_extra_turn",
                             "no_echo", "no_repeat")}
    for w in worlds:
        for f in w["failed"]:
            aimed.setdefault(f, []).append(w["id"])
    for k, who in aimed.items():
        if not who:
            failures.append(f"no control world trips {k!r}")
    if not any(w["expect_pass"] for w in worlds):
        failures.append("no positive world")
    for f in failures:
        print(f"FAIL e2 control: {f}", file=sys.stderr)
    if failures:
        return lc.EXIT_GATE
    print(f"PASS e2 controls: 1 positive world, {len(worlds) - 1} negative worlds; "
          f"every judgment is tripped by at least one world")
    return lc.EXIT_OK


def main(argv):
    ap = argparse.ArgumentParser(prog="eval_ladder.py e2")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args(argv)

    def stage(paths, log):
        items = load_prompt_file(args.prompts)
        if args.limit:
            items = items[: args.limit]
        model, tok, meta = lc.load_for_eval(args.ckpt, device=args.device, claim=True)
        kind = str(getattr(meta["cfg"], "kind", "") or "").lower()
        lc.seal(paths, ckpt=args.ckpt, stage="e2", device=args.device, protocol="chatml",
                ckpt_kind=kind, prompts=str(args.prompts),
                prompts_sha256=lc.sha256_file(args.prompts), n_items=len(items))
        lc.require(kind == "sft",
                   f"{args.ckpt} has cfg.kind={kind!r}; E2 is the SFT-only stage and a base "
                   f"checkpoint's ChatML prefix measures an unseen sequence, not capability")
        log.write(f"E2 on {args.ckpt} kind={kind} n={len(items)} max_new={args.max_new}\n")
        from train import generate_batch

        rows, controls = [], {}
        if args.control:
            controls = {"worlds": control_worlds()}
            bad = [w["id"] for w in controls["worlds"] if not w["ok"]]
            lc.require(not bad, f"control worlds failed before the live run: {bad}")
        for item in items:
            text, ids = build_prompt(tok, item)
            gen = generate_batch(model, [ids], args.max_new, args.device, 0.0,
                                 tokenizer=tok)[0]
            row = judge(tok, text, ids, gen, args.max_new,
                        {"kind": kind, "max_new": args.max_new})
            row["id"] = item["id"]
            rows.append(row)
            log.write(f"{item['id']}: pass={row['passed']} stop={row['stop_reason']} "
                      f"n={row['n_tokens']} failed={row['fail_class']}\n")
        lc.write_items(paths, rows)
        stopped = sum(1 for r in rows if r["stop_reason"] in ("im_end", "eos"))
        format_ok = all(not ({"prefix_ok", "no_extra_turn", "no_echo"} & set(r["fail_class"]))
                        for r in rows)
        quality_ok = all(not ({"nonempty", "no_repeat"} & set(r["fail_class"])) for r in rows)
        gates = {
            "stopped": stopped == len(rows),
            "format": format_ok,
            "quality": quality_ok,
            "controls": all(w["ok"] for w in controls.get("worlds", [])) if controls else True,
        }
        failed = sorted({f for r in rows for f in r["fail_class"]})
        summary = {
            "ckpt": args.ckpt, "ckpt_sha256": meta["ckpt_sha256"], "step": meta["step"],
            "protocol": "chatml", "ckpt_kind": kind, "n": len(rows),
            "passed": sum(1 for r in rows if r["passed"]), "gates": gates,
            "stopped_rate": round(stopped / len(rows), 4) if rows else 0.0,
            "fail_classes": failed, "controls": controls,
            "inputs": {"prompts": str(args.prompts),
                       "prompts_sha256": lc.sha256_file(args.prompts),
                       "vocab_id": meta["vocab_id"]},
            "exit_code": lc.EXIT_OK if (gates["stopped"] and gates["format"]
                                        and gates["controls"]) else lc.EXIT_GATE,
        }
        log.write(f"E2 gates={gates} failed_classes={failed}\n")
        return summary

    return lc.run_stage("e2", stage, ckpt=args.ckpt)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
