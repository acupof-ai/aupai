#!/usr/bin/env python3
"""Sampled pass@k HumanEval, sharing the greedy path's judge and controls.

The p1 acceptance spec (docs/standards/p1_data_recipe.md): temp 0.2 / top-p 0.95
/ 20-sample pass@1, sharing ONE judge with the greedy path, both reported. This
file imports judge / truncate / run_control / STOPS from eval.humaneval_gen --
the baseline 0/164 and the SFT 3/164 were scored with that judge, and a sampled
number is only comparable on the same scorer. The generation loop is the greedy
loop with argmax replaced by nucleus sampling: same cfg.seq window, eos tid 1,
stop check every 16 tokens, max_new 280.

pass@1 estimator: c/n per problem -- for k=1 the unbiased estimator
1 - C(n-c,1)/C(n,1) = c/n. Greedy is n=1.

    CUDA_VISIBLE_DEVICES="" python3 eval/humaneval_sample.py --control   # scorer self-check, CPU
    python3 eval/humaneval_sample.py --ckpt <ckpt>                       # greedy + 20-sample arms
"""

import argparse
import contextlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eval_artifacts import attest, open_artifact  # noqa: E402

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# One judge, one stop list, one control set: imported, not copied.
from eval.humaneval_gen import (  # noqa: E402
    DATA_PATH,
    STOPS,
    TOK_PATH,
    judge,
    run_control,
    truncate,
)
from eval.shard import label as shard_label  # noqa: E402
from eval.shard import select as shard_select  # noqa: E402
from eval.shard import runs_full_control  # noqa: E402
from eval.shard import validate as shard_validate  # noqa: E402


def hits_real_stop(text, entry_point):
    """True if `text` ends on a STOP that is not the function re-declaring itself.

    This is the sampled arm's equivalent of humaneval_gen.truncate's 66-14 exemption,
    applied to the every-16-token in-loop check so generation is not stopped early on a
    `\\ndef <entry_point>(` self-hit. A naive `any(st in text)` -- the old sampled-arm test
    -- treats the re-declaration as a stop, which the greedy path explicitly does not.
    Mirrors truncate()'s scan, including re-scanning after a skipped self occurrence."""
    for st in STOPS:
        i = text.find(st)
        while i != -1:
            if (st == "\ndef "
                    and text[i:i + 12 + len(entry_point)].startswith(f"\ndef {entry_point}(")):
                i = text.find(st, i + 1)
                continue
            return True
    return False


def nucleus(logits, temperature, top_p):
    """logits: [1, V] -> [1, 1] sampled id. Standard nucleus: sort, keep the
    smallest set whose cumulative mass reaches top_p (always keeping rank 0)."""
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sp, dim=-1)
    keep = cum <= top_p
    keep[..., 1:] = keep[..., :-1].clone()
    keep[..., 0] = True
    sp = sp * keep
    sp = sp / sp.sum(dim=-1, keepdim=True)
    return si.gather(-1, torch.multinomial(sp, 1))


def merge_shards(paths, args):
    """Cardless merge of N shard preds into the one read point.

    Verifies the partition before any number: the headers' shard ids are exactly 0..k, and
    across them every problem appears in exactly one shard (eval/shard.py's fixed-position
    partition is what makes that exact). Refuses a gap or duplicate. Scores greedy and the
    n-sample pass@1 over the union; samples are independent draws, so per-problem c/n and
    its mean over problems combine from the rows each shard already judged.
    """
    if not paths:
        sys.exit(f"no shard files match {args.merge_glob!r}")
    headers, greedy, samples = {}, {}, {}
    n = None
    for path in paths:
        hdr = None
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ph = r.get("phase")
            if r.get("_header"):
                hdr = r
                continue
            if ph == "greedy":
                key = ("g", r["task_id"])
            elif ph == "sample":
                key = ("s", r["task_id"])
            else:
                continue  # sample_summary is a per-shard number; recomputed over the union
            if key in greedy or key in samples:
                sys.exit(f"duplicate {key} across shards in {path} -- refuse to double-count")
            if key[0] == "g":
                greedy[r["task_id"]] = r
            else:
                samples[r["task_id"]] = r
        if not hdr or not hdr.get("shard_n"):
            sys.exit(f"{path} is not a shard artifact (no shard_n in header)")
        si, sn, n = hdr["shard_i"], hdr["shard_n"], hdr["n"]
        if si in headers:
            sys.exit(f"shard {si} present in two files")
        headers[si] = hdr
    got = set(headers)
    if got != set(range(max(got) + 1)):
        sys.exit(f"shard set {sorted(got)} is not contiguous 0..k -- refuse to score a gap")
    sn = max(got) + 1
    # Every shard must carry the SAME problem set that shard_select assigns it. Greedy and
    # sampled must cover the same problems.
    if set(greedy) != set(samples):
        sys.exit("greedy and sample rows cover different problem sets across the shards")
    nprob = len(samples)
    # The union must be exactly eval/shard's partition over the full dataset: every problem
    # appears in exactly one of the contiguous shards.
    probs_all = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    want = {probs_all[i]["task_id"] for i in range(len(probs_all))}
    if set(samples) != want:
        miss = sorted(want - set(samples))[:3]
        extra = sorted(set(samples) - want)[:3]
        sys.exit(f"shards cover {nprob} problems, dataset has {len(want)} "
                 f"(missing e.g. {miss}, extra e.g. {extra})")
    gp = sum(int(r["ok"]) for r in greedy.values())
    ge = sum(int(r.get("empty")) for r in greedy.values())
    sp = sum(int(r["c"]) / n for r in samples.values())
    n_samples = nprob * n
    se = sum(1 for r in samples.values() for s in r["samples"] if not s["gen"].strip())
    print(f"merged {sn} shards, {nprob} problems, n={n}\n", flush=True)
    print(f"HUMANEVAL pass@1 (greedy) = {gp}/{nprob} = {100 * gp / nprob:.2f}%  "
          f"empty {ge}/{nprob}", flush=True)
    print(f"HUMANEVAL pass@1 (n={n}, temp={headers[0]['temp']}, top_p={headers[0]['top_p']}) = "
          f"{sp}/{nprob} = {100 * sp / nprob:.2f}%", flush=True)
    print(f"sampled empty completions = {se}/{n_samples} = {100 * se / n_samples:.1f}%", flush=True)
    print("(canonical control: shard 0 ran the full-set gate pre-generation; each shard's own "
          "sample_summary held its in-shard control)", flush=True)
    return 0


def _selftest():
    """Known answers for the sampled arm's stop/truncate handling. Cardless, no model.

    The red world: the sampled gen() used to (a) call truncate() with NO entry_point and
    (b) test stops with a naive substring check, so a completion that re-declares the
    function it was asked to write -- a normal completion-style answer -- was cut to empty
    and scored differently from the greedy arm on the identical string. These cases pin both
    the final truncation and the every-16-token in-loop stop to the entry_point-aware form
    the greedy path (humaneval_gen) uses."""
    # A self re-declaration carrying a real body. truncate WITHOUT entry_point (the old
    # sampled call) empties it; WITH entry_point the body survives -- this is the assertion
    # that reddens the old version.
    ep = "truncate_number"
    redecl = "\ndef truncate_number(number: float) -> float:\n    return number - int(number)\n\nx = 1"
    kept = truncate(redecl, ep)
    assert "return number - int(number)" in kept, (
        f"truncate cut the function's own re-declared def to {kept!r}; the sampled pass@1 "
        "would read a valid body as empty and disagree with the greedy scorer")
    legacy_empty = truncate(redecl)
    assert legacy_empty == "", "truncate(entry_point=None) must preserve the legacy verbatim cut"
    # The in-loop stop helper must NOT fire on the self-hit (even before the body is emitted,
    # at the "\ndef truncate_number(" prefix), but MUST fire on a DIFFERENT top-level def.
    assert not hits_real_stop("\ndef truncate_number(", ep), "self re-declaration read as a stop"
    assert hits_real_stop("\ndef other_function(", ep), "a different top-level def must stop"
    assert hits_real_stop("\nclass X:", ep), "a column-0 class must stop"
    # After the real body, a SECOND distinct def still cuts.
    assert hits_real_stop(kept + "\ndef other(", ep), "a later distinct def must stop"
    print("humaneval_sample selftest OK: self re-declared def keeps its body (truncate + "
          "in-loop stop entry_point-aware); a different top-level def still cuts")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--max_new", type=int, default=280)
    ap.add_argument("--device", default="cuda:0", help="cuda:0 (default) or cpu (cardless; set CUDA_VISIBLE_DEVICES=)")
    ap.add_argument("--threads", type=int, default=None, help="torch threads when --device cpu")
    ap.add_argument("--limit", type=int, default=None, help="first N problems (chain dry-run)")
    ap.add_argument("--expect-greedy", default=None,
                    help="optional 'passes,empties' reproduction gate for a SPECIFIC prior ckpt; "
                         "default reports the greedy numbers and never exits on them. The only "
                         "value that was ever meaningful here was 3,72 (format_sft_0909); a "
                         "from-scratch pretrain ckpt has no fixed expected greedy answer")
    ap.add_argument("--n", type=int, default=20, help="samples per problem (SmolLM published 24 at this setting)")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--no-greedy", action="store_true", help="skip the greedy arm (reproduced separately)")
    ap.add_argument("--rstrip_nl", action="store_true",
                    help="strip the prompt's trailing newline(s) before generation -- the v41_ced_0923 "
                         "GATE protocol (prereg amendment_1, user ruling 2026-09-23); matches what "
                         "bigcode/DeepSeek feed base models. Without it the bare column-0 newline is "
                         "fed as-is and a re-declared column-0 def is cut to empty. judge still sees "
                         "the stripped text the model was given")
    ap.add_argument("--no-sample", action="store_true",
                    help="greedy arm only -- the mid-run CPU trend read (stop rule 4); no n-sample "
                         "arm and no sampled control, which need the post-training GPU shards")
    ap.add_argument("--preds", default=None,
                    help="explicit preds artifact path (default: data/eval/...); a mid-run CPU "
                    "trend read writes under runs/")
    ap.add_argument("--shard_i", type=int, default=None,
                    help="multi-card shard: generate only fixed-order problem indices i with "
                         "i %% --shard_n == shard_i. Pair with --shard_n; N cards cover every "
                         "problem exactly once (eval/shard.py, same partition as humaneval_gen).")
    ap.add_argument("--shard_n", type=int, default=None, help="total number of shards")
    ap.add_argument("--merge-glob", default=None,
                    help="cardless: glob of shard preds files to merge into one read and score, "
                         "then exit. Verifies exact problem coverage 0..N-1 and contiguous shards.")
    ap.add_argument("--control", action="store_true", help="run the known-answer controls only, no model (CPU)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--selftest", action="store_true", help="known answers for the entry_point truncate path; cardless")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    import glob as _glob

    if args.merge_glob is not None:
        return merge_shards(sorted(_glob.glob(args.merge_glob)), args)

    probs_all = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    shard_validate(args.shard_i, args.shard_n)
    if not runs_full_control(args.shard_i, args.shard_n):
        print(f"CONTROL skipped on shard {args.shard_i}/{args.shard_n} "
              "(full-set canonical gate runs on shard 0 only)", flush=True)
    else:
        print(f"HumanEval sampled: n={args.n}, temp={args.temp}, top_p={args.top_p}", flush=True)
        run_control(probs_all)
    if args.control:
        return
    if args.shard_n is None and args.limit is not None:
        probs_all = probs_all[: args.limit]
    probs = [p for _, p in shard_select(probs_all, args.shard_i, args.shard_n)]
    print(f"HumanEval sampled: {len(probs)}/{len(probs_all)} problems, n={args.n}, "
          f"temp={args.temp}, top_p={args.top_p}"
          + (f" shard {args.shard_i}/{args.shard_n}" if args.shard_n is not None else ""), flush=True)
    if not args.ckpt:
        ap.error("--ckpt required (unless --control or --merge-glob)")
    is_cpu = str(args.device).startswith("cpu")
    if is_cpu:
        if args.threads:
            torch.set_num_threads(args.threads)
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            # Cardless is the point of --device cpu; the env must be EMPTY explicitly so a
            # default change can never silently put a "cpu" run on physical GPU 0.
            sys.exit("REFUSING: --device cpu but CUDA_VISIBLE_DEVICES is unset -- set it empty "
                     "(CUDA_VISIBLE_DEVICES=) to run cardless.")
    elif not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("REFUSING: CUDA_VISIBLE_DEVICES is unset, so cuda:0 is physical "
                 "GPU 0 -- tileRL's card. Set it to your granted card, or pass --device cpu "
                 "with CUDA_VISIBLE_DEVICES= to run cardless.")

    from scripts.loader import load_checkpoint
    from tokenizers import Tokenizer
    model, cfg = load_checkpoint(args.ckpt, device=args.device)
    model.eval()
    tok = Tokenizer.from_file(TOK_PATH)
    torch.manual_seed(args.seed)

    def gen(prompt, temperature, entry_point):
        """The greedy loop with argmax replaced by nucleus sampling at
        temperature > 0. Everything else verbatim from humaneval_gen.gen.

        entry_point is REQUIRED: truncate must exempt the model's own
        "\\ndef <entry_point>(" re-declaration (humaneval_gen.truncate's 66-14
        exception). The sampled arm used to call truncate() with no entry_point,
        so a re-declared same-named def -- a valid, common completion-style answer
        -- was cut to empty and scored differently from the greedy path on the
        identical string. The in-loop stop check applies the same exemption so
        generation is not ended 16 tokens early on the self-hit."""
        ids = tok.encode(prompt.rstrip("\n") if args.rstrip_nl else prompt).ids
        x = torch.tensor([ids], device=args.device)
        new = []
        # CPU runs fp32 with no autocast, same as humaneval_gen: bf16 autocast is a cuda-only
        # context and the chain must run cardless on a checkpoint dry-run.
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if not is_cpu else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            for step in range(args.max_new):
                lg = model(x[:, -cfg.seq:])[0][:, -1]
                nxt = lg.argmax(-1, keepdim=True) if temperature <= 0 else nucleus(lg, temperature, args.top_p)
                tid = nxt.item()
                if tid == 1:
                    break
                new.append(tid)
                x = torch.cat([x, nxt], 1)
                if step % 16 == 15 and hits_real_stop(tok.decode(new), entry_point):
                    break
        return truncate(tok.decode(new), entry_point)

    preds_path = args.preds or os.path.join(
        ROOT, "data", "eval",
        f"preds_humaneval_sample_{os.path.basename(str(args.ckpt).rstrip('/'))}"
        + (".rstripnl" if args.rstrip_nl else "")
        + (f".limit{args.limit}" if args.shard_n is None and args.limit is not None else "")
        + shard_label(args.shard_i, args.shard_n) + ".jsonl")
    t0 = time.time()
    # The greedy reproduction gate is OPTIONAL and ckpt-specific. 3/72 was the measured
    # format_sft_0909 answer: that checkpoint, that scorer. A from-scratch pretrain ckpt
    # (the CED run's read point) has no fixed expected greedy pass/empty pair, so gating it
    # on 3/72 aborts the sampled arm on every such ckpt before the sampled read point runs.
    # --expect-greedy opts into the gate for a rerun that must reproduce a specific prior
    # number; the default reports the greedy numbers and proceeds.
    expected = None
    if args.expect_greedy:
        try:
            _ep, _ee = (int(v) for v in args.expect_greedy.split(","))
            expected = (_ep, _ee)
        except ValueError:
            ap.error("--expect-greedy must be 'passes,empties', e.g. 3,72")
    greedy_pass = greedy_empty = 0
    sample_pass = 0.0
    ctrl_pass = 0
    sample_empty = 0
    with open_artifact(preds_path, force=args.force, run=args.run) as fout:
        out_path = fout.name
        fout.write(json.dumps({
            "_header": 1,
            "ckpt": os.path.basename(str(args.ckpt).rstrip("/")),
            "data": os.path.basename(args.data),
            "n": args.n, "temp": args.temp, "top_p": args.top_p, "seed": args.seed,
            "max_new": args.max_new, "stops": STOPS, "n_problems": len(probs),
            "shard_i": args.shard_i, "shard_n": args.shard_n,
            "rstrip_nl": args.rstrip_nl,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, ensure_ascii=False) + "\n")

        # Phase 1: greedy. With --expect-greedy the sampled arm is gated on the specific
        # prior ckpt's reproduction; without it the numbers are reported and the sampled
        # arm runs (a fresh pretrain ckpt has no canonical greedy answer to reproduce).
        if not args.no_greedy:
            for i, p in enumerate(probs, 1):
                g = gen(p["prompt"], 0.0, p["entry_point"])
                # Judge the same bytes the model was fed: on the rstrip arm that is the
                # stripped prompt, else exec would reintroduce the removed newline.
                _jprompt = p["prompt"].rstrip("\n") if args.rstrip_nl else None
                ok = judge(p, g, _jprompt)
                empty = not g.strip()
                greedy_pass += int(ok)
                greedy_empty += int(empty)
                fout.write(json.dumps(
                    {"phase": "greedy", "task_id": p["task_id"], "gen": g, "ok": ok, "empty": empty},
                    ensure_ascii=False) + "\n")
                fout.flush()
                if i % 20 == 0 or i == len(probs):
                    print(f"  greedy {i}/{len(probs)}  pass@1 = {greedy_pass}/{i}  "
                          f"empty {greedy_empty}/{i}  ({time.time() - t0:.0f}s)", flush=True)
            print(f"\nHUMANEVAL pass@1 (greedy) = {greedy_pass}/{len(probs)} = "
                  f"{100 * greedy_pass / len(probs):.2f}%  empty {greedy_empty}/{len(probs)}", flush=True)
            if expected is not None and (greedy_pass, greedy_empty) != expected:
                sys.exit(
                    f"GREEDY REPRODUCTION FAILED: {greedy_pass}/{len(probs)} pass, "
                    f"{greedy_empty}/{len(probs)} empty "
                    f"(expected {expected[0]}/{expected[1]}). The shared "
                    "judge/STOPS/truncate diverged from humaneval_gen; NOT running the sampled arm.")
            if expected is not None:
                print("greedy reproduction OK -- running sampled arm", flush=True)

        if args.no_sample:
            # Greedy-only mid-run CPU trend read: skip the n-sample arm (20x the greedy
            # work) and its control; they run on the post-training GPU shards.
            pass
        else:
            # Phase 2: n samples per problem.
            for i, p in enumerate(probs, 1):
                samples = []
                for _ in range(args.n):
                    c = gen(p["prompt"], args.temp, p["entry_point"])
                    _jprompt = p["prompt"].rstrip("\n") if args.rstrip_nl else None
                    samples.append({"gen": c, "ok": judge(p, c, _jprompt)})
                c = sum(s["ok"] for s in samples)
                sample_pass += c / args.n
                ctrl_pass += int(judge(p, p["canonical_solution"]))
                sample_empty += sum(not s["gen"].strip() for s in samples)
                fout.write(json.dumps(
                    {"phase": "sample", "task_id": p["task_id"], "c": c, "samples": samples},
                    ensure_ascii=False) + "\n")
                fout.flush()
                if i % 20 == 0 or i == len(probs):
                    print(f"  sample {i}/{len(probs)}  pass@1(n={args.n}) = "
                          f"{100 * sample_pass / i:.2f}%  ({time.time() - t0:.0f}s)", flush=True)

            # The known-answer control through the SAMPLED phase's own judge call, plus
            # the sample empty rate: a pass@1 that quietly counts empty completions as
            # failures is a format number, and a sampled figure without its control is
            # not a figure (e1-58).
            n_samples = len(probs) * args.n
            fout.write(json.dumps(
                {"phase": "sample_summary", "control_canonical_pass": ctrl_pass,
                 "control_n": len(probs), "sample_empty": sample_empty,
                 "sample_empty_n": n_samples}, ensure_ascii=False) + "\n")
            print(f"\nCONTROL canonical_solution through sampled path = "
                  f"{ctrl_pass}/{len(probs)} (must be {len(probs)})", flush=True)
            print(f"sampled empty completions = {sample_empty}/{n_samples} = "
                  f"{100 * sample_empty / n_samples:.1f}%", flush=True)
            if ctrl_pass != len(probs):
                sys.exit(
                    f"SAMPLED CONTROL FAILED: canonical_solution scored {ctrl_pass}/{len(probs)} "
                    "through the sampled phase's judge -- the pass@1 above is not a figure.")

    attest(out_path)
    if args.no_sample:
        print(f"\nHUMANEVAL greedy-only pass@1 = {greedy_pass}/{len(probs)} = "
              f"{100 * greedy_pass / len(probs):.2f}%  empty {greedy_empty}/{len(probs)} "
              f"(mid-run trend read; no n-sample arm)", flush=True)
    else:
        print(f"\nHUMANEVAL pass@1 (n={args.n}, temp={args.temp}, top_p={args.top_p}) = "
              f"{sample_pass}/{len(probs)} = {100 * sample_pass / len(probs):.2f}%", flush=True)
    print(f"preds saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
