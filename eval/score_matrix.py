#!/usr/bin/env python3
"""Score one checkpoint with the metrics that move on its type, and say which
metrics do not apply.

Type comes from ck["cfg"], not the filename:
- cfg["kind"] (stamped by the entry points) is authoritative
- epochs > 1 -> sft (pretrain forces epochs = 1)
- the ledger row that produced the checkpoint names rlvr -> rl
- else base

A metric that does not apply to a type is recorded under "skipped" with the
reason, never as 0. A base checkpoint reads zero on every generative eval
(ckpt_0830v1_0.8b: math-500 0/500, degenerate loops); an inapplicable 0 and
a measured 0 must look different in the ledger.

    python eval/score_matrix.py --ckpt ckpt_0830v1_0.2b.pt [--ckpt ...]
    python eval/score_matrix.py --ckpt X.pt --json runs/score_matrix.jsonl

Every metric here must prove it moves across the 0.2b->3.24b span; a metric
that does not move has no resolution and does not belong in the matrix.
"""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from cache_guard import set_vocab_id  # noqa: E402
from domain_loss import domain_loss_seqs, seqs_fp, val_seqs  # noqa: E402

from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

# Type -> the metrics that apply. Generative evals read zero on a base
# checkpoint, so they are not in base's list: they go under "skipped".
# English MC sits at chance on every 200M checkpoint measured (ceval moves,
# the English three do not, same model same scale -- b0's finding), so base
# gets ceval only as a z-score tripwire.
APPLIES = {
    "base": ["domain_loss", "minimal_pairs", "mc_ceval", "lambada_zh", "math_v2_like", "l1_fewshot"],
    "sft": ["domain_loss", "minimal_pairs", "mc_full", "math_hard", "math_500", "code_500", "code_500_v2", "pass_at_k"],
    "rl": ["domain_loss", "minimal_pairs", "mc_full", "math_hard", "math_500", "code_500", "code_500_v2", "pass_at_k"],
}
# Fixed subset for every stage-1/stage-2 milestone: the metrics that fit in
# <60 min on the lane card. pass_at_k and math_hard stay out (hours); they run
# only at 15B and 30B. de's harness milestone (t39) and b0's readout_30b.py
# consume the record.
PROFILES = {
    "milestone": ["domain_loss", "mc_full", "math_500", "code_500", "code_500_v2"],
}
SKIP_REASON = {
    "math_hard": "generative; a base checkpoint reads zero (ckpt_0830v1_0.8b: math-500 0/500)",
    "math_500": "generative; a base checkpoint reads zero",
    "code_500": "generative; a base checkpoint reads zero",
    "code_500_v2": "generative; a base checkpoint reads zero",
    "pass_at_k": "needs a policy (SFT or RL); a base checkpoint has none",
    "mc_full": "hellaswag/piqa unreachable from this machine (pod HF egress broken); not a signal judgement -- run --benchmarks hellaswag piqa on a box with egress. English MC also at chance on every 200M measured, ceval stays as tripwire",
    "lambada_zh": "base-panel metric (frozen panel, docs/lessons/base_eval_panel.md #3)",
    "math_v2_like": "base-panel metric (frozen panel, docs/lessons/base_eval_panel.md #4)",
    "l1_fewshot": "reasoning panel L1 (docs/lessons/reasoning_panel.md §2); few-shot continuation math",
}

# Seed variance (sd) per metric at 0.2b, df=3, from ckpt_p02_s0..s3.
# A move smaller than the readable-move threshold (4.65*sd, t-based df=3)
# is not readable. Recorded next to each metric so nobody reads noise as signal.
# Source: facts/base_eval.json (be.*_seed_variance), runs/score_matrix.jsonl.
NOISE_THRESHOLDS = {
    "minimal_pairs": {"sd_pt": 2.47, "readable_move_pt": 11.5, "source": "be.minimal_pairs_seed_variance"},
    "math_v2_like": {"sd_pt": 3.11, "readable_move_pt": 14.5, "source": "be.math_v2_like_seed_variance"},
    "lambada_zh": {"sd_pt": 1.02, "readable_move_pt": 4.8, "source": "be.panel_expressive_seed_variance"},
    "mc_ceval": {"sd_pt": 1.27, "readable_move_pt": 5.9, "source": "be.panel_expressive_seed_variance"},
    # domain_loss: sd=0.0516 nat (ds.seed_variance_0p2b), readable_move=0.24 nat
    "domain_loss": {"sd_nat": 0.0516, "readable_move_nat": 0.24, "source": "ds.seed_variance_0p2b"},
    # l1_fewshot: binomial delta=1.4/sqrt(N); N=500 -> 6.3pt. No seed variance yet
    # (b0 hand-running); label noise is binomial, seed variance is model-side.
    "l1_fewshot": {"binomial_delta_pt": 6.3, "n_items": 500, "source": "reasoning_panel.md §3"},
}


def classify(cfg, ckpt_name, log=None):
    """The checkpoint's type, from cfg with the ledger as the fallback for RL.

    cfg carries no kind stamp yet (the entry points predate it); epochs>1 is
    the SFT signal because pretrain forces epochs=1. RL is invisible in cfg --
    rlvr_trainer saves the base cfg with grad_ckpt flipped -- so the producing
    ledger row is the only authoritative source for it. The name is used as a
    key into the ledger, never as a pattern: 'sft' in a filename proves nothing."""
    kind = cfg.get("kind") if isinstance(cfg, dict) else getattr(cfg, "kind", None)
    if kind:
        return kind
    epochs = cfg.get("epochs", 1) if isinstance(cfg, dict) else getattr(cfg, "epochs", 1)
    if epochs > 1:
        return "sft"
    log = log or os.path.join(ROOT, "runs", "experiments.jsonl")
    if os.path.exists(log):
        stem = ckpt_name[: -len(".pt")] if ckpt_name.endswith(".pt") else ckpt_name
        with open(log, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                cmd = str(row.get("cmd", ""))
                if f"--out {stem}" in cmd or f"--name {stem[len('ckpt_') :]}" in cmd:
                    if "rlvr" in cmd:
                        return "rl"
                    if "sft" in cmd:
                        return "sft"
    return "base"


def read_cfg(ckpt_path):
    """cfg only, no model load."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ck.get("cfg", {}), ck.get("vocab_id")


def metric_domain_loss(model, tok, seq, device, mix_path):
    """Per-domain loss on the rows train.py actually held out.

    Was the head of each domain's alphabetically-first shard, which stopped being val
    when train.py started shuffling before slicing: 0.625% of the scored docs landed in
    val against 0.587% by chance, so every per-domain nat ever recorded here was
    TRAINING-SET loss (tilerl measured it, 2026-09-01). val_seqs reconstructs val the
    way train.py builds it -- same cache, same seeded shuffle, same val_frac.

    A domain with no shards is SKIPPED, never scored as zero, and the skip is reported:
    a mean over a silently smaller set of domains is a different metric wearing the same
    name.
    """
    mix = json.load(open(mix_path, encoding="utf-8"))
    out, skipped = {}, []
    for name in mix["domains"]:
        rows = val_seqs(name, tok)
        if rows is None:
            skipped.append(name)
            continue
        loss, ntok = domain_loss_seqs(model, rows, device)
        if loss is None:
            skipped.append(name)
            continue
        # head_fp on the same terms as domain_loss.py's CLI: the readout refuses when
        # the two sides disagree AND when the field is absent (62/b0, 2026-09-01), so a
        # record written here without it would be unreadable by the guard rather than
        # merely unverified.
        out[name] = {"loss": round(loss, 4), "tokens": ntok, "head_fp": seqs_fp(rows)}
    if not out:
        return None, f"no domain had val rows to score ({len(skipped)} skipped: {skipped[:5]})"
    vals = [d["loss"] for d in out.values()]
    out["unweighted_mean"] = round(sum(vals) / len(vals), 4)
    out["_split"] = "val"  # the record says which split it is: the old numbers were train
    if skipped:
        out["_skipped"] = skipped
    return out, None


def metric_minimal_pairs(ckpt_path):
    """b0's eval/base_matrix.py: token-aligned Chinese minimal pairs across five
    syntactic dimensions, with BPE-merge handling (a pair is skipped, not scored,
    if the edit changes tokenization) and a --swap known-answer mode. Floor is
    50% by construction. The pair bank is 277 pairs -- a de-risked prototype,
    not the 1000-pairs-per-dimension resolution target."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    try:
        r = subprocess.run(
            [sys.executable, "eval/base_matrix.py", "--ckpt", ckpt_path, "--out", out],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=1800,
        )
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
            return None, f"base_matrix.py exited {r.returncode}: {tail[0][:200]}"
        return json.load(open(out, encoding="utf-8")), None
    finally:
        os.unlink(out)


def _run_eval_json(script, ckpt_path, extra_args=None, timeout=3600):
    """Shell out to an eval/<script>.py --ckpt --out <tmp> and parse its JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    try:
        r = subprocess.run(
            [sys.executable, f"eval/{script}", "--ckpt", ckpt_path, "--out", out] + (extra_args or []),
            capture_output=True, text=True, cwd=ROOT, timeout=timeout,
        )
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
            return None, f"{script} exited {r.returncode}: {tail[0][:200]}"
        return json.load(open(out, encoding="utf-8")), None
    finally:
        os.unlink(out)


def metric_lambada_zh(ckpt_path):
    """b0's eval/lambada_zh.py: LAMBADA-zh last-token prediction (panel #3).
    Open-vocab top-1/top-5 (floor ~= 0) plus a 2-way known-answer reading
    (real final token vs same-length distractor, floor 50%). Data must be
    built first from held-out prose: eval/lambada_zh.py --build."""
    data = os.path.join(ROOT, "data", "eval", "lambada_zh.jsonl")
    if not os.path.exists(data):
        return None, "lambada_zh.jsonl not built (eval/lambada_zh.py --build --src <held-out prose>)"
    return _run_eval_json("lambada_zh.py", ckpt_path, ["--data", data])


def metric_math_v2_like(ckpt_path):
    """b0's eval/math_v2_like.py: math-hard v2 likelihood twin (panel #4).
    Scores the gold answer span against a same-token-length wrong answer
    (one deterministic digit edit) conditioned on the solution prefix.
    Floor 50%; per-family reporting; pairs that break token-length alignment
    are skipped and counted."""
    return _run_eval_json("math_v2_like.py", ckpt_path)


def metric_l1_fewshot(ckpt_path):
    """Reasoning panel L1: few-shot continuation math (docs/lessons/reasoning_panel.md §2).
    3 solved demos in prompt, model continues, exact-match on boxed/last number.
    N=497, binomial delta=1.4/sqrt(N)≈6.3pt. Script: eval/l1_fewshot.py."""
    script = os.path.join(ROOT, "eval", "l1_fewshot.py")
    if not os.path.exists(script):
        return None, "eval/l1_fewshot.py not found"
    return _run_eval_json("l1_fewshot.py", ckpt_path)


def _run(cmd, patterns, env=None):
    """subprocess, parse {name: value} from stdout lines matching
    `patterns` (name -> regex with one float group). (values, error).
    env merges into os.environ for K/TEMP-style knobs."""
    full_env = {**os.environ, **env} if env else None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=3600, env=full_env)
    except Exception as e:
        return None, f"{cmd[0]} failed to run: {e}"
    out = {}
    for line in r.stdout.splitlines():
        for name, pat in patterns.items():
            m = re.search(pat, line)
            if m:
                out[name] = float(m.group(1))
    if not out:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        return None, f"{cmd[0]} produced no score: {' | '.join(tail)[:200]}"
    return out, None


def metric_mc(ckpt_path, tok_path, benchmarks):
    """run_eval prints '  <display name>  25.1%' lines (ceval displays as
    'C-Eval (zh)'), so parse every percent line by its display name rather
    than by the benchmark flag."""
    try:
        r = subprocess.run(
            [
                sys.executable,
                "eval/run_eval.py",
                "--ckpt",
                ckpt_path,
                "--tokenizer",
                tok_path,
                "--benchmarks",
                *benchmarks,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=3600,
        )
    except Exception as e:
        return None, f"run_eval.py failed to run: {e}"
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r"\s*(.+?)\s{2,}([\d.]+)%", line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    if not out:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        return None, f"run_eval.py produced no score: {' | '.join(tail)[:200]}"
    return out, None


def metric_math_hard(ckpt_path, tok_path, ngpu=1):
    return _run(
        ["bash", "eval/eval_hard.sh", ckpt_path, str(ngpu)],
        {"math_hard": r"=\s*([\d.]+)\s*%"},
        env={"TOKENIZER": tok_path},
    )


def metric_math_500(ckpt_path, tok_path, ngpu=1):
    return _run(
        ["bash", "eval/eval_math.sh", ckpt_path, str(ngpu)],
        {"math_500": r"=\s*([\d.]+)\s*%"},
        env={"TOKENIZER": tok_path},
    )


def metric_code_500(ckpt_path, tok_path, ngpu=1):
    return _run(
        ["bash", "eval/eval_code.sh", ckpt_path, str(ngpu)],
        {"code_500": r"=\s*([\d.]+)\s*%"},
        env={"TOKENIZER": tok_path},
    )


def metric_code_500_v2(ckpt_path, tok_path, ngpu=1):
    # t51: clean holdout (gen_code_v2 families, absent from all SFT/pretraining
    # sources). code_500 is carved from an SFT source and measures template recall.
    return _run(
        ["bash", "eval/eval_code.sh", ckpt_path, str(ngpu)],
        {"code_500_v2": r"=\s*([\d.]+)\s*%"},
        env={"TOKENIZER": tok_path, "HOLDOUT": "data/eval/code_holdout_v2_500.jsonl", "TAG": "v2"},
    )


def metric_pass_at_k(ckpt_path, tok_path, ngpu=1):
    # Sharded via eval_hard.sh instead of single-GPU math_hard.py.
    # pass@k is computed on the merged full rows, so sharding changes speed, not the number.
    return _run(
        ["bash", "eval/eval_hard.sh", ckpt_path, str(ngpu)],
        {"pass_at_1": r"pass@1\(greedy\)\s+([\d.]+)%", "pass_at_8": r"pass@8\s+([\d.]+)%"},
        env={"K": "8", "TEMP": "0.8", "TOKENIZER": tok_path},
    )


# Degeneration repeat rate. All parameters live here, not in prose -- a metric
# config outgrows its name. Source: facts/base_eval.json (be.degeneration_rate).
DEGEN_CONFIG = {
    "ngram_len": 8,
    "repeat_threshold": 3,
    # The degenerate CONDITION, not the denominator: a generation with < n words cannot
    # form the n-gram, so it is non-degenerate -- but it stays in N. Canonical is in-N
    # (fb's first version + b0's recompute); excluding short generations moved math
    # 0-shot by 5.4pt (16.9% -> 22.3%).
    "min_words": 8,
    "unit": "whitespace",
    # This metric reads the last 300 characters of each generation, not the whole one.
    # The comment here used to say the eval scripts store gen[-300:]; they store the
    # full text (code_500 preds run to 1020 chars, 21/500 over 300), so the truncation
    # is this scorer's own and discards data already on disk. Measured cost on p324:
    # whitespace code_500 58.2% tail vs 58.8% full, every other field unchanged.
    # Widening it changes recorded numbers, so it is fb's call, not a silent edit.
    "window": "gen[-300:]",
    # CJK character n-gram: Chinese text has no whitespace word boundaries, so the
    # whitespace 8-gram undercounts CJK degeneration (math_500 25.6% is a floor).
    "cjk_ngram_len": 12,
    "cjk_repeat_threshold": 3,
    "cjk_min_chars": 36,
    "cjk_majority_threshold": 0.3,
}


def degeneration_rate(path, temperature, greedy=None):
    """Fraction of generations with a repeated n-gram, over one prediction file.

    A generation is degenerate if any ngram_len-gram (whitespace tokens) appears
    >= repeat_threshold times. The denominator N is every row: a generation with
    < min_words tokens cannot form the n-gram, so it is non-degenerate, but it
    stays in N (canonical in-N; excluding short generations moved math 0-shot 5.4pt).

    The report carries the decode temperature because a format metric under greedy
    is a decoder property, not a model property (SFT greedy 55.8% vs t=0.8 20.1%).
    A number without a temperature cannot go on the board.

    Two known pitfalls (measured, not theoretical):
    1. This scorer reads gen[-300:], so it measures TAIL degeneration. The preds files
       hold the full generation, so widening the window costs nothing but a rescore.
       On p324 it moves one field: whitespace code_500 58.2% -> 58.8%. Left as the tail
       because the recorded numbers were measured that way.
    2. The 300-char window holds different token counts across languages (Chinese
       is dense), so cross-domain absolute values carry a density confound.
       Within-domain comparison is unaffected.
    """
    if not os.path.exists(path):
        return None, f"no prediction file {os.path.relpath(path, ROOT)}"
    cfg = DEGEN_CONFIG
    n = deg = deg_cjk = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if greedy is not None and r.get("greedy") != greedy:
                continue
            n += 1  # denominator = every row; a short generation is non-degenerate, not absent
            # The eval scripts now store the FULL gen; the metric's window is the tail
            # (DEGEN_CONFIG["window"]). Old files already store the tail, and [-300:] of
            # a 300-char string is itself, so this is backward compatible.
            text = r.get("gen", "")[-300:]
            words = text.split()
            if len(words) >= cfg["min_words"]:
                counts = {}
                for i in range(len(words) - cfg["ngram_len"] + 1):
                    ng = tuple(words[i : i + cfg["ngram_len"]])
                    c = counts.get(ng, 0) + 1
                    if c >= cfg["repeat_threshold"]:
                        deg += 1
                        break
                    counts[ng] = c
            # CJK character 12-gram: Chinese text has no whitespace word boundaries,
            # so the whitespace 8-gram undercounts. Only for CJK-majority text.
            cjk = sum(1 for c in text if '一' <= c <= '鿿')
            if cjk > len(text) * cfg["cjk_majority_threshold"] and len(text) >= cfg["cjk_min_chars"]:
                chars = list(text)
                cg_counts = {}
                for i in range(len(chars) - cfg["cjk_ngram_len"] + 1):
                    ng = tuple(chars[i : i + cfg["cjk_ngram_len"]])
                    c = cg_counts.get(ng, 0) + 1
                    if c >= cfg["cjk_repeat_threshold"]:
                        deg_cjk += 1
                        break
                    cg_counts[ng] = c
    if n == 0:
        # Not "too short to form the n-gram" -- that was the old message, and it read as
        # a property of the generations when the truth was that there were none. An empty
        # preds file reported as a benign scoring skip: the vacuous shape, in the scorer
        # written to catch degeneration (de, 2026-09-01, code_500_v2 on a 0-byte file).
        return None, f"no rows in {os.path.relpath(path, ROOT)} ({os.path.getsize(path)} bytes)"
    return {
        "rate": round(deg / n, 4),
        "degenerate": deg,
        "cjk_rate": round(deg_cjk / n, 4),
        "cjk_degenerate": deg_cjk,
        "n": n,
        "temperature": temperature,
        **cfg,
    }, None


def _add_degeneration(record, key, pred_path, temperature, greedy=None, after=None):
    """Compute the degeneration rate from a generative metric's prediction file
    and store it beside the metric.

    `after` names the metric that was supposed to WRITE pred_path. When that metric
    errored, the file on disk is whatever a previous run left there, and scoring it
    reports a number the current run did not produce. On 2026-09-01 the step15000
    milestone recorded math_500_degeneration 0.026 and code_500_degeneration 0.518
    beside math_500 ERROR and code_500 ERROR -- both rates computed from preds files
    an 03:34 run had written, published in a 04:31 record as that run's result. The
    numbers happened to describe the same checkpoint, so nothing looked wrong; had
    the leftovers been another checkpoint's, or a truncated shard merge, the record
    would have read identically. A derived number must name the run whose bytes it
    read, and refuse when that run did not write them.
    """
    if after and _errored(record, after):
        record["metrics"][key] = {
            "error": f"{after} did not produce a prediction file this run "
                     f"({record['metrics'][after].get('error', '')[:120]}); "
                     f"whatever is at {os.path.relpath(pred_path, ROOT)} belongs to an earlier run"
        }
        print(f"  {key:15s} SKIPPED: {after} errored; refusing to score a previous run's file", flush=True)
        return
    v, err = degeneration_rate(pred_path, temperature, greedy=greedy)
    record["metrics"][key] = v if v else {"error": err}


def _errored(record, name):
    """True when metric `name` ran and recorded an error. Absent is not errored:
    a metric outside the profile was never asked to write anything."""
    m = record["metrics"].get(name)
    return isinstance(m, dict) and "error" in m


def write_records(path, records):
    """Replace same-(ckpt, profile) records, keep others and unparseable lines.
    The matrix is the current state, not a history.

    The key is (ckpt, profile), not ckpt: a milestone-profile record must never
    replace a checkpoint's full record (2026-08-31, t39 dry run). A record
    without a profile reads as 'full', so existing rows need no migration.

    An exclusive lock on path + '.lock' serializes concurrent writers: without
    it, two score_matrix processes on different ckpts can interleave their
    read-modify-write cycles, and the later writer overwrites the earlier's
    fresh record. Both print 'wrote N record(s)', both exit 0, and a record
    vanishes with no log to say so."""
    keys = {(r["ckpt"], r.get("profile", "full")) for r in records}
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            existing = []
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            if (r.get("ckpt"), r.get("profile", "full")) in keys:
                                continue
                        except Exception:
                            pass
                        existing.append(line)
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(existing)
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _selftest_write_worker(path, ckpt):
    """Module-level worker for the concurrent-write selftest (must be picklable)."""
    write_records(path, [{"ckpt": ckpt, "v": 1}])


def selftest():
    """Known answers for the logic that breaks silently: type classification,
    the MC display-name parser, same-ckpt record replacement. A check that
    fires on correct code is as useless as one that never fires, so every
    assertion is a case with a known right answer."""
    import tempfile

    # kind stamp wins over sft epochs; epochs>1 is sft; no stamp, no ledger row -> base
    assert classify({"kind": "rl", "epochs": 3}, "ckpt_x.pt", log="/nonexistent") == "rl"
    assert classify({"epochs": 3}, "ckpt_x.pt", log="/nonexistent") == "sft"
    assert classify({"epochs": 1}, "ckpt_selftest_no_such_row.pt") == "base"
    # the ledger join is the only RL signal: rlvr in the producing command
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "experiments.jsonl")
        with open(log, "w", encoding="utf-8") as f:
            f.write(json.dumps({"cmd": "python algorithms/rlvr_trainer.py --out ckpt_rl_one.pt"}) + "\n")
            f.write(json.dumps({"cmd": "python sft.py --name sft_one"}) + "\n")
        assert classify({"epochs": 1}, "ckpt_rl_one.pt", log=log) == "rl"
        assert classify({"epochs": 1}, "ckpt_sft_one.pt", log=log) == "sft"
        # replacement: same ckpt updated in place, the other ckpt kept
        p = os.path.join(d, "m.jsonl")
        write_records(p, [{"ckpt": "a.pt", "v": 1}, {"ckpt": "b.pt", "v": 1}])
        write_records(p, [{"ckpt": "a.pt", "v": 2}])
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]  # noqa: SIM115
        assert {(r["ckpt"], r["v"]) for r in rows} == {("a.pt", 2), ("b.pt", 1)}, rows
        # concurrent writers: every record must survive. Without flock, the later
        # writer reads stale content and overwrites the earlier's record — both
        # print "wrote N record(s)", both exit 0, and a record vanishes silently.
        import multiprocessing
        p4 = os.path.join(d, "m4.jsonl")
        ctx = multiprocessing.get_context("fork")
        procs = [ctx.Process(target=_selftest_write_worker, args=(p4, f"ckpt_{i}.pt")) for i in range(4)]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
        rows4 = [json.loads(l) for l in open(p4, encoding="utf-8")]  # noqa: SIM115
        assert len(rows4) == 4, f"concurrent write lost records: {len(rows4)}/4 survived"
    # the MC parser keys on display name, not flag name (ceval prints "C-Eval (zh)")
    m = re.match(r"\s*(.+?)\s{2,}([\d.]+)%", "  C-Eval (zh)        25.1%")
    assert m and (m.group(1).strip(), m.group(2)) == ("C-Eval (zh)", "25.1")
    # degeneration rate: known-answer cases. A repeated 8-gram is degenerate;
    # a short generation (< min_words) stays in N as non-degenerate, not excluded.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"gen": "a b c d e f g h " * 3, "greedy": True}) + "\n")  # 8-gram x3 -> degenerate
        f.write(json.dumps({"gen": " ".join(str(i) for i in range(20)), "greedy": True}) + "\n")  # no repeat
        f.write(json.dumps({"gen": "a b c", "greedy": True}) + "\n")  # < 8 words -> in N, non-degenerate
        p1 = f.name
    try:
        v, err = degeneration_rate(p1, 0)
        assert err is None, err
        assert v["n"] == 3 and v["degenerate"] == 1 and v["rate"] == round(1 / 3, 4), v
    finally:
        os.unlink(p1)
    # the greedy filter selects one arm (pass_at_k's sampled rows are greedy=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"gen": "a b c d e f g h " * 3, "greedy": True}) + "\n")
        f.write(json.dumps({"gen": " ".join(str(i) for i in range(20)), "greedy": False}) + "\n")
        p2 = f.name
    try:
        v2, err2 = degeneration_rate(p2, 0.8, greedy=False)
        assert err2 is None, err2
        assert v2["n"] == 1 and v2["degenerate"] == 0 and v2["temperature"] == 0.8, v2
    finally:
        os.unlink(p2)
    # An empty preds file is not "generations too short": the old message named a
    # property of generations that did not exist. It must name the file and its size.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        p3 = f.name
    try:
        v3, err3 = degeneration_rate(p3, 0)
        assert v3 is None and "0 bytes" in err3 and os.path.basename(p3) in err3, err3
    finally:
        os.unlink(p3)
    # after=: a degeneration rate must not be computed from a file the errored metric
    # did not write. The red case carries the REAL leftover shape -- a scoreable preds
    # file on disk beside an ArtifactExists error -- because that is exactly what
    # published math_500_degeneration 0.026 under a math_500 ERROR (step15000).
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"gen": "a b c d e f g h " * 3, "greedy": True}) + "\n")
        p_after = f.name
    try:
        errored = {"metrics": {"math_500": {"error": "bash produced no score: ArtifactExists"}}}
        _add_degeneration(errored, "math_500_degeneration", p_after, 0, after="math_500")
        got = errored["metrics"]["math_500_degeneration"]
        assert "error" in got and "earlier run" in got["error"], got
        # green: the same file, the same call, with the metric having succeeded
        ok_rec = {"metrics": {"math_500": {"math_500": 1.2}}}
        _add_degeneration(ok_rec, "math_500_degeneration", p_after, 0, after="math_500")
        assert ok_rec["metrics"]["math_500_degeneration"]["rate"] == 1.0, ok_rec
        # a metric outside the profile was never asked to write: absent is not errored
        absent = {"metrics": {}}
        _add_degeneration(absent, "math_500_degeneration", p_after, 0, after="math_500")
        assert absent["metrics"]["math_500_degeneration"]["rate"] == 1.0, absent
    finally:
        os.unlink(p_after)
    print("selftest OK")


def _metric(name, fn, record, *args, **kwargs):
    """Run a metric, print start/result, store it. Real-time output so a 2h
    silent run doesn't look dead. Wall time recorded so a milestone profile
    has a measured budget, not a guess."""
    print(f"  {name:15s} ... running", flush=True)
    t0 = time.time()
    v, err = fn(*args, **kwargs)
    elapsed = round(time.time() - t0, 1)
    entry = v if v else {"error": err}
    if isinstance(entry, dict):
        entry["_wall_s"] = elapsed
    record["metrics"][name] = entry
    if err:
        print(f"  {name:15s} ERROR: {err} ({elapsed}s)", flush=True)
    else:
        print(f"  {name:15s} {v} ({elapsed}s)", flush=True)


def _pick_card():
    """A free card from the ones this process was given, not cuda:0 by default.

    2026-09-01: scoring lrprobe_0.85 died with OutOfMemoryError on GPU 0 -- a card the
    run never used (training held 1-7) and a third party held at 95.16 of 95.22 GiB. The
    checkpoint was already saved and fine, but the traceback named an OOM on a 95 GiB
    card, and that read as the training run exhausting memory. It cost the team twenty
    minutes and a launch-blocker that was not there.

    CUDA_VISIBLE_DEVICES already remaps indices, so cuda:0 IS the first allowed card when
    it is set -- the failure only happens when it is not, which is exactly how the scorer
    is invoked after a run. Pick the emptiest visible card instead of the first one.
    """
    if not torch.cuda.is_available():
        return "cpu"
    # CUDA_VISIBLE_DEVICES, when the caller sets it, IS the "cards this run was given" --
    # torch then only sees those, so the loop below is already restricted to them. When it
    # is unset (how the post-run scorer is actually invoked) every card is visible and the
    # emptiest is the right pick, NOT the first: measured on the pod 2026-09-02 while arm
    # 1.2 trains, cards 1-7 hold 23.2 GiB free each and card 0 holds 94.9. Preferring the
    # training set would have put scoring on the busiest cards in the box.
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            continue  # a card we cannot query is a card we should not choose
        if free > best_free:
            best, best_free = i, free
    # Say which card and why, at the top. The contention line already existed but only
    # printed inside the OOM handler, under a truncated traceback where nobody read it.
    print(f"scoring on cuda:{best} ({best_free / 2**30:.1f} GiB free of "
          f"{torch.cuda.device_count()} visible card(s))", flush=True)
    return f"cuda:{best}"


def score(ckpt_path, mix_path, tok_path, device, ngpu=1, metrics=None, profile="full"):
    ckpt_name = os.path.basename(ckpt_path)
    cfg, vocab_id = read_cfg(ckpt_path)
    kind = classify(cfg, ckpt_name)
    wanted = metrics if metrics else APPLIES[kind]
    known = set().union(*APPLIES.values())
    bad = [m for m in wanted if m not in known]
    if bad:
        raise ValueError(f"unknown metrics {bad}; choose from {sorted(known)}")
    print(f"\n{ckpt_name}  type={kind}  {len(wanted)} metrics", flush=True)
    record = {
        "ckpt": ckpt_name,
        "profile": profile,  # (ckpt, profile) is the key: a milestone record must never replace a full one
        "type": kind,
        "vocab_id": vocab_id,
        "measured": subprocess.run(["date", "+%Y-%m-%d"], capture_output=True, text=True).stdout.strip(),
        "metrics": {},
        "skipped": {},
        "noise_thresholds": {k: v for k, v in NOISE_THRESHOLDS.items() if k in wanted},
    }
    needs_model = "domain_loss" in wanted
    model = tok = seq = None
    if needs_model:
        print(f"  {'model':15s} ... loading", flush=True)
        model, cfg = load_checkpoint(ckpt_path, device=device, dtype=torch.bfloat16)
        tok = load_tokenizer(tok_path, cfg)
        model.eval()
        seq = getattr(cfg, "seq", 4096)
        # train.VOCAB_ID from the checkpoint, before val_seqs reaches _domain_seqs. Only
        # train.build_tokenizer sets it, and nothing here calls that -- so without this the
        # module global stays None, every cache stamp reads as a mismatch, and the domain_loss
        # metric retokenizes the training caches and re-stamps them with an empty
        # vocabulary (fb, 2026-09-02, caught on ppl.py two minutes into a live run).
        # val_seqs checks freshness per domain; this half cannot live there, because the
        # fingerprint comes from the checkpoint and val_seqs is not given one.
        set_vocab_id(cfg)
        print(f"  {'model':15s} loaded", flush=True)

    if "domain_loss" in wanted:
        _metric("domain_loss", metric_domain_loss, record, model, tok, seq, device, mix_path)
    if model is not None:
        del model
        torch.cuda.empty_cache()
    if "minimal_pairs" in wanted:
        _metric("minimal_pairs", metric_minimal_pairs, record, ckpt_path)
    if "mc_ceval" in wanted:
        _metric("mc_ceval", metric_mc, record, ckpt_path, tok_path, ["ceval"])
    if "lambada_zh" in wanted:
        _metric("lambada_zh", metric_lambada_zh, record, ckpt_path)
    if "math_v2_like" in wanted:
        _metric("math_v2_like", metric_math_v2_like, record, ckpt_path)
    if "l1_fewshot" in wanted:
        _metric("l1_fewshot", metric_l1_fewshot, record, ckpt_path)
    if "mc_full" in wanted:
        # hellaswag/piqa excluded: pod HF egress broken, datasets unreachable.
        # Run run_eval.py --benchmarks hellaswag piqa on a box with egress.
        _metric("mc_full", metric_mc, record, ckpt_path, tok_path, ["ceval", "mmlu", "arc-easy"])
    if "math_hard" in wanted:
        _metric("math_hard", metric_math_hard, record, ckpt_path, tok_path, ngpu)
        _add_degeneration(record, "math_hard_degeneration",
                          os.path.join(ROOT, f"data/eval/hard_{ckpt_name}.jsonl"), 0,
                          after="math_hard")
    if "math_500" in wanted:
        _metric("math_500", metric_math_500, record, ckpt_path, tok_path, ngpu)
        _add_degeneration(record, "math_500_degeneration",
                          os.path.join(ROOT, f"data/eval/preds_{ckpt_name}.jsonl"), 0,
                          after="math_500")
    if "code_500" in wanted:
        _metric("code_500", metric_code_500, record, ckpt_path, tok_path, ngpu)
        _add_degeneration(record, "code_500_degeneration",
                          os.path.join(ROOT, f"data/eval/preds_code_{ckpt_name}.jsonl"), 0,
                          after="code_500")
    if "code_500_v2" in wanted:
        _metric("code_500_v2", metric_code_500_v2, record, ckpt_path, tok_path, ngpu)
        _add_degeneration(record, "code_500_v2_degeneration",
                          os.path.join(ROOT, f"data/eval/preds_code_v2_{ckpt_name}.jsonl"), 0,
                          after="code_500_v2")
    if "pass_at_k" in wanted:
        _metric("pass_at_k", metric_pass_at_k, record, ckpt_path, tok_path, ngpu)
        # The sampled arm (t=0.8): pass_at_k's eval_hard.sh writes greedy + sampled rows
        # to the same hard_<ckpt>.jsonl, so select greedy=False.
        _add_degeneration(record, "pass_at_k_degeneration",
                          os.path.join(ROOT, f"data/eval/hard_{ckpt_name}.jsonl"), 0.8, greedy=False,
                          after="pass_at_k")

    for m, reason in SKIP_REASON.items():
        if m not in wanted:
            record["skipped"][m] = reason
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append")
    ap.add_argument("--mix", default=os.path.join(ROOT, "data/mix_scale_3.24b.json"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data/tokenizer.json"))
    ap.add_argument("--json", help="append one record per checkpoint here")
    ap.add_argument("--selftest", action="store_true", help="known answers; run before believing any record")
    ap.add_argument("--ngpu", type=int, default=1,
                    help="GPUs for sharded evals (1 on the lane card, 7 on the training block)")
    ap.add_argument("--metrics", nargs="+", default=None,
                    help="subset of metrics to run (default: all that apply to the ckpt type)")
    ap.add_argument("--profile", choices=list(PROFILES), default=None,
                    help="predefined metric subset (overrides --metrics)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.ckpt:
        ap.error("--ckpt is required")

    metrics = PROFILES[a.profile] if a.profile else a.metrics

    device = _pick_card()
    records = []
    failed = []
    for ck in a.ckpt:
        try:
            rec = score(ck, a.mix, a.tokenizer, device, a.ngpu, metrics, a.profile or "full")
        except Exception as e:
            # "SKIPPED (OutOfMemoryError: ...)" read as "the checkpoint OOMed when
            # saving", which would be a large conclusion -- training fits but writing
            # does not. The save succeeded at 987 MB; SCORING is what ran out of memory.
            # A message whose subject is not the thing that failed is the same defect
            # class as everything else found today (fb, 2026-09-01).
            #
            # It also exited 0 and wrote no record, so an automatic post-checkpoint run
            # failed twice and was found by someone reading an unrelated log. The
            # planned save policy is ~28 milestones over three days: this would have
            # failed 28 times while score_matrix_present stayed red the whole time, and
            # a red nobody can act on is the same as no signal.
            #
            # The message is NOT truncated. [:90] cut CUDA's OOM line at "GPU 0 has a
            # total capacity of 95.22 GiB o" -- exactly before the allocated/free/
            # reserved figures that say whether this process was greedy or the card was
            # already occupied. The surviving text read as "a scorer wants 95 GB"; the
            # full line says it failed to allocate 96 MiB, which is the opposite
            # diagnosis. A handler that trims the part naming the cause is the defect
            # it is reporting.
            detail = " ".join(str(e).split())
            failed.append((os.path.basename(ck), f"{type(e).__name__}: {detail}"))
            print(f"\n{os.path.basename(ck)}: SCORING FAILED -- the checkpoint is fine, "
                  f"this run produced no metrics\n  {type(e).__name__}: {detail}",
                  flush=True)
            if isinstance(e, torch.cuda.OutOfMemoryError):
                free, total = torch.cuda.mem_get_info()
                print(f"  card state now: {free / 2**30:.1f} GiB free of "
                      f"{total / 2**30:.1f} GiB -- if free is small, another process "
                      f"holds the card and this is contention, not a greedy scorer",
                      flush=True)
            continue
        records.append(rec)
        print(f"\n{rec['ckpt']}  type={rec['type']}", flush=True)
        for m, v in rec["metrics"].items():
            if "error" in v:
                print(f"  {m:15s} ERROR: {v['error']}", flush=True)
            elif m == "domain_loss":
                print(f"  {m:15s} mean={v['unweighted_mean']:.4f} across {len(v) - 1} domains", flush=True)
            else:
                print(f"  {m:15s} {v}", flush=True)
        for m, why in rec["skipped"].items():
            print(f"  {m:15s} SKIPPED: {why}", flush=True)

    if a.json:
        write_records(a.json, records)
        print(f"\nwrote {len(records)} record(s) to {a.json}")

    # ANY failure exits nonzero, not only an all-fail. The planned save policy is ~28
    # milestones and this runs automatically per checkpoint: one silent failure in
    # twenty-eight is exactly the case that goes unnoticed, and the all-or-nothing
    # condition made a single loss indistinguishable from success to any caller
    # checking the exit code. Attempted is not a result (de, 2026-09-01).
    if failed:
        for name, why in failed:
            print(f"FAILED {name}: {why}", flush=True)
        print(f"{len(failed)}/{len(a.ckpt)} checkpoint(s) produced NO metrics", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
