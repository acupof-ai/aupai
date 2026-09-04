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
import traceback
from collections import Counter

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
    "base": ["domain_loss", "minimal_pairs", "mc_ceval", "lambada_zh", "math_v2_like",
             "l1_fewshot", "domain_bpb", "lambada_en", "humaneval_bpb"],
    # A FOURTH TYPE, for a foreign-tokenizer control (Pythia-160m). Only the per-BYTE metrics
    # are DEFINED here -- not "not run", defined. minimal_pairs is Chinese minimal pairs whose
    # premise is that the edit does not change tokenization, which is a statement about OUR
    # BPE; under a NeoX BPE the pairs are not token-aligned and the metric has no meaning.
    # lambada_zh and math_v2_like score option likelihoods over Chinese text a 50k English
    # BPE fragments differently, and mc_ceval is Chinese MC. So the control panel is the three
    # byte-denominated metrics, and the 2x2 table says "undefined for this tokenizer" in the
    # other cells rather than "unmeasured" (1e's ruling 2026-09-03, option 3).
    "control": ["domain_bpb", "lambada_en", "humaneval_bpb"],
    "sft": ["domain_loss", "minimal_pairs", "mc_full", "math_hard", "math_500", "code_500", "code_500_v2", "pass_at_k"],
    "rl": ["domain_loss", "minimal_pairs", "mc_full", "math_hard", "math_500", "code_500", "code_500_v2", "pass_at_k"],
}
# Fixed subset for every stage-1/stage-2 milestone: the metrics that fit in
# <60 min on the lane card. pass_at_k and math_hard stay out (hours); they run
# only at 15B and 30B. de's harness milestone (t39) and b0's readout_30b.py
# consume the record.
PROFILES = {
    "milestone": ["domain_loss", "mc_full", "math_500", "code_500", "code_500_v2"],
    # The control comparison: the three metrics that mean the same thing under two different
    # tokenizers. Run it on OUR checkpoint and on the control with --hf; the pair is the panel.
    "control": ["domain_bpb", "lambada_en", "humaneval_bpb"],
}
SKIP_REASON = {
    "math_hard": "generative; a base checkpoint reads zero (ckpt_0830v1_0.8b: math-500 0/500)",
    "math_500": "generative; a base checkpoint reads zero",
    "code_500": "generative; a base checkpoint reads zero",
    "code_500_v2": "generative; a base checkpoint reads zero",
    "pass_at_k": (
        "needs a policy (SFT or RL); a base checkpoint has none. AND NOT ADDED TO THE BASE "
        "PANEL for HumanEval either, which b0-13 asked for (6e's ruling 2026-09-03): "
        "generative code scoring reads 0.0 at every scale anyone here has measured. "
        "be.gold_bpb_falls_while_generation_scores_zero has five points -- p324 3.24B, 8B, "
        "15B, 16b_pin, 22b_step24000 -- and code_500 generative accuracy is 0.0 at ALL FIVE "
        "while gold BPB falls 1.08724 -> 0.91778 across them. 0 == 0 is not a comparison, and "
        "a resident matrix pays that cost at every checkpoint forever. humaneval_bpb is what "
        "carries this axis instead. THE BOUND IS 'NOT READABLE THROUGH 22B TOKENS', not 'not "
        "readable at 206M': the ceiling is what was measured, not what was assumed. WHEN it "
        "becomes readable is UNMEASURED -- five points, five zeros, zero non-zero points, so "
        "there is no slope to extrapolate and no threshold to name. That is a gap in what has "
        "been measured, NOT a prediction that it stays zero. A one-off run to answer 'what "
        "does the ruler the reader recognises say' is a different question from a resident "
        "panel metric, and is fine to run on request."
    ),
    "mc_full": "hellaswag/piqa unreachable from this machine (pod HF egress broken); not a signal judgement -- run --benchmarks hellaswag piqa on a box with egress. English MC also at chance on every 200M measured, ceval stays as tripwire",
    "lambada_zh": "base-panel metric (frozen panel, docs/lessons/base_eval_panel.md #3)",
    "math_v2_like": "base-panel metric (frozen panel, docs/lessons/base_eval_panel.md #4)",
    "l1_fewshot": "reasoning panel L1 (docs/lessons/reasoning_panel.md §2); few-shot continuation math",
}

# Per-KIND skip reasons, consulted before SKIP_REASON. Keyed by (kind, metric) because
# SKIP_REASON is keyed by metric alone: writing control's reasons into it replaced the base
# panel's reason for the same metric name. ruff caught that one as a duplicate literal, but the
# collision is real independent of the literal -- a base checkpoint would have been told
# "undefined for this tokenizer" about its own panel metric.
#
# UNDEFINED is not UNMEASURED, and that distinction is the whole point of this table. Each of
# these metrics has a premise that is a statement about OUR vocabulary, so under a foreign
# tokenizer there is no number to not-have-measured (1e's ruling 2026-09-03, option 3).
KIND_SKIP_REASON = {
    ("control", "minimal_pairs"):
        "undefined for this tokenizer: the pair bank's premise is that the edit does not change "
        "tokenization, which is a property of OUR BPE, not a property of the sentences",
    ("control", "lambada_zh"):
        "undefined for this tokenizer: Chinese option likelihoods under a 50k English BPE are "
        "not on a comparable scale to ours",
    ("control", "mc_ceval"): "undefined for this tokenizer: Chinese MC under an English BPE",
    ("control", "math_v2_like"): "undefined for this tokenizer: Chinese math options",
    ("control", "domain_loss"):
        "undefined ACROSS tokenizers: nats per TOKEN is not comparable when the same text is a "
        "different number of tokens per side -- domain_bpb is the comparable reading",
}

# Seed variance (sd) per metric at 0.2b, df=3, from ckpt_p02_s0..s3.
# A move smaller than the readable-move threshold (4.65*sd, t-based df=3)
# is not readable. Recorded next to each metric so nobody reads noise as signal.
# Source: facts/base_eval.json (be.*_seed_variance), runs/score_matrix.jsonl.
NOISE_THRESHOLDS = {
    "minimal_pairs": {"sd_pt": 2.47, "readable_move_pt": 11.5, "source": "be.minimal_pairs_seed_variance"},
    # SATURATED at >=200M/4B, and this is the row that says so rather than a note elsewhere.
    # Measured on runs/score_matrix.jsonl: 2-way with a 50% floor, so headroom is 100 - score.
    #   0.2b seeds  76.69 / 77.22 / 83.53 / 78.75   headroom 16.5-23.3pt   sd 3.11pt (the source)
    #   p324 3.24B  95.05                            headroom  4.95pt
    #   15b_s1      97.41 | p200m_4b 97.44 | rehearse 97.54   headroom 2.46-2.59pt
    # Headroom 2.56pt against a readable move of 14.5pt is a ratio of 0.177: even an arm that
    # took ALL the remaining headroom would move less than this metric can resolve. And the
    # three checkpoints above sit inside 0.13pt of each other -- below the binomial se of
    # 0.288pt at n=3012 (95% CI +-0.56pt) -- so a 15B model, a 200M@4B model and a rehearse
    # run are ONE POINT here. The metric still discriminates at 0.2b (76.7 -> 83.5 across
    # seeds, p03/p04 87.7/87.9, p08 94.7); what it has is a scale ceiling, not broken noise.
    # So: do not put it in an A/B reading at 200M or above. b0, 2026-09-03, at 1e's request.
    "math_v2_like": {"sd_pt": 3.11, "readable_move_pt": 14.5, "source": "be.math_v2_like_seed_variance",
                     "saturated_at_or_above": "200M/4B",
                     "headroom_pt_at_200m": 2.56, "headroom_over_readable_move": 0.177},
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


def is_hf_dir(path):
    """True for a HuggingFace checkpoint DIRECTORY, by structure and never by name.

    config.json plus at least one weight file. Structure, not the name, for the reason
    classify() already gives about names: a directory called "pythia-160m" proves nothing, and a
    control checkpoint could be named anything. torch.load on a directory raises IsADirectoryError
    (a confusing one, from inside torch), so this has to be answered before read_cfg."""
    if not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    return any(os.path.exists(os.path.join(path, f)) for f in
               ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
                "pytorch_model.bin.index.json"))


def read_cfg(ckpt_path):
    """cfg only, no model load. An HF directory carries no cfg of ours -- that is not an error."""
    if is_hf_dir(ckpt_path):
        # NOT {} silently: the record must be able to say this checkpoint is foreign, and a
        # vocab_id of None would otherwise read as "an old checkpoint of ours that predates the
        # stamp". "hf" is a value, absence is not.
        return {"kind": "control", "hf": True}, "hf"
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
        # per_row=True IS LOAD-BEARING AND COSTS NOTHING HERE. b0-23 pairs on BLOCKS, and
        # the blocks only exist if the scorer asks for them: this call is where the data
        # leg's end-of-run score lost them. That record was written 32 min before the
        # per-block code landed, so it carried 9 domain scalars and no blocks, and the leg
        # had to be rescored on a lane card to get an N2 exit number. The reported scalar
        # is unaffected -- domain_loss_seqs keeps the flat reduction="sum" as its return
        # value and adds the rows beside it (verified: the rescore reproduced all 9 domains
        # and the mean 1.9443 to the digit).
        loss, ntok, per = domain_loss_seqs(model, rows, device, per_row=True)
        if loss is None:
            skipped.append(name)
            continue
        # head_fp on the same terms as domain_loss.py's CLI: the readout refuses when
        # the two sides disagree AND when the field is absent (62/b0, 2026-09-01), so a
        # record written here without it would be unreadable by the guard rather than
        # merely unverified.
        out[name] = {"loss": round(loss, 4), "tokens": ntok, "head_fp": seqs_fp(rows),
                     "blocks": [{"ce_sum": ce, "n_tokens": nt} for ce, nt in per]}
    if not out:
        return None, f"no domain had val rows to score ({len(skipped)} skipped: {skipped[:5]})"
    vals = [d["loss"] for d in out.values()]
    out["unweighted_mean"] = round(sum(vals) / len(vals), 4)
    out["_split"] = "val"  # the record says which split it is: the old numbers were train
    if skipped:
        out["_skipped"] = skipped
    return out, None


def _capture_failure(r):
    """A failed subprocess's non-empty output lines, from BOTH streams.

    `(r.stderr or r.stdout)` was here and it discards stdout ENTIRELY whenever stderr holds
    anything at all -- one `UserWarning` is enough. That is not hypothetical: eval/domain_bpb.py
    puts every refusal it has on stdout (`REFUSING: no domain produced a number` at :292,
    `SKIPPED (round-trip ...)` at :267, and zero sys.exit/SystemExit), while scripts/loader.py:136
    warns on stderr for every old-format checkpoint. 6 of 10 domain_bpb error rows in
    runs/score_matrix.jsonl therefore record the vocab_id warning as the cause and the refusal
    nowhere (audit_0904 E18; MT-12 counted 10, 3 of those had a real stderr cause and 1 a bare
    source line).

    stdout FIRST: a refusal is what the script chose to say, a warning is what leaked. The 3-line
    tail the callers keep should hold the former when both exist.
    """
    return [ln for ln in (r.stdout + r.stderr).strip().splitlines() if ln.strip()]


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
            # BOTH streams, concatenated -- see _capture_failure's docstring for why `or` is
            # wrong. base_matrix.py has no refusal path today, so nothing is known to be lost
            # here; the expression is fixed anyway because the next refusal it gains would be
            # invisible, and `_run` eleven lines below already does it this way.
            tail = _capture_failure(r)[-1:] or ["no output"]
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
            # The LAST line of a traceback is not reliably the exception. Python 3.12 ends a
            # frame with a caret line, and a subprocess can write past the traceback, so the
            # one-line version recorded domain_bpb's failure as the SOURCE LINE
            # "ours_tok = load_tokenizer(a.tokenizer, None)" -- which names where, never what.
            # ONE row in runs/score_matrix.jsonl carries that string
            # (ckpt_data_leg_206m_8b.pt.step10000); the comment said "all three" before the rows
            # were counted, and 6 others carry a different failure -- the stream shadowing
            # _capture_failure now fixes (audit_0904 E18).
            lines = _capture_failure(r)
            exc = [ln for ln in lines if re.match(r"^\w[\w.]*(Error|Exception|Exit|Interrupt)\b", ln)]
            # With an exception line, one line IS the answer. Without one the traceback was
            # truncated, and the last line alone is exactly what was useless before -- keep
            # three so the record at least says which frames.
            tail = exc[-1:] or lines[-3:] or ["no output"]
            return None, f"{script} exited {r.returncode}: {' | '.join(t.strip() for t in tail)[:400]}"
        return json.load(open(out, encoding="utf-8")), None
    finally:
        os.unlink(out)


def metric_domain_bpb(ckpt_path, mix_path, hf=False):
    """eval/domain_bpb.py: per-domain bits per UTF-8 byte over the same held-out bytes.

    THE CONTROL ARM'S DOMAIN METRIC, and not a replacement for domain_loss. domain_loss is
    nats per TOKEN over our own ids, which is (a) unreadable by a foreign tokenizer's model and
    (b) not comparable across tokenizers at all. Every existing record and threshold is in
    domain_loss's units, so both live side by side."""
    extra = ["--mix", mix_path] + (["--hf"] if hf else [])
    return _run_eval_json("domain_bpb.py", ckpt_path, extra, timeout=5400)


def metric_lambada_en(ckpt_path, hf=False):
    """eval/lambada_en.py: greedy last-word accuracy plus per-byte NLL of the target."""
    return _run_eval_json("lambada_en.py", ckpt_path, ["--hf"] if hf else None, timeout=5400)


def metric_humaneval_bpb(ckpt_path, hf=False):
    """eval/humaneval_bpb.py: gold BPB of the 164 canonical solutions, teacher-forced.

    Gold BPB and NOT pass@k, because at 200M pass@k is 0 on both arms and 0 == 0 is not a
    comparison (be.gold_bpb_falls_while_generation_scores_zero: code_500 generative accuracy
    sat at 0.0 across a ladder while gold BPB fell 1.087 -> 0.918)."""
    return _run_eval_json("humaneval_bpb.py", ckpt_path, ["--hf"] if hf else None, timeout=5400)


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
    vanishes with no log to say so.

    DUPLICATE KEYS RAISE, both on the way in and on the way out (b0-15). The
    matrix is 'the current state, not a history', so two rows under one key
    have no defined current state -- and the audit that refuses them at commit
    time (scripts/ledger_audit.py duplicates()) justifies itself by asserting
    THIS function cannot produce them. It could, in two ways, both measured:

      1. `records` itself carrying one key twice wrote both rows. The dedup
         above builds a SET of keys and filters existing lines by it; it never
         looks at `records` for internal collisions.
      2. A file that ALREADY held two rows for some other key kept them: those
         lines are not in `keys`, so they are copied to `existing` verbatim.
         This is how the live matrix reached 22 duplicate keys -- union merges
         put them there and every later write preserved them, exit 0, silently.

    Case 2 raises on a key this call does not touch, which is deliberate: this
    is the only code that reads the whole file, so it is the only place the
    corruption is visible before the commit hook. The message names the keys and
    says to fold them, because a writer that repaired them silently would erase
    the evidence that a merge is producing them."""
    dup_in = sorted(k for k, n in Counter(
        (r["ckpt"], r.get("profile", "full")) for r in records).items() if n > 1)
    if dup_in:
        raise ValueError(
            f"write_records was handed {len(dup_in)} duplicate key(s) in ONE call: {dup_in}. "
            f"The matrix is current state, not history, so two rows under one "
            f"(ckpt, profile) have no defined current state. Fold them before writing.")
    keys = {(r["ckpt"], r.get("profile", "full")) for r in records}
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            existing = []
            kept = Counter()
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            k = (r.get("ckpt"), r.get("profile", "full"))
                            if k in keys:
                                continue
                            kept[k] += 1
                        except Exception:
                            pass
                        existing.append(line)
            dup_out = sorted(k for k, n in kept.items() if n > 1)
            if dup_out:
                raise ValueError(
                    f"{path} already holds {len(dup_out)} duplicate key(s) that this call does "
                    f"not touch: {dup_out}. A union merge is the usual source (.gitattributes "
                    f"marks this file merge=union, and the in-process lock cannot see across "
                    f"branches). Refusing to write on top of a file with no defined current "
                    f"state -- fold the duplicates, keeping the row you can justify, then rerun. "
                    f"This is not repaired automatically: that would hide the merge producing it.")
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

    # --control: an HF DIRECTORY is recognised by structure, the control panel is exactly the
    # three byte-denominated metrics, and the other cells say UNDEFINED rather than unmeasured.
    import tempfile as _tf2
    with _tf2.TemporaryDirectory() as _hd:
        assert not is_hf_dir(_hd), "an empty directory is not an HF checkpoint"
        # BOTH conditions, each isolated. Asserting only "config.json alone is not enough" cannot
        # catch a dropped config.json check, because the weight-file half still rejects the
        # directory -- verified: removing the config check left this selftest green. So each half
        # gets a world where it is the ONLY thing standing between the input and a wrong answer.
        open(os.path.join(_hd, "model.safetensors"), "wb").write(b"")
        assert not is_hf_dir(_hd), "a weight file WITHOUT config.json is not an HF checkpoint"
        os.remove(os.path.join(_hd, "model.safetensors"))
        open(os.path.join(_hd, "config.json"), "w").write("{}")
        assert not is_hf_dir(_hd), "config.json ALONE is not enough; a weight file is required"
        open(os.path.join(_hd, "model.safetensors"), "wb").write(b"")
        assert is_hf_dir(_hd), "config.json + a weight file IS an HF checkpoint"
        # By STRUCTURE, never by name: a name-based check would call this one HF too.
        _named = os.path.join(_hd, "pythia-160m-step2000")
        os.makedirs(_named)
        assert not is_hf_dir(_named), "is_hf_dir matched on the NAME, not the structure"
        # read_cfg must not torch.load a directory, and "foreign" must be a VALUE, not absence:
        # vocab_id None would read as "an old checkpoint of ours predating the stamp".
        _cfg, _vid = read_cfg(_hd)
        assert _cfg.get("kind") == "control" and _cfg.get("hf") is True, _cfg
        assert _vid == "hf", f"vocab_id {_vid!r} for a foreign checkpoint must be a value"
        assert classify(_cfg, os.path.basename(_hd)) == "control", classify(_cfg, "x")

    # The per-KIND skip reason must not collide with the per-metric one. SKIP_REASON is keyed by
    # metric alone, so control's reasons written there replaced the BASE panel's reason for the
    # same metric -- a base checkpoint would have been told "undefined for this tokenizer" about
    # its own panel metric. ruff caught that as a duplicate literal; the collision itself is
    # silent, so it gets an assertion.
    # The panel is named EXPLICITLY, not compared to PROFILES["control"] -- the first version
    # asserted `_cw == PROFILES["control"]`, which is true for whatever that list happens to hold,
    # so dropping a metric from the panel stayed green. A test that reads its expectation from the
    # thing under test has no expectation.
    _want = ["domain_bpb", "lambada_en", "humaneval_bpb"]
    assert PROFILES["control"] == _want, PROFILES["control"]
    _cw, _, _csk = dispatch("control", PROFILES["control"])
    _bw, _, _bsk = dispatch("base", PROFILES["control"])
    assert _cw == _want and _bw == _want, (_cw, _bw)
    assert APPLIES["control"] == _want, APPLIES["control"]
    assert "undefined for this tokenizer" in (_csk.get("minimal_pairs") or ""), _csk
    assert "nats per TOKEN" in (_csk.get("domain_loss") or ""), _csk.get("domain_loss")
    # ...and the SAME metric must NOT carry that reason for a base checkpoint.
    assert "undefined for this tokenizer" not in (_bsk.get("lambada_zh") or ""), _bsk["lambada_zh"] if "lambada_zh" in _bsk else None
    # domain_loss is absent from the control panel by construction, not by a runtime guard.
    assert "domain_loss" not in APPLIES["control"], APPLIES["control"]

    # _mix_for: the mix comes from the CHECKPOINT unless --mix was named. Worlds are real
    # torch.save files, because the bug being guarded is a type assumption -- cfg is a dict on
    # every real checkpoint, and a getattr-only read returns the fallback for all of them while
    # looking like it consulted the checkpoint.
    import torch
    with tempfile.TemporaryDirectory() as _td:
        _mixp = os.path.join(_td, "mix_probe.json")
        with open(_mixp, "w", encoding="utf-8") as fh:
            json.dump({"total_tokens": 1, "domains": {}}, fh)
        _dictck = os.path.join(_td, "ckpt_dictcfg.pt")
        torch.save({"cfg": {"mix": _mixp, "epochs": 1}, "model": {}}, _dictck)
        _fb = os.path.join(ROOT, "data/mix_scale_3.24b.json")
        assert _mix_for(_dictck, _fb) == _mixp, "a dict cfg's mix must be used"
        assert _mix_for(_dictck, _fb, explicit=True) == _fb, "--mix must win when named"

        # A cfg naming an absent mix falls back, and SAYS so -- silently scoring against the
        # ladder's mix is how a record acquires a basis nobody chose.
        _gonck = os.path.join(_td, "ckpt_goneMix.pt")
        torch.save({"cfg": {"mix": "data/mix_that_does_not_exist.json", "epochs": 1},
                    "model": {}}, _gonck)
        assert _mix_for(_gonck, _fb) == _fb, "an absent mix must fall back"

        # No mix in cfg at all: fall back rather than raise.
        _nock = os.path.join(_td, "ckpt_noMix.pt")
        torch.save({"cfg": {"epochs": 1}, "model": {}}, _nock)
        assert _mix_for(_nock, _fb) == _fb, "a cfg without a mix must fall back"
        # An unreadable checkpoint must not take down a scoring run.
        assert _mix_for(os.path.join(_td, "nope.pt"), _fb) == _fb, "unreadable ckpt must fall back"

    # A RAISING metric costs its own entry and nothing else. domain_loss raises through
    # val_seqs -> cache_guard, which is designed to refuse; before this, that took score()'s
    # whole record with it and threw away every metric already measured.
    _rec = {"metrics": {}, "skipped": {}}
    _metric("ok_one", lambda: ({"value": 1}, None), _rec)

    def _boom():
        raise RuntimeError("cache stamps disagree")

    # The raise must be CAUGHT by _metric, not by this selftest. Letting it propagate here would
    # also exit nonzero -- with the raw RuntimeError and no assertion named, which reads exactly
    # like a guard firing. Twice today a bad world passed for that reason.
    try:
        _metric("raiser", _boom, _rec)
    except Exception as _e:
        raise AssertionError(
            f"_metric let {type(_e).__name__} escape; score()'s per-checkpoint except then "
            "discards every metric already measured for that checkpoint"
        ) from _e
    _metric("ok_two", lambda: ({"value": 2}, None), _rec)
    assert _rec["metrics"]["ok_one"]["value"] == 1, _rec
    assert _rec["metrics"]["ok_two"]["value"] == 2, "a raise must not stop later metrics"
    assert "RuntimeError" in _rec["metrics"]["raiser"]["error"], _rec["metrics"]["raiser"]
    # The traceback, not only the message: "domain_loss failed" is the same string whether the
    # cause was cache_guard, a vocab mismatch or an unreadable shard. Presence asserted before
    # indexing -- a bare _rec[...]["_traceback"] raises KeyError, which exits nonzero naming no
    # assertion and reads like the guard firing. Third instance of that trap today.
    assert "_traceback" in _rec["metrics"]["raiser"], \
        f"a raised metric records no traceback, only {sorted(_rec['metrics']['raiser'])}"
    assert any("_boom" in ln for ln in _rec["metrics"]["raiser"]["_traceback"]), \
        _rec["metrics"]["raiser"]["_traceback"]
    # An (None, err) return is still an error entry, not an exception -- the two paths must not
    # have merged into one.
    _metric("returner", lambda: (None, "no shards"), _rec)
    assert _rec["metrics"]["returner"]["error"] == "no shards", _rec["metrics"]["returner"]
    assert "_traceback" not in _rec["metrics"]["returner"], "a returned error has no traceback"

    # _capture_failure keeps BOTH streams. The world this fails in is the real one: domain_bpb
    # refuses on stdout and loader warns on stderr, and `(r.stderr or r.stdout)` recorded the
    # warning while dropping the refusal for 6 of 10 rows (E18). A real subprocess, not a stub,
    # because the defect lives in what subprocess.run returns.
    _cap = subprocess.run(
        [sys.executable, "-c",
         'import sys, warnings\n'
         'warnings.warn("checkpoint has no vocab_id (old format)")\n'
         'print("REFUSING: no domain produced a number")\n'
         'sys.exit(1)\n'],
        capture_output=True, text=True,
    )
    _lines = _capture_failure(_cap)
    assert any("REFUSING" in ln for ln in _lines), \
        f"the refusal was dropped; _capture_failure returned {_lines}"
    assert any("vocab_id" in ln for ln in _lines), \
        f"the warning was dropped; both streams are kept, not one: {_lines}"
    # Order is load-bearing: callers keep a 3-line TAIL, so with stderr first a long warning
    # could push the refusal out of the window. Verified against the old expression, which
    # returns the warning alone and fails the first assertion above.
    assert _lines.index(next(ln for ln in _lines if "REFUSING" in ln)) < \
        _lines.index(next(ln for ln in _lines if "vocab_id" in ln)), \
        f"stdout must come first so the tail keeps the refusal: {_lines}"

    # domain_loss.py's standalone CLI must take the mix from the checkpoint too -- the same
    # defect, the same fix, and 44 found it by reading 3415e9e rather than by running anything.
    # Asserted HERE because domain_loss.py's own --selftest requires a --ckpt, so the pre-commit
    # hook lists it as unrunnable: a check living in that file would never execute. An AST read,
    # not a substring, and it asserts the CLI CALLS _mix_for rather than that the name appears --
    # an import with no call site is exactly what a copy-paste of this fix would leave behind.
    import ast as _ast
    _dl = os.path.join(HERE, "domain_loss.py")
    with open(_dl, encoding="utf-8") as _fh:
        _dl_src = _fh.read()
    _dl_main = next((n for n in _ast.parse(_dl_src).body
                     if isinstance(n, _ast.FunctionDef) and n.name == "main"), None)
    _calls = {_ast.unparse(n.func) for n in _ast.walk(_dl_main) if isinstance(n, _ast.Call)} \
        if _dl_main else set()
    assert "_mix_for" in _calls, \
        "domain_loss.py's CLI does not call _mix_for; it would score against the ladder mix"
    # And the mix it hands to domain_files must be that result, not a.mix.
    _files_args = [_ast.unparse(n.args[0]) for n in _ast.walk(_dl_main)
                   if isinstance(n, _ast.Call) and _ast.unparse(n.func) == "domain_files"
                   and n.args] if _dl_main else []
    assert _files_args == ["mix_path"], \
        f"domain_loss.py passes {_files_args} to domain_files, not the checkpoint's mix"

    # THIS FILE'S OWN domain_loss_seqs CALL MUST ASK FOR per_row, AND THE RECORD MUST CARRY
    # THE BLOCKS. The data leg's end-of-run score was written here without per_row: 9 domain
    # scalars, no blocks, and b0-23 had to rescore the leg on a lane card to get a pairing
    # unit. Nothing failed -- the record was well-formed and the scalars were right, which is
    # why it shipped. An AST read of THIS module, on the same terms as the _mix_for check
    # above and for the same reason: a substring search passes on a commented-out call, and
    # the arity is the half that a copy-paste breaks (per_row=True returns THREE values).
    _sm_src = open(os.path.abspath(__file__), encoding="utf-8").read()
    _seq_calls = [n for n in _ast.walk(_ast.parse(_sm_src)) if isinstance(n, _ast.Call)
                  and _ast.unparse(n.func) == "domain_loss_seqs"]
    assert _seq_calls, "no domain_loss_seqs call found in score_matrix.py -- the scorer moved"
    for _c in _seq_calls:
        _kw = {k.arg: _ast.unparse(k.value) for k in _c.keywords}
        assert _kw.get("per_row") == "True", (
            "score_matrix's domain_loss_seqs call does not pass per_row=True, so every record "
            "it writes carries domain scalars and no blocks -- b0-23 cannot pair on it and the "
            "loss is silent (this is exactly how the data leg's score shipped blockless)")
    # and the record built from it must actually store them under "blocks". AN AST READ, NOT A
    # SUBSTRING: my first version of this line was `assert '"blocks": [{"ce_sum"' in _sm_src`,
    # and it was GREEN against a mutation that deleted the storing code -- because the literal
    # inside the assertion is itself part of the source it searches, so the check found ITSELF.
    # A guard that reads the whole file for a string it contains can never fail. So: find the
    # dict assigned to out[name] and require a "blocks" key whose value is built from `per`.
    _out_dicts = [n.value for n in _ast.walk(_ast.parse(_sm_src))
                  if isinstance(n, _ast.Assign) and isinstance(n.value, _ast.Dict)
                  and any(_ast.unparse(t).startswith("out[") for t in n.targets)]
    assert _out_dicts, "no `out[...] = {...}` record literal found -- the writer moved"
    for _d in _out_dicts:
        _keys = {_ast.unparse(k).strip("'\"") for k in _d.keys if k is not None}
        assert "blocks" in _keys, (
            f"the record score_matrix writes has keys {sorted(_keys)} and no 'blocks': per_row "
            "is requested but the rows are dropped on the floor, which is indistinguishable "
            "from the blockless record the data leg shipped")
        _blocks_val = next(_ast.unparse(v) for k, v in zip(_d.keys, _d.values)
                           if k is not None and _ast.unparse(k).strip("'\"") == "blocks")
        assert "per" in _blocks_val, (
            f"'blocks' is present but not built from the per_row rows: {_blocks_val}")

    # kind stamp wins over sft epochs; epochs>1 is sft; no stamp, no ledger row -> base
    assert classify({"kind": "rl", "epochs": 3}, "ckpt_x.pt", log="/nonexistent") == "rl"
    assert classify({"epochs": 3}, "ckpt_x.pt", log="/nonexistent") == "sft"
    assert classify({"epochs": 1}, "ckpt_selftest_no_such_row.pt") == "base"

    # e1-22: a metric a checkpoint type cannot carry must reach `skipped`, never a score.
    # THE DISPATCH ITSELF, not a copy of it. My first draft reimplemented the four lines
    # inside this selftest and asserted on the reimplementation -- which stays green while
    # score() regresses, because the two share no code. A second copy of the logic is a
    # second thing to keep in step, and the defect being fixed here WAS two lists
    # disagreeing. So `dispatch` is a module-level function that score() calls, and these
    # assertions exercise the same object the real path does.
    #
    # --metrics code_500 on a base checkpoint: the case that put a ChatML zero into the
    # ledger as measured. code_500 hands the model <|im_start|>user...assistant, absent
    # from 168,000 sampled corpus rows (AGENTS.md:200).
    w, un, sk = dispatch("base", ["code_500"])
    assert w == [], f"code_500 must not be scored on a base checkpoint, got {w}"
    assert un == ["code_500"], un
    assert "code_500" in sk and "REQUESTED" in sk["code_500"], sk
    # --profile milestone: one list run across every type on purpose. base keeps the two
    # it can carry and the generative ones are skipped, so the same command works on base
    # and sft and the record says which metrics each could actually carry.
    w, un, sk = dispatch("base", PROFILES["milestone"])
    assert w == ["domain_loss"], f"base milestone should keep only domain_loss, got {w}"
    assert {"mc_full", "math_500", "code_500", "code_500_v2"} <= set(sk), sk
    # ...and an sft checkpoint carries the whole milestone list, or this check would be
    # passing by refusing everything.
    w, un, sk = dispatch("sft", PROFILES["milestone"])
    assert w == PROFILES["milestone"], f"sft must carry the full milestone list, got {w}"
    assert un == [], un
    # The default path is untouched: no --metrics means APPLIES[kind], all applicable.
    w, un, _ = dispatch("base", APPLIES["base"])
    assert w == APPLIES["base"] and un == [], (w, un)
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

        # DUPLICATE KEYS RAISE (b0-15). Both holes measured before the fix: each wrote
        # successfully and exited 0, and ledger_audit.py's duplicates() justifies refusing a
        # second row at commit time by asserting this writer cannot produce one.
        #
        # Hole 1: one call carrying the same key twice. The dedup builds a SET of keys and
        # never looks at `records` for internal collisions -- measured: 2 rows for 1 key.
        try:
            write_records(p, [{"ckpt": "c.pt", "v": 1}, {"ckpt": "c.pt", "v": 2}])
            raise AssertionError("write_records accepted the same key twice in ONE call; before "
                                 "the guard this wrote both rows and exited 0")
        except ValueError as e:
            assert "ONE call" in str(e) and "c.pt" in str(e), e
        # ...and the refused call must not have touched the file.
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]  # noqa: SIM115
        assert {(r["ckpt"], r["v"]) for r in rows} == {("a.pt", 2), ("b.pt", 1)}, \
            f"a refused write still modified the file: {rows}"

        # Hole 2: the file already holds two rows for a key this call does NOT touch. Those
        # lines are absent from `keys`, so they were copied through verbatim -- which is how
        # the live matrix reached 22 duplicates via union merges, every write preserving them.
        p5 = os.path.join(d, "m5.jsonl")
        with open(p5, "w", encoding="utf-8") as f:
            f.write(json.dumps({"ckpt": "x.pt", "v": 1}) + "\n")
            f.write(json.dumps({"ckpt": "x.pt", "v": 2}) + "\n")
        try:
            write_records(p5, [{"ckpt": "y.pt", "v": 1}])
            raise AssertionError("write_records wrote on top of a file with two rows under one "
                                 "key; before the guard this exited 0 and kept both")
        except ValueError as e:
            assert "already holds" in str(e) and "x.pt" in str(e), e
        # The corruption is REPORTED, never silently repaired: a writer that folded them would
        # erase the evidence that a merge is producing them.
        rows = [json.loads(l) for l in open(p5, encoding="utf-8")]  # noqa: SIM115
        assert len(rows) == 2 and not any(r["ckpt"] == "y.pt" for r in rows), \
            f"the refused write repaired or partially applied: {rows}"
        # The SAME key differing only by profile is two legitimate rows, not a duplicate -- a
        # guard keyed on ckpt alone would refuse the milestone/full pair the key exists for.
        p6 = os.path.join(d, "m6.jsonl")
        write_records(p6, [{"ckpt": "z.pt", "profile": "full", "v": 1},
                           {"ckpt": "z.pt", "profile": "milestone", "v": 1}])
        write_records(p6, [{"ckpt": "w.pt", "v": 1}])
        rows = [json.loads(l) for l in open(p6, encoding="utf-8")]  # noqa: SIM115
        assert len(rows) == 3, f"(ckpt, profile) pairs were treated as duplicates: {rows}"
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
    has a measured budget, not a guess.

    ONE metric raising takes down ONE metric. Every metric_* returns (value, error) and the
    convention held until one of them raised instead -- domain_loss does, through
    val_seqs -> cache_guard, which is designed to refuse. That propagated to score()'s caller,
    where the per-checkpoint except discarded the record whole: the four metrics already
    measured, minutes of card time, and the checkpoint's row in the ledger, thrown away by a
    metric that was never going to work on that checkpoint. A partial record is the useful
    artifact -- the point of `metrics` and `skipped` being separate keys is that a record says
    what it could and could not carry (44, reviewing 3415e9e).

    The traceback goes into the record, not just the message: an eval that fails inside a
    dependency (cache_guard, a tokenizer mismatch, a shard read) is diagnosed by where, and the
    message alone reads as "domain_loss failed" for every one of those causes.
    """
    print(f"  {name:15s} ... running", flush=True)
    t0 = time.time()
    try:
        v, err = fn(*args, **kwargs)
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        v, err = None, f"{type(e).__name__}: {' '.join(str(e).split())}"
        record["metrics"][name] = {"error": err, "_wall_s": elapsed,
                                  "_traceback": traceback.format_exc().splitlines()[-6:]}
        print(f"  {name:15s} RAISED: {err} ({elapsed}s) -- the other metrics continue",
              flush=True)
        return
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


def dispatch(kind, requested):
    """(metrics to score, metrics asked for but impossible, skipped -> reason).

    ONE place decides what a checkpoint type can carry, because the defect this replaces
    was two lists disagreeing: score() validated a requested metric name against the
    UNION of all three types (every real name passes) and never against APPLIES[kind], so
    `--metrics code_500` and `--profile milestone` routed a base checkpoint into the three
    generative ChatML metrics. Worse, the skip bookkeeping keyed on the requested list, so
    such a metric was missing from `skipped` -- it had been scored, and its zero entered
    the ledger as measured. A base checkpoint reads zero on those because the prompt hands
    it <|im_start|>user...<|im_start|>assistant, a prefix occurring 0 times in 168,000
    sampled corpus rows (AGENTS.md:200); score_code_exec.py:9-31 measured 1.6% of
    generations carrying a code fence under ChatML against 94.4% under 1-shot continuation
    on the same checkpoint family. That zero is a format artifact, not a capability.

    Skipped, never raised: `--profile milestone` is deliberately one list run across every
    type, and the record is supposed to say which metrics each type could carry.

    A module-level function rather than inline code, so selftest() can exercise the REAL
    dispatch. My first draft reimplemented these lines inside the selftest and asserted on
    the copy, which stays green while this regresses -- the exact shape of the bug above.
    """
    wanted = [m for m in requested if m in APPLIES[kind]]
    unusable = [m for m in requested if m not in APPLIES[kind]]
    def reason(m):
        """The per-kind reason if there is one, else the metric's general one."""
        return KIND_SKIP_REASON.get((kind, m)) or SKIP_REASON.get(m)

    skipped = {m: reason(m) for m in set(SKIP_REASON) | {k[1] for k in KIND_SKIP_REASON
                                                        if k[0] == kind}
               if m not in APPLIES[kind] and reason(m)}
    for m in unusable:
        skipped[m] = (f"REQUESTED but does not apply to a {kind} checkpoint -- "
                      + (reason(m) or f"{kind} accepts {APPLIES[kind]}"))
    return wanted, unusable, skipped


def _mix_for(ckpt_path, fallback, explicit=False):
    """The mix this CHECKPOINT was trained on, from its own cfg. --mix wins when named.

    The mix is a property of the checkpoint, not of the invocation. --mix defaulted to the
    ladder's mix_scale_3.24b.json, so scoring a non-ladder checkpoint read domain rows for
    domains it never trained on and cache_guard refused -- correctly, since the seqs
    fingerprint belongs to another corpus. run_ddp.sh's end-of-run scoring passes no --mix at
    all, so EVERY non-ladder run would fail its own automatic scoring and exit nonzero with a
    good checkpoint (fb hit this on p500m step2500 and again on p200m_4b_0902, 2026-09-02).

    cfg is a DICT on every real checkpoint -- verified on ckpt_p200m_4b_0902.pt.step500:
    type dict, cfg['mix'] = 'data/mix_200m_4b.json'. A getattr-only read would return the
    fallback for all of them while looking like it consulted the checkpoint.
    """
    if explicit:
        return fallback
    try:
        cfg, _ = read_cfg(ckpt_path)
    except Exception:
        return fallback
    mix = cfg.get("mix") if isinstance(cfg, dict) else getattr(cfg, "mix", None)
    if not mix:
        return fallback
    p = mix if os.path.isabs(mix) else os.path.join(ROOT, mix)
    # A cfg naming a mix that is gone must not silently score against the ladder's: say so and
    # fall back, so the record's basis is visible in the log rather than inferred.
    if not os.path.exists(p):
        print(f"note: {os.path.basename(ckpt_path)} names mix {mix!r}, which is absent here; "
              f"falling back to {os.path.basename(fallback)}", flush=True)
        return fallback
    return p


def score(ckpt_path, mix_path, tok_path, device, ngpu=1, metrics=None, profile="full"):
    ckpt_name = os.path.basename(ckpt_path)
    cfg, vocab_id = read_cfg(ckpt_path)
    kind = classify(cfg, ckpt_name)
    requested = metrics if metrics else APPLIES[kind]
    known = set().union(*APPLIES.values())
    bad = [m for m in requested if m not in known]
    if bad:
        raise ValueError(f"unknown metrics {bad}; choose from {sorted(known)}")
    # APPLICABILITY, not just name validity. The check above asks whether a metric name
    # exists ANYWHERE (the union of all three types), which every real metric name
    # passes -- so `--metrics code_500` and `--profile milestone` (domain_loss, mc_full,
    # math_500, code_500, code_500_v2) routed a BASE checkpoint into the three generative
    # ChatML metrics. Those hand the model <|im_start|>user...<|im_start|>assistant, a
    # prefix that occurs 0 times in 168,000 sampled corpus rows (AGENTS.md:200), so the
    # zero they produce measures response to an unseen prefix rather than capability
    # (score_code_exec.py:9-31: 1.6% of generations carry a fence under ChatML against
    # 94.4% under 1-shot continuation, same checkpoint family).
    #
    # An inapplicable metric goes to `skipped` with its reason -- NOT raised, and not
    # scored. That is this file's own contract eleven lines into the docstring: "an
    # inapplicable 0 and a measured 0 must look different in the ledger". Raising would
    # satisfy the letter of it by refusing to write either, but it also breaks
    # `--profile milestone`, which is one list deliberately run across every checkpoint
    # type; the profile's whole purpose is that the same command works on base and sft
    # and the record says which metrics each could carry.
    wanted, unusable, preskipped = dispatch(kind, requested)
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
    hf = bool(cfg.get("hf")) if isinstance(cfg, dict) else False
    # domain_loss uses OUR loader and OUR ids, so it can never run on a foreign checkpoint. It is
    # already absent from APPLIES["control"], so dispatch() has skipped it -- this line only keeps
    # the model load from being attempted if someone forces it with --metrics.
    needs_model = "domain_loss" in wanted and not hf
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
    # The three cross-tokenizer metrics. Each takes `hf` so the SAME code path scores both arms:
    # one implementation, two tokenizers, which is what makes the pair comparable. A second
    # implementation per arm is how two arms end up measuring two different things.
    if "domain_bpb" in wanted:
        _metric("domain_bpb", metric_domain_bpb, record, ckpt_path, mix_path, hf)
    if "lambada_en" in wanted:
        _metric("lambada_en", metric_lambada_en, record, ckpt_path, hf)
    if "humaneval_bpb" in wanted:
        _metric("humaneval_bpb", metric_humaneval_bpb, record, ckpt_path, hf)
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

    # dispatch() already decided this, keyed on APPLIES[kind] rather than on `wanted`.
    # Keying on wanted meant an EXPLICITLY requested inapplicable metric was absent from
    # `skipped` -- it had been scored, so its ChatML-induced 0 entered the ledger as a
    # measured value, which is what the docstring above forbids.
    record["skipped"].update(preskipped)
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
    # Whether the caller NAMED a mix, not whether a.mix is set -- argparse fills the default
    # in either case, so `a.mix != default` cannot tell an explicit `--mix <the default>` from
    # a silent fallback. sys.argv is the only place that distinction survives.
    mix_given = any(x == "--mix" or x.startswith("--mix=") for x in sys.argv[1:])

    metrics = PROFILES[a.profile] if a.profile else a.metrics

    device = _pick_card()
    records = []
    failed = []
    for ck in a.ckpt:
        try:
            rec = score(ck, _mix_for(ck, a.mix, explicit=mix_given), a.tokenizer, device,
                        a.ngpu, metrics, a.profile or "full")
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
