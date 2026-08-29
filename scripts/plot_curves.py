#!/usr/bin/env python3
"""Plot training curves from run logs -> plots/<name>.png.

Parses the plain-text lines the trainers print (pretrain / SFT / RLVR):
  step N/M loss X            train loss
  step N/M val X             periodic validation (Cfg.val_every)
  ep E/K train X val Y       validation loss at epoch end
  step N/M acc A loss L ...  RLVR reward + loss
  math-500: c/n = P%         holdout accuracy (eval)

Usage: python scripts/plot_curves.py runs/*.log   (no args = all of runs/)
Called automatically at the end of train.py / sft.py / sft_math.py / rlvr_trainer.py.
"""

import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(ROOT, "plots")
STEP_RE = re.compile(r"^step (\d+)/(\d+) loss (-?[\d.]+)")
VAL_RE = re.compile(r"^ep (\d+)/(\d+) train ([\d.]+) val ([\d.]+)")
RL_RE = re.compile(r"^step (\d+)/(\d+) acc ([\d.]+) loss (-?[\d.]+)")
# Under a data mix Cfg.epochs is forced to 1, so the epoch-end val line fires once;
# this is the only regex that makes a val CURVE rather than a val point.
STEPVAL_RE = re.compile(r"^step (\d+)/(\d+) val ([\d.]+)")


def parse(path):
    step, loss, val, rl, sval = [], [], [], [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        if m := RL_RE.match(line):
            rl.append((int(m[1]), float(m[3]), float(m[4])))
        elif m := STEP_RE.match(line):
            step.append(int(m[1]))
            loss.append(float(m[3]))
        elif m := STEPVAL_RE.match(line):
            sval.append((int(m[1]), float(m[3])))
        elif m := VAL_RE.match(line):
            val.append((int(m[1]), float(m[3]), float(m[4])))
    return step, loss, val, rl, sval


def ema(xs, a=0.9):
    out, v = [], None
    for x in xs:
        v = x if v is None else a * v + (1 - a) * x
        out.append(v)
    return out


def plot(path):
    step, loss, val, rl, sval = parse(path)
    if not step and not rl:
        return None
    name = os.path.splitext(os.path.basename(path))[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    if step:
        ax.plot(step, loss, alpha=0.3, label="train loss")
        ax.plot(step, ema(loss), label="train loss (ema)")
        if sval:  # already in steps, no rescaling needed
            ax.plot([e for e, _ in sval], [v for _, v in sval], "o-", label="val loss")
        elif val:
            per_ep = max(step) / max(e for e, _, _ in val)
            ax.plot([e * per_ep for e, _, _ in val], [v for _, _, v in val], "o-", label="val loss")
        ax.set_ylabel("loss")
    if rl:
        s = [r[0] for r in rl]
        ax.plot(s, [r[1] for r in rl], label="reward (train acc)")
        ax.set_ylabel("reward")
        ax2 = ax.twinx()
        ax2.plot(s, [r[2] for r in rl], color="gray", alpha=0.5, label="grpo loss")
        ax2.set_ylabel("loss")
        ax2.legend(loc="upper right")
    ax.set_xlabel("step")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    os.makedirs(PLOTS, exist_ok=True)
    out = os.path.join(PLOTS, name + ".png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "runs", "*.log")))
    for p in paths:
        out = plot(p)
        print(f"{p} -> {out}" if out else f"{p}: no curves found")
