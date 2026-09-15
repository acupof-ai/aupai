#!/usr/bin/env python3
"""Figures for docs/reports/v41_round1/report.md.

All numbers are measured artifacts from the v41 r3 / phi-SFT round, not estimates:
- HE greedy rstrip CLEAN/156 and gold-bpb from runs/he_r3_step{k}_{rstrip,bpb}.log on pod.
- E0 / psft n=10 temp0.2 per-sample mean from runs/e0_{e0,psft}_result.json.
- Non-code from runs/noncode_eval_psft.json (MMLU n13564, ARC-E 2221, lambada 5153/1000).
Regenerate: python3 docs/reports/v41_round1/make_figs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# --- Fig 1: training trajectories (greedy HE CLEAN and gold-bpb) -----------
steps = [18000, 24000, 36000, 37000]
he_greedy_clean = [11.54, 16.03, 21.79, 17.95]      # %, n=1 greedy rstrip CLEAN/156
bpb_steps = [12000, 18000, 24000, 36000, 37000]
gold_bpb = [0.6377, 0.5599, 0.4920, 0.4735, 0.4697]  # per-task mean, lower is better

fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
ax1.plot(steps, he_greedy_clean, "o-", color="#1f6fb2", label="HumanEval CLEAN (greedy, %)")
ax1.set_xlabel("training step")
ax1.set_ylabel("HumanEval CLEAN pass@1 (%)", color="#1f6fb2")
ax1.tick_params(axis="y", labelcolor="#1f6fb2")
ax1.set_ylim(0, 30)
ax2 = ax1.twinx()
ax2.plot(bpb_steps, gold_bpb, "s--", color="#c0504d")
ax2.set_ylabel("HumanEval gold-bpb (lower better)", color="#c0504d")
ax2.tick_params(axis="y", labelcolor="#c0504d")
ax2.spines["top"].set_visible(False)
ax2.set_ylim(0.7, 0.4)
ax1.axvspan(34264, 38070, color="grey", alpha=0.12)
ax1.text(34350, 1.5, "anneal", fontsize=8, color="grey")
ax1.set_title("r3 trajectory: generative pass@1 vs teacher-forced bpb")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig1_trajectory.png"))
plt.close(fig)

# --- Fig 2: E0 vs psft, both benchmarks (n=10 per-sample mean) -------------
bench = ["HumanEval\nCLEAN/1560", "MBPP\nCLEAN/3380"]
e0 = [18.72, 26.89]
psft = [20.45, 25.62]
x = range(len(bench)); w = 0.36
fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.bar([i - w/2 for i in x], e0, w, label="r3 final (E0)", color="#6a8caf")
ax.bar([i + w/2 for i in x], psft, w, label="after phi SFT", color="#d98c5f")
ax.axhline(30, color="red", ls=":", lw=1.5)
ax.text(1.42, 30.6, "30% gate", color="red", fontsize=9)
for i, v in enumerate(e0):
    ax.text(i - w/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
for i, v in enumerate(psft):
    ax.text(i + w/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylabel("pass@1, per-sample mean (%)")
ax.set_xticks(list(x)); ax.set_xticklabels(bench)
ax.set_ylim(0, 34)
ax.set_title("E0 vs phi-SFT (n=10, t=0.2): neither delta significant")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig2_sft_code.png"))
plt.close(fig)

# --- Fig 3: non-code ability, r3 vs psft ------------------------------------
dims = ["MMLU\n(13564)", "ARC-E\n(2221)", "LambadaEN\n(5153)", "LambadaZH\n2-way(1000)"]
r3v = [23.34, 39.08, 21.93, 64.82]
psv = [23.70, 35.88, 19.23, 63.05]
x = range(len(dims))
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.bar([i - w/2 for i in x], r3v, w, label="r3 final", color="#6a8caf")
ax.bar([i + w/2 for i in x], psv, w, label="after phi SFT", color="#d98c5f")
for i, v in enumerate(r3v):
    ax.text(i - w/2, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
for i, v in enumerate(psv):
    ax.text(i + w/2, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
ax.set_ylabel("accuracy (%)")
ax.set_xticks(list(x)); ax.set_xticklabels(dims)
ax.set_ylim(0, 75)
ax.set_title("General ability: narrow code SFT slightly hurts language/reasoning")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig3_noncode.png"))
plt.close(fig)

# --- Fig 4: HumanEval n=10 score histogram (E0), c_i/10 distribution --------
# E0 HE: 115 tasks c=0, 15 tasks c=10, 34 middle. Distribution from measured
# bimodal readout (98 panel / merged preds).
buckets = list(range(11))
# Exact histogram computed from data/eval/e0_he_merged.n10temp0.2.jsonl over the
# 156 clean tasks (8-task union excluded): c=0..10 counts.
counts = [108, 6, 8, 2, 3, 2, 3, 3, 1, 5, 15]
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.bar(buckets, counts, color="#4f7a4f")
ax.set_xlabel("# correct samples out of 10 per HumanEval task")
ax.set_ylabel("# tasks")
ax.set_title("Bimodal competence (E0 HumanEval, 156 clean tasks)")
ax.text(0.3, 101, "108 never\ncorrect", fontsize=8, color="#7a3030")
ax.text(9.0, 18, "15 always\ncorrect", fontsize=8, color="#2a5a2a")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig4_bimodal.png"))
plt.close(fig)

print("wrote", sorted(os.listdir(FIGS)))
