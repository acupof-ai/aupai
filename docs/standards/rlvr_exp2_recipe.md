---
question: What exactly is run for experiment 2 — RLVR sample efficiency against pretraining tokens — and what must be true before a card is spent on it?
status: recorded
source: algorithms/rlvr_trainer.py at 09041709; docs/lessons/efficiency_gap_views.md:369,376-381,441; runs/controller_board.md:45; facts/data_scaling.json#ds.b_unidentified_from_val_traces; facts/contamination.json#cont.novel_ops_frozen_sets
---

# RLVR recipe, experiment 2

Every constant below is **chosen** unless it cites a measurement. The distinction is the
point of the document: nothing here is derived from a fitted scaling exponent, because
`ds.b_unidentified_from_val_traces` establishes that the exponent is not identifiable from
a 200M run's val trace, and no other measured quantity in the tree sets these values.

## 0. Two definitions of "experiment 2" are in circulation, and they are different experiments

This has to be settled before anything runs. What is written down:

| source | experiment 2 is |
|---|---|
| `efficiency_gap_views.md:441`, `controller_board.md:45` | RLVR on **2-3 ladder checkpoints differing only in pretraining tokens** (0.4b/1.6b/3.24b); readout **pass@1 vs rollouts consumed**; deliverable d(capability)/d(RL sample) as a function of pretraining tokens |
| fb, 2026-09-05, by message | **sample efficiency per token consumed**, RL rollouts+updates against pretraining tokens, read out by the **conversion-rate curve on the S/P sets** |

The S/P sets were built for **experiment 1** and their own fact says so
(`cont.generator_families_in_owm`, config.why: "for the conversion-rate curve's skill (4c,
synthesis experiment 1)"). Fusing the two is a design decision that appears in no document.

**It is also the better experiment, and the reason is a difference in kind rather than in
degree** (e1's sharpening). The docs' version reads out on math-500 at 30% containment
(`cont.holdout_v2`), which cannot separate retrieval from reasoning at all. The S/P sets
are constructed-absent — the header's own `absence_basis` field says the operator, its rule
and its phrasing "were invented 2026-09-05, after every corpus in the mix was built", and a
scan is not what the claim rests on. Those are two epistemic classes, not two points on a
containment scale. So this recipe writes fb's version, and states the substitution as new
rather than citing it.

What is lost by the substitution: the docs' version measures capability the model might
plausibly acquire from RL on real math. This one measures acquisition of a skill the model
has certainly never seen. Those are different questions, and the second cannot be
generalised to the first without an argument this recipe does not make.

## 1. The blocker: two of the three ladder points cannot run this at all

`rlvr_trainer.py:193-200` **refuses a base checkpoint**, by classification and not by
filename:

    kind = classify(cfg, os.path.basename(args.resume))
    if kind == "base":
        raise SystemExit("refusing: ... Every prompt here is ChatML, a prefix a base
        checkpoint has never seen, so the rewards would measure format rather than
        reasoning. Run SFT first, or pass an SFT/RL checkpoint.")

Of the three ladder points, only **p324 has an SFT** (`sft_p324_v2/v3/v5`, status ok in
`runs/experiments.jsonl`). p04 and p16 have base checkpoints and no SFT run at all.

So experiment 2 as specified — a curve **across** pretraining-token checkpoints — cannot be
run today on one card. It needs two more SFT runs first, on the same pack and recipe as
p324's, or the "curve" is a single point.

This also collides with fb's ruling that both arms use a base checkpoint in continuation
format. **The trainer will not do it**, and its refusal is deliberate: continuation-format
RLVR is recorded there as a different method, not a fallback (e1-22, 2026-09-02). Resolution
in §3.

## 2. What the code actually does, against what its docstring says

Read at `09041709`. Three disagreements matter for the recipe:

**The GSPO ratio is identically 1.** `rlvr_trainer.py:110` sets `old_lp = seq_lp.detach()`
in the same forward, so `ratio = exp(seq_lp - old_lp) == 1` and the clip at 1±0.2 can never
bind. The effective loss is plain policy gradient plus the KL term — which is what the
module docstring line 8 calls GRPO. **`--clip_eps` is a live flag that changes nothing at
any value.** Do not report this run as GSPO and do not tune clip_eps.

**There is a KL anchor and a third forward.** `ref_model` is a frozen deepcopy of the SFT
weights, never synced (`:211-215`); KL is the k3 estimator on length-normalized per-token
means, `kl_beta` default 0.02 (`:113-116`). Every group costs three forwards: rollouts,
policy, reference.

**No token accounting exists.** The log dict holds `reward, n, loss, n_loss, gnorm, gen`
(`:247`), where `n` counts responses, not tokens. Nothing sums generated or consumed
tokens. **The primary x-axis of this experiment is not currently instrumented**; §5 is the
smallest patch that adds it.

Other measured facts about the run: prompts per step are `--batch` (default 4) **globally,
not per GPU** — under DDP every rank seeds `random.seed(1337+step)` and samples the same
prompts (`:265-267`). Generation has no KV cache and re-forwards the whole prefix at every
decode step, truncated to the last 1024 tokens (`rlvr_generate.py:19-20`).

## 3. Arms, checkpoints and format

fb's ruling was: both arms on the same base checkpoint family, both scored in continuation
format, the RL rollout prompt format the same as the scoring format. The first half is
impossible (§1). The recipe holds the *intent* — format must not be a second variable — and
changes the level at which it is held:

| | RL arm | pretraining-token arm |
|---|---|---|
| checkpoint | `ckpt_sft_p324_v5.pt` | `ckpt_p324.pt` + n S-instances of continued training |
| format seen in training | ChatML (`format_prompt`, `rlvr_trainer.py:274`) | same ChatML wrapper on the same instances |
| format at scoring | ChatML, identical string | ChatML, identical string |

**Both arms are ChatML, both start from the SFT checkpoint. This is compliance with fb's
ruling, not a deviation from it.** What fb asked for is that format not be a second
variable. `rlvr_trainer.py:193` refuses base + continuation by classification, and its
refusal cites the measured size of the thing being controlled: `be.l1_fewshot_p324` puts
the format effect at **+38.2pt** with the model held fixed. An effect that large is not a
nuisance parameter next to a conversion rate — it is bigger than the signal. Given the
code, ChatML on the SFT checkpoint is the only configuration in which format is constant
*and* both arms can run at all, so it is the only way to honour the instruction.

The pretraining-token arm is *continued pretraining on S instances*, not RL — it is the
denominator the RL arm is compared against, and it must see the identical prompt strings.

## 4. Verifier

**Not `rlvr_reward.py`.** Measured: `reward_fn("42", "42") == 0.0` — the reward path
requires `\boxed{}` in the generation (`rlvr_reward.py:19-20, 86-88`) and the S/P sets
contain no `\boxed{}` anywhere (grep: 0 in all 10,192 rows).

**The failure is silent and total, not a low score.** e1 traced the path further than I
did and is right: `keep = 1.0 if 0 < sum(r) < len(r) else 0.0` (`:298`) drops every
all-zero group, then `if not kept: continue` (`:303`) skips the step *before* the forward.
With `\boxed{}` absent, every group is degenerate at every step, so **the run takes zero
optimizer steps and exits 0**, and the final checkpoint is bit-identical to the initial
one. The only trace is `degen N` inside the periodic log line (`:375`), and `n_degenerate`
resets to 0 each interval (`:384`) — a per-interval count, never a total. A 500-step run
prints "all groups degenerate, skipped" 500 times and looks like a completed run.

So the recipe also requires: **refuse when the degenerate fraction over the first 20 steps
is 1.0.** "No gradient was ever applied" and "RL did not help" produce the same artifact,
and only the refusal separates them.

Per fb's ruling, the frozen prompts are not changed. A separate exact-match verifier:

- decode with `skip_special_tokens=True`, take the **last integer** in the generation
  (`-?\d+`), compare to `answer` as an int.
- **Take the last, not any.** The answer string already appears somewhere in the
  *instruction* in 122 of 1000 S_test rows; a scorer searching prompt+completion jointly
  would false-positive at that rate.
- Negative answers matter: 284 of 1000 S_test answers are negative, and `-55` tokenizes as
  two tokens `['-','55']`. Match on the decoded string, never on token ids.
- Guessing floor is low: the most common answer covers 2.7% of S_test, 1.6% of P_test.


## 5. Chosen constants, each with its reason

| constant | value | chosen because |
|---|---|---|
| `--max_new` | **64** | measured: reference solution+answer is p95 54 tokens, max 60 over all 10,192 rows. The default 512 would spend 8× the generation budget on padding. **This is the single largest cost lever in the recipe, and the least safe number in it** — see below |

**The max_new caveat, e1's, and it cuts against the number I chose.** The p95 of 54 is a
property of the *reference* solutions. What gets truncated is the *model's* generation, and
a model that has not learned the format emits longer, not shorter. A generation cut at 64
scores 0 under exact match and is then indistinguishable from a wrong answer — the metric
degrades in the direction that looks like "the skill was not acquired", which is the one
reading this experiment must not manufacture. So: **the log dict gets a truncation counter
in the same patch as the token counter (§2), and 64 stands only until a pilot reports the
p95 of actual generations.** If truncation exceeds ~1% of rollouts, raise it.

| `--group_size` | **8** | the trainer's default and the pass@k gate's k. Also sets the FP8 pad q=2. |
| `--batch` | **4** | trainer default; global, not per-GPU (§2) |
| `--temperature` / `--top_p` | **0.8 / 0.95** | trainer defaults, and the pass@k gate is specified at 0.8 |
| `--lr` | **1e-6** | trainer default; fp32 master weights exist specifically so this survives bf16 ULP |
| `--kl_beta` | **0.02** | trainer default. Not tuned: with the ratio pinned at 1 (§2), KL is the only thing bounding the update |
| `--clip_eps` | leave at 0.2 | inert (§2) |
| `--steps` | **set from the token budget, not chosen directly** | see below |
| seeds | **2 per curve point** | 62's conversion-rate spec. No RLVR design in the tree names a seed count |

**Tokens per RL step — the stated assumption.** With max_new 64, group_size 8, batch 4:

    generated tokens per step  =  4 prompts x 8 responses x <=64 tokens  =  <=2,048

This is an upper bound; early `<eos>` reduces it. It is **not** the compute per step: with
no KV cache, generating those 2,048 tokens costs sum-over-t of a full forward, so
*forward-token* volume is roughly 64× higher. The recipe reports sampled tokens as the
x-axis and records the forward-token multiplier beside it, because the first is what the
comparison needs and the second is what the card actually spends.

## 6. Readout

Per fb's ruling, **two columns against the same pretraining baseline**:

| column | counts | answers |
|---|---|---|
| **generated tokens** (primary) | every sampled token, zero-reward rollouts included | what does this capability cost in compute actually spent |
| **trained-on tokens** (secondary) | tokens inside kept groups only | how much signal per token of supervision |
| **consumed tokens** | 0 on this arm, always | the column experiment 1's arm fills, carried here so the two ledgers join |

The ratio between the first two is fixed by the constants above — it is
`kept_groups / total_groups`, nothing else — so both columns come from one run and the row
records `group_size`, `max_new` and the kept fraction so the ratio is reconstructible.

**Both ledgers carry all the column names, with the other side's at zero** (4c's ruling,
2026-09-05). This arm generates and consumes nothing; experiment 1's continued-pretraining
arm consumes and generates nothing. A single `tokens` column would add an exposure the model
READS to a rollout the model WRITES, and the pretraining-vs-RL comparison these two ledgers
exist for is exactly a comparison of those two quantities. The zero is written as a literal:
a reader joining the ledgers cannot tell an absent column from an unrecorded one.
`runs/rlvr_tokens_<out>.jsonl` carries `tok_generated`, `tok_trained` and `tok_consumed: 0`
per logged step; experiment 1's row is `runs/prereg.jsonl#conversion_rate_0905`,
`token_accounting`.

The y-axis is **accuracy on S_test minus accuracy on P_test**, both at n=1000, matching the
sets' own contract. P is not a baseline to subtract for noise; it is the format control. S
and P rising together is format acquisition, S lagging and closing is the skill.

**S_test's readout and its floor are pinned to a sha, and the floor is not 25%.** The 4-way
set is `data/probes/novel_ops/S_test_4way.jsonl`, and it must be scored **per program** —
`diamond_chain` and `diamond_chain4` separately, never pooled. Content-free floor 0.364 and
0.294 respectively (`facts/contamination.json#cont.novel_ops_frozen_sets`,
`four_way_content_free_floor`); a pooled number cancelled a z=+12.81 cell against a z=−12.19
one in an earlier build and read as chance. That floor is a maximum over the battery's rules
and is a lower bound, not a certificate — it rose twice on a byte-identical artifact as rules
were added. The operational floor is the no-injection control arm's own score. P_test's
readout is **mean per-token NLL**, not 4-way: a 2-operand chain has no intermediate, so the
carry never fires and all three readings agree on 1000/1000 items, which makes a 4-way P
score a different quantity from a 4-way S score (e1's amendment 1).

Curve points: pretraining-arm n in **{1, 8, 64, 256}**. 512 and 4096 are NOT run — at 104.0
tokens/doc, 4096 exposures is 425,984,000 injected tokens against a 500-step arm's
131,072,000, i.e. 325%, which is fine-tuning on S with pretraining mixed in rather than an
injection. Above 256 the answer is reported as a bound, never as a measured zero (4c's ruling
on e1's measurement). The RL arm's x-values are wherever its token counter lands; the two
curves are plotted against a shared token axis, which is the whole comparison.

## 7. Preconditions. None of these is closed, so this recipe is NOT schedulable

1. **Two more SFT runs**, on p04 and p16, on the same pack and recipe as p324's. Without
   them "a curve across pretraining-token checkpoints" is a single point (§1). This is a
   precondition of the same kind as the rest of this list, not a note in the prose (e1).
2. **pass@8 − pass@1 ≥ 15pt** on `ckpt_sft_p324_v5.pt` (AGENTS.md hard rule,
   `eval/math_hard.py --k 8 --temperature 0.8`). If RL has no headroom to exploit, the curve
   measures nothing. Never run on this checkpoint.
3. **Generation throughput, measured.** `efficiency_gap_views.md:371` names this as the
   prerequisite and says it is unmeasured: at 200M rollout throughput is the binding
   constraint, and nothing in `facts/efficiency.json` covers decode. Until it is measured,
   no schedule for this experiment is honest.
4. **The MDE, pre-registered from n and the base rate** (`efficiency_gap_views.md:376`).
   Needs the base rate from precondition 2, so it cannot be computed here.
5. **A pilot's generation-length p95**, to confirm or raise `max_new` (§5).

Two more that are cheap and belong in the run itself rather than before it:

- **A negative control at the same rollout budget** — constant reward — or a rising curve
  cannot be separated from drift under the KL anchor.
- **Sets verified by sha256** against `cont.novel_ops_frozen_sets` at run time. A curve
  scored against any other hash is a different measurement.
