# Trainable CSA2 indexer — design (STE vs auxiliary loss), v41f-defined

Status: PROPOSAL for fb + de. Design only; no `.py` changes in this PR. Implementation
does not start until this and `v41f_train_checkpoint_design.md` (#447) are both approved.

---

## 0. The problem and the faithfulness boundary

The two-level selector in `v41f/indexer.py` is a faithful port of the inference-only
upstream module and every selection in it is discrete:

- level 2 (entry selection): `index_score.topk(k).indices.sort().values`, then
  `torch.where(idxs < compress_lens, idxs+offset, -1).int()` — integer indices
  (`v41f/indexer.py` `Indexer.forward`; ref `third_party/deepseek_v41_ref/model_ref.py.ref:578-580`).
- level 1 (block prefilter): `select_candidate_blocks` reduces each block by `amax`,
  pins the newest block to +inf, takes `topk`, and scatters a **bool** mask
  (`v41f/indexer.py`; ref `:607-613`).

The only trainable tensors are `Indexer.wq_b.weight` and `Indexer.weights_proj.weight`,
which enter through `score()` (`v41f/indexer.py:80-91`). Both selection outputs stop the
gradient: `.indices` and a bool scatter have no derivative. Under today's single
`shifted_cross_entropy` (`v41f/loss.py:14`) these two parameters receive **no gradient at
all** — they sit in the optimizer as dead weights, updated only if an explicit auxiliary
path is added.

**Faithfulness ruling (per prereg rules, not attributed upstream).** The vendored model
has no differentiable path through the indexer and no auxiliary objective for it. The ref
module ends in hard `topk(...).int()` and hard bool scatter; there is no straight-through,
soft, Gumbel, or load-balancing term anywhere in `model_ref.py.ref`. The mechanism's
provenance note (ref `:94`, "Names match DeepSeek-V3.2-Exp, where this mechanism first
appeared") is a literature pointer, not a training recipe. **Any training signal for the
indexer is a v41f-defined decision.** The prereg row states this explicitly; no claim is
made that DeepSeek trains the indexer this way, and the inference numerics must stay
bit-identical whether the training path is on or off (§3).

Scope: prefill training only. v41f trains teacher-forced prefill; the decode path
(`start_pos != 0`) is inference-only and is never asked to carry gradients. Level-1
candidate prefilter is config-gated off in v41f-S (`candidate_source_layer=-1`; the model
call at `v41f/attention.py:175` passes no `candidates`), so the design covers it but it is
dormant on the current small shape.

---

## 1. Options

### Option A — straight-through estimator (STE) on the hard selection

Keep the forward exactly as today (hard `topk`, integer gather, bit-identical KV actually
attended), but let the gradient pass through the selection as if it were the identity:

    soft = score                       # continuous [b,s,t], already masked in-place
    hard_idx = soft.topk(k).indices    # discrete, used by the forward
    # surrogate: selected entries = soft.gather(hard_idx); backward treats the
    # gather mask as constant and sends dL/d(selected score) into `soft` at picked slots

The attended compressed KV is still the hard-gathered one — forward numerics do not change.
On the backward, only the `k` selected entries per query receive gradient, through the
continuous `score` that produced them. `wq_b` / `weights_proj` then learn "raise the score
of entries the main CE found useful," evaluated through the real attention output, not a
surrogate target.

Cost: small. Needs an explicit custom-autograd `Function` (or
`score[...].detach()*0 + gather(score)` style) because native `topk().indices` carries no
grad. No new loss term, no scalar weight to tune, and it cannot perturb the main CE in the
forward (the hard selection is unchanged).

Known weakness (labeled, not hidden): STE is a biased estimator — gradient magnitude ignores
that selection is a step function, and non-selected entries get exactly zero signal even
when they were a near-tie. It trains the selected slots, never the margin against rejected
slots. It also cannot regularize *which* entries are picked (no diversity/collapse term).

### Option B — auxiliary differentiable objective over `score`

Leave the hard forward untouched and add a side loss computed on the continuous `score`
(pre-topk), with a tunable coefficient `λ`. Two candidate targets:

- **B1, margin / ranking loss.** Push the score of the hard-selected entries above the best
  non-selected entry by a margin (a hinge over the same `topk` partition, using `score`
  values so it is differentiable). This sharpens the decision the forward already made.
- **B2, load/coverage balance.** Penalize concentration of selections across compressed
  positions (an entropy or cv² term over a softmax over entry-use), the MoE load-balancing
  analogue. Target: stop the indexer collapsing onto a few always-picked positions.

Cost: a new term in the loss, a coefficient that must be scheduled and swept, and a risk
that the auxiliary objective fights the main CE (it optimizes a property of the *scores*,
not the LM outcome). B2 in particular can be anti-correlated with quality — balanced use is
not the same as useful use — and needs a metric showing it helps downstream before it is
believed (repo rule: a policy gets a metric, not just a term).

### Option C — hybrid (STE + optional balance aux)

STE carries the main-CE signal to the selected slots; B2 is added only with a small
coefficient and only if an ablation shows selection collapse and a downstream gain. This is
the recommended target but **in two separately-gated landings**, not one change:

1. STE alone, on/off switch, default **off**, proven bit-identical forward and dead-grad
   → live-grad backward.
2. Balance aux later, its own prereg amendment and ablation, default off. No balance term
   is written in the first change (YAGNI; the collapse it prevents is unmeasured).

**Recommendation: implement Option A first; treat B/C as a recorded follow-up gated on a
measured collapse or a measured downstream loss.** Rationale: STE is the only option whose
training signal is the main CE itself through the real attention path (lowest risk of an
auxiliary that diverges from the goal), it needs no tunable scalar, and it cannot change
inference numerics. B2 adds a tunable regularizer for a failure mode that has not been
observed on v41f-S; the repo's standing rule is not to build against an unmeasured
requirement.

---

## 2. Interfaces (no behavior change to the faithful modules)

The training hook must attach to the continuous score the indexer already computes, and must
reuse that exact tensor — never recompute it. Recomputing the score in a training-only path
is the fed-dead-weight shape (#434/#436/#438): an allclose over fed values goes green while
the real default path leaves `wq_b`/`weights_proj` unused.

- **`Indexer.score()` returns the masked continuous score.** Today `forward` masks
  `index_score` in-place and then topks it. The change exposes the post-visibility-mask
  continuous tensor (the same object the hard topk reads) so a training adapter can attach
  STE without a second projection. Selection math is unchanged.
- **Selector is factored as `score -> hard idxs` with a named seam.** The seam returns
  `(idxs_int, score_continuous)`; inference and the bit-exact tests consume `idxs_int`
  exactly as now. The STE adapter wraps the gather of `score_continuous` at selected slots;
  it is constructed only when training-with-STE is enabled.
- **Compressor interface is unchanged.** The softmax-gated compressor
  (`v41f/compressor.py`) produces the latent; the indexer scores `index_key(latent)`. STE
  differentiates `score` w.r.t. `wq_b`/`weights_proj` only. Whether gradient flows further
  into `index_key`/the compressor latent is an explicit decision: **first landing keeps the
  STE target on score only** (indexer-local params), so compressor/attention numerics and
  their existing faithful allcloses are untouched. Extending STE grad into `index_key` is a
  later, separately-tested option.
- **Candidate prefilter (level 1) keeps a hard bool mask forward.** When enabled
  (`candidate_source_layer >= 0`), its forward is unchanged. If a level-1 training signal is
  wanted later it gets its own STE seam over the block-`amax` reduction's underlying scores,
  not a soft block mask in the forward. v41f-S leaves it off, so no level-1 STE is built
  now.
- **Loss.** `v41f/loss.py` gains no term for Option A. The STE gradient reaches the
  indexer through the existing `shifted_cross_entropy` → sparse-attn → gathered-KV path;
  nothing is added to the scalar loss. Option B, if ever built, adds a separate
  `indexer_aux_loss(score, idxs)` composed by the trainer as `loss = ce + λ*aux`, never
  edited into the CE function.

---

## 3. Switch, default, and main-CE preservation

- A single config flag, proposed `indexer_train_mode: "off" | "ste"` (default `"off"`;
  future `"ste+aux"`). `off` is byte-identical to today on both forward and backward —
  proven by an equality test against the flag-off path, not assumed.
- Default off means the faithful P0/P1 allclose suite, the inference checkpoint, and every
  existing numerics test see no change. The indexer params remain outside the fp32-master
  optimizer group while off (consistent with #447 §1.1: indexer params are saved
  structurally but carry no m/v until a gradient path exists). Turning `ste` on moves them
  into the optimizer group; #447's membership rule ("in optimizer group ⇒ has master") then
  covers them with no special case.
- STE changes only the **backward**. Assertions:
  - forward: hard-selected idxs and attended logits identical between off and ste
    (`torch.equal(idxs)`, logits bit-exact on CPU fp32);
  - backward off: `wq_b.weight.grad is None` or zeros (dead, as today);
  - backward ste: `wq_b.weight.grad` / `weights_proj.weight.grad` are non-zero, finite, and
    flow **only** from CE — no auxiliary scalar exists yet, so any non-CE gradient is a
    wiring fault.
- No tunable coefficient exists in Option A, so there is no knob that can distort the main
  CE. The risk surface is restricted to a biased-but-scaled gradient; a gradient-norm guard
  (indexer grad norm finite and not exploding relative to attn grads) is logged, not used to
  silently clip.
- `eval()`/inference never instantiates the STE adapter: it is a training-only
  custom-autograd path, and the inference loader/format is untouched.

---

## 4. Mutation / gate test list

New `tests/v41f/test_p1_indexer_train.py`, registered in pre-commit SELFTEST_FILES in the
same PR (an unregistered selftest is no gate). Direct-runner compatible; temp state
process-private per the #441 lesson. Numbered for review; each mutant must turn a NAMED
assertion red, not merely crash.

Correctness (green on real code):

1. **off = today**: with `indexer_train_mode="off"`, selected idxs and logits are
   bit-identical to a run on the unmodified selector; both indexer grads are None/zero.
2. **STE forward hard**: with `"ste"`, `torch.equal` of hard idxs vs off, and logits
   bit-identical (forward does not soften).
3. **STE backward soft/nonzero**: finite, non-zero grads on both `wq_b.weight` and
   `weights_proj.weight`; shapes equal the parameter shapes; grad reaches indexer params
   from CE alone.
4. **gradient provenance**: only `wq_b`/`weights_proj` receive new grad; compressor /
   `index_key` / attention param grads are unchanged between off and ste (indexer-local
   scope, §2).
5. **score-seam identity**: the continuous tensor the adapter reads `is` the tensor the
   hard topk consumed (same `data_ptr`/object), proving no recomputation.
6. **masked positions excluded**: visibility/candidate -inf slots carry no gradient and are
   never selected; the pinned newest block (level 1) is always selected and behaves under
   autograd without NaN.
7. **optimizer-group membership**: off ⇒ indexer params absent from optimizer param group
   (no m/v); ste ⇒ present with fp32 master under #447. Loads/saves round-trip either way.
8. **inference isolation**: `eval()`/inference checkpoint path contains no STE Function;
   logits equal the off path.

Mutants (each names the assertion that dies):

- M1 replace STE backward with zero / detach the score ⇒ gate 3 (nonzero grad) fails.
- M2 make the forward soft (weighted gather instead of hard) ⇒ gate 2 `torch.equal(idxs)`
  and logits bit-exact fail — pins "forward stays hard."
- M3 recompute score on the training side instead of reusing the seam tensor ⇒ gate 5
  (object identity) fails; this is the fed-dead-weight regression.
- M4 route STE grad into `index_key`/compressor ⇒ gate 4 (scope) fails.
- M5 drop the visibility mask before the STE gather ⇒ gate 6 fails (a -inf slot gets grad).
- M6 leave STE on under `eval()` ⇒ gate 8 fails.
- M7 forget to move indexer params into the optimizer group when ste ⇒ gate 7 fails (params
  have grad but are never stepped — a silent no-training fault).
- M8 default the flag to `"ste"` ⇒ gate 1 fails (default-off is a hard property, and the
  faithful allclose suite would move).
- M9 (level-1, only when enabled) soften the block mask ⇒ a forward-identical gate fails.

Non-goals recorded N/A: no balance/ranking aux in this change (Option B/C deferred, needs a
measured collapse + downstream metric); no STE gradient into compressor/index_key yet;
decode path stays inference-only and untested for grad; `λ` scheduling does not exist until
an aux term does.

---

## 5. Open questions for de/fb

1. STE custom-autograd placement: a small `v41f/indexer_ste.py` (training-only, never
   imported by inference) vs an inline `torch.autograd.Function` in `indexer.py`. Recommend
   the separate file so the faithful port file stays a 1:1 read against the ref.
2. Confirm indexer-local STE scope (grad to `wq_b`/`weights_proj` only) for the first
   landing, vs extending into `index_key` immediately. Recommend local.
3. Gate the first STE landing on a short CPU train-smoke asserting indexer grads are
   non-zero and the main-CE loss still decreases — no claim it improves HumanEval; it only
   proves the params are trained. A real usefulness claim needs the GPU ablation later
   (off vs ste, downstream HumanEval delta), which is a separate prereg question.
4. Flag name/grouping with #447's optimizer-membership rule — confirm
   `indexer_train_mode="ste"` is the single switch that also moves master/m/v.
