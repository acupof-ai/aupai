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
soft, Gumbel, or load-balancing term anywhere in `model_ref.py.ref`. The mechanism's provenance note (ref `:94`, "Names match DeepSeek-V3.2-Exp, where this
mechanism first appeared") is a literature pointer, not a training recipe. The single
Gumbel in the file (`model_ref.py.ref:1285-1292`, a standalone `sample()` doing
Gumbel-max token decoding after the MTP head) is unrelated to the indexer and must not be
read as an upstream soft/straight-through indexer path; it is post-head sampling, not
selection. **Any training signal for the indexer is a v41f-defined decision.** The prereg
row states this explicitly; no claim is made that DeepSeek trains the indexer this way, and the inference numerics must keep the hard selection identical (equal idxs/weight) with logits held to a sub-ULP bound whether the training path is on or off (§3).

Scope: prefill training only. v41f trains teacher-forced prefill; the decode path
(`start_pos != 0`) is inference-only and is never asked to carry gradients. Level-1
candidate prefilter is config-gated off in v41f-S (`candidate_source_layer=-1`; the model
call at `v41f/attention.py:175` passes no `candidates`), so the design covers it but it is
dormant on the current small shape.

---

## 1. Options

### Option A — straight-through estimator (STE) wired into the attention softmax

Keep the forward exactly as today (hard `topk`, integer gather, bit-identical KV actually
attended), but give the hard per-slot selection indicator a differentiable twin that is
consumed BY the attention the main loss runs through. The STE tensor is not a side output; a
surrogate that is computed and never fed to a consumer leaves `score.grad`/`wq_b.grad` None
(this was the defect in the first draft of this option: `score.gather(hard_idx)` on its own is
a dangling tensor, and the real path `score -> topk -> .int -> sparse_attn` discards it —
independently measured, both grads are None whether or not the gather exists).

Concrete construction (de d4f, measured on the real path 2026-09-17):

    # selector emits, alongside hard idxs, the CONTINUOUS scores at the selected slots:
    sc = score.gather(-1, hard_idx)          # [b,m,k], a view of the SAME score tensor
    a  = one-hot / hard selection weight actually used in the gather (forward)
    sf = softmax(sc)                          # differentiable over the k selected slots
    p  = a + (sf - sf.detach())              # STE: forward == a, backward == d softmax

`p` is passed to `sparse_attn` as the selection weight in the softmax it already computes
over the k gathered slots (plus the sink). Forward: `sf.detach()` cancels `sf`, so
`torch.equal(p, a)` is True (the STE tensor IS the hard weight bit-for-bit). The attention
output is then equal to the hard path within a sub-ULP bound, NOT bit-exact: `a` is the
uniform weight `1/k` across the k selected slots, so it factors out of
`exp/Σexp` algebraically, but `p` carries the non-constant fp expansion `(sf-sf.detach())`,
and evaluating that expansion through `exp`/denom does not cancel bit-for-bit — measured
(de d4f) logits max_abs fp32 **2.98e-8**, bf16 **2.44e-4**, a constant shift of every
logit, not a relative redistribution. Gate 2 therefore asserts TWO things separately:
`torch.equal(p, a)` and logits within a one-ULP tolerance (de's measured values), never
bit-identical logits. Backward: the `- sf.detach()` cancels in value but not in the graph,
so gradient flows through the already-computed softmax `sf` into `sc`, into `score`, into
`wq_b`/`weights_proj` — measured forward `torch.equal(p,a)` True and
`wq_b.weight.grad = 111.836`, finite. The selection the main CE used is still the hard one;
only the gradient is soft.

`wq_b` / `weights_proj` then learn "raise the indexer score of entries the attended output
found useful," differentiated through the real attention and the real main CE — not a
surrogate target.

Cost: small. The seam exposes the gathered `sc`; the STE is one add with a detached term;
no new loss term and no scalar weight, so it cannot perturb the main CE forward (hard
selection unchanged). No custom `torch.autograd.Function` is needed — the
`x + (f(x) - f(x).detach())` identity is the standard STE and keeps the forward a plain
hard tensor.

Do NOT use a multiplicative `score * log(gate)` coupling: it is not element-wise across the
selection and produces NaN grads; the additive STE above is the measured-good form.

Known weakness (labeled, not hidden): STE is a biased estimator — gradient magnitude ignores
that selection is a step function, and non-selected entries get exactly zero signal even
when they were a near-tie. It trains the selected slots, never the margin against rejected
slots. It also cannot regularize *which* entries are picked (no diversity/collapse term).
A second, subtler bias specific to this construction: the hard weight `a = 1/k` is UNIFORM,
and adding a value-cancelling soft term leaves the forward a pure constant rescale — so in
forward the chosen entries receive no relative preference, and in backward the signal says
"the attended output was sensitive to these slots' scores," not "entry X should outrank
entry Y within the chosen set." It is a relevance signal on the selected set, not a ranking
signal; that ranking margin is exactly what an Option B1 aux loss would add if ever needed.

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

1. STE alone, on/off switch, default **off**, proven to keep the hard forward (equal
   idxs/weight, sub-ULP logits) and turn dead grads into live finite grads;
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
- **Selector is factored as `score -> (hard idxs, selected scores)` with a named seam.** The
  seam returns the integer `idxs_int` the gather uses AND the continuous scores gathered at
  those slots, `sc = score.gather(-1, hard_idx)` (a view of the same `score` object, not a
  recompute). Inference and the bit-exact tests consume `idxs_int` exactly as now. When STE
  is on, the adapter builds `p = a + (softmax(sc)-softmax(sc).detach())` (Option A) and
  passes `p` INTO `sparse_attn` as the per-slot selection weight in its softmax — the STE
  tensor is a consumer input, never a dangling side tensor (the first draft's
  `score.gather(...)` left unused was the bug: grads stayed None). It is constructed only in
  training-with-STE; the off path never builds `sc`/`p`.
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
  existing numerics test see no change. The indexer params use the three-state lifecycle of
  `v41f_train_checkpoint_design.md` §6 (#447): off = **present-dormant** (module built,
  bf16 weights saved with `param_meta grad=False`, no master/m/v); ste = **in-group**
  (AdamW group ⇒ fp32 master + m/v). A level-1 indexer module that v41f-S does not
  instantiate is **absent** (never in the blob). Turning the mode on is the only thing that
  moves `wq_b`/`weights_proj` present-dormant → in-group, with no format migration; #447's
  membership rule ("in optimizer group ⇒ has master") covers them with no special case.
- STE changes only the **backward**; the forward idxs are hard and the STE weight bit-equals
  the hard weight, but the downstream logits are NOT claimed bit-exact (see Option A: a
  uniform `1/k` cancels through softmax only algebraically, not bit-for-bit). Assertions:
  - forward idxs/weight: hard-selected idxs identical off-vs-ste (`torch.equal(idxs)`), and
    `torch.equal(p, hard_weight_a)` — the STE tensor is bit-for-bit the hard uniform weight;
  - forward logits: within a ONE-ULP bound of the off path, not bit-exact — measured de d4f
    max_abs fp32 2.98e-8, bf16 2.44e-4 (a constant shift). A genuinely soft M2 mutant moves
    logits by 5–6 orders of magnitude more, so the ULP gate still separates hard from soft;
  - backward off: `wq_b.weight.grad is None` or zeros (dead, as today);
  - backward ste: `wq_b.weight.grad` / `weights_proj.weight.grad` are non-zero and finite,
    and reach them THROUGH the `p -> sparse_attn softmax -> CE` consumer graph — proven by
    the grad being present when `p` is fed and None when the identical gathered `sc` is left
    unconsumed. (de measured `wq_b.grad=111.836` finite on the real path; the additive-STE
    forward-identity/backward-soft identity is independently proven on a minimal tensor.)
    No auxiliary scalar exists yet, so any non-CE gradient is a wiring fault.
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
   bit-identical to a run on the unmodified selector (the STE path is never built); both
   indexer grads are None/zero.
2. **STE forward: hard tensor + sub-ULP logits (two separate assertions).** With `"ste"`:
   (a) `torch.equal` of hard idxs vs off AND `torch.equal(p, hard_weight_a)` — the STE
   tensor is bit-for-bit the hard uniform weight `1/k` (M1c kills this); (b) logits differ
   from the off path by at most a ONE-ULP bound, NOT bit-exact — de d4f measured max_abs
   fp32 2.98e-8 / bf16 2.44e-4. The same ULP bound must reject a genuinely soft M2 mutant
   by 5–6 orders of magnitude, proving it separates "hard weight, sub-ULP eval" from
   "softened forward." Asserting bit-identical logits here is WRONG and must not be written:
   the uniform weight cancels through softmax only algebraically.
3. **STE backward soft/nonzero**: finite, non-zero grads on both `wq_b.weight` and
   `weights_proj.weight`; shapes equal the parameter shapes; grad reaches indexer params
   through the `p -> sparse_attn softmax -> CE` consumer graph, from CE alone.
9. **STE tensor is actually consumed (the first-draft regression).** With gathered `sc` and
   `softmax(sc)` built but NOT passed to `sparse_attn` (a dangling tensor), both indexer
   grads are None; feeding the STE `p` into the sparse_attn softmax flips them to gate-3
   nonzero. This gate separates a wired STE from a computed-but-unused surrogate.
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
8. **inference isolation**: `eval()`/inference checkpoint path builds no STE `sc`/`p`;
   logits equal the off path.

Mutants (each names the assertion that dies):

- M1 zero the STE backward (`p=a` with no soft graph, or `sf.detach()` only) ⇒ gate 3
  nonzero-grad fails.
- **M1b compute `sc`/`p` but never pass `p` into `sparse_attn` (dangling surrogate) ⇒ gate
  9 fails — the exact first-draft defect that left the as-written grads None.**
- M1c drop the `-sf.detach()` cancellation (`p = a + sf`) ⇒ gate 2(a) `torch.equal(p,a)`
  fails (p is no longer the hard weight) and logits leave the sub-ULP bound.
- M1d replace the additive STE with a multiplicative `score * log_softmax(...)` coupling ⇒
  NaN/non-finite at gate 3/6 (the form de measured bad).
- M2 make the forward soft (weighted gather instead of hard) ⇒ gate 2(a) idxs/weight
  equality and gate 2(b)'s sub-ULP logit bound both fail (the soft form moves logits 5–6
  orders beyond the ULP tolerance) — pins "forward stays hard."
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

1. STE placement: no `torch.autograd.Function` is needed — the additive
   `a+(sf-sf.detach())` identity is enough. The open choice is where the gathered `sc`/`p`
   crosses into `sparse_attn` (an optional per-slot selection-weight argument, default the
   hard one-hot) and whether the adapter lives in a small training-only
   `v41f/indexer_ste.py` so the faithful `indexer.py` stays a 1:1 read against the ref.
   Recommend the separate file + an opt-in sparse_attn weight the off path never builds.
2. Confirm indexer-local STE scope (grad to `wq_b`/`weights_proj` only) for the first
   landing, vs extending into `index_key` immediately. Recommend local.
3. Gate the first STE landing on a short CPU train-smoke asserting (a) gate 9 — an
   unconsumed `p` leaves grads None while the wired `p` yields the gate-3 nonzero finite
   grads — and (b) the main-CE loss still decreases. No claim it improves HumanEval; it only
   proves the params train. A usefulness claim needs the later GPU ablation (off vs ste,
   downstream HumanEval delta), a separate prereg question.
4. Flag name/grouping with #447 §6's optimizer-membership rule — confirm
   `indexer_train_mode="ste"` is the single switch that moves the indexer params
   present-dormant → in-group (master/m/v) and that an absent level-1 module is excluded.
