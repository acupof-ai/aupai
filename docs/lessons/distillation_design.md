---
question: How should the next version train by distillation from Qwen3.8-27B — which route, which seeds, what cost, what would falsify it?
status: open
source: facts/distillation.json (teacher vocab, seed inventory, throughput — all measured 2026-09-08); cards pending user decision
---

# Distillation pipeline design (Qwen3.8-27B teacher)

**Status 2026-09-08: PAUSED by user ruling — no cards for distillation, the 8 cards stay
with tileRL. The design is closed at run-ready state; section 6 is the pick-up checklist.**

4c's dispatch 2026-09-08. Deliverable: this design + a prereg row. Not code.

## 0. The vocab constraint, verified not accepted

Our vocab is 32,773 (facts/tokenizer.json#tok.vocab_size). The teacher's is **248,320** —
248,044 BPE entries (247,587 merges, Qwen2Tokenizer, read on the pod from
`/work/Qwen3.8-27B-NVFP4/tokenizer.json`, 2026-09-08) padded to a multiple of 128 in the
embedding table (`model.language_model.embed_tokens.weight [248320, 5120]`, read from
`model.safetensors`, 2026-09-09). Ratio 7.57x either way. Two different
vocabularies have no token-level probability alignment, so logits KL is not a tuning
problem — it is undefined. facts/distillation.json#distill.vocab_mismatch_blocks_logits_kl.

## 1. Route decision: sequence-level distillation

| route | verdict | why |
|---|---|---|
| 1. Sequence-level — teacher generates text, student SFTs on it | **CHOSEN** | vocabulary-agnostic; works today; every component (SFT pack, correctness verifiers, held-out panel) already exists |
| 2. Rebuild our vocab to the teacher's | REJECTED | a 248K-slot embedding+head would dominate a 200M student; invalidates every checkpoint including the six-point ladder and p324 — the same arithmetic was run for a 151K vocab and declined (docs/standards/0830v1_gates.md:691). The 2026-08-29 freeze makes any rebuild a user-owned decision, and unfreeze condition 2 is about OUR corpus changing, not about a teacher existing |
| 3. Cross-vocab logits distillation over aligned character spans | REJECTED as engineering | open research; no off-the-shelf method; the failure mode (misaligned spans silently teaching wrong distributions) is worse than route 1's known cost |

Route 1's known cost is real: the student sees the teacher's mode, not its distribution.
The mitigation is rejection sampling — generate K per prompt, keep the verifiably-correct
ones — which recovers distribution mass on the questions that matter, the ones with
checkable answers. This is the STaR/RFT recipe and it is the only one of the three whose
risk is a measured number rather than a research programme.

## 2. Seed set

**Source: the instruction side of existing SFT sets, not new synthesis and not the corpus.**

| seed | rows | register | why |
|---|---|---|---|
| openo1_sft.jsonl | 77,685 | long-CoT reasoning | already prompt-shaped; mean output 5,832 chars says the instructions elicit the register we want |
| gsm8k_sft.jsonl | 7,471 | math word problems | answer-verifiable cheaply |
| code: `data/sft/code_with_tests/0000_mined.jsonl` | **0 (empty)** | verifiable code | the supply task that gates the code half — mine function+test pairs from code_py_starcoder/rp1t before generation starts |

(facts/distillation.json#distill.seed_inventory, measured 2026-09-08.)

**Sampling: K=4 per prompt, T=0.7.** High enough that four samples differ, low enough that
most are coherent. Openo1: 77,685 × 4 = 310,740 generations. gsm8k: 7,471 × 4 = 29,884.

**Why not the corpus as seeds:** corpus documents are not prompts; mining problem
statements from math_owm is a second mining project stacked on generation. The SFT
instruction sets are already prompt-shaped and already cover the reasoning register.

**Why not eval prompts:** math-500 and code-500 are held-out. Training on them
contaminates the only falsification instrument we have.

**Acceptance filter — resolved 2026-09-08 (4c's question, measured the same day).**
Both seed files carry only `instruction`/`output` and no independent golds (openo1 keys =
`[instruction, output]`; gsm8k keys = `[instruction, output, source]`). Filtering new
generations against answers extracted from the seed file's own output is circular: it
measures agreement with the original teacher, not correctness, and a wrong original answer
inverts the filter. The filter is therefore per-domain, and every path is independent of
the seed file's output:

| domain | filter | independence |
|---|---|---|
| gsm8k (7,471) | answer match against the ORIGINAL GSM8K golds (`#### N`), re-fetched with the dataset — the golds our file dropped | fully independent |
| openo1-math | two-stage: (a) symbolic/numeric re-derivation where the problem permits (arithmetic, determinate algebra); (b) consensus — keep generations where ≥3 of K=4 agree — for the rest. Consensus measures confidence, not correctness; the acceptance rate is labelled a consensus rate. Leg (b) is conditionally activated — see the teacher-correctness premise below | independent of the original output |
| code | execute the teacher's code against the tests the problem embeds (openo1 code problems carry unittest assertions in the instruction) | execution is independent of any model output |

The filter may NOT extract an answer from the seed file's output and call a match
"correct". The pilot measures each filter's acceptance rate separately; a domain whose
rate is too low to fill its share drops out, and the dataset composition follows the
measured rates.

**Pilot readout — consensus vs difficulty, pre-registered 2026-09-08 before any pilot
data (4c order).** The ≥3/4 consensus leg selects teacher-stable problems, and
teacher-stable may mean easy. Fixed in advance: (a) correctness via the symbolic path on
the symbolic-verifiable intersection — the leg stands only if the consensus subset's
correctness is above the filtered-out subset's (point estimate, CI reported); (b) the
difficulty proxy is instruction length in chars (problem-side, no parser; step count
rejected because its parser definition would itself be a degree of freedom); collapse =
consensus-subset median length < 0.7× the filtered-out subset's median, with ≥100
problems per subset. If difficulty collapses: stratified sampling by length terciles
(boundaries from the full seed distribution), accept within tercile, per-tercile
consensus rates reported; if the longest tercile cannot fill its share within the service
budget, the consensus leg drops and openo1-math shrinks to the symbolic-verifiable subset
only. No silent drop, no post-hoc threshold. The standing condition rules only if the
symbolic-verifiable intersection yields both subsets with n ≥ 100 each (a 15pp
correctness gap needs ≈120 per subset for 80% power at α=0.05); below that the leg is
**unmeasured**, not passed, and the check re-runs on a larger dedicated sample. The
intersection's instruction-length distribution (median, p90) is reported alongside the
population's; if the intersection median is < 0.7× the population median, the slice is
systematically easier and the verdict is again unmeasured. A number is valid only on the
population it was measured on, and that population is written down.

**Teacher correctness premise, corrected 2026-09-08 (tilerl-0a, relayed by 4c).** The
teacher is **91% correct on level-5 math** at cap 6144, not 64% — the 64% was a lower
bound that counted the 32 unscored cap-hit samples as wrong; unscored is unknown, not
wrong (facts/distillation.json#distill.teacher_correctness_level5_math). This opens the
question the next owner judges FIRST (section 6, open question 1): is the ≥3/4 consensus
leg worth running at 91%? The leg exists to remove 9% of errors; its price is a
difficulty distribution that may collapse, and a collapsed SFT set teaches an easy-only
student. Recorded proposal (amendment 5): activate the leg only if the pilot clears the
pre-registered gates (standing condition, n≥100 per subset, representativeness, no
collapse) AND the accepted set's residual error is ≤ 4.5% on the symbolic-verifiable
intersection — at least half of the teacher's 9%; otherwise train on unfiltered teacher
output and accept the 9% noise as SFT-tolerable. The pre-registered pilot numbers are
what the judgment runs on, not the 64% premise.

**Length distributions join the mandatory reported quantities (tilerl-0a's criterion,
2026-09-08).** With the cap-hit rate, the **token-length distributions of cap-hit and
non-cap-hit samples are reported together** — the rate says how much is lost, the
distribution says which end. A continuous transition into the cap means genuinely-wrong
answers are mixed in; a gap before the cap means pure truncation (the 32 level-5 cut-offs
were confirmed via a 159-token gap at 1889–2048). The consensus comparison likewise
reports both subsets' **output-length distributions**: output length cannot be the
pre-registered difficulty proxy (it is known only after generation — circular), but the
reader judges the stacking from the distributions. The stacking is measured, not assumed:
cap-hit samples re-ran to a 3331-token mean against 1386 for naturally-terminating ones
(2.4×), while length and correctness are nearly uncorrelated inside naturally-terminating
samples (0–500 tok: 86% correct; 1000–1500: 100%) — truncation filters long reasoning,
and reasoning length is the common latent variable behind both filters.

**Cap-hit rate is a third pre-registered quantity (4c, 2026-09-08; tilerl-0a measurement
the same day).** Truncation is a second difficulty filter and it is aligned with the
consensus filter: long-reasoning problems hit the cap more, and high-consensus problems
skew easy — both drop hard problems, and they stack. cap 2048 truncated 32% of level-5
math generations, and 84% of those cut off were correct answers. Rules: (a) cap-hit rate
is a mandatory reported quantity per domain; (b) truncated samples (finish_reason=length
or output tokens == cap) are removed BEFORE any subset comparison — post-comparison
removal leaves a truncation-rate difference inside the correctness difference; (c) the
big-batch cap is chosen from a calibration batch of 200 problems generated once at cap
8192: one run yields the full truncation-rate-vs-cap curve — any smaller cap's rate is
the mass of the measured length distribution above it — and the length segment each cap
drops. The cap is set at the curve's knee (another +1024 tokens buys no meaningful
drop), subject to the dropped segment passing the registered collapse check: the
median-ratio ≥ 0.7 rule applied to the truncation-dropped subset versus the rest. A 5%
truncation that cuts the hardest end off entirely is too high; a 12% truncation that
leaves the distribution standing is acceptable. If no cap up to model context passes
both, the domain is reported cap-bound and its share is renegotiated. A batch run at cap 2048 that looks fine is a batch where 32% of rows are
half-answers. Offline baseline (measured 2026-09-08, 500-row sample seed 42,
facts/distillation.json#distill.openo1_answer_extractability): 81.0% of openo1 rows
carry an `<Output>` tag; of those, 78.3% yield an extractable numeric answer (last
number), so the answer-extraction population is 63.4% of the file. 19.0% have no
`<Output>` tag. The symbolic path's true coverage —
problems whose answer can be re-derived independently — is smaller than the extraction
population and is a pilot measurement, not this baseline. Of the 19.0% no-tag rows, the
7.0pp that are code-like belong to the code domain (embedded-unittest execution), not
math; the remaining 12.0pp is discarded as unjudgeable — cheaper than rescuing.

## 3. Teacher generation cost

Measured 2026-09-08 by tilerl-0a, relayed by 4c
(facts/distillation.json#distill.teacher_throughput_nvfp4): **130.3 tok/s per H20**,
NVFP4, batch 8, tp=1 — generation throughput, not forward. 105.0 tok/s incl. load and
JIT is a lower bound. That is 469k token/hr/card.

The planning number is the cap-6144 one: mean 3331 token/sample → **141 samples/hr/card**.
(At cap 2048 the mean is 1400 → 335/hr, but that rate is contaminated — cap 2048
truncates 32% of level-5 math and 84% of the cut-offs were correct answers, section 2.)

Full-scale openo1: 310,740 generations (77,685 × K=4) at 141/hr = **2,204 card-hours ≈
92 card-days** on one card; ≈ 12 days on 8 cards, where the 8-card figure is an
unverified linear extrapolation (multi-process interference unmeasured). gsm8k adds
29,884 generations, expected shorter.

Pilot: 1,000 generations ≈ 7.1 card-hours. Cap calibration batch (section 2): 200
problems × 4 = 800 generations at cap 8192 ≈ 7–8 card-hours (same order; the batch
measures its own mean).

Cards are tileRL's by user grant; the pilot needs cards back — a user decision 4c is
reporting with this budget. The bf16 copy is broken (MMLU 0.0%) and is not a fallback.

## 4. Falsification criterion

The route is falsified if the distilled student does not beat the unlooped control at
equal tokens and equal active params on the held-out panel (math-500, code-500, minimal
pairs) beyond the remeasured noise floor. σ̂ = 0.0516 does not transfer — it was measured
on KDA+MLA+AttnRes and the student is a different architecture; the remeasurement is a
prerequisite of the first MDE claim (4c ruling 2026-09-08, landed in
docs/standards/0830v1_gates.md).

A cost go/no-go gate sits BEFORE scaling, distinct from the falsifier: pilot 1,000
generations, measure throughput and acceptance, compute the full-scale cost. If the
measured cost prices the target dataset above the service-time budget 4c names, stop and
report — the route is correct but unaffordable, and that is a user decision.

## 5. What would change this design

- tilerl-0a's throughput numbers below the pilot's break-even (section 3 fills in, then
  section 4's gate fires).
- A working cross-vocab alignment method with a measured success case (route 3 reopens).
- The user ruling on vocab rebuild (route 2 reopens, owned by the user, not this design).

## 6. Pick-up checklist (cards arrive → run)

The leg is paused by user ruling. When cards are granted, run in this order:

**1. Judge open question 1 first** — is the ≥3/4 consensus leg worth running at 91%
teacher correctness (section 2 premise)? Do not run a design premised on 64%. The
recorded proposal is conditional activation at ≤4.5% residual error; the pilot's
pre-registered numbers are what the judgment runs on.

**2. Calibration batch** — 200 openo1 problems, K=4, T=0.7, cap 8192, seed 42. One run
yields the full truncation-rate-vs-cap curve (any smaller cap's rate is the mass of the
measured length distribution above it) plus the length segment each cap drops:

```bash
# Run on the pod. Service URL/port: read from tileRL's serving config — the NVFP4 27B
# service is tileRL's; this design does not hard-code its port. If the seed's training
# template differs from the service default chat template, apply the seed template.
python3 - <<'EOF'
import json, random, urllib.request

SERVICE = "http://127.0.0.1:<PORT>/v1/chat/completions"  # from tileRL config
MODEL = "Qwen3.8-27B-NVFP4"
SEED_FILE = "/work/aupai/data/openo1_sft.jsonl"
OUT = "/work/aupai/data/distill/calibration_200.jsonl"

random.seed(42)
rows = [json.loads(l) for l in open(SEED_FILE)]
with open(OUT, "w") as f:
    for i, r in enumerate(random.sample(rows, 200)):
        for k in range(4):
            body = json.dumps({"model": MODEL,
                "messages": [{"role": "user", "content": r["instruction"]}],
                "max_tokens": 8192, "temperature": 0.7, "seed": 42 + i * 4 + k}).encode()
            req = urllib.request.Request(SERVICE, data=body,
                headers={"Content-Type": "application/json"})
            resp = json.load(urllib.request.urlopen(req, timeout=600))
            c = resp["choices"][0]
            f.write(json.dumps({"id": i, "k": k, "finish_reason": c["finish_reason"],
                "tokens": resp["usage"]["completion_tokens"],
                "completion": c["message"]["content"]}) + "\n")
EOF
```

**3. Mandatory readouts** — every one reported, no exceptions:
- per-domain truncation rate (cap-hit / total)
- token-length distributions of cap-hit vs non-cap-hit samples (a gap before the cap =
  pure truncation; a continuous transition = genuinely-wrong answers mixed in)
- output-length distributions of the consensus (≥3/4) vs filtered-out subsets
- symbolic-verifiable intersection: n per subset (floor 100) and its instruction-length
  distribution vs the population (median 222, p90 463)

**4. Set the big-batch cap** at the curve's knee, subject to the dropped segment passing
the median-ratio ≥ 0.7 collapse check (section 2).

**5. Cost gate** — 141 samples/hr/card at cap 6144
(facts/distillation.json#distill.teacher_throughput_nvfp4); full-scale openo1 ≈ 92
card-days single-card, ≈ 12 days on 8 (unverified extrapolation). If the budget named
for the leg can't cover it, stop and report.

The amendment chain (runs/prereg.jsonl#distill_qwen27b_0908, amendments 1–5) carries
every criterion; this section is the index.
