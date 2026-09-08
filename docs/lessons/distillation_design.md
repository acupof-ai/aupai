---
question: How should the next version train by distillation from Qwen3.8-27B — which route, which seeds, what cost, what would falsify it?
status: open
source: facts/distillation.json (teacher vocab, seed inventory, measured 2026-09-08); teacher throughput pending tilerl-0a's measurement
---

# Distillation pipeline design (Qwen3.8-27B teacher)

4c's dispatch 2026-09-08. Deliverable: this design + a prereg row. Not code.

## 0. The vocab constraint, verified not accepted

Our vocab is 32,773 (facts/tokenizer.json#tok.vocab_size). The teacher's is **248,044 BPE**
(247,587 merges, Qwen2Tokenizer, read on the pod from
`/work/Qwen3.8-27B-NVFP4/tokenizer.json`, 2026-09-08). Ratio 7.57x. Two different
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
| openo1-math | two-stage: (a) symbolic/numeric re-derivation where the problem permits (arithmetic, determinate algebra); (b) consensus — keep generations where ≥3 of K=4 agree — for the rest. Consensus measures confidence, not correctness; the acceptance rate is labelled a consensus rate | independent of the original output |
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
only. No silent drop, no post-hoc threshold. Offline baseline (measured 2026-09-08, 500-row sample seed 42,
facts/distillation.json#distill.openo1_answer_extractability): 81.0% of openo1 rows
carry an `<Output>` tag; of those, 78.3% yield an extractable numeric answer (last
number), so the answer-extraction population is 63.4% of the file. 19.0% have no
`<Output>` tag. The symbolic path's true coverage —
problems whose answer can be re-derived independently — is smaller than the extraction
population and is a pilot measurement, not this baseline. Of the 19.0% no-tag rows, the
7.0pp that are code-like belong to the code domain (embedded-unittest execution), not
math; the remaining 12.0pp is discarded as unjudgeable — cheaper than rescuing.

## 3. Teacher generation cost

**PENDING.** The NVFP4 27B service's throughput (tok/s), concurrency ceiling, and
per-sample cost are tilerl-0a's measurements, asked 2026-09-08. The bf16 copy is broken
(MMLU 0.0%) and is not a fallback. Route feasibility is decided by "how many accepted
samples per service-day", not by the algorithm — this section gets the measured numbers
before the pilot, and the pilot's own first 1,000 generations re-check them.

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
