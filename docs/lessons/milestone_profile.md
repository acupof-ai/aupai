---
question: "milestone profile: fixed-subset eval at every stage-1/stage-2 milestone, <60 min"
status: recorded
source: "t53 2026-08-31; runs/milestone_p324_v2.jsonl; eval/score_matrix.py --profile milestone"
cadence_corrected: "e1-10 2026-09-02: 532K tok/s and the 1.7-4 h cadence were p324 (200M) figures -- 76K/card x the 7-card block, never measured as an aggregate. 500M measures 94.9K, so 9.5-22.4 h and ~6 milestones. Wall times below are still p324's and are NOT rescaled."
---

# Milestone profile

Fixed subset that runs at every stage-1/stage-2 milestone. de's `harness
milestone` (t39) and b0's readout_30b.py consume the record.

Three premises this profile was sized against came from p324, a 200M checkpoint,
and none holds for the 500M run:

| sized against | measured for 500M |
|---|---|
| 532K tok/s = 76K/card x 7 cards | 11,857 tok/s/gpu x 8 = 94.9K total |
| milestone every 1.7-4 h | every 9.5-22.4 h, ~6 in the run, not 28 |
| runs on GPU 7, the lane card | no lane card: `lane_card: null`, `block_cards: 0-7` |

532K was never measured or logged as an aggregate: `git log --all -S "532000"`
returns nothing, and the string enters the history only as prose. It is
76,000 x 7 = 532,000 exactly, the per-card figure from `README.md:21` ("Measured
2026-08-31 on 7xH20 at the 0.2B point: 76K tok/s/GPU") times the 7-card block of
that era, with GPU 7 held out as the lane. Two independent confirmations of the
x7: both cadence endpoints fall out of the milestone grid at 532K
(3.24e9/532K = 1.69 h -> "1.7"; 8e9/532K = 4.18 h -> "4"), and `harness
milestone` hard-codes "3.24B ~= 3531 at 16x2x4096x7" -- the same 7.

Throughput from `runs/p500m_20b_0902.log:57-63` on the pod, steps 170-230, b32
accum1 seq4096 world=8 grad_ckpt fp8 d1024 L32, 2026-09-02. Cards from
`runs/card_assignment.json`, whose note states in-run milestone generation sweeps
cannot run because the scorer needs a whole 95 GiB card and not a gap.

The sparser cadence costs nothing here: it makes a <60 min budget easier, and
what shrinks is the readout's sample count. The missing lane card is what bites —
for the 20B run this is an end-of-run instrument, scored by `run_ddp.sh` after
torchrun exits with all eight cards free.

## Profile contents

`score_matrix --profile milestone` = 5 metrics:

| Metric | What it measures | Wall time on p324 |
|---|---|---|
| domain_loss | per-role loss on 7 domains × 262K tokens | 71.5s |
| mc_full | C-Eval + MMLU + ARC-Easy (multiple choice) | 122.6s |
| math_500 | greedy math on 500 held-out problems | 112.0s (CJK rep_stop) |
| code_500 | greedy code on 500 held-out problems | 722.5s |
| code_500_v2 | greedy code on 500 clean problems | 810.3s |
| **Total** | | **1838.9s = 30.6 min** |

Those wall times are p324's. At 2.4x the parameters with O(T²) generation (see
Known defects), 30.6 min at 500M is unmeasured — time the first 500M run before
quoting the budget again.

pass_at_k and math_hard stay out (hours); they run only at 15B and 30B.

## Decode improvements (2026-08-31)

Baseline math_500 took 702s with batch=64, no rep_stop. Two fixes:

1. **batch=256** (code_zh.py, math_zh.py): a 200M model on a 96 GB card decodes
   256 prompts at once. No speedup for math (O(T²) defect, below) but enables
   rep_stop to amortize.

2. **rep_stop** (train.py generate_batch): whitespace 8-gram repeated 3×,
   checked every 32 tokens. Stops degenerate outputs at ~100 tokens instead of
   512. Works for English/code (77.6%/85.2% degenerate rates on code).

3. **CJK rep_stop** (train.py + score_matrix.py): character 12-gram (same ≥3×
   rule) for CJK-majority text (>30% CJK chars). Chinese text has no whitespace
   word boundaries, so the whitespace 8-gram never fired there. CJK rep_stop
   reduced math_500 wall time from 733.9s to 112.0s (85% faster; avg gen tokens
   512→77). Reports both `rate` (whitespace) and `cjk_rate` (character).

## Known defects

- **O(T²) prefix recomputation** (train.py generate_batch): no KDA/MLA state
  carried across token steps; each step recomputes the full prefix. Generation
  is O(T²) per sequence. Batch size does not help — total work is the same.
  Fix is tilerl's, not tonight's.
- **Sandbox timeout** (datagen/sandbox_exec.py, fixed 0557844): subprocess.run
  timeout killed only the unshare parent; the grandchild `python3 -I code.py`
  survived, holding stdout pipes open and blocking communicate() indefinitely.
  Fixed: Popen + start_new_session=True + os.killpg on timeout. Self-check
  verifies no survivors.

## Reproduction

```bash
# Run the milestone profile on one free card. During the 20B run there is no
# free card (block_cards 0-7); this is an end-of-run invocation.
CUDA_VISIBLE_DEVICES=7 python3 eval/score_matrix.py --ckpt ckpt_p324.pt --profile milestone

# Record lands in runs/milestone_<ckpt>.jsonl
```
