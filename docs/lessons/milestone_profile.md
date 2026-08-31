---
question: "milestone profile: fixed-subset eval at every stage-1/stage-2 milestone, <60 min on lane card"
status: recorded
source: "t53 2026-08-31; runs/milestone_p324_v2.jsonl; eval/score_matrix.py --profile milestone"
---

# Milestone profile

Fixed subset that runs at every stage-1/stage-2 milestone on GPU 7 (lane card).
Milestones arrive every 1.7-4 h at 532K tok/s; the profile must finish in <60 min
so the readout does not fall behind. de's `harness milestone` (t39) and b0's
readout_30b.py consume the record.

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
# Run the milestone profile on GPU 7 (lane card)
CUDA_VISIBLE_DEVICES=7 python3 eval/score_matrix.py --ckpt ckpt_p324.pt --profile milestone

# Record lands in runs/milestone_<ckpt>.jsonl
```
