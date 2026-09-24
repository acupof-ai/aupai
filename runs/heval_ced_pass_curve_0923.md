# Other-architecture counts on the early CED checkpoints (v41_ced_0923)

Recorded 2026-09-23 by 66. Kept because these are counts in the single digits and one
checkpoint's number cannot be read against another's unless the TASK SET matches -- the
first-60 figure is over the same 60 file positions in
`data/eval/humaneval/humaneval_164.jsonl`, taken in the data file's own order
(`eval/shard.py:select` enumerates the same order, so a sharded and a single-process run
compare directly).

| step | arm | tasks | pass | source |
|---|---|---|---|---|
| 2000 | rstrip, single process | first 60 | **5/60** | pid 521265 (still running) |
| 2000 | rstrip, single process | all 164 | **5/164** | pid 521265, closed later the same day |
| 4000 | rstrip, 8 CPU shards | first 60 | **4/60** | shards started 10:32:07Z, merged by eval/e0_merge_score.py |
| 4000 | rstrip, 8 CPU shards | all 164 | **4/164 = 2.44%** (CLEAN 4/156 = 2.56%) | same |

Both numbers are single-digit counts: a one-task difference is not a trend. The gate is
HumanEval pass@1 >= 30% at the end of the 30.0B schedule, so at step 4000 (5% of 38,146)
neither reading is informative about the outcome.

Method check: the first-60 mapping was validated by reproducing 5/60 on step2000 before
quoting the step4000 figure from it.

Later points are produced by `runs/heval_auto_loop.sh` (MIN_STEP=6000), which writes
`runs/heval_merged_step<N>_result.json` for each.

## Wall clock and cost -- what the instrument can and cannot resolve

Wall clock for the step4000 read: shards launched **10:32:07Z**, last preds file **11:41**
(that is the slowest shard; the merge result is 11:43). So ~69 min from launch, of which the
slowest shard is ~49 min -- the earlier "41 min" quoted a narrower span without saying which.

The co-resident cost was first reported as +1.11% median s/step from a 20-interval window on
each side of the launch. **That figure is withdrawn: it is inside the noise of this
instrument.** Adjacent 100-step bins of the same log, with NO eval running, spread 3.28% peak
to peak and move by up to 2.81% bin to bin. Re-measured over windows wide enough to average
that out (base 5000-6270 n=128, during 6280-7410 n=114, after 7420-8200 n=79), the medians
are 3.6604 / 3.6638 / 3.6702 s/step: **+0.09% during, +0.27% after**. The honest statement is
that the eight CPU shards cost the training run less than this log can resolve, not that they
cost 1.11%.

## Loop cap (1e's question, 2026-09-23)

The step scan orders by the NUMBER (`sed 's/.*\.step//' | sort -n`), not by the path string:
`sort -t. -k3 -n` looks numeric but runs the key to end-of-line, and measured, it puts
step10000/12000 before step6000/8000.

`MAX_ITER` counts POLLS, not evaluations: a round with no complete checkpoint increments it
too. With a 600 s poll and ~2000 steps between saves at ~3.7 s/step, one checkpoint cycle
costs ~13 iterations (12 polls + the eval round). At step 7800 of 38,146 with 15 checkpoints
left, the remaining run needs ~202 iterations, so the first cap of 200 would have expired in
the last hours of training. It is now 800, and reaching it prints an explicit
`=== STOPPED: iteration cap` line rather than stopping silently.

Completion criteria for a checkpoint, both required: (a) the training log has printed the
step line for step+10 (save_checkpoint is synchronous -- it is the `save_checkpoint(ckpt_path
+ f".step{step}", ...)` call in the training loop, above the `runlog(` that emits the step
line, so that line proves the save returned); (b) the
file size has not changed for 120 s. A fixed size does not work. Measured at the time:
step2000/4000 were 13021167860 B while step6000 was 13021167924 B (the save is not
byte-stable); 2000 and 4000 have since been rolled off the pod, so that half of the comparison
is no longer re-testable and is reported as measured-at-the-time.

## Known defect in the first version of the loop (found by 3b on the live pod)

`echo "$step" >> "$DONE"` ran unconditionally: the script has no `set -e` and the merge's
status was masked by a `| tail`. `e0_merge_score.merge()` raises SystemExit on a shard gap,
a duplicate `(task_id, sample_idx)` or a short sample count, all before writing anything --
so on 2026-09-23 step 12000 was recorded done while only **157/164** tasks merged (shard 3
timed out at `eval/humaneval_gen.py:69`, its 7 missing ids all == 3 mod 8) and no result file
existed. Because the DONE grep was the only de-duplication, that step was never retried.

Fixed: a step counts as done only when the DONE line AND `heval_merged_step<N>_result.json`
both exist (`done_and_produced`, used by the skip test and by the post-merge bookkeeping), a
failed merge prints `MERGE-FAILED` and is retried up to 3 times, and a step that exhausts the
retries is written to `heval_auto.giveup` and skipped in the log rather than silently
defaulting to done.
