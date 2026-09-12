# v41_gate_0911 run-end sequence

Ordered steps for the hour the 30B gate run ends (~2026-09-13 10Z). Every command is run on
the pod from `/work/aupai` unless stated. Card allocation stays the controller's: ask fb for
the free card before steps 5 and 7 (two different cards; SFT starts while the panel is read).
Nothing here is launched ahead of the run ending — steps 0 and every REFUSING gate enforce
that. Dry-parsed cardless on the pod 2026-09-12: `exp.py done -h`, `pod_backup.sh --dry-run`
(142 paths), `score_matrix.py --help`, `harness.py launch --help`.

Tokens: the run does 30.0B at 786,432 tokens/step = 38,146 steps. Final architecture name
per facts/v41.json (`3.2b-a352m-e48`: 3.22B total / 352.3M active, 48 experts). Final
checkpoint convention `ckpt_{arch}_{tokens}_{date}.pt`.

## 0. Confirm the run actually ended and the cards are free

```bash
cd /work/aupai
python3 scripts/card_claim.py status
ls -la ckpt_v41_gate_0911.pt
```

Require: no live `v41_gate_0911` claim or torchrun process, every card 0 MiB in
`nvidia-smi`, and the final `ckpt_v41_gate_0911.pt` present (not only `.stepN`). If the
process was killed without a final unsuffixed save, the newest `.step<N>` is the final
checkpoint — use its N in step 2 and note it in the exp result.

## 1. Read the standard HumanEval column and val, then close the exp row

The gate number is the STANDARD (continuation) arm, the metric of
runs/prereg.jsonl#v41_gate_0911. Greedy 164 on the final ckpt, one card:

```bash
CARD=<controller-granted> CUDA_VISIBLE_DEVICES=<card> \
python3 eval/humaneval_gen.py --ckpt ckpt_v41_gate_0911.pt \
  --run v41gate_final_cpu --force 2>&1 | tee runs/he_standard_final.log
```

Read pass@1 from `HUMANEVAL pass@1 (greedy) = N/164`. The preds file is versioned by
`--run` at `data/eval/preds_humaneval_ckpt_v41_gate_0911.pt.v41gate_final_cpu.jsonl` (read
the exact path from the script's `preds saved:` line). Take final val from the last training
log line `step 38146/38146 val X` (grep `runs/v41_gate_0911*.log`). Then:

```bash
python3 scripts/exp.py done --name v41_gate_0911 --status ok \
  --result "HumanEval standard greedy pass@1 = N/164 (preds <preds saved path>); final val X" \
  --finding "gate >=30% met / not met at 30B under the standard arm" \
  --decision "post-gate SFT per runs/prereg.jsonl#v41_sft_0913 regardless; >=30% is the pretrain gate" \
  --reading_artifact data/eval/preds_humaneval_ckpt_v41_gate_0911.pt.v41gate_final_cpu.jsonl
```

Use `--started "2026-09-12 02:07"` if exp complains about multiple open rows.

## 2. Hard-link the final checkpoint under the naming convention (no move)

```bash
ln ckpt_v41_gate_0911.pt ckpt_3.2b-a352m-e48_30b_20260913.pt
ls -li ckpt_v41_gate_0911.pt ckpt_3.2b-a352m-e48_30b_30b_20260913.pt
sha256sum ckpt_v41_gate_0911.pt ckpt_3.2b-a352m-e48_30b_20260913.pt
```

The two names must share one inode and one sha256. Hard-link, never `mv`: train.py and the
roller key on `ckpt_v41_gate_0911.pt`, and a move breaks resume/audit paths. Date is the
UTC run-end date; if the run ends on 09-12 use `20260912`.

## 3. Back up to durable storage

```bash
bash scripts/pod_backup.sh
```

Backs up final/milestone `ckpt_*.pt` (never `.stepN` rollers), `runs/*.jsonl`, `facts/`,
tokenizer and mix json to `/mnt/data02/aupai_backup` and writes a sha256 MANIFEST. Refuses
unless the destination is a separate mounted filesystem. Confirm the MANIFEST lists
`ckpt_3.2b-a352m-e48_30b_20260913.pt`.

## 4. Full base score_matrix panel on one card

The full APPLIES['base'] panel (10 metrics), no `--metrics` pin, so domain_loss and
mc_ceval are scored on GPU in the same record as the rest (the he6k CPU matrix ran those two
cardless and the other seven were CUDA-refused; the final record is one consistent GPU run).
`harness launch` sets CUDA_VISIBLE_DEVICES from `--cards` itself, so do not export it here:

```bash
python3 scripts/harness.py launch sm_gate_final --cards <card> \
  --output runs/score_matrix.jsonl -- \
  python3 eval/score_matrix.py --ckpt ckpt_3.2b-a352m-e48_30b_20260913.pt \
  --mix data/mix_v41_gate.json --ngpu 1 --json runs/score_matrix.jsonl
```

Minutes are an ESTIMATE, from g/i smoke idle-card `_wall_s` (smoke shape, dedicated card):
domain_loss ~1 min, minimal_pairs/mc_ceval/lambada_zh/math_v2_like ~0.6-0.8 each,
domain_bpb ~1.3, humaneval_bpb ~0.7, lambada_en ~2.4, l1_fewshot ~13-17 (dominates).
Total ~25-30 min plus checkpoint load. `api_cloze` is expected to record an error (its
35.1 GB token cache is unbuilt); that is a known skip, not a panel failure.

## 5. Launch the post-gate SFT on a second card

Only after the panel card is assigned and the SFT card is confirmed free (a different card).
The launcher runs its own cardless pack gate and live-claim gate, then trains and reads its
own ChatML by-name HumanEval (runs/prereg.jsonl#v41_sft_0913):

```bash
HYPOTHESIS="ChatML code-instruction SFT moves HumanEval by-name pass@1 above 5/164 from an unseen-prefix 0/164 base" \
RESUME=ckpt_3.2b-a352m-e48_30b_20260913.pt CUDA_VISIBLE_DEVICES=<card2> CARD=<card2> \
setsid nohup bash runs/v41_sft_0913.sh > runs/v41_sft_0913.launch.log 2>&1 </dev/null &
```

Detached via setsid; poll `runs/v41_sft_0913.launch.log`. It refuses if the gate still
holds the card or the pack vocab_id mismatches. Acceptance: by-name pass@1 >=5/164 vs the
paired base `--chatml` control; <5/164 drops the recipe.

## 6. Notify (fb sends)

fb messages `agent-infer-be` and `tilerl-a3` that the gate run has ended and all eight cards
are free. 3b does not send these.
