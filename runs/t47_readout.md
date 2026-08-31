# t47 readout — WSD two-stage join for the staged 15B→30B run

## Verdict
The staged (WSD) join works. Stage 1 ends at stable lr with no anneal; stage 2 resumes
under a different mix at the same stable lr, continues the absolute step, rebuilds the
data plan from the new mix, passes both stamps, and prints the join loudly.

## What train.py does today on --resume with a new mix/total_tokens
- step: continues absolute (step = resume_step from the ckpt).
- data plan: rebuilt from the new --mix (build_schedule reads mix_path; Xtr built before total_steps).
- total_steps: recomputed from the new mix's row count.
- lr schedule: already WSD-shaped (warmup → stable 1.0 → cosine warmdown in the last
  `warmdown` fraction), NOT cosine-to-total. Needed `warmdown`/`anneal_frac` as tunable
  floats (added: 81e8cf0/8199ee0), applied by is-not-None so `--warmdown 0` lands as 0.0.
- both stamps: _assert_mix_domains runs on the new mix (corpus_fp + env fingerprint).

## Rehearsal (0.2b → 0.3b, 50 + 50 steps, through harness launch)
Stage 1 (mix_scale_0.2b, --warmdown 0 --anneal_frac 0, 50 steps):
  lr flat at 1.00e-02 (mult 1.0) every step, all [main] — no anneal. ends val 5.658.
  ckpt_p47_s1.pt.step50 saved (959MB, opt+step).

Stage 2 (--resume step50, mix_scale_0.3b — a DIFFERENT stamped mix, --warmdown 0.10, to 100):
  WSD JOIN: resumed at step 50/100 under mix data/mix_scale_0.3b.json | lr_mult 1.0000 |
            warmdown 0.1 anneal_frac 0.1 | warmdown starts at step 90
  - step continues at 50 (not 0).
  - lr_mult 1.0 at the join == stage 1's last lr (1.00e-02): exact stable-lr handoff.
  - plan rebuilt from the new mix (corpus_fp lists all 7 domains of 0.3b).
  - both stamps pass (corpus_fp printed, env fingerprint OK).
  - lr stays 1.0 through step 80 [main]; warmdown only begins at step 90 (last 10%).
  - loss 5.716 → 4.454, val 4.717.

## The WSD recipe for the 15B→30B launch
- stage 1: mix_15b_stage1.json with anneal_frac=0, --warmdown 0 → warmup + stable, ends at stable lr.
- stage 2: --resume <15B ckpt> --mix mix_30b.json --warmdown 0.10 → join at stable lr,
  anneal only the last 10% at 30B. total_steps recomputes from the 30B mix.

## Changes landed
--warmdown/--anneal_frac tunable floats (is-not-None apply for the falsy-0 trap); a loud
WSD JOIN line at resume; both keys in _FROZEN_KEYS (recipe). test asserts --warmdown 0
lands as 0.0. artifact: runs/p47_s1.log, runs/p47_s2.log.
