---
question: What must `launch_30b.sh --dry` prove before READY means the run can start and will train on what the recipe names?
status: proposed
source: b0 audit 2026-08-31; each guard carries a failing case reproduced against the live repo and pod
---

# READY means the run can start (b0 audit)

## The class

Today's five near-misses share one shape: **an artifact was declared ready by a check
that did not test the property the consumer needs.**

| artifact | what the check tested | what the consumer needed |
|---|---|---|
| 197-shard token cache | its own build stamp | the corpus the trainer reads |
| hand-stamped `UNKNOWN` fingerprint | the stamp file exists | the stamp matches the bytes |
| register write refusal | am I in a linked worktree | can I write here safely |

The last one had the sign backwards. The worktree was the safe place to write, and the
tree it redirected to was the one where sessions overwrite each other.

The launch gate has the same shape. `launch_30b.sh --dry` computes the mix contract,
prints it, and then derives READY from `_blocked` alone. The contract result is displayed
and discarded.

Four guards close the class at the gate, plus one that closes the gate's own defect. Each
is stated with the failing case that is red before the change and green after.

## G0 — READY must consume the contract it prints

The gate already computes the right answer and throws it away.

```
$ bash launch_30b.sh --dry --stage 1          # against a mix with weights summing to 1.69013
== launch_30b stage 1 readiness ==
  [FAIL] mix_contract weights(domains)+weights(_blocked) = 1.69013, not 1.0
READY: all domains stamped, none blocked.
EXIT=0
```

`[FAIL]` and `READY` on adjacent lines, exit 0. Any contract violation passes: a dropped
domain, a silent reweight, a landed domain with no stamp file at all.

A second defect sits behind it. `check_mix_30b_contract` tests stamp presence with
`os.path.exists`, while `train.py` reads the stamp and compares it to the live bytes.
They disagree on the exact input that cost us the morning:

| input | harness contract | train.py `_assert_mix_domains` |
|---|---|---|
| stamp file with `{"fingerprint": "UNKNOWN"}` | PASS | refuses at startup |

The gate is weaker than the runtime it gates, so the gate cannot be the last word.

**Failing case.** A mix whose `_blocked` is empty and whose weights sum to 1.69013 must
exit non-zero. A domain stamped `UNKNOWN` must exit non-zero.

**Fix.** Fold the contract state into READY. In `check_mix_30b_contract`, replace the
existence test with a read requiring 16 lowercase hex.

## G1 — the token cache belongs to this corpus

Measured on the pod, all seven stage-1 domains agree today:

| domain | shards | fingerprint (cache `.srcfp` = live = stamped) |
|---|---|---|
| en_c4 | 166 | 225e0de8caced5f4 |
| cot | 13 | 388496b76ed9bf88 |
| math_owm | 127 | 580e04daf8376488 |
| zh_web | 909 | a0d44fc44a289d60 |
| textbook_30b | 79 | 3f237c5191cb8571 |
| wiki_chat | 12 | b864d32f9452a7c8 |
| code_rp1t | 235 | d8b9b18ba080f487 |

The keying is stronger than the 197-vs-236 story assumed. `train.py:1366-1372` gates
reuse on a conjunction: cache exists, shards exist, `.vocab` matches `VOCAB_ID`,
`.srcfp` matches the live corpus fingerprint, and cache mtime is not older than the
newest shard. A shard-count change moves the corpus fingerprint, so that scenario is
already caught.

Two holes remain, both verified.

**Tokenizer behavior is not in the key.** `vocab_fingerprint` (train.py:1195-1204)
hashes the id-to-token map and nothing else. On the pod, against the real
`data/tokenizer.json`, three edits each keep `vfp = 0bce3584bc24f255` while changing the
token stream:

| edit | vocab identical | VOCAB_ID | encoding |
|---|---|---|---|
| `add_prefix_space` False to True | yes | collides | differs |
| normalizer null to NFKC | yes | collides | differs |
| move one merge to the front | yes | collides | `reading` [2335,330] to [16307,15278] |

`same_vocab` stays true, `same_source` stays true, mtime is untouched, so `fresh` is
true and every `/data00/tokens_*.pt` is reused against a tokenizer that segments
differently. The repo already found one axis of this bug and fixed it by putting
`--fone` in the cache **name** (train.py:1341-1344), for the stated reason that it
"changes the token stream while leaving the vocabulary fingerprint identical." This is
the same defect on three more axes.

**The corpus fingerprint reads 128KB per shard.** `_corpus_fp` hashes name, size, and
sha256 of the first and last 64KB. An equal-length edit in a shard's interior leaves it
unchanged. This is a deliberate cost trade, documented at `corpus_fingerprint.py:13`, and
the right one at 108GB. It belongs in this document so the gate states what its
fingerprint covers, rather than leaving a reader to assume content equality.

**Failing case.** Copy `data/tokenizer.json`, move one entry in `merges` to the front,
leave the vocab untouched. The gate must go red. Today `VOCAB_ID` is unchanged and the
cache is reused.

**Fix.** Write a `cache + ".tokfp"` sidecar holding `sha256(TOK_PATH bytes)[:16]` and add
`same_tok` to the conjunction. A sidecar rather than a change to `vocab_fingerprint`,
because `vocab_id` also means something in checkpoints and should keep meaning it.

The gate's own job is smaller: **compare the three values and print them.** The
comparison exists in `train.py`, inside the training process, after torchrun has brought
up seven ranks. The property is true today by measurement and untested at the gate.

## G2 — pod sync covers what the training run reads

The pre-launch drift gate runs with `--scope training`. `scripts/launch_30b.sh`,
`scripts/harness.py`, and `data/mix_*.json` all classify as `docs`, so drift in the
launcher, the harness, or a mix weight does not refuse the launch at pre-flight.

This is a gap, not a hole. `train.py:1783` calls `pod_drift.check_pod(ROOT)` unscoped,
covering all 177 manifest entries, and raises at startup. The cost of the gap is
therefore wasted time, not wrong data: the operator learns at first step rather than at
the gate.

One item is outside both. `data/tokenizer.json` is listed in `pod_drift.SCOPE` but ships
zero manifest entries, because `scoped_paths()` enumerates `git ls-files` and the file is
gitignored (.gitignore:5). The single most identity-bearing file in the run is named by
the drift gate and covered by neither tier.

**Failing case.** Append a byte to the pod's `data/tokenizer.json`. `--dry` stays green,
and `grep -c data/tokenizer.json data/pod_head_manifest.txt` returns 0.

**Fix.** Add `scripts/launch_30b.sh` to `_ENTRY_POINTS["training"]`; the BFS follows its
citation of harness. Cover `data/tokenizer.json` by content hash rather than by
`git ls-files`.

## G3 — the environment is the one that was verified

No check compares the current environment to a previously verified one before a launch.
`check_env_fp_present` verifies that checkpoints **carry** an `env_fp`, which is a
different property. `env_fp` is compared only on `--resume` (train.py:1838-1848).

The 2026-08-30 restart dropped the writable layer with `liger_kernel`, `fla`, `flask`,
`opencc`, and `trackio`. `check_env_importable` catches exactly this and does not run on
the launch path.

The same restart wipes `/data00`, where 163 GiB of token caches live. A restart between
the dry run and the launch leaves the gate's verdict unchanged and the caches gone, which
costs about 15 minutes of regeneration at 8 workers. Correctness is unaffected. The gate
should still print it, so the operator learns it at the gate.

**Failing case.** Uninstall one required package on the pod. `--dry` stays green.

**Fix.** Call `check_env_importable` from the gate and compare `env_fingerprint()` to the
recorded value.

## G4 — the resolved flags are the effective values

Nothing compares the `FLAGS` string literal to the recipe documents. One of its own flags
was already being dropped.

`train.py:1733-1735` applies parsed flags with `if hasattr(Cfg, k) and v`. Zero is falsy.
Auditing all thirteen stage-1 flags against the `Cfg` defaults:

| flag | commanded | Cfg default | effective |
|---|---|---|---|
| seed | 0 | 42 | **42, dropped** |
| warmdown | 0 | 0.65 | 0.0, rescued explicitly |
| anneal_frac | 0 | 0.1 | 0.0, rescued explicitly |
| attn_res_blocks | 0 | 0 | 0, dropped, identical by luck |

The rescue loop at train.py:1739 covers `warmdown` and `anneal_frac` by name, and `seed`
was not on the list. `ckpt_pretrain_15b_s1.pt.step500` records `seed 42`, so the running
job's value is observed rather than inferred.

`attn_res_blocks` is the same bug hidden by a matching default. Change that default and
the recipe silently stops applying.

Two recipe values cannot be expressed in `FLAGS` at all. The card count comes from
`data/mix_scale_run_config.json` at launch time, and the gate prints neither it nor
`NGPU`. Editing the card list to dodge a busy card changes `world` with byte-identical
`--dry` output.

**Failing case.** `--seed 0` must leave `Cfg.seed == 0`. `--attn_res_blocks 0` must land
by intent rather than by coincidence.

**Fix.** Replace the truthiness test with `is not None` against the parser's own
defaults, which removes the falsy class and makes the rescue loop unnecessary. Print the
effective values from the same dict `save_checkpoint` serialises, so the banner and the
checkpoint cannot disagree.

Timing is load-bearing. A mid-run auto-resume picks up whatever `train.py` says at that
moment, so patching during stage 1 would apply seed 0 and rebuild the data plan on a
different shuffle. The fix lands after stage 1 ends.

## Scope note

Five corpus checks — `check_mix_not_unfiltered`, `check_mix_shards`,
`check_score_input_fresh`, `check_mix_supply`, `check_corpus_fp` — hardcode the domain
list to `Cfg.mix = data/mix_scale_3.24b.json`, which shares zero domains with
`mix_15b_stage1.json`. Their PASS lines describe the retired ladder corpus.
`_assert_mix_domains` runs on the actual `--mix` file, so the launch itself stays
guarded. The green board an operator reads is about a different corpus.

Parameterizing those checks onto the launch mix requires `PROVENANCE.md` blocks for the
seven stage-1 domains first. Without them, `check_corpus_fp` redirected at the stage-1
mix fails 0/7 on the provenance clause alone. Two changes, in that order.

## Delivery

Each guard is a check plus its failing case. The gate consumes every one and exits
non-zero when any is red, so the dry run becomes the only thing a person has to read.

| guard | change | owner |
|---|---|---|
| G0 | READY folds in the contract state; the stamp test reads 16 hex | de |
| G1 | `.tokfp` sidecar in the freshness conjunction; gate prints the three fingerprints | de |
| G2 | `launch_30b.sh` into training entry points; tokenizer covered by content hash | de |
| G3 | `check_env_importable` and the env fingerprint on the launch path | de |
| G4 | `is not None` apply; banner from the `save_checkpoint` dict; after stage 1 | de |

The scope-note work is separate and ordered: `PROVENANCE.md` blocks for the seven
stage-1 domains, then parameterize the five checks onto the launch mix.
