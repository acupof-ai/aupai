---
question: How should the token cache store only what a mix draws, without silently changing what is sampled?
status: recorded
source: tilerl-2/de-2 design; measurements in facts/efficiency.json#eff.cache_load_gates_startup
---

# Token cache stores demand, not supply (de-2)

## The waste

`train.py` loads every domain's full token cache on every rank before the first
step. Stage 1 measured 149 GiB and 386 s of startup before step 1. The caches
hold each domain's whole supply, but a mix draws a fraction of it:

| domain | cached | drawn (stage 1) | epochs | bytes read but never sampled |
|---|---|---|---|---|
| zh_web | 79.3 GiB / 21.29B tok | 1.65B tok | 0.08 | ~97% |
| code_rp1t | 28.2 GiB / 7.569B tok | (full) | ~1 | small |

The read happens per rank at every launch and every resume.

## The cap, and the constraint that shapes it

Cap at tokenize time to `demand x (1 + margin)`, after the existing
`random.Random(Cfg.seed).shuffle(texts)` at `train.py:1382`.

After the shuffle, not before. A prefix cap takes the first N shards, so shard
numbering — an artifact of how the corpus was fetched — would decide what stage 1
trains on. zh_web at 0.08 epochs makes this the difference between a sample and
an arbitrary subset (tilerl-78).

## What the cache is a function of

A capped cache is derived from four inputs, and the failure mode this repo keeps
hitting is a derived artifact that stays valid after its source changes:

| input | today | why it must be in the stamp |
|---|---|---|
| source bytes | `.srcfp` | already covered |
| vocabulary | `vocab_id` | already covered |
| sample seed | nothing | decides *which* rows survive the cap |
| demand x margin | nothing | decides *how many*; a larger demand needs a rebuild, a smaller one may reuse |

A cache whose stamp disagrees with the live values rebuilds. Precedent: the 0.2b
run swapped its source, the cache was rebuilt against the new source, and
training kept reusing the old one because nothing compared them.

## The seed question: a separate `sample_seed`, defaulting to `Cfg.seed`

tilerl-78 raised the cost: capping after the shuffle makes the cache a function
of `Cfg.seed`, so a seed sweep pays a full retokenize per arm — 19 min for
zh_web. Take the separate field, for a reason stronger than the cost.

`Cfg.seed` drives four different things: the data shuffle (1382), the val
permutation (1620), the batch generator (1561), and **weight init** (1746). The
seed-variance arms (`p02_s0..s3`, `ds.seed_variance_0p2b` = 0.0516 nat) exist to
measure *initialisation and ordering* noise at a fixed corpus. Binding the cache
to `Cfg.seed` would silently change the training data of every such arm — the
measurement would then mix data variance into a number the repo reads as init
variance, and every threshold derived from it (`readable_move_nat` 0.24,
`per_role_domain_loss` 0.1176 in the readout pre-registration) would be measuring
something other than what it claims.

So: `Cfg.sample_seed`, defaulting to `Cfg.seed`. A seed sweep sets `--seed` and
leaves `sample_seed` pinned; the arms then share one cache and differ only in
what they are meant to differ in. Changing the sample is possible but must be
deliberate.

This is the reverse of the `--fone` precedent at `_domain_cache_path`, and
deliberately so: `--fone` changes the token stream, so it belongs in the cache
*name* and the two caches coexist. `sample_seed` and `demand` select rows from
one stream; they belong in the stamp, where a mismatch forces one rebuild rather
than accumulating a cache per value.

## Correctness gate

The cap is wrong if it changes what training sees beyond sampling noise. The
falsifying measurement, not a claim of care:

- a short run on a capped cache and on the uncapped one, same `sample_seed`,
  must agree on per-domain loss within the measured seed noise (0.0516 nat,
  `ds.seed_variance_0p2b`); a divergence larger than that means the cap is not
  drawing a uniform sample
- changing `sample_seed` or raising `demand` rebuilds; lowering `demand` reuses
- startup bytes drop by the ratio the stamp predicts, measured, not assumed

## Margin

`margin` covers epoch cap and schedule slack. It is a stamp field, not a
constant: at 0.08 epochs almost any margin is a large absolute saving, and at
~1 epoch the cap is inert. Start at 0.25 and record what the mix actually drew.

## Composition with tilerl-1

tilerl-1 (load only the drawn rows / mmap) attacks the same bytes at load time;
this attacks them at tokenize time. They compose — a capped cache is smaller to
mmap — and neither depends on the other. Sequencing matters only in that this one
changes what is on disk, so tilerl-1 should be measured against a capped cache
once both land.
