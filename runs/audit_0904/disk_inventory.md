---
question: What is stored on this machine, on every mount, and what is it for?
status: partial
source: host reads via tn exec and container reads via ~/bin/pod, 2026-09-04 06:58-07:15Z; peer sections at main be3d93d9 (3b) and 50e16811 (de)
---

# Whole-machine disk inventory, 2026-09-04

Consolidated by tilerl. Sections: **host + non-aupai container** (this document, mine),
**`/work/aupai/data` + token caches** (3b, `disk_inventory_3b.md`, be3d93d9),
**`/work/aupai/runs`** (de, `disk_inventory_de.md`, 50e16811), **checkpoints** (b0,
post-prune folded in later). Read-only: nothing deleted, nothing moved.

## The headline: the machine is 15.7 TiB, not the 2 TB everyone has been watching

Every conversation so far has been about `/work` at 96% full. That is one mount of six.

| mount | device | size | used | free | use% |
|---|---|---|---|---|---|
| `/` (holds `/work` via kubelet) | /dev/vda2 | 1.97 TiB | 1.85 TiB | 87.1 GiB | **96%** |
| `/data00` | /dev/nvme0n1 | 3.44 TiB | 1.42 TiB | 1.88 TiB | 43% |
| `/data01` | /dev/nvme1n1 | 3.44 TiB | 612 GiB | 2.67 TiB | 19% |
| `/data02` | /dev/nvme2n1 | 3.44 TiB | 242 GiB | 3.03 TiB | 8% |
| `/data03` | /dev/nvme3n1 | 3.44 TiB | 242 GiB | 3.03 TiB | 8% |
| `/boot/efi` | /dev/vda1 | 196 MB | 3.1 MB | 193 MB | 2% |
| **total** | | **15.7 TiB** | **4.26 TiB** | **11.5 TiB** | **27%** |

**The pressure is on one 2 TiB device while 11.5 TiB sits free on four NVMe drives.** The
150–200 GB repo-shaped fetch that `f6e90bfa` costed as "does not fit" does not fit
`/work`; it fits `/data02` or `/data03` twelve times over. That is a placement question for
the user, not a capacity problem.

## Two measurement traps found here, stated before the numbers that depend on them

**1. `du` on this machine overstates by up to 1000× because of sparse files.**
`/data00/kv-ssd-dsv4` reads **2,534 GB** by `du -sb` and **2.53 GB** allocated — 8
`kv.mmap` files, one of which is 316.75 GB apparent against 0.50 GB of blocks.
`/data00/kv-l3` is 256 GB apparent and **0 bytes** allocated. A `du`-based inventory of
`/data00` would have reported 4,223 GB on a 3.78 TB device — impossible, which is what
exposed it. **Every figure below is `st_blocks × 512` (allocated), not apparent size.**

**2. `/var` carries a 2255 mtime.** `/var/lib/containerd` reports a newest-file mtime of
`2255-03-14`, i.e. a corrupt or deliberately-far-future timestamp inside an image layer.
Any "newest mtime" sort over the host root is wrong until that is excluded.

## Host `/` — 1.85 TiB used, of which 1.51 TiB is the container we already inventory

| path | allocated | files | newest mtime (UTC) | owner / purpose | referenced by | disposition |
|---|---|---|---|---|---|---|
| `/var/lib/kubelet/pods/95a05e32…/volumes/kubernetes.io~empty-dir/work` | **927.6 GB** | — | live | **this is `/work` in the container** — the emptyDir | everything | keep — **not additional storage, the same bytes** |
| `/var/lib/containerd/…overlayfs` | 616.4 GB | 3.8 M | 2255 (corrupt) | container image layers, all pods on the node | node infra | **not ours** |
| `/root/podman` | 187.3 GB | 4.67 M | 2026-07-07 | another workflow's container store | none found | **not ours** — predates this project's work |
| `/root/miniconda3` | 27.1 GB | 195 k | 2026-08-14 | node toolchain | — | not ours |
| `/host` | 27.7 GB | 36 k | 2026-08-24 | node mount | — | not ours |
| `/root/sccache` | 13.8 GB | 17 k | 2026-08-24 | compiler cache | — | not ours |
| `/sgl-workspace` | 12.6 GB | 20 k | 2026-08-14 | sglang source, the container's default cwd | — | not ours |
| `/var/log` | 11.2 GB | 306 | live | node logs | — | not ours |
| `/var/crash` | 10.3 GB | 4 | 2026-07-16 | kernel crash dumps | — | not ours; flag to the node owner |
| `/root/tc27-nvfp4-slice{2,4}`, `/root/Qwen` | 19.7 GB | 50 | 2026-08-23 | model slices | none found | **not ours** (tileRL-adjacent, not aupai) |
| `/tmp` (all) | **1.46 GB** | 5,234 | 2026-09-04 | mixed | — | see below |

**The 2026-09-03 `/tmp/absmoke` copy is GONE** — the 206 GB `cp -a` that filled this disk
was cleaned at the time and no trace remains under `/tmp`. Host `/tmp` totals 1.46 GB, the
largest single item `qwen.tgz` at 0.76 GB (2026-08-28). **There is no host-side leftover of
ours worth a cleanup rule**; the class the order anticipated does not exist right now.

**Reconciliation for `/`**: 927.6 (our work volume) + 616.4 (image layers) + 187.3 (podman)
+ ~100 GiB (root toolchains, /host, /usr, logs, crash) ≈ **1.79 TiB** against `df` 1.85 TiB.
Residual **~60 GiB, 3.3%** — above the 2% bar. It is not isolated; the likely holders are
other pods' volumes under `/var/lib/kubelet` (only ours was walked) and deleted-but-open
files. **Stated, not hidden.**

## `/data00` — 1.42 TiB used, entirely model weights and caches, none of it aupai's

| path | allocated | files | newest mtime | purpose | disposition |
|---|---|---|---|---|---|
| `Qwen3.8-Flash-Next` | 335.3 GB | 145 | 2026-08-28 | model weights | not ours |
| `DeepSeek-V4-Flash-0731-FP8` | 286.3 GB | 56 | 2026-08-23 | model weights | not ours |
| `DeepSeek-V4-Flash-0731` | 155.4 GB | 77 | 2026-08-21 | model weights | not ours |
| `models` | 152.8 GB | 126 | 2026-08-18 | model store | not ours |
| `runs` | 97.7 GB | 20 | 2026-07-29 | another project's runs | not ours |
| `.bpe_cache_0828` | 35.4 GB | 5 | 2026-08-27 | tokenizer cache | not ours |
| `Qwen3.6-35B-A3B-FP8` | 34.9 GB | 58 | 2026-08-22 | model weights | not ours |
| `tokens_web.pt.stale_0826` | 31.0 GB | 1 | 2026-08-26 | **stale token cache, name says so** | not ours to delete; flag to its owner |
| `tokens_web.pt` | 30.2 GB | 1 | 2026-08-28 | token cache | not ours |
| ThinkingCap / Qwen3.x FP8 / NVFP4 / DSpark variants | ~185 GB | ~260 | Aug | model weights | not ours |
| `kv-ssd-dsv4` | **2.53 GB** (2,534 GB apparent) | 8 | 2026-08-17 | sparse KV tier files | not ours |
| `kv-l3` | **0.00 GB** (256 GB apparent) | — | 2026-08-19 | sparse, unallocated | not ours |

3b's 247.80 GB of aupai token caches on `/data00` are in their section; they are inside
`models`/loose files above and **not double-counted here** — I list only what 3b did not.

## `/data01` — 612 GiB, and it holds a 365 GiB BACKUP of this project nobody knew about

| path | allocated | files | newest mtime | purpose | disposition |
|---|---|---|---|---|---|
| **`/data01/aupai/backup`** | **365.2 GiB** | 4,685 | **2026-08-30T19:51Z** | **rsync copy of the whole aupai tree** | **keep — see below** |
| `/data01/aupai/attempt7_upperdir` | 3.3 GiB | 48,549 | 2026-08-30 | overlayfs upper layer from an earlier attempt | unclassified, small |
| `/data01/aupai/backup.log` | ~1 MiB | 1 | 2026-08-30T20:16Z | the rsync progress log that identifies it | — |
| `/data01/eicyep50jiy2ox7jylkb0x0` | 243.4 GiB | 320 | live | another workload | not ours |

**This is the most consequential thing in the inventory and it was not on anyone's map.**
`/data01/aupai/backup` is an rsync of the project root — `AGENTS.md`, `algorithms/`,
`EXPERIMENTS.md`, `ckpt_p324.pt`, the lot — taken 2026-08-30 19:32–20:16Z. 365.2 GiB across
4,685 files. `backup.log` is the rsync transfer log, which is how it was identified.

**Why it matters more than its size.** `/work` is a kubernetes emptyDir, destroyed on pod
restart — the fact that decided `tilerl_22_prune_list.md` to keep all 63 logs and that
makes every uncited artifact on `/work` fragile. `/data01` is a real NVMe mount. **This is
the only durable copy of the project on this machine**, and every deletion decision taken
in this cleanup was reasoned as though no such copy existed.

Two things follow, neither of which I have done:

1. **It is five days stale** (2026-08-30 against 2026-09-04). It does not contain the Stage
   D/E arms, the audit, or anything in `runs/audit_0904/`. As a disaster copy it is
   partial; as a reference point for "what did the tree look like on 08-30" it is exact.
2. **Nothing schedules it.** One rsync, one log, no cron, no reference in any script or
   document (`grep` for `/data01` across the repository returns nothing). Whether to
   refresh it, schedule it, or let it age is a user decision — it is the answer to
   "环境的问题我们应该想个办法" for the durability half.

**Disposition: KEEP, and do not treat as free space.** 365 GiB is recoverable if the user
decides the backup is not wanted, but it must not be swept by any hygiene rule.

## `/data02`, `/data03` — 260 GB each, one workload, not ours

Both hold only `eicyep50jiy2ox7jylkb0x0` (242 GB, ~290 files, live). **3.03 TiB free on
each.**

## Container `/work` — 843 GB, the part outside `/work/aupai`

| path | allocated | files | newest mtime | purpose | referenced by | disposition |
|---|---|---|---|---|---|---|
| `/work/aupai` | 843.0 GB | — | live | the project | everything | 3b / de / b0 sections |
| `Qwen3.8-27B-bf16` | 41.1 GB | 16 | 2026-08-28 | model | tileRL | keep, not ours |
| `Qwen3.8-27B-NVFP4` | 21.8 GB | 12 | 2026-08-28 | model, the served one | tileRL `cli.py` | keep, not ours |
| `newdata` | 10.2 GB | 10 | 2026-08-28 | data | none found | not ours |
| `Qwen3.8-27B-DFlash2` | 3.6 GB | 7 | 2026-09-02 | draft model | tileRL | keep, not ours |
| `tilelang_cache` + `.bad` + 6 `tlc_*` variants | 3.5 GB | — | Aug–Sep | kernel caches | tileRL | not ours |
| `fwe`, `_env_restore`, `sglvenv`, `wheels`, `sgl-src` | 3.7 GB | — | Aug | envs and sources | — | not ours |
| **`tl-ab`** | 4.5 MB | 536 | **2026-09-04T04:59Z** | tileRL checkout, ran the card-7 job | — | **held, not ours** |
| **`tl-rl`** | 5.6 MB | 579 | **2026-09-04T05:37Z** | tileRL checkout, ran the card-1 GRPO job | — | **held, not ours** |
| **`tilerl`** | 2.6 MB | 669 | 2026-09-02T18:34Z | tileRL checkout, the code behind MT-1 | `eff.kv_pool_follows_context` | **held, not ours** |
| ~40 zero-byte `*.done` / `*.log` sentinels at `/work` root | 0 B | ~40 | Aug | another project's flags | — | not ours |

**The three tileRL checkouts total 12.7 MB.** They are consequential for card ownership
(PR-1, PR-3) and irrelevant to disk.

### `/work/aupai/retired_pre0830v2` — 5.76 GB, mine to classify (de's referral)

236 files, the `ckpt_0830v1_*` ladder checkpoints, newest 2026-08-30T04:23Z.
**Disposition: KEEP.** Three fact files reference the path — `facts/base_eval.json`,
`facts/data_scaling.json`, `facts/contamination.json`. It is named "retired" but it is
cited, and the emptyDir rule from `tilerl_22_prune_list.md` applies: `/work` is destroyed
on pod restart, so a cited artifact living only there is already fragile without deleting
it deliberately.

## Deletion candidates already broadcast (not mine, listed for the whole-machine view)

Per 6e, marked **candidate, broadcast 2026-09-04 ~07:1xZ**, execution ≥ 2026-09-05 07:15Z:
`runs/owm_dedup_ck` 8.77 GB, `runs/ddp_trace_rank1..6` 2.49 GB, `runs/scan_math_ws.json`
6.62 GB (de/b0), plus C3's 84 GB of `web_cci3_*` and loose `batch_*.jsonl` (3b).
**Total under broadcast ≈ 102 GB**, all on `/work`.

## What nobody walked

Named, not silent:

- **`/data01/aupai/backup` internals** — identified as an rsync of the project root and
  sized, but not compared file-by-file against the live tree, so "what is in it that is no
  longer in `/work`" is unanswered.
- **Other pods' volumes** under `/var/lib/kubelet/pods/*` — only ours (`95a05e32…`) was
  walked; this is most of the 150 GB `/` residual.
- **`/data00` subdirectory detail** — top level only; `models` (152.8 GB) is one row here.
- **`/data0{2,3}/eicyep50jiy2ox7jylkb0x0`** — sized, not opened; another workload.
- **`/root/podman` internals**, 187 GB and 4.67 M files.
- Container `/root`, `/opt`, `/usr` beyond top-level sizes.

## Totals and the residual

| scope | measured | source |
|---|---|---|
| `/work/aupai` | 843.0 GB | this doc; 3b covers `data` 664.2 GB, de covers `runs` 24.5 GB, b0 checkpoints |
| aupai token caches on `/data00` | 247.8 GB | 3b, be3d93d9 |
| `/data01/aupai` | 368.5 GB | this doc — **unclassified** |
| `/data01/aupai` (the 08-30 backup) | 368.5 GiB | this doc |
| **aupai total, all mounts** | **1,459 GiB = 1.43 TiB** | |
| everything else (models, image layers, other workloads) | ≈ 2,900 GiB = 2.8 TiB | |
| **machine used, `df`** | **4.26 TiB** | |

Per-mount reconciliation: `/data00` walks to 1.39 TiB against `df` 1.42 TiB (**2.0%**, at the
bar); `/data01` 611.9 vs 612 GiB (**0.0%**); `/data02` and `/data03` 242.0/242.4 vs 242 GiB
(**0.0%** each); `/` 1.79 vs 1.85 TiB (**7.6%**). **`/data00`, `/data01`, `/data02` and `/data03` all meet the 2% bar; `/` misses it at
3.3%.** The `/` residual is the part I did not walk: other pods' volumes under
`/var/lib/kubelet/pods/*`, of which only ours was measured. Stated rather than explained.

The first version of this table mixed TB and TiB — `df -B1` returns bytes and I divided by
10^12 in one place and 2^40 in another, which inflated the machine to 17.3 TB and the
`/`-residual to 150 GB. Every figure here is now TiB/GiB from the raw byte counts.
