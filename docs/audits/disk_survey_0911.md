---
question: "Whole-machine disk 2026-09-11: what occupies >=2 GiB, who owns it, who could approve deletion (read-only, nothing deleted)"
status: measured
source: "host via `tn exec` (python 3.7, du/df/crictl/stat) + sglang-test container via ~/bin/pod; 2026-09-11 ~08:30Z"
---

# Whole-machine disk survey 2026-09-11

Read-only. **Nothing was deleted.** Numbers are `du -x` (one filesystem, excludes
bind mounts) GiB; "newest mtime" is the newest file inside the path at depth 1.
The only constrained filesystem is the **2.0 TB root `/dev/vda2` at 91% (189 GB
free)**. The four 3.5 TB NVMe mounts are at 8-16% and are not pressure.

## Filesystems

| mount | device | size | used | use% | notes |
|---|---|--:|--:|--:|---|
| `/` (also every overlay rootfs) | /dev/vda2 | 2.0T | 1.8T | 91% | **the only constrained disk; 189 GB free** |
| `/data00` (host) | /dev/nvme0n1 | 3.5T | 511G | 16% | teacher models + old bpe cache |
| `/data01` | /dev/nvme1n1 | 3.5T | 244G | 8% | EIC agent RocksDB volume |
| `/data02` | /dev/nvme2n1 | 3.5T | 243G | 8% | EIC agent RocksDB volume; pod backup target |
| `/data03` | /dev/nvme3n1 | 3.5T | 243G | 8% | EIC agent RocksDB volume |
| `/work` (in container) | /dev/vda2 | 2.0T | 1.8T | 91% | same physical root disk as `/` |
| tmpfs /dev/shm | tmpfs | 4.0T | 1.6G | 0% | RAM, not disk |

**Topology finding (load-bearing):** the sglang-test container's `/data00` is
**not** the host NVMe `/data00`. Inside the container `df /data00` resolves to the
overlay on `/dev/vda2` (inode verified identical to containerd snapshot 1650); the
host NVMe `/data00` holds a different, older token set. So the gate token caches
98 builds into the container's `/data00/tokens_*.pt` are physically written to the
91%-full **root disk**, not to an NVMe. The container's `/work` is a bind to
`/dev/vda2` as well.

**NVMe is NOT writable inside the container (corrected after probe):** the in-container
`/mnt/data02` and `/mnt/nvme_probe` mountpoints for nvme2n1 are dead binds — `mkdir`
returns ENOENT (`df` lists them but they are inaccessible). `/mnt/data01`,
`/mnt/nvme02`, `/mnt/recover` appear writable but `df` resolves them to the **overlay**
(root disk), so they are ordinary root-disk directories, not NVMe. Net: as of
2026-09-11 the container has **zero reachable writable NVMe**; all four 3.5 TB NVMe
devices (host /data00-/data03, ~10.4 TB free) are usable only from the host via
`tn exec`. This is also why a byte-identical copy of a model on host `/data00` is NOT
usable by an in-container job — it must be copied into the container/root disk (proven
by the Qwen3.8-NVFP4 delete/restore, noted in runs/deletion_0911b.txt row 4).

## Entries >= 2 GiB, largest first

| path | GiB | owner | newest mtime inside | what it is | who could approve deletion |
|---|--:|---|---|---|---|
| `/var/lib/containerd/.../snapshots/1650/fs/data00/aupai_raw/ultradata` | 148.4 | aupai (0e) | 2026-09-11 | raw UltraData parquet working copy on the **root overlay** (sglang-test layer), actively converting | 0e + fb; likely intermediate but 0e has not confirmed disposability |
| `/var/lib/containerd/.../snapshots/1650/fs/tmp` (1661 `tmp*`, 1598 dated 09-11) | 188.4 | aupai + tileRL | 2026-09-11 | container `/tmp` scratch: 2.9G-each worker temp dirs (183G), `enc4_s1` 24.3G, torchinductor 4.8G; sits on root overlay | aupai for `tmp*`/torchinductor (scratch, verify no live pid); `enc4_s1` owner unknown — see below |
| `/root/podman/storage/vfs` (host) | 174.4 | host/container owner (aupai/fb per tilerl-58) | 2026-07-08 **(>7d stale)** | podman VFS image/layer store: 418 layer dirs, 438 image dirs, 52 containers, all from July 6-8 | fb (host owner); deleting breaks envs until re-pull; not a tileRL session's call |
| sglang-test layer 1650 `/data00/tokens_*.pt` (gate caches, ~17 files) | 218.5 | aupai (98) | 2026-09-11 | pretokenized gate caches written to the **root overlay** container /data00 (l2_dc 57.1, starcoder + _dc 29.6 each, math + _dc ~21.9 each, textbook 14.3, keep + _dc 9.8 each, c4 + _dc 7.4 each, wiki 1.8, cot + _dc, en, rp1t + _dc) | 98 + fb; REQUIRED for the gate run — do not delete; duplicate of the host-NVMe older set is a separate item |
| snapshot 1623 (`/tmp` 46.6, `/root` 32.1, `/sgl-workspace` 16.8) | 97.6 | old sglang-test layer (container attempt 7, Exited) | 2026-08-30 **(>7d stale)** | writable layer of the PREVIOUS sglang-test container (task attempt 7, exited 3 weeks ago); superseded by running attempt 8 / snapshot 1650 | fb/infra; an exited container's layer, not mounted |
| `/work/aupai/data/corpus/code_ultra_l3_noexec` | 91.0 | aupai (0e) | 2026-09-11 | static-only L3 aggregate in progress (all 147 shards, no exec filter) | 0e; live until promoted to `code_ultra_l3_noexec_dc` |
| `/data00/DeepSeek-V4-Flash-0731` (host NVMe) | 155.4 | aupai (distill/arch ref) | 2026-08-20 **(>7d stale)** | DeepSeek V4-Flash teacher/reference weights | fb; reference model, needed for any future distill comparison |
| `/work/aupai/data/corpus/zh_web` | 83.7 | aupai (retired 0830v1 corpus) | (corpus dir) | Chinese web corpus from the pre-pivot plan; not in mix_v41_gate | fb; retired data plan, but corpus bytes were kept at the pivot |
| snapshot 1530 (`/sgl-workspace` 12.7, `/tmp` 5.8, `/usr` 3.7) | 22.3 | another sglang container (task 2d255408) | recent | overlay layer; container identity not resolved to a friendly name | fb/infra |
| `/data00/Qwen3.6-35B-A3B-FP8` (host NVMe) | 34.9 | aupai/tileRL (model) | 2026-08-22 **(>7d stale)** | Qwen 3.6 35B FP8 weights | model owner session / fb |
| `/data00/.bpe_cache_0828` (host NVMe) | 35.4 | aupai | 2026-08-28 **(>7d stale)** | pre-rebuild BPE/tokenizer work cache | fb; tied to the superseded 32,773 vocab work |
| `/data00/Qwen3.8-27B-FP8` + `Qwen3.6-27B-FP8` (host NVMe) | 57.6 | aupai/tileRL | 2026-08-20/21 **(>7d stale)** | two Qwen FP8 weight sets | model owner / fb |
| `/data00/DeepSeek-V4-Flash-DSpark-draft` (+`-fp8` 18.6) (host NVMe) | 47.3 | aupai | 2026-08-10 **(>7d stale)** | DSpark draft model two formats | fb |
| `/work/aupai/data/corpus/code_ultra_l2_dc` | 54.0 | aupai (0e) | 2026-09-11 | finished L2 decontaminated aggregate (554 shards, fp adf2ff20) | 0e; REQUIRED gate domain |
| `/work/aupai/ckpt*.pt[.*]` (in /work/aupai) | 61.6 | aupai | 2026-08-23 to 2026-09-09 | checkpoints incl v41 smoke x3 (12 GB each), 30B e48 milestones (5.6 GB x3), old `.step` rolling saves (~0.4 GB) | controller per pod deletion list; `.stepN` rolling saves are excluded from backup by policy |
| `/work/aupai/data/corpus/code_py_starcoder` + `_dc` | 27.5 + 27.4 | aupai (ae-7 / corpus) | 2026-09-11 | original and decontaminated starcoder copies; zero-hit shards hard-linked so the extra ~27.4G is mostly the rewritten 281 shards | fb; once launch reads only `_dc`, the pre-_dc original is the disposable copy (same pattern for every _dc pair) |
| `/data00/Qwen3.8-27B-NVFP4` + `ThinkingCap-...-NVFP4` (host NVMe) | 42.8 | tileRL/aupai | 2026-08-20 **(>7d stale)** | NVFP4 weight sets (one also at /work 21.8G) | model owner / fb |
| `/work/Qwen3.8-27B-NVFP4` (container, on root) | 21.8 | tileRL/aupai | 2026-08-18 **(>7d stale)** | duplicate of an NVMe-hosted weight set, on the constrained disk | fb/model owner; candidate duplicate |
| `/work/aupai/data/corpus/math_owm_stage2` + `_dc` | 20.1 + 20.0 | aupai | 2026-09-11 | original + decontaminated math copies | fb; pre-_dc original disposable post-launch |
| snapshot 1558 (`/usr` 13.7) | 13.7 | sglang container layer | — | overlay /usr layer | fb/infra |
| `/work/aupai/data/sft` | 13.0 | aupai | — | SFT packs (several unstamped/superseded, inventoried in cont.sft_clean_pack boundary) | fb; 21 older packs await named disposition |
| `/data00/fw2raw` (host NVMe) | 18.0 | aupai/tileRL | 2026-08-26 **(>7d stale)** | framework-to-raw model conversion output | fb/model owner |
| `/work/aupai/data/p1` | 15.1 | aupai | — | p1 keep-set sources backing the hard-linked code_keep_p1 domain | fb; source of code_keep_p1 hard links |
| `/work/aupai/data/raw` | 10.9 | aupai | — | raw fetches | fb |
| `/var/log/atop` (host) | 11.4 | host/system | 2026-09-11 | atop process recorder (only 09-08..09-11 present) | host admin; short retention already |
| `/work/aupai/data/corpus/textbook_30b` | 7.6 | aupai (stopped) | — | teacher textbook synthesis, stopped by the 2026-09-10 pivot | fb; retired teacher-synthesis plan |
| `/work/aupai/data/corpus/code_ultra_l3` (exec partial) | 7.6 | aupai (0e) | — | execution-filtered partial L3 arm, NOT promoted, reserved for a later A/B | 0e/fb; kept deliberately per the noexec user order |
| `/root/miniconda3` (host) | 25.2 | host env | — | system python | host admin |
| `/root/sccache` (host) | 13.8 | build tooling | — | compiler cache | host/admin or build owner |
| `/work/aupai/data/corpus/code_keep_p1` + `_dc` | 8.9 + 6.5 | aupai | 2026-09-11 | assembled flat keep set + decontaminated copy | fb; pre-_dc disposable post-launch |
| `/work/aupai/data/corpus/en_c4_stage2` + `_dc` | 8.1 + ~8.1 | aupai | — | original + decontaminated en c4 | fb; pre-_dc disposable post-launch |
| `/sgl-workspace` (sglang-test layer, /work view outside aupai) | 12.4 | sglang/tilerl | — | sglang checkout + env at container root | fb/infra |
| `/host/cuda-12.9` (host, seen in container /host) | 7.9 | host toolchain | — | CUDA 12.9 toolkit; tilerl-58: KEEP, it is the env | not disposable |
| EIC RocksDB `/data0{1,2,3}/eicyep...` | 243+242+242 | EIC platform service (not aupai/tilerl) | 2026-07-15 **(>7d stale)** | `eic-agent` RocksDB `.sst` state across three NVMe | EIC service owner / platform — outside this team |
| `/root/tc27-nvfp4-slice4` + `slice2` + `/root/Qwen` (host) | 6.9 + 6.4 + 6.4 | model build scratch | recent/slice work | NVFP4 slicing scratch + Qwen copy | build owner / fb |
| snapshot 1650 layer `/root/.cache` + `.triton` + `.rustup` | 10.3 + 1.9 + 1.5 | shared dev cache | 2026-09-11 | pip/triton/rust caches inside the container layer | disposable caches, regenerate; fb |
| `/work/newdata` | 10.2 | unknown (pod, pre-pivot) | 2026-08-28 **(>7d stale)** | unidentified data dir | owner must be identified before any action |
| `/work/tilerl-s-wt-pred` | 5.2 | tileRL session (NOT tilerl-58: their tree is `tilerl-s-wt-pod`) | 2026-09-08 | earlier spec/prefill worktree per 58 | tileRL a3/cc/65/5f |
| `/host/tilelang-main` | 1.0 | tilelang | — | referenced tilelang checkout; 58: KEEP | not disposable |
| `/tmp` (host itself) | 1.5 | host | — | host /tmp (distinct from container /tmp) | host admin |
| journal (host) | 3.9 | systemd | live | journald logs | host admin |
| `/work/tilelang_cache` + `.bad` | 3.8 + 1.0 | all tileRL/tilelang sessions | 2026-09-11 | TileLang JIT cache (regenerates, first compile slower); `.bad` is a quarantined corrupt cache | any tileRL session can acknowledge; `.bad` ask who renamed it |
| `/work/fwe` | 1.2 | unknown | — | small project dir | owner |
| `/work/_env_restore` | 1.2 | env restore scratch | — | environment restore | fb |
| `/work/tl-rl8` | 1.0 | tileRL (a3/65 per 58, disclaimed by 58) | 2026-09-04 | RL worktree | tileRL a3 |
| `/work/tilerl-s-eval25-ctl` | 0.9 | controller run? (a3/65 per 58) | 2026-08-29 **(>7d stale)** | eval25 controller dir | tileRL a3/65 |
| `/work/sglvenv` | 1.1 | sglang env | — | python virtualenv | fb/infra |

## Stale (>7 days) and duplicated, called out separately

- **Pre-pivot/old model weights on the host NVMe** (all mtime <= 2026-08-26): DeepSeek
  0731 155G, DSpark draft 47G, three Qwen FP8 93G, two NVFP4 43G, fw2raw 18G,
  .bpe_cache_0828 35G. ~390G on a disk that is only 16% full — not pressure today.
- **Weight duplicates across disks:** Qwen3.8-27B-NVFP4 exists on host NVMe (21.8G)
  and on the constrained root `/work` (21.8G). The container gate token caches
  (218G) sit on the root overlay while a separate older token set is on host NVMe
  `/data00` — same directory name, two different physical files, not hard-linked
  (different filesystems).
- **July podman VFS store** 174G, last touched 2026-07-08 (65 days stale).
- **Exited sglang-test attempt-7 layer** (snapshot 1623) 98G, mtime 2026-08-30.
- **Container `/tmp` scratch** 188G on the root overlay, 1598 of 1661 temp dirs
  created 2026-09-11 — active worker scratch, not stale, but on the constrained disk.

## Operational note (factual, not a recommendation)

The gate-run token caches and the active UltraData raw copy are being written to
the root disk because the container `/data00` is an overlay path on `/dev/vda2`,
and the container currently has **no writable NVMe mount** (the /mnt binds are dead
or overlay; the 10.4 TB free on nvme0-3 is reachable only from the host). Moving the
caches off root therefore needs a working NVMe bind-mount into the container first;
a 2026-09-10 attempt to `mkdir /mnt/data02/tokens` already failed ENOENT. The
approver column is the only prescriptive content; the user decides.

## Method and gaps

- `du -x -B1` on host and in container; `df -h`; `crictl ps/pods/images/inspect`;
  overlay snapshot -> container mapping via /proc/self/mountinfo; inode cross-check
  that container /data00 == snapshot 1650.
- Items between ~0.9 and 2 GiB included where ownership was informative; the table
  threshold is >=2 GiB per the order, with a handful below for the tileRL questions.
- container identity of overlay snapshots 1530/1558 (task prefix 2d255408) and the
  friendly name were not fully resolved; classified fb/infra.
- `/work/newdata` (10.2G), `/work/fwe`, and `/tmp/enc4_s1` (24.3G) have no
  confirmed owner; tilerl-58 disclaimed the tileRL-named dirs and pointed to
  a3/cc/65/5f, and said enc4_s1 looks like an encrypted build scratch (safe only if
  no compiler is running — a live-compiler pid check is required before touching).
