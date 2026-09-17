# Infra provisioning rebuild after the 2026-09-16 pod loss

Scope: standing up a NEW node so the data funnel has somewhere to run. This is the
infrastructure precondition that `docs/standards/data_pipeline_rebuild_0916.md` (PR #404,
opening scope note) assumes and does not describe — persistent volumes, the static pod, the `~/bin/pod`
transport, H20 visibility, code delivery, claims, and backup. Data regeneration (tokenizer
gates, corpora, KenLM, pools, κ labels) lives in that document and is not repeated here.

Write-only. Nothing here has been executed; the old pod is permanently stopped and must not
be probed. Every command is a recipe for the node the user provisions next, gated on the
user's release.

## State tags

| tag | meaning |
|---|---|
| `[ON-MAIN]` | code/doc survives in git; clone or PR gets it. Nothing to recover. |
| `[LOST-REDO]` | lived only on the emptyDir and was wiped; regenerate (data steps in #404). |
| `[SURVIVES-PERSISTENT]` | sits on a hostPath/separate disk, not the emptyDir; verify by reading before rebuilding — it may have survived. |
| `[EXTERNAL-FETCH]` | pulled from an upstream mirror at provision time; not stored by us. |
| `[OPEN-PR]` | on a feature branch, not on `main`; fetch the branch, do not assume main holds it. |

## 1. Root cause and the two hard rules

**Root cause.** The working tree `/work/aupai` was a Kubernetes **emptyDir**
(`scripts/harness.py:348` `EPHEMERAL_MOUNTS = ("/work",)`). An emptyDir is created when the
pod is scheduled and **deleted with the pod**; on 2026-09-16 the static pod was permanently
removed and every byte under it — roughly 662 GB (measured at teardown, not a recorded
fact: checkpoints, corpora, token caches, runs/ ledgers, and the gate tokenizer copy — was
wiped. The container deleting
cleanly was the failure, not a crash: nothing about an emptyDir survives pod removal.

`check_root_durable` named this exact risk for the whole campaign
(`scripts/harness.py:3789-3794`): it FAILs when the root is on `/work` unless a recent backup
marker exists on a durable mount. The FAIL was knowingly carried under `--force` because
mid-campaign nobody relocates a standing tree (`scripts/harness.py:3775-3778`). Knowing the
risk and accepting it is not a mount; the pod still deleted empty.

**Hard rule 1 — the new AUPAI_ROOT must be a persistent hostPath or a separately attached
disk, never an emptyDir.** Bind the working tree at `/work/aupai` (or set `AUPAI_ROOT`) to
a mount whose lifecycle is the NODE's, not the pod's: a hostPath on host NVMe, or a
dedicated disk mounted into the container at that path. Then a pod removal leaves the bytes
in place and a replacement pod remounts them. An overlay/emptyDir at the root is refused at
provision time, not accepted and --force'd.

The repo already has the syscall machinery for this and documents why a naive bind does not
propagate: `scripts/host_mount_into_container.py` (open_tree(CLONE) + move_mount; the
private-/work-mount reason a host-side bind is invisible) and the idempotent,
read-verified attach wrapper `scripts/attach_nvme_caches.py` (st_dev differs from `/` AND a
known sha prefix matches — a dropped mount otherwise survives as an empty look-alike
directory and silently triggers a 247.8 GB cache rebuild).

**Hard rule 2 — the backup destination must be a real mounted filesystem, and a backup must
actually land on it.** `scripts/pod_backup.sh` refuses to copy to a path that is not a
mountpoint (`scripts/pod_backup.sh:59-71`); the failure this guards was already paid for on
2026-08-30 — a directory that merely existed at `/mnt/data02`, was not a mount, and would
have backed up onto the same ephemeral disk (`scripts/pod_backup.sh:29-46`). On the lost
node the intended `/mnt/data02` target was not mounted inside the container, so no durable
backup existed; `root_durable`'s 48h marker (`scripts/harness.py:3731`) could not go green
on bytes that were never written. At provision: attach the backup disk, confirm
`mountpoint(1)` / st_dev from inside the container, run one `pod_backup.sh`, and confirm the
marker and MANIFEST sha on the durable mount **before** any GPU work starts.

## 2. Open questions to answer on the node before rebuilding

These cannot be resolved from a laptop and are not assumed either way:

1. **Do the host NVMe `/data00`–`/data03` physically survive the pod removal?** They are
   node disks, not emptyDir, so they are expected to. Read them from the host first.
   `[SURVIVES-PERSISTENT]`
2. **Teacher weights** at `/data00/models/Qwen3.6-35B-A3B-FP8` were on a hostPath NVMe, not
   the emptyDir, and were out of scope of the loss audit — `ls /data00/models/` before any
   re-download. `[SURVIVES-PERSISTENT]`
3. **The static-pod spec is not in the repo.** No pod/sglang/static YAML is tracked
   (confirmed: zero matches under the repo). The `sglang-test` static pod was created
   out-of-band on the node via the kubelet. Recover the manifest from the node/controller
   records; this runbook cannot regenerate it. `[LOST-REDO]` (spec only).
4. **Did `pod_backup.sh` ever produce a durable MANIFEST?** If `/mnt/data02` was ever
   correctly mounted and a backup ran, final/milestone checkpoints and runs/facts may exist
   there. Check for `aupai_backup/MANIFEST` on every persistent disk before treating
   checkpoints as gone. `[SURVIVES-PERSISTENT]`
5. **Host NVMe token caches** (the 22 caches, ~231 GB, `scripts/attach_nvme_caches.py`) may
   still hold gate-vocab caches; verify the recorded sha prefix before deciding to
   re-pretokenize. `[SURVIVES-PERSISTENT]`

## 3. Provisioning order

Dependency order; each step verifies before the next starts. No step launches training.

1. **Node + static pod with persistent volumes.** Recover/apply the `sglang-test` static-pod
   spec (open question 3). In it: attach host NVMe `/data00`–`/data03`, mount AUPAI_ROOT
   from a persistent hostPath/disk (hard rule 1), and attach a real backup disk at
   `/mnt/data02` (hard rule 2). Image is the sglang cu129 image; all 8 H20 must be visible
   (`nvidia-smi` from inside shows 8 devices). `scripts/pod:4-8` documents the static-pod +
   crictl transport and why the node kubelet cannot `kubectl exec`.
2. **Transport wrappers (laptop, per machine).** The wrappers are tracked:
   `ln -sf "$PWD/scripts/pod" ~/bin/pod && ln -sf "$PWD/scripts/podput" ~/bin/podput`.
   Gate: `scripts/test_pod_wrappers.sh`, laptop-only. `[ON-MAIN]`
3. **Clone the code on the persistent root.** Fresh clone onto the mounted root; `/work`
   on the lost node was never a git repo (files were pushed), which is part of why nothing
   there was recoverable from git. On the new node make the root a real checkout on the
   durable disk. `[ON-MAIN]`
4. **Python/uv environment.** `uv sync` inside the container (python 3.12, torch cu129 per
   the old image; verify on the new image). Provisioning is otherwise driven by
   `scripts/bootstrap_pod.sh`, idempotent one stage at a time: verify → fetch → build →
   vocab → check → caches (`scripts/bootstrap_pod.sh:8-16`). The `caches` stage re-attaches
   the NVMe mount host-side and verifies by read. `[ON-MAIN]`
5. **Confirm hard rules from inside the container before data work:** root is not under
   `/work`-as-emptyDir (`scripts/harness.py check` → `root_durable` PASS, not --force),
   `mountpoint /mnt/data02` holds, one backup round-trip verified.
6. **Card grants.** Allocation is a controller decision recorded in
   `runs/card_assignment.json`; `harness launch` refuses a card outside the grant. Claims
   live in `runs/claims/` in the tree the job runs from. Re-establish the grant file before
   any launch; do not infer ownership from `nvidia-smi`. `[ON-MAIN]`
7. **Code delivery.** Code reaches a node by PR merge to `main` then
   `scripts/pod_push.sh <files>` plus `scripts/pod_sync_check.sh`; never bare `podput` for
   code, and no advance podput of unmerged code. Data bytes may still be copied directly.
   `[ON-MAIN]`

## 4. Local surviving assets and copy-in points

Bytes that survived off-pod on a laptop; copy these in rather than regenerating.

| asset | local path | copy-in point | state |
|---|---|---|---|
| gate tokenizer (32,768 BPE, `<eos>=1`, `[NUM]=32767`) | `/Users/bytedance/code/aupai-de/data/tokenizer.json` | new root `data/tokenizer.json` | `[SURVIVES-PERSISTENT]` verified 2026-09-16: 2,287,069 B, sha256 prefix `c6d5eec97c6af1ba` |

After copy, gate the actual file (a bare `--selftest` runs internal known-answer worlds and
ignores `--tokenizers`, so it does NOT read the copied file):

```bash
python3 -c "import sys;sys.path.insert(0,'scripts');import harness as h;print(h.check_tokenizer_roundtrip('.'));print(h.check_pinned_ids('.'))"  # load + NUL/tab/hanzi/digits round-trip; <eos>=1, [NUM]=32767
python3 scripts/tokenizer_eval.py --veto_only --tokenizers data/tokenizer.json --domains <a-present-corpus-dir>  # round-trip, 256 bytes, ref fertility <=1.55, scoped hanzi
```

`check_tokenizer_roundtrip`/`check_pinned_ids` need no corpus (run against the repo root);
`--veto_only` (ae #442) skips the 800-row held-out split so it runs before a gate-scale corpus
exists, but needs one shard-bearing `--domains` dir for the hanzi/ref-fertility reading (on the
zero-Chinese gate mix the hanzi veto is N/A, not failed). Confirm sha256 still prefixes
`c6d5eec97c6af1ba` after copy. Do not retrain unless a gate fails (full recipe in #404 §1).
Other local working trees
(`aupai-de`, `aupai-66-*`) hold code already on `main` or on feature branches — they are
not data backups and need no copy beyond what git carries.

## 5. Branch notes

- The data rebuild and all current labeler/ledger code are on `main` (`#400`, `#404`).
- **v41f P0/P1 are on `main` (no longer branch-only).** Hyper-Connections (`v41f/hyperconn.py`),
  the v41f MoE (`v41f/moe.py`), the MTP **loss** pieces (`v41f/mtp.py`), the assembled
  `v41f/block.py` and whole-net `v41f/model.py`, plus engram/compressor/indexer/rope/window/
  norm_gate and `tests/v41f/p0_*` / `p1_*` are merged and green — a rebuilt node that checks out
  `main` can build v41f. The only remaining piece is the **DSpark draft-block (P3)**, still
  `[OPEN-PR]` in #454 (`0e-v41f-mtp`); do not assume the draft/decode block is on main.
- The L3 multi-endpoint/`reason`/ledger labeler WIP on session 66's local branch
  (`66-l3-locked-ledger`) is superseded by `datagen/l3_label_drive.py` on `main` (#400);
  keep the branch for history, do not deploy it. The 50k drive in #404 §7 uses the merged
  driver.

## 6. Full artifact state table

| artifact | state | recovery |
|---|---|---|
| repo code, scripts/pod, podput, bootstrap, pod_backup, mounts, harness | `[ON-MAIN]` | clone |
| gate tokenizer | `[SURVIVES-PERSISTENT]` | laptop copy, §4; gates then copy |
| teacher A3B weights `/data00/models/...` | `[SURVIVES-PERSISTENT]` | read `/data00/models/` first |
| host NVMe `/data00–03`, token caches | `[SURVIVES-PERSISTENT]` | attach + sha-verify |
| possible prior backup `aupai_backup/MANIFEST` | `[SURVIVES-PERSISTENT]` | scan persistent disks |
| static-pod `sglang-test` spec | `[LOST-REDO]` | recover out-of-band, not from repo |
| AUPAI_ROOT tree, checkpoints, runs, facts on `/work` | `[LOST-REDO]` | persistent remount prevents recurrence; contents per #404 / facts re-record |
| corpora `en_c4_stage2_dc`, `code_py_starcoder_dc` | `[LOST-REDO]` + `[EXTERNAL-FETCH]` | #404 §2 |
| KenLM, L2 pools, locked κ sets, 50k labels | `[LOST-REDO]` | #404 §3–7 |
| v41f P0/P1 (HC/MoE/MTP loss/Block/整网) | `[ON-MAIN]` | merged; build v41f from `main` |
| v41f DSpark draft-block (P3) | `[OPEN-PR]` | #454 branch `0e-v41f-mtp` |

## 7. What "done" means for provisioning

The node is provisioned, and only then may #404 begin, when all hold, read from inside the
container:

1. `df`/st_dev show AUPAI_ROOT on a persistent disk; deleting a test pod leaves a marker
   file in the tree present after pod recreation (the direct test of hard rule 1).
2. `/mnt/data02` is a mountpoint and `scripts/pod_backup.sh --dry-run` accepts it; a real
   run writes MANIFEST whose sha-verified files read back from the disk.
3. `scripts/harness.py check` has `root_durable` PASS with no `--force`.
4. `nvidia-smi` shows 8 H20 and the grant file names the allocation. (GPU numerics are
   gated separately — `docs/standards/gpu_smoke_gate_0917.md`, #469; this document owns only
   disk/mount items.)
5. `/data00/models/` and the token caches are accounted for (survived-and-verified or
   scheduled for fetch), and the surviving tokenizer passes its gates.

## 8. Node-ready checklist — disk and mount (operator)

The box-by-box version of §7's disk items. Tick every one before #404 starts. "Where" says
HOST (a shell on the node, via `tn`/`crictl` on the host namespace) or CONTAINER (inside the
`sglang-test` container, via `scripts/pod`). A mount is a **separate filesystem identified by
a different `st_dev` from `/`** — never judge it by the directory existing or by `mountpoint`
alone (`mountpoint` is absent/misleading across the two namespaces; the repo's own gates use
`st_dev`). Do not proceed past a FAIL: accepting an overlay/emptyDir at the root, or a backup
"disk" that is the same ephemeral fs, is exactly the 2026-09-16 loss.

### A. AUPAI_ROOT survives pod removal (hard rule 1)

- [ ] **A1 — host NVMe present (HOST).** `ls -ld /data00 /data01 /data02 /data03` and
  `stat -c '%n %d' / /data00 /data01 /data02 /data03`.
  Expect: the `/data0x` device numbers differ from `/` (separate filesystems) and the dirs
  exist. FAIL action: attach/format the node NVMe before scheduling the pod; an empty dir on
  `/` is not persistent.
- [ ] **A2 — AUPAI_ROOT bound to persistent storage in the pod spec (HOST).** Inspect the
  recovered static-pod spec (§3.1): the `/work/aupai` mount (or `AUPAI_ROOT`) is a
  hostPath/volume on node NVMe, **not** an `emptyDir: {}`.
  Expect: a hostPath/persistent volume entry; no `emptyDir` for the working tree. FAIL: fix
  the spec and re-apply; never launch against an emptyDir root.
- [ ] **A3 — root is a separate fs from inside (CONTAINER).**
  `python3 -c "import os;print(os.stat('/work/aupai').st_dev, os.stat('/').st_dev)"`.
  Expect: the two numbers differ. FAIL: the bind did not propagate into the mount namespace
  (the documented private-/work case) — attach with `scripts/host_mount_into_container.py`
  (`open_tree(CLONE)` + `move_mount`) or re-apply the spec; do not train.
- [ ] **A4 — delete-pod persistence test (CONTAINER, decisive).** Write a sentinel:
  `echo $$ > /work/aupai/PERSIST_PROBE`, then delete/recreate the test pod per the controller
  (or have provisioning cycle it), and `cat /work/aupai/PERSIST_PROBE` from the new
  container.
  Expect: the file and its content are present after recreation. FAIL: root is still ephemeral
  — stop, A2/A3 are wrong; nothing run against this tree is safe. (Remove the probe after.)
- [ ] **A5 — `root_durable` PASS, no `--force` (CONTAINER).** From `/work/aupai`:
  `python3 scripts/harness.py check 2>&1 | grep root_durable`.
  Expect: `[PASS] root_durable …` (a fresh backup marker on a durable mount is what makes it
  pass once the root itself stays on `/work`; see C). FAIL/WARN: do not waive; follow its
  named move/backup instruction.

### B. `/mnt/data02` is a real mounted filesystem (hard rule 2)

- [ ] **B1 — distinct device from inside (CONTAINER).**
  `python3 -c "import os;print(os.stat('/mnt/data02').st_dev==os.stat('/').st_dev)"`.
  Expect: `False` (different device). FAIL: it is a directory on the container's own fs —
  attach the backup disk (`scripts/attach_nvme_caches.py` host-side, then it is visible in the
  container); a directory that merely exists is the 2026-08-30 failure.
- [ ] **B2 — backup dry-run accepts the mount (CONTAINER).**
  `bash scripts/pod_backup.sh --dry-run`.
  Expect: `would back up N path(s) …` and exit 0 (no "not a mounted filesystem" refusal).
  FAIL: the destination is not a separate fs; do not run a real backup until B1/B2 pass.
- [ ] **B3 — real backup round-trip, sha-read back (CONTAINER).** Run
  `bash scripts/pod_backup.sh`, then check the marker and read a listed file back:
  `ls -l /mnt/data02/aupai_backup/MANIFEST` and verify one `sha256sum -c` entry from inside
  `/mnt/data02/aupai_backup`.
  Expect: MANIFEST present, fresh (within the 48h `root_durable` window), and the checksum
  verifies. FAIL: backup landed on the wrong fs or bytes do not match — re-attach and rerun;
  this confirmation must precede any GPU work.

### C. Account for surviving persistent data before rebuilding (§2)

- [ ] **C1 — teacher weights (HOST/CONTAINER).** `ls /data00/models/` for
  `Qwen3.6-35B-A3B-FP8`; record present or to-fetch. Skip the re-download only if verified.
- [ ] **C2 — token caches re-attached and sha-verified (CONTAINER).**
  `python3 scripts/attach_nvme_caches.py --verify-only` (host-side attach first per §3.4).
  Expect: `OK: already mounted and readable` with the recorded sha prefix matching; a dropped
  mount otherwise survives as an empty look-alike and silently triggers a 247.8 GB rebuild.
  FAIL: attach host-side, or schedule re-pretokenize in #404.
- [ ] **C3 — prior-backup MANIFEST scan (HOST).** On every persistent disk, look for an
  earlier `aupai_backup/MANIFEST` before treating any checkpoint as gone; list what it covers
  to the controller.

GPU/device numerics and the 8×H20 visibility acceptance are NOT here — they are
`docs/standards/gpu_smoke_gate_0917.md` (#469). Data regeneration (tokenizer gates, corpora,
KenLM, pools, labels) is `docs/standards/data_pipeline_rebuild_0916.md` (#404). This checklist
is only the persistent-disk/mount block; all three must pass before the node is "done".
