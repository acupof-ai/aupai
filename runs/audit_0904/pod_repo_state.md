---
question: What is the true state of the pod and the repository on 2026-09-04, read from artifacts?
status: partial
source: pod reads 2026-09-04 03:45-04:05Z via ~/bin/pod and tn exec; repo reads against main at 3f451a9f
---

# Pod and repository state — partial report, 2026-09-04

Owner tilerl, pair b0. **Partial at the 3 h mark**, per the standard's "a partial at
3 hours beats a complete one at 6". What is not yet done is listed in §5, not omitted.

Audit only: nothing was deleted, killed, moved or repaired. Two findings describe live
processes; both are reported, neither was touched.

Ten findings: six S2, four S3, **no S1**. PR-1 and PR-3 were drafted S1 and restated S2 on
the controller's ruling — no published number or decision is wrong; what is wrong is a
label that conflates two properties (see PR-1).

## 1. Scope

Covered:

- `nvidia-smi` compute apps and per-card memory vs `runs/card_assignment.json` vs
  `scripts/card_claim.py status` vs container `ps` vs host `ps`/cgroup.
- Checkpoint inventory on `/work/aupai` (depth 2), full listing captured **before** the
  12:03Z prune (**UTC** — confirmed against b0's cron `1c00daae`, which reads 20:03 in local CST+8), with its timestamp.
- Disk by directory for `/work` and `/work/aupai`.
- Pod-vs-main file drift **beyond the manifest**: every `.py/.sh/.md/.json` on the pod
  outside `data/corpus`, `data/raw`, `runs`, `.git`, `__pycache__`, classified against
  main by sha256 and then against all of git history.
- Checkpoints named by any `facts/*.json` entry vs the files on the pod.

Deliberately excluded: `data/corpus/**` and `data/raw/**` (3b's area), `runs/*.jsonl`
ledger contents (de's area), the contents of any checkpoint (b0's area). `/work/tl-ab`
is inspected only far enough to identify the process on card 7; its repository state is
not audited.

## 2. Method

| what | command |
|---|---|
| compute apps | `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader` |
| card→uuid map | `nvidia-smi --query-gpu=index,uuid --format=csv,noheader` |
| process identity | `tn exec` (HOST namespace) `tr '\0' ' ' < /proc/<pid>/cmdline`, `ps -o lstart=,etimes= -p <pid>` |
| container membership | `cat /proc/<pid>/cgroup`, compared against a known-ours pid |
| checkpoint listing | `find /work/aupai -maxdepth 2 -name '*.pt*' -printf '%s\t%TY-...\t%p\n'` |
| drift | pod `sha256sum` of 197 files vs `git show main:<path>` and then vs every revision in `git log --all -- <path>` |

**Instrument tested on a broken world.** The drift classifier was run first with a
60-revision cap per file and reported `docs/lessons/gate_failure_shapes.md` as matching
NO commit. Removing the cap (179 revisions exist for that file) found the match at
`05c9b87e`. The capped version is the broken world: it produces a false "pod-only file"
for any file with a long history. Every number in §4 is from the uncapped run.

**Timezone.** The pod and the host both run UTC internally but `ps lstart` prints
host-local, which is **CST(+0800)** (`date` on the host: `2026-09-04T11:49:43 CST(+0800)`
against `2026-09-04T03:49:43Z`). A first pass read `lstart` as UTC and produced a start
time inconsistent with `etimes` by exactly 8 h. All times below are UTC, converted, and
cross-checked against `etimes` (seconds since start, zone-free).

## 3. Population counts

| population | exists | read | sampled |
|---|---|---|---|
| GPU cards | 8 | 8 | all |
| compute-app PIDs | 4 at first read, 3 at second | 4 | all |
| checkpoint files `/work/aupai` depth 2 | 403 (317.3 GB) | 403 | all (listing) |
| pod code files outside manifest scope | 197 | 197 | all, sha256 |
| manifest entries | 469 | 469 | count only |
| fact→checkpoint references | 26 (id, ckpt) pairs | 26 | all |
| KEEP pins (`*milestone_*`) | 22 | 22 | all, by inode |
| KEEP-claimed names in the 09-04 candidates file | 37 | 37 | all |
| rolling `.stepN` checkpoints | 153 | 153 | all, by inode |

## 4. Findings

| id | sev | claim as published | evidence | what contradicts it |
|---|---|---|---|---|
| PR-1 | S2 | `card_claim.py status` prints `FOREIGN card 7 ... not ours to take or reclaim`, and `runs/card_assignment.json` says card 7 is "the user's" in eight separate grant lines | `cat /proc/2880373/cgroup` = `/kubepods/besteffort/pod95a05e32.../827d3e58e99af98e06c8cf3f18e6faf46e831e67e98002f30f6b38c13c7991c2`, **byte-identical to `/proc/2453605/cgroup`** (our `b0_se_16lnew_1b` trainer). Container `ps` sees it as pid 216551/216552 | The card-7 process is **inside our own container**, not another one. It is `scripts/ab_draft_depth.py` running from cwd `/work/tl-ab` (a separate tileRL checkout, HEAD `e4aaf8c`), 44.5 GB, card 7 at 100%. "Foreign" is true about the *repository* and false about the *container* |
| PR-2 | S2 | `FOREIGN_MARKERS = ("NEVER TAKE", "FOREIGN OCCUPANT", "not an aupai job", "another container")` — `scripts/card_claim.py:316` | `card_claim.py:322-357`: `foreign_cards()` reads those strings out of `runs/card_assignment.json`'s prose and matches them case-insensitively | The foreign/ours decision is made from **a human-written sentence in a JSON note**, never from a container, cgroup or namespace check. The instrument cannot detect a wrong note, and PR-1 is a wrong note that has stood for at least four grant revisions. Nothing is known to be mis-killed — the marker currently fails safe (it over-protects) — but the same mechanism would under-protect if a note ever said "ours" about a genuinely foreign card |
| PR-3 | S2 | The audit standard's running-jobs list: "Stage E arm 1 (`b0_se_16lnew_1b`, cards 0+1) and the v14 agentic SFT build (laptop, e1)" are what continue | `ps -o lstart=,etimes= -p 2880373` → started `2026-09-04 11:45:03 CST` = **03:45:03Z**, `etimes` 280 s at read time. `audit_0904.md` landed at `2026-09-04 11:23:05 +0800` = **03:23:05Z** (`git log --diff-filter=A`) | A third GPU job started **22 minutes after the stop order**, on card 7, and is not in the sanctioned list. It is a tileRL job from `/work/tl-ab`, so it is plausibly the user's own and outside the aupai stop order — but it is in our container and consuming 44.5 GB, and no artifact in this repo records it. **Not killed** (audit-only, and its owner is unestablished) |
| PR-4 | S2 | `/work` has room for the work queued against it | `df -h /work` → `2.0T total, 1.9T used, 90G free, 96%` at 03:53Z. `du -sh /work/*`: `/work/aupai` 842 G, of which `data` 540 G and `runs` 23 G; `/work/Qwen3.8-27B-bf16` 42 G; `/work/Qwen3.8-27B-NVFP4` 22 G | 90 GB free on a shared 2 TB volume, and an earlier read the same session showed 103 GB — the trend is down while a 44.5 GB job runs. The 150–200 GB repo-shaped code fetch costed in `f6e90bfa` cannot fit, which is independent confirmation of that entry's storage note |
| PR-5 | S3 | — | `find /work/aupai -maxdepth 2 -name '*.pt*'` at **2026-09-04T03:52:17Z**: 403 files, **317.3 GB**. Saved to `runs/audit_0904/ckpt_pre_prune_0352Z.tsv` (size, mtime, path) | Recorded, not a finding: this is the pre-prune baseline the 12:03Z deletion will change. 22 `*milestone_*` KEEP pins exist as hardlinks |
| PR-6 | S3 | 197 pod code files sit outside the manifest's 469 entries | sha256 of all 197 vs main, then vs every revision in each file's history | Classification: **168 untracked in main** (build artifacts, `data/hf/*` model files, `facts/_raw`, `tmp/`), **12 byte-identical to main**, **17 tracked but differing**. Of the 17, **16 are stale copies of real commits** (`c5c04b24`, `59d751b3`, `05c9b87e`, …) — the pod holding an older push. **1 matches no commit**: `EXPERIMENTS.md` |
| PR-7 | S3 | `EXPERIMENTS.md` on the pod matches no commit in 79 revisions | pod copy 184,083 B / 1413 lines, mtime 2026-09-03T20:15:56Z; main 228,542 B / 1473 lines. `diff`: pod header says `209 runs, 52 completed`, main says `220 runs, 60 completed` | Not pod-only work: the pod copy is strictly **older** and its two `running` rows for `b0_sd_unlooped`/`b0_sd_looped` were later rewritten with results on main. A stale push mid-edit, not a divergent authority. Harmless today because nothing on the pod reads it |
| PR-8 | S2 | 26 (fact id, checkpoint) references exist across `facts/*.json` | existence test on the pod at 03:5xZ | **16 present, 10 absent.** Two of the ten are **my parser's artifacts**, not real: `ckpt_params_leg_438m_3p76b.pt{'statistic':` and `...step{2500` come from splitting a brace-expansion string and a nested dict, so the true absent count is **8**. The 8 overlap the `ckpt_facts_sources_present` WARN already in the harness — this is not a new discovery, it is a second instrument agreeing |
| PR-9 | S2 | `runs/pod_ckpt_candidates_2026-09-04.txt` line 15 records the inode-pin mechanism (§162): a KEEP line in a text file is invisible to the roller, which exempts by inode | pod: `st_ino` of every `*.pt*`, cross-referenced against names containing `.milestone_`; selector `*.pt.step*` minus `.interrupt` = 153 (b0's `ckpt_*.step[0-9]*` = 149; the difference is the bare-prefix `ckpt.pt.stepN` files) | **153 rolling `.stepN` files, 12 inode-pinned, 141 unpinned.** Of the 141, exactly **one is named as the source of a live fact**: `ckpt_pretrain_15b_s1.pt.step15000` → `be.adjacent_checkpoint_jitter`. Its run is **not active** (see PR-10), so no roller will rotate it today — the exposure is latent, not live. The other 140 belong to finished runs no fact cites |
| PR-10 | S3 | — | `ps -eo pid,etimes,args` in the container: pid `1238204`, **255,245 s = 2.95 days**, `tail -f runs/ms_ckpt_pretrain_15b_s1.pt.step15500.log` | An orphaned `tail -f` has held a file descriptor since 2026-09-01. Harmless (no GPU, negligible CPU, the log is not large) and **not killed** — reported per the audit's no-kill rule. It is the reason a naive `ps | grep pretrain_15b_s1` reads as "that run is alive", which is what made PR-9's exposure look live before the cmdline was read |

## 5. What this audit has NOT checked

Named, not silent:

- **`runs/pod_ckpt_candidates_2026-09-04.txt` vs the inventory — DONE, and it found nothing
  new.** All 37 KEEP-claimed names were matched against the 03:52Z listing: 3 are absent
  (`ckpt_p200m_4b_0902.pt.step500`, `.pt.step1000`, and a third that turned out to be my
  regex clipping `ckpt_pretrain_30b_s2.pt.step17500` to `.pt`). The two real losses are
  **already recorded on line 8 of that same file** — e1 retired the claim on 2026-09-03
  after verifying by `stat`. A second instrument agreeing, not a discovery.
- **KEEP pins by inode — DONE**, see PR-9. What was not done: verifying that each of the
  12 pinned inodes has `nlink ≥ 2` and that its milestone partner is the file the claim
  names, rather than a same-inode file with an unrelated name.
- **`runs/claims/`** is **pod-only** (`/work/aupai/runs/claims/`, one file
  `b0_se_16lnew_1b.json` at 02:31Z); it does not exist in the repo, because `pod_push.sh`
  excludes `runs/`. My earlier read failed against the local path, which was my error, not
  a missing directory.
- **The 168 untracked pod files** were classified as a group by directory, not read
  individually. Some are under `tmp/` and `facts/_raw/` and may matter to 44's area.
- **Orphan processes beyond GPU holders.** One found by accident (PR-10); no systematic
  sweep of CPU-only orphans, held file descriptors, or stale locks was run.
- **`/work/tl-ab` itself** — its git state, its disk share, and whether the job on card 7
  writes anywhere inside `/work/aupai`.
- **The vanished PID.** `nvidia-smi` listed pid `2878900` (40,240 MiB) at 03:49:0xZ and it
  was gone from the very next query seconds later. Not traced; it may have been a normal
  exit or a second `/work/tl-ab` process.

## 6. Open questions for the controller

1. Card 7 runs a `/work/tl-ab` job **inside our container**, started 22 min after the stop
   order. Is that the user's own work (leave it), or does the stop order cover it?
2. `card_claim.py` decides "foreign" from prose markers in a JSON note (PR-2). Should the
   fix — a cgroup/namespace check — be filed for after the audit, or is the note simply
   corrected to say "our container, another repository"?
3. PR-8's 8 genuinely-absent fact checkpoints duplicate an existing harness WARN. Does
   this area own re-deriving them, or does 44's facts audit?
4. `/work` is at 96% with 90 GB free while a 44.5 GB non-roster job runs. Does the 12:03Z
   prune proceed as planned regardless, or should its freed space be measured first?
5. PR-9's one latent exposure (`ckpt_pretrain_15b_s1.pt.step15000`, unpinned, sourcing
   `be.adjacent_checkpoint_jitter`) is safe only while that run stays dead. Pin it now, or
   leave it and accept that a resume of `pretrain_15b_s1` costs that fact its source?

## Pair check (b0, 2026-09-04)

PR-9, PR-4 and PR-6/7 as assigned. Every figure read on the pod in this pass, not taken from the
report's numbers. **PR-9 and PR-4 hold exactly. PR-6/7 I could not reproduce — my count differs and
the population rule is not published, so neither number is checkable yet.**

### PR-9 — HOLDS, every number
Independent enumeration: **153** plain `.stepN`, **12** sharing an inode with one of the 22
`*milestone*` files, **141** unpinned. Identical to PR-9. `ckpt_pretrain_15b_s1.pt.step15000` read
`ino=84187898 nlink=1`, 959,429,841 B — unpinned, as claimed. **Now pinned** at tilerl's request:
`ckpt_pretrain_15b_s1.milestone_keep_tilerl_adjacentjitter.pt`, same inode 84187898, nlink 1 → 2,
zero extra bytes.

A GLOB RECONCILIATION worth recording, because the next reader will hit it: `ckpt_*.step[0-9]*`
returns **149**, not 153. The nine `ckpt.pt.stepN` files under the bare prefix are the difference.
153 is the right population — the roller globs `ckpt_path + ".step*"` and `ckpt_path` varies per
run — but a reader reaching for `ckpt_*` gets 149 and concludes PR-9 overcounted.

### PR-4 — HOLDS
`df -h /work` → `2.0T 1.9T 90G 96%`, raw `93331444` KiB available = 89.0 GiB, matching the 03:53Z
reading. (88G after the pin above and arm 1's 669 MB checkpoint — still 96%.)

### PR-6/7 — NOT REPRODUCED, and the population rule is why
In-scope code directories plus the repo root, extensions `.py/.sh/.md/.json/.txt`: **538 walked,
130 outside the manifest's 481 entries.** PR-6 says 197 outside 469. The 130 are dominated by
untracked scratch — `_b0_*.py`/`_b0_*.sh` (nine of them MINE, from Stage D probes),
`ab_launch*.sh`, `arith_v*.sh`, `bench_eff/ddp_trace_rank[0-6].json`, plus `CLAUDE.md`,
`EXPERIMENTS.md`, `README.md`.

Three candidate causes, indistinguishable from here: whether the walk includes `runs/` and `data/`
(mine excludes them — the manifest omits them by design since the pod writes rows there; including
them gives 506 or 6442 depending on extension set), whether the extension sets differ, or whether
PR-6 counts only files TRACKED in main while mine counts everything on disk the manifest does not
name. **"197 files sit outside the manifest" is not checkable without the population rule beside
it** — the charter's principle 3 applied to this audit's own reports rather than to the code.

The 469-vs-481 entry count is probably mine to explain: the manifest is rewritten by every
`pod_push.sh --all` and I ran one at 03:21Z that added `scripts/b0_se_launch_arm2.sh`.

### A defect in MY first attempt, recorded because it is this audit's recurring shape
My initial manifest parser took `line.split()[-1]` as the path — the format is
`sha  path  category  mode`, so it read the mode column and found **2 entries in a 481-line file**.
Printing the entry count is the only reason "529 files outside a 2-entry manifest" did not become a
published finding. Four enumeration defects across three auditors in this audit now (58's
`json|jsonl` truncation, my `A/B`-as-path tokeniser, my number-regex boundaries, this) and zero
checking defects.

## Pair check (tilerl, 2026-09-04) — reply to b0, and b0's model_training.md recomputed

### PR-6/7: b0's first hypothesis is right, and the counts close

b0 listed three indistinguishable causes. It is the first one: **whether the walk includes
`data/`.** My rule, which the report should have carried and did not:

> pod-side `find`, files **on disk** (not "tracked in main"), extensions `.py .sh .md
> .json` — **no `.txt`** — excluding only `./data/corpus/*`, `./data/raw/*`, `./runs/*`,
> `./.git/*`, and every `__pycache__`. Note what that does *not* exclude: the rest of
> `data/`.

Measured against one manifest revision:

| | walked | outside manifest |
|---|---|---|
| rule A as published | 624 | **197** |
| rule A minus everything under `data/` | — | **135** |
| b0's rule B | 538 | **130** |

**62 of my 197 are under `data/`** — `data/hf/{SmolLM2-360M,SmolLM2-135M,Qwen2.5-0.5B,pythia-160m}`
(model config/tokenizer json), `data/vocab_sweep/`, `data/controls/`. b0's walk never
enters them; the manifest omits them by design. Removing them lands on **135 against b0's
130**, and the residual 5 is the extension set — b0 counts `.txt`, I do not. So the two
numbers were never in conflict: they differ by two named populations, both now stated.

b0's third hypothesis (tracked-in-main vs on-disk) is **not** a cause here: my classifier
walks everything on disk and then classifies against main, so untracked files are counted,
not filtered.

### The manifest revision, which neither of us named

469 lines when I read it (mtime 03:50:57Z), 481 for b0, **482** now
(sha256 `99f3f081bc49d5ce48e9d4276bed2b57974039d8f394bab282e7cc4deea36d99`, mtime
04:17:39Z). `pod_push.sh --all` rewrites it on every full push — b0 identifies their 03:21Z
push as the 469→481 step. Every "N outside the manifest" is a claim about a revision, and
until this table none of ours named one. **That is the finding; the arithmetic was never
the problem.**

### Two defects in my own reconciliation

Recorded because this audit keeps finding the same shape in its own instruments:

- My first reconstruction of b0's walk ran `find ./docs -printf '%P\n'`, which prints
  **relative to `./docs`** — so every path came out `standards/…` instead of
  `docs/standards/…`, matched nothing in the manifest, and produced a confident **510
  outside**. It looked like a finding for as long as it took to check one path.
- I suspected `cut -d' ' -f3` was mis-parsing the two-space-separated manifest and
  re-derived with `awk '{print $2}'`. Identical counts. The suspicion was wrong; recorded
  as checked-and-negative rather than dropped.

### PR-9 pin — confirmed done by b0

`ckpt_pretrain_15b_s1.milestone_keep_tilerl_adjacentjitter.pt`, inode 84187898, nlink 1→2,
zero extra bytes. `be.adjacent_checkpoint_jitter` now has an inode-protected source.
b0's glob reconciliation is adopted into PR-9's evidence column above.

### My recompute of b0's model_training.md: MT-1, MT-4, MT-2 — all three CONFIRMED

| finding | verdict | what I did |
|---|---|---|
| MT-1 | **CONFIRMED, and dated** | `/work/tilerl/src/tilerl/cli.py:57` reads `num_blocks=max(256, (ctx * 8) // BLOCK_TOKENS)`, `kv_cache.py:19 BLOCK_TOKENS = 16`. Computed: ctx 2048→1024 blocks, 4096→2048, 8192→4096 — **exactly 8.0× the declared budget at every context**, against the fact's "HALF". b0 could not reach that repo's history; I can, and it adds provenance: `git log -S"num_blocks=256"` dates the hardcoded value's removal to **`48ae458`, 2026-09-02 16:29:50 +0800** ("fix(cli): the token budget follows the model's context"), whose parent holds `num_blocks=256` at `cli.py:72`. So the fact was **correct when measured 2026-08-30** and was superseded three days later without being updated — a different defect from "wrong", and the one worth recording. One correction to b0: the fact cites `cli.py:66` / `kv_cache.py:37`, not `:57` / `:19`; the lines moved with the code, which is itself the argument for citing a commit rather than a line. |
| MT-4 | **CONFIRMED as written, and fixed since** | `shape_key()` at `scripts/launch_tests.py:95` now yields `scripts/test_arch_L32.py@d768L12h6f2304` vs `@d768L16h6f2304` — **distinct**, so two arms can both hold a certificate. File mtime **04:01Z, after b0's report**: the defect was real and is repaired. The live `runs/launch_tests.json` still holds two **bare** keys, both L12 (recorded 03:10 / 03:14); the L16 row b0 watched get erased is still gone. |
| MT-2 | **CONFIRMED** | `runs/trace_p200m_3step.json` — **59,446,282 bytes on the pod, absent from this checkout**, b0's figure to the byte. Cited by **4** fact ids: `eff.step_class_breakdown_p200m_4card`, `eff.step_roofline_p200m_4card`, `eff.optimizer_step_gpu_cost_p200m`, `eff.clip_and_sync_cost_p200m`. The S2 framing is right — a 57 MB artifact that cannot be pulled is a constraint needing a decision, not an oversight needing a fix. |

**A false positive I nearly filed on MT-4.** `rows_for()` returns the L12 row for an L16
query through its bare-path fallback, which reads exactly like the defect surviving its own
repair. It is not: `launch_gate.py:347-350` re-compares `row["shape"]` against
`LAUNCH_SHAPE`. Negative control run — an L16 query against the live file yields
`differs: ['layers']`, so the gate flags it. The fallback is safe **because a second check
exists**, not because the lookup is exact, and a reader of `rows_for` alone would conclude
otherwise.

**Disclosure on MT-1.** I am not a disinterested party: `/work/tilerl` is my own project's
repository, the KV pool code is mine, and `48ae458` — the commit that supersedes the fact —
is my own. I recomputed the arithmetic from the file rather than from memory of having
written it. Discount accordingly.
