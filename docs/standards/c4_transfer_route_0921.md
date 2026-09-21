---
question: How do the 32 surviving rp1t_c4 raw files get from the digest machine into the pod container, when no direct ssh route exists in either direction?
status: measured
source: measured 2026-09-21 on the laptop, digest (n37-112-238) and pod arle; sha256 compared at each hop
---

# Moving the rp1t_c4 raw bytes onto the pod

`en_c4_stage2_dc` is the one gate domain whose source cannot be re-fetched: `data.together.xyz`
returns 403 at the host level (verified again 2026-09-21 from the pod itself, not only from
digest). The 32 surviving `rp1t_c4` raw files are the only bytes consistent with the historical
corpus fingerprint, so they have to be physically moved. This page is the route that works.

**32, not 34: count `*.jsonl`, never the directory entries.** `ls | wc -l` in `$SRC` returns
**34** because the directory also holds `fetch_stats.json` and `fetch_stats.log`; `ls *.jsonl |
wc -l` returns **32**. The 32 jsonl files total **27,055,283,205 B (25.197 GiB)**; summing all 34
entries gives 27,055,291,175 B, and `du -sb .` gives 27,055,295,271 B because it adds the
directory's own 4,096-byte block. All three round to 25.197 GiB. The verification command at
the end of this page uses the `*.jsonl` glob and prints 32, which is the number that must match
— a reader who counts directory entries will conclude two shards are missing.

## Why the obvious routes do not

| route | result | what it actually means |
|---|---|---|
| pod container → digest | unreachable | different network segment (`/dev/tcp/<digest>/22` unreachable from the container) |
| pod host → digest | unreachable | same machine as the container; same IP, same result |
| digest → pod host via ssh | **not tested** | the address one reaches for here, `n37-112-238.byted.org`, **is digest itself** (`hostname -f` on digest returns it). An `ssh` to it is a self-login and its `Permission denied` says nothing about the pod |
| `tn push` (sftp) from the laptop | `sftp: "Failure" (SSH_FX_FAILURE)` on every destination tried | sftp is unusable against this host; **and `tn push` exits 0 on it**, so rc is not the signal |

The last row is the one worth remembering: `tn push` reported failure on stdout and returned
**0**. Anyone gating on the exit code would record a transfer that never happened.

## The route that works

Two hops, sha-verified at each one. `/work` in the container is a hostPath onto the pod host's
`/data00/aupai_work`, so a write that lands on the host is immediately visible in the container.

```bash
SRC=/data00/home/chenkailun.c/aupai-cimap/wt-3b/data/raw/rp1t_c4
DST=/data00/aupai_work/aupai_c4        # host path; container sees /work/aupai_c4

# every file, in a loop, one at a time (see the rate note below before parallelising)
for f in $(ssh digest "ls $SRC"); do
  # HOP 1 -- digest to the laptop, streamed
  ssh digest "cat $SRC/$f" > /tmp/gate_transfer/$f
  # HOP 2 -- laptop to the pod host; tn write streams stdin and renames atomically
  cat /tmp/gate_transfer/$f | tn write $DST/$f
  # verify BOTH halves, per file, before moving on
  a=$(ssh digest "sha256sum $SRC/$f" | cut -d' ' -f1)
  b=$(shasum -a 256 /tmp/gate_transfer/$f | cut -d' ' -f1)
  c=$(~/bin/pod "sha256sum /work/aupai_c4/$f" | cut -d' ' -f1)
  [ "$a" = "$b" ] && [ "$b" = "$c" ] || { echo "MISMATCH $f"; break; }
  rm -f /tmp/gate_transfer/$f
done

# then, in the container, confirm all 32 against the digest-side list
~/bin/pod "cd /work/aupai_c4 && sha256sum *.jsonl | sort > /tmp/pod_sums.txt; wc -l < /tmp/pod_sums.txt"
ssh digest "cd $SRC && sha256sum *.jsonl | sort > /tmp/digest_sums.txt; wc -l < /tmp/digest_sums.txt"
```

## What to check, and what not to trust

- **`tn write`, not `tn push`.** `write` streams stdin and does a temp+rename, so a reader sees
  the old file or the new one and never a half-written one. `push` is sftp and fails here.
- **`pod "cat > file"` writes 0 bytes and returns 0.** stdin is not forwarded through `pod`
  (known shape, recorded in AGENTS.md). The file that results has the sha of the empty string,
  `e3b0c442...`, which is how it was caught the first time. Use `tn write`, or verify by sha.
- **`tn push`'s source path must be local to the laptop.** A digest path is stat'd on the pod
  host and reports `no such file or directory` — the error names a path that was never in
  scope, not a missing file.
- **Per-file sha, all three sides.** HOP 1 and HOP 2 each fail independently; a single
  end-to-end check after the loop cannot say which hop corrupted a byte.

## Rate

Both hops measured separately on 2026-09-21, laptop → digest → pod host:

| hop | measured | size |
|---|---|---|
| 1 — digest → laptop (`ssh digest "cat file"`) | **10 MiB/s** | 64 MiB in 5 s |
| 2 — laptop → pod host (`tn write`) | **0.44–0.53 MiB/s** | 64 MiB in 146 s; 16 MiB in 29 s |

**Hop 2 is the bottleneck by ~20×**, and the hop-2 points are close (16 MiB → 0.53, 64 MiB →
0.44, and an independent 32 MiB → 0.65), so the rate is **roughly constant, not degrading** —
unlike the finemath fetch, a serial total here is a sum of like-sized terms.

### The transfer must fit a per-stream timeout — chunk it

**A `tn write` stream is killed 300 s after it starts, at any speed.** Measured directly with a
single 250 MiB stream: it died at `held connection: no response for 5m0s` with `rc=1` and
`elapsed=300s` exactly, having landed **172,163,072 B = 164.2 MiB** (0.547 MiB/s that run).

**Cut by SECONDS, not by bytes.** A byte ceiling is not a constant — it is `rate × 300 s`, and
the rate moves (0.44–0.65 MiB/s across runs). A "196 MB limit" computed from one run's rate
would send an oversized chunk the moment the link slows. Budget **240 s** (60 s of margin) and
convert at the rate you are actually seeing; re-measure per chunk rather than trusting a number
from an earlier one.

### `tn write` FAILS OPEN on timeout — verify every chunk by hash

This is the part that costs a silent corruption if it is missed. After the 250 MiB stream died
at the window, the host held:

```
big.bin       172,163,072 B     <- the FINAL name
*.tn-tmp      none              <- no leftover temp file
head -c 1MB   identical to the source's first 1MB   <- a true prefix of the source
```

**A truncated fragment was renamed into place under the correct name.** `tn write`'s atomic
temp+rename was not violated — a truncated file *is* a valid new file under that contract. The
result is a file that exists, is non-empty, has correct content as far as it goes, correct name,
and the wrong length. **`rc=1` is the only external signal.**

```bash
# per chunk: transfer, then hash IMMEDIATELY -- never gate on rc alone, never on file existence
cat "$chunk" | tn write "$DST/parts/$(basename "$chunk")"; rc=$?
for try in 1 2 3; do
  a=$(sha256sum "$chunk" | cut -d' ' -f1)
  b=$(tn exec "sha256sum $DST/parts/$(basename "$chunk")" | cut -d' ' -f1)
  [ "$rc" = 0 ] && [ "$a" = "$b" ] && break
  cat "$chunk" | tn write "$DST/parts/$(basename "$chunk")"; rc=$?
done
[ "$a" = "$b" ] || { echo "CHUNK UNRECOVERABLE: $chunk"; exit 1; }
```

**A chunk that dies at the window is the normal case, not an error** — the chunk boundary is
what makes the retry cheap. **And the final whole-shard sha is the last line of defence, not a
formality:** if a truncated chunk ever reaches the concatenation, a mismatch there is the only
thing that catches it.

### Throughput: per-stream, and the probe caveat

Four concurrent 8 MiB writes finished in 12 s = **2.46 MiB/s aggregate** against 0.65 MiB/s
single-stream, **3.8×** — the limit is per stream, not per host, so concurrency is still the
right lever (each stream carries its own 300 s window).

**But that number was measured in the probe interval and does not transfer to real payloads.**
8 MiB at 12–13 s never approaches the window, so it measures throughput while the window is
irrelevant. **A probe answers "is the route up"; it does not answer "will this payload finish".**
The speed-up for chunked, real-sized payloads is **unmeasured**.

The measured figures are also in `facts/corpus_supply.json#cs.c4_transfer_route_0921`; read them
there rather than from this page, which would go stale.

## Scope

This moves **bytes only**. It does not build the corpus, decontaminate it, or produce a token
cache — those are `docs/standards/data_pipeline_rebuild_0916.md` §2b onward. The destination
`/data00/aupai_work/aupai_c4` is raw input; the built domain is `data/corpus/en_c4_stage2_dc/`,
which must be a new directory with its own build stamp (a corpus domain carries the fingerprint
of what produced it).
