---
question: How do the 34 surviving rp1t_c4 raw files get from the digest machine into the pod container, when no direct ssh route exists in either direction?
status: measured
source: measured 2026-09-21 on the laptop, digest (n37-112-238) and pod arle; sha256 compared at each hop
---

# Moving the rp1t_c4 raw bytes onto the pod

`en_c4_stage2_dc` is the one gate domain whose source cannot be re-fetched: `data.together.xyz`
returns 403 at the host level (verified again 2026-09-21 from the pod itself, not only from
digest). The 34 surviving `rp1t_c4` raw files are the only bytes consistent with the historical
corpus fingerprint, so they have to be physically moved. This page is the route that works.

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

# then, in the container, confirm all 34 against the digest-side list
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

**A `tn write` stream is killed after a fixed window, independent of speed.** Measured
2026-09-21: four concurrent streams each died at ~196 MB with `held connection no response for
5m0s`, and 300 s × 0.65 MiB/s = 196 MB — the arithmetic matches, so **the 5-minute window is
what ended them, not a rate collapse.** A real 843 MB shard at 0.65 MiB/s needs ~21.6 min and
therefore **cannot be sent as one stream at all.**

**This invalidates the parallel figure below for real work.** The 3.8× was measured with 8 MiB
probes finishing in 12–13 s, which never come close to the window. **A probe answers "is the
route up"; it does not answer "will this payload finish".** The two intervals are different, and
a number from one does not transfer to the other.

**The working shape is: cut each shard into ~80 MB chunks, stream each chunk, concatenate on the
host, then verify.**

```bash
# per shard, per chunk: ~80 MB is ~125 s at 0.65 MiB/s -- inside the 5-minute window
split -b 80m shard.bin /tmp/chunk_
for c in /tmp/chunk_*; do
  cat "$c" | tn write /data00/aupai_work/aupai_c4/parts/$(basename "$c") || exit 1   # 3 retries
done
# concatenate on the pod HOST (not in the container: /work is the same filesystem)
tn exec "cd /data00/aupai_work/aupai_c4/parts && cat chunk_* > ../shard.bin"
# then the three-way sha, as below
```

**Retry each chunk independently**, up to 3 times. A chunk that dies at the window is the normal
case, not an error — the chunk boundary is what makes the retry cheap.

### Throughput: per-stream, and the probe caveat

Four concurrent `tn write`s of 8 MiB each finished in 12 s = **2.46 MiB/s aggregate** against
**0.65 MiB/s single-stream, 3.8×** — the limit is per stream rather than per host. **Read that
number with the paragraph above: it was measured in the probe interval and says nothing about
whether 4 parallel streams of 80 MB chunks complete.** Parallelism is still the right lever
(each stream carries its own window, so more streams carry more bytes per window), but the
speed-up for real payloads is **unmeasured**.

The measured figures are also in `facts/corpus_supply.json#cs.c4_transfer_route_0921`; read them
there rather than from this page, which would go stale.

## Scope

This moves **bytes only**. It does not build the corpus, decontaminate it, or produce a token
cache — those are `docs/standards/data_pipeline_rebuild_0916.md` §2b onward. The destination
`/data00/aupai_work/aupai_c4` is raw input; the built domain is `data/corpus/en_c4_stage2_dc/`,
which must be a new directory with its own build stamp (a corpus domain carries the fingerprint
of what produced it).
