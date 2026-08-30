# Rebuilding a corpus domain

fineweb2 `web_hq` was lost on 2026-08-30 and could not be rebuilt: the shards were
gone from `/work`, `/data00`, and `data/raw`, and nothing recorded how they had been
produced. The 0.2b budget point trained on a substitute and was discarded. This
standard exists so the next loss costs a command, not a day.

A domain is rebuildable when three things are written down and one of them is checked.

## The three records

Every domain in a `data/mix_scale_*.json` has a block in `data/PROVENANCE.md`:

1. **Fetch** — the literal command, including `HF_ENDPOINT=https://hf-mirror.com`
   (huggingface.co is unreachable from the pod; the mirror and modelscope.cn are
   reachable, measured 2026-08-30) and `--revision <commit sha>`. Without a pinned
   revision the same command returns different data later, which is not a rebuild.
2. **Build** — the command chain from raw to shards, with every flag. For `web_hq`
   that is `build_corpus.py` → `score_corpus.sh` → `build_web_hq.sh --keep 0.40` →
   `clean_web.py`. A flag left out of the record is a flag that will be guessed.
3. **Result** — the `corpus_fingerprint.py` value, the document count, the byte count,
   and the date. This is what a rebuild is checked against.

## The check

`harness check` fails when a domain named by the default mix has no provenance block,
or when its recorded fingerprint does not match what is on disk. A rebuild is finished
when the fingerprint matches, not when the files appear.

## Rules

- **Download on the pod, never upload from a laptop.** A path that runs through one
  person's machine cannot be reproduced by anyone else. **`/work` is the only durable
  path inside the container.** It is a real ext4 mount (`/dev/vda2`, `st_dev` 65026);
  `/data00` *inside the container* is not a mount at all — it shares `st_dev` with `/`,
  so it is the container's own overlay and a restart erases it. The durable
  `/data00`–`/data03` NVMe live on the **host** and are not mounted in the container
  (`scripts/harness.py:45`). Both paths report the same free space, because the overlay's
  upper layer sits on the same underlying filesystem — **free space is not evidence of a
  separate disk**, and that is what fooled this document and the fetch target below.
  Verify a target with `os.stat(p).st_dev != os.stat("/").st_dev`, never with
  `disk_usage().free`. (`/work` itself is a Kubernetes emptyDir: it survives a container
  restart but not a pod deletion. Durable-to-durable it is not; durable enough for a
  fetch it is.)
- **Read `data/PROVENANCE.md` before fetching anything.** The hf-mirror workaround was
  recorded there on 2026-08-28 and independently re-derived, wrongly, on 2026-08-30.
  Knowledge that is not read is not knowledge.
- **A domain directory holds real files.** No symlinks, nothing pointing at a build in
  progress. A corpus under construction stays in its own directory and enters a mix by
  name.
- **Derived caches carry their source fingerprint.** `/data00/tokens_<domain>.pt` is
  reused whenever it exists, without looking at the source directory. On 2026-08-30 the
  source was swapped, the cache was rebuilt from the new source, and training reused it
  with nothing raising. A cache whose recorded source fingerprint does not match the
  directory must be rebuilt, not reused.
