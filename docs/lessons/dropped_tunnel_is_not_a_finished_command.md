---
question: Why did free disk space keep falling for ten minutes AFTER the delete reported success?
status: measured
source: pod overlay df, 2026-09-03 06:40-07:05Z; host ps/proc for pids 176214/176232/176263/176276
---

# A dropped tunnel is not a finished command

Running a 3-step smoke test on the pod, the `tn` connection dropped:

```
tn: held connection: no response for 5m0s (daemon crashed or SSH dropped)
[exited with code 1]
```

That exit code is **tn's**. The command it carried — `crictl exec … bash -lc
'cp -a /work/aupai/. /tmp/absmoke/ && … train.py …'` — kept running inside the
container for another ten minutes, because nothing about the tunnel dying
signals the remote process.

The disk was full by then (`overlay 2.0T 2.0T 0 100%`), so `rm -rf
/tmp/absmoke` went out. It exited 0. And free space kept falling:

```
167G -> 164G -> 161G -> 160G -> 156G
```

**A successful delete cannot be followed by shrinking free space.** The `cp`
was still writing into the directory the `rm` was walking. Only after finding
pid 176276 on the host (`ps -eo pid,args`, cmdline read and confirmed to
contain `absmoke`) and killing it by exact pid did the number hold still; a
second `rm` cleared the 13 GB it had re-created.

## The mirror image of a rule that already existed

`AGENTS.md` already says: a long job started with `pod "<cmd>"` in the
foreground dies with the tunnel while **the container process keeps running as
an orphan**, so long jobs must `setsid` detach.

That is the same mechanism, stated for the launch side. This is the teardown
side of it, and the launch-side rule does not imply the teardown-side one — I
had read the first and still read `[exited with code 1]` as "the command
ended".

## What made it expensive rather than merely wrong

The command was `cp -a /work/aupai/. /tmp/absmoke/` — copying a training tree
copies its **dataset**: 206 GB, filling a shared 2 TB disk while another
session's training job was mid-run and due to checkpoint. What the smoke test
actually needed was two files. The tree it needs now is symlinks plus the two
files under test: **204 KB**.

## Rules

**A local exit code describes the local process.** For anything sent through a
tunnel, "did it finish" is answered by looking for the process on the far side,
not by reading the status the tunnel returned.

**A cleanup that does not converge has not finished.** Free space that keeps
falling after a successful `rm`, a count that keeps rising after a successful
kill — the delete is racing a writer that is still alive. Re-issuing the
cleanup does not help; find the writer.

**Copy the files under test, not the tree that contains them.** `cp -a` on a
training directory takes the corpus with it.

Corollary, same class, same session: `podput <local> <remote>` does not
validate that `<remote>` is absolute. `podput model.py model.py` wrote into the
container's default cwd — a different project's source tree.
