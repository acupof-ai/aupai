# Incremental batch jobs

Any batch job expected to run more than 10 minutes must:

1. Save each shard's output to its own file as soon as that shard finishes.
2. On startup, skip shards whose output file already exists.

Criterion: kill the job at any point and restart it; at most one shard of work
is lost.

Incident: `datagen/train_quality_head.py` saved once with `np.save` after all
shards finished. A two-hour job at 50% had nothing on disk — neither "use the
finished half" nor "add cards and parallelize" was possible. The script was
left running; this rule exists so the next batch job does not repeat it.

Pattern: one output file per shard (`out.{shard:03d}.npy`), written inside the
shard loop, not after it. A merge step, if needed, reads the shard files and is
itself rerunnable.
