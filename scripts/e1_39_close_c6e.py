"""Append e1-39: C6e delivered. One append, no rewrite.

# restartable: the only write is a single append to runs/tasks.jsonl at the end. A re-run refuses
# if the id is already present, so appending twice is impossible; nothing here touches a card, the
# pod, or a checkpoint.
"""
import json
import os
import subprocess

ROOT = "/Users/bytedance/code/aupai-e1"
TASKS = os.path.join(ROOT, "runs", "tasks.jsonl")
NOW = subprocess.run(["date", "-u", "+%Y-%m-%d %H:%M"], capture_output=True,
                     text=True).stdout.strip()

rows = [json.loads(l) for l in open(TASKS, encoding="utf-8") if l.strip()]
# THE HIGHEST ID OVER EVERY LINE, not over the folded latest-per-id view: the fold hides ids whose
# newest row is `done`, which is how e1-31/e1-32 collided and the harness refused them.
used = {r["id"] for r in rows if r.get("id")}
assert "e1-39" not in used, "e1-39 already exists; refusing to append a second row"

ROW = {
    "id": "e1-39",
    "owner": "e1",
    "socket": "uds:/tmp/cc-socks/56482.sock",
    "pair": "3b",
    "prior": "defect-fix",
    "state": "done",
    "opened": NOW,
    "closed": NOW,
    "task": "C6e (6e's ruling): eval/l1_fewshot.py writes an identity header as row 0 of every "
            "predictions file, and score_matrix.metric_l1_fewshot transcribes an existing artifact "
            "only when that header's checkpoint sha256 matches the file on disk. No header or a "
            "mismatch records 'artifact unverifiable, not re-run'; --force stays an operator flag "
            "and is never passed by the metric. Plus ruling (4): the .REFUSED sidecars get a line "
            "in eval/README.md and their presence is recorded in the row.",
    "why": "A complete 497-row L1 result at acc 0.0181 sat in data/eval while its score_matrix row "
           "said ERROR, and could not be published because the rows carry q/gen/ok only -- so "
           "filename-and-mtime was the whole identity, and this repo has rewritten same-named "
           "checkpoints. Two audits (E18, E21) read the seven ArtifactExists rows as failures "
           "before anyone opened the files.",
    "reading": "a matching artifact transcribes with from_artifact=true; a rewritten checkpoint or "
               "a header-less file refuses with a named reason; the transcription rescores from the "
               "rows with the script's own scorer rather than trusting the row's ok= field",
    "result": "DELIVERED at 29b31367. Identity is the checkpoint FILE's sha256, not the cfg's "
              "vocab_id -- vocab_id is None on exactly the checkpoints that need it "
              "(ckpt_b0_sd_equalcompute.pt is one), so a vocab_id identity compares None to None "
              "and matches on the whole population. The filename rule moved into "
              "l1_fewshot.artifact_path(), one implementation, because the verifier needs the name "
              "the writer builds; NOT called preds_path, because that identifier is "
              "harness.check_reported_path_is_written's marker for the stale-variable defect and a "
              "correct call to a function of that name is read as the defect (the check matches the "
              "ast.Name, not its use) -- it refused the first commit. 13 mutations verified RED AT "
              "THEIR OWN ASSERTION, 6 on score_matrix and 7 on test_l1_fewshot_2x2; three of my "
              "first six were red for the wrong reason (a crash or a neighbouring case) and were "
              "rewritten. Two defects found in my own code by mutating: verify_artifact's refusal "
              "named '--ckpt is a directory', a cause it had not checked (ckpt_sha256 also returns "
              "None for an absent path); and removing the empty-artifact refusal crashed on "
              "binomial_delta, not acc -- that one line stands in front of THREE divisions by "
              "total, and guarding them individually would turn an artifact that measured nothing "
              "into acc 0.0 / delta 0.0 / present 0.0. Group 6 of test_l1_fewshot_2x2 rewritten "
              "from source-text greps to calling the function: the four old checks went red when "
              "the path moved, on a file whose behaviour was identical. The equalcompute number "
              "stays UNTRANSCRIBED per ruling (2) -- the real artifact is still header-less, so it "
              "takes the refuse branch, verified on the pod.",
    "evidence": "29b31367 (eval/l1_fewshot.py, eval/score_matrix.py, eval/l1_2x2_diagnose.py, "
                "eval/test_l1_fewshot_2x2.py, eval/README.md); "
                "runs/audit_0904/eval_heldout.md (E18, E21, E22)",
    "note": "RULING (4) CORRECTION, for de's disk inventory: the .REFUSED sidecars are NOT "
            "design-only. open_artifact writes them beside the ARTIFACT, so they live in "
            "data/eval/, never in runs/ -- `find /work/aupai/runs -name '*.REFUSED*'` returns 0 "
            "and says nothing about whether any were written. Six exist on the pod, all under "
            "/work/aupai/data/eval/ (2026-09-02 06:17 to 2026-09-04 01:13). There are 6 for 7 "
            "refusal rows because the sidecar is named after the artifact: before 5a989647 put the "
            "checkpoint into the artifact name, six refusals on six checkpoints shared one "
            "sidecar. Recorded in eval/README.md and the metric's docstring.",
}

with open(TASKS, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(ROW, ensure_ascii=False) + "\n")
print(f"appended e1-39 done at {NOW}")
