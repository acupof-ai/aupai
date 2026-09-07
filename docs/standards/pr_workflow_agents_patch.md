# AGENTS.md Coordination patch — the gh PR workflow

Written by de 2026-09-07 for 4c to apply under the claim. **I am not editing `AGENTS.md` myself
here**: it is a symlink to `~/.claude/CLAUDE.md`, the global instructions this session runs on, and
a peer ruling is not authority to edit my own instructions. The mechanism (`harness claim-file`)
would let me; the reason not to is that the file is the user's, not the round's.

Insert after the bullet beginning **"Each session works in its own worktree on its own branch"**,
in the Coordination section.

---

- **Code goes through a GitHub PR; ledger-only commits keep `merge_main` (user order 2026-09-07).**
  A commit touching anything outside `runs/*.jsonl` and `EXPERIMENTS.md` is code:

  ```bash
  git push -u origin HEAD
  gh pr create --base main --head <branch>
  # CI must be green on the PR's HEAD sha
  # your second reader approves ON THE PR, with `artifact:` or `case:` in the approval body
  # THE REVIEWER, never the author: gh pr merge --merge
  ```

  Ledger-only commits still merge with `scripts/merge_main.sh <branch>` — immediate, union-merged,
  CAS. That split is the ruling and not a shortcut: the union driver and the compare-and-swap are
  what make an append-only ledger mergeable without a person, and putting a review cycle in front
  of every experiment row would buy nothing and cost every measurement a round trip.

  Four consequences worth knowing before the first PR, each measured rather than assumed:

  - **`--merge`, never `--squash`.** tilerl squashes; we cannot. Three checks key on shas that must
    stay on main — `pod_drift`'s `data/pod_synced_head` stamp, `tasks_closed_by_commit`, and
    `main_advances_by_ancestry`. A squash gives the merged commit a new sha, so a close row or a pod
    stamp written before the merge names a commit main never holds. `--merge` keeps branch shas
    reachable and all three predicates unchanged.
  - **Approval is necessary, not sufficient.** The approval body must contain `artifact:` or
    `case:`. GitHub approval is a click; a review row names what the reviewer opened, and
    `review_present` FAILs 30 minutes after a close if it names neither. `review.jsonl` stays for
    rulings and non-code reviews; `scripts/review_row_lookup.py --pr` reads either source and
    either satisfies the gate.
  - **`merge_main` refuses post-flip code and prints these commands.** The refusal is per commit
    and by the commit's own committer date, so a branch carrying pre-flip code work still drains
    through `merge_main` — you are not asked to follow a rule that did not exist when you wrote the
    commit. `AUPAI_CONTROLLER=1` overrides, logged to `runs/friction.jsonl`.
  - **The pod push moves to the PR merger, in the same step as the merge (4c's ruling
    2026-09-07).** The existing rule — "a commit that touches a file in the manifest's scope is
    pushed to the pod by its committer in the same step" — cannot hold for a PR: at commit time the
    code is not on main yet, and the pod runs what main holds. So for CODE, whoever runs
    `gh pr merge --merge` pushes the pod in the same step and stamps main's sha; for LEDGER commits
    the committer still pushes, unchanged. The obligation moves with the act that puts the code on
    main, which is now the reviewer's, not the author's.

---

## What this patch deliberately does not say

- **No flip date.** `AUPAI_PR_FLIP_EPOCH` unset means the gate is inert, so the transition is a
  separate act. Naming a date here before the switch is thrown would make the document wrong for
  however long the gap is.
- **No claim that CI needed changing.** `.github/workflows/ci.yml:2` is already
  `on: [push, pull_request]`, so a PR's head sha gets CI today. Measured, not built.
- **Nothing about branch protection.** Whether GitHub itself enforces "the reviewer merges" is a
  repository setting, not a rule in this file, and I have not been asked to change repo settings.
  Until it is set, the ordering is a convention the PR gate does not check.

## Rule-coverage row

`agents_rules_covered` requires every new bullet to map to a check or an explicit manual reason.
Add to the Rule coverage table:

| Rule | Enforced by |
|---|---|
| Code goes through a GitHub PR; ledger-only commits keep `merge_main` | `merge_main.sh --selftest`'s nine pr-gate worlds (`_code_pr_gate`); the approval-body half by `scripts/review_row_lookup.py --selftest` |
| A push now happens AFTER the merge, not in the same step (4) | manual: the ORDER of two operator actions leaves no artifact recording which came first. `pod_drift --check` catches the consequence — a stamp naming a sha main does not hold reads as drift — but not the discipline |
