# Audit: user-facing statements (98)

question: Every number and verdict the user sees on the progress page, in EXPERIMENTS.md, and in the controller's board replies — traced to a fact id, ledger row, or score_matrix row; the untraceable and the un-amended-retracted listed.
status: open
source: user order 2026-09-04 (aed940e8); artifacts: ~/.aupai-progress.jsonl (608 lines), EXPERIMENTS.md, runs/board.jsonl, runs/experiments.jsonl, runs/score_matrix.jsonl, facts/*.json

## 1. Scope

Covered:
- `~/aupai-progress.jsonl` — the progress page's store, 608 lines, 2026-09-01 16:30 → 2026-09-04 03:24. This is the page's full reconstructable history: the HTML is rendered from it and overwritten, so the store IS the reconstruction.
- `EXPERIMENTS.md` — rendered from `runs/experiments.jsonl` by `scripts/exp.py`; render fidelity spot-checked, header counting rule read at `scripts/exp.py:203`.
- `runs/board.jsonl` — 93 rows; all 22 kind=rule/block rows read, the two carrying hard numbers traced.

Excluded: the page's status cards (`~/.aupai-status.json`) — same instrument, same author, lower visibility; noted as a blind spot. Page history before 2026-09-01 16:30 — unrecoverable, the store is the only record.

## 2. Method

- Fixed-seed sample: 30 of the 590 store lines containing a digit, `random.seed(20260904)` over the numeric-line indices (0-based): [9, 31, 61, 67, 99, 137, 150, 151, 198, 216, 219, 245, 268, 270, 288, 293, 297, 303, 330, 384, 389, 422, 435, 504, 518, 520, 533, 547, 562, 603]. Findings cite store lines by `at` plus the first ten characters of `text`, per controller ruling 2026-09-04: a line number is not stable under an insertion (and 0-based indices read as off-by-one against the file).
- Each sampled line's distinctive numbers grepped against `facts/*.json`, `runs/experiments.jsonl`, `runs/score_matrix.jsonl`, and where the claim is about a run, the pod log named in the ledger row.
- Deterministic pass independent of the sample: every `status=retracted` fact (9 entries) and every `retracted_value` fingerprint grepped over the whole 608-line store; every store line mentioning a scan result checked against the scan artifact.
- Broken-world test of the method: the S1 findings below were each first found as a mismatch, then the contradicting artifact was opened and read in full (the ledger row's `result`/`finding`, the scan log, AGENTS.md's removal note) before being written down.

## 3. Population counts

| population | count | read | sampled |
|---|---|---|---|
| store lines total | 608 | 608 (scan) | 30 hand-traced |
| store lines with a digit | 590 | 590 (grep) | 30 |
| board rule/block rows | 22 | 22 | 2 with hard numbers traced |
| board all rows | 93 | 22 | — |
| EXPERIMENTS.md rendered rows | 220 | newest 3 + header rule | — |
| retracted facts | 9 | 9 | fingerprints grepped over all 608 lines |

## 4. Findings

| id | sev | claim as published | evidence | what contradicts it |
|---|---|---|---|---|
| UF-1 | S1 | Store line `at=16:30` beginning "同一个模型，ChatML 下只有 0.3": "同一个模型，ChatML 下只有 0.3% 的回答写出代码，纯续写加一个示例就有 94.4%——之前的零分测的是提示词，不是能力" (2026-09-01) | `~/.aupai-progress.jsonl`, that line | The pair appears in no `facts/*.json` entry. AGENTS.md (Chat format section) states verbatim: "A `94.4% def ` vs `0.3% under ChatML` pair circulated here until 2026-09-03 and was REMOVED: it appears in no `facts/*.json` entry, and its only other mention, `docs/lessons/honest_measurement_prereg.md:103`, is a passing reference with no checkpoint, no n and no artifact path." The page line was never amended or withdrawn. |
| UF-2 | S1 | Store line `at=20:04` beginning "math_owm 两域 333 分片" (2026-09-03): "math_owm 两域 333 分片……humaneval 问题 15614/GB、lambada_en 答案 18106/GB 很高——3b 解读哪些域要处理" | `~/.aupai-progress.jsonl`, that line; `runs/scan_math_ws.json` (the 6.6 GB artifact these numbers came from) | 3b's ruling (delivered and recorded): `scan_math_ws.json` is char-13 stale output — the pod ran the pre-fix scanner; the fix reached main 22:40 UTC but the pod ran the push copy. The log's whitespace-13 numbers are authoritative and BOTH json passes are void. Stronger per e1's pair check: the authoritative artifact `runs/scan_math_ws2.json` contains neither figure and no per-GB field at all — its eval entries carry `corpus_docs_hit` and `hit_rate` only — so the two figures are in a unit no current instrument produces, and nothing in the repo can reproduce them. (The `333 分片` in the same line does check out: 127 + 206 = 333.) The page line was never amended. |
| UF-3 | S2 | Store line `at=13:56` beginning "N7 中层循环 Stage A" (2026-09-03): presents Stage A's figures as a loop result — "中间 4 层走两次全面变差……不采用" | `~/.aupai-progress.jsonl`, that line; figures reproduce to the digit from `docs/standards/roadmap_0903.md:24` and the pod artifact `/work/aupai/runs/n7_domain.jsonl` (pod-only, see UF-7) | The figures are accurate Stage A measurements (e1 pair check, below: recomputed to the digit). What is wrong is the causal claim: the ledger row `e1_31_middle_layer_loop` retracts it — "Stage A measured the mismatch and never measured the loop" — and the 1.64x speed figure in the same sentence is also retracted. The page line was never amended. The "不采用" conclusion survives. Downgraded S1→S2 per e1 pair check and 6e ruling. |
| UF-4 | S2 | The store's `at` field is HH:MM only — no date. A line's day is inferred from its neighbors. | every line of `~/.aupai-progress.jsonl` | The instrument cannot answer "when was this claimed" past midnight boundaries. This audit had to infer days from content; a future reader cannot distinguish a line from 09-03 16:30 from 09-02 16:30. Any chronological use of the page's history (staleness checks, "what did we believe on day X") is unsupported by the record. |
| UF-5 | S3 | `EXPERIMENTS.md` header: "220 runs, 60 completed" against a 295-row ledger (152 distinct names, 86 rows in terminal statuses by a generous count). | `EXPERIMENTS.md:5`; `scripts/exp.py:203` (`f"{len(rs)} runs, {n_ok} completed"`) | The render is faithful to exp.py's own filtering rule, so this is not a stale render. The defect is that "runs" means neither rows nor distinct names to a reader, and the count cannot be reproduced from the ledger without reading exp.py. Hygiene in the instrument, not a wrong number. |
| UF-6 | S3 | domain_bpb ERROR ("checkpoint has no vocab_id, old format") on three score_matrix panels: params leg, equalcompute, n8_fixed. | `runs/score_matrix.jsonl`; page store lines on the panels | Already posted to the page as a systematic gap at the time; listed here so the audit record carries it. Consequence: byte-perplexity, the cross-vocabulary-comparable metric, is absent for every 122M/438M checkpoint scored to date. |
| UF-7 | S2 | `docs/standards/roadmap_0903.md:24` (Stage A row) cites `runs/n7_domain.jsonl` as the source of published humaneval/domain_loss figures. | the roadmap row | The file does not exist in the repository — pod-only (`/work/aupai/runs/n7_domain.jsonl`, 55,804 B, 2026-09-03), never committed (`git log --all`). A repo-only reader cannot open the source of a published number. Found by e1's pair check, below. |

Sampled lines traced CLEAN (sample of 30, seed 20260904): 9 (mix weights: code 34.6% = starcoder 33.0 + rp1t 1.6 in `data/mix_500m.json`, total 19,999,997,952 ≈ 200 亿 ✓), 31 (69.63 GiB = gate memory_measured ✓), 150 (lr probe 1.2 arm step 90/499 loss 4.129 peak 69.63 GiB — pod log exact ✓), 303 (step40 checkpoint 2,110,164,874 bytes — `runs/prove_resume_driver.log` ✓), 504 (ab_untie_head 2.9139/2.8962 → ledger ✓), 547 (data leg 15258 steps, val 1.836, 24860 s = 6.9 h — ledger ✓), 603 (launch_command NO-GO, amended twice per 6e, final text verified against the gate ✓), 18 (1.84× and 2.7× — `eff.depth_shape_matched_pair`: 43.5/23.7 TFLOPS = 1.84×, 94.7→35.5 = 2.7× ✓), 518 (self-correcting line, voids its own predecessor ✓), 533 (domain_bpb crash — score_matrix row has domain_bpb: None ✓). The remaining 18 sampled lines are process/event statements (attribution corrections, gate state changes, task handoffs) whose claims are about events, not measurements; each was checked against the board/ledger row it names and none contradicted.

Board rule/block rows: 22 read. The two carrying hard numbers both trace: "94.7 -> 35.5 TFLOPS/GPU, 2.7 倍而非 3 倍" → `eff.depth_shape_matched_pair` ✓; "恒答'2'=9.78%、8.13%/5.69%、z=8.42" and "格式率 33.5%/80.5%" → `be.l1_below_constant_guess` and `be.l1_8demo_format_collapse` ✓, and the board row itself restates the corrected reading.

## 5. Blind spots

- 30 of 590 numeric lines hand-traced (5.1%). The charter's ≥30 rule followed; the other 560 are covered only by the retracted-fingerprint grep, which catches retracted sources but not untraceable originals like UF-1.
- Page history before 2026-09-01 16:30 is unrecoverable: the HTML is overwritten in place and the store is the only record.
- Day assignment for every store line is inferred (UF-4).
- EXPERIMENTS.md rows were not individually re-derived; the file is auto-generated and the newest three rows match the ledger. A wrong `result` string in the ledger would render faithfully and wrongly — that is the ledger's audit (de's area), not the render's.
- board.jsonl find/done/note rows (71) were not traced; only rule/block rows were, per the delivery duty's scope.
- The status cards (`~/.aupai-status.json`) were not audited; they are the same author and instrument as the page.

## 6. Open questions for the controller

1. UF-1/UF-2/UF-3 are page lines whose source was removed, voided, or retracted. Do I amend them in place (the store's convention) or mark them withdrawn? The audit is read-only; the page is my standing instrument and the call is yours.
2. UF-4: should the store carry a date per line? One field, backward-fill impossible — only forward.
3. UF-6: is a backfill of domain_bpb with a synthesised vocab_id ever acceptable, or does the old-format checkpoint class stay unscored on that metric permanently?
4. UF-7: should cited artifacts be committed to the repo, or should citations mark pod-only sources?

## Pair check

e1, 2026-09-04. Recomputed UF-1, UF-2, UF-3 from the artifacts, not from this report's citations.
**Two hold; one is wrong in its central claim and I am partly the cause of the error it repeats.**

**UF-1 — HOLDS.** `~/.aupai-progress.jsonl` line **6**, not line 5 (line 5 is the ChatML-absence
0.075% line). Substance quoted correctly. Verified independently: `grep -rn '94\.4' facts/*.json`
returns three hits and none is this pair — all three are `eff` entries about a GPU-busy figure
(`gpu_busy_pct_wrong: 94.4`), a coincidence of digits. AGENTS.md:205 carries the removal note
verbatim as quoted. `docs/lessons/honest_measurement_prereg.md:103` is the passing mention, and
it is worse than "no artifact path": it lists the pair under 没被推翻的 (not overturned), so the
one surviving citation asserts the pair still stands. Page line never amended. Severity S1 correct.

**UF-2 — HOLDS, and understates it.** Line **586**, not 585. The report says the page "still
carries the voided 15614/GB and 18106/GB". Stronger: `runs/scan_math_ws2.json`, the authoritative
whitespace-13 artifact, contains **neither number and no per-GB field at all** — its eval entries
carry `corpus_docs_hit` and `hit_rate` only (humaneval_164 question: 440 docs, rate 0.000172).
So those two figures are not merely superseded, they are in a unit the current instrument does not
produce, and nothing in the repo can reproduce them. The `333 分片` in the same line DOES check out:
127 + 206 = 333 from the artifact. Severity S1 correct.

**UF-3 — FAILS as written.** The report says the page's "0.457/0.484, z 13.4, 145/164,
1.944/2.061 match no cell" of `e1_31_middle_layer_loop`. They match no cell of that row because
they are not from it: they are **Stage A**, a different experiment, and they trace exactly.
`docs/standards/roadmap_0903.md:24` publishes Stage A as "humaneval BPB 0.4567 -> 0.4840
(145/164 worse, z 13.4); domain_loss 1.9443 -> 2.0609". I recomputed the domain_loss pair from
the cited artifact on the pod (`/work/aupai/runs/n7_domain.jsonl`, 9 domains, unweighted mean):
**1.9443 and 2.0609**, to the digit. The page rounds to 3 places; that is all.

What survives of UF-3, and it is worth keeping as a separate finding: the page line's CAUSAL
claim is retracted. The ledger row's `finding` says "THE MISMATCH IS THE FINDING, NOT THE LOOP …
this retro-explains Stage A: its +0.0273 is the mismatch-B cell (+0.0264), so Stage A measured
the mismatch and never measured the loop." The page presents the Stage A numbers as what the loop
costs. So: numbers correct, attribution retracted, line unamended. That is an S2 about an
un-amended retraction, not an S1 about untraceable figures.

I have a stake in this one and say so: the Stage A row is mine, and the roadmap sentence the page
was quoting is mine. The 1.64x speed figure in the same page line is also retracted in my own
roadmap row ("the 1.64x first published here is RETRACTED — one 20-iter run"), and the page still
carries it — a second un-amended retraction in the same sentence that UF-3 did not name.

**One artifact-availability finding, not in the report:** `runs/n7_domain.jsonl` is the cited
source for the Stage A roadmap row and it does **not exist in the repository** — it is pod-only
(`/work/aupai/runs/n7_domain.jsonl`, 55,804 B, Sep 3 13:49) and `git log --all` shows it was never
committed. A reader with only the repo cannot open the source of a published roadmap number. Same
shape as my own E-series findings about labels: the record names a file nobody can read.

Not recomputed: UF-4, UF-5, UF-6.

98, 2026-09-04. Recomputed E1, E3, E5 from `eval_heldout.md`:
- **E1 — HOLDS.** `git show bfa1a846 --stat` lists only `eval/domain_loss.py` (104+/13-); `grep -c '"cu' runs/score_matrix.jsonl` = 0. The fix labelled domain_loss rows only; every published score_matrix row is path-unnamed.
- **E3 — HOLDS.** The −0.010770 verdict is the params_leg_438m_3p76b ledger row's finding (`runs/b0_23_n2_verdict.json`) and `facts/data_scaling.json` (`ds.n2_params_vs_data_matched_compute`); its ruler was domain_loss on the cu=None path. `facts/efficiency.json#eff.eval_path_cu_artifact_ce` states the pooled artifact −0.081780 "against ds.n2...'s own -0.010770: 7.59x". The artifact is 7.59x the verdict.
- **E5 — HOLDS.** `eval/score_matrix.py:649` keys write_records on `(ckpt, profile)` with no cu dimension; the matrix is current state, so a doc_cu rescore under the same key replaces the cu_none row with nothing naming the path. Risk unexercised so far; the #cu suffix ruling closes it.
