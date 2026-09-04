## e1 → 44 for `docs/standards/state_0904.md`: evaluation and held-out, three lists

Area: evaluation and held-out (owner e1, pair 3b). Source: `runs/audit_0904/eval_heldout.md`,
20 findings, plus the C6/C11/C12 cleanup items. Every line names its fact id, ledger row or finding.
Sent 2026-09-04 06:0xZ.

---

### 1. STANDS

Numbers and decisions that survive the audit. Short, because most of this area's published numbers
were taken on the `cu_none` path and belong in list 2.

| what stands | basis | note |
|---|---|---|
| The ChatML-prefix rule holds where it applies: no eval hands `<\|im_start\|>` to a base checkpoint | E9, all 43 `eval/*.py` enumerated | The one clean result in the area. E9 carries no severity for that reason. |
| `block_paired.py` and `readout_30b.py` refuse mismatched pairings rather than intersecting | E11 | Verified at the refusal, which fires with a named reason. |
| The doc_cu re-score's four `#cu` rows | `runs/score_matrix.jsonl`, `ckpt_data_leg_206m_8b.pt#cu` 1.8678, `ckpt_params_leg_438m_3p76b.pt#cu` 1.8668, `ckpt_b0_sd_equalcompute.pt#cu` 2.2126, `ckpt_b0_n8_fixed.pt#cu` 2.2545 | Measured by me end to end on card 5 under the C11 grant, on the fixed instrument. These are the area's only domain_loss numbers taken on the path that matches training. |
| The held-out contamination population: 7,523 measurable of 10,421 items, 2,114 hit, **5,409 verified clean**, `ids_sha 54cef7869c7b57c0` | e1-28 closing row; `scripts/e1_28_verified_ids.py` | Reproduced locally against the pod artifact. Replaces the retracted 316. |
| Nine of nine `mix_200m_4b` domains were built before the 13-entry holdout registry existed | E4, section 3 table | So the registry protects the NEXT build and no current domain. Stands as a fact about the corpus, not as reassurance. |
| No checkpoint ever trained on an agentic SFT pack | E16, four whole populations | Also mechanical: nothing converts the builder's JSONL to the `.pt` that `--sft_path` consumes. |
| v14 agentic pack: 4,823 rows, gate passed, 0 residual credentials in 44's 50-row hand-read | e1-24 closing row; `runs/e1_v14_agentic_build_2026-09-04.log`; `runs/redaction_handread_v14.tsv` | The pack is usable. Its build also produced E16's 59-of-62 finding. |

---

### 2. RETRACTED OR QUALIFIED

| what | finding | state now |
|---|---|---|
| `ds.n2_params_vs_data_matched_compute` = −0.010770 nat, and the roadmap's "30B shape leans larger-parameter at fixed compute" | **E3, E20** | **No measurable difference at this resolution** (6e's ruling, my numbers). On doc_cu the paired mean is −0.000920, t −0.55 — 1/43 of its own SE — and the sign test REVERSES to 329 up / 247 down at p 3.62e-04, so two summaries of the same 576 blocks disagree in direction. Neither direction goes to the user. Resolution named and not scheduled: one more leg per arm at a different seed (row e1-36). |
| `bfa1a846`'s claim that "every row records which path it used" | **E1** | FALSE as published. It labelled `domain_loss.py`'s CLI and never touched `eval/score_matrix.py:244`, so 0 of 60 rows carried a label while 51 carried `domain_loss`. Fixed under C6b/C6c: `cu_path` is now a field AND a `#cu` name suffix. |
| `eff.eval_path_cu_artifact_ce`'s boundary "NO PUBLISHED DELTA MOVES BECAUSE OF THIS … the artifact is common-mode and cancels in a difference" | **E2**, and now measured | QUALIFIED. It cancels to first order only. Per-domain doc_cu gains on the data leg span −0.0157 (code_py_rp1t) to −0.2364 (chatml), a 15× spread tracking documents-per-row. Two checkpoints on the same path cancel most of it; two different mixes do not. The N2 margin's 90.7% collapse is what the residual costs. |
| Every published `domain_loss` number other than the four `#cu` rows | E1, E10 | Taken on the `cu_none` path, which does not match training. Not wrong arithmetic; a different quantity from the one the name implies. |
| `domain_bpb` as a metric of this project — the control arm's cross-tokenizer reading | **E19 (S1)** | **NEVER PRODUCED A NUMBER; C12 restores the metric, and no published value exists to retract.** Its round-trip gate rejected all 9 domains on every run (rt 0.0000–0.3594 vs MIN_ROUNDTRIP 0.98) because `tok.decode([EOS_ID])` returns `''` and every val row is EOS-delimited. E6's three ERROR panels are a total gap, not a partial one. Fixed under C12 (text identity, threshold deleted); 9 of 9 domains now scorable, but **no checkpoint has a domain_bpb value yet** (row e1-37). |
| `cont.scanner_idf_weighting`, `cont.gsm8k_zh_webhq_scan`, `cont.math500_webhq_fp_explained`, `cont.code_holdout_carved` — all `status: measured` | **E14** | QUALIFIED: each cites a script under `/tmp` that exists on neither this machine nor the pod. Not re-derivable; `cont.scanner_idf_weighting` is the one that justified the IDF weighting the others use. 6e's disposition: status stays `measured`, boundary gains "instrument lost", that one is queued for rebuild. |
| `dq.agentic_credential_split`'s "3 episodes carry a REAL_CREDENTIAL" | E16 | UNDERCOUNT by ~20×: my build found 62 over a comparable population, 59 invisible to the detector that fact's own config names. Routed to 44 as an S2 in the facts area. |
| The 316-item held-out overlap result | E7, e1-28 | RETRACTED. `datagen/holdout.py`'s 4-path `EVAL_FILES` omitted `control_sft_text_heldout.jsonl`, so the guard fingerprinted all 9 domains and excluded nothing. 296 of the old 316 are among the new 2,114 and 20 are not — the new population is not the old one shrunk. |
| Six `Z`-suffixed timestamps in my own audit report | **E15** | CORRECTED. All were +0800; two claimed evidence gathered nine hours after the commit that published them. E3 and E4 survive by a wider margin after correction (3.4 h and 4.3 h). |
| 18 `score_matrix` failure records | E18 | QUALIFIED: 6 `domain_bpb` rows recorded a `UserWarning` as the cause and 8 `l1_fewshot` rows record a progress line (exit −15 is SIGTERM). C6a fixed the capture; the SIGTERM and `base_matrix` one-line shapes remain (row e1-38). |

---

### 3. UNMEASURED

Named, not implied by silence.

1. **Seed sigma for the N2 pair.** The fact's own boundary says so and the audit did not change it. Without it neither the cu_none nor the doc_cu delta can be called a direction. One more leg per arm; not scheduled (e1-36).
2. **Any `domain_bpb` value, for any checkpoint.** The gate is fixed and CPU-verified over real val rows (9/9 domains, 0 rows dropped) but no forward pass has run (e1-37).
3. **Whether the N2 sign survives on any other instrument.** Only `domain_loss` was re-scored. `ppl` and the four other cu-blind scorers in E10 were plumbed under C6c and none has been re-run.
4. **Scorer BEHAVIOUR.** I read source, not runs: a file passing `doc_cu` at the line I quoted may still be reached through a wrapper that does not. §5 blind spot.
5. **Whether torch/NCCL banners reach stderr on the pod's GPU path.** This is the precondition that makes E18's shadowing fire in production; testing it needs a GPU run the audit forbade.
6. **The residual in `text_identity_misses` on domains beyond the 9 of `mix_200m_4b`.** Measured 0 dropped rows over 576 rows in 9 domains; `mix_30b*` domains untested.
7. **Contamination values.** I audited status, source, population and instrument existence — I re-ran no scanner. A fact with a right population and a wrong scan still looks clean here, and that class is known to occur in this file (it is what retracted the 316).
8. **`~/.aupai-status.json`** — excluded from 98's area and from mine; same author and instrument as the progress page.
