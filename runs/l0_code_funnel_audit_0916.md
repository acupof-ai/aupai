# L0/L1 funnel audit — English code corpora (2026-09-16)

Read-only, CPU-only, no deletion. One strided sample per domain: 6 shards (every
`total/6`-th), first 2,000 nonblank docs/shard ≈ **12,000 docs/domain** (phase 1) and
first 1,500 ≈ **9,000/domain** (phase 2). Sampling caveat: only shard *heads*, not a
uniform draw over every line; treat rates as indicative, not population-exact.

Domains: `code_ultra_l2_dc`, `code_ultra_l3_noexec_dc`, `code_ultra_l3_stub_dc`,
`code_keep_p1_dc`, `code_py_starcoder_dc`, `code_py_rp1t_dc`. Raw counts in
`runs/l0_code_audit_phase1.json`, `..._phase2.json` (this audit's machine output).

## 1. What L0 each domain actually gets

| domain | build-time language gate | production L0 tier | near-dedup applied |
|---|---|---|---|
| code_ultra_l2 | `ast.parse` python (builder) | `reject_light` only | exact content-hash at build; **no near-dedup** |
| code_ultra_l3_noexec | none (task/analysis/solution prose+code assembly) | `reject_light` only | exact only; **no near-dedup** |
| code_ultra_l3_stub | `ast.parse` python (100%) | `reject_light` only | exact only; **no near-dedup** |
| code_keep_p1 | none on assembly — hardlinks into `code_rp1t_dd09`+`b2v2_dd`+`code_dedup08` (multilingual) | `reject_light` on sources | char-5gram MinHash 0.8 (`code_dedup08` subset only) |
| code_py_starcoder | `ast.parse` python (100%) | `reject_light` | **not** in `code_dedup08` |
| code_py_rp1t | `ast.parse` python slice (100%) | `reject_light` | char-5gram MinHash 0.8 (`code_dedup08`) |

`reject_light` (`datagen/build_corpus.py:98`) is only `short n<100`, `long n>200k`,
`bad_bytes>0.1%`, holdout. No symbol/dup/boilerplate/structural cleaning. Confirmed by
the trigger counts:

| domain | short% | long% | bad_bytes% |
|---|---:|---:|---:|
| ultra_l2 | 0.48 | 0 | 0 |
| l3_noexec | 0 | 0.07 | 0 |
| l3_stub | 0.025 | 0 | 0 |
| keep_p1 | 0.31 | 0 | 0.008 |
| starcoder | **2.04** | 0.075 | 0.008 |
| rp1t | 0 | 0 | 0 |

`code_lang_validate` is a **C/JS/Java/C++ lexer + brace-balance check, not a parser** —
Python is a *foreign* language to it. Python domains rely on `ast.parse`; C-family has no
parser in the environment by design.

## 2. Web L0 rules must NOT be applied to code (false-kill evidence)

The web tier `reject_reason` symbol rule is `len([\d\W]) / nonspace > 0.35`. Two problems,
both measured:

**(a) The deployed regex over-counts whitespace.** `[\d\W]` matches space/newline/tab
(`\W` includes them) while the denominator excludes whitespace, so indentation alone drives
the ratio and it can exceed 1.0 (observed max 1.59). Deployed vs whitespace-corrected
trigger rate:

| domain | deployed `sym>0.35` % | whitespace-corrected % |
|---|---:|---:|
| ultra_l2 | 89.7 | 5.10 |
| l3_noexec | 96.7 | 0.38 |
| l3_stub | 99.5 | 1.12 |
| keep_p1 | 89.1 | 1.91 |
| starcoder | 85.8 | 3.22 |
| rp1t | 81.9 | 0.34 |

The 82–99% is mostly the whitespace bug, not code punctuation. Even corrected, 3–5% of
ultra_l2/starcoder cross it — real, punct-heavy Python. Either way the rule is unsafe on
code. (This is a latent defect in the **web** path too; flagging, not fixing in this task.)

**(b) Other web rules:**

| web rule | l2 | l3_noexec | l3_stub | keep_p1 | star | rp1t | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `digit/nonspace>0.2` % | 0.33 | 0.18 | 0.41 | 0.29 | 0.57 | 0 | safe-ish, low |
| `dup_lines set/total<0.7` % | 6.0 | 1.4 | 1.2 | **15.3** | **10.7** | **11.4** | fires a lot; needs code-aware read |
| `nav_menu` % | 0.36 | 0.01 | 0.21 | 0.85 | 0.33 | 0.09 | low |
| CJK `boiler`/`unfinished`/`garbage_topic` | — | — | — | — | — | — | not applicable (Chinese-literal) |

`dup_lines` flags 6–15% — the one web-style signal with real volume on code. Whether those
are genuine near-dup docs vs repeated imports/blank/`}` lines is decided in §3 samples, not
by the raw rate.

## 3. Candidate Gopher code/en signal rates

% of sampled docs tripping each (phase 1 + parser-level phase 2):

| signal | l2 | l3_noexec | l3_stub | keep_p1 | star | rp1t |
|---|---:|---:|---:|---:|---:|---:|
| long no-punct line (≥100 chars, <2% punct) — any | 3.72 | 2.94 | 0.22 | 2.08 | **4.15** | 2.11 |
| … such lines >10% of doc | 0.09 | 0 | 0 | 0.13 | 0.08 | 0.03 |
| dup paragraphs (≥3, unique<0.8) % | 0.56 | 0.03 | 0 | 0.79 | 1.16 | 1.28 |
| English boilerplate ≥3 hits % | 0.12 | 0.01 | 0 | 0.15 | 0.25 | 0.31 |
| heavy URLs % | 0.55 | 0.03 | 0 | 1.22 | 1.37 | 1.61 |
| comment-dominated (>70% `#` lines) % | 0.48 | 0.04 | 0 | 0.15 | 0.40 | 1.25 |
| autogen banner % | 0.09 | 0.09 | 0.08 | 0.37 | **3.17** | **1.90** |
| vendored marker % | 0.02 | 0 | 0 | 0.03 | 0.40 | 0.17 |
| minified shape (maxline≥500, mean≥120) % | 0 | **1.92** | 0.22 | 0.02 | 0 | 0 |
| all funcs stub (pass/…/NotImplemented) % | 0.09 | 0 | 0 | 0.31 | 0.48 | 0.59 |
| ≥80% funcs stub % | 0.11 | 0 | 0 | 0.32 | 0.57 | 0.68 |
| ≥5 TODO/FIXME % | 0.24 | 0.01 | 0 | 0.40 | 0.75 | 1.14 |

Comment-fraction p99 is high for python (rp1t 0.82, l2 0.59) — so a naive
"comment-dominated" rule would false-kill heavily-commented real code; the >70% threshold
keeps the rate low but the positives need a hand-read.

## 4. Parse-channel (language ID) findings

| domain | whole-doc `ast.parse` % | reading |
|---|---:|---|
| ultra_l2 | 95.6 | python; ~4.4% fail |
| l3_noexec | 48.0 | **format, not breakage** — full_content = task/analysis/solution prose wrapping code; a whole-doc parse gate would delete half a valid domain |
| l3_stub | 100 | python by construction |
| keep_p1 | 55.0 | **multilingual by origin** (rp1t): fail buckets are real `LANG:js 9.5%`, `java 4.8%`, `cpp 4.7%`, `foreign:ruby 6.9%`, not junk |
| starcoder | 100 | build-time ast gate already applied |
| rp1t | 100 | build-time python-slice ast gate already applied |

`code_lang_validate` names Python as `foreign:python`, so its output on a python domain's
parse-failures is "carries python lexical markers", not a reject verdict — the breakdown
buckets are raw and need per-language gating, not a single accept/reject.

## 5. AST-level dedup — current status and gap

- **No AST/syntactic-equivalence dedup exists.** Every dedup path is lexical:
  - `build_corpus.MinHashLSH` — char 5-gram on whitespace/punct-stripped text, 128-perm.
  - `code_dedup_build.py` — same char-5gram MinHash, 0.8, union-find, over starcoder+rp1t
    only (`code_dedup08`); ultra and most of keep_p1 excluded.
  - `near_dedup_postpass.py` — normalises (strip comments/strings, numbers→#, non-keyword
    idents→@) then word-3gram Jaccard≥0.5, but this is token-lexical, not parse-based.
- Gap: two files identical after rename/reorder/whitespace/comment changes, but below the
lexical Jaccard threshold, are not clustered; conversely char-5gram clusters can merge
files that share a long boilerplate header. A true AST-canonical dedup (parse → canonical
tree / normalised subtree hashes) is absent.
- Recommendation (listed, **not built this task** per instruction): an opt-in
python AST canonicaliser behind the same builder-side, module-sha-stamped placement as
`code_lang_validate`; gated by a known-answer set (renamed-idents/reordered-fns must
cluster; different logic must not). C-family needs a real parser the pod lacks.

## 6. Rules shipped after hand-reading positives

Each candidate was run over the corpus and its real positives read before a threshold was
adopted. Three of the four text rules were **rejected** because their positives are real code:

| candidate | measured volume | hand-read verdict |
|---|---|---|
| dup_lines (web form) | 6–15% | **reject** — unit tests (repeated asserts), docstring `"""`/`Args:` lines, repeated decorators |
| all-funcs-stub | 0.3–0.6% | **reject** — zope.interface/ABC/Protocol declarations, micropython `.pyi` stubs |
| long no-punct / minified | 0.2–4% | **reject** — module docstrings, comments, MIT license text; l3_noexec hits are dense problem prose |
| **strict autogen banner** | starcoder 2.55%, rp1t 0.66%, ≤0.03% elsewhere | **ship** — every positive was a Django migration or AutoRest/protobuf SDK file |

Shipped: `datagen/code_quality_rules.py::is_autogen_banner` — true only when the **leading**
comment/module-docstring header carries BOTH a generated-marker (`auto-generated`,
`generated by`, `@generated`, …) AND a tool/do-not-edit cue (django, protoc/protobuf,
autorest, swagger, grpc, `do not edit`, …). The conjunction + header gate are what reject the
measured false-fires: `autogenerated = False`, an inline `# do not modify the cache`, and an
in-function docstring. `--selftest`: 7 genuine-banner positives + 7 real-code negatives, plus
header-gate and conjunction controls; all pass.

Per-domain rate from the committed module (same sample basis as phase 2):

| domain | strict autogen % |
|---|---:|
| ultra_l2 | 0.000 |
| l3_noexec | 0.011 |
| l3_stub | 0.000 |
| keep_p1 | 0.033 |
| starcoder | **2.547** (228/8953) |
| rp1t | **0.656** (59/9000) |

**Placement / wiring.** The module sits beside the builders like `code_lang_validate`, not in
`filters/PIPELINE_FILTERS`, so no frozen domain's `filters_fp` moves. It classifies only; it is
NOT wired into a live build in this change (plan B — no data regeneration). A future code
rebuild that opts in records `module_sha256` in its own stats and applies the verdict (drop or
tag) at build time. autogen files should also be run through the existing lexical dedup; the
banner catches the rename/format variants the char-5gram threshold misses.
