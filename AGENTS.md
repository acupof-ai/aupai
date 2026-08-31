# aupai — 200M reasoning LLM, coding and math (KDA + gated MLA hybrid, optional Attention Residuals)

**Objective changed 2026-08-30, by the user.** This was a 200M *Chinese* LLM. It is now a
reasoning model targeting coding and math capability at ~30B tokens, and the corpus follows
the capability rather than the language: roughly 60:40 English-leaning, because code is
written in English and the math and chain-of-thought sources are overwhelmingly English.
Chinese web drops from a planned 16B to 3-4B. The scaling law is no longer the deliverable.

One consequence is already known and gates the corpus build: the frozen 32,784-slot
vocabulary was fitted on Chinese web and cosmopedia, so it now faces a material
distribution change plus a third distribution — code — that it has never seen.
`tokenizer_eval` runs against a sample of the new composition **before any fetch**, and a
failure is a rebuild decision that invalidates every existing checkpoint. What survives the
change and what it supersedes: `docs/standards/0830v1_gates.md`.

Architecture: NoPE throughout — no RoPE, no learned position embeddings; KDA state carries all position information. Attention is gated MLA, full causal over the 4096-token sequence (document-masked). The 1024-token sliding window was removed 2026-08-30: `infer_local.py` never implemented it, so every generation ran a wider attention than training. Attention Residuals are on by default.

## Writing rules (all docs, commit messages, and replies)

- No metaphors. They distort.
- No big words, no verdict-first tone, no invented compressed terms, no filler explanation, no spoken/speech register.
- Delete anything a competent reader already knows.
- Every rewrite must raise information density. Rephrasing without new information is a no-op.
- 3+ consecutive prose paragraphs: check whether a table, list, or grouping works instead.
- Target: simple, clear, coherent, specific, accurate, complete. Short units and strict logical order matter more than completeness of phrasing.

## 0830v1 reset (2026-08-30)

Pre-0830v1 conclusions are zeroed: no checkpoint, run, or recipe is a baseline. Kept: corpus bytes (`data/corpus/*`), the tokenizer, reusable methods (`docs/lessons/kept_methods.md`), dataset properties (`facts/`). The experiment log restarts empty with 0830v1. Full history is in git log before this date; nothing was kept "just in case".

## Layout

| path | responsibility |
|---|---|
| `train.py` `sft.py` `sft_math.py` `serve.py` `chat.py` `infer.py` | entry points |
| `scripts/` | ops: harness, exp log, eval shards, SFT packing, tokenizer builds |
| `datagen/` | corpus generation and augmentation |
| `filters/` | data cleaning |
| `eval/` | benchmarks |
| `algorithms/` | RL |
| `mathbank/` | synthetic math generators |
| `workflows/` | corpus JS |
| `data/corpus/*` | corpus bytes (gitignored except `sample/`) |
| `data/mix_scale_*.json` | the 0830v1 mixes |
| `data/tokenizer.json` | the frozen vocabulary |
| `docs/lessons/` | research, with frontmatter |
| `docs/audits/` | source audits |
| `docs/standards/` | standards and recipes |
| `facts/` | measurements, each with its config |
| `runs/` | logs and `experiments.jsonl` |

## Hard constraints

- **Tokenizer frozen 2026-08-29.** Rebuild only under the three unfreeze conditions (see Tokenizer), and copy the live file to `data/tokenizer_<name>.json` first. A rebuild invalidates every checkpoint trained on the old vocabulary.
- **Vocabulary identity.** Score every checkpoint with the vocabulary it was trained on; checkpoints and packs carry `vocab_id`, and a mismatch refuses. For an older checkpoint pass `--tokenizer`.
- **GPUs.** All 8 cards belong to this repo (GPU7's tileRL reservation was released 2026-08-30). The controller session allocates them; ask before starting a GPU process. Kill by exact PID, never `pkill -f`. A process the controller cannot account for gets killed.
- **Lanes: a 7-card training block, and one lane card for everything else.** `world` is 7, so 8 cards leave exactly one for evals, probes, and verification runs — there is no arrangement that yields more. The block's card indices are allocation and the controller names them (`cards` in `data/mix_scale_run_config.json`); the lane is whichever card is not in `cards`. Two rules follow, and the second is the one that cost time:
  - **Small jobs queue on the lane card. They never spill into the block, not even onto a card that is idle at that instant.** A 7-card run needs all seven *simultaneously*, so one 10-minute eval on one block card blocks a 55-minute training job completely — contention only slows, occupancy stops. On 2026-08-30 a bf16 A/B waited ~40 minutes for a window, and the window it finally got was closed within seconds by a confirmatory eval landing on a block card.
  - **The lane holds one job at a time.** The round routinely wants two or three concurrent probes; they serialize. The previous version of this rule named a single bench card without saying jobs must queue on it, so three concurrent small jobs spilled into the block *by necessity* — an under-provisioned lane is violated for cause, not by carelessness, and a rule people must break is not a rule.
  - When the block is idle and no 7-card job is pending, the controller may lend block cards out explicitly. Idle is not the same as free: a card's owner is the script still running or the job the controller has queued, never the instantaneous `nvidia-smi` row.
- **Long jobs detach.** `pod "<cmd>"` in the foreground dies with the tn tunnel after 5 minutes, but the container process keeps running — it becomes an orphan holding a whole card at 100%. One such orphan silently contaminated a seven-card profile before anyone noticed. Always `setsid nohup ... </dev/null &`, then poll the log.
- **Language.** Repo artifacts (code, docs, commits) in English; user-facing text in Chinese.
- **Shared files.** Announce before editing `train.py`/`sft*.py`/`AGENTS.md`, commit promptly, hand the file back.
- **CI gates.** ruff E9/F, py_compile, `test_arch_compat`, `eqcheck`, `holdout` on every push.
- **Derived artifacts carry the fingerprint of what produced them.** The failure mode: a derived artifact stays valid after its source changes, and nothing raises. Three instances, each bought with an incident: checkpoints carry `vocab_id` (a k5 SFT trained at loss 4.77 instead of 1.28 with nothing raising); token caches carry `.srcfp` of their source directory (the 0.2b run — source swapped, cache rebuilt against the new source, training kept reusing it); corpus shards carry `filters_fp`, a content hash of the `filters/*.py` that produced them. The fingerprint covers what actually takes effect, not the nominal version: content hash, not git sha — uncommitted edits still change what a build keeps, and a sha cannot see them (same reasoning as `corpus_fingerprint`'s "content-based, not mtime-based").

## Entry points

| task | command |
|---|---|
| Launch any GPU or corpus job | `python scripts/harness.py launch <name> [--training] [--hypothesis "..."] -- <cmd>` — exp row first, card allocation from controller config, startup gate for training, monitor on process-gone/log-silent |
| Pretrain | `./run_ddp.sh [train.py flags]` — wraps `torchrun ... train.py --fp8` on all 8 GPUs |
| SFT | `scripts/run_sft.sh <name> <resume_ckpt> <sft_pt> [sft_math.py args]` |
| Eval, one metric | `scripts/eval_hard.sh <ckpt> [ngpu]` |
| Eval, full matrix | `scripts/eval_all.sh <ckpt> [tokenizer]` — math-hard, math-500, MC suite, digit head |
| Score matrix | `scripts/score_matrix.py --ckpt <ckpt> [--json runs/score_matrix.jsonl]` — per-type metrics; generative SKIPs on base, never 0 |
| Pod drift | `scripts/pod_sync_check.sh` — sha256 of tracked code vs /work/aupai; exit 1 on DIFF/MISSING |
| Measure everything unscored | `python scripts/harness.py measure` |
| pass@k gate for RL | `python eval/math_hard.py --ckpt X --k 8 --temperature 0.8` — needs pass@8 − pass@1 ≥ 15pt |
| Corpus | `python datagen/build_corpus.py --domain X --source Y --target_tokens 6e9`; `--dry --limit N` prints the rejects histogram. Math generators: `mathbank/vet_programs.py` is the registry root that reaches `math_programs_l*` |
| AttnRes A/B | `NGPU=6 STEPS=500 scripts/run_ablation.sh` |
| FP8 NaN probe | `COMPILE=1 GC=0 BS=8 MUON=1 STEPS=60 python scripts/nan_probe.py` (pod) |
| Reachability | `python scripts/reachability.py` — which scripts are reachable from entry points; `runs/reachability.txt` is the committed listing with fate rulings |
| Count cleaned code | `python scripts/count_cleaned_code.py` — token counts over cleaned corpus domains |
| Checkpoint info | `python scripts/ckpt_info.py <ckpt>` — config, vocab_id, step count from a checkpoint |
| Perplexity | `python eval/ppl.py --ckpt <ckpt>` — perplexity over a text sample |
| Lambda probes | `python scripts/assemble_lambda_probe.py` / `python scripts/validate_lambda_probe.py` — t05 lambda-curriculum probes (3b, deprioritised but live) |

## Run pretraining

```bash
./run_ddp.sh --mix data/mix_scale_3.24b.json --name <name> [--attn_res] [--warmup 150] [--lr_scale 0.5]
```

Any `--flag` in `train.py`'s parser overrides `Cfg.<flag>` — a fixed whitelist (seq/batch/accum/vocab/seed/attn_every/attn_res_blocks/val_*/warmup + the boolean flags), not a reflection over `Cfg`; a `Cfg` field without a parser entry cannot be set from the CLI. The 0830v1 budget points are six mixes — `mix_scale_{0.2b,0.3b,0.4b,0.8b,1.6b,3.24b}.json` — identical weights, scaled `total_tokens`. Five are a ×2 geometric series: three points would exactly identify the three parameters of E + B/D^β and leave no residual degrees of freedom to expose a bad fit. Checkpoints save as `ckpt_{name}.pt`; naming convention `ckpt_{arch}_{tokens}_{date}.pt`.

On the pod, launch detached (see Pod):

```bash
pod "cd /work/aupai && setsid nohup bash -c './run_ddp.sh --mix data/mix_scale_3.24b.json --name k9 > runs/k9.log 2>&1' </dev/null >/dev/null 2>&1 &"
```

## Record a run

Every GPU run gets a row before it starts and a result when it ends:

```bash
python scripts/exp.py start --name <name> --cmd "<command>" --hypothesis "<question the run answers>"
python scripts/exp.py done  --name <name> --result "math-hard 3.6%" --finding "<what the number means>" --decision "<what changes>" --status ok
python scripts/exp.py render   # rewrites EXPERIMENTS.md, newest first
```

`hypothesis` is written BEFORE the run starts. `result` is the number; `finding` is its interpretation, not the number; `decision` is what changes because of it.

## Harness — `python scripts/harness.py`

| subcommand | use |
|---|---|
| `check` | invariants; exit 1 on FAIL; runs in CI |
| `--selftest` | every check must FAIL on its broken world |
| `ledger` | checkpoint, provenance, math-hard score, one row per checkpoint |
| `gaps` | what is not measured, stated out loud |
| `measure` | run the full eval matrix and record it |
| `stages` | postconditions per stage |

### The checks

| check | asserts | on FAIL |
|---|---|---|
| `mix_not_unfiltered` | the default mix names no `web` domain | someone repointed `Cfg.mix` at an unfiltered mix; revert |
| `mix_shards_present` | every default-mix domain has shards (GPU boxes only) | tokenize the missing domain; SKIP on machines without GPUs |
| `no_oversized_blob` | no tracked file over 5MB | `git rm --cached` it; large files are gitignored |
| `tokenizer_roundtrip` | NUL, tab, hanzi, digits decode to the exact bytes | the vocabulary drops a byte; rebuild with `initial_alphabet` |
| `pinned_ids` | `<eos>=1`, `[NUM]=32772` | a rebuild moved the specials; re-pin or update the check |
| `no_stale_running` | no experiments row is `running` over 24h | the job died without `exp.py done`; close the row |
| `guard_on_path` | `train.py main()` calls the mix guard | the guard moved off the entry path; restore it |
| `facts_well_formed` | every fact carries its config; guarded phrases absent | a fact landed without its measurement config; add it |
| `entrypoints_ran` | cited scripts exist (FAIL); tried ones have an ok run (WARN) | the doc rotted, or the command never succeeded — run it |
| `entrypoints_table_present` | AGENTS.md has ≥1 entry-point row citing a script | the table was deleted; restore it |
| `docs_root_clean` | zero `.md` directly under `docs/` | classify the file into `lessons/`/`audits/`/`standards/` |
| `lessons_have_frontmatter` | every lessons/audits doc has `question`/`status`/`source` | add the frontmatter |
| `fact_refs_resolve` | every `facts/<file>.json#<id>` citation resolves; retracted citations WARN | fix the citation or the fact |
| `doc_commands_exist` | every `.sh`/`.py` cited in a command block exists | the doc rotted; fix the command or the file |
| `score_matrix_present` | every status=ok training run has a score-matrix record for its checkpoint | run `scripts/score_matrix.py --ckpt <ckpt> --json runs/score_matrix.jsonl` |

## Add a check

Append a `(name, asserts, incident, run, broken)` tuple to `CHECKS` in `scripts/harness.py`. `run(root) -> (state, evidence)`; `broken()` builds a world where `run` must report FAIL — by **mutating a real artifact**: copy the real file and break it. Never hand-write a world; it shares the check's own assumptions (a scan's self-check was green while the scan always returned clean).

```python
def check_thing(root):
    doms, err = read_mix(os.path.join(root, cfg_default("mix")))
    if err:
        return FAIL, err
    return PASS, f"{len(doms)} domains"

def _broken_thing():
    # the REAL mix with its domains emptied -- mutated, not hand-written
    import json, shutil
    d = _tmp_repo()
    p = os.path.join(d, cfg_default("mix"))
    shutil.copy(os.path.join(ROOT, cfg_default("mix")), p)
    obj = json.load(open(p))
    obj["domains"] = {}
    json.dump(obj, open(p, "w"))
    return d
```

Selftest also asserts the broken world holds a file at a repo-real path. Known ceiling: it catches worlds built on made-up paths, not worlds that mutate one real file and hand-write the rest.

## Land research

`docs/lessons/<topic>.md` (audits: `docs/audits/`), opening with frontmatter:

```
---
question: What does this research answer?
status: measured | recorded | open | retracted
source: command, artifact path, or arXiv id
---
```

Cite a fact as `facts/<file>.json#<id>`; the id must exist. Numeric conclusions belong in `facts/*.json`; the lesson references them.

## Eval — what each set can and cannot say

| set | resolution | caveat |
|---|---|---|
| math-hard, 1032 problems | — | v1 retired as metric of record: our own generators contaminated it; continuity only |
| math-500 | saturated | 0.0% contamination on the pod corpus; 30% of questions have a containment hit in the math SFT corpus, so post-SFT values are inflated, base values clean (`facts/contamination.json`) |
| MC suite | low | three of five sit at chance |

## Data

### Mix

Per-domain weight, epoch cap, anneal weight. `train.py` builds the schedule and consumes it in order, so `Cfg.epochs` is forced to 1. **It is the only data path** — a named-but-missing mix raises. The flat-corpus fallback was deleted: it once trained on 244KB in silence. `data/mix_sample.json` is the 2,000-document sample a checkout ships.

### Chat format

ChatML, owned by `scripts/loader.format_prompt / format_example / format_history`. The pretraining chat domain renders in ChatML too, so SFT does not teach the format from nothing. `scripts/test_sft_pack.py` checks the loss mask directly (CI): every masked span ends at `assistant\n`, the turn terminator is supervised.

### Synthetic data

One distinction decides the mix weight: **anchored rephrasing** vs **from-scratch generation**. Test: are the output's numbers and entities a subset of its declared source. Methods and failure modes: `docs/lessons/kept_methods.md`.

### Numbers — `--fone`

BPE splits numbers by frequency, not place value (1640 → `16|40`). FoNE gives each number one `[NUM]` token carrying a Fourier value, scored ten-way per digit. `--fone` changes the data format everywhere: pack with `prepare_sft_math.py --fone`; a checkpoint whose flag disagrees with the pack raises. `scripts/fone_digit_acc.py --ckpt X` scores the digit head. Failure mode: multiple answer formats on identical prompts leave termination underdetermined — one format per prompt.

### Vocabulary identity

Every checkpoint is scored with the vocabulary it was trained on. `data/tokenizer.json` is rebuilt in place; ids do not survive a rebuild and size does not identify a vocabulary. Checkpoints and packs carry `vocab_id` (a hash of the id→token map) and `sft_math.py` refuses a mismatch. The loudest skipped-check bug: a k5 SFT trained at loss 4.77 instead of 1.28 with nothing raising.

## Tokenizer

Frozen 2026-08-29. A rebuild is allowed only under the three unfreeze conditions, and invalidates every checkpoint trained on the old vocabulary.

- **Gates** (`scripts/tokenizer_eval.py --tokenizers <paths>`): round-trip lossless and all 256 bytes are vetoes; hanzi whole-char ≥ 0.95 is a veto; ref fertility ≤ 1.55 and never-used ≤ 0.01 are regression guards.
- **Build** (`scripts/build_tokenizer.py`): always pass `initial_alphabet=ByteLevel.alphabet()` — without it NUL silently drops; stratified equal-byte sample per domain.
- **Measure** with `scripts/tokenizer_report.py --selftest` — mandatory before believing any number it prints.
- **Unfreeze conditions — three, and nothing else:**
  1. The model outgrows the fitted 12–20K optimum (arXiv 2407.13623).
  2. The corpus distribution changes materially.
  3. An extrinsic test — two pretrains differing only in the vocabulary — says a candidate is better.

Facts (fingerprint, sizes, gate values, frontier, sweeps): `facts/tokenizer.json`.

## Pod

- `pod` is at `~/bin/pod` — **not in the default PATH**. A session once misjudged "no pod access" for this reason.
- 8×H20, all usable. `/work/aupai` is not a git repo — push files.
- **`tn exec` and `~/bin/pod` are two different filesystem views with the same hostname.** `tn exec` runs on the host (where `crictl` lives); `~/bin/pod` runs inside the container. A file present in one view is not necessarily present in the other. Read and write container artifacts — code, logs, checkpoints, corpus — with `~/bin/pod` only. Use `tn exec` for host-level queries (`nvidia-smi`, `ps`, `crictl`). GPU and process state are machine-wide, so those queries agree on both sides — that is why the confusion survived for days.
- **`setsid`, not `nohup`**: `pod` runs through `crictl exec`; when that session ends the kernel kills the whole process group, and `nohup` only blocks SIGHUP. Launch:

```bash
pod "cd /work/aupai && setsid nohup bash -c '<cmd> > runs/x.log 2>&1' </dev/null >/dev/null 2>&1 &"
```

- **`CUDA_VISIBLE_DEVICES`, not `cuda:N`**: fla/Triton kernels launch on the current device; `cuda:1` raises illegal memory access.
- **File transfer into the container: `podput <local> <remote-abs-path>`** (gzip+base64, 100KB cap — the real limit is the remote argv, not a constant). `tn push` lands on the HOST filesystem and is invisible inside the container — never use it to deliver code or data to the repo.
- **Push code via `scripts/pod_push.sh <files>`, never bare `podput`.** It refuses to push a file with uncommitted changes, and re-runs the drift gate after. **It does not pull, and no workflow here should.** Every session shares this tree and this `.git`, so another session's commit is already in HEAD the moment it is made — there is nothing to fetch from them, and `origin` is not how they reach each other. The `--autostash` that used to accompany the pull stashed and restored the *whole* dirty tree, i.e. five other sessions' uncommitted edits, on every push: `git checkout` on a file you did not write, automated. A push copies one session's local state into the pod's global state; in a multi-session tree that state is stale by default (2026-08-30: a push rolled back 3b's `datagen/build_corpus.py` row-group feature, commit e39146e, and its new launcher died on `unrecognized arguments: --rg_mod`).
- `uv sync` after dependency changes.
- **Outbound network: `curl -4`, always.** The pod's IPv6 egress is broken, curl tries IPv6 first, and the failure surfaces as `Errno 99 / Cannot assign requested address` — which reads as "this host is unreachable" and is actually "the local address family is unusable". `urllib` and most probe scripts do not fall back to IPv4. On 2026-08-30 this produced a whole reachability matrix of false negatives and nearly retired the pretraining code slot: with `-4`, `data.together.xyz` (RedPajama-1T's manifest host) answers 200 at 2.66GB per file. `Errno 99` is a local error, never a statement about the remote.
- **What is reachable, measured 2026-08-30 with `-4`:** hf-mirror `resolve/main/<file>` paths 200 but its metadata API 403s (anti-scrape) — so a dataset with a *published URL manifest* is easier to fetch here than one needing the Hub API, the opposite of the usual ranking. `huggingface.co` itself times out on both families.
- **Reachability changes without notice, so a fetcher carries a mirror chain.** 2026-08-31 from ~14:30: `hf-mirror.com` rc=28 (timeout) for the rest of the day; `www.modelscope.cn` 200 in 0.08 s and `/datasets/AI-ModelScope/<name>/resolve/master/<file>` serves HF datasets; `data.together.xyz` 200 all day. The math and CoT fetches stalled for hours on the one dead host. Rule: every HF-hosted source lists `[hf-mirror, modelscope, huggingface.co]` with a verified name map; a 10 s `curl -4` probe per host before fetching and on any timeout/403; failover continues from the same `.part`; `fetch_stats.json` records which host served each file (`t37`).
- **`cd` inside a backgrounded chain stays in it.** `pod "cd X && cmd & followup"` runs `followup` in the original cwd: the `&` backgrounds the whole `cd X && cmd` list in a subshell. Everything that needs the cwd goes inside the chain; everything outside it uses absolute paths.

## Rules kept from before the reset

| Rule | Incident in one line |
|---|---|
| A stage is done when its falsifying measurement exists, not when it produced a file | Three write-ups one night, zero runs of the metric of record |
| Every CHECKS entry carries `broken()`; `--selftest` asserts FAIL there | Four guards shipped the same defect in one afternoon, selftest green |
| Broken worlds mutate a real artifact, never a hand-written one | Three of six checks were dead while selftest passed — both halves believed the same fiction |
| A metric carries a known-answer case | `tokenizer_report.py` printed four wrong numbers in one day |
| Install probes measure teacher-forced AND free-running in the same run | Free-running-only scored 0.0→0.0 and would have retired a correct path |
| A null landing in a pre-registered cell does not certify that cell | A pre-registered null was written up as settled; the amendment is labelled as written afterwards |
| A permanent red is the same as no signal | CI red on a clean checkout; a red nobody acts on is no signal |
| Before a two-arm test, name what else changed with the variable | The 36%-vs-5% ablation: the freed 31% went to web, so the verdict was web's |

**Measurement numbers carry their configuration.** Every wrong number this repo has published was a value that depends on its measurement configuration, printed without it. `tokenizer_report.py --selftest` catches the class with one assertion: ten times the text must not move a per-character or per-word ratio. Known answers come in pairs and must differ by 60 points. `sample_corpus`'s `shards` and `clip` are part of every metric's definition.

**Two failure modes specific to the harness:**

- `cfg_default` raises rather than returning None: an annotation once made it return None, and two checks reported SKIP "chosen on purpose" while `check` exited 0.
- The ledger takes names from the scores: `--name X` attributes a score to `ckpt_X` without the checkpoint appearing in any command — the top of the table was once missing.

**E2E — `E2E_GPU=<idx> python scripts/test_e2e.py`.** The only test of the joins: mix → tokenize → pretrain → checkpoint → load → pack → SFT → generate. It asserts `vocab_id` equality end to end, cos(pretrained, post-SFT embedding) > 0.9, and that the pretrain actually stepped (cos(fresh init, checkpoint) < 0.9). `E2E_GPU` is required and there is no CPU half — a cardless run could only re-check what `harness check` covers. CI does not run it; it will not pick a card (the pod's GPUs are shared).

**Before committing model/optimizer changes:** `python scripts/test_arch_compat.py` (CPU: AttnRes fwd/bwd, legacy-ckpt round-trip, optimizer grouping/schedule/snapshot, KDA decay init). Old checkpoints keep loading: `HybridLM.load_state_dict` remaps fused keys and auto-disables AttnRes; consumers build the model from `ck["cfg"]`, never from the live `Cfg` class. `ruff format && ruff check` on touched files (line length 110; CI gates E9/F).

**Fact store — `facts/*.json`.** Measurements live here, one file per domain, never in prose. Required per entry: `id`, `value`, `measured` (YYYY-MM-DD), `source` (command or artifact), `config` (non-empty), `uncertainty`, `status` (`measured` / `recorded` / `unmeasured` / `retracted`). `unmeasured` and `retracted` entries also need `claim`, `audit`, `refuted_by`. Optional: `unit`, `guard_phrases` (must not reappear in AGENTS.md), `boundary` (what the measurement cannot answer). `facts_well_formed` enforces the required fields. Domain files: `facts/tokenizer.json` (fingerprint, sizes, gate values, frontier, sweeps), `facts/contamination.json` (math holdout containment), `facts/data_scaling.json` (scaling-law and token-budget measurements), `facts/multilingual.json` (corpus stats, fertility, zh:en ratio/transfer evidence, token supply), `facts/data_quality.json` (hand-read quality audits, filter retention, domain loss drops, quality-head AUC), `facts/base_eval.json` (base-model eval resolution at 200M, minimal-pair construction rules, sample-size math, MC language confound), `facts/efficiency.json` (hardware specs, throughput, hybrid-ratio and AttnRes evidence).

**Coordination.** Several sessions share this tree. Announce before editing `train.py`/`sft*.py`/`AGENTS.md`, commit promptly, hand the file back. Commit messages in English, one concern per commit.

Three rules from the day one session's `git checkout` erased another session's uncommitted device gate in `scripts/test_arch_compat.py` (recovered only because a copy had been `podput` to the pod minutes earlier):

- Never run `git checkout` / `git restore` on a file you did not write. Undo your own edits by hand.
- Run `ruff format` over a whole file only if you created it. On a shared file, format the lines you touched; a 61-line reformat buries someone else's work and invites the checkout that deletes it.
- Commit as soon as a change works. The working tree protects nothing — `git fsck` cannot recover what was never staged.
- Stage by path, never `git add -A` / `git add .` / `git commit -a` in a shared tree. 2026-08-31: a `git add -A` under the message "pod manifest: refresh" swept 26 files and 533K insertions — including 234MB under `data/_corpus_unsanitized/` and five other sessions' uncommitted work — into one commit (d535674). The pre-commit hook (`scripts/hooks/pre-commit`, installed by `harness install-hooks`) refuses staged files >5MB and new `data/` paths not in the allow-list, so the blob never enters history.
- A commit that touches a file in `data/pod_head_manifest.txt` is pushed to the pod by its committer in the same step (`scripts/pod_push.sh <file>`; it regenerates the manifest and refuses if that changed — commit the manifest and push again). The pod runs the pushed copy, not HEAD; 2026-08-31 the drift gate stopped the A/B launch twice on files another session had committed and not pushed.
- Corpus directories named by any ladder mix (`data/mix_scale_*.json`) are frozen: they carry `build_corpus`'s stamp and every ladder point and A/B reads them. New corpus goes to a new directory that the 30B mix names (`data/corpus/code_rp1t/`, not `data/corpus/code/`). 2026-08-31: ten new shards written into `data/corpus/code/` changed its fingerprint and `_assert_mix_domains` stopped the A/B at startup — correctly.
