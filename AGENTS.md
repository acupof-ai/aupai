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

## Writing rules (all docs, commit messages, register rows, and replies)

The standard is `docs/standards/writing.md` (user, 2026-08-31): no metaphors, no filler, no verdict-first tone, few quotes and parentheses, formulas set as formulas, bold for the key part only, three consecutive paragraphs become a table, every rewrite raises density, four review passes before hand-over.

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
- **A kill is not finished until `nvidia-smi` says the card is free.** Killing what you launched does not kill what it launched. 2026-09-01: after the milestone watcher's chain was killed by exact PID, `eval/run_eval.py` (pid 313429) still held GPU7 at 5.7 GB / 95% — a grandchild reparented to init whose pgid still named the dead leader, so `ps` by pgid could not see it as an orphan and only the card showed it. It would have contended with the next job on the lane. After any kill of a GPU job: read `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`, and kill by exact PID whatever still holds memory. A killed process can stay in the process table as a zombie: `kill -0 <pid>` returns 0 and `ps -p <pid>` prints a row for it, so neither says whether the kill worked. Read `ps -o stat= -p <pid>`: `Z` is dead (e1, 2026-09-03: three scan pids read as surviving `kill` and `kill -9` for ten minutes while the card had been free since the first signal). Killing the local wrapper (a `~/bin/pod` call, a timed-out foreground command) does not kill the process it started in the container: read the container's `ps` after every local kill and kill by exact PID there (e1, 2026-09-03: a CPU scoring run of 10,421 items kept running on the pod after its local wrapper was killed).
- **Lanes: a 7-card training block, and one lane card for everything else.** `world` is 7, so 8 cards leave exactly one for evals, probes, and verification runs — there is no arrangement that yields more. The block's card indices are allocation and the controller names them (`cards` in `data/mix_scale_run_config.json`); the lane is whichever card is not in `cards`. Two rules follow, and the second is the one that cost time:
  - **Small jobs queue on the lane card. They never spill into the block, not even onto a card that is idle at that instant.** A 7-card run needs all seven *simultaneously*, so one 10-minute eval on one block card blocks a 55-minute training job completely — contention only slows, occupancy stops. On 2026-08-30 a bf16 A/B waited ~40 minutes for a window, and the window it finally got was closed within seconds by a confirmatory eval landing on a block card.
  - **The lane holds one job at a time.** The round routinely wants two or three concurrent probes; they serialize. The previous version of this rule named a single bench card without saying jobs must queue on it, so three concurrent small jobs spilled into the block *by necessity* — an under-provisioned lane is violated for cause, not by carelessness, and a rule people must break is not a rule.
  - When the block is idle and no 7-card job is pending, the controller may lend block cards out explicitly. Idle is not the same as free: a card's owner is the script still running or the job the controller has queued, never the instantaneous `nvidia-smi` row.
  - **When there is no lane card at all — `NGPU=8`, as p500m_20b_0902 runs — co-residency is judged by host IO and seconds, not by metric class.** The rule stated on 2026-09-02 was "likelihood evals may share a card, generative ones wait", derived from one eval (`score_matrix`, 2.3 GiB). MEASURED against the run's own control — `--save_every 500`, and steps 500/1000/1500/2000 all read 7K tok/s/gpu with **no eval running**, because a 2.1 GB `torch.save` plus a val pass costs 78 s by itself: `score_matrix`'s four likelihood metrics cost **46 s**, cheaper than the control; `l1_fewshot` (generative) **209 s**; `ppl` **109 s** and climbing when it was killed. The class was never the variable. What separates `ppl` from `score_matrix` inside one class is that `ppl` `torch.load`s a whole token cache per domain — 85 GB for `zh_web`, ~166 GB across the nine. So: an eval that reads a token cache off `/data00` waits for the run; one that only loads a checkpoint costs about what a save costs. `python3 scripts/eval_load_cost.py` is the table, with the unmeasured evals listed as unmeasured rather than as zero.
  - **Judge the cost in seconds against what the run already spends on itself, never by the printed ETA.** ETA extrapolates a single 10-step interval over 19,151 steps, so one interval 54 s slow prints as 29 lost hours, and every checkpoint save prints ~99 h. Total across every dip in the first 1990 steps: 10.3 min of 6.04 h elapsed, 2.8% (`docs/lessons/gate_failure_shapes.md` §50).
- **Long jobs detach.** `pod "<cmd>"` in the foreground dies with the tn tunnel after 5 minutes, but the container process keeps running — it becomes an orphan holding a whole card at 100%. One such orphan silently contaminated a seven-card profile before anyone noticed. Always `setsid nohup ... </dev/null &`, then poll the log.
- **Language.** Repo artifacts (code, docs, commits) in English; user-facing text in Chinese.
- **Shared files.** Announce before editing `train.py`/`sft*.py`/`AGENTS.md`, commit promptly, hand the file back.
- **CI gates.** ruff E9/F, py_compile, `test_arch_compat`, `eqcheck`, `holdout` on every push.
- **A deletion needs a per-file check for glob and runtime loaders.** No static analysis sees a runtime glob. `scripts/reachability.py` is a citation graph -- a doc mention is an edge, so "reachable" can mean "named by a doc nobody runs" -- and `mathbank/vet_programs.py:37` globs `math_programs_l*_ext*.py`, so 23 live generators read as unreferenced to a name scan. Grep for `glob`/`importlib` over a directory before deleting anything in it.
- **Derived artifacts carry the fingerprint of what produced them.** The failure mode: a derived artifact stays valid after its source changes, and nothing raises. Three instances, each bought with an incident: checkpoints carry `vocab_id` (a k5 SFT trained at loss 4.77 instead of 1.28 with nothing raising); token caches carry `.srcfp` of their source directory (the 0.2b run — source swapped, cache rebuilt against the new source, training kept reusing it); corpus shards carry `filters_fp`, a content hash of the `filters/*.py` that produced them. The fingerprint covers what actually takes effect, not the nominal version: content hash, not git sha — uncommitted edits still change what a build keeps, and a sha cannot see them (same reasoning as `corpus_fingerprint`'s "content-based, not mtime-based").

## Entry points

| task | command |
|---|---|
| Launch any GPU or corpus job | `python scripts/harness.py launch <name> [--training] [--hypothesis "..."] -- <cmd>` — exp row first, card allocation from controller config, startup gate for training, monitor on process-gone/log-silent |
| Pretrain | `./run_ddp.sh [train.py flags]` — wraps `torchrun ... train.py --fp8` on all 8 GPUs |
| SFT | `scripts/run_sft.sh <name> <resume_ckpt> <sft_pt> [sft_math.py args]` |
| Eval, one metric | `eval/eval_hard.sh <ckpt> [ngpu]` |
| Eval, full matrix | `eval/eval_all.sh <ckpt> [tokenizer]` — math-hard, math-500, MC suite, digit head |
| Score matrix | `eval/score_matrix.py --ckpt <ckpt> [--json runs/score_matrix.jsonl]` — per-type metrics; generative SKIPs on base, never 0 |
| Pod drift | `scripts/pod_sync_check.sh` — sha256 of tracked code vs /work/aupai; exit 1 on DIFF/MISSING |
| Measure everything unscored | `python scripts/harness.py measure` |
| pass@k gate for RL | `python eval/math_hard.py --ckpt X --k 8 --temperature 0.8` — needs pass@8 − pass@1 ≥ 15pt |
| Corpus | `python datagen/build_corpus.py --domain X --source Y --target_tokens 6e9`; `--dry --limit N` prints the rejects histogram. Math generators: `mathbank/vet_programs.py` is the registry root that reaches `math_programs_l*` |
| AttnRes A/B | `NGPU=6 STEPS=500 scripts/run_ablation.sh` |
| FP8 NaN probe | `COMPILE=1 GC=0 BS=8 MUON=1 STEPS=60 python eval/nan_probe.py` (pod) |
| Reachability | `python scripts/reachability.py` — which scripts are reachable from entry points; `runs/reachability.txt` is the committed listing with fate rulings |
| Provision an empty pod | `bash scripts/bootstrap_pod.sh [verify\|fetch\|build\|vocab\|check]` — idempotent, one stage at a time, stopping on error rather than feeding a broken artifact forward. Launching the pretrain is deliberately NOT a stage |
| Count cleaned code | `python datagen/count_cleaned_code.py` — token counts over cleaned corpus domains |
| Checkpoint info | `python scripts/ckpt_info.py <ckpt>` — config, vocab_id, step count from a checkpoint |
| Perplexity | `python eval/ppl.py --ckpt <ckpt>` — perplexity over a text sample |
| Lambda probes | `python eval/assemble_lambda_probe.py` / `python eval/validate_lambda_probe.py` — t05 lambda-curriculum probes (3b, deprioritised but live) |
| Cursor rehearsal | `python3 scripts/rehearse_cursor.py --ckpt <ckpt> --steps 50` (pod, stopped window) — gates a row cursor before a resume. It VERIFIES an existing cursor; `scripts/replay_cursor.py` RECONSTRUCTS a missing one, so they are opposites and neither supersedes the other. Assertion 3 (wrong-order: srcfp + shuffle seed) runs cardless today; assertions 1–2 (continuity, fresh-run identity) are still waiting on a 50-step block — `runs/review.jsonl:19` notes, "rehearsal implements assertion 3 only; continuity/identity (assertions 1-2) deferred to the window 50-step block". A ledger row saying someone is waiting on an output is reader evidence; no scan can see it |
| Progress page | `python3 scripts/progress_feed.py` — writes `~/aupai-progress.html`; 98 runs it by hand every 5–20 minutes. **This row IS its reader.** A tool only ever launched from a terminal has no citation anywhere in the tree, so every reachability scan reports it unreferenced by construction — de-5 deleted it on that reading (a477ad1) and froze the page. Same shape as `vet_programs.py:37`'s glob: a runtime dependency static analysis cannot see. The fix for an operational tool is a row here, not leaving it bare |

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
| `score_matrix_present` | every status=ok training run has a score-matrix record for its checkpoint | run `eval/score_matrix.py --ckpt <ckpt> --json runs/score_matrix.jsonl` |

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

ChatML, owned by `scripts/loader.format_prompt / format_example / format_history`. **The pretraining corpus effectively contains no ChatML** — `<|im_start|>` occurs 0 times in 168,000 rows sampled across all 42 domains, and the chat domain is `问：/答：` plain text in 4000 of 4000 rows (de, 2026-09-01). Stated as a bound, not as zero: 0 of 4000 puts a domain's rate below 0.075% at 95% (rule of three), so `wiki_chat`'s 372,827 rows could still hold ~279 ChatML documents. The bound is what survives contact — a zero is overturned by one counterexample, and a format present in under 0.1% of the chat domain is not a format the model learned. The line that stood here said the opposite and was believed for weeks.

The consequence is not that ChatML is under-taught. `eval/code_zh.py` and `eval/math_zh.py` prompt through `format_prompt`, so **every generative number ever taken on a base checkpoint handed the model a prefix that appears nowhere in its training data**, then scored the continuation — the model repeats the input or drifts to web boilerplate, which is what an unseen prefix produces. Same checkpoint, continuation prompt with one demo: 94.4% of generations contain `def ` against 0.3% under ChatML. **Base generative zeros taken before this date measure response to an unseen prefix, not capability.** Rules that follow: base evals prompt in continuation format, which is in-distribution; a base eval may not introduce a token sequence absent from pretraining; SFT does teach ChatML from nothing, so the SFT-side loss-mask check below is unaffected. `scripts/test_sft_pack.py` checks the loss mask directly (CI): every masked span ends at `assistant\n`, the turn terminator is supervised.

### Synthetic data

One distinction decides the mix weight: **anchored rephrasing** vs **from-scratch generation**. Test: are the output's numbers and entities a subset of its declared source. Methods and failure modes: `docs/lessons/kept_methods.md`.

### Numbers — `--fone`

BPE splits numbers by frequency, not place value (1640 → `16|40`). FoNE gives each number one `[NUM]` token carrying a Fourier value, scored ten-way per digit. `--fone` changes the data format everywhere: pack with `datagen/prepare_sft_math.py --fone`; a checkpoint whose flag disagrees with the pack raises. `probes/fone_digit_acc.py --ckpt X` scores the digit head. Failure mode: multiple answer formats on identical prompts leave termination underdetermined — one format per prompt.

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
- **A PID is only meaningful in the namespace that read it. GPU UUID and cmdline are the only cross-boundary identities.** The host and the container number the same process differently: `tn exec` sees a rank as 1738493 while `~/bin/pod` sees it as 1382917. Neither is wrong and neither resolves in the other view. Two failures from this on 2026-09-01, in opposite directions: a job queued behind `while [ -d /proc/1302052 ]` used a *host* pid inside the *container*, where it does not exist — the guard was false on its first evaluation, so the job launched immediately onto cards a running probe held and contended with it for ~80 steps; and separately two sessions quoted pid sets to each other that neither could look up. So: "kill by exact PID" implicitly means "by a PID read in the namespace you are about to kill from" — read and kill in the same view. To hand a process to another session, or to guard on one, name it by GPU UUID (`nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory`) plus its cmdline, both of which are machine-wide. **Never guard a launch on `[ -d /proc/<pid> ]` across the boundary**; poll a log string instead, which is namespace-independent.
- **`setsid`, not `nohup`**: `pod` runs through `crictl exec`; when that session ends the kernel kills the whole process group, and `nohup` only blocks SIGHUP. Launch:

```bash
pod "cd /work/aupai && setsid nohup bash -c '<cmd> > runs/x.log 2>&1' </dev/null >/dev/null 2>&1 &"
```

- **Card claims live where the job runs.** `scripts/card_claim.py` reads and writes `runs/claims/` in the tree it is invoked from, so a claim acquired on the pod is released on the pod; a `release` typed on the laptop prints `no claim` and changes nothing (b0, 2026-09-03). `harness launch` acquires for the job's python/torchrun descendant and the monitor releases when it sees the job end (de-30); a claim bound to a shell pid is refused (de-34).
- **`CUDA_VISIBLE_DEVICES`, not `cuda:N`**: fla/Triton kernels launch on the current device; `cuda:1` raises illegal memory access.
- **File transfer into the container: `podput <local> <remote-abs-path>`** (gzip+base64, 100KB cap — the real limit is the remote argv, not a constant). `tn push` lands on the HOST filesystem and is invisible inside the container — never use it to deliver code or data to the repo.
- **Push code via `scripts/pod_push.sh <files>`, never bare `podput`.** It refuses to push a file with uncommitted changes, and re-runs the drift gate after. **It does not pull, and no workflow here should.** Every session shares this tree and this `.git`, so another session's commit is already in HEAD the moment it is made — there is nothing to fetch from them, and `origin` is not how they reach each other. The `--autostash` that used to accompany the pull stashed and restored the *whole* dirty tree, i.e. five other sessions' uncommitted edits, on every push: `git checkout` on a file you did not write, automated. A push copies one session's local state into the pod's global state; in a multi-session tree that state is stale by default (2026-08-30: a push rolled back 3b's `datagen/build_corpus.py` row-group feature, commit e39146e, and its new launcher died on `unrecognized arguments: --rg_mod`).
- `uv sync` after dependency changes.
- **`pod_push` only ever ADDS: a deletion on `main` needs a second explicit step on the pod.** The deleted file stays there and `pod_drift --check` still passes, because the manifest asserts that the files it lists match, never that unlisted ones are absent. 69 files deleted from `main` on 2026-09-02 were all still on the pod with every gate green.
- **Only a `refusing:` line means nothing shipped.** `--all` printing `0 push, 0 delete` is a real sync — the stamp still advances — so "no files moved" is not the failure signal. The refusals go to stdout, where a `| tail -2` eats them and the command reads as success: filter for `refusing|REFUS`, never by position.
- **`pod_drift.py --write` regenerates from HEAD, `--write-index` from the index.** During a merge HEAD is still the pre-merge commit, so `--write` drops the files the merge is *adding* and the manifest describes neither side (2026-09-02: six new test files silently removed mid-merge). Resolve a manifest conflict with `--write-index`.
- **Outbound network: `curl -4`, always.** The pod's IPv6 egress is broken, curl tries IPv6 first, and the failure surfaces as `Errno 99 / Cannot assign requested address` — which reads as "this host is unreachable" and is actually "the local address family is unusable". `urllib` and most probe scripts do not fall back to IPv4. On 2026-08-30 this produced a whole reachability matrix of false negatives and nearly retired the pretraining code slot: with `-4`, `data.together.xyz` (RedPajama-1T's manifest host) answers 200 at 2.66GB per file. `Errno 99` is a local error, never a statement about the remote.
- **What is reachable, measured 2026-08-30 with `-4`:** hf-mirror `resolve/main/<file>` paths 200 but its metadata API 403s (anti-scrape) — so a dataset with a *published URL manifest* is easier to fetch here than one needing the Hub API, the opposite of the usual ranking. `huggingface.co` itself times out on both families.
- **Reachability changes without notice, so a fetcher carries a mirror chain.** 2026-08-31 from ~14:30: `hf-mirror.com` rc=28 (timeout) for the rest of the day; `www.modelscope.cn` 200 in 0.08 s and `/datasets/AI-ModelScope/<name>/resolve/master/<file>` serves HF datasets; `data.together.xyz` 200 all day. The math and CoT fetches stalled for hours on the one dead host. Rule: every HF-hosted source lists `[hf-mirror, modelscope, huggingface.co]` with a verified name map; a 10 s `curl -4` probe per host before fetching and on any timeout/403; failover continues from the same `.part`; `fetch_stats.json` records which host served each file (`t37`).
- **`cd` inside a backgrounded chain stays in it.** `pod "cd X && cmd & followup"` runs `followup` in the original cwd: the `&` backgrounds the whole `cd X && cmd` list in a subshell. Everything that needs the cwd goes inside the chain; everything outside it uses absolute paths.
- **The pod is frozen from a training launch until that run prints its first step.** A launch reads `data/pod_synced_head` and the manifest at startup (`run_ddp.sh:36-41`), and `build_mix` then spends minutes loading token caches before the first step — 156 GB and ~2.5 minutes for `mix_200m_4b`. A push landing inside that window turns the drift gate red on a run that is already committed to the cards, and the failure names the pushed file rather than the push. Whoever wants to push waits for the first step line; whoever launches says so when the run is up (2026-09-02: a `profile_step_cost.py` push landed between p200m_4b_0902's launch and its first step — harmless only because the run reads `train.py`).
- **Check a launch line's shape against `facts/efficiency.json` before it reaches a card.** The fact store already holds the answer for the shapes people try, down to which ranks die first: `eff.microbatch_32_oom` records that micro-batch 32 × accum 1 OOMs at seq 4096 fp8 at 93.8/95.2 GB with ranks 3/6 first, and states the verdict — grow the effective batch via accum, not micro-batch. p200m_4b_0902 launched on 32×1 twice on 2026-09-02 and reproduced that fact to two decimals (93.78 GiB, rank3 first), because nothing reads the fact store at launch time. `harness` check `launch_line_vs_oom_facts` now joins the two sides (44-20, 2026-09-02): a stop-window launch line or a running experiments row matching a recorded OOM config on (dim, layers, batch, accum, seq) is FAIL; grad_ckpt and world are printed for adjudication, never joined on.
- **A fact's `source` names only checkpoints that still exist.** A checkpoint on the pod deletion list (`runs/pod_ckpt_candidates_*.txt`) must be KEEP-claimed by name; a source that names a pruned, zeroed, or misnamed checkpoint is a fact defect, not a footnote. `eff.kda_mla_growth_ratio_l32`'s step1500 was pruned with nothing red and the same day's list nearly took step2000/2500/3000 too (b0, 2026-09-02). `harness` check `ckpt_facts_sources_present` (44-22) joins every fact's source/config against the newest listing and FAILs both classes — `[deletion-candidate]` (on the list, unkept) and `[absent]` (not in the listing: pruned, zeroed, or misnamed) — naming the fact and the file on both sides. A source the fact's own uncertainty/boundary already names as gone is WARN, not FAIL: honest provenance stays visible (three tiers, fb ruling 2026-09-02). Names match exactly: a fact that shortens a name is wrong, and the check says so.

## Ten gate-failure rules (compressed from `docs/lessons/gate_failure_shapes.md`)

| Rule | Shapes | §refs |
|---|---|---|
| Verify premises before acting, sources before citing; a correct conclusion does not certify its argument | 15 | §8 §14 §18 §37 §38 §46 §49 §52 §57 §60 §66 §70 §83 §87 §92 |
| A criterion must express the property asked; test it on known-answer positive and negative worlds before trusting output | 35 | §9 §10 §23 §26 §29 §31 §34 §35 §40 §45 §48 §54 §56 §61 §65 §67 §69 §71 §72 §73 §74 §75 §76 §77 §78 §80 §81 §82 §84 §85 §88 §89 §90 §91 §93 |
| Artifacts carry their producer's identity; missing identity refuses, never rebuilds | 5 | §4 §24 §43 §44 §47 |
| Failures must be loud: checks before the write, raise or exit nonzero, never print-and-continue | 6 | §7 §13 §25 §27 §51 §59 |
| State the vision before the number; outside it, label unmeasured, not absent | 10 | §3 §5 §6 §17 §19 §28 §30 §32 §36 §53 |
| Every number carries its basis: source type, resolution, algorithm; label extrapolation | 12 | §1 §11 §12 §20 §21 §50 §55 §62 §63 §64 §79 §86 |
| Retractions travel as wide as the ruling and name the todos they void; constraints are machine checks, not prose | 5 | §16 §22 §42 §58 §68 |
| Shared resources are explicitly exclusive; co-residency is judged by each implementation's measured cost in seconds against the run's own spend, never by metric class | 2 | §15 §33 |
| Run a deletion candidate before judging it; broadcast the list, delete after 24h unclaimed | 2 | §39 §41 |
| What happened only on the pod did not happen; bring it back to the repo the same day | 1 | §2 |

Full cases live in the shapes doc; new shapes land there first and this table follows.

## Rule coverage

Every rule below maps to a check that enforces it or an explicit reason none can.
`agents_rules_covered` FAILs on a bullet that maps to neither, and the manual count is
ratcheted against a literal — raising it takes a commit saying which rule became
unenforceable. Coverage proves a mapping was made, not that it is honest, so the manual
column states what each check cannot see. A rule that is only prose is one people break
for cause: tonight `harness task` refused to run in a worktree, so "run it in the main
checkout" sent a session into the one tree where sessions overwrite each other.

| Rule | Enforced by |
|---|---|
| Tokenizer frozen 2026-08-29 | `pinned_ids` |
| Vocabulary identity | manual: enforced at load since 7aacbac (2026-09-03): sft_math.py refuses a vocab_id mismatch and prints the matching id; before 7aacbac the guard key was `vocab` and the assert key `vocab_id`, so the check never fired (shape §70) |
| GPUs | manual: card ownership is a controller decision, not a file state |
| A kill is not finished until `nvidia-smi` says the card is free | manual: the rule is an operator sequence -- kill, read the card, kill what remains -- and no artifact records whether the second step happened; lane_respected catches the orphan holding a card now, which is the consequence, not the discipline |
| Lanes: a 7-card training block, and one lane card for everything else | manual: the lane/block split is allocation policy; lane_respected checks the instant, not the policy |
| Small jobs queue on the lane card. They never spill into the block, not even o | manual: queueing is operator behaviour over time; lane_respected catches the instantaneous violation |
| The lane holds one job at a time | manual: same: lane_respected sees now, not the queue discipline |
| When there is no lane card at all — `NGPU=8`, as p500m_ | manual: `scripts/eval_load_cost.py` now records the deciding quantity, host bytes per token cache, measured on the pod 2026-09-03 (22 caches, 247.8 GB), and derives each mix's total: mix_500m is 166.2 GB, reproducing the rule's ~166 GB from file sizes rather than estimates (`160e9 < b < 172e9` asserted in `--selftest`). What stays manual is the tok/s delta. That needs a live training run to differ against, and only three evals have one |
| Judge the cost in seconds against what the run already | manual: how a human reads a log field. The fix that IS checkable is on the instrument — ETA as a window mean, or the per-interval overrun printed beside it — and that edits `train.py`, frozen for p500m_20b_0902 (de-27, stop-window list) |
| Long jobs detach | `no_foreground_pod_training` |
| Language | manual: no automatic judge of whether prose is English or Chinese-for-the-user |
| Shared files | manual: announcing an edit happens in conversation, outside the repo |
| CI gates | CI |
| Derived artifacts carry the fingerprint of what produced them ->R3 | `corpus_fp_matches` |
| Card claims live where the job runs | manual: the claim files sit in the tree the job runs from and no check reads the pod's runs/claims from here; scripts/test_launch_claims.py asserts the launch path acquires and the monitor releases, card_claim.py --selftest asserts a shell pid is refused |
| `CUDA_VISIBLE_DEVICES`, not `cuda:N` | `device_set_honoured` |
| File transfer into the container: `podput <local> <remote-abs-path>` | manual: the 100KB cap is enforced by podput itself, which refuses |
| pod is at ~/bin/pod — not in the default PATH. A session onc | `pod_drift` |
| `tn exec` and `~/bin/pod` are two different filesystem views with the same hos | manual: a fact about the environment; the mistakes it prevents are interactive |
| `setsid`, not `nohup` | `no_foreground_pod_training` |
| Push code via `scripts/pod_push.sh <files>`, never bare `podput` | `pod_drift` |
| Never `git stash` in this repository ->R8 | `no_shared_stash` |
| The index must equal HEAD before you merge: commit your | manual: which order a session ran merge and add in is not recoverable from the repo. What IS checked is the consequence: a wip commit lands on the branch where dirty_aged and the behind-main hook see it. The rule's own history is the reason it stays prose -- the previous version was a correct measurement of the wrong branch shape, and no artifact records which shape a merge had |
| A conflicting path needs a commit first, and read which | manual: same -- the sequence happens in a terminal; the consequence IS checked, a wip commit lands on the branch where dirty_aged and the behind-main hook see it |
| `pod_push` only ever ADDS: a deletion on `main` needs a second exp | manual: the deletion is an operator sequence -- delete here, then delete there -- and the second half happens on a filesystem no check reads; pod_drift compares the manifest against the pod, and a file in neither is invisible to it by construction |
| Only a `refusing:` line means nothing shipped | manual: how a human reads pod_push's stdout; the transcript is not an artifact, so nothing records whether the reader's filter could see a refusal at all |
| `pod_drift.py --write` regenerates from HEAD, `--write-index` from | manual: which flag a session typed is not recoverable from the manifest it produced -- both write the same file, and a manifest built from the wrong side is well-formed |
| Outbound network: `curl -4`, always | `curl_ipv4` |
| What is reachable, measured 2026-08-30 with `-4` | manual: a record of a measurement, not a rule to enforce |
| Reachability changes without notice, so a fetcher carries a mirror chain | manual: fetchers do carry chains; asserting 'a chain is present' would match a comment |
| `cd` inside a backgrounded chain stays in it | manual: a shell fact; no artifact records the mistake |
| The pod is frozen from a training launch until that run | manual: the window is defined by two events in different places -- a launch timestamp on the pod and a push from a laptop -- and nothing records the second. `pod_drift` sees the drift that results, which is the consequence; whether a push landed inside someone's startup window is not recoverable from any artifact |
| cfg_default raises rather than returning None: an annotation | manual: a note on how checks are written, not a rule to enforce |
| The ledger takes names from the scores: --name X attributes | manual: a note on how the ledger reads, not a rule to enforce |
| Each session works in its own worktree on its own branch: gi | manual: worktree topology is per-machine, not in the repo |
| Commit in your worktree as soon as a change works, at most 3 | manual: same deadline as above, enforced by dirty_aged |
| runs/.jsonl ledgers merge by union (.gitattributes); row ide | `no_ghost_running` |
| scripts/pod_push.sh pushes only content reachable from main; | `pod_drift` |
| The shared corpus, checkpoints, and GPUs on the pod are unch | `pod_drift` |
| Run ruff format over a whole file only if you created it. On | manual: reformat scope is a review judgement |
| Commit as soon as a change works, and never later than 30 mi | manual: dirty_aged/untracked_aged enforce the deadline; 'as soon as' is judgement |
| Stage by path, never git add -A / git add . / git commit -a | manual: git history cannot show which command staged a commit |
| A commit that touches a file in data/pod_head_manifest.txt i | `pod_drift` |
| Corpus directories named by any ladder mix (data/mix_scale_ | `ladder_config_frozen` |

51 rules: 17 checked, 34 manual. The count is regenerated from `harness check`'s
`agents_rules_covered` line, not maintained by hand — it was stale at "35 rules: 14
checked, 21 manual" while the code said 36/13/23, which is the same drift the table
itself had before the check began reading it.

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

**Coordination.** Several sessions work on this repository. Announce before editing `train.py`/`sft*.py`/`AGENTS.md`, commit promptly, hand the file back. Commit messages in English, one concern per commit.

**One worktree per session (from 2026-08-31 evening).** Six sessions in one working tree share one index: a file left dirty blocked others' moves four times in one afternoon, staged files were swept into other sessions' commits four times, and a hook built the manifest from another session's staged move. Rules replace none of this; isolation does.

- Each session works in its own worktree on its own branch: `git worktree add ../aupai-<name> -b <name>` (from this repository; the branch starts at `main`). The controller keeps `/Users/bytedance/code/aupai` on `main` as the integration tree and is the only session that commits there directly.
- Commit in your worktree as soon as a change works, at most 30 minutes after touching a file. Merge into `main` at least every 30 minutes: `git -C /Users/bytedance/code/aupai merge --no-edit <name>`; if it conflicts, `git merge main` in your worktree, resolve there, merge again. Never rebase a branch someone else has merged.
- **Never `git stash` in this repository.** `.git/refs/stash` is one stack shared by every worktree — not per-worktree like HEAD and the index — so two sessions stashing in the same window each pop the other's entry, applying a diff they never wrote to a tree it was not made against (e1 and b0, 2026-09-02; nothing was lost, and that was luck). `no_shared_stash` reports a non-empty stack.
- **The index must equal HEAD before you merge: commit your paths, or `git reset`.** A `git merge --no-edit main` refuses with ANY staged change, including on a path the merge does not touch — MEASURED eight ways on 2026-09-02 (`docs/lessons/gate_failure_shapes.md` §49). The rule that stood here said the opposite, because it was measured on a fast-forward: a fast-forward updates the working tree like a checkout and only wants the paths it touches clean, while a three-way merge must write the index and refuses to write over a record that differs from HEAD, related path or not. Once both branches have commits — every time a merge is actually needed — three-way is the normal case. Unstaged non-conflicting changes still merge fine; the shorter rule is that a clean index always does.
- **A conflicting path needs a commit first, and read which path it is.** `git merge` names the file in its `would be overwritten` line. A derived artifact — `data/pod_head_manifest.txt`, which the hook regenerates on every commit, so anyone's commit makes it collide — is nobody's work: `git checkout -- data/pod_head_manifest.txt`, then merge again. A file you wrote gets a path-limited `git commit <paths> -m "wip: ..."` first. A `wip:` commit is yours on your branch; a stash is everyone's. `AUPAI_BEHIND_MAIN_OK=1` is not the way out of either: it commits onto code main has moved past.
- `runs/*.jsonl` ledgers merge by union (`.gitattributes`); row identity is `(name, started)` / `id`, never position. `data/pod_head_manifest.txt` is regenerated by the hook on the merge commit — a conflict there is resolved by regenerating, never by hand.
- `scripts/pod_push.sh` pushes only content reachable from `main`; a branch-only file is refused. CI reads `main`.
- The shared corpus, checkpoints, and GPUs on the pod are unchanged by this; `harness launch` and the allocation file still own them.

**Every delivery has a second reader (user order, 2026-08-31 22:00).** Fixed pairs: de↔ 44, tilerl↔ b0, 3b→ b0, e1→ 3b, fb→ 44; `runs/roster.json` is the roster of record. `harness task done` requires `--reviewer`, a roster member other than the owner, and refuses otherwise. The reviewer writes a row to `runs/review.jsonl` naming the artifact path or failing case they actually opened; a review that names neither is not a review. `review_present` WARNs while a review is pending and FAILs 30 minutes after the close — a sleeping reviewer never blocks a close, and a review that never arrives never stays invisible. Reviews of controller rulings keep the 15-minute window below.

**The controller is reviewed too (user order, 2026-08-31).** Every controller ruling that changes data composition, launches or kills a job, sets a recipe value or threshold, or reports a number to the user is sent to the reviewer session (44) as it is issued; a challenge within 15 minutes must name a failing case or an artifact, otherwise the ruling stands; urgent kills execute first and are reviewed after. Challenges and outcomes: `runs/review.jsonl`; accepted corrections go into the gates doc under the controller's name. The controller made four evidenced errors that day that no one was positioned to catch first.

Three rules from the day one session's `git checkout` erased another session's uncommitted device gate in `scripts/test_arch_compat.py` (recovered only because a copy had been `podput` to the pod minutes earlier):

- Never run `git checkout` / `git restore` on a file you did not write. Undo your own edits by hand.
- Run `ruff format` over a whole file only if you created it. On a shared file, format the lines you touched; a 61-line reformat buries someone else's work and invites the checkout that deletes it.
- Commit as soon as a change works, and never later than 30 minutes after touching a file — path-scoped, `wip:` in the message if unfinished. A commit is not a pod push; nothing runs it until `pod_push.sh`. The working tree protects nothing — `git fsck` cannot recover what was never staged — and in a shared tree a file left dirty for hours blocks every other session's move of it and gets swept into their commits (2026-08-31: three files, four incidents in one afternoon). User ruling 2026-08-31: nothing stays uncommitted.
- Stage by path, never `git add -A` / `git add .` / `git commit -a` in a shared tree. 2026-08-31: a `git add -A` under the message "pod manifest: refresh" swept 26 files and 533K insertions — including 234MB under `data/_corpus_unsanitized/` and five other sessions' uncommitted work — into one commit (d535674). The pre-commit hook (`scripts/hooks/pre-commit`, installed by `harness install-hooks`) refuses staged files >5MB and new `data/` paths not in the allow-list, so the blob never enters history.
- **A hook edit made in a branch worktree does not run until it is merged.** `.git/hooks/pre-commit` is a symlink to `../../scripts/hooks/pre-commit` resolved against **main's** worktree, so every worktree executes main's copy. The consequence, not the mechanism, is what bites: edit a hook in your worktree, commit, watch it not fire, and conclude your change is broken — it is not, it was never loaded. Verify a hook change by running its logic directly against a deliberately-broken input before trusting it (2026-09-01: a readout commit landed with its own selftest red under five green hook lines, and the fix for that ran the old hook too). **This includes the `SELFTEST_FILES` registration, not just hook logic**: adding your file to that set in your own worktree gates nothing, and the hook still prints a `selftests` timing line — a small number reads as "ran, fast" when it means "ran zero of them" (e1, 2026-09-02: `build_agentic_sft.py` was registered on the e1 branch, every commit printed `selftests 0.03s`, and the selftest had never run at a single commit). The test is one line — `readlink -f "$(git rev-parse --git-common-dir)/hooks/pre-commit"`, then grep that file for your own filename; a timing line cannot tell you whether the run was empty.
- The hook runs `--selftest` on staged files in its `SELFTEST_FILES` map. A file carrying a selftest that is not in the map is unguarded: the hook checks what it happens to check, not what the commit changed. Add the path when you add a selftest.
- A commit that touches a file in `data/pod_head_manifest.txt` is pushed to the pod by its committer in the same step (`scripts/pod_push.sh <file>`; it regenerates the manifest and refuses if that changed — commit the manifest and push again). The pod runs the pushed copy, not HEAD; 2026-08-31 the drift gate stopped the A/B launch twice on files another session had committed and not pushed.
- Corpus directories named by any ladder mix (`data/mix_scale_*.json`) are frozen: they carry `build_corpus`'s stamp and every ladder point and A/B reads them. New corpus goes to a new directory that the 30B mix names (`data/corpus/code_rp1t/`, not `data/corpus/code/`). 2026-08-31: ten new shards written into `data/corpus/code/` changed its fingerprint and `_assert_mix_domains` stopped the A/B at startup — correctly.
