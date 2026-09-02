---
question: pod 上 179 个不在 main 的 .py 文件，每一个是什么
status: recorded
source: b0 (3b-4, reassigned from 3b 2026-09-02). Classification is mechanical: a file's
  bytes are hashed as a git blob and looked up with `git cat-file --batch-check` against
  the integration tree. That answers "were these exact bytes ever recorded by git", with
  no commit window and no branch limit -- a hand-written file's blob cannot coincidentally
  exist in the object store.
---

# pod 独有的 179 个 .py：分类，不是清理

**本文只做分类：一个字节都没删，一个文件都没移。**

**交付物是分类和证据，一个字节都没删。** fb 的硬边界二：先知道每个文件是什么，再动它。

## 0. 先纠正两个前提

**一、pod 上没有 `.git`。** `ls -d .git` → 不存在，`git ls-files` → 0 行。所以
"从未进过 git"**在 pod 上判定不了**——必须拿 pod 的文件内容去问集成树的对象库。
第一次尝试在 pod 上跑 `git ls-files` 得到 419（等于全部 .py），那是"没有仓库所以
零个被跟踪"，不是"419 个是孤儿"。**一个空的跟踪集合会让每个文件看起来都是孤儿。**

**二、总数是 179。** main 跟踪 241 个 .py，pod 上有 419 个，差集 **179**。

**我第一版写 214，那是错的，根因是 locale collation。** `comm` 要求两个输入用
同一种排序，而我的两份清单一份在 macOS 默认 locale 下 sort、一份在容器里 sort。
加 `LC_ALL=C` 重排后 214 → 179，**多出来的 35 个是假阳性：它们是 main 当前跟踪的
活文件，字节与 HEAD 完全相同**（`algorithms/rlvr_reward.py`、
`datagen/build_chat_qa.py` 等）。**按 214 那版执行删除，会删掉建 chat_qa 语料的
那个脚本。**

**这个错误的形状值得记**：`comm` 在输入排序不一致时**不报错**，它安静地给出一个
错的差集。两份清单各自都是对的，**错在比较它们的那一步，而那一步没有任何校验**。

168 我仍未复现，口径不明。**记下差异，不假设谁对。**

## 1. 分类结果

| 类别 | 数量 | 处置 |
|---|---|---|
| 字节在 git 历史里，**但路径仍被 main 跟踪** | 1 | **不动**——是活文件 |
| 字节在 git 历史里，路径已退役 | **20** | 真候选，三关照走；**其中 3 个本轮扣下，见 §1.1** |
| **从未进过 git** | **158** | **一律保留**，逐个说明见 §3 |

**158 这个数在修正前后没有变化**，所以 §2 的"118 个在根目录"和 §3 的逐条明细
不受 collation 错误影响。**变的只有"可删"那一侧：56 → 20，缩了 64%。**

**三份清单以文件形式提交，供独立对账**（`docs/lessons/_pod_inventory/`）：
`all_179.txt`、`candidates_20.txt`、`never_in_git_158.txt`。**清单是文件不是消息**
——一份只在消息里的清单，对方对不了账。

### 1.1 本轮扣下的三个

- **`datagen/scan_delivered_holdout.py`** —— 有提交 `2965ded` 但不在 HEAD。它是
  跑出 `FULL 6180175 rows | holdout HITS 0` 那个 P0 结果的脚本，**而那个结果至今
  没有落成 fact 或写进 PROVENANCE**。删掉它，零命中结论再也无法复现。
  **已落盘（3b，`e0dcccf`）**：fact `dq.code_py_starcoder_holdout_zero`，绑定语料
  指纹 `e1a14839e11f3e6f`。**前置满足，但脚本仍建议留**——它是那个结论的复现器，
  而复现器比结论便宜。
- **`datagen/cascade_code_py.py`**、**`datagen/classify_parse_failures.py`** ——
  `git log --all -- <path>` 两个都返回空，但字节在对象库里。**已查清（3b）：它们
  在 3b 的分支上、本轮提交、未并入 main**——正是"未合并分支"那种情况。
  **它们是活的测量脚本，处置是合进 main，不是删。** 保留这一条的记录，因为
  判定方法在这里正好证明了自己的边界：**"字节在历史里"不等于"这条路径的历史可
  追溯"，而两者的处置相反（一个可删、一个该合）。**

**判定方法可复算**：把文件内容按 `blob <len>\0<bytes>` 算 sha1，交给
`git cat-file --batch-check`。命中即证明这些字节被 git 记录过；miss 即证明没有。
**不限提交数、不限分支、不限时间**——比"和最近 N 个提交逐个比"强一档。

**注意 56 这个数的含义**：它说的是**内容**在历史里，不是**这个路径**在历史里。
一个文件可能是某个已删除脚本的副本、或某个被重命名文件的旧版。**所以"可删"仍
需要第二关（glob 与运行时加载器检查），fb 的三关不能只过一关。**

## 2. 158 个从未进 git 的文件，按位置

| 位置 | 数量 |
|---|---|
| 仓库根目录 | 118 |
| `bench_eff/` | 16 |
| `tmp/` | 11 |
| `probes/` | 6 |
| `data/` | 5 |
| `scripts/` | 2 |

**118 个在根目录**，这本身是最重要的发现：**根目录是 pod 上一次性脚本的默认落点**，
而根目录没有任何 gitignore 规则挡住它们，也没有任何检查会注意到它们出现。

## 3. 158 个文件逐条

按最后修改日期分组。**日期是唯一可靠的归属线索**——git 身份是共享的 `cklxx`，作者从历史里查不出来（今晚已证实两次）。


### 2026-09-01 — 30 个

- `bench_eff/t61.py` — Is the inductor-fusion group partly the same quantisation wo
- `bench_eff/t61c.py` — 无 docstring
- `bench_eff/t61d.py` — 无 docstring
- `bench_eff/t61e.py` — 无 docstring
- `bench_eff/t61f.py` — 无 docstring
- `bench_eff/t61g.py` — Split the quantisation work into HEAD (FP8_HEAD=1, not live)
- `bench_eff/t61h.py` — 无 docstring
- `dupscan.py` — Exact-duplicate rate per domain, measured on the bytes the t
- `neardup.py` — Near-duplicate rate per domain: MinHash+LSH, then exact Jacc
- `probes/de_boxcount.py` — 无 docstring
- `probes/de_cells.py` — 无 docstring
- `probes/de_chatml_scan.py` — 无 docstring
- `probes/de_prompt_in_gen.py` — 无 docstring
- `probes/de_restart.py` — 无 docstring
- `probes/de_scorer_diff.py` — 无 docstring
- `t65.py` — Near-duplicate rate per domain, at the normaliser that ACTUA
- `tmp/de_code0.py` — 无 docstring
- `tmp/de_codex1b.py` — 无 docstring
- `tmp/de_constant.py` — 无 docstring
- `tmp/de_ctrl.py` — 无 docstring
- `tmp/de_deg.py` — 无 docstring
- `tmp/de_inturn.py` — 无 docstring
- `tmp/de_math16.py` — 无 docstring
- `tmp/de_restate.py` — 无 docstring
- `tmp/de_sftperf.py` — 无 docstring
- `tmp/de_stopseq.py` — 无 docstring
- `tmp/de_turn_cases.py` — 无 docstring
- `tmp_junk_rate.py` — 无 docstring
- `valcheck.py` — Does domain_loss.py's 'head' actually land in train.py's val
- `xdup.py` — Cross-domain exact duplication: no pass ever compared two do

### 2026-08-31 — 14 个

- `bench_short_conv.py` — t02 short_conv gate: the shifted-multiply-add form must equa
- `bench_short_conv2.py` — t02: does the shifted form beat conv1d UNDER torch.compile?
- `ckpt_cost.py` — t38: checkpoint write cost at d=1024 with optimizer state. L
- `cotprobe.py` — 无 docstring
- `count30b.py` — 无 docstring
- `enprobe.py` — 无 docstring
- `envp.py` — 无 docstring
- `envp2.py` — 无 docstring
- `envpf.py` — 无 docstring
- `fm.py` — 无 docstring
- `msprobe.py` — 无 docstring
- `ratetest.py` — 无 docstring
- `restamp.py` — 无 docstring
- `stamps.py` — 无 docstring

### 2026-08-30 — 38 个

- `audit2.py` — 无 docstring
- `audit3.py` — 无 docstring
- `bench_eff/analyze_ddp_idle.py` — 无 docstring
- `bench_eff/diff_r4_r0.py` — rank4 vs rank0 per-kernel diff: where does rank4 spend more/
- `bench_eff/parse_ddp_ranks.py` — Per-rank compute vs NCCL time: is allreduce wait or real tra
- `bench_eff/parse_kernels.py` — Per-kernel raw output: name, self CUDA time/step, count, gri
- `ce.py` — 无 docstring
- `cert.py` — 无 docstring
- `conn.py` — 无 docstring
- `conn2.py` — 无 docstring
- `conn4.py` — 无 docstring
- `conn4b.py` — 无 docstring
- `conn5.py` — 无 docstring
- `cpcheck.py` — 无 docstring
- `cpcount.py` — 无 docstring
- `cpschema.py` — 无 docstring
- `deg_probe.py` — 无 docstring
- `diag2.py` — 无 docstring
- `diag_rlvr.py` — 无 docstring
- `exactcnt.py` — 无 docstring
- `fence_probe.py` — 无 docstring
- `ftype.py` — 无 docstring
- `measure2.py` — 无 docstring
- `measure3.py` — 无 docstring
- `measure_code.py` — 无 docstring
- `minhash2.py` — 无 docstring
- `minhash_code.py` — 无 docstring
- `msco.py` — 无 docstring
- `nccl_test.py` — Minimal NCCL allreduce probe: no compile, no model, just NCC
- `overlay_check.py` — 无 docstring
- `prep.py` — 无 docstring
- `probe2.py` — 无 docstring
- `probe3.py` — 无 docstring
- `probe4.py` — 无 docstring
- `probe5.py` — 无 docstring
- `probe6.py` — 无 docstring
- `probe_code.py` — 无 docstring
- `rpschema.py` — 无 docstring

### 2026-08-29 — 25 个

- `anyscore.py` — Re-score both probes with 'gold anywhere in the generation'.
- `arith_probe.py` — Bare arithmetic, no word problem. If the model fails here, n
- `arith_probe_fone.py` — Bare arithmetic through the FoNE channel. Directly comparabl
- `bench_eff/bench_eff.py` — Step-time breakdown + full-causal-vs-1024-window A/B for the
- `bench_eff/bench_eff2.py` — Phase timing + full-vs-window A/B for the 200M hybrid on H20
- `bench_eff/bench_opt.py` — Muon/AdamW optimizer phase timing only (CUDA events, 5 iters
- `bench_eff/parse_ddp.py` — Parse the 7-GPU DDP trace: kernel breakdown + cutlass GEMM a
- `bench_eff/parse_trace.py` — Parse a chrome trace into the step-time breakdown table.
- `costprobe.py` — What single-digit splitting costs, measured on the real corp
- `dump.py` — 无 docstring
- `dump5.py` — 无 docstring
- `fmt_probe.py` — Three prompt formats, on the same problems, so the trade is
- `fone_check.py` — Does FoNE actually fix the two defects the harness measured
- `mk_distil_src.py` — Problems for the teacher to solve, with the eval held out.
- `mk_distil_src2.py` — 无 docstring
- `numtok.py` — How this vocabulary splits numbers. Every arithmetic method
- `overlap.py` — Is the probe testing on data the model trained on?
- `probe3_bpe.py` — The BPE half of the FoNE comparison. Same held-out cases, sa
- `probe3_patch.py` — 无 docstring
- `reason.py` — Reasoning quality, separated from arithmetic. Does it pick a
- `reason2.py` — Split the failures: arithmetic versus reasoning.
- `run_scale.py` — 无 docstring
- `scripts/_audit_sample.py` — Build audit samples on the pod: cosmopedia (with bands) + we
- `see_teacher.py` — 请解答下面的数学题。要求：
- `vocabq.py` — Is the vocabulary too small, or split the wrong way?

### 2026-08-28 — 27 个

- `cmp_dist.py` — cosmopedia and web on the SAME ruler: our own quality head.
- `contam.py` — Every new source has to be scanned for eval questions before
- `cosmo_sample.py` — Sample cosmopedia into the same {t,y} shape the web hand-lab
- `cosmo_stats.py` — 无 docstring
- `data/rebuild_gate.py` — Rebuild gate: rebuilt math corpus + synthetic batches vs bot
- `data/sft/contam2.py` — Eval-contamination + near-dup scan (stdlib only, self-valida
- `data/sft/dedup08.py` — Near-dup dedup at Jaccard>=0.8, per-source, greedy (keep fir
- `doclen.py` — 无 docstring
- `failbias.py` — Are the 32% unanswered documents a random sample, or systema
- `fwe_content.py` — What Fineweb-Edu-Chinese actually contains, by reading it ra
- `inspect_cosmo.py` — 无 docstring
- `masked2.py` — 无 docstring
- `mt_probe.py` — 无 docstring
- `nothink.py` — 无 docstring
- `packsmoke.py` — Pack a handful of ChatML examples and verify the mask lands
- `probe27b.py` — 无 docstring
- `rawout.py` — 无 docstring
- `spamdiag.py` — 无 docstring
- `struct_test.py` — 无 docstring
- `tok_cov.py` — Does a small training sample lose rare Chinese characters fr
- `tok_sweep.py` — How much corpus does a 32K BPE actually need? Time vs vocab
- `tput.py` — 无 docstring
- `trad_resid.py` — 无 docstring
- `verify_t2s.py` — 无 docstring
- `verify_vocab.py` — The rebuilt vocabulary, checked on the corpus it will actual
- `vocab_cmp.py` — Why does the new vocab compress better? Read words from the
- `web_tokens.py` — Web's token count after t2s. The old 8.11B was measured on t

### 2026-08-26 — 6 个

- `check_cot.py` — Check SFT raw data for CoT reasoning.
- `check_number_tok.py` — Check how numbers are tokenized — ByteLevel BPE often splits
- `classify_errors.py` — Generate 30 RLVR examples and classify error patterns.
- `prep_math_data.py` — Stage-2 math SFT data prep: hold out 500, dedup, normalize,
- `rlvr.py` — RLVR: GRPO on 218K Chinese math problems with verifiable \\b
- `scripts/refix_math_answers.py` — Recompute the gold answer of every data/math/*.jsonl row wit

### 2026-08-25 — 5 个

- `debug_rlvr_gen.py` — Debug: see what model generates for RLVR math prompts.
- `diag_tokenizer.py` — Diagnose tokenizer ByteLevel issue on pod.
- `eval_ppl.py` — PPL evaluation: computes perplexity on held-out validation s
- `gen_test.py` — Quick generation test with ckpt_sft.pt.
- `test_boxed.py` — Test if model can generate \\boxed{} with greedy decoding.

### 2026-08-24 — 10 个

- `analyze_data.py` — Sample and analyze data distribution before cleaning.
- `check_repetitive.py` — Check what's being flagged as repetitive.
- `data/raw/fable5_sft/scripts/analyse_fable5.py` — 
- `data/raw/fable5_sft/scripts/clean_fable5.py` — 
- `download_data.py` — Download Chinese datasets for expanded training.
- `download_sft.py` — Download SFT datasets from HuggingFace and save to /work/aup
- `download_sft2.py` — Download SFT datasets from HuggingFace via mirror, with retr
- `eval_mmlu.py` — Simple MMLU-style eval: log-likelihood scoring for multiple
- `fp8_compute.py` — 
- `test_sft.py` — Quick test: load SFT checkpoint and generate text with fla m

### 2026-08-23 — 3 个

- `debug_nan.py` — Debug NaN: layer-by-layer check.
- `test_triton.py` — Test Triton scan kernel against PyTorch reference.
- `triton_scan.py` — Triton kernel for Gated Delta Net recurrence scan — 10-50x f


## 4. 有 docstring 的比例

72 / 158 个带 docstring。**没有 docstring 的那些是最难判的**：
一个 `audit2.py`／`ce.py`／`conn4b.py` 这样的名字，加上没有说明，
**除了写它的那次会话，没人能说出它做过什么**。而会话已经结束。

## 5. 不能做的事，写在这里

- **不删任何 §2 里的文件。** 它们的字节在 git 里找不到对应物，**那是唯一的一份**。
- **不移动它们。** 移动会改变 `pod_drift` 的读数，而发车正在用那个读数。
- **56 个"可删"的也不在这次动。** fb 的三关只过了第一关。

## 6. 下一步（发车后）

1. 56 个字节在历史里的：过第二关（glob／加载器引用检查）和第三关（59 门 selftest），再删。
2. 158 个从未进 git 的：**先按 §3 的日期分组找当事人**。fb 的规则是活的进 git、
   死的删（须先证明与 git 历史逐字节相同）、不确定的保住并说明查过什么——
   **这 158 个全部属于第三类，因为它们定义上就证不出"与历史相同"。**
3. **根目录需要一条规则。** 118/158 落在根目录不是巧合，是没有任何东西阻止它。

