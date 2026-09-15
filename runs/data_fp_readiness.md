---
question: r3-final 之后 A/B 续训与 phi SFT 实际会校验的数据指纹，在 pod 上是否全部就绪且一致
status: measured
source: pod /data00 token caches + stamps, ckpt_v41_r3_0914.pt.step38000 cfg/cursor, data/mix_v41_r3.json, data/sft/sft_phi_codeexercises_v42_65m_0914.pt + phi_l3_stub_holdout_manifest.jsonl, 2026-09-15 01:0x UTC；read-only（无 GPU、无生成、无数据写）
---

# data fingerprint readiness — post-r3 A/B + phi SFT

两个下游消费者代码里实际核的指纹，不是名义版本号：

- A/B 续训走 `train.py`：cache 路径解析（`_domain_cache_path` 2138，按 mix 的 `cache_exclude` 加 `.excl<id>` 后缀）、`.vocab`/`.srcfp`/`.seed` 新鲜度（2234-2245）、cursor triple（`_assert_mix_derived_against` 2597）。
- SFT 走 `sft_math.py --check_pack`：`vocab_id`、`holdout_fp`（对 data/eval/holdout_hashes.txt）、fone 三项（sft_math.py:233-304）。

**两组 holdout 指纹互相独立，勿混**（本盘点初稿混过一次）：

| | eval holdout | pretrain 排除 manifest |
|---|---|---|
| 文件 | `data/eval/holdout_hashes.txt` | `data/sft/phi_l3_stub_holdout_manifest.jsonl` |
| 内容 | 305,007 个 HumanEval/MBPP 等评测 hash（+2 注释行） | 273,879 个 code_ultra_l3_stub_dc 文档（确定性 2% 切片） |
| 指纹 | sha256[:16] **10d9c13fd4ffa359** | sha256 前缀 **56d12083dc30bdf2** |
| 谁核 | SFT pack 的 `holdout_fp`，check_pack 逐字节比 | r3 pretrain cache 经 mix `cache_exclude` 排除，保证 SFT 目标未被预训练 |
| 实测 | pack=10d9c13f == live=10d9c13f | manifest 273,879 行 sha56d；r3 mix 带 cache_exclude=56d，loader 读 `.excl` cache |

## 就绪 / 异常表

| # | 对象 | 期望 | pod 实测 | 判定 |
|---|---|---|---|---|
| 1 | `tokens_textbook_claude_v41_dc.pt` | T 臂教科书 cache，amendment_3 基数 2,075 packed rows / 8,501,275 tok | 在，flat int32 8,502,553 token（2075×4097=8,501,275 吻合） | READY |
| 2 | 同 `.vocab`/`.srcfp`/`.seed` | f1f860970d15d623 / e473e748ae3ea4e2 / 42 | 全部精确匹配；srcfp 与 corpus build_corpus_stats fingerprint 一致 | READY |
| 3 | `corpus/textbook_claude_v41_dc/` | 1,938 docs，13-gram 0 hit | rows_final 1938，distinct_problems_hit 0，decontam_fp 0aefe6a2 | READY |
| 4 | SEG137 可训池 | 2,075 − 5% val 103 = 1,972；4ep=7,888 行 | 算术一致；val 切分在 build_mix 启动时发生，go 时 dry-run 再确认 | READY（结构性） |
| 5 | 六 anneal 域 cache srcfp | == r3 cursor 六值 | stub 12ec3cd2 / keep d7b4f3a0 / l2 adf2ff20 / math 4b1469b / en c59c2e42 / cot 0d9f4959，六/六匹配，vocab 全 f1f8 | READY |
| 6 | r3 cache_exclude 链路 | mix 带 56d → loader 读 `.excl` cache | ckpt cfg mix=`data/mix_v41_r3.json`（该 mix stub 带 cache_exclude 56d）；train.py:2708 填 `_CACHE_EXCLUDE`、2143 拼 `.excl`；`.excl56d…pt` 在（19.58GB，srcfp `12ec3cd2\|exclude=56d12083dc30bdf2:273879`） | READY，见注 A |
| 7 | SFT 包存在/形状 | 23,637 rows × 4097 | 在，774,729,109 B，input_ids (23637,4097) | READY |
| 8 | SFT 包 vocab_id | f1f860970d15d623 | 匹配 r3 ckpt | READY |
| 9 | SFT 包 holdout_fp | == live eval holdout 10d9c13f | 精确匹配 | READY |
| 10 | SFT 包 fone | r3 cfg fone=False ↔ 无 values 键 | 两侧均 False，assert 通过 | READY |
| 11 | SFT 包来源未预训练 | r3 吃 `.excl` stub cache（273,879 已排除） | 由 #6 成立；manifest sha56d 在位 | READY，见注 A |
| 12 | 包体 token 口径 | — | build_stats supervised_body **63,703,751（63.70M）**；labels!=-100 独立算 63.97M；名义"65.72M"在文件内无对应字段 | 见 W1，不进任何门 |
| 13 | r3 final ckpt | `ckpt_v41_r3_0914.pt` step 38,070 | 最新滚动 step38000（total 38070，warmup500/warmdown0.65）；final 未生成，SFT RESUME 有 `[ -f ]` 拒启动保护 | PENDING（预期） |

**注 A（#6/#11 的推断链，非直接进程证据）**：容器内此刻 `ps` 只有 defunct python，未能直接读到 r3 进程打开的 fd。推断依据是三处静态事实且互相印证：r3 ckpt 记录 mix=`data/mix_v41_r3.json`；该 mix stub 带 `cache_exclude: 56d12083dc30bdf2`；train.py loader 唯一按该字段拼 cache 路径。ckpt cursor 里 stub srcfp 是**裸** `12ec3cd2` 不构成反证——存进 ckpt 的 `fps[name]=_corpus_fp(ddir)` 永远是裸 corpus 目录 fp（train.py:2428），复合后缀只在 cache `.srcfp` stamp 与 `_same_source` 比对中出现。r3 cursor stub=3,421,631 行也落在 excl 池 1,194,545×~2.86 epoch 的量级内（full 单行池仅 1,218,934 行，排除 24,389=2.00%）。r3 final 落地后可在 final 上再读一次 mix 名做闭环，成本一行。

## 异常 / 需关注（无阻断项）

- **W1 — "65.72M" 名义数无字段支撑。** 文件实测两个口径：63.70M（build_stats supervised_body_tokens）与 63.97M（labels!=−100 求和，27 万 token 差异是 prompt 边界 mask 口径）。65.72M 可能是打包流程的近似/含 doctest 臂的合计，不被 check_pack 或任何门校验。对外引用用 63.70M supervised body；无需改包。
- **W2 — A/B 域名照 cursor，不照 prereg 散文。** prereg arms 段写 anneal 名 `code_ultra_l3_dc`，r3 实际 cursor 与 cache 是 **`code_ultra_l3_stub_dc`**（amendment_3 dry-run 已照 stub 做）。重建 T/C mix 时域名逐字照 r3 final ckpt cursor；六个可绑定 cache 全部是 plain `*_dc` 的 srcfp（cursor 不存 exclude 后缀，见注 A），A/B mix **不应**再带 cache_exclude（amendment_3："All seven domains bind PLAIN"）。
- **W3 — 笔记本某分支工作树有未提交的 holdout 大改，与 pod 无关。** 本地 `data/eval/holdout_hashes.txt` 被改成 29,924 行（fp ff6553a210b4f56a）并改了 `datagen/holdout.py`。pod 与 main 是 10d9c13f（305,007），SFT 在 pod 跑不受影响；但不要在笔记本该工作树上跑 check_pack，会报 holdout_fp 不匹配。

## 结论

13 项：11 READY、1 PENDING（r3 final 未生成，预期）、1 纯口径（W1）。无缺失文件、无 stamp 不匹配、无 vocab/holdout/fone 冲突，SFT 来源未被 r3 预训练这一条由 mix cache_exclude 链路保证。go 时两件事必须在 r3 final 之后：T/C mix 用 `--resume-cursor <final.pt>` 重建并过 dry-run（SEG 冻结 137，域名照 cursor 的 stub 名、不带 cache_exclude）；SFT launcher 的 RESUME 指到存在的 final ckpt。
