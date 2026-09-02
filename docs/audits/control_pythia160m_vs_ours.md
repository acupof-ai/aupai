---
question: 同算力下，我们的 206M 架构+配方 vs Pythia-160M，在同一份 SFT 数据上谁的留出 loss 更低
status: open
source: pod 2026-09-02/03。两臂同时在跑（我们臂卡 0、对照臂卡 1 三点扫描）；本文先落表头与口径，数字回填。e1，fb 指派
---

# 对照实验：同一份 SFT 数据，我们的 206M vs Pythia-160M

## 1. 表头（结论之前必须先读的口径）

| | 我们臂 | 对照臂 |
|---|---|---|
| 模型 | `ckpt_p200m_4b_0902.pt` | `EleutherAI/pythia-160m` revision **step2000** |
| 参数（总） | **206,128,200** | **162,322,944** |
| 参数（非 embedding） | **172,508,232** | **85,056,000** |
| LM head | **tied**（与 `tok.weight` 共享，`train.py:472`） | **untied**（`embed_out` 是独立的 38,633,472） |
| 预训练 token | **3,999,997,952** | **4,194,304,000**（2000 步 × 1024 × 2048） |
| SFT pack（文本） | `control_sft_text_train.jsonl`，sha256 `3b443c60…` | 同一份，同一个 sha256 |
| 留出集（文本） | `control_sft_text_heldout.jsonl`，sha256 `85bd9016…` | 同一份 |
| 留出 example ids sha256 | `70f88f6ee3433a7d` | 必须相同 |
| SFT token | **201,396,229**（49,157 行 × 4097） | 待填（自己的 tokenizer 分词后） |
| SFT epoch | **1** | **1** |
| seq | **4096** | **1024**（Pythia max_position 2048） |
| 超长丢弃 | **1,113** | 待填 |
| 优化器 | Muon（2D 矩阵）+ AdamW（embedding/scalar） | AdamW（全参数） |
| lr | 既有配方 `--lr_scale` 默认 | **三点扫描 {3e-5, 1e-4, 3e-4}，取留出 loss 最低** |
| 词表 | 32,773（`vocab_id 0bce3584bc24f255`） | 50,304（tokenizer 50,277 + 2 个 ChatML special） |

**参数口径两列都报，不许只放一个。** 拿 206.1M 对 162M 是一回事，对 85M 是另一回事。
**方向对我们不利也照写**：对照模型比我们**小**，不是大。

**而且两侧的"非 embedding"不是同一个量，因为 head 一边 tied 一边没有。** 我们 206,128,200 里
embedding 只算一份 33,619,968（head 与 tok 共享权重；checkpoint 存了两份张量，state_dict 求和
得 239,748,168，减去重复的 head 才是 206,128,200 —— 这个差正好是一个 embedding 矩阵）。Pythia
的 `embed_in` 与 `embed_out` 是**两份独立权重**，各 38,633,472。所以：

- **总参数**：206,128,200 vs 162,322,944 —— 可比，我们大 27%
- **非 embedding**：172,508,232 vs 85,056,000 —— 可比，我们大 103%
- **"embedding 占比"**：我们 16%，Pythia 48% —— **不可直接对比**，因为 Pythia 把 head 又算了一遍

报表里两行都给，并把 tied/untied 标出来；只报"非 embedding"会让我们看起来大一倍而不解释为什么。

**token 量相近但不对齐**：Pythia 多 **4.9%**。写"相近（+4.9% pythia）"，不写"对齐"。

**revision 边界**：step2000 由下载 URL 路径推断，**config.json 里没有 step 字段**，文件内容无法自证是哪一步。

## 2. 这个对照能支持什么结论，不能支持什么

**能**：在同一份数据、同一 epoch 下，"我们的**数据+架构+配方合起来**"与一个同期公开基线的差距。

**不能**：把差距归因到架构、数据、配方中的**任何单独一项**。三处必然不同，每一处都是一个混杂：

1. **优化器不同**。我们用 Muon 带 2D 矩阵；在外来模型上套 Muon 是在数据之外加第二个干预。
2. **lr 不同，且不可换算**。Muon 的更新是正交化过的，它的 lr 与 AdamW 的 lr 不在同一标尺。（我第一版把 `Cfg.embed_lr × lr_scale = 1e-2` 当成"同量级"给了对照臂，**实测发散**：4 步 loss 11.52 → 13.27。1e-4 才下降到 2.49。）
3. **seq 不同**（4096 vs 1024，Pythia 的位置上限是 2048）。两侧的超长丢弃数都报。

**对照臂做三点 lr 扫描，就是为了堵住"对照没调好"这一条**。选择依据是**留出 loss**，不是训练 loss —— 用训练 loss 选会挑出记得最狠的那个 lr，那样反驳只是从"没调好"变成"挑着过拟合选的"。

## 3. 单位：每 supervised byte，不是每 token

两侧 tokenizer 不同，**同一份留出文本在两侧是不同的 token 数**。各自除以自己的 token 数，比的是两个不同的量。字节是两侧唯一共享的分母，而**只有 completion 参与 loss**（prompt 全 mask），所以分母是 **supervised bytes**。

留出集：**10,641 examples**，我们臂 989 行 / 4,051,933 token / **supervised bytes 12,266,806**。
两臂的分母是**同一个函数**（`eval_heldout_ours.supervised_bytes`，对照臂 import 它），不是两份实现。

per-token 也会报，标注**不可跨臂比较**，因为它是训练日志里那个数。

## 4. 数据构成（从文本实测，不要读 `src` 字段）

```
源 856,153 行 → 去重 532,027（去掉 324,126 重复）
  train    521,386 行  799,098,158 B
  heldout   10,641 行   16,190,349 B（严格每 50 条，与 train 重叠 0）
leak check: humaneval 0/164 命中、code_holdout 0/156 命中
```

**内容构成（6,000 行全文件抽样）：36.5% 数学应用题、8.1% 带代码围栏。**

**`src` 字段会误导，不作构成依据。** 它记的是"哪个列表先产出这一行"。两个源列表的重叠比明面上多——`prepare_sft.SOURCES` 里已含 `school_math_r1_zh.jsonl`，其内容与 `prepare_sft_math` 的 `school_math_train.jsonl` 相同——所以去重保留前者，tag 读出 99.8% code_general / 0.2% math_cot，与实际内容矛盾。

## 5. 结果

待回填。两臂完成后写入：各自 held-out loss/supervised byte、对照臂三点表与选中点、以及**训练 loss 会不会选出不同的点**（会的话说明这个扫描起了作用）。

## 6. 过程中修掉的、会污染这个结论的东西

按被发现的顺序，每条都在上卡前：

| 缺陷 | 后果（若未修） |
|---|---|
| pack 未洗牌（`prepare_sft` 洗，我这个没洗） | 两臂各按来源顺序训练，最后看到的来源主导结果；且与 `be.sft_v3/v4/v5` 不可比 |
| 留出集由每臂各自切 | 我们臂会训练在对照臂的验证集上，且多训 2% |
| `resize_token_embeddings(len(tok))` | 把 Pythia embedding 从 50,304 缩到 50,279，**丢 25 行训练过的权重** |
| 对照臂 lr 1e-2 | 发散；结论会变成"我们赢"而实际是对照炸了 |
| leak 检查在空 population 上返回 clean | 0 行 pack 被标记为"干净" |
| `check_sft_ready` 带子 `(0.5, 60.0)` | 拒绝 repo 里**每一个**真实 pack（实测都在 79.3%） |
| `run_sft.sh` 硬编码卡 0 | 两臂撞同一张卡，表现为"对照很慢" |
| `card_claim` 记 `os.getpid()` | 锁永远批准；两个 job 都拿到同一张卡 |
| `sft_math.py` vocab assert 守卫读错键 | 错词表 pack 会静默以 ~4 倍 loss 训练 |
| `run_sft.sh` 从不传 `--hypothesis` | exp 行的假设永远空白 |

最后两条不是这个实验引入的，是既有代码在"单卡并发 + 独立 gate"这个新用法下暴露的。
