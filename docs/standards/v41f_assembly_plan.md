# v41f 整网装配接线 — 施工顺序设计（P0，只设计）

Status: PROPOSAL for fb + 98. 本 PR 只有这一份文档，不改任何 `.py`。前置：#454（DSpark 草稿块）、
#447（训练 ckpt）、#456（indexer STE）三篇返修合并之后才开工；`v41f/model.py`、`v41f/train.py`
是共享文件，由 de 一人串行接。

现状（main `0ba88c03`，实测）：`tests/v41f` 全绿 **87 passed in 54.20s**。本文件要保证的是——
每一步接完之后，这 87 条里**属于该步的那部分必须保持绿**，且每步只动该步该动的东西。

---

## 0. 接线前的可复现基线

```bash
python -m pytest tests/v41f -q          # 0ba88c03: 87 passed in 54.20s
python tests/v41f/p0_selftest.py        # P0 逐模块对拍
python tests/v41f/p1_selftest.py        # P1 整 Block / 整网
python scripts/v41f_param_count.py      # 904,583,784 total / 210,950,760 active
```

`v41f_small()` 四个子系统全部关闭（`engram_layer_ids=()`、`n_mtp_layers=0`、
`dspark_block_size=0`、`dspark_target_layer_ids=()`），所以**今天的整网对拍门是在所有新机制关闭下
取得的**。接线阶段的第一个风险就是这个：把某个子系统打开之后，整网 logits 会变，而
`test_whole_model_logits_allclose` 拿的是 `v41f_small`，它不会告诉你任何一个新机制接错了。
每一步的对拍策略因此必须**新开一对配置**（见 §2 的 general 规则），不能靠"旧门还绿"。

---

## 1. 接线顺序总表

顺序由三件事定：谁的依赖最少、谁的**形状**会改动共享文件、谁会把 ckpt 格式带进来。

| 步 | 内容 | 动到的文件 | 前置 |
|---|---|---|---|
| **A** | Engram 接进 `V41FModel`（`self.engrams` 槽 + `engram_hash`） | `v41f/model.py` | #447 可能已落；先落者掌术语 |
| **B** | DSpark draft 接进 `V41FModel`（`self.mtp` + 训练入口） | `v41f/model.py` | **#454 合并** |
| **C** | indexer STE 接线（非默认路径） | `v41f/attention.py`（+ `v41f/indexer_ste.py`） | **#456 合并** |
| **D** | ckpt 覆盖三个新子系统 | `v41f/ckpt.py`（+ `v41f/master.py`） | **#447 合并**，A/B/C 之后 |

A 与 B 都改 `v41f/model.py`，**串行**；C 不动 `model.py`，可在 B 之后或与 B 并行，但它改
`attention.py`，与任何正在改 `attention.py` 的会话互斥；D 最后，因为它的前置是"参数集合已经定形"。

---

## 2. 每一步的前置依赖与回归门

### 2.1 通用对拍规则（每一步都适用）

新增机制**不能**只用 `v41f_small` 验。每步要开一对配置：

- `v41f_small(**over)`：**机制开、形状仍小**（层数/专家数不变），用来和 ref 对拍；
- 生产 `V41FConfig()`：形状全开，只验**不 NaN、参数对得上**，不验数值（ref 在 CPU 上跑不了这个
  形状——1982 个参数叶子、48 专家、4096 seq）。

理由是仓库已经吃过一次的亏：`_model_args` 手工镜像时漏了 `route_scale`/`norm_eps`，整网 bf16 差
0.70，被当成"bf16 路由噪声"追了很久。**单一形状来源**（从同一份 `v41f_small()` 派生 ModelArgs）是
那次修好的根因，装配阶段照抄这条纪律。

### 2.2 步 A — Engram

**前置（两条，都是硬前置）：**

1. **`engram_compressed_vocab_size` 必须由我们自己的 tokenizer 实测填入。** 现在 config 默认
   `0`，而 `NgramHashState.__init__` 有 `assert vocab_size == args.engram_compressed_vocab_size`
   （`v41f/engram.py:138`）。0 会让构造直接炸——这是**好**的失败（loud），但意味着必须先跑
   `build_compressed_token_map(data/tokenizer.json)` 并把实测值写进配置。charter §6 已经点名这条：
   哈希乘数由压缩词表大小派生，错配会静默整体重哈希。
2. **`V41FConfig` 缺 `engram_num_embeddings` 字段，而 ref 的 `ModelArgs` 有。** 实测：

   ```
   ModelArgs 有：engram_compressed_vocab_size, engram_head_dim, engram_layer_ids,
                engram_max_ngram_size, engram_n_heads, engram_num_embeddings,
                engram_pad_id, engram_vocab_size
   V41FConfig 有：除 engram_num_embeddings 之外的全部
   ```

   `EngramLayout.from_args` 读 `args.engram_num_embeddings`（`v41f/engram.py:120` 与
   ref `engram_ref.py.ref:122`），而 `layout.num_embeddings[i]` 就是 `nn.Embedding` 的行数
   （`v41f/engram.py:194`）。所以：

   - `EngramLayout.from_args(cfg)` 直接 `AttributeError`（响亮，好）；
   - 更麻烦的是 **§2.1 那条"单一形状来源"机制结构上带不动这个字段**。`test_p1_model._model_args`
     用 `dataclasses.fields(ModelArgs) ∩ asdict(cfg)` 派生（`:37-46`），字段只存在于 ref 一侧，
     交集必然丢它——ref 侧落回默认 `()`，我们的侧得另找路子填。**两侧就在"表的行数"这个字段上
     分叉**，而这正是 engram 唯一会静默配错的量。

   实测：生产档（`engram_layer_ids=(1,)`、`max_ngram=4`、`n_heads=4`、`engram_vocab_size=65536`）
   素数桶和是 **786,862**。做法：把 `engram_num_embeddings` 加进 `V41FConfig`（ref 也是配置字段，
   加了更忠实），值由 `EngramLayout.from_args` 派生后写回，交集就能自动带上两侧。

   顺带一条干净的基线：今天 `V41FConfig` 的字段**全部**存在于 `ModelArgs`，交集丢弃集合为空
   （实测 `[]`），所以这条纪律在此之前从没暴露过短板——engram 是第一个。

3. **`Engram.q_weight` / `k_weight` 的精度是"默认 dtype 的产物"，不是指定的；而 ref 的忠实值是 bf16。**
   `v41f/engram.py:197-198` 用 `torch.ones(...)` 建，实测在 bf16 默认下是 **bfloat16**（同
   `embed.weight`/`wkv.weight`）。**ref 侧同样如此，且这是可证的**：

   - ref `Engram.__init__`（`model_ref.py.ref:345-346`）也是 `nn.Parameter(torch.ones(...))`，
     **不在任何 `set_dtype` 块里**；
   - ref 全文只有一个 `set_dtype` 用在模型构造上：`model_ref.py.ref:940` 的
     `with set_dtype(torch.float32):`，包的是 **HC 的六张表**（`:941-946` 的 `torch.empty(...)`）；
   - ref 的 `Linear` 是显式的（`:218` `dtype = dtype or default_dtype`，`default_dtype` 在
     `:1191` 定为 fp8 或 bf16）；
   - ref 的进程默认 dtype 是 bf16（`:1296`）。

   所以 ref 里三类的处理**各不相同**：Linear 走 `default_dtype` 显式、HC 表 `set_dtype(fp32)`
   显式、engram 的 `q_weight/k_weight` 走 ambient。忠实值因此是 **bf16**，
   `q_weight.float() * k_weight.float()`（ref `:348`，我们 `engram.py:208`）就是 ref 在数学里
   把 bf16 提回 fp32 的地方。

   **要修的是"隐式"，不是"精度"**：给显式 `dtype=torch.bfloat16`，让它不随谁的 ambient 漂。
   若为了训练稳定性决定改成 fp32，那是一条**偏离忠实**的 v41f 自定决定，要按 DSpark 训练形态
   那样记为自定并给理由——不能写成"对齐 head/HC 的原生 fp32 处理"，因为 ref 并没有把两者同等
   对待（HC 显式包了 `set_dtype`，engram 没有）。这条差异我们在 `v41f/hyperconn.py:60` 的注释里
   已经记录过一次，是同一个判据的正例。

代价记录：786,862 × 128 = **100.7M 参数**落在层的 engram 表里，是当前 904.6M 的约 11%。charter §3
写的是"engram 表当前 rows=0（待 tokenizer 素数桶，计 0）"，接线时必须重跑
`scripts/v41f_param_count.py --engram-rows 786862` 覆盖那一段数字。

**接线内容：**
- `V41FModel.__init__`：`self.engram_hash = NgramHashState(cfg, EngramLayout.from_args(cfg), tokenizer)`，
  `self.engrams[i] = Engram(cfg, i, layout)`（仅 `i in cfg.engram_layer_ids`）。
- `forward` 里那两行**已经按 ref 写好了**（`model.py:88-90`），不需要改控制流。
- tokenizer 从哪来：`NgramHashState` 要的是 `tokenizer.backend_tokenizer.decode/id_to_token`
  （`v41f/engram.py:55,59,61`），即一个 HF `tokenizers.Tokenizer`。**`v41f/` 下目前没有任何
  tokenizer 加载函数**（`grep` 为空），仓库里 `data/tokenizer.json` 在 pod 上且 gitignored。所以
  这一步要定一个口：`V41FModel(cfg, tokenizer=...)` 显式传入，**不要**在 `model.py` 里读磁盘路径。

**回归门：**

| 门 | 内容 | 期望 |
|---|---|---|
| G-A1 | `tests/v41f/test_p0_engram.py` 全部 | 保持绿（哈希逐位、门控 1e-5、cache 前缀） |
| G-A2 | `test_whole_model_logits_allclose`（`v41f_small`，engram 仍关） | **必须仍是 0.0**——开机制不能动关机制的路径 |
| G-A3 | 新增：`v41f_small(engram_layer_ids=(1,), engram_compressed_vocab_size=<实测>, engram_num_embeddings=<派生>)` 对 ref `Transformer` 整网 logits | bf16 全序列，逐位或 atol 5e-2 |
| G-A4 | 新增：注入点位置——engram 在 Block **之前**、且**早于** target hidden 读取（ref `:1258-1265`） | 用 spy 抓 `h` 的 `data_ptr`/数值，独立重算 |
| G-A5 | 新增：`engram_layer_ids=()` 与 `(1,)` 的参数集合差 == 恰好该层的 engram 参数 | 差集断言，不是计数 |
| G-A6 | `scripts/v41f_param_count.py --engram-rows 786862` 与整网 `sum(p.numel())` 对账 | 相等 |

G-A3 的对拍要用**同一份 `v41f_small()` 派生 ModelArgs**（`_model_args` 现有函数），只 over 两个
engram 字段——这是 §2.1 那条纪律的具体落法。

**已知会红、且要一并改的：** `tests/v41f/test_p1_model.py::test_param_count_reconciles_to_script`
现在断言 `v41f_small` 的参数按分量对账。engram 一开，分量表要加一行；这条测试的**结构**不动，数值
按新配置重算。

### 2.3 步 B — DSpark draft

**前置：**

1. **#454 合并**，且其 P0 已修（draft 只能看 `window_size` 个 main 槽——`dspark_causal_topk` 必须把
   main 索引限制到最近 `window_size`，不是全量 `arange(main_len)`）。这是**接线前**必须定死的，因为
   接线之后这个函数就是训练路径的一部分。
2. `v41f/config.py` 补 `dspark_noise_token_id`（#454 会带进来；main 上现在没有这个字段，而
   `forward_embed` 读它）。
3. `v41f_small` 已经有 `dspark_target_layer_ids=(8,)` / `dspark_block_size=5` / `n_mtp_layers=1`
   的**生产**默认值，但 small 把三个都置空。接线时的对拍配置要重新打开它们。

**接线内容（按 ref `Transformer.__init__:1207-1213` / `forward:1276-1282`）：**
- `self.mtp = nn.ModuleList([DSparkBlock(cfg, i, n_target, max_batch_size) for i in range(cfg.n_mtp_layers)])`；
- `self.mtp[i].embed = self.embed`、`self.mtp[i].head = self.head`（ref 在 `:1212-1213` 做绑定）。
  **这是 Modules 赋值，会把 `embed`/`head` 注册进 `mtp[i]` 的 `_modules`。** 实测：

  ```
  keys: ['emb.weight', 'child.emb.weight']        len(sd) == 2
  same storage (data_ptr): True
  ```

  两个 key、同一块 storage。对 `state_dict()`/save 来说这是**写两份**（磁盘 dedup 是另一回事），
  对 `strict=True` 的 load 来说是**必须两份都在**，任何一边被删都会报 Missing key。接线的选择是
  "排除 `mtp.*.embed/head`（embed/head 只存顶层）还是接受两个 key"——**必须先定**，因为它同时决定
  ckpt 的键集、#447 `param_meta` 的键集校验、以及步 D 的 G-D1。**这一点本文件不替你决定**，见 §7.3。

  另一条（实测 4）：**`v41f/block.py` 没有 `engram` 属性**，engram 挂在模型级 `self.engrams`
  （`model.py:76`）。所以 ref 的 `layer.engram` 写法在这里对应 `self.engrams[i]`，是 2.3 之前
  就有意做的结构差异，不是接线要补的东西。
- `main_hidden` 的**收集点已经写好了**（`model.py:91-93`，Block **之前**、`h.mean(dim=2)`），
  与 ref `:1264-1267` 一致，且有 `test_dspark_target_hidden_is_pre_block_attn_input` 钉着。接线
  **不要动这三行**。
- 训练入口：`forward_train_embed`（#454 已提供）取 `main_hidden` + gold ids。整网的
  `forward` 保持只返回 `(logits, main_hidden)`；MTP 的 loss 在 trainer 里组合，**不塞进
  `v41f/loss.py`**（该文件只管主 CE；多 token loss 的加权是 v41f 自定，需独立 prereg）。

**回归门：**

| 门 | 内容 | 期望 |
|---|---|---|
| G-B1 | `tests/v41f/test_p0_dspark.py` 全部（#454 的 15 条） | 绿，且含修好的窗口 mutant |
| G-B2 | `test_whole_model_logits_allclose`（`v41f_small`，dspark 仍关） | **仍 0.0** |
| G-B3 | `test_dspark_target_hidden_is_pre_block_attn_input` | 绿且未被削弱 |
| G-B4 | 新增：`v41f_small(dspark_block_size=5, dspark_target_layer_ids=(k,), n_mtp_layers=1)` 下，主干 logits 与"无 mtp"配置**逐位相等** | mtp 不进主干前向，接了不该改主干输出 |
| G-B5 | 新增：整网 `state_dict()` 的 key 集合 == 主干 ∪ `mtp.*`，且 `mtp.*.embed/head` 的处置与 §7.3 的裁决一致 | 排除规则可断言 |
| G-B6 | 新增：`main_hidden` 只有 1 个 target 时 shape `[b,s,dim]`，2 个时 `[b,s,2*dim]`，且 `main_proj` 输入维度 == `dim*n_target` | 拼接维是 last dim（ref 如此），别接成序列维 |
| G-B7 | 生产档构造不 NaN（前向一次） | 只需有限性 |
| G-B8 | 新增：`mtp.*` 的参数在 optimizer 组中的成员身份按 #447 的规则明确（tied embed/head 不重复计数） | 参数计数与 optimizer 组一致 |

### 2.4 步 C — indexer STE

**前置：#456 合并且其 P0 已修**——`surrogate` 必须真的接进 loss 路径。当前设计里的写法实测
`wq_b.grad is None`（见 #456 评审）。可用的构造（已验：forward 逐位相等、`wq_b.grad` 有限非零）：

```python
a  = torch.softmax(attn.masked_fill(hm.unsqueeze(2) == 0, -inf), dim=-1)   # 硬选中的稀疏注意力权重
p  = a + (torch.softmax(sc, dim=-1).unsqueeze(2) - torch.softmax(sc, dim=-1).detach().unsqueeze(2))
```

**插入点（这是本次预研要回答的问题）：** 不在 `model.py`，在 `v41f/attention.py:200` 那一行

```python
o = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale)
```

`idxs` 在 `:198` 由 `[window_idxs ; comp_idxs]` 拼成。STE 要替换的是**那条调用**对压缩段权重的
处理：窗口段没有 score、不参与 STE，压缩段的 `a` 由 `sc` 决定。所以 seam 的落点是
`Attention.forward` 内部：`_compress()` 已经返回了 `idxs`，还要**同时**把 `sc` 递出来（`_compress`
现在的返回是 `(compress_kv, idxs)`，`:177` / `:180` 两处 return），于是 `forward` 里按
`compress_ratio` 与开关决定走 `sparse_attn` 还是 `sparse_attn_ste`。

这条与 §2.1 通用规则冲突的一点要说清：STE 是**训练专属**路径，`eval()` 与推理永远走原调用，所以
"关机制的路径仍逐位相等"这条门在 C 步是**最强**的（G-C2 直接跑 off 路径对比）。

**回归门：**

| 门 | 内容 | 期望 |
|---|---|---|
| G-C1 | `tests/v41f/test_p0_indexer.py`、`test_p0_attention.py` | 绿（硬 topk、两层索引、RoPE 位置） |
| G-C2 | `indexer_train_mode="off"` 时，选中 idxs 与 logits 与未接线前**逐位相等** | `torch.equal` |
| G-C3 | `v41f_small` 整网 logits 仍 0.0 | 默认 off ⇒ 完全无影响 |
| G-C4 | `"ste"` 时：forward 逐位相等、`wq_b`/`weights_proj` grad 有限非零、其余参 grad 不变 | 按 #456 §4 gate 2/3/4 |
| G-C5 | seam 拿到的 `sc` 与硬 topk 消费的是**同一对象**（`data_ptr`） | 防重算（fed-dead-weight 形状） |
| G-C6 | 生产档：`ste` 打开时 indexer 参数进入 optimizer 组，off 时不进 | 与 #447 的成员规则一致 |

### 2.5 步 D — ckpt 覆盖

**前置：#447 合并** + A/B/C 全部落定（参数集合已定形）。#447 的 §1.1 分组需要在接线后重算：
- compressor 的 fp32 分组（#447 评审 P0-1）；
- **engram 参数**：表（`embed.weight`）与 `wkv`、`q_weight`、`k_weight` 实测在 bf16 默认下都是
  bf16（`v41f/engram.py:194-198`，只有 `head` 是显式 fp32）。**这四项的忠实值就是 bf16**（ref 的
  `q_weight/k_weight` 是 ambient，而 ref 进程默认是 bf16，见 §2.2 前置 3）。要加的是显式
  `dtype=torch.bfloat16`，让它不随 ambient 漂；分组按 `param.dtype` 派生即自动归 bf16-native，
  无需在 #447 的名单里单列。
- **`NgramHashState` 的 buffer 全部 `persistent=False`**（`primes/offsets/multipliers/token_map/cache`），
  即重建、不进 ckpt。这与现在的 `ckpt.py` 行为一致，但 #447 的 `param_meta` 生成器如果按
  `state_dict()` 枚举，会**漏掉**它们——不会有错，但要确认是"设计如此"而不是"没看见"。
- `mtp.*.embed/head` 的排除规则（见 G-B5）。

**门：** G-D1 三个新子系统的参数集合与 `param_meta` 键集相等；G-D2 存取往返后 `test_p1_ckpt.py`
全绿；G-D3 `v41f_small`（全关）的 ckpt 与今天**逐位兼容**（老 ckpt 能开）。

---

## 3. 逐步的"整网 bit-exact 门"清单（汇总）

一条原则：**每一项新机制都必须有一条"关着它时整网逐位不变"的门**，和一条"开着它时与 ref 对上"的
门。两者缺一，接线错误就会以"数值差不多"的形式藏进来。

| 机制 | 关着时（回归） | 开着时（对拍） |
|---|---|---|
| Engram | `v41f_small` 整网 0.0（G-A2） | G-A3：`engram_layer_ids=(1,)` + 实测压缩词表 + 派生行数，对 ref |
| DSpark | 主干 logits 与无 mtp 逐位相等（G-B4） | G-B1+G-B6：`test_p0_dspark` + 拼接维/参数命名空间 |
| indexer STE | off 时 idxs/logits 逐位相等（G-C2） | G-C4/G-C5：grad 非零且只落在 indexer |
| ckpt | 老 ckpt 可开（G-D3） | G-D1：键集 == 参数集 |

**今天 87 条全绿是这四步的地板，不是天花板**：它们的配置里三个机制都是关的，所以每一步都要**新增**
那两条门，而不是指望旧门变红来发现问题。

---

## 4. main_hidden 收集点（明确写死，接线时不要动）

`v41f/model.py:91-93`：

```python
if i in self.target_layer_ids:
    main_hiddens.append(h.mean(dim=2))
h, pre_mix, state = layer(h, 0, pre_mix, state)
```

三条契约，逐条都有 ref 行号：
1. **读的是 Block 的输入，不是输出** —— append 在 `layer(...)` 之前（ref `:1264-1267`），已由
   `test_dspark_target_hidden_is_pre_block_attn_input` 钉住（拿 spy 抓真实输入 + 反恒真 gap 1.30）。
2. **`h.mean(dim=2)`** —— 把 `hc_mult` 路残差流压成一路，得到 `[b,s,dim]`。
3. **`torch.cat(main_hiddens, dim=-1)`** —— 拼在**最后一维**（ref `Transformer` 同样 `dim=-1`），
   所以 `main_proj: dim*n_target → dim`。多个 target 层时序列维不变、特征维翻倍。

另一条同位置契约：**engram 注入在这三行之前**（`model.py:88-90`）。顺序是
`engram 注入 → target hidden 读取 → Block`。ref 正是这个顺序，改动它会让 MTP 读到被 engram 改过的
hidden。

---

## 5. indexer STE 的插入点（明确写死）

```
v41f/attention.py
  :175  idxs = self.indexer(x, qr, state.index_k, freqs, 0, window_len)
  :177  return state.compress_kv, idxs          <- _compress 的返回，需要多带一个 sc
  :180  return state.compress_kv, state.topk_idxs
  :195  comp_kv, comp_idxs = self._compress(...)
  :198  idxs = torch.cat([idxs, comp_idxs], dim=-1)
  :200  o = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale)   <- STE 替换这一行
```

- **不是 `model.py` 的活**：STE 在 attention 内部，整网控制流不变。
- `_compress` 现在返回 `(kv, idxs)`，要变成多返回一个连续 score（`Indexer.forward` 里
  `index_score` 在 `:115` 被 topk 消费后就没再传出，`v41f/indexer.py:114-116`）。
- 窗口段与压缩段在 `:198` 已经 concat 成一条 `idxs`，但**只有压缩段有 score**。STE 的实现要么在
  concat 前分别调用、要么在 softmax 权重上按段拼 `p`——这个选择要在 #456 的实现里定，本文件只
  指出落点与它为什么不在 `model.py`。
- 开关默认 `off`，off 时 `:200` 那一行**原样执行**，这是 G-C2 逐位门的来源。

---

## 6. 与 #447 / #456 的交叉点（术语要对齐，不能各说各的）

- **#447 的分组没有"在模型里但不在 master/optim"这一档**，而 C 步要求 indexer 参数 `off` 时在模型
  里、不在 optimizer 组，`ste` 时才进。后落的那一篇要在先落那一篇的**术语**里把这一档写清。
- **#447 的 §1.1 要按接线后的实际参数集合重算**（compressor fp32、engram 表与 `q/k_weight`、
  `mtp.*`）。评审 P0-1 已给判据：`group` 应由 `param.dtype` 派生，而不是手写名单——engram 一进来，
  手写名单必然再漏一次。
- **#456 的 STE 与 #447 的 optimizer 成员规则是同一个开关的两个后果**：`indexer_train_mode="ste"`
  同时决定 STE 路径与"indexer 进 optimizer 组"。写成两个独立开关会产生"开了 STE 但没进 optimizer"
  的静默不训练（#456 的 M7 就是这条）。

---

## 7. 本设计不回答的问题（留给实现 PR 或裁决）

### 7.0 已裁（fb，2026-09-17）

1. **tokenizer 显式传参，不建 `v41f/tokenizer.py` 磁盘加载器，`model.py` 绝不读磁盘。**
   模型构造收 tokenizer 对象（或已构建好的 compressed map + 派生行数），加载由训练/推理入口做一次
   再注入。理由与 `build_compressed_token_map(tokenizer)` 的纯函数签名一致，且避免单测为了构造模型
   先落一个 tokenizer 文件。
2. **§7.3 tied embed/head 双 key：倾向"衔接时显式只注册一份、两模块共享同一注册名"**，让
   `state_dict()` 保持一个参数一个名字（与 #447 严格键集自洽）。具体写法在步 D 的实现 PR 里对着
   `state_dict().keys()` 实测敲定；判据是**不留两个 `data_ptr` 相同的 key**。
3. **engram `q_weight/k_weight`：必须显式 dtype。** ⚠️ fb 给的理由是"按忠实参考该是 fp32 就 fp32
   （对齐 head/HC 的原生 fp32 处理）"——**这条前提经查不成立**，见 §2.2 前置 3：ref 的 HC 表显式
   包在 `set_dtype(torch.float32)`（`model_ref.py.ref:940-946`）里，engram 的 `q_weight/k_weight`
   没有，走的是 ambient bf16。**要修的是隐式，忠实值仍是 bf16**；若定要改 fp32，是一条需记为自定
   的偏离。这一条在接线 PR 里仍需 fb 确认一次。

### 7.1 其余待裁项

1. （已裁，见 §7.0.1）
2. MTP 多 token loss 的加权与 target 层选取：charter §6 已标注"官方未给，需自定并 prereg"，本文件
   只接线不算 loss。
3. （已裁方向，见 §7.0.2；实现细节留步 D）
4. 步 C 与步 A/B 的并行度：C 不动 `model.py`，理论上可与 B 并行，但两者都要跑 `v41f_small` 整网
   对拍。若 fb 要压缩日历时间，C 可以先行（它默认 off，风险最低）。
4. 步 C 与步 A/B 的并行度：C 不动 `model.py`，理论上可与 B 并行，但两者都要跑 `v41f_small` 整网
   对拍。若 fb 要压缩日历时间，C 可以先行（它默认 off，风险最低）。
