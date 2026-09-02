---
question: 把模型从 train.py 切出来，需要搬哪些符号、block 接口怎么定、老 ckpt 怎么保证还能加载
status: recorded
source: b0-8（fb 转达用户方向 2026-09-02，pair tilerl）。行号取自 train.py（2,751 行）本次
  读取时的 HEAD；prior eff.attnres_internal。**这是设计页，一行代码都没动**——run 活着
  期间 main 的 train.py 零编辑，diff 在 b0 worktree 里做，run 结束那刻可合。
---

# 把模型切出去：一个结构实验 = 一个类 + 一个 Cfg 开关

**目标不是"文件更小"，是让一次奇葩结构实验的成本等于一个类加一个开关。**
（搬迁量 382 行 = 13.9%，不是最初写的 22%——这个数字不支撑"文件更小"，
所以更不该拿它当理由。）

用户点名的例子是 DFlash 式 draft head 共享主干——那件事今天要改 2,751 行文件里的
模型定义，而模型定义和训练环、信号处理绞在一起。

## 0. 业界怎么切（一句话，prior 已填）

**主流做法是"模型是一个包，训练器 import 它"**：nanoGPT 的 `model.py` / `train.py` 两分、
HF transformers 的 `modeling_*.py` 一个文件一个架构、torchtitan 的 `models/` 下按架构分目录。
**共同点不是目录形状，是依赖方向单向**：模型不 import 训练器。**我们今天是双向的**——
`train.py` 里模型和 `RunLog`、`Cfg`、DDP 装配同居，而 `sft.py`／`sft_math.py`／
`infer_local.py` 从 `train` import 模型符号，所以任何模型改动都经过训练器这个瓶颈。

## 1. 搬什么：符号清单与行号

**行号是本次读取时的，做 diff 前重新确认**——train.py 在 run 期间不动，但合并会移动它。

### 进 `model.py`（纯模型，不 import 训练器）

| 符号 | 行 | 说明 |
|---|---|---|
| `RMSNorm` | 278-287 | 归一化 |
| `rms_scale` | 288-294 | `AttnRes` 用的 gain-free RMS |
| `DeltaRecurrence` | 295-357 | **KDA block**；状态在类内部（见 §2） |
| `GatedMLA` | 358-414 | **gated-MLA block**，NoPE |
| `SwiGLU` | 631-647 | FFN |
| `Source` | 648-661 | `AttnRes` 的输入契约（`v` + `scale`） |
| `AttnRes` | 662-688 | 深度注意力 |
| `Block` | 689-708 | **接口所在，见 §2** |
| `remap_legacy_state_dict` | 709-731 | 老 ckpt 键名重映射 |
| `HybridLM` | 732-875 | 模型本体，含 FoNE head（`:757-758`） |

**合计 382 行**（十个区间逐个相加），占 train.py 2,751 行的 13.9%。

**不要用首尾相减**：278-875 的跨度是 598 行，但 FP8 那一族（415-630，216 行）夹在
中间且留守（见下），598 - 216 = 382。**本页前三版写的"约 600 行"就是这么错的**——
把跨度当成了搬迁量，而跨度里有一块不搬。

### 留在 `train.py`

`Cfg`（171）、`RunLog`（44）、`Muon`（876）、`build_optimizers`（1092）、
`set_schedule`（1151）、`save_checkpoint`（1323）、数据/游标/DDP 全部。
**判据：模型前向不需要它。** `Muon` 是优化器不是模型，即使它和模型耦合最紧。

### 边界情况，逐个定

- **`SOFTCAP`（168）**：模型前向读它（logit softcap），但它是 env 读出来的全局。
  **搬 `model.py` 并在 `train.py` 里 re-export**，否则 `sft.py` 的 import 断。
- **FP8 那一族**（`FP8LinearFunction` 415、`FP8Linear` 464、`_fp8_ok` 480、`_fp8_mm` 492、
  `patch_liger_flce_fp8` 526、`convert_to_fp8_compute` 579、`_convert_to_fp8_legacy` 621）：
  **不搬**。它们是 module 替换器（对已构造的模型做手术），依赖 Liger／CUDA capability，
  是训练配置不是架构。**搬了会让 `model.py` import torch 之外的东西。**
- **`HAS_FA`**：模型内部分支用，**跟着 `model.py` 走**。

## 2. Block 接口：残差流进出，KDA 状态在类内

**现状（`train.py:701-706`，`Block.forward` 到 `sublayers` 的返回），这是契约，
不是实现细节：**

```python
def forward(self, x, cu=None):          # x: [B, T, d] 残差流；cu: doc 边界或 None
    x = x + self.mixer(self.n1(x), cu)  # pre-norm，残差在 Block 里加
    return x + self.ffn(self.n2(x))

def sublayers(self, cu=None):           # AttnRes 用：(深度注意力, norm, 变换) 三元组
    return ((self.ar1, self.n1, lambda t: self.mixer(t, cu)),
            (self.ar2, self.n2, self.ffn))
```

**三条必须写进设计、否则一个结构实验会悄悄破坏它：**

**一、`forward(x, cu)` → 同形状 `x`。** 残差流进出，形状 `[B, T, d]` 不变。
**一个新 block 只要满足这条，就能塞进 `HybridLM.blocks` 的任意位置。**

**二、KDA 状态不在接口里，在 `DeltaRecurrence` 内部。** 两条证据，一反一正：

- **反向（没有 state 在流动）**：`self.state` 在整个 train.py 只出现在 `Muon:959`，
  那是优化器动量，不是 KDA。
- **正向（签名本身不接收也不返回 state）**：`DeltaRecurrence.forward` 的签名是
  `train.py:314` `def forward(self, x, cu=None)` —— **只有残差流和 doc 边界，没有
  state 入参**；唯一的 return 在 `train.py:355`
  `return self.o(out.reshape(B, T, D).to(x.dtype))` —— **只返回一个张量，不返回
  (out, state) 元组**。一个不接收、不返回 state 的 forward，无法参与跨 block 的
  状态传递。

**反向证据只能说"我没找到"，正向证据说的是"接口上没有位置放它"。** 这一条是整页的
作废条件，所以两条都要有（fb 要求，2026-09-02）。

**因此"位置在 KDA state"这个性质是 block 局部的**：`GatedMLA` 是 NoPE，位置信息由序列
里的 KDA 层提供，**但那是通过残差流传的，不是通过一个 state 参数**。

> **对 draft head 共享主干的直接含义**：draft head 可以读任意层的残差流输出，
> **不需要拿到 KDA 状态**。这让 DFlash 式实验变成"再加一个读 `x` 的 head"，
> 而不是"改 block 的状态传递协议"。**这是本设计页最重要的一条结论。**

**三、`sublayers()` 是 AttnRes 的耦合点，新 block 必须提供它或显式声明不支持。**
`AttnRes` 需要 `(ar, norm, transform)` 三元组来对每个子层做深度注意力。
**一个不实现 `sublayers()` 的 block，在 `attn_res=True` 时会静默跳过深度注意力**——
那是"配置说开了、实际没开"的经典形状。**设计要求：`Block` 基类提供默认
`sublayers()`，子类不实现就用默认；`AttnRes` 遇到 `None` 时抛，不跳过。**

## 3. `test_arch_compat` 必须覆盖什么

**核心风险不是"切错了"，是"切完老 ckpt 加载看起来成功但权重错位"。**
`remap_legacy_state_dict`（709）的存在证明这个风险已经实现过一次
（`w1|w3 -> w13`、`k_up|v_up -> kv_up` 等键名融合）。

**四条，前两条是硬要求：**

1. **老 ckpt 往返**：加载一个搬迁前存的 checkpoint，`load_state_dict(strict=True)`
   必须过。**`strict=True` 是要点**——`strict=False` 会把"少加载了一半权重"读成成功。
2. **前向逐位相等**：同一 ckpt、同一输入、同一 seed，搬迁前后的 logits
   `torch.equal`（不是 `allclose`）。**搬移代码不该改变任何一个 bit**；如果只能过
   `allclose`，说明搬迁改了计算顺序，那是另一件事，要单独裁定。
3. **`sublayers()` 契约**：每个 block 类型在 `attn_res=True` 下都真的用上了 `AttnRes`
   ——断言深度注意力的参数出现在梯度里，**不是断言配置为 True**。
4. **import 方向单向**：`model.py` 的 import 集合里没有 `train`。
   **一行 `assert "train" not in sys.modules` 式的检查即可**，它防的是下一个人
   "顺手" import 回去。

## 4. 下游 import 不能断

**`sft.py` 和 `sft_math.py` 顶部各 import 12 个符号**（实测，顶部两者完全相同）：
```
Cfg, HybridLM, RunLog, SOFTCAP, build_optimizers, convert_to_fp8_compute,
ddp_even_len, doc_cu_seqlens, opt_snapshot, save_checkpoint, set_schedule, setup_ddp
```
**其中只有 `HybridLM` 和 `SOFTCAP` 会搬走。** 另外
`infer_local.py:35` import `AttnRes, Source, remap_legacy_state_dict`（三个全搬走），
`fone.py:159` import `generate_batch`（不搬），
**`sft_math.py:148` 在函数体里 `from train import HAS_FA`（搬走）**——
这一条是本页第一版漏掉的：顶部 import 块之外还有一处延迟 import，
`grep 'from train import'` 才看得到，读顶部 12 个符号的清单看不到。

**会断的是三个文件，不是四个**：`sft.py`（`HybridLM`/`SOFTCAP`）、
`sft_math.py`（那两个 + `HAS_FA`）、`infer_local.py`（三个全搬）。
`fone.py` 不断——它只要 `generate_batch`，那个不搬。

**结论：`train.py` 必须 re-export 搬走的 6 个符号**
（`HybridLM`、`SOFTCAP`、`HAS_FA`、`AttnRes`、`Source`、`remap_legacy_state_dict`）。
**re-export 是本次唯一允许的"加行"**（约 5 行），因为它换掉的是三处 import 修改。

## 5. 删除式：搬出去的不许在别处加回来

**验收就是 fb 的那句**：行数净减，且 59 门检查结论完全不变。

- `train.py` 减 382 行，`model.py` 加 382 行 + re-export 约 8 行（6 个符号 + import +
  `__all__`）→ **净加约 8 行，不是净减。** 这一点必须提前说清，不要在验收时才发现。
  （前三版这里写"净加 5 行"，算法没错，错在搬迁量的前提是 600 而不是 382，
  re-export 的符号数是 2 而不是 6。**推导继承前提的状态**，见
  `docs/standards/writing.md`。）
- **真正的净减来自第二步**：切出去之后，`Block`／`HybridLM` 里那些
  `getattr(cfg, "attn_res", False)` 式的防御性读取可以收敛成显式参数。
  **但那是行为可能变化的改动，不属于这次搬迁**——按 fb 的话，那是另一个检查。
- **所以本次交付的诚实描述是"结构变化，行数近似不变"。** 用户要的是
  "一个结构实验 = 一个类 + 一个开关"，那个能力由 §2 的接口契约提供，不由行数提供。

## 6. 顺序与硬约束

1. tilerl 读这页 → 对 §2 的三条接口契约和 §3 的四条覆盖点表态
2. **tilerl 把本页合进 main**（roster：integration 持有 main 的合并权；b0 不自合）
3. b0 在自己的 worktree 里做 diff（**main 的 train.py 零编辑**）
4. run 结束那一刻，`test_arch_compat` 四条全过 + 59 门 diff IDENTICAL → 合

**run 活着期间不合、不推、不动 main 的 train.py。** 训练路径在 `pod_head_manifest.txt`
里被运行读取，改它等于在飞行中换发动机。

## 7. 作废条件

- **`Block.forward` 的签名变化**：§2 的契约作废，整页重写。
- **KDA 状态改成跨 block 传递**：§2 第二条作废，且 draft head 的结论跟着变——
  **那会把 DFlash 式实验从"加一个 head"变成"改协议"**。
- **行号漂移**：§1 的行号只在本次 HEAD 有效，做 diff 前重新确认。
  **本页已经因此失效过一次，修的时候又错了一次，两次都值得记：**

  *第一次（漂移）*：第一版行号取自 `f5df4b6` 的 train.py（2,722 行，`RMSNorm` 在
  277、`DeltaRecurrence.forward` 在 313）。合 main 后 train.py 在 import 块加 1 行、
  `main()` 里加 28 行，§1 的十个符号号全部低 1。**写对了、然后被自己的合并改错，
  和一开始就写错，事后看长得一样**——这就是"行号不是引用、行内容才是"的由来
  （`docs/standards/writing.md`）。

  *第二次（过度修正）*：发现"全体低 1"后，我把 §2 的 `701-706` 也加了 1。
  但 `Block` 在 689 而不是 690——`Block` 之前的插入是 0 行，之后才有 28 行，
  **偏移不是全局常数**。`701` 本来就是对的。**"全体差 1" 是从十个样本归纳出的
  规律，不是逐个核对的结果**；修正必须和断言走同一道核对，否则就是用一个错
  换另一个错。现在 §1／§2 的 25 处行号都由脚本按行内容验过（`BAD count: 0`）。
