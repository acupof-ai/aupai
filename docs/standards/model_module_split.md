---
question: 把模型从 train.py 切出来，需要搬哪些符号、block 接口怎么定、老 ckpt 怎么保证还能加载
status: recorded
source: b0-8（fb 转达用户方向 2026-09-02，pair tilerl）。行号取自 train.py（2,751 行）本次
  读取时的 HEAD；prior eff.attnres_internal。**这是设计页，一行代码都没动**——run 活着
  期间 main 的 train.py 零编辑，diff 在 b0 worktree 里做，run 结束那刻可合。
---

# 把模型切出去：一个结构实验 = 一个类 + 一个 Cfg 开关

**目标不是"文件更小"，是让一次奇葩结构实验的成本等于一个类加一个开关。**
（搬迁量 363 行 = 13.2%，不是最初写的 22%——这个数字不支撑"文件更小"，
所以更不该拿它当理由。**实测净加 60 行**，见 §5。）

用户点名的例子是 DFlash 式 draft head 共享主干——那件事今天要改 2,751 行文件里的
模型定义，而模型定义和训练环、信号处理绞在一起。

## 0. 业界怎么切（一句话，prior 已填）

**主流做法是"模型是一个包，训练器 import 它"**：nanoGPT 的 `model.py` / `train.py` 两分、
HF transformers 的 `modeling_*.py` 一个文件一个架构、torchtitan 的 `models/` 下按架构分目录。
**共同点不是目录形状，是依赖方向单向**：模型不 import 训练器。**我们今天是双向的**——
`train.py` 里模型和 `RunLog`、`Cfg`、DDP 装配同居，而 `sft.py`／`sft_math.py`／
`infer_local.py` 从 `train` import 模型符号，所以任何模型改动都经过训练器这个瓶颈。

## 1. 搬什么：符号清单与行号

**行号只在 train.py blob `ad0f6e130083379a309c0ea43577fef61fd8e0e9` 上有效**
（`git hash-object train.py`；该 blob = `bb0c378:train.py`，2,751 行）。
**sha 对不上就不要信下面的数字，按符号名重新定位**（`grep -n '^class RMSNorm'`
之类；分支 `b0` 上有一个三通道校验脚本，但它按 fb 的裁定不进 main——这页的行号
在 split 落地时会一起删掉，见 §6）——train.py 在 run 期间不动，但**合并会移动它，今晚已经移动过一次**
（见 §7）。**没有 sha 的行号等于没有行号**：它无法区分"漂移了"和"一开始就错"
（fb 要求，2026-09-02）。

### 进 `model.py`（纯模型，不 import 训练器）

| 符号 | 行 | 说明 |
|---|---|---|
| `RMSNorm` | 278-287 | 归一化 |
| `rms_scale` | 288-294 | `AttnRes` 用的 gain-free RMS |
| `DeltaRecurrence` | 295-357 | **KDA block**；状态在类内部（见 §2） |
| `GatedMLA` | 358-408 | **gated-MLA block**，NoPE |
| `SwiGLU` | 631-647 | FFN |
| `Source` | 648-661 | `AttnRes` 的输入契约（`v` + `scale`） |
| `AttnRes` | 662-688 | 深度注意力 |
| `Block` | 689-708 | **接口所在，见 §2** |
| `remap_legacy_state_dict` | 709-731 | 老 ckpt 键名重映射 |
| `HybridLM` | 732-862 | 模型本体，含 FoNE head（`:757-758`） |

**合计 363 行**（十个区间逐个相加），占 train.py 2,751 行的 13.2%。

**不要用首尾相减**：278-875 的跨度是 598 行，但 FP8 那一族（415-630，216 行）夹在
中间且留守（见下），598 - 216 = 382。**本页前三版写的"约 600 行"就是这么错的**——
把跨度当成了搬迁量，而跨度里有一块不搬。

**382 也还是错的，第四版才对（363）**：`HybridLM` 的区间原写 732-875、`GatedMLA` 原写
358-414，**真实结束行是 862 和 408**。865-873 是 "Muon optimizer" 段注释加
`POLAR_EXPRESS`（Muon 的），411-412 是 FP8 段注释加 `_FP8_MAX_E4M3`（FP8 的），
**四段都属于留守的代码**。按 382 切会直接产生 `NameError: POLAR_EXPRESS` 和四条
`F821 Undefined name _FP8_MAX_E4M3`——这两个错正是这么发现的。
**一个区间有两端，而校验脚本只验起始行**（`732` 确实是 `class HybridLM`，所以它放行）。
脚本现在两端都验，并会点名被吞掉的那个符号。

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

**抛在构造期，不在 `forward`**（tilerl 定，2026-09-02）。理由是这个条件
**静态可判**：哪些 block 实现了 `sublayers()` 在模型构造完就确定，不随 step 变。
构造期抛，错误出现在 step 0 之前、栈里带着那个 block 的类名；forward 里抛，最好
情况 step 1 崩，**最坏情况某个条件分支上的 block 到 step 8000 才第一次被问到**。
**这条规则存在的理由正是"配置说开了、实际没开"不能悄悄跑下去，那就该在开跑前喊。**

**落点是 `HybridLM.__init__`，不是 `AttnRes.__init__`（实现时更正，fb 2026-09-02 接受）。**
`AttnRes.__init__` 的签名是 `(self, d, dyn_q=False, rank=64)`——**它手上没有任何 block
引用，判不了"这个 block 有没有 sublayers()"**。`HybridLM.__init__` 里 `attn_res` 和
`self.blocks` 同时在作用域内，那才是判据能看见它所需事实的地方，而且仍满足 tilerl 的
两条要求：**step 0 之前抛，消息里带那个 block 的类名**。
**判据必须落在能看见它所判事实的作用域里**——写页面时我按"谁的契约"选了落点，
而正确的选法是"谁看得见证据"。

## 3. `test_arch_compat` 必须覆盖什么

**核心风险不是"切错了"，是"切完老 ckpt 加载看起来成功但权重错位"。**
`remap_legacy_state_dict`（709）的存在证明这个风险已经实现过一次
（`w1|w3 -> w13`、`k_up|v_up -> kv_up` 等键名融合）。

**五条，前两条是硬要求，第五条是静态的：**

1. **老 ckpt 往返**：加载一个搬迁前存的 checkpoint，`load_state_dict(strict=True)`
   必须过。**`strict=True` 是要点**——`strict=False` 会把"少加载了一半权重"读成成功。
2. **前向逐位相等**：同一 ckpt、同一输入、同一 seed，搬迁前后的 logits
   `torch.equal`（不是 `allclose`）。**搬移代码不该改变任何一个 bit**；如果只能过
   `allclose`，说明搬迁改了计算顺序，那是另一件事，要单独裁定。
   **前提：比较必须在同一个进程里做**（tilerl 定，2026-09-02）——搬迁前后各构造
   一次模型、同一份权重、同一个输入、**同一个进程**。跨进程比较会引入 cuBLAS
   workspace 和 autotune 缓存的差异，那时逐位相等会**假红**，而**假红比不做检查
   更糟：它会让人以为搬移改了计算，去查一个不存在的 bug**。
   tilerl 核过 kernel 选择的三处（`HAS_FA` 是模块级 try-import 的布尔、Liger FLCE
   在 `Cfg.compile` 之外显式构造、`chunk_kda` 是 fla 的直接调用），**三处都只依赖
   import 成不成功，不依赖先后**，所以搬移不改变哪个包能 import 成功，逐位相等
   可以要求。
3. **`sublayers()` 契约**：每个 block 类型在 `attn_res=True` 下都真的用上了 `AttnRes`
   ——断言深度注意力的参数出现在梯度里，**不是断言配置为 True**。
4. **import 方向单向**：`model.py` 的 import 集合里没有 `train`。
   **一行 `assert "train" not in sys.modules` 式的检查即可**，它防的是下一个人
   "顺手" import 回去。
5. **`ruff check --select F821,F401` 在拆分后的两个文件上为零**（fb 加，2026-09-02）。
   **理由是实测：前四条全绿的同时 `_FP8_MAX_E4M3` 是未定义的。** `test_arch_compat`
   跑的路径不进 FP8，所以它 exit 0；是 hook 里的 ruff F821 一秒指出来的（同一次还
   抓到 `LigerFusedLinearCrossEntropyLoss` 被我连同 kernel try-import 块一起删掉，
   而 FLCE 的 loss 路径留在 `train.py`）。
   **报错的是留下来的那一半，不是搬走的那一半**（e1 复现更正，2026-09-02）：坏切割下
   `ruff --select F821` 在 `model.py` 上是 All checks passed，因为常量被搬**进**了它；
   六个使用点留在 `train.py`。**所以这道检查必须跑两个文件**——只查新文件会全绿。
   **前四条都作用在"搬走的代码"上，对"留下来但丢了名字的代码"一句话说不出**——
   这个缺口不是靠增加动态用例能补的，只能靠一道静态检查。F401 同时挡住
   re-export 写漏和多写。

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

- **实测（`be845ec`）**：`train.py` 2,751 → 2,369（**-382**），`model.py` **+442**，
  **净加 60 行**。不是估的净加 8。
- **60 从哪来**：re-export **19 行**（13 个符号，不是 6）、`model.py` 自己的 docstring 与
  import 头 **62 行**（其中 kernel 的 try-import 块约 35 行）、`AttnRes` 新守卫约 19 行。
  **没有重复**：`train.py` 现在 `flash_attn` 的 import 为零，`HAS_FA`／`SOFTCAP`／
  `chunk_kda` 全部来自 `model`。
- **前三版这里的估算全部作废，而算法一次都没错**：写"净加 5 行"时前提是搬 600、
  re-export 2 个符号；写"净加 8 行"时前提是搬 382、re-export 6 个符号。
  **推导继承前提的状态**（`docs/standards/writing.md`）——三次算对，三次前提错。
- **真正的净减来自第二步**：切出去之后，`Block`／`HybridLM` 里那些
  `getattr(cfg, "attn_res", False)` 式的防御性读取可以收敛成显式参数。
  **但那是行为可能变化的改动，不属于这次搬迁**——按 fb 的话，那是另一个检查。
- **所以本次交付的诚实描述是"结构变化，行数近似不变"。** 用户要的是
  "一个结构实验 = 一个类 + 一个开关"，那个能力由 §2 的接口契约提供，不由行数提供。

## 6. 顺序与硬约束

1. ~~tilerl 读这页 → 对 §2 的三条接口契约和 §3 的四条覆盖点表态~~
   **已完成 2026-09-02**：三条接口契约 + 四条覆盖点全部同意，各补一条约束
   （§2 第三条抛的位置、§3 第二条同进程前提，已并入正文）。tilerl 按行内容复核了
   §2 的两行证据（签名与唯一 return），**整页不作废**。
2. **tilerl 把本页合进 main**（roster：integration 持有 main 的合并权；b0 不自合）
3. **做 diff 前，在分支 `b0` 上跑 `python3 scripts/check_split_page_lines.py`**
   ——按行内容核 §1／§2 的 25 处行号 + 比对 stamp 的 blob sha，8 个破坏世界全抓。
   **不是可选步骤**：本页行号已经被一次合并改错过，而那次合并是我自己做的。
   **脚本只在 b0 上，按 fb 裁定不进 main**（一个脚本守一份一次性文档是新增表面）。
4. b0 在自己的 worktree 里做 diff（**main 的 train.py 零编辑**）
5. run 结束那一刻，`test_arch_compat` 四条全过 + 59 门 diff IDENTICAL → 合
6. **合入的同一个提交里，删掉 §1／§2 的行号和 stamp**，页面改为按符号名引用
   （`RMSNorm`、`DeltaRecurrence.forward`），脚本一并删除。
   **行号的寿命到 diff 落地为止**——之后 `model.py` 就是权威，本页是历史。
   放在同一个提交，是因为分两步就会有一段时间页面上的行号指向已经不存在的
   train.py 区间（§1 的十个符号那时已经在 model.py 里）。

**run 活着期间不合、不推、不动 main 的 train.py。** 训练路径在 `pod_head_manifest.txt`
里被运行读取，改它等于在飞行中换发动机。

## 7. 作废条件

- **`Block.forward` 的签名变化**：§2 的契约作废，整页重写。
- **KDA 状态改成跨 block 传递**：§2 第二条作废，且 draft head 的结论跟着变——
  **那会把 DFlash 式实验从"加一个 head"变成"改协议"**。
- **行号漂移**：§1 的行号只在本次 HEAD 有效，做 diff 前重新确认。
  **本页已经因此失效过一次，修的时候又错了一次，两次都值得记：**

  *第一次（漂移）*：第一版行号取自 `b9519f8` 的 train.py（2,722 行，`RMSNorm` 在
  277、`DeltaRecurrence.forward` 在 313）。合 main 后 train.py 在 import 块加 1 行、
  `main()` 里加 28 行，§1 的十个符号号全部低 1。**写对了、然后被自己的合并改错，
  和一开始就写错，事后看长得一样**——这就是"行号不是引用、行内容才是"的由来
  （`docs/standards/writing.md`）。

  *第二次（过度修正）*：发现"全体低 1"后，我把 §2 的 `701-706` 也加了 1。
  但 `Block` 在 689 而不是 690——`Block` 之前的插入是 0 行，之后才有 28 行，
  **偏移不是全局常数**。`701` 本来就是对的。**"全体差 1" 是从十个样本归纳出的
  规律，不是逐个核对的结果**；修正必须和断言走同一道核对，否则就是用一个错
  换另一个错。现在 §1／§2 的 25 处行号都由脚本按行内容验过（`BAD count: 0`）。
