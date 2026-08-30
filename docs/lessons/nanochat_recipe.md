---
question: "nanochat 的 midtrain 阶段是什么,我们缺了什么?"
status: recorded
source: "github.com/karpathy/nanochat @ master, 2026-08-31: runs/speedrun.sh, scripts/base_train.py, scripts/chat_sft.py, tasks/common.py, tasks/gsm8k.py (read via gh api, not from summaries)"
---

# nanochat 的实际配方

**读的是仓库,不是博客。两条早先的转述是错的,先更正。**

## 更正 1:没有 midtrain 阶段

`runs/speedrun.sh` 的全部流水线:

```
nanochat.dataset  →  tok_train  →  tok_eval  →  base_train  →  base_eval  →  chat_sft  →  chat_eval
```

**四个阶段,没有 midtrain。** 早先说"nanochat 有一个我们缺的 midtrain 阶段"来自搜索摘要和一篇讲 nanochat+DiLoCo 的论文,不是这个仓库。**我们没有缺一个阶段。**

## 更正 2:它训在 8 tok/param,低于 Chinchilla

```bash
scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8
```

而 `base_train.py:58` 的帮助文本写着 `Chinchilla=20`,默认值是 `12`。**speedrun 用的是 8。**

所以此前那张对照表里"nanochat = 20 tok/param"是第三方 gist 的旧配置。真实情况是:

| | 参数 | tok/param |
|---|---|---|
| nanochat speedrun | d24 | **8** |
| aupai 现在 | 206M | **15.7** |

**我们的 token:参数比是它的两倍,结果仍然是零。所以差别不在这个比值上。**

## 真正的配方:任务教学在 SFT 里,靠混合与过采样

`scripts/chat_sft.py:163-166`:

```python
SmolTalk(split="train"),                                          # 460K rows of general conversations
*[MMLU(subset="all", split="auxiliary_train") for _ in range(3)], # 100K rows per epoch
*[GSM8K(subset="main", split="train")        for _ in range(4)],  #   8K rows per epoch
```

- 约 **792K 行**:SmolTalk 58% / MMLU 38% / GSM8K 4%
- **过采样的实现就是把同一个数据集对象放进列表 N 次**(`TaskMixture` 的注释:"if you wish to oversample any task, just pass it in multiple times")
- CLI 里两个 flag 直接叫 `--mmlu-epochs`(默认 3)和 `--gsm8k-epochs`(默认 4),**帮助文本写明各自教什么**:MMLU "teaches Multiple Choice",GSM8K "teaches Math and Tool Use"

**它们训的是 benchmark 自己的 train split。** `MMLU(split="auxiliary_train")`、`GSM8K(split="train")`。这是标准做法,但要说出来:**报出来的 GSM8K 数是 in-distribution 的。** 我们从来没有在 math-hard 或 code-500 的训练分布上训过。

验证集按同样比例构造(`stop=5200`、`stop=420`,注释写 "to match the train ratios"),所以 val bpb 跨阶段可比。

## 最值得抄的一条:工具调用是数据里本来就有的标注

`tasks/gsm8k.py` 不发明工具协议 —— GSM8K 原始答案里就带 `<<12/60=0.2>>` 这种计算器标注,它把这个拆成两个**有类型的消息片段**:

```python
assistant_message_parts.append({"type": "python",        "text": expr})    # 12/60
assistant_message_parts.append({"type": "python_output", "text": result})  # 0.2
```

**模型学的是"写出表达式"和"写出结果"两件分开的事,而不是"心算出结果"。**

这一条对我们直接相关。我们的 `eqcheck` 实测:**8% 的生成含等式,其中 81.7% 算错**。把等式拆成 expr / output 两个片段之后,**结果那一半可以由工具产生,不由模型产生** —— 一个 200M 模型不需要会算术,只要格式允许它把算术交出去。

## 其它可直接对照的差异

| | nanochat | aupai |
|---|---|---|
| SFT 优化器 | `--load-optimizer 1`,**从预训练热启动** | 不继承 |
| SFT LR | `init 0.8×base`,**warmup 0**,warmdown 占后 50%,final 0 | warmup 20 绝对步 |
| SFT 数据 | 三个数据集混合,按任务过采样 | 一个混合包,不按任务分权 |
| 预训练数据 | FineWeb-Edu(英文,已筛) | 54% 中文 cosmopedia,真英文 0.75% |
| 词表 | 65,536,4.8 chars/token | 32,784,~1.7 chars/token(中文) |

**按字符算,它的 token 预算是我们的近 3 倍密度** —— 跨词表比 token 数是跨分母比较,和今天犯过三次的错是同一个。

## 结论

1. **不需要新增阶段。** 需要的是把 SFT 从"一个混合包"改成"按任务过采样的混合",并且**在代码里写明每个数据集教什么**。
2. **工具调用的格式化是最便宜的一条**,而且我们的 eqcheck 数字正好指向它:算术错误率 81.7% 是一个可以用格式绕开的问题,不是必须用规模解决的问题。
3. **token:参数比不是我们和 nanochat 的差别** —— 我们是 15.7,它是 8。差别在数据构成(它是筛过的英文,我们是中文百科)和 in-distribution 训练(它训 benchmark 的 train split)。
