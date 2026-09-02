---
question: "why does a sentence that states a measurement and then draws a conclusion from it get read as all-measurement, and what makes the pair separable"
status: recorded
source: "de, 2026-09-02 and 2026-09-03. Six instances: train.py:189 (a bisect number plus a ratio inferred from it), train.py:217 (the inference alone, contradicting :189 by an order of magnitude), Cfg._plan_trimmed (a name asserting what the code did not do), scripts/harness.py:1955 _noted_gone (two facts sharing one field), the `step N/N` completion criterion (correct at every layer, measuring total_steps mod 10 rather than completion), and train.py:1034 (a run-end save inferring plan-complete from a missing step, and storing the inference as a cursor -- 213,164 rows overstated). Three of the six were written by someone who already knew the rule, including this document's own first draft quoting 2.27x as grad_ckpt's cost. tilerl asked for the entry after the first."
---

# 事实和推断不能共处一句

一句话里同时放测量值和从它推出的结论,读者拿走的是结论,而结论的来源
——测过、还是推过——在传递中丢失。丢失不是因为读者粗心:一个句子只有一个
可信度,而句子的可信度由它最强的部分决定。

作者也是读者。本文初稿就犯了它要讲的错(第 2 节),而第 5、6 两条是在本文落地
之后一小时内发现的,一条在我自己写的 fact 里、一条在存盘路径上——不是谦辞,是这一类最短的证据:知道这条规则不足以遵守它。

## 六个实例

### 1. `train.py:189` — 测量后面跟一个推断,同一个注释

```python
batch = (
    32  # throughput_bisect 2026-08-27: 90K tok/s at batch 32 no-ckpt; 72 needs grad_ckpt (2.4x slower)
)
```

`90K tok/s at batch 32` 是 bisect 测出来的。`2.4x slower` 不是——那一天没有
跑过 batch 72 开 grad_ckpt 的计时,2.4x 是从「72 不开 ckpt 会 OOM」加上
一个内存-时间的换算推出来的。两半共处一个分号。

`facts/efficiency.json#eff.batch_ceiling` 抄了整句,`source` 写的是
`train.py:121,131 comments`——即注释本身成了来源。`status: recorded`,不是
`measured`,这一点是诚实的;但 value 里两半仍然不分。

### 2. `train.py:217` — 只剩推断,数字还变了

```python
grad_ckpt = False  # costs 25% wall-clock for ~15GB savings; batch 32 fits without it on H20
```

同一个文件里,`:189` 说 2.4x,`:217` 说 25%。两个数字差一个数量级,共存了
数周,因为读到其中一处的人没有理由去读另一处。

而 2026-09-01 在 32 层上实测(`facts/efficiency.json#eff.grad_ckpt_inverts_with_depth`)是
**1.116x**——两个都不对。`:189` 的 2.4x 在 12 层是对的,`:217` 的 25% 无出处。
12 层的两臂测量(`facts/efficiency.json#eff.grad_ckpt_300m_two_arm`)给 **2.27x**,
支持 `:189` 的量级而不支持 `:217`。

**这段初稿自己犯了本文要讲的错**,记在这里:第一版写的是「第二个 A/B
(b8a4 无 ckpt vs b16a2 开 ckpt)量到 2.27x」,读起来像 grad_ckpt 的代价。回去读
两个 arm 的启动行才发现**两臂差两个变量**——grad_ckpt,以及 micro-batch 16 对 8。
有效 batch 两边都是 32,但省下的激活显存被花在了不同的 micro-batch 上,所以
2.27x 是两者的联合效果。它能定的只是**量级**,足以裁掉 25%,不足以当成
grad_ckpt 的代价。干净的第三臂(grad_ckpt=False,batch 16 × accum 2)没有跑——
那正是这个形状在避开的 OOM 配置(`eff.microbatch_32_oom`)。

写进 fact 前那个数还根本不在 fact store 里:我以为在,`grep 2.27` 命中两条
`nccl_version: 2.27.3`。**引用一个数之前先确认它有条目**,否则文档在引用自己。

**一句话里的推断会脱离它的测量独立传播**,然后被改写、被引用、和原测量矛盾,
而没有任何一步是恶意或粗心的。

### 3. `Cfg._plan_trimmed` — 名字断言了代码没做的事

标志位的计算是正确的:

```python
Cfg._plan_trimmed = any(v > 0 for v in used.values())
```

它知道的是「cursor 在建 plan 前 seed 了 `used[]`」。它的**名字**说的是
「plan 已被裁剪」。而 `:1597` 从来不裁剪——`want = int(rows * frac * weight)`
用的是完整预算,不看 cursor。

两个消费点都照名字用:`:2175` 加回 `resume_step`(需要「本段」的语义,这条
恰好成立),`:2235` 设 `i0 = 0`(需要「plan 里没有已读行」,这条也恰好成立)。
**两处都对,而名字是错的**——所以没人去查。tilerl 读出这一点时,依据不是行为,
是名字与代码的对照。

名字是最短的句子。它没有空间放「测过」还是「以为」。

### 4. `harness.py:1955 _noted_gone` — 两个事实共处一个**字段**

同一形状再上一层。判据是:

```python
note = f"{entry.get('uncertainty') or ''} {entry.get('boundary') or ''}"
named = name in note or (len(tail) >= 5 and tail in note)
return bool(named and re.search(r"prun|delet|zero|...", note, re.I))
```

名字和 gone-word 各查一遍,互不知道位置。所以一条「在 X 上测的」加一条
「**Y** 被剪了」——两句都真、说的是两个 ckpt——合起来让 X 读作已披露。实测
返回 `True`。

不是句子共处,是**字段**共处;拼接把两个事实变成一个字符串,判据在字符串上
做 AND,位置信息在拼接那一步就没了。

### 5. `step N/N` — 每一层都对,而量的不是那件事

这一条不是句子也不是名字,是**判据**,而它是这一类里最难发现的形态。

`eff.resume_inflates_total_steps` 写过:415 个 log 里 35 个打出 `step N/N`,
所以「有 run 跑到过终点」。2026-09-03 200M 那个 resume 真的跑完了,而它对这个
scan **完全不可见**——最后一条 step 行是 `3810/3814`,因为 step 行每 10 步打
一次,而 3814 不是 10 的倍数。

回头看那 35 条:**total_steps 全部 ≡ 0 (mod 10)**。

```
step N/N 量的是:total_steps 是日志间隔的倍数
它声称量的是:run 跑到了终点
```

每一层都对。正例上全中,反例上会红,连细节都是对的——`grep -oE 'step
([0-9]+)/\1'` 会把 `step 1460/14606` 读成跑完,换成 awk 逐字段比较是必要的修
正。**而整体量错了对象**,并且它的样本恰好排除了唯一能推翻它的那个 run。

改成收尾行判据后:51 个 log 有 `ep N/M train`,其中 **49 个训练了非零 loss**。
另 2 个(`rehearse_resume`、`e2e_tmp`)在**零步之后照样打收尾行并存盘**——就是
16000/7998 那个。所以判据必须两半都要,而这一半只有在见到反例之后才写得出来。

和前四条的关系:前四条是一个句子里事实与推断不分,这一条是**判据与它声称的
属性不是同一件事**,而两者的后果一样——读者拿走结论,来源丢失。区别在于判据
不会露出破绽:它在能构造的每个 broken world 上都表现正确。

### 6. `:1034` — 同一个形状,写进了 artifact

`train.py:2561` 存 run-end checkpoint 时不传 `step`;`:1034` 把「没有 step」读成
`no step (run-end save): the plan is complete`,于是写 plan-complete 计数。

**一个事实(没传 step)和一个从它推出的结论(所以 plan 跑完了),在同一行,而
结论被当成数据存进了文件。** 那个推断在写下时是真的——跑完循环就等于跑完
plan。`--max_steps` 用来约束 resume 之后就不真了。

代价是可量的:同一个 run 的两个 ckpt 相差几秒,`.ep1` 的 cursor 和是
976,384 = 3814 × 256(实读行数),裸 `.pt` 是 1,189,548,**多 213,164 行 =
0.87B token 这个 run 从没训过**。从裸 ckpt resume 会静默跳过它们。
`facts/efficiency.json#eff.run_end_cursor_overstates_under_max_steps`。

这一条是在本文落地一小时后发现的,在同一个 repo 的存盘路径上。

## 为什么这一类不会被发现

| 机制 | 后果 |
|---|---|
| 一个句子只有一个可信度 | 由最强的半句决定,弱的那半继承它 |
| 推断比测量短、更好用 | 引用时被优先抄走,`2.4x` 比一整句 bisect 结果好放进表格 |
| 消费点照名字/结论用 | 名字错而两个消费点都对,没人有理由回去查 |
| 拼接消灭位置 | 两个各自为真的事实,合成一个假的结论 |
| 判据在每个 broken world 上都对 | 它量错了对象,而错法不会露出破绽(第 5 条) |
| 推断被存成数据 | 后来的读者拿到的是一个数字,不是一个当时为真的推理(第 6 条) |

六条都不需要任何人犯错,后三条(初稿的 2.27x、第 5、第 6)还是知道规则的人当天写的。

## 规则

- **一句一件事**:测量值和从它推出的结论分开成两句,推断那句以「推出」「按…换算」开头。数字后面直接跟它的来源命令或 artifact 路径。
- **名字只能断言代码做到的事**。`_plan_trimmed` → `_cursor_seeded`:后者是标志位真正知道的。名字改了要重判每个消费点——`:2235` 的 `i0 = 0` 重判后不变,理由写进注释,因为「不变」和「没看」在代码里长得一样。
- **一个字段一个事实**。判据要在拼接前工作:按句切分,要求名字和词在**同一句**里。跨字段拼接后做 AND,等于承认位置不重要。
- **检测法**:一个注释或 value 里出现两个数字,而只有一个能追到命令或 artifact——另一个是推断。同一文件里两处描述同一个量而数字不同,两处都不可信,去测。
- **引用一个数之前确认它有 fact 条目**。本文初稿引了 2.27x,而 `grep 2.27 facts/` 命中的是两条 `nccl_version: 2.27.3`——数在我脑子里,不在 fact store 里。一份引用自己的文档和一份引用测量的文档,在读者眼里长得一样。

## 已做与未做

`_plan_trimmed` → `_cursor_seeded` 已改(8c61642),两个消费点已重判。
`_noted_gone` 的同句要求由 44 落地(31093f9)。
`train.py:189` / `:217` 的注释未改:`train.py` 在 p200m_4b_0902 续训期间冻结,
排在下一个 stop window,和 resume 修复同一个解冻窗口。
`eff.batch_ceiling` 的 value 仍然两半不分,`status: recorded` 是它现在唯一的
标注——**未改**,因为改它要同时改三个引用点(`arch_efficiency_2x.md:318`、
`stop_window_2026-09-02.md:17`、`throughput_survey.md:197`),而其中一处引的正是
那个无出处的 25%。
