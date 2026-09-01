---
question: 500M 读数设计——最早在第几个 token,能可靠区分"500M 这次在起作用"和"又是上一轮那个结果";在 500M 任何 checkpoint 落地之前钉死
status: open
source: fb tasking 2026-09-01("这个答案值三天机时"); builds on readout_30b_prereg.md、reasoning_panel.md §6、honest_measurement_prereg.md。决策规则在数出来之前写死
---

# 500M 读数设计(预登记)

## 0. 问题精确化

"起作用"不是"500M 比 200M 强"——是 **500M 在 matched-token 上的仪器曲线离开 200M 的曲线,超过仪器噪声**。比较轴是 token 数,不是 checkpoint 序号:500M@T vs 200M@T,T 相同。200M 曲线 = `ckpt_pretrain_30b_s2`。**pod /work/aupai 实测(2026-09-01):存活仅 5 点**——step17500(=16.056B,里程碑名"16b")、step24000(=22.020B,"22b")、step25500/26000/26500(=23.396/23.855/24.314B,尾部 500 步密档);0–16B 无存活 checkpoint,跑程止于 step26500(24.314B)。token 换算:200M 跑 batch 16 × accum 2 × 4096 × world 7 = **917,504 tok/step**(里程碑名 16b/22b 反推唯一吻合 world=7)。**matched-token 能力轴从 16B 才开始;16B 以下的 500M checkpoint 用绝对阈值(仪器地板 + 2δ)读,是 weaker 读法,如实标注。**

**0–16B 的替代轴(fb 线索,已核实+已修正):训练日志的 val 损失存活,但跨跑不干净。** `runs/pretrain_15b_s1.log` 有 32 个 val 点(0.46–14.7B,2.917→2.116),`pretrain_30b_s2.log` 21 个(15.1–24.3B,2.120→2.085)。train.py 的 val 是**每域 pool 的前缀切分**(:1817,跑内 held-out,不是跨跑固定集)。跨跑只在 **pool 文件相同的域**上干净:500m 与 30b_stage2 共享 5 域(cot、en_c4_stage2、math_owm_stage2、textbook_30b、zh_web),4 域新增(chat_qa、chatml、code_py_rp1t、code_py_starcoder)、2 域丢弃(code_rp1t、wiki_chat)。**日志聚合值被新增/丢弃域污染——是趋势参考,不是干净对比**(同 step17500 readout 的 "different text" REFUSED 类;我上一版说"对比干净"是错的,在此修正)。干净仪器 = 在 5 个存活 200M checkpoint 上对共享 5 域重跑 per-domain val,与 500M checkpoint 对比,覆盖 16B+(与能力轴相同)。**0–16B 干净 val 不存在:没有 checkpoint,日志聚合值不干净。启动要求:500m 共享域的 token cache 必须复用 200M 的同一 pool 文件(或至少前缀同内容),否则连 16B+ 的干净 val 对比也没有。**

"最早第几个 token" = 主读数仪器首次越过阈值、且**连续两个 checkpoint 成立**的第一个 checkpoint;报分辨率区间 [T_prev, T*],不报单点。

## 1. 仪器盘点:谁有 headroom,谁是饱和的

| 仪器 | 200M 读数 | headroom | 速度 | 角色 |
|---|---|---|---|---|
| 生成通过率(R3 族,format-free + shuffle 对照 + model_turn 抽取) | 8.13% vs 2.40%(30B) | **0→~40%,唯一有信号且有量程** | 慢(全 sweep ~1h/档) | **主读数** |
| math_v2_like two-way | ~95% | 向上饱和,只能测退化 | 快(log-prob,~15min) | 护栏,不测进步 |
| L0' 代码判别(reasoning_panel §6) | 未建,200M 基线未知 | 未知——200M 若 ≤60% 则既有 headroom 又快 | 快 | §5 分支:建成且 200M 低 → 升为快主读数 |
| domain loss / BPB | 有曲线 | 趋势仪器,无能力读数 | 快 | 连续性监控,不判能力 |
| 退化率(重复窗口) | 16B 贪心 99.4% / 采样 71.0% | **前段唯一有量程的:99.4%→更低是曲线,0%→0% 不是** | 极快(无评分无 gold) | **先导指标,每档跑(§10)** |
| ceval 类知识 | 21.6%(21.9B) | 有,但答的不是"推理起作用" | 中 | 辅读数 |

**关键张力:读得出信号的仪器(生成)慢,快的仪器(math_v2_like)在 200M 已饱和。** 所以读数节奏由生成仪器的成本决定,除非 L0' 分支成立(§5)。

## 2. 决策规则(测前写死)

复用 `eval/readout_30b.py` 的三态机器(moved/floor/flat,阈值来自 prereg),不新造 harness:milestone = 500M@T,paired = 200M@T 同 token checkpoint,同 held-out head 打分。

- **WORKING:** 主读数(生成通过率 Δ vs 200M@T)> 12.6pt @ 492(2δ),连续两个 checkpoint。
- **NULL:** 全跑程主读数从未 moved。报"null over full run"+ 达到的灵敏度(可检出的最小 Δ)。
- **护栏跳闸:** math_v2_like < 80%(honest_prereg OOE-e)→ 停读生成,报"配比伤了判别仪器",数据问题不是规模问题。
- **warmup 不可读:** warmup 结束前的 checkpoint 不判(同 readout_30b 的 WARMUP-CONFOUND 逻辑);500M 的 warmup 步数由 tilerl 配方定,落地后写死边界。
- **floor 不是读数:** 提取率 <20% 的档拒报通过率(honest_prereg §4);2δ 以下 = floor。

## 3. 节奏与预算(三天机时)

- **存档要求(fb 2026-09-01 终裁,tilerl 执行):** **每 1B 一个里程碑 + 精确钉 16B 一个**(200M 存活档里唯一落在 20B 量程内的匹配点);**前 5B 每 0.4B 加密**(无 200M 对照段,只要分辨率);**1.25B 网格退役**(只买到一个匹配点,不值得让分辨率让路)。步数算术(tilerl 2026-09-01 落地):`save_every = 1e9 ÷ 917,504 = 1089.91 → 1090 步**(网格点间隔 1.00008B;第 16 个网格点 step 17440 = 16.0013B,±1.3M token,0.008%——"1B 网格"在执行层是这个精度,不存在整数步精确落在 16B)。**精确钉点 = step 17500**:500M 与 200M 同为 917,504 tok/step(b32 world=7 或 b16+accum2 都是这个数),步对齐即 token 精确对齐,200M 的 16b 里程碑就是 step17500——钉同一个步,不是钉"16.000B"。**里程碑一个不删,只有滚动档参与回收。**单档 1.84 GB(Muon 单动量 + bf16,tilerl 实测),~20 里程碑 ≈ 37 GB(可用 401 GB)。
- **滚动档不带 opt/step(tilerl 2026-09-01 实测 ckpt_shape500_probe.pt:keys 无 opt,step=None)——matched-token 比较轴只能用里程碑档。**
- **每档成本:** 生成 sweep ~1h(0/1/3/8 × 数学+代码,492 题,`--eval-from 8`,rep_stop=False,model_turn 抽取)+ math_v2_like ~15min + domain loss ~10min ≈ 1.5h。
- **分配:** 快护栏每档都跑;生成 sweep 每 3 档跑一次(~每 3B token)。30B 跑程 ≈ 10 次生成 + 30 次快读 ≈ 17.5h,三天预算内留足 L0' 与复测。
- **若 L0' 分支成立:** 快主读数每档跑,生成降为每 5 档确认——分辨率从 3B  sharpen 到 1B。

## 4. L0' 分支(建成前是本文最大的不确定性)

- 规格照 reasoning_panel §6:逐题胜率,标签来自执行;易扰动层(单算子变异,>90% 偏好未变异 = 仪器自检)+ 难扰动层(模型生成失败解,**从一个 checkpoint 冻结**,跨 checkpoint 复用)。
- 数据活是 44 的(reasoning_panel §5 点名 3b/44)。交付:数据集 + 脚本 + selftest,**在 500M 第一批 checkpoint 落地之前**。
- 分支(测前写死):200M@30B 基线 ≤60% → L0' 升为快主读数,§3 节奏改快档;>80% → 饱和,留作护栏;60–80% → 与生成并列双主读数,任一 moved 即 WORKING。

## 5. 格式分支(已解决:格式在语料里)

mix_500m.json(96b5fdd,9 域)实测组成:math_owm_stage2 26.4%(OpenWebMath,含 \boxed 解答)、cot 8.1%、**chatml 0.77% + chat_qa 0.75%(答案格式显式喂入)**、textbook_30b 10.2%、code_py_starcoder 33.0%。**分支落点:有 → 生成读数按能力读,阈值照 §2。** 与上一轮的关键差别:200M 语料 ChatML 上界 <0.075%,这一轮 chatml/chat_qa 是显式 1.5%——500M 的生成数若仍地板,不能再归因于"语料没有格式"。token cache 落地后抽查 chatml/chat_qa 域确含答案形态(\boxed/答案是/fence),作为本分支的验证脚注。

## 6. 报告纪律

- 每个数字携带抽取规则 + 测量配置(fb 2026-09-01 两条裁定)。
- 读数进展发 board `readout` 课题;WORKING/NULL 判决同时发 board + 写本文 §7。
- 对照同行:通过率、shuffle 对照、鹦鹉下限、提取率,四个数一行(honest_prereg §4)。

## 8. 2026-09-01 增补:退化率仪器(fb 裁定 + 44 设计)

**触发:de 的 16B 档 code-500 实测**——4500 条生成里 `def ` 出现率 0.2%,退化率贪心 99.4% / 采样 T=0.8 71.0%。主读数(生成通过率)在它最需要工作的前段是死的:8.13% 是 22B 档带示例的数,16B 档同模型贪心下几乎不产生文本。

**裁定(fb):退化率升为快护栏指标。设计(44):先导指标,不取代任何能力仪器,与 L0' 并存分工。**

- **测什么:** 自由生成的重复窗口率(短窗口反复到 token 预算用完)。和 free_running 的 FR/TF 崩塌、bin-1 即时掉下去是同一现象的文本级测法——**正是深度这个赌要修的那件事**。深度若对,退化率先于通过率动。
- **为什么是先导不是判决(三条,缺一不可):** (a) 解码敏感——贪心 99.4% vs 采样 71.0%,解码配置 alone 就是 28pt,读数必须钉死配置;(b) 必要不充分——退化率动可以来自解码/格式/短续写,不一定是能力;(c) 不随能力单调——早停的短续写退化更少但不更好。**WORKING 判决仍在能力仪器(通过率/L0');退化率单独动 = "校准在改善,能力未确认",是 LEAD。**
- **钉死:** 贪心为主(确定性、无种子噪声、最坏情况量程最大),T=0.8 为辅;0-shot 固定提示族(示例改变退化);同一 harness 跨模型跨 checkpoint。阈值:Δ > 4pt @ 500 条(p≈0.7 的 2δ 二项)。
- **分段结构:** A 段(0–16B,无 200M 能力对照)退化率先导 + 绝对阈值;B 段(16B+)能力 matched-token 全仪器。**1B 分辨率的实现 = 退化率(免费,每档)+ L0'(快,~15min/档)两个一起;通过率仍每 3B 确认。**
- **200M 基线:** 同一 harness 在 5 个存活 checkpoint(16B/22B/尾部三档)重跑退化率,~30 min。de 的 16B 数和 free_running 的 22B 数提示族不同,不能直接拼曲线。
- **报告纪律增补(de):** 评测脚本最后一行打印的预测文件路径可能不带 `--run` 版本号而真实文件带——读输出以真实版本化路径为准,不照日志末行读。

## 9. 结果(测后填)

(空——在 500M 数出来之前,本节不存在内容。)

## Sources

- `docs/lessons/readout_30b_prereg.md` + `eval/readout_30b.py`(三态读数机器,复用)
- `docs/lessons/reasoning_panel.md` §6(L0' 规格、冻结纪律)、§阈值(12.6pt @ 492)
- `docs/lessons/honest_measurement_prereg.md`(R3 基线、model_turn 抽取、20% 拒报线、OOE-e 护栏)
- `runs/readout_ckpt_pretrain_30b_s2.pt.step17500.txt`(200M 曲线已有点)
- `eval/math_v2_like.py`(护栏仪器,floor 50%,swap 对照)
