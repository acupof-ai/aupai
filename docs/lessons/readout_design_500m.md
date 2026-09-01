---
question: 500M 读数设计——最早在第几个 token,能可靠区分"500M 这次在起作用"和"又是上一轮那个结果";在 500M 任何 checkpoint 落地之前钉死
status: open
source: fb tasking 2026-09-01("这个答案值三天机时"); builds on readout_30b_prereg.md、reasoning_panel.md §6、honest_measurement_prereg.md。决策规则在数出来之前写死
---

# 500M 读数设计(预登记)

## 0. 问题精确化

"起作用"不是"500M 比 200M 强"——是 **500M 在 matched-token 上的仪器曲线离开 200M 的曲线,超过仪器噪声**。比较轴是 token 数,不是 checkpoint 序号:500M@T vs 200M@T,T 相同。200M 曲线 = `ckpt_pretrain_30b_s2`。**pod /work/aupai 实测(2026-09-01):存活仅 5 点**——step17500(16B)、step24000(22B)、step25500/26000/26500(尾部 500 步密档,~22–24.3B);0–16B 无存活 checkpoint,跑程止于 step26500(~24.3B)。**matched-token 轴从 16B 才开始;16B 以下的 500M checkpoint 用绝对阈值(仪器地板 + 2δ)读,是 weaker 读法,如实标注。**

"最早第几个 token" = 主读数仪器首次越过阈值、且**连续两个 checkpoint 成立**的第一个 checkpoint;报分辨率区间 [T_prev, T*],不报单点。

## 1. 仪器盘点:谁有 headroom,谁是饱和的

| 仪器 | 200M 读数 | headroom | 速度 | 角色 |
|---|---|---|---|---|
| 生成通过率(R3 族,format-free + shuffle 对照 + model_turn 抽取) | 8.13% vs 2.40%(30B) | **0→~40%,唯一有信号且有量程** | 慢(全 sweep ~1h/档) | **主读数** |
| math_v2_like two-way | ~95% | 向上饱和,只能测退化 | 快(log-prob,~15min) | 护栏,不测进步 |
| L0' 代码判别(reasoning_panel §6) | 未建,200M 基线未知 | 未知——200M 若 ≤60% 则既有 headroom 又快 | 快 | §5 分支:建成且 200M 低 → 升为快主读数 |
| domain loss / BPB | 有曲线 | 趋势仪器,无能力读数 | 快 | 连续性监控,不判能力 |
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

- **存档要求(给 tilerl):** save_every ≤ 2B token,且在与 200M 存档匹配的 token 点(1.25B 的倍数)必须有存档。粗于 2B,"最早 token"的答案分辨率不值三天机时。
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

## 7. 结果(测后填)

(空——在 500M 数出来之前,本节不存在内容。)

## Sources

- `docs/lessons/readout_30b_prereg.md` + `eval/readout_30b.py`(三态读数机器,复用)
- `docs/lessons/reasoning_panel.md` §6(L0' 规格、冻结纪律)、§阈值(12.6pt @ 492)
- `docs/lessons/honest_measurement_prereg.md`(R3 基线、model_turn 抽取、20% 拒报线、OOE-e 护栏)
- `runs/readout_ckpt_pretrain_30b_s2.pt.step17500.txt`(200M 曲线已有点)
- `eval/math_v2_like.py`(护栏仪器,floor 50%,swap 对照)
