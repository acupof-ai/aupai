---
question: step24000(200M/30B token)的诚实能力测量——在 500M 跑完前让它有东西可比;以及 200M 尺度上什么仪器读得出推理信号
status: open
source: fb tasking 2026-09-01; builds on docs/lessons/reasoning_panel.md, fewshot_arm_prereg.md, free_running_prereg.md, eval/score_math_formatfree.py。预登记:新臂的任何数出来之前写死;在飞的 3/8 数学臂见 §0
---

# step24000 诚实能力测量(预登记)

## 0. 已经测过的(不重复)

- **ChatML 生成分数:格式门不开。** 数学 1439 条生成 0 条含 \boxed(gold 494/500 有);代码 288 条里 3 条带 fence。历史上每个 base checkpoint 的生成分数都是 OOD 提示产物(fb/de 2026-09-01 全量确认)。
- **free_running(e1,`runs/free_running.json`,已测):** teacher-forced top-1 中位 0.727(代码)/0.6875(数学)→ free-running 中位 0.0/0.021(naive),ratio 0.00/0.03;gold 对数概率只赢 45%/52.8% 的采样序列。**崩塌级,不是温和级;且模型不偏好金答案。**
- **few-shot arm(e1/de,在飞):** 数学 3-shot + 8-shot,GPU2/GPU3,读数 A/B/C 在 `fewshot_arm_prereg.md`。
- **format-free 打分在 ChatML 生成上:** 17.9% vs shuffle 对照 20.1%——低于随机,但生成本身 OOD,不构成能力读数。
- **p324 L1(`reasoning_panel.md`):** 数学 3-shot 0.2%,代码全剂量 0/497;成形剂量-反应:数学 25.4%/53.5%/63.6%(0/1/3-shot,单调、边际递减),代码 97.8–100%(天花板无量程)。

## 1. 本预登记加什么

1. 数学 0-shot、1-shot 臂(补全 0/1/3/8 四档)。
2. 代码臂 0/1/3/8——**从未跑过**(code_fewshot 需小补丁,见 §8)。
3. **抄示例对照(鹦鹉下限)**——新,任何现有 prereg 都没有。
4. 提取率优先 + 20% 拒报线(de 的 L2 规则,沿用)。
5. "确实不会" vs "会但表达不出来" 判据,测前写死(§6)。
6. 预期,测前写死(§5)。
7. 判断:200M 尺度什么仪器读得出信号(§7)。

## 2. Harness 与提示(钉死)

- **checkpoint:** `ckpt_pretrain_30b_s2.pt.step24000`(200M,d=1024,L=12,30B token)。
- **数学:** `eval/l1_fewshot.py`;`build_prompt` 纯续写 `题目：{q}\n解答：{a}`;demos = math_test_500 行 0..n−1 的完整 gold 解;**`--eval-from 8` 对全部四档** → 同一 492 题集,只有 demo 数在动。greedy(t=0),max_new 512,**rep_stop=False**(成形率需要;通过率不受影响——正确答案不触发重复停止)。
- **代码:** `eval/code_fewshot.py` 打补丁后;提示格式钉死 `题目：{q}\n```python\n{code}\n```\n运行输出：\n{out}`,目标以 `题目：{q}\n```python\n` 结尾;提取到首个闭合 fence,无 fence 取整段续写;sandbox 执行、stdout 行匹配。
- **数学打分三器:** (a) 严格:`\boxed` 或 `答案是:`(l1 内置);(b) format-free:末位数字(`score_math_formatfree` 逻辑)+ 同聚合层 shuffle 对照;(c) 成形:boxed/答案是 出现 + 含任意数字。
- **代码打分三器:** (a) 执行通过;(b) 成形:非空续写 + 含 `def `;(c) 空续写率。
- **de 的 L2 尺子(代码,最长可解析片段):** 落地后用它重打代码通过率。本预登记钉的是尺子契约(提取率优先、同聚合层 shuffle 对照、自带已知答案),不是实现。**按 fb 指示:跑等尺子。**

## 3. 对照(三个,缺一不可)

1. **shuffle 对照,同聚合层:** 数学 format-free 逐行(每条生成对另一题的 gold 末位数字);代码——每条生成的代码对另一题的 expected_output 执行(离线重打 preds)。统计量是逐行通过率,对照就是逐行;若报 pass@k,对照在题目聚合层重建(`score_math_formatfree` 已示范:8 条采样的并集对随机 gold,17.9% vs 20.1% 就是这样现形的)。
2. **抄示例对照(鹦鹉下限):** 每档 n,把 n 个 demo 的参考解用同一打分器对全部 492 题打分,报 max over demos(最强鹦鹉)。动机:de 发现生成在复述示例的 `is_prime` 函数体。**模型通过率 − 鹦鹉下限 < 2pt → 通过率是复述,不是解题。**
3. **gold round-trip(打分器自检):** `code_fewshot --selfcheck`(gold 全过、错解全挂)+ `score_math_formatfree --selftest` + gold 答案在严格/format-free 两器下 ~100%。模型跑之前跑,不通过不跑模型。

## 4. 报告顺序(提取率优先)

1. 成形/提取率,每档每域,先于一切通过率。
2. **拒报线:提取率 <20% 的档,不报通过率**——报"提取地板,通过率不可读"。
3. 通过率与对照同行:严格、format-free、shuffle 对照、鹦鹉下限,四个数一行。
4. 仪器存在阈值:**2δ = 12.6pt @ 492**(δ=1.4/√492)。通过率 <12.6pt = 地板,不是读数(`reasoning_panel` 常设规则)。

## 5. 预期(测前写死)

基于 p324 L1(数学 3-shot 0.2%;代码全剂量 0%;成形剂量-反应 25.4/53.5/63.6;代码天花板)与 step24000 = 10× token:

- **E1 成形剂量-反应:** 数学含数字率 0-shot 30–50%、1-shot 50–65%、3-shot 60–75%、8-shot 65–80%,单调非减、边际递减;代码非空 ~95–100% 全档,`def ` ≥90% 全档(天花板,无量程)。
- **E2 通过率地板:** 数学严格 ≤3% 全档;format-free 在 shuffle 对照 +2pt 以内全档;代码 ≤3% 全档。
- **E3 鹦鹉下限低:** 数学 2–8%(小整数巧合);代码 0–2%。
- **E4 8-vs-3:** 成形 +3–7pt(数学);通过率不变(地板)。
- **E5 判别完好:** math_v2_like two-way 在 step24000 ≥90%(阶梯上 94.69;30B token 不应降低它)。

**预期外的结果及预指派含义(测前写死,不许事后二选一):**

- **OOE-a:** 数学 format-free 任一档 > shuffle +5pt → 30B token 上有可读算术能力;500M 有了正基线。结论从"无仪器"变"仪器存在、弱"。
- **OOE-b:** 代码通过率 >12.6pt → 生成代码仪器存在;SFT/代码政策工作重开。
- **OOE-c:** 数学 0-shot 成形 <15%(低于 p324 的 25.4%)→ 退化或提示错配;先查再读任何其他数。
- **OOE-d:** 鹦鹉下限与模型通过率差 <2pt → 通过率是复述;照此报,不做能力解读。
- **OOE-e:** math_v2_like <80% → 判别随 token 崩塌;30B 配比伤了唯一能用的仪器——数据问题,不是规模问题。

## 6. 判据:"确实不会" vs "会但表达不出来"(测前写死,互斥)

- **R1 确实不会:** 最佳档(成形最高)提取率 ≥20% 且 format-free 通过率 ≤ shuffle+2pt 且严格 ≤3% 且 通过率−鹦鹉下限 ≤2pt。模型产出答案形状的文本,不比随机或复述好。配上 free_running FR/TF ≤0.05(已测),判决:**200M/30B 上生成式推理缺席,不是格式阻塞。**
- **R2 判别残留(注意:不是经典的"会但表达不出来"):** 全部档提取率 <20%(格式诱导失败)且 math_v2_like ≥90%。模型能在两个候选里分对错,但产不出、**且不偏好金答案**(gold beats sampled 0.45/0.528 ≈ 随机)。经典"会但表达不出来"需要金偏好 ≥60%,当前已测数不满足——如实写成"判别残留",不借经典版的光。
- **R3 会一点:** 提取率 ≥20% 且 format-free > shuffle+2pt 但 <12.6pt → 弱可读信号;报 Δ 和剂量-反应,不下能力判决。
- **R4 不可判:** 提取率有的档 ≥20% 有的 <20%,或对照含糊 → 报数不判;写明加哪个档能判。

## 7. 判断:200M 尺度什么仪器读得出信号(fb 的问题)

- **生成类仪器在 200M/30B:全格式脆弱或地板**(本预登记的预期发现;p324 的发现;free_running FR~0)。
- **判别类仪器:** math_v2_like two-way ~95%(读得出);MC 套件 3/5 随机(混合);gold-vs-sampled ~0.5(读不出)。
- **文献(上午综述,`small_model_composition.md`):** ≤500M 模型在 600B token 以下没有生成式推理先例;SmolLM2-135M 的 MMLU 31.5 是 **cloze 形式**测的——判别,不是生成。这个领域自己在小规模的读数就是判别/cloze。
- **预登记的回答(待 §5/§6 的数确认):** 200M 尺度读得出信号的仪器族是 **log-prob 判别**(two-way 偏好、cloze)。若 step24000 确认 R1(生成地板)且判别完好,500M 读数按判别优先设计:(a) math_v2_like + 代码判别(`reasoning_panel.md` §6 的 L0' 规格——建)作为主读数;(b) 生成读数只有在 500M 语料包含答案格式时才有意义——否则 500M 的生成数会重演这次仪器故障;(c) 500M 对照预登记为判别 Δ,生成为辅。
- **若判别也崩**(R2 不满足、math_v2_like <80%):更强的结论——200M/30B 没有任何仪器读得出推理信号;500M 读数设计从"什么仪器读得出"重新开始,不从现有面板继承。

## 8. 执行

- pod,一张卡,~1 小时;**等 de 的 L2 代码尺子落地再跑**(fb 指示)。
- 数学 0/1/3/8 全档重跑(`--eval-from 8`,同一 492 题)。在飞的 3/8 臂用默认 rep_stop=True 且 3-shot 默认评 497 题——通过率仍可作独立确认,但剂量-反应比较用本预登记的同题集 sweep(避免总体变化混淆,de 在 `split_rows` 修的正是这个)。
- 代码臂前补丁:`code_fewshot.py` DEMO_POOL 3→8、`choices=[0,1,3,8]`、加 `--eval-from`(默认 8);补丁后重跑 `--selfcheck`。
- 产物:preds 落 `data/eval/`(gitignored,pod 侧),`attest` 哈希;读数写 §9。

## 9. 结果(测后填)

(空——在数出来之前,本节不存在内容。)

## Sources

- `docs/lessons/reasoning_panel.md`(L1 阈值 12.6pt、p324 读数、代码判别 L0' 规格)
- `docs/lessons/fewshot_arm_prereg.md`(在飞的 3/8 数学臂,A/B/C 读数)
- `docs/lessons/free_running_prereg.md` + `runs/free_running.json`(FR 已测:0.0/0.021)
- `eval/score_math_formatfree.py`(de 的 L2 尺子,数学侧;同聚合层对照的示范)
- `eval/l1_fewshot.py`、`eval/code_fewshot.py`(harness)
- `docs/lessons/small_model_composition.md` §1–2(文献:小规模无生成式推理先例;cloze 是领域惯例)
