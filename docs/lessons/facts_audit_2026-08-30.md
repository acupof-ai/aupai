---
question: "facts/ 里有多少条 measured fact 的 config 在今天的架构和语料下已不成立——20 条抽样审计,判据抽样前写死"
status: recorded
source: "aupai-fb 任务 2026-08-30;facts/*.json"
---

# facts 抽样审计(2026-08-30)

## 判据(抽样前写死,2026-08-30)

架构分界:旧架构 = 0830v1 及更早(1024 滑窗、attn_res_blocks=4、vocab 32773、chunk 64,
已退至 pod retired_pre0830v2/);新架构 = 0830v2+(KDA + full causal MLA + AttnRes,
滑窗已删 b3cad87,chunk_size=32)。语料分界:data/corpus/web/ 已并入 web_hq(无质量切),
留出集已从 web 剔除。

对每条 sampled fact,判 **"config 不成立"** 当且仅当满足任一:

- **C1 架构锚**:数值测于旧架构 checkpoint,且 fact 未标注"仅旧架构校准用",
  且该数值可被用作新架构曲线/六点的锚点。
  方法规则类(公式、协议、文献、构造规则)不算;已标注 OLD-arch-only 的不算
  (标签本身就是 config,它作为校准仍然成立)。
- **C2 语料锚**:config 引用的语料状态已不成立(data/corpus/web/ 路径、
  质量切假设、留出前的语料状态)。
- **C3 默认值锚**:config 引用的代码默认值已变(warmup、chunk_size 等),
  且 fact 把默认值当实际值用。

抽样:seed=20260830,`random.Random(seed).sample` 从全部 facts/*.json 中
status=="measured" 的 fact 里抽 20 条。

## 结果

(抽样后填)
