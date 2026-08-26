export const meta = {
  name: 'gen-corpus-v4',
  description: 'Secondary expansion: 60 entries x 30 domains, reading primary docs',
  phases: [
    { title: 'Write', detail: '120 agents, 15 docs each' },
  ],
}

const ROOT = '/Users/bytedance/code/aupai'

const DOMAINS = [
  '职场', '投资', '学习', '健康', '亲密关系', '育儿', '城市生活', '互联网', '创业', '体育',
  '烹饪', '旅行', '消费', '社交', '写作', '编程', '理财', '买房', '养老', '宠物',
  '音乐', '电影', '游戏', '时尚', '农业', '制造业', '医疗', '教育', '法律', '媒体',
  '广告', '零售', '物流', '餐饮', '酒店', '银行', '保险', '房地产', '汽车', '能源',
  '环保', '公益', '政府', '宗教', '哲学', '科学', '艺术', '文学', '军事', '社区',
]

const STYLE = `你在为一个小型中文语言模型写训练语料。这是二级扩散：把一级扩散文档（一个条目的基础变体）扩展到具体领域。一级文档是"这个规律是什么"，二级文档是"这个规律在这个领域里长什么样"。

## 硬规则
1. 先读一级文档，抓住它的核心机制和因果链。
2. 扩展到指定领域时，不是换名词，是换成本领域真实的机制：这个领域里的"损失"具体是什么、"反馈"具体是什么、"约束"具体是什么。
3. 必须用这个领域里的具体例子，不要泛泛而谈。
4. 因果链用 → 连接，每步具体。
5. 必须包含失效边界：这个规律在这个领域里什么条件下不成立。
6. 禁止套话：不要"总之""综上所述""首先其次最后""值得注意的是"。
7. 每篇 400-600 字，必须点出骨架代号（F/E/S/C/L/M/P/G/T/I）。
8. 推理逻辑遵循四动作：还原机制（链在领域里的映射）、找反例（领域里的失效边界）、跨领域（和一级文档的同构检验）、反向预测（领域里的隐含前提）。

## 格式范本
**C2 在投资里**
一级文档讲的是损失厌恶：丢一百块比捡一百块更痛。搬到投资里，这个机制变成：亏一万的痛苦远大于赚一万的快乐，所以散户拿不住盈利单、死扛亏损单。链映射：损失敏感 → 拒绝兑现亏损 → 亏损单越拖越大 → 回本越远越不肯卖。这个领域里的具体例子：2015 年股灾，很多人从 5000 点扛到 3000 点不卖，不是因为判断会涨，是因为卖了就承认亏了。失效边界：当止损由机制而非人执行时（程序化交易的硬性止损线），规律被绕过；所以交易系统比交易心态管用。和一级文档的同构检验：机制相同（损失敏感），场景不同（金额放大、决策频率降低），是同构不是类比。踩的骨架：M、P。`

const N_ENTRIES = 60
const N_DOMAINS = 30
const BATCH = 15

// 每个条目 2 个 agent，每个 agent 15 个领域
const batches = []
for (let i = 0; i < N_ENTRIES; i++) {
  for (let half = 0; half < 2; half++) {
    const domains = []
    for (let k = 0; k < BATCH; k++) {
      domains.push(DOMAINS[(i * 7 + (half * BATCH + k) * 13) % 50])
    }
    batches.push({ entryIdx: i, domains })
  }
}

function writerPrompt(batch, idx) {
  const pad = n => String(n).padStart(3, '0')
  const primaryFile = `${ROOT}/data/corpus/primary/entry_${pad(batch.entryIdx + 1)}.jsonl`
  const outFile = `${ROOT}/data/corpus/batch4_${pad(idx + 1)}.jsonl`
  const domainList = batch.domains.map((d, k) => `${k + 1}. ${d}`).join('\n')
  return `${STYLE}

## 你的任务
用 Read 工具读取一级扩散文档：${primaryFile}

把它扩展到以下 ${batch.domains.length} 个领域：
${domainList}

## 输出方式
用 Write 工具把 ${batch.domains.length} 篇写到文件：${outFile}

格式：每行一个 JSON 对象，UTF-8，共 ${batch.domains.length} 行：
{"type":"secondary","title":"条目名在领域名","content":"正文全文"}

要求：
- content 是完整正文，含 **标题** 行；换行在 JSON 里写成 \\n。
- title：{条目名}在{领域}。
- 不要 markdown 代码块包裹文件内容，不要任何额外解释，文件里只有 JSONL。
- 一篇都不能少。

写完后只返回清单（不要返回正文）：
{"batch":${idx + 1},"file":"${outFile}","count":${batch.domains.length},"chars":总字数}`
}

const MANIFEST = {
  type: 'object',
  properties: {
    batch: { type: 'integer' },
    file: { type: 'string' },
    count: { type: 'integer' },
    chars: { type: 'integer' },
  },
  required: ['batch', 'file', 'count', 'chars'],
}

log(`${N_ENTRIES} entries × ${N_DOMAINS} domains → ${batches.length} batches`)

phase('Write')
const manifests = await parallel(batches.map((batch, idx) => () =>
  agent(writerPrompt(batch, idx), {
    label: `v4-${String(idx + 1).padStart(3, '0')}`,
    phase: 'Write',
    agentType: 'general-purpose',
    schema: MANIFEST,
  })
))

const ok = manifests.filter(Boolean)
const docs = ok.reduce((s, m) => s + m.count, 0)
const chars = ok.reduce((s, m) => s + m.chars, 0)
log(`done: ${ok.length}/${batches.length} batches, ${docs} docs, ${chars} chars`)
return { batches: batches.length, ok: ok.length, docs, chars, missing: batches.length - ok.length }
