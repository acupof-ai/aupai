export const meta = {
  name: 'gen-corpus-v3',
  description: 'Primary expansion: 60 entries, one general variant each',
  phases: [
    { title: 'Write', detail: '5 agents, 12 docs each' },
  ],
}

const ROOT = '/Users/bytedance/code/aupai'

const ENTRIES = [
  'A1 能量守恒', 'A2 熵增', 'A3 最小阻力路径', 'A4 尺度效应', 'A5 相变与临界点', 'A6 惯性',
  'B1 演化=变异+筛选+时间', 'B2 活着就是逆着熵走', 'B3 适应有代价', 'B4 红皇后效应', 'B5 局部最优陷阱', 'B6 冗余换稳健',
  'C1 激励决定行为', 'C2 损失比收益更痛', 'C3 省力是本能', 'C4 人只看得见自己在找的东西', 'C5 习惯是压缩的决定', 'C6 身份先于观点', 'C7 相对比绝对重要',
  'D1 稀缺是一切价格的来源', 'D2 竞争把利润压向零', 'D3 分工放大产出', 'D4 沉没成本', 'D5 边际递减', 'D6 杠杆', 'D7 网络效应', 'D8 外部性',
  'E1 制度是固化的激励', 'E2 规模改变性质', 'E3 协调成本是隐形税', 'E4 公地悲剧', 'E5 信念会自我实现', 'E6 地位是零和的', 'E7 稳健的系统都有冗余和缓冲', 'E8 信任降低一切成本',
  'F1 地图不是疆域', 'F2 技术是叠上去的', 'F3 抽象让复杂可以堆叠', 'F4 信息会腐烂', 'F5 复制便宜创造昂贵', 'F6 工具改变使用者', 'F7 测量什么就得到什么',
  'G1 大多数事情是随机加一点趋势', 'G2 不可逆的决定要慢可逆的要快', 'G3 极端事件决定长期结果', 'G4 缓慢累积突然爆发', 'G5 时间是最大的杠杆', 'G6 机会成本是真正的成本', 'G7 简单规则+迭代大于复杂计划',
  'H1 测量误差', 'H3 学习靠错误', 'H6 壁垒四种', 'H11 二八定律', 'H15 三问定位法',
]

const STYLE = `你在为一个小型中文语言模型写训练语料。这是一级扩散：把每个条目用日常语言重新讲一遍，用全新的例子，保留因果链和失效边界。一级文档是二级扩散（领域扩展）的基础，所以要写得扎实、可扩展。

## 硬规则
1. 口语化说人话，像跟聪明的朋友解释。具体例子优先，不要抽象定义堆砌。
2. 必须用全新的例子，不要重复母本已有的（房间变乱、丢一百块、水往低处流那些）。
3. 因果链用 → 连接，每步具体。
4. 必须包含失效边界：这条规律在什么条件下不成立。
5. 禁止套话：不要"总之""综上所述""首先其次最后""值得注意的是"。
6. 每篇 400-600 字，必须点出骨架代号（F/E/S/C/L/M/P/G/T/I）。
7. 推理逻辑遵循四动作：还原机制（拆链追问）、找反例（失效边界）、跨领域（留扩展空间）、反向预测（隐含前提）。

## 格式范本
**C2 损失比收益更痛**
丢一百块的难受，比捡一百块的高兴强烈得多。大脑对损失的敏感度是收益的两倍左右，这不是性格问题，是硬件设定。新例子：炒股的人拿不住盈利单、死扛亏损单——赚 10% 就急着卖，亏 30% 反而装死，因为卖亏损等于承认损失，痛苦被放大。链：损失敏感 → 拒绝兑现亏损 → 亏损单越拖越大 → 回本越远越不肯卖。失效边界：当止损由机制而非人执行时（程序化交易的硬性止损线），规律被绕过——所以交易系统比交易心态管用。隐含前提：人对损失的感受是即时的、情绪化的，而对收益的感受是延迟的、理性的。踩的骨架：M、P。`

const BATCH = 12
const batches = []
for (let i = 0; i < ENTRIES.length; i += BATCH) {
  batches.push(ENTRIES.slice(i, i + BATCH).map((e, k) => ({ entry: e, idx: i + k })))
}

function writerPrompt(batch, batchIdx) {
  const lines = batch.map((a, k) => {
    const file = `${ROOT}/data/corpus/primary/entry_${String(a.idx + 1).padStart(3, '0')}.jsonl`
    return `${k + 1}. ${a.entry} → 用 Write 写到 ${file}`
  }).join('\n')
  return `${STYLE}

## 你的任务（${batch.length} 篇）
对以下每个条目，各写一篇一级扩散文档：
${lines}

## 输出方式
每个条目写一个文件，用 Write 工具，路径如上。每个文件一行 JSON：
{"type":"primary","title":"条目名","content":"正文全文"}

要求：
- content 是完整正文，含 **标题** 行；换行在 JSON 里写成 \\n。
- 不要 markdown 代码块，不要额外解释。
- 一篇都不能少。

写完后只返回清单：
{"batch":${batchIdx + 1},"count":${batch.length},"files":[${batch.map(a => `"entry_${String(a.idx + 1).padStart(3, '0')}.jsonl"`).join(',')}]}`
}

const MANIFEST = {
  type: 'object',
  properties: {
    batch: { type: 'integer' },
    count: { type: 'integer' },
    files: { type: 'array', items: { type: 'string' } },
  },
  required: ['batch', 'count', 'files'],
}

log(`${ENTRIES.length} entries → ${batches.length} batches`)

phase('Write')
const manifests = await parallel(batches.map((batch, idx) => () =>
  agent(writerPrompt(batch, idx), {
    label: `v3-${String(idx + 1).padStart(3, '0')}`,
    phase: 'Write',
    agentType: 'general-purpose',
    schema: MANIFEST,
  })
))

const ok = manifests.filter(Boolean)
const docs = ok.reduce((s, m) => s + m.count, 0)
log(`done: ${ok.length}/${batches.length} batches, ${docs} docs`)
return { batches: batches.length, ok: ok.length, docs }
