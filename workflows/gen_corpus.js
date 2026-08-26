export const meta = {
  name: 'gen-corpus',
  description: 'Generate ~2000 Chinese training docs from the skeleton seed framework',
  phases: [
    { title: 'Write', detail: '147 writer agents, ~14 docs each' },
  ],
}

const ROOT = '/Users/bytedance/code/aupai'

const SKELETONS = [
  ['F', '反馈', '结果回头改变原因：正反馈放大，负反馈拉回'],
  ['E', '耗散', '不持续投入维护的东西，默认走向混乱衰败'],
  ['S', '筛选', '多样性 + 淘汰 = 自动适应环境'],
  ['C', '复利', '小变化率 × 足够长时间 = 巨大差距'],
  ['L', '约束', '最稀缺的东西决定整体形状和上限'],
  ['M', '边际', '多做一单位的收益代价随数量变化'],
  ['P', '路径依赖', '早期选择锁死后来选项，改比选贵'],
  ['G', '涌现', '简单个体简单规则互动出整体新性质'],
  ['T', '权衡', '得到一样往往要放弃另一样'],
  ['I', '信息差', '谁知道什么、知道多少，决定谁占便宜'],
]

const ENTRIES = [
  'A1 能量守恒', 'A2 熵增', 'A3 最小阻力路径', 'A4 尺度效应', 'A5 相变与临界点', 'A6 惯性',
  'B1 演化=变异+筛选+时间', 'B2 活着就是逆着熵走', 'B3 适应有代价', 'B4 红皇后效应', 'B5 局部最优陷阱', 'B6 冗余换稳健',
  'C1 激励决定行为', 'C2 损失比收益更痛', 'C3 省力是本能', 'C4 人只看得见自己在找的东西', 'C5 习惯是压缩的决定', 'C6 身份先于观点', 'C7 相对比绝对重要',
  'D1 稀缺是一切价格的来源', 'D2 竞争把利润压向零', 'D3 分工放大产出', 'D4 沉没成本', 'D5 边际递减', 'D6 杠杆', 'D7 网络效应', 'D8 外部性',
  'E1 制度是固化的激励', 'E2 规模改变性质', 'E3 协调成本是隐形税', 'E4 公地悲剧', 'E5 信念会自我实现', 'E6 地位是零和的', 'E7 稳健的系统都有冗余和缓冲', 'E8 信任降低一切成本',
  'F1 地图不是疆域', 'F2 技术是叠上去的', 'F3 抽象让复杂可以堆叠', 'F4 信息会腐烂', 'F5 复制便宜创造昂贵', 'F6 工具改变使用者', 'F7 测量什么就得到什么',
  'G1 大多数事情是随机加一点趋势', 'G2 不可逆的决定要慢可逆的要快', 'G3 极端事件决定长期结果', 'G4 缓慢累积突然爆发', 'G5 时间是最大的杠杆', 'G6 机会成本是真正的成本', 'G7 简单规则+迭代大于复杂计划',
]

const DOMAINS = [
  '职场', '投资', '学习', '健康', '亲密关系', '育儿', '城市生活', '互联网', '创业', '体育',
  '烹饪', '旅行', '消费', '社交', '写作', '编程', '理财', '买房', '养老', '宠物',
  '音乐', '电影', '游戏', '时尚', '农业', '制造业', '医疗', '教育', '法律', '媒体',
  '广告', '零售', '物流', '餐饮', '酒店', '银行', '保险', '房地产', '汽车', '能源',
  '环保', '公益', '政府', '宗教', '哲学', '科学', '艺术', '文学', '军事', '社区',
]

const PHENOMENA = [
  '银行挤兑', '内卷', '学区房', '网红过气', '共享单车泡沫', '外卖补贴大战', '芯片卡脖子', '老龄化',
  '少子化', '远程办公', '知识付费', '健身房跑路', '奶茶店倒闭潮', '直播带货', '拼多多崛起', '日本失去的三十年',
  '郁金香泡沫', '高考', '考公热', '房价', '股市散户亏钱', '基金抱团', '偶像塌房', '饭圈互撕',
  '游戏氪金', '滴滴快的大战', '社区团购', '在线教育崩盘', '新能源车价格战', '光伏产能过剩', '躺平', '颜值经济',
  '银发经济', '孤独经济', '宠物经济', '地摊经济', '零工经济', '注意力经济', '会员订阅', '免费模式',
  '马太效应', '二八定律', '颠覆式创新', '私域流量', '种草经济', '下沉市场', '消费降级', '品牌溢价',
  '平替', '临期食品', '二手经济', '数字游民', '副业刚需', '延迟退休', '养老金缺口', '药品集采',
  '医美热', '露营热', '飞盘热', '马拉松热', '电竞入奥', '剧本杀', '脱口秀走红', '国潮',
  '汉服热', '盲盒', '手办', '二次元', 'B站弹幕', '小红书种草', '抖音算法', '信息茧房',
  '标题党', '谣言传播', '网络暴力', '回声室', '群体极化', '拖延症', '手机依赖', '信息焦虑',
  '懂得很多道理却过不好一生',
]

const STYLE = `你在为一个小型中文语言模型写训练语料。母本是一套"世界底层规律"框架：10 个骨架代号 + 55 个条目。你要模仿母本风格写新内容。

## 骨架表
F 反馈：结果回头改变原因，正反馈放大，负反馈拉回
E 耗散：不持续投入维护的东西，默认走向混乱衰败
S 筛选：多样性加淘汰，自动适应环境
C 复利：小变化率乘足够长时间，变成巨大差距
L 约束：最稀缺的东西决定整体形状和上限
M 边际：多做一单位的收益代价随数量变化
P 路径依赖：早期选择锁死后来选项
G 涌现：简单个体按简单规则互动，出现整体新性质
T 权衡：得到一样往往要放弃另一样
I 信息差：谁知道什么、知道多少，决定谁占便宜

## 条目表（交叉引用只能用这些代号）
A1 能量守恒 / A2 熵增 / A3 最小阻力路径 / A4 尺度效应 / A5 相变与临界点 / A6 惯性
B1 演化=变异+筛选+时间 / B2 活着就是逆着熵走 / B3 适应有代价 / B4 红皇后效应 / B5 局部最优陷阱 / B6 冗余换稳健
C1 激励决定行为 / C2 损失比收益更痛 / C3 省力是本能 / C4 人只看得见自己在找的东西 / C5 习惯是压缩的决定 / C6 身份先于观点 / C7 相对比绝对重要
D1 稀缺是一切价格的来源 / D2 竞争把利润压向零 / D3 分工放大产出 / D4 沉没成本 / D5 边际递减 / D6 杠杆 / D7 网络效应 / D8 外部性
E1 制度是固化的激励 / E2 规模改变性质 / E3 协调成本是隐形税 / E4 公地悲剧 / E5 信念会自我实现 / E6 地位是零和的 / E7 稳健的系统都有冗余和缓冲 / E8 信任降低一切成本
F1 地图不是疆域 / F2 技术是叠上去的 / F3 抽象让复杂可以堆叠 / F4 信息会腐烂 / F5 复制便宜创造昂贵 / F6 工具改变使用者 / F7 测量什么就得到什么
G1 大多数事情是随机加一点趋势 / G2 不可逆的决定要慢可逆的要快 / G3 极端事件决定长期结果 / G4 缓慢累积突然爆发 / G5 时间是最大的杠杆 / G6 机会成本是真正的成本 / G7 简单规则+迭代大于复杂计划

## 风格规则
1. 口语化，说人话，像跟聪明的朋友解释。具体例子优先，不要抽象定义堆砌。
2. 每篇必须点出踩在哪几个骨架代号上（F/E/S/C/L/M/P/G/T/I）。
3. 因果链用 → 连接，每一步要具体，不能跳。
4. 禁止套话：不要"总之""综上所述""首先其次最后""在当今社会""值得注意的是"。
5. 不要重复母本已有的例子（房间变乱、丢一百块、水往低处流那些），全部用新场景。
6. 字数：散文 400-600 字；结构化条目 200-350 字；现象拆解 500-700 字；问答的答 300-500 字。
7. 交叉引用只用条目表里的真实代号（A1-G7），不要引用 N 开头的代号。

## 格式范本

[散文条目]
领导一表扬谁，谁就干得更起劲，这叫正反馈。行为带来奖励，行为被强化，职场里几乎所有的劲头都是这么来的。但反馈的方向决定结果：表扬加班，加班就越来越多；表扬产出，摸鱼就越来越少。所以看一个团队往哪走，不要看墙上的标语，看奖励落在哪。这是 F 反馈最直接的形态，它和 C1 激励决定行为是同一件事的两面，踩的骨架是 F 和 I。

[结构化条目]
**N017 市场是最狠的筛选器** \`S\`
不需要谁判断对错，亏钱的策略自己会消失。
链：策略有差异 → 市场淘汰亏钱的 → 留下的继续跑 → 环境一变，昨天的赢家今天被淘汰。
关联：B1、D2、B4

[现象拆解]
**内卷**
每个人都更努力了，但没有人过得更好，这是 B4 红皇后效应：你进步，对手也进步，相对位置不变。它踩在 F 反馈上——努力的回报被竞争者的努力互相抵消。稀缺的是位置，不是努力（L 约束），所以努力的边际收益递减（D5），到最后多学一小时只是不掉队。出口不在更努力，在换一条人少的赛道；但换赛道的第一步一定是变差的，这正是 B5 局部最优陷阱困住大多数人的地方。

[问答]
问：为什么懂得很多道理，还是过不好这一生？
答：因为知识存在理性那一层，行为由习惯和情绪驱动，两层之间没有直通的管道。这是 C5 习惯和 C3 省力本能的合力：默认行为走的是被踩平的老路，不是新懂的道理。改行为靠改环境和重复，不靠再多懂一条道理。所以问"我该怎么办"之前，先问"我的环境把哪条路踩平了"。踩的骨架是 P 和 L。`

function buildAssignments() {
  const out = []
  const proseAngles = ['从一个身边小事切入', '从一个反直觉的场景切入', '从一个历史或新闻案例切入']
  const structAngles = ['给个人用的版本', '给组织用的版本', '给市场竞争用的版本']

  let i = 0
  for (const [code, name] of SKELETONS)
    for (const d of DOMAINS)
      for (const a of proseAngles) {
        if (i % 2 === 0) out.push({ type: 'prose_entry', skeleton: `${code} ${name}`, domain: d, angle: a })
        i++
      }

  let ncode = 1
  i = 0
  for (const [code, name] of SKELETONS)
    for (const d of DOMAINS)
      for (const a of structAngles) {
        if (i % 3 === 0) out.push({ type: 'struct_entry', ncode: `N${String(ncode++).padStart(3, '0')}`, skeleton: `${code} ${name}`, domain: d, angle: a })
        i++
      }

  const phAngles = [
    '整体拆解：它踩在哪几个骨架和条目上',
    '谁占便宜谁吃亏',
    '长期会走向哪里',
    '和哪些条目连着，顺着连接走一圈',
    '如果外力干预会怎样',
  ]
  for (const p of PHENOMENA)
    for (const a of phAngles)
      out.push({ type: 'phenomenon', subject: p, angle: a })

  for (const e of ENTRIES) {
    const name = e.split(' ').slice(1).join(' ')
    out.push({ type: 'qa', q: `为什么说「${name}」？` })
    out.push({ type: 'qa', q: `「${name}」在日常生活里有什么例子？` })
    out.push({ type: 'qa', q: `「${name}」和别的条目怎么配合着用？` })
  }
  for (const [code, name] of SKELETONS) {
    out.push({ type: 'qa', q: `什么是${name}（${code}）？` })
    out.push({ type: 'qa', q: `${name}（${code}）的正例和反例各是什么？` })
    out.push({ type: 'qa', q: `怎么用${name}（${code}）分析一个具体问题？` })
    out.push({ type: 'qa', q: `${name}（${code}）和其他骨架怎么配合？` })
  }
  const dq = [
    d => `在${d}里，最稀缺的是什么（L）？`,
    d => `在${d}里，反馈在放大还是在拉回（F）？`,
    d => `在${d}里，哪一步走了就回不来（P）？`,
    d => `用三问法（稀缺、反馈方向、不可逆步骤）分析${d}里的一个具体问题`,
  ]
  for (const d of DOMAINS)
    for (const f of dq)
      out.push({ type: 'qa', q: f(d) })

  return out
}

function writerPrompt(batch, idx) {
  const pad = n => String(n).padStart(3, '0')
  const file = `${ROOT}/data/corpus/batch_${pad(idx + 1)}.jsonl`
  const lines = batch.map((a, k) => {
    if (a.type === 'prose_entry') return `${k + 1}. [散文条目] 骨架=${a.skeleton}；场景=${a.domain}；角度=${a.angle}`
    if (a.type === 'struct_entry') return `${k + 1}. [结构化条目] 代号=${a.ncode}；骨架=${a.skeleton}；场景=${a.domain}；角度=${a.angle}`
    if (a.type === 'phenomenon') return `${k + 1}. [现象拆解] 主题=${a.subject}；角度=${a.angle}`
    return `${k + 1}. [问答] 问题=${a.q}`
  }).join('\n')
  return `${STYLE}

## 你的任务（${batch.length} 篇）
${lines}

## 输出方式
用 Write 工具把 ${batch.length} 篇写到文件：${file}

格式：每行一个 JSON 对象，UTF-8，共 ${batch.length} 行：
{"type":"prose_entry|struct_entry|phenomenon|qa","title":"短标题","content":"正文全文"}

要求：
- content 是完整正文：散文直接写；结构化条目含 **代号 名字** 那一行；问答含"问：...\n答：..."两部分（\\n 是 JSON 转义的换行）。
- title：散文=自拟条目名（4-10字）；结构化=代号+名字；现象=现象名；问答=问题原文。
- 不要 markdown 代码块包裹文件内容，不要任何额外解释，文件里只有 JSONL。
- 一篇都不能少。

写完后只返回清单（不要返回正文）：
{"batch":${idx + 1},"file":"${file}","count":实际篇数,"chars":总字数}`
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

const BATCH = 14
const assignments = buildAssignments()
const nb = Math.ceil(assignments.length / BATCH)
const buckets = Array.from({ length: nb }, () => [])
assignments.forEach((a, j) => buckets[j % nb].push(a))

log(`${assignments.length} assignments → ${nb} batches`)

phase('Write')
const manifests = await parallel(buckets.map((batch, idx) => () =>
  agent(writerPrompt(batch, idx), {
    label: `w${String(idx + 1).padStart(3, '0')}`,
    phase: 'Write',
    agentType: 'general-purpose',
    schema: MANIFEST,
  })
))

const ok = manifests.filter(Boolean)
const docs = ok.reduce((s, m) => s + m.count, 0)
const chars = ok.reduce((s, m) => s + m.chars, 0)
log(`done: ${ok.length}/${nb} batches, ${docs} docs, ${chars} chars`)
return { batches: nb, ok: ok.length, docs, chars, missing: nb - ok.length }
