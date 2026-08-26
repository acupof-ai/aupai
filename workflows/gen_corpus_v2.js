export const meta = {
  name: 'gen-corpus-v2',
  description: 'Generate ~1400 reasoning docs: mechanism restoration, contradiction, isomorphism, reverse prediction, field analysis',
  phases: [
    { title: 'Write', detail: '~101 writer agents, 14 docs each' },
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

// 表面矛盾的条目对
const PAIRS = [
  ['D5 边际递减', 'D7 网络效应'],
  ['B1 演化=变异+筛选+时间', 'E1 制度是固化的激励'],
  ['C3 省力是本能', 'C5 习惯是压缩的决定'],
  ['E8 信任降低一切成本', 'C1 激励决定行为'],
  ['B6 冗余换稳健', 'E7 稳健的系统都有冗余和缓冲'],
  ['D2 竞争把利润压向零', 'D1 稀缺是一切价格的来源'],
  ['G1 大多数事情是随机加一点趋势', 'G4 缓慢累积突然爆发'],
  ['A2 熵增', 'B2 活着就是逆着熵走'],
  ['C2 损失比收益更痛', 'G3 极端事件决定长期结果'],
  ['F5 复制便宜创造昂贵', 'F2 技术是叠上去的'],
  ['P 路径依赖', 'G7 简单规则+迭代大于复杂计划'],
  ['L 约束', 'C 复利'],
  ['M 边际', 'T 权衡'],
  ['S 筛选', 'G 涌现'],
  ['E 耗散', 'F 反馈'],
  ['C1 激励决定行为', 'C6 身份先于观点'],
  ['D8 外部性', 'E4 公地悲剧'],
  ['F1 地图不是疆域', 'F7 测量什么就得到什么'],
  ['A5 相变与临界点', 'G1 大多数事情是随机加一点趋势'],
  ['B5 局部最优陷阱', 'G5 时间是最大的杠杆'],
]

const STYLE = `你在为一个小型中文语言模型写训练语料。母本是一套"世界底层规律"框架：10 个骨架代号 + 55 个条目。这套语料训练的不是记忆，而是四个理解动作：还原机制、找反例并解释、跨领域检验同构、反向预测。你写的每一篇都必须是一次真实的推理，不是复述。

## 骨架表
F 反馈 / E 耗散 / S 筛选 / C 复利 / L 约束 / M 边际 / P 路径依赖 / G 涌现 / T 权衡 / I 信息差

## 条目表（引用代号只能用这些）
A1 能量守恒 / A2 熵增 / A3 最小阻力路径 / A4 尺度效应 / A5 相变与临界点 / A6 惯性
B1 演化=变异+筛选+时间 / B2 活着就是逆着熵走 / B3 适应有代价 / B4 红皇后效应 / B5 局部最优陷阱 / B6 冗余换稳健
C1 激励决定行为 / C2 损失比收益更痛 / C3 省力是本能 / C4 人只看得见自己在找的东西 / C5 习惯是压缩的决定 / C6 身份先于观点 / C7 相对比绝对重要
D1 稀缺是一切价格的来源 / D2 竞争把利润压向零 / D3 分工放大产出 / D4 沉没成本 / D5 边际递减 / D6 杠杆 / D7 网络效应 / D8 外部性
E1 制度是固化的激励 / E2 规模改变性质 / E3 协调成本是隐形税 / E4 公地悲剧 / E5 信念会自我实现 / E6 地位是零和的 / E7 稳健的系统都有冗余和缓冲 / E8 信任降低一切成本
F1 地图不是疆域 / F2 技术是叠上去的 / F3 抽象让复杂可以堆叠 / F4 信息会腐烂 / F5 复制便宜创造昂贵 / F6 工具改变使用者 / F7 测量什么就得到什么
G1 大多数事情是随机加一点趋势 / G2 不可逆的决定要慢可逆的要快 / G3 极端事件决定长期结果 / G4 缓慢累积突然爆发 / G5 时间是最大的杠杆 / G6 机会成本是真正的成本 / G7 简单规则+迭代大于复杂计划

## 硬规则
1. 口语化说人话，但推理必须硬：每一步追问都要具体，不许"可能""也许"糊弄。
2. 因果链用 → 连接。
3. 禁止套话：不要"总之""综上所述""首先其次最后""值得注意的是"。
4. 每篇必须点出骨架代号（F/E/S/C/L/M/P/G/T/I）。
5. 字数按各类型要求。
6. 核心要求：机制还原必须说出失效边界；反例碰撞必须说出主语/层面/尺度差异；同构检验必须给出判同构还是类比的那个细节；反向预测必须说出隐含前提。做不到这四点的篇目是废品。

## 格式范本

[机制还原]
**D2 的机制还原**
链：高利润 → 吸引进入者 → 供给增加 → 降价 → 利润回到平均。
逐步追问：第一步，进入者凭什么进得来？——要资金、技术、牌照可得，这一步不必然，壁垒就在这里。第二步，供给增加为什么一定降价？——要产品同质、买家对价格敏感，差异化产品不降价。第三步，降价为什么不被需求增长抵消？——需求增长慢于供给扩张时才成立，新市场初期需求暴涨可以抵消。
失效边界：进入壁垒高、产品差异化强、需求增长快于供给，三者任一成立，利润就压不到零。持续高利润不是反例，是这条规律的边界条件被满足了。踩的骨架：S、F。

[反例碰撞]
**边际递减 vs 网络效应：矛盾吗？**
表面：D5 说越多越不值钱（第二十口饭反胃），D7 说越多越值钱（第一千万个用户让电话更有用）。
碰撞：主语不同。D5 的主语是单个消费者对同一物品的重复消费——同一个人吃到第二十口，边际效用递减；D7 的主语是系统层面的用户网络——新用户改变的是所有其他用户的价值。一个在个体层面递减，一个在系统层面递增，尺度不同，不矛盾。
什么条件下真的矛盾：当网络增长的收益全部归同一个体时（一个人买第十部电话自己用），D5 赢，D7 不适用。踩的骨架：M、L。

[跨领域同构]
**E 耗散在房间、组织、代码库**
房间：不打扫 → 灰尘杂物累积 → 变乱。维护成本随时间大致线性。
组织：不维护流程 → 沟通失真累积 → 协调成本上升。不维护之后逐渐变乱，但到某个点突然崩（关键人离职）。
代码库：不重构 → 技术债累积 → 修改成本指数上升。不维护之后不是线性变乱，是指数——每次改动都建立在之前的妥协上。
判定：房间和组织是同构（线性累积、逐渐衰败），代码库只是类比——它的衰败是指数的，机制细节不同。类比能帮记忆，不能用来推理：不能用打扫房间的节奏去安排重构。踩的骨架：E、A4。

[反向预测]
**反向预测：公地悲剧**
起点：一个群体共享一份资源，每个个体多拿的收益归自己，代价由所有人分摊。
我的推理（不看结论）：个体多拿的收益是自己的，代价只承担 1/N → 每个个体都有动机多拿 → 所有人都多拿 → 资源消耗速度超过恢复速度 → 资源耗尽。
隐含前提（推的时候必须说）：个体之间没有协商机制；资源没有自我恢复能力或恢复慢于消耗；没有外部管制；个体是短期理性的。
对照：结论一致。但前提里任何一条不成立结论就变——有协商（社区自治）、资源可再生（渔场休渔）、有管制（配额），公地就不一定悲剧。踩的骨架：L、F、I。

[实战三问]
**实战三问：社区团购大战**
事件：2020 年互联网巨头扎堆社区团购，烧钱补贴，一年后死伤大半。
踩的三段：D2 竞争把利润压向零——高利润吸引进入者，供给爆炸；D6 杠杆——烧的是融资，亏损被放大，资金链断就出局；G3 极端事件——一次融资失败就归零，一百次小赢抵不过。
链的对应：高利润（生鲜高频刚需）→ 进入者（巨头加创业公司）→ 供给增加 → 补贴战降价 → 利润为负 → 谁先没钱谁出局。
哪段不成立：D7 网络效应——社区团购没有网络效应，用户对平台无忠诚，补贴停用户走，"越多人用越值钱"在这里不成立，所以赢家通吃没有发生，烧钱换不来护城河。踩的骨架：S、F、L。`

function buildAssignments() {
  const out = []

  // 1. 机制还原：55 条目 + 10 骨架 × 3 角度
  const mechAngles = [
    '逐步追问：把链拆开，每步问为什么必然发生，卡住的地方就是失效边界',
    '失效边界：这条规律在什么条件下不成立，给出具体反例场景',
    '前提检验：链的每一步隐含了什么假设，假设不成立会怎样',
  ]
  for (const e of ENTRIES)
    for (const a of mechAngles)
      out.push({ type: 'mechanism', subject: e, angle: a })
  for (const [code, name] of SKELETONS)
    for (const a of mechAngles)
      out.push({ type: 'mechanism', subject: `${code} ${name}`, angle: a })

  // 2. 反例碰撞：20 对 × 3 角度 + 自选 240
  const conAngles = [
    '表面矛盾在哪：把两条规律的冲突点摆出来',
    '为什么不矛盾：主语、层面、尺度的差异是什么',
    '什么条件下真的矛盾：让冲突成立的具体条件',
  ]
  for (const [p1, p2] of PAIRS)
    for (const a of conAngles)
      out.push({ type: 'contradiction', pair: `${p1} × ${p2}`, angle: a })
  for (let k = 0; k < 240; k++) {
    const e1 = ENTRIES[k % ENTRIES.length]
    const e2 = ENTRIES[(k * 7 + 3) % ENTRIES.length]
    out.push({ type: 'contradiction', pair: `${e1} × ${e2}`, angle: '自选角度：找出它们表面冲突的地方并解决它' })
  }

  // 3. 跨领域同构：10 骨架 × 10 三元组 × 2 角度 + 55 条目 × 2 领域
  const isoAngles = ['检验同构：三个实例是不是同一个机制', '识别类比：哪个实例只是说法像、过程不像']
  for (const [code, name] of SKELETONS)
    for (let k = 0; k < 10; k++) {
      const t = [DOMAINS[k % 50], DOMAINS[(k + 17) % 50], DOMAINS[(k + 34) % 50]]
      for (const a of isoAngles)
        out.push({ type: 'isomorphism', skeleton: `${code} ${name}`, domains: t.join('、'), angle: a })
    }
  for (let k = 0; k < ENTRIES.length; k += 1)
    for (let r = 0; r < 2; r++) {
      const d1 = DOMAINS[(k + r * 13) % 50]
      const d2 = DOMAINS[(k + r * 13 + 25) % 50]
      out.push({ type: 'isomorphism', skeleton: ENTRIES[k], domains: `${d1}、${d2}`, angle: '检验同构：两个实例机制细节是否一致' })
    }

  // 4. 反向预测：55 条目 × 2 + 10 骨架 × 5 + 开放 140
  for (const e of ENTRIES) {
    out.push({ type: 'prediction', subject: e, angle: '给出起点条件，先自己推再对照，说出隐含前提' })
    out.push({ type: 'prediction', subject: e, angle: '换一个场景重设起点，推完对照，错在哪一步' })
  }
  for (const [code, name] of SKELETONS)
    for (let k = 0; k < 5; k++)
      out.push({ type: 'prediction', subject: `${code} ${name}`, angle: `在${DOMAINS[(k * 7 + ENTRIES.length) % 50]}场景里设一个起点，推完对照` })
  for (let k = 0; k < 140; k++) {
    const d = DOMAINS[k % 50]
    out.push({ type: 'prediction', subject: d, angle: `从${d}里选一个具体局面作为起点，不看任何结论自己推，说出隐含前提` })
  }

  // 5. 实战三问：80 现象 × 2 + 开放 140
  for (const p of PHENOMENA) {
    out.push({ type: 'field', subject: p, angle: '踩哪三段 + 链怎么对应 + 哪段其实不成立' })
    out.push({ type: 'field', subject: p, angle: '换个时间尺度重答：短期踩哪三段，长期哪段失效' })
  }
  for (let k = 0; k < 140; k++) {
    const d = DOMAINS[k % 50]
    out.push({ type: 'field', subject: d, angle: `从${d}里编一件今天可能发生的具体事，做三问` })
  }

  return out
}

function writerPrompt(batch, idx) {
  const pad = n => String(n).padStart(3, '0')
  const file = `${ROOT}/data/corpus/batch2_${pad(idx + 1)}.jsonl`
  const lines = batch.map((a, k) => {
    if (a.type === 'mechanism') return `${k + 1}. [机制还原] 对象=${a.subject}；角度=${a.angle}`
    if (a.type === 'contradiction') return `${k + 1}. [反例碰撞] 对=${a.pair}；角度=${a.angle}`
    if (a.type === 'isomorphism') return `${k + 1}. [跨领域同构] 骨架/条目=${a.skeleton}；领域=${a.domains}；角度=${a.angle}`
    if (a.type === 'prediction') return `${k + 1}. [反向预测] 对象=${a.subject}；角度=${a.angle}`
    return `${k + 1}. [实战三问] 对象=${a.subject}；角度=${a.angle}`
  }).join('\n')
  return `${STYLE}

## 你的任务（${batch.length} 篇）
${lines}

## 输出方式
用 Write 工具把 ${batch.length} 篇写到文件：${file}

格式：每行一个 JSON 对象，UTF-8，共 ${batch.length} 行：
{"type":"mechanism|contradiction|isomorphism|prediction|field","title":"短标题","content":"正文全文"}

要求：
- content 是完整正文，含 **标题** 行；换行在 JSON 里写成 \\n。
- title：4-15 字的短标题。
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
    label: `v2-${String(idx + 1).padStart(3, '0')}`,
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
