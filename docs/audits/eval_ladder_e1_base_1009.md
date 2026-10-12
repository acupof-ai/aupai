---
question: 正确的 Engram 基座 ckpt_v42s2.pt 在 base continuation 协议下是否生成稳定？
status: measured
source: runs/ladder/ckpt_v42s2/e1/{items.jsonl,summary.json}（pod），E1 模块 eval/ladder/e1_base_gen.py
---

# E1 问题清单 — ckpt_v42s2.pt（base continuation）

依据 `runs/ladder/ckpt_v42s2/e1/`（items.jsonl + summary.json，pod 2026-10-09 09:18 UTC）：n=16，passed=5，exit_code=3，stopped_rate=0.0。

门（裁定 R8/R9）：`nonempty` `protocol` `no_echo` `lang` `no_repeat`；`stopped` 只记数不当门。门值：{"generation_stable": false, "lang": true, "no_echo": true, "no_repeat": false, "nonempty": true, "protocol": true}

## 失败分布

| fail_class | 题数 |
|---|---:|
| no_repeat | 11 |

## R9 两种读法的敏感度

R9 的两条：最大 8-gram 重复次数 > 2（计数规则）或重复覆盖比例 > 0.5（覆盖规则）。逐题看哪条开火：

| id | max 8-gram 次数 | 覆盖比例 | 最长重复 span | 哪条开火 | 字面 span 读法下 |
|---|---:|---:|---:|---|---|
| zh-cont-01 | 2 | 0.6094 | 0.3438 | +覆盖 | pass |
| zh-cont-02 | 2 | 0.875 | 0.2344 | +覆盖 | pass |
| zh-cont-03 | 2 | 0.8594 | 0.2812 | +覆盖 | pass |
| zh-cont-04 | 6 | 0.5469 | 0.4688 | 计数+覆盖 | FAIL |
| zh-qa-01 | 3 | 1.0 | 0.5938 | 计数+覆盖 | FAIL |
| zh-qa-02 | 3 | 1.0 | 0.6875 | 计数+覆盖 | FAIL |
| zh-qa-03 | 2 | 0.5938 | 0.1875 | +覆盖 | pass |
| zh-qa-04 | 3 | 1.0 | 0.6406 | 计数+覆盖 | FAIL |
| en-cont-01 | 3 | 0.7031 | 0.2188 | 计数+覆盖 | FAIL |
| en-cont-02 | 3 | 0.4062 | 0.1406 | 计数 | FAIL |
| en-qa-01 | 8 | 0.8594 | 0.625 | 计数+覆盖 | FAIL |

- 6 题两条规则都开火；1 题只由计数规则开火；**4 题只由覆盖规则开火**（zh-cont-01/02/03、zh-qa-03，最长重复 span 只有 0.19–0.34）。
- 换成字面「最长重复 span > 0.5」读法，11 题里仍有 7 题 FAIL，E1 的结论（FAIL）不变，只是失败计数从 11 降到 7。

## 逐题证据

| id | lang | stop_reason | n_tokens | 判定 | 输出前 70 字 |
|---|---|---|---:|---|---|
| zh-cont-01 | zh | max_new | 64 | no_repeat | 胃肠道会流向肌肉，肌肉会流向肾脏，肾脏会流向肾脏，肾脏会流向� |
| zh-cont-02 | zh | max_new | 64 | no_repeat | 但是电动牙刷的特点是不能够满足人们的需求，而是能够满足人们的需求，所以电动牙刷的特点是不能� |
| zh-cont-03 | zh | max_new | 64 | no_repeat | 它能够在实际使用中使用，而且能够在实际使用中使用，因为它们的使用寿命长，而且它们的使用寿命长，而且它们 |
| zh-cont-04 | zh | max_new | 64 | no_repeat | 记住语法的意义。 在语法上，我们可以看到，语法是指在语法中，语法中，语法中，语法中，语法中，语法中，语法中， |
| zh-qa-01 | zh | max_new | 64 | no_repeat | 地球到月球的距离大约是多少？ 答：地球到月球的距离大约是多少？ 答：地球到月球的距� |
| zh-qa-02 | zh | max_new | 64 | no_repeat | 人体每天需要喝多少水？ 答：人体每天需要喝多少水？ 答：人体每天需要喝多少水？ 答：人体每� |
| zh-qa-03 | zh | max_new | 64 | no_repeat |   1. 计算长方形的面积：  面积 = 左边长 + 右边长 + 左边长 + 右边长 面积 = 8 * 5 + 8 * 5 + 8 * 5  |
| zh-qa-04 | zh | max_new | 64 | no_repeat | 老年痴呆症有没有遗传倾向。 老年痴呆症有没有遗传倾向。 老年痴呆症有没有遗传 |
| en-cont-01 | en | max_new | 64 | no_repeat |  and you will be able to compete in a KCBS BBQ competition in the same |
| en-cont-02 | en | max_new | 64 | no_repeat |  the drive is already mounted. If it is, you can use the following ste |
| en-cont-03 | en | max_new | 64 | pass |  def denormalize(x):     """Scale the values from the range [0, 1] to  |
| en-cont-04 | en | max_new | 64 | pass |  1. Start by moving all terms involving \(x\) to one side:    \[ 2x -  |
| en-qa-01 | en | max_new | 64 | no_repeat |  Yes, it is. A: I am a mentor. I am a coach. I am a mentor. I am a men |
| en-qa-02 | en | max_new | 64 | pass |  To find the distance traveled, we can use the formula: Distance = Spe |
| en-qa-03 | en | max_new | 64 | pass |  The boiling point of water is the temperature at which water boils. A |
| en-qa-04 | en | max_new | 64 | pass |  A list is a collection of elements that can be added to or removed fr |

## 判读

- 11/16 的续写在 64 token 内退化为复读：题干整句回环（zh-qa-01/02/04）、句子循环（zh-cont-01/02/03、en-cont-01、en-qa-01）、公式复读（zh-qa-03）、短语复读（zh-cont-04）。
- 16/16 的 `stop_reason` 是 `max_new`。base 在散文前缀上没有 EOS 线索，按 R8 只记数，不作门。
- 通过的 5 题（en-cont-03/04、en-qa-02/03/04）是连贯续写；en-qa-04 的覆盖比例 0.44，判据没有误伤正常平行结构。
- 结论：按交接文档 §6「base 协议下生成稳定」这一门，E1 **FAIL**，阶段 B 停在此级。后续是阶段 C（格式与停止数据）与阶段 D（逐级 SFT）。
