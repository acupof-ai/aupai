#!/usr/bin/env python3
"""L2 ext1 programs: 3-4 arithmetic steps; combination, unit conversion, remainder,
average, perimeter. 40 distinct families reusing mathcommon pools.

Contract per family: fn(rng) -> (instruction, lines[list[str]], ans).
- instruction: natural Chinese, >=3 sentence phrasings via rng.randrange(3).
- lines: per-line equations `X op Y = Z` (ops + - x /, unicode X /). Trailing
  Chinese unit allowed after Z. Every line independently verified (LHS eval'd to
  equal RHS number). Last line's numeric == num(ans).
- ans: int or fractions.Fraction. No floats, no '%' in LHS, no `……` notation.
"""
import random
from fractions import Fraction
from mathcommon import (
    ANIMALS, FOOD, FRUITS, GOODS, NAMES, STATIONERY, UNIT_N, UNIT_ZHI,
    num,
)

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L2", name, fn))


# --- unit conversion ----------------------------------------------------------

# a m b cm -> total cm
def cmCombined(rng):
    a = rng.randint(1, 9)
    b = rng.randint(10, 99)
    unit = rng.choice(["彩带", "绳子", "花边", "布料"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}量得一根{unit}长{a}米{b}厘米，合多少厘米？",
        f"一段{unit}长{a}米又{b}厘米，用厘米表示是多少？",
        f"把{a}米{b}厘米的{unit}改写成厘米，一共多少厘米？",
    ][t]
    a_cm = a * 100
    total = a_cm + b
    lines = [
        f"{a} × 100 = {a_cm}厘米",
        f"{a_cm} + {b} = {total}厘米",
    ]
    return ins, lines, total


_reg("cmCombined", cmCombined)


# h hours m min -> total min
def minTotal(rng):
    h = rng.randint(1, 5)
    m = rng.randint(5, 59)
    act = rng.choice(["做实验", "看纪录片", "写作业", "参加活动"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}{act}用了{h}小时{m}分钟，一共多少分钟？",
        f"{act}持续了{h}时{m}分，换算成分钟是多少？",
        f"{n1}花了{h}个小时{m}分钟，总共是几分钟？",
    ][t]
    h_min = h * 60
    total = h_min + m
    lines = [
        f"{h} × 60 = {h_min}分",
        f"{h_min} + {m} = {total}分",
    ]
    return ins, lines, total


_reg("minTotal", minTotal)


# a kg b g -> total g
def gCombined(rng):
    a = rng.randint(1, 8)
    b = rng.choice([100, 200, 300, 400, 500, 600, 700, 800, 900])
    obj = rng.choice(["面粉", "大米", "白糖", "牛肉", "苹果"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}买来的{obj}重{a}千克{b}克，合多少克？",
        f"一袋{obj}重{a}千克又{b}克，用克表示是多少？",
        f"把{a}千克{b}克的{obj}改写成克，一共多少克？",
    ][t]
    a_g = a * 1000
    total = a_g + b
    lines = [
        f"{a} × 1000 = {a_g}克",
        f"{a_g} + {b} = {total}克",
    ]
    return ins, lines, total


_reg("gCombined", gCombined)


# n days -> w weeks + leftover days
def weeksLeftDays(rng):
    k = 7
    full = rng.randint(3, 8)
    left = rng.randint(1, 6)
    n = k * full + left
    obj = rng.choice(["培训班", "夏令营", "复习计划", "观察记录"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"一个{obj}持续{n}天，其中正好有{left}天比整星期还要多，整星期部分占多少天？",
        f"{obj}共{n}天，除去多余的{left}天，其余正好是几个整星期？其中整星期共多少天？",
        f"数一数：{n}天里有{left}天是零头，剩下的整星期一共多少天？",
    ][t]
    lines = [
        f"{full} × {k} = {k * full}天",
        f"{n} - {k * full} = {left}天",
    ]
    return ins, lines, left


_reg("weeksLeftDays", weeksLeftDays)


# x cm -> whole meters + leftover cm
def cmToMPart(rng):
    left = rng.randint(10, 99)
    m = rng.randint(1, 8)
    n = m * 100 + left
    obj = rng.choice(["毛线", "电线", "丝带", "缎带"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"把{n}厘米的{obj}绕成整米，能绕出{m}米，还剩多少厘米？",
        f"{n1}有一卷{n}厘米的{obj}，剪成{m}米最长的整段后，余下多少厘米？",
        f"{n}厘米的{obj}按每{m}米一段裁剪，最后一截不足整米，还有多少厘米？",
    ][t]
    lines = [
        f"{m} × 100 = {m * 100}厘米",
        f"{n} - {m * 100} = {left}厘米",
    ]
    return ins, lines, left


_reg("cmToMPart", cmToMPart)


# x g -> whole kg + leftover g
def gToKgPart(rng):
    left = rng.choice([100, 200, 300, 400, 500, 600, 700, 800, 900])
    kg = rng.randint(1, 6)
    n = kg * 1000 + left
    obj = rng.choice(["绿豆", "花生", "小米", "玉米", "黄豆"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}称得{n}克{obj}，装成整千克能装{kg}千克，还余多少克？",
        f"{n}克{obj}打包为整千克，正好装出{kg}千克，剩下零头多少克？",
        f"把{n}克{obj}分装成每袋整千克，装了{kg}千克后仍有余，余下多少克？",
    ][t]
    lines = [
        f"{kg} × 1000 = {kg * 1000}克",
        f"{n} - {kg * 1000} = {left}克",
    ]
    return ins, lines, left


_reg("gToKgPart", gToKgPart)


# --- remainder ---------------------------------------------------------------

# kids sit rows of k -> full rows, leftover
def rowsLeft(rng):
    k = rng.randint(6, 9)
    full = rng.randint(5, 10)
    left = rng.randint(1, k - 1)
    n = k * full + left
    where = rng.choice(["报告厅", "礼堂", "体育馆", "剧院"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{where}有{n}名学生，每排坐{k}人，坐满{full}排后还剩几人？",
        f"{n}个同学在{where}看演出，每{k}人坐一排，坐下{full}排后还有几人？",
        f"把{n}名学生排成每排{k}人，排好{full}排后余下几人？",
    ][t]
    lines = [
        f"{full} × {k} = {k * full}人",
        f"{n} - {k * full} = {left}人",
    ]
    return ins, lines, left


_reg("rowsLeft", rowsLeft)


# candy into bags of k -> full bags, leftover
def candyBags(rng):
    k = rng.randint(3, 9)
    full = rng.randint(4, 9)
    left = rng.randint(1, k - 1)
    n = k * full + left
    obj = rng.choice(["颗糖", "块饼干", "个巧克力", "颗奶糖"])
    place = rng.choice(["超市", "糖果店", "仓库", "食品店"])
    t = rng.randrange(3)
    ins = [
        f"{place}有{n}{obj}，每{k}个装一袋，装满{full}袋后还剩多少？",
        f"把{n}{obj}每{k}个装一袋，装好{full}袋后余下几个？",
        f"{n}{obj}平均装入每袋{k}个的袋中，装满{full}袋还余多少？",
    ][t]
    lines = [
        f"{full} × {k} = {k * full}{obj}",
        f"{n} - {k * full} = {left}{obj}",
    ]
    return ins, lines, left


_reg("candyBags", candyBags)


# eggs into cartons of k -> full cartons, leftover
def eggCartons(rng):
    k = rng.randint(3, 8)
    full = rng.randint(4, 9)
    left = rng.randint(1, k - 1)
    n = k * full + left
    obj = rng.choice(["鸡蛋", "鸭蛋", "鹌鹑蛋"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}捡了{n}个{obj}，每{k}个装一盒，装满{full}盒后还剩几个？",
        f"共有{n}个{obj}，每盒装{k}个，正好装满{full}盒，剩下的散装还有几个？",
        f"{n}个{obj}按每盒{k}个装，装好{full}盒后盒外还有几个？",
    ][t]
    lines = [
        f"{full} × {k} = {k * full}个",
        f"{n} - {k * full} = {left}个",
    ]
    return ins, lines, left


_reg("eggCartons", eggCartons)


# pencils into boxes of k -> full boxes, leftover
def boxLeftPencil(rng):
    k = rng.randint(6, 12)
    full = rng.randint(4, 8)
    left = rng.randint(1, k - 1)
    n = k * full + left
    obj = rng.choice(["铅笔", "钢笔", "蜡笔", "水彩笔"])
    place = rng.choice(["文具店", "仓库", "学校商店"])
    t = rng.randrange(3)
    ins = [
        f"{place}进了{n}支{obj}，每{k}支装一盒，装满{full}盒后还剩几支？",
        f"把{n}支{obj}每{k}支装成一盒，装好{full}盒后余下多少支？",
        f"{n}支{obj}按每盒{k}支的数量装箱，正好装箱{full}盒，盒子外还剩几支？",
    ][t]
    lines = [
        f"{full} × {k} = {k * full}支",
        f"{n} - {k * full} = {left}支",
    ]
    return ins, lines, left


_reg("boxLeftPencil", boxLeftPencil)


# sticks into bundles of k -> full bundles, leftover
def bundlesLeft(rng):
    k = rng.randint(3, 9)
    full = rng.randint(5, 10)
    left = rng.randint(1, k - 1)
    n = k * full + left
    obj = rng.choice(["根筷子", "根甘蔗", "根小棒", "枝花"])
    place = rng.choice(["菜市场", "果园", "早市", "花店"])
    t = rng.randrange(3)
    ins = [
        f"{place}卖{n}{obj}，每{k}根扎成一把，扎好{full}把后还剩几根？",
        f"{n}{obj}按每把{k}根捆扎，正好捆{full}把，散落在外还有几根？",
        f"扎把：{obj}共{n}根，每{k}根一把，扎满{full}把后余几根？",
    ][t]
    lines = [
        f"{full} × {k} = {k * full}根",
        f"{n} - {k * full} = {left}根",
    ]
    return ins, lines, left


_reg("bundlesLeft", bundlesLeft)


# flowers into vases of k -> full vases, leftover
def vaseLeft(rng):
    k = rng.randint(2, 6)
    full = rng.randint(4, 8)
    left = rng.randint(1, k - 1)
    n = k * full + left
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}摘了{n}枝花，每{k}枝插一瓶，插满{full}瓶后还剩几枝？",
        f"插花：{n}枝花每{k}枝一瓶，正好插满{full}瓶，剩下的还有几枝？",
        f"{n}枝花每{k}枝扎成一把插{full}瓶，剩下的散枝有多少？",
    ][t]
    lines = [
        f"{full} × {k} = {k * full}枝",
        f"{n} - {k * full} = {left}枝",
    ]
    return ins, lines, left


_reg("vaseLeft", vaseLeft)


# --- average -----------------------------------------------------------------

# three scores, drop lowest -> avg of remaining two
def avgDropLow(rng):
    a = rng.randint(60, 90)
    b = a + rng.randint(2, 8)
    c = b + rng.randint(2, 8)
    s = a + b + c
    s2 = s - a
    avg = Fraction(s2, 2)
    subj = rng.choice(["语文", "数学", "英语", "科学"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}三次{subj}测验分别考了{a}、{b}、{c}分，去掉最低的一次，另两次的平均分是多少？",
        f"三次成绩{a}、{b}、{c}分，划去最低分{a}分后，剩下两次的平均分是几分？",
        f"{n1}{subj}三次分数为{a}、{b}、{c}，不计最低那次，余下两次平均为多少分？",
    ][t]
    lines = [
        f"{a} + {b} + {c} = {s}分",
        f"{s} - {a} = {s2}分",
        f"{s2} ÷ 2 = {num(avg)}分",
    ]
    return ins, lines, avg


_reg("avgDropLow", avgDropLow)


# four daily temperatures average
def avgTemp(rng):
    k = 4
    vals = [rng.randint(18, 34) for _ in range(k)]
    s = sum(vals)
    avg = Fraction(s, k)
    city = rng.choice(["上午", "中午", "午后", "傍晚"])
    t = rng.randrange(3)
    ins = [
        f"一天{vals[0]}、{vals[1]}、{vals[2]}、{vals[3]}时的气温分别是{vals[0]}、{vals[1]}、{vals[2]}、{vals[3]}摄氏度，这四次的平均气温是多少？",
        f"某地四点测得的温度依次为{vals[0]}、{vals[1]}、{vals[2]}、{vals[3]}摄氏度，求温度的平均值。",
        f"{city}四个时刻气温为{vals[0]}、{vals[1]}、{vals[2]}、{vals[3]}摄氏度，平均气温是多少？",
    ][t]
    lines = [
        f"四次气温和 = {' + '.join(map(str, vals))} = {s}摄氏度",
        f"{s} ÷ {k} = {num(avg)}摄氏度",
    ]
    return ins, lines, avg


_reg("avgTemp", avgTemp)


# total distance / total time = average speed
def avgSpeedRound(rng):
    v1, t1 = rng.randint(3, 24), rng.randint(2, 4)
    v2, t2 = rng.randint(3, 24), rng.randint(2, 4)
    d1, d2 = v1 * t1, v2 * t2
    d = d1 + d2
    tt = t1 + t2
    avg = Fraction(d, tt)
    act = rng.choice(["跑步", "骑车", "徒步", "滑雪"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}前半程以每小时{v1}千米的速度{act}了{t1}小时，又以每小时{v2}千米{act}了{t2}小时，全程平均速度是多少千米/时？",
        f"{n1}{act}两段：先每小时{v1}千米走{t1}小时，再每小时{v2}千米走{t2}小时，全程平均每小时走多少千米？",
        f"两段路程中，前{t1}小时时速{v1}千米，后{t2}小时时速{v2}千米，总路程除以总时间求平均速度是多少？",
    ][t]
    lines = [
        f"{v1} × {t1} = {d1}千米",
        f"{v2} × {t2} = {d2}千米",
        f"{d1} + {d2} = {d}千米",
        f"{d} ÷ {tt} = {num(avg)}千米/时",
    ]
    return ins, lines, avg


_reg("avgSpeedRound", avgSpeedRound)


# three days pocket money average
def avgMoney(rng):
    k = 3
    vals = [rng.randint(3, 15) for _ in range(k)]
    s = sum(vals)
    avg = Fraction(s, k)
    day = rng.choice(["周一", "周二", "周三"])
    t = rng.randrange(3)
    ins = [
        f"{day}起连续三天零花钱分别是{vals[0]}、{vals[1]}、{vals[2]}元，平均每天多少元？",
        f"三天的零花钱为{vals[0]}、{vals[1]}、{vals[2]}元，求平均每天的花费。",
        f"{day}以来三天每天各得{vals[0]}、{vals[1]}、{vals[2]}元，平均一天几元？",
    ][t]
    lines = [
        f"{' + '.join(map(str, vals))} = {s}元",
        f"{s} ÷ {k} = {num(avg)}元",
    ]
    return ins, lines, avg


_reg("avgMoney", avgMoney)


# total pages over n days = avg per day
def avgPages(rng):
    d1, d2, d3 = (rng.randint(6, 30) for _ in range(3))
    s = d1 + d2 + d3
    avg = Fraction(s, 3)
    obj = rng.choice(["一本书", "一本故事书", "一本练习册"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    title = rng.choice(['小侦探', '环球旅行', '神奇植物', '海洋游记'])
    ins = [
        f"{n1}三天分别读了{d1}、{d2}、{d3}页{obj}，平均每天读多少页？",
        f"《{title}》{d1}、{d2}、{d3}三天各读{d1}、{d2}、{d3}页，平均每天读几页？",
        f"把{d1}、{d2}、{d3}页的总页数平均分到3天，求每天平均读多少页？",
    ][t]
    lines = [
        f"{d1} + {d2} + {d3} = {s}页",
        f"{s} ÷ 3 = {num(avg)}页",
    ]
    return ins, lines, avg


_reg("avgPages", avgPages)


# laps run over days, average then double for a goal
def avgRunsDouble(rng):
    a, b, c = (rng.randint(2, 10) for _ in range(3))
    s = a + b + c
    avg = Fraction(s, 3)
    goal = avg + 2
    obj = rng.choice(["圈", "趟", "个来回"])
    t = rng.randrange(3)
    ins = [
        f"小明三天分别跑了{a}、{b}、{c}{obj}，平均每天跑多少{obj}？若再多跑2{obj}就达标，达标是多少？",
        f"三天跑步{a}、{b}、{c}{obj}求平均，平均再添2{obj}即为目标值，这个目标是多少？",
        f"小明跑了{a}、{b}、{c}{obj}，求这三天的平均，再在平均上多计2{obj}为完成目标，目标是多少？",
    ][t]
    lines = [
        f"{a} + {b} + {c} = {s}{obj}",
        f"{s} ÷ 3 = {num(avg)}{obj}",
        f"{num(avg)} + 2 = {num(goal)}{obj}",
    ]
    return ins, lines, goal


_reg("avgRunsDouble", avgRunsDouble)


# --- perimeter ---------------------------------------------------------------

# square side -> perimeter then fence twice
def squareFence(rng):
    side = rng.randint(5, 25)
    p2 = side * 4
    total = p2 * 2
    place = rng.choice(["正方形花坛", "正方形菜地", "正方形操场一角"])
    him = rng.choice(["篱笆", "围栏", "栏杆"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"一个{place}边长{side}米，沿围一圈需要多少米{him}？若围两圈呢？",
        f"{place}边长为{side}米，{n1}绕着转一圈后又走一圈，共走多少米？",
        f"{place}每条边{side}米，围四周两圈共需{him}多少米？",
    ][t]
    lines = [
        f"{side} × 4 = {p2}米",
        f"{p2} × 2 = {total}米",
    ]
    return ins, lines, total


_reg("squareFence", squareFence)


# rectangle perimeter then per-meter cost
def rectFenceCost(rng):
    l = rng.randint(8, 40)
    w = l - rng.randint(2, 6)
    p2 = 2 * (l + w)
    per = rng.choice([5, 8, 10, 12])
    cost = p2 * per
    place = rng.choice(["花园", "菜园", "操场", "苗圃"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"长方形{place}长{l}米、宽{w}米，周围每米{per}元的篱笆，围一圈要多少元？",
        f"{n1}给长{l}米、宽{w}米的{place}装每米{per}元的围栏，总共花多少元？",
        f"长方形{place}的长是{l}米、宽是{w}米，沿四周钉每米{per}元的栅栏，共需多少钱？",
    ][t]
    lines = [
        f"{l} + {w} = {l + w}米",
        f"{l + w} × 2 = {p2}米",
        f"{p2} × {per} = {cost}元",
    ]
    return ins, lines, cost


_reg("rectFenceCost", rectFenceCost)


# triangle perimeter then ribbons/cost
def triangleFence(rng):
    a, b, c = (rng.randint(4, 20) for _ in range(3))
    s = a + b + c
    per = rng.choice([6, 9, 11])
    cost = s * per
    place = rng.choice(["三角花圃", "三角菜地"])
    t = rng.randrange(3)
    ins = [
        f"一个{place}三条边分别是{a}、{b}、{c}米，四周修每米{per}元的护栏，一共多少元？",
        f"{place}的周长得先算：三边{a}、{b}、{c}米求和后再乘{per}元，护栏总价多少？",
        f"三角地三边各{a}、{b}、{c}米，围一圈用每米{per}元的护栏，要付多少元？",
    ][t]
    lines = [
        f"{a} + {b} = {a + b}米",
        f"{a + b} + {c} = {s}米",
        f"{s} × {per} = {cost}元",
    ]
    return ins, lines, cost


_reg("triangleFence", triangleFence)


# playground perimeter then N laps
def playLaps(rng):
    l = rng.randint(30, 60)
    w = l - rng.randint(10, 20)
    p2 = 2 * (l + w)
    laps = rng.randint(2, 5)
    total = p2 * laps
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"操场长{l}米、宽{w}米，{n1}沿着操场跑了{laps}圈，一共跑多少米？",
        f"长方形操场长{l}米、宽{w}米，绕一圈再跑{laps}圈共跑多少米？",
        f"跑操：操场周长跑{laps}圈，长{l}米、宽{w}米，总路程是多少米？",
    ][t]
    lines = [
        f"{l} + {w} = {l + w}米",
        f"{l + w} × 2 = {p2}米",
        f"{p2} × {laps} = {total}米",
    ]
    return ins, lines, total


_reg("playLaps", playLaps)


# ribbon perimeter then leftover
def frameLeft(rng):
    l = rng.randint(6, 20)
    w = rng.randint(4, l - 2)
    p2 = 2 * (l + w)
    ribbon = p2 + rng.randint(4, 20)
    left = ribbon - p2
    obj = rng.choice(["照片", "画", "奖状", "地图"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"给一张长{l}厘米、宽{w}厘米的{obj}镶一圈花边，买了{ribbon}厘米，还剩下多少厘米？",
        f"{obj}长{l}厘米、宽{w}厘米，沿四边围花边共需周长，买了{ribbon}厘米后余多少？",
        f"用{ribbon}厘米的{obj}点缀，{obj}四周长是长{l}加宽{w}再乘2，剩下多少厘米？",
    ][t]
    lines = [
        f"{l} + {w} = {l + w}厘米",
        f"{l + w} × 2 = {p2}厘米",
        f"{ribbon} - {p2} = {left}厘米",
    ]
    return ins, lines, left


_reg("frameLeft", frameLeft)


# --- two-item purchase -------------------------------------------------------

# two fruits by 斤
def fruitBundle(rng):
    q1, q2 = rng.randint(2, 8), rng.randint(2, 8)
    p1, p2 = rng.randint(3, 12), rng.randint(3, 12)
    o1, o2 = rng.sample(FRUITS, 2)
    n1 = rng.choice(NAMES)
    c1, c2 = q1 * p1, q2 * p2
    s = c1 + c2
    t = rng.randrange(3)
    ins = [
        f"{n1}买了{q1}斤{p1}元一斤的{o1}和{q2}斤{p2}元一斤的{o2}，一共多少元？",
        f"苹果摊位：{q1}斤{o1}、{q2}斤{o2}，单价分别是{p1}元和{p2}元，共付多少？",
        f"{n1}买菜买水果：{o1}每斤{p1}元买{q1}斤，{o2}每斤{p2}元买{q2}斤，共几元？",
    ][t]
    lines = [
        f"{q1} × {p1} = {c1}元",
        f"{q2} × {p2} = {c2}元",
        f"{c1} + {c2} = {s}元",
    ]
    return ins, lines, s


_reg("fruitBundle", fruitBundle)


# adult + child ticket x2 each (family of 4)
def ticketFamily(rng):
    adult = rng.randint(15, 60)
    child = rng.randint(8, adult - 2)
    a2, c2 = adult * 2, child * 2
    s = a2 + c2
    n1 = rng.choice(NAMES)
    place = rng.choice(["动物园", "游乐园", "科技馆", "海洋馆"])
    t = rng.randrange(3)
    ins = [
        f"{n1}一家四口去{place}，成人票{adult}元，儿童票{child}元，两个大人和两个小孩共买4张票，一共要多少元？",
        f"去{place}买票：大人每张{adult}元雇2张，小孩每张{child}元雇2张，总金额是多少？",
        f"{n1}和父母、妹妹共4人去{place}，大人票{adult}元，儿童票{child}元，买4张票共多少元？",
    ][t]
    lines = [
        f"2 + 2 = 4人",
        f"{adult} × 2 = {a2}元",
        f"{child} × 2 = {c2}元",
        f"{a2} + {c2} = {s}元",
    ]
    return ins, lines, s


_reg("ticketFamily", ticketFamily)


# count q sets of a + one single b
def comboMeal(rng):
    q = rng.randint(2, 4)
    combo = rng.randint(15, 30)
    single = rng.randint(5, 12)
    set_total = combo * q
    s = set_total + single
    item = rng.choice(["套餐", "饭盒", "盒饭", "套餐盒"])
    plus = rng.choice(["一杯豆浆", "一个煎蛋", "一份小菜", "一杯酸奶"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}组了{q}份{combo}元的{item}，再加{single}元的{plus}，一共多少元？",
        f"食堂{q}份{item}每份{combo}元，另加{plus}{single}元，共付多少？",
        f"买{q}个{combo}元的{item}和一份{single}元的{plus}，一共多少钱？",
    ][t]
    lines = [
        f"{combo} × {q} = {set_total}元",
        f"{set_total} + {single} = {s}元",
    ]
    return ins, lines, s


_reg("comboMeal", comboMeal)


# buy n of item then pay with bill -> change
def buyChangeBill(rng):
    q = rng.randint(2, 6)
    p = rng.randint(3, 15)
    cost = q * p
    while cost > 50:
        p = rng.randint(3, 12)
        cost = q * p
    bill = rng.choice([20, 50, 100])
    while bill < cost:
        bill += 10 if bill == 20 else 50 if bill == 50 else 100
        if bill > 200:
            bill = 100
    change = bill - cost
    obj = rng.choice(GOODS)
    unit = rng.choice([UNIT_N, UNIT_ZHI])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}买了{q}{unit}{p}元的{obj}，付了一张{bill}元，找回多少元？",
        f"{obj}每{unit}{p}元，{n1}买{q}个共多少元，再付{bill}元应找零多少？",
        f"买{q}{unit}{p}元的{obj}应付{q * p}元，付{bill}元后找回多少？",
    ][t]
    lines = [
        f"{q} × {p} = {cost}元",
        f"{bill} - {cost} = {change}元",
    ]
    return ins, lines, change


_reg("buyChangeBill", buyChangeBill)


# two kinds of stationery
def schoolKit(rng):
    q1, q2 = rng.randint(2, 6), rng.randint(2, 6)
    p1, p2 = rng.randint(2, 10), rng.randint(2, 10)
    o1, o2 = rng.sample(STATIONERY, 2)
    n1 = rng.choice(NAMES)
    c1, c2 = q1 * p1, q2 * p2
    s = c1 + c2
    t = rng.randrange(3)
    ins = [
        f"{n1}买了{q1}支{p1}元的{o1}和{q2}支{p2}元的{o2}，一共花了多少元？",
        f"文具采购：{q1}支{o1}每支{p1}元，{q2}支{o2}每支{p2}元，总价多少？",
        f"{o1}每支{p1}元买{q1}支，{o2}每支{p2}元买{q2}支，共需多少元？",
    ][t]
    lines = [
        f"{q1} × {p1} = {c1}元",
        f"{q2} × {p2} = {c2}元",
        f"{c1} + {c2} = {s}元",
    ]
    return ins, lines, s


_reg("schoolKit", schoolKit)


# buy 3 get 1: pay only 2 items each price
def buyThreeGetOne(rng):
    unit_price = rng.randint(5, 20)
    promo = rng.randint(3, 6)
    n_free = promo // 3
    n_pay = promo - n_free
    total = n_pay * unit_price
    obj = rng.choice(FOOD)
    unit = rng.choice([UNIT_N, UNIT_ZHI])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{obj}每{unit}{unit_price}元，{n1}买{promo}个，其中有{n_free}个满赠免单，只需付{n_pay}个的钱，一共多少元？",
        f"促销满额赠送：{obj}{unit_price}元一个，买满{promo}个赠{n_free}个，实际付{n_pay}个的钱，共多少？",
        f"{obj}{promo}个中免单{n_free}个，只按{n_pay}个每个{unit_price}元结算，应付多少元？",
    ][t]
    lines = [
        f"{promo} - {n_free} = {n_pay}个",
        f"{n_pay} × {unit_price} = {total}元",
    ]
    return ins, lines, total


_reg("buyThreeGetOne", buyThreeGetOne)


# two legs of clothing with quantities
def clothesBuy(rng):
    q1, q2 = rng.randint(1, 3), rng.randint(1, 3)
    p1, p2 = rng.randint(30, 90), rng.randint(25, 70)
    o1, o2 = rng.choice(["衬衫", "帽子"]), rng.choice(["T恤", "袜子"])
    n1 = rng.choice(NAMES)
    c1, c2 = q1 * p1, q2 * p2
    s = c1 + c2
    dur = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{dur}买{q1}件{p1}元的{o1}和{q2}双{p2}元的{o2}，一共多少钱？",
        f"买衣服：{q1}件{o1}每件{p1}元，{q2}双{o2}每双{p2}元，共多少元？",
        f"{o1}单价{p1}元买{q1}件，{o2}单价{p2}元买{q2}双，总共多少元？",
    ][t]
    lines = [
        f"{q1} × {p1} = {c1}元",
        f"{q2} × {p2} = {c2}元",
        f"{c1} + {c2} = {s}元",
    ]
    return ins, lines, s


_reg("clothesBuy", clothesBuy)


# drinks for n people + snacks
def partyDrinks(rng):
    n = rng.randint(4, 8)
    per = rng.randint(3, 6)
    drink_total = n * per
    snack = rng.randint(10, 30)
    s = drink_total + snack
    where = rng.choice(["聚会", "野餐", "生日会"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}的{where}有{n}人，每人喝{per}元的饮料，再买{snack}元零食，一共多少元？",
        f"{where}上{n}人每人{per}元饮料，外加{snack}元小吃，总花费多少？",
        f"{n}人每人{per}元饮品再加{snack}元零嘴，算一算一共多少元？",
    ][t]
    lines = [
        f"{n} × {per} = {drink_total}元",
        f"{drink_total} + {snack} = {s}元",
    ]
    return ins, lines, s


_reg("partyDrinks", partyDrinks)


# --- time --------------------------------------------------------------------

# depart hour + travel hours = arrive; then convert leftover? realistically just arrival
def trainArrive(rng):
    dep = rng.randint(6, 14)
    dur = rng.randint(2, 6)
    arr = dep + dur
    city = rng.choice(["北京", "上海", "广州", "成都", "武汉"])
    town = rng.choice(["西安", "杭州", "南京", "重庆"])
    t = rng.randrange(3)
    ins = [
        f"一列火车{dep}点从{city}出发，开{dur}小时到达{town}，几时到达？",
        f"{city}开往{town}的车{dep}点发车，行驶{dur}小时，上午几时到站？",
        f"火车{dep}点发车，历经{dur}小时抵达{town}，到站时刻是几点？",
    ][t]
    lines = [
        f"{dep} + {dur} = {arr}时",
    ]
    return ins, lines, arr


_reg("trainArrive", trainArrive)


# movie start/end duration
def movieLength(rng):
    start = rng.randint(13, 19)
    dur = rng.randint(2, 3)
    end = start + dur
    t = rng.randrange(3)
    ins = [
        f"一场电影{start}点开演，演了{dur}小时，几点结束？",
        f"电影{start}时开场，时长{dur}小时，散场是几时？",
        f"{start}点放映的影片连放{dur}小时，几点放完？",
    ][t]
    lines = [
        f"{start} + {dur} = {end}点",
    ]
    return ins, lines, end


_reg("movieLength", movieLength)


# study h hours/day x days + extra = total hours
def dailyStudy(rng):
    h = rng.randint(1, 4)
    days = rng.randint(3, 6)
    base = h * days
    extra = rng.randint(1, 3)
    total = base + extra
    subj = rng.choice(["数学", "英语", "语文"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}每天学{subj}{h}小时，学了{days}天，再加{extra}小时，一共几小时？",
        f"每天学{h}小时，连学{days}天后再多学{extra}小时，总计多少小时？",
        f"{subj}学习{h}小时持续{days}日，之后的{extra}小时加在一起共多少小时？",
    ][t]
    lines = [
        f"{h} × {days} = {base}小时",
        f"{base} + {extra} = {total}小时",
    ]
    return ins, lines, total


_reg("dailyStudy", dailyStudy)


# week work hours
def weekWork(rng):
    per_day = rng.randint(6, 9)
    days = rng.randint(4, 6)
    total = per_day * days
    act = rng.choice(["上班", "练习", "锻炼"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}每天{act}{per_day}小时，一周工作{days}天，共多少小时？",
        f"每天{act}{per_day}小时，{days}天加起来一共多少小时？",
        f"{act}时间每天{per_day}小时，按每周{days}天计，每周合计几小时？",
    ][t]
    lines = [
        f"{per_day} × {days} = {total}小时",
    ]
    return ins, lines, total


_reg("weekWork", weekWork)


# bed + sleep hours = wake (no wrap)
def sleepWake(rng):
    bed = rng.randint(20, 22)
    dur = rng.randint(8, 10)
    wake = (bed + dur) % 24
    if wake == 0:
        wake = 24
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}晚上{bed}点睡觉，睡了{dur}小时，第二天几点起床？",
        f"{bed}点就寝，共睡{dur}小时，次日几时清醒？",
        f"夜里{bed}点入睡，经过{dur}小时到第二天早上，几点起床？",
    ][t]
    lines = [
        f"{bed} + {dur} = {bed + dur}时",
        f"{bed + dur} - 24 = {wake}时",
    ]
    return ins, lines, wake


_reg("sleepWake", sleepWake)


# bus minutes per stop x stops (+ wait)
def busStops(rng):
    per = rng.randint(2, 6)
    stops = rng.randint(5, 12)
    ride = per * stops
    wait = rng.randint(3, 8)
    total = ride + wait
    place = rng.choice(["动物园", "火车站", "博物馆"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"公交车每站停{per}分钟，{n1}坐{stops}站，再加上等车{wait}分钟，一共多少分钟？",
        f"乘车共经{stops}站，每站{per}分钟，再加等车{wait}分钟，全程几分钟？",
        f"到{place}要坐{stops}站，平均每站{per}分钟，先等车{wait}分钟，总计几分钟？",
    ][t]
    lines = [
        f"{per} × {stops} = {ride}分钟",
        f"{ride} + {wait} = {total}分钟",
    ]
    return ins, lines, total


_reg("busStops", busStops)


# --- combination (counts/quantities) -----------------------------------------

# packs of per + loose = total
def packPlusLoose(rng):
    packs = rng.randint(3, 8)
    per = rng.randint(8, 15)
    boxed = packs * per
    loose = rng.randint(1, per - 2)
    total = boxed + loose
    obj = rng.choice(["饼干", "糖果", "奶糖", "曲奇"])
    box = rng.choice(["盒", "袋", "罐"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}买来{packs}{box}{obj}，每{box}{per}个，又另外买了{loose}个，一共多少个？",
        f"{packs}{box}{obj}，每{box}装{per}个，再加散装的{loose}个，总共有多少？",
        f"货物分{packs}{box}整装和{loose}个散装，整装每{box}{per}个，共多少个？",
    ][t]
    lines = [
        f"{packs} × {per} = {boxed}个",
        f"{boxed} + {loose} = {total}个",
    ]
    return ins, lines, total


_reg("packPlusLoose", packPlusLoose)


# rows of seats + extra seats
def rowsSeats(rng):
    rows = rng.randint(6, 15)
    each = rng.randint(6, 12)
    seated = rows * each
    extra = rng.randint(2, each - 2)
    total = seated + extra
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"阶梯教室有{rows}排座位，每排{each}个，最前面再加{extra}个，一共多少个座位？",
        f"教室{rows}排、每排{each}座，另添{extra}个加座，总座位数是多少？",
        f"{rows}排座位每排{each}个，再多{extra}个备用座，一共几个座位？",
    ][t]
    lines = [
        f"{rows} × {each} = {seated}个",
        f"{seated} + {extra} = {total}个",
    ]
    return ins, lines, total


_reg("rowsSeats", rowsSeats)


# eggs: hens lay per day x days = total
def eggsDays(rng):
    per_day = rng.randint(3, 8)
    days = rng.randint(4, 9)
    total = per_day * days
    add = rng.randint(1, 5)
    total2 = total + add
    bird = rng.choice(["鸡", "鸭", "鹅"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}家的{bird}每天下{per_day}个蛋，连下{days}天后又下了{add}个，一共多少个？",
        f"{bird}日均产蛋{per_day}个，产了{days}天再多{add}个，合计多少个？",
        f"养{bird}每天收{per_day}个蛋，连续{days}天再加{add}个，共收多少个蛋？",
    ][t]
    lines = [
        f"{per_day} × {days} = {total}个",
        f"{total} + {add} = {total2}个",
    ]
    return ins, lines, total2


_reg("eggsDays", eggsDays)


# stairs: floors x steps per floor + extra
def stairsTotal(rng):
    floors = rng.randint(3, 8)
    per_floor = rng.randint(10, 18)
    steps = floors * per_floor
    extra = rng.randint(1, 6)
    total = steps + extra
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"从地面走到第{floors}层，每层{per_floor}级台阶，最后一层再迈{extra}级到顶，共多少级？",
        f"上到{floors}楼要走{floors}段，每段{per_floor}级再加{extra}级，共多少级台阶？",
        f"楼梯每层{per_floor}级，爬到第{floors}层后再多{extra}级，总级数是多少？",
    ][t]
    lines = [
        f"{floors} × {per_floor} = {steps}级",
        f"{steps} + {extra} = {total}级",
    ]
    return ins, lines, total


_reg("stairsTotal", stairsTotal)


# animals legs total: two kinds
def mixedLegs(rng):
    a_count = rng.randint(2, 8)
    b_count = rng.randint(2, 8)
    pair = rng.choice([("小鸡", "小狗"), ("小鸭", "小马"), ("小猫", "兔子")])
    kind, legs = pair[0], rng.choice([2, 4])
    kind2, legs2 = pair[1], 4 if legs == 2 else 2
    t1, t2 = a_count * legs, b_count * legs2
    s = t1 + t2
    t = rng.randrange(3)
    ins = [
        f"院子里有{a_count}只{kind}和{b_count}只{kind2}，它们一共有多少条腿？",
        f"{a_count}只{kind}每只{legs}条腿，{b_count}只{kind2}每只{legs2}条腿，合计几条腿？",
        f"数腿：{a_count}只{kind}与{b_count}只{kind2}，两种动物共多少条腿？",
    ][t]
    lines = [
        f"{a_count} × {legs} = {t1}条腿",
        f"{b_count} × {legs2} = {t2}条腿",
        f"{t1} + {t2} = {s}条腿",
    ]
    return ins, lines, s


_reg("mixedLegs", mixedLegs)


# --- money / change / balance -------------------------------------------------

# two bills of a + coins, total then spend -> left
def walletBalance(rng):
    a = rng.randint(10, 50)
    coins = rng.randint(5, 20)
    have = a * 2 + coins
    spend = have - rng.randint(5, have - 5)
    left = have - spend
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}钱包里有{a}元和{a}元两张，还有{coins}元零钱，花掉{spend}元后还剩多少元？",
        f"手头有两张{a}元和{coins}元零钱，支出{spend}元后剩余多少元？",
        f"带{a}元两张加{coins}元，去商场花{spend}元，还剩下多少元？",
    ][t]
    lines = [
        f"{a} × 2 = {a * 2}元",
        f"{a * 2} + {coins} = {have}元",
        f"{have} - {spend} = {left}元",
    ]
    return ins, lines, left


_reg("walletBalance", walletBalance)


# pay with two bills, buy two items, change
def payTwoBills(rng):
    q1, q2 = rng.randint(1, 3), rng.randint(1, 3)
    p1, p2 = rng.randint(5, 20), rng.randint(5, 20)
    c1, c2 = q1 * p1, q2 * p2
    s = c1 + c2
    bill = 100
    while s >= 100:
        p1, p2 = rng.randint(3, 15), rng.randint(3, 15)
        c1, c2 = q1 * p1, q2 * p2
        s = c1 + c2
    change = bill - s
    o1, o2 = rng.sample(GOODS, 2)
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}买{q1}个{p1}元的{o1}和{q2}个{p2}元的{o2}，付一张{bill}元，应找回多少？",
        f"两样购物：{q1}个{o1}（{p1}元/个）与{q2}个{o2}（{p2}元/个），付款{bill}元找回多少？",
        f"先算两件商品总价，再求找零：买{q1}个{p1}元的{o1}和{q2}个{p2}元的{o2}，付{bill}元应找回多少元？",
    ][t]
    lines = [
        f"{q1} × {p1} = {c1}元",
        f"{q2} × {p2} = {c2}元",
        f"{c1} + {c2} = {s}元",
        f"{bill} - {s} = {change}元",
    ]
    return ins, lines, change


_reg("payTwoBills", payTwoBills)


# income - two expenses = balance
def budgetRemain(rng):
    total = rng.randint(60, 200) * 5
    e1 = total // rng.randint(5, 8)
    e2 = total // rng.randint(6, 9)
    used = e1 + e2
    rest = total - used
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{total}元，第一笔{e1}元，第二笔{e2}元，还剩多少元？",
        f"预算共{total}元，两项花费{e1}元和{e2}元，结余是多少元？",
        f"手头{total}元先后用去{e1}元、{e2}元，还剩下多少钱？",
    ][t]
    lines = [
        f"{total} - {e1} = {total - e1}元",
        f"{total - e1} - {e2} = {rest}元",
    ]
    return ins, lines, rest


_reg("budgetRemain", budgetRemain)


# weight: two animals combined then compare to a bag
def weightCombine(rng):
    w1, w2 = rng.randint(3, 20), rng.randint(3, 20)
    total = w1 + w2
    bag = total + rng.randint(2, 8)
    a1, a2 = rng.sample(ANIMALS, 2)
    t = rng.randrange(3)
    ins = [
        f"{a1}重{w1}千克，{a2}重{w2}千克，合起来比一袋{bag}千克的米还差多少？",
        f"{a1}体重{w1}千克、{a2}体重{w2}千克，相加之和比{bag}千克的米少几千克？",
        f"把{w1}千克的{a1}和{w2}千克的{a2}加在一起，比{bag}千克少几千克？",
    ][t]
    did = bag - total
    lines = [
        f"{w1} + {w2} = {total}千克",
        f"{bag} - {total} = {did}千克",
    ]
    return ins, lines, did


_reg("weightCombine", weightCombine)


# two children then grown; combined double
def doubleCombined(rng):
    a, b = rng.randint(5, 20), rng.randint(5, 20)
    s = a + b
    doubled = s * 2
    o1, o2 = rng.sample(NAMES, 2)
    unit = rng.choice(["本", "个", "枚", "张"])
    t = rng.randrange(3)
    ins = [
        f"{o1}有{a}{unit}，{o2}有{b}{unit}，两人所有加起来再翻一倍，共有多少？",
        f"先把{o1}的{a}{unit}与{o2}的{b}{unit}相加，所得和再乘2，结果是多少？",
        f"{o1}、{o2}分别有{a}和{b}{unit}，把这些合起来扩大一倍，一共多少？",
    ][t]
    lines = [
        f"{a} + {b} = {s}{unit}",
        f"{s} × 2 = {doubled}{unit}",
    ]
    return ins, lines, doubled


_reg("doubleCombined", doubleCombined)


# quantity left after giving pairs
def givePairs(rng):
    have = rng.randint(10, 40)
    pairs = rng.randint(2, 6)
    gave = pairs * 2
    left = have - gave
    obj = rng.choice(["个", "颗", "块"])
    noun = {"个": "苹果", "颗": "糖", "块": "饼干"}[obj]
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{have}{obj}{noun}，每次给{n2}给2{obj}，共给了{pairs}次，还剩多少？",
        f"{n1}手头{have}{obj}，每次分出2{obj}，给出{pairs}次后还剩几个？",
        f"分享{noun}：原本{have}{obj}，每次拿出2{obj}共{pairs}次，剩余多少？",
    ][t]
    lines = [
        f"给出数量 = {pairs} × 2 = {gave}{obj}",
        f"剩余数量 = {have} - {gave} = {left}{obj}",
    ]
    return ins, lines, left


_reg("givePairs", givePairs)


if __name__ == "__main__":
    from run_math_short import verify
    rng = random.Random(42)
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines}"
            ok += 1
    print(f"L2_ext1 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")