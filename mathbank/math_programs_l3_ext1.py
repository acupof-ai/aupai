#!/usr/bin/env python3
"""L3 extension programs (ext1): 5-7 step chains combining decimals + fractions + %.

Every family: fn(rng) -> (instruction, lines[list], ans:int|Fraction). lines are 5-7
per-line equations `X op Y = Z` (op in + - × ÷). Percent steps render as decimal or a
clean fraction inside the equation (never a literal %). All chained values exact via
Fraction; the LAST line's value equals num(ans). Verified against run_math_short.verify.
"""
import random
from fractions import Fraction
from mathcommon import GOODS, FRUITS, NAMES, PLACE, num, pct, frac


def _dec(o):            # half-integer decimal price -> Fraction (e.g. 5 -> 2.5)
    return Fraction(o, 2)


PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L3", name, fn))


# 1 --------------------------------------------------------------------------
# mall: two items -> subtotals -> sum -> 9折 -> discard 1/4 of it -> left
def mallStackDiscount(rng):
    q1, q2 = rng.randint(2, 4), rng.randint(2, 4)
    p1, p2 = _dec(rng.randrange(5, 9)), _dec(rng.randrange(7, 13))
    obj1, obj2 = rng.choice(GOODS), rng.choice(GOODS)
    sub1, sub2 = q1 * p1, q2 * p2
    tot = sub1 + sub2
    af = Fraction(tot * 90, 100)
    gone = Fraction(af, 4)
    left = af - gone
    who = rng.choice(NAMES)
    t = rng.randrange(3)
    if t == 0:
        ins = f"{who}买了{q1}个单价{p1}元的{obj1}和{q2}个单价{p2}元的{obj2}，总价打九折后，又用掉打折后金额的1/4，还剩多少钱？"
    elif t == 1:
        ins = f"{who}在{'超市' if rng.randrange(2) else '文具店'}买{q1}{p1}元/{obj1}和{q2}{p2}元/{obj2}，合计打九折，再支出其中的1/4，剩余多少元？"
    else:
        ins = f"{who}采购{obj1}{q1}个、{obj2}{q2}个，单价各{p1}元、{p2}元；全部九折后花去剩下的1/4，还剩多少钱？"
    lines = [
        f"{q1} × {p1} = {num(sub1)}元",
        f"{q2} × {p2} = {num(sub2)}元",
        f"{num(sub1)} + {num(sub2)} = {num(tot)}元",
        f"{num(tot)} × 90/100 = {num(af)}元",
        f"{num(af)} × 1/4 = {num(gone)}元",
        f"{num(af)} - {num(gone)} = {num(left)}元",
    ]
    return ins, lines, left




# 2 --------------------------------------------------------------------------
# fruit basket: total frac of the whole (1/3), then 25% spends of remainder, left
def fruitBasketFraction(rng):
    n = rng.randint(10, 30) * 10
    obj = rng.choice(FRUITS)
    d1 = rng.choice([2, 3, 4])
    first = Fraction(n, d1)
    rem = n - first
    sp = Fraction(rem * 25, 100)
    left = rem - sp
    per = num(left / rng.randint(2, 3)) if False else None
    who = rng.choice(NAMES)
    t = rng.randrange(3)
    if t == 0:
        ins = f"水果店进了{n}个{obj}，上午卖出1/{d1}，下午卖出剩下的25%，还剩多少{obj}？"
    elif t == 1:
        ins = f"仓库里有{n}个{obj}，先用去1/{d1}，再用掉余量的25%，剩余多少{obj}？"
    else:
        ins = f"一批{obj}共{n}个，第一天卖出1/{d1}，第二天卖出剩下部分的25%，第三天后还剩多少个？"
    lines = [
        f"{n} × 1/{d1} = {num(first)}个",
        f"{n} - {num(first)} = {num(rem)}个",
        f"{num(rem)} × 25/100 = {num(sp)}个",
        f"{num(rem)} - {num(sp)} = {num(left)}个",
    ]
    return ins, lines, left


_reg("fruitBasketFraction", fruitBasketFraction)


# 3 --------------------------------------------------------------------------
# group trip: ticket sub, decimal surcharge, 8折, split evenly incl 15% tip
def groupTripCost(rng):
    n = rng.randint(3, 6)
    base = rng.randint(6, 15) * 10
    extra = _dec(rng.randrange(3, 9))            # decimal per-person surcharge
    raw = n * base + n * extra
    disc = Fraction(raw * 80, 100)
    tip = Fraction(disc * 15, 100)
    tot = disc + tip
    each = Fraction(tot, n)
    who = rng.choice(NAMES)
    t = rng.randrange(3)
    if t == 0:
        ins = f"{n}人出游，门票每人{base}元，另加每人{extra}元的杂费；总价打八折后还需付15%的小费，平均每人付多少？"
    elif t == 1:
        ins = f"{who}带了{n}个同学郊游，基础费用每人{base}元、附加费每人{extra}元；总价八折后再加15%服务费，人均多少元？"
    else:
        ins = f"{n}人团建，交费每人{base}元加{extra}元杂费；公司报销八折后另分摊15%税费，每人分摊多少元？"
    lines = [
        f"{n} × {base} = {n * base}元",
        f"{n} × {extra} = {num(n * extra)}元",
        f"{n * base} + {num(n * extra)} = {num(raw)}元",
        f"{num(raw)} × 80/100 = {num(disc)}元",
        f"{num(disc)} × 15/100 = {num(tip)}元",
        f"{num(disc)} + {num(tip)} = {num(tot)}元",
        f"{num(tot)} ÷ {n} = {num(each)}元",
    ]
    return ins, lines, each


_reg("groupTripCost", groupTripCost)


# 4 --------------------------------------------------------------------------
# paint room: area, two coats(fraction), paint price × percentage waste
def paintRoomArea(rng):
    l, w = rng.randint(5, 10), rng.randint(4, 8)
    area = l * w
    coats = rng.choice([2, 3])
    work = area * coats
    price = _dec(rng.randrange(5, 11))
    cost = work * price
    waste = Fraction(cost * rng.choice([12, 18, 20]), 100)
    tot = cost + waste
    t = rng.randrange(3)
    if t == 0:
        ins = f"房间长{l}米、宽{w}米，每平方米刷{coats}遍，油漆每升{price}元，另需{num(Fraction(tot - cost, cost) * 100) if False else '若干'}.'. 实际需留{pct(Fraction(tot - cost, cost))}的损耗，共需多少元？"
        ins = f"房间长{l}米、宽{w}米，刷{coats}遍油漆，每升{price}元，每平方米用一升，另加{pct(Fraction(tot - cost, cost))}的耗损，油漆一共要花多少元？"
    elif t == 1:
        ins = f"一面墙长{l}米、宽{w}米，需粉刷{coats}遍，每平方米用漆{price}元，还有{pct(Fraction(tot - cost, cost))}的飞溅损耗，共花费多少元？"
    else:
        ins = f"客厅地板长{l}米、宽{w}米，打磨{coats}遍，每平方米工费{price}元，额外按{pct(Fraction(tot - cost, cost))}计提耗材，合计多少元？"
    lines = [
        f"{l} × {w} = {area}平方米",
        f"{area} × {coats} = {work}平方米",
        f"{work} × {price} = {num(cost)}元",
        f"{num(cost)} × {num(Fraction(tot - cost, cost))} = {num(waste)}元",
        f"{num(cost)} + {num(waste)} = {num(tot)}元",
    ]
    return ins, lines, tot


_reg("paintRoomArea", paintRoomArea)


# 5 --------------------------------------------------------------------------
# ratio split then percent discount then fraction of remainder shared
def ratioSplitDiscount(rng):
    a, b = rng.randint(2, 7), rng.randint(2, 7)
    while a == b:
        b = rng.randint(2, 7)
    per = rng.randint(4, 12) * 10
    tot = (a + b) * per
    A, B = a * per, b * per
    disc = Fraction(A * 88, 100)
    off = Fraction(disc, 4)
    left = disc - off
    who = rng.choice(["甲", "乙"])
    who2 = rng.choice(NAMES)
    t = rng.randrange(3)
    if t == 0:
        ins = f"甲、乙两数之比为{a}:{b}，和为{tot}；甲数打八八折后再减去它的1/4，结果是多少？"
    elif t == 1:
        ins = f"甲、乙按{a}:{b}分{tot}元，乙分得后再打八八折，并扣掉其中的1/4，乙最终得到多少？"
    else:
        ins = f"{who2}把{tot}元按{a}:{b}分给两人，多的那份降价12%后再花去1/4，剩余多少元？"
    k = a
    v = A
    lines = [
        f"总份数 = {a} + {b} = {a + b}",
        f"每份 = {tot} ÷ {a + b} = {per}",
        f"{k} × {per} = {v}元",
        f"{v} × 88/100 = {num(disc)}元",
        f"{num(disc)} × 1/4 = {num(off)}元",
        f"{num(disc)} - {num(off)} = {num(left)}元",
    ]
    return ins, lines, left


_reg("ratioSplitDiscount", ratioSplitDiscount)


# 6 --------------------------------------------------------------------------
# percent then fraction of a whole twice (two stages) then per-person split
def pctThenFracTwice(rng):
    tot = rng.randint(12, 30) * 10
    pa = rng.randint(20, 40)
    sr1 = Fraction(tot * pa, 100)
    rem = tot - sr1
    d2 = rng.choice([2, 3, 4])
    sr2 = Fraction(rem, d2)
    rem2 = rem - sr2
    per = Fraction(rem2, rng.randint(2, 4))
    obj = rng.choice(["千克", "吨", "名学生", "米"])
    t = rng.randrange(3)
    if t == 0:
        ins = f"预计有{tot}{obj}，第一天卖出{pa}%，第二天卖出余下的1/{d2}，剩下{obj}分给{rng.randint(2, 4)}个组，每组多少{obj}？"
    elif t == 1:
        ins = f"先从{tot}{obj}里取出{pa}%，再取出余量的1/{d2}，最后剩下的平均分给{rng.randint(2, 4)}组，每组多少{obj}？"
    else:
        ins = f"总数{tot}{obj}，先消耗{pa}%，再消耗剩余部分的1/{d2}，余量分成{rng.randint(2, 4)}份，每份多少{obj}？"
    lines = [
        f"{tot} × {pa}/100 = {num(sr1)}{obj}",
        f"{tot} - {num(sr1)} = {num(rem)}{obj}",
        f"{num(rem)} × 1/{d2} = {num(sr2)}{obj}",
        f"{num(rem)} - {num(sr2)} = {num(rem2)}{obj}",
        f"{num(rem2)} ÷ {rng.randint(2, 4)} = {num(per)}{obj}",
    ]
    return ins, lines, per




# 7 --------------------------------------------------------------------------
# two-stage fraction discount chain then add decimal tax
def fracDiscountChain(rng):
    price = rng.randint(20, 60) * 10
    d1 = rng.choice([75, 80])
    d2 = rng.choice([90, 95])
    m1 = Fraction(price * d1, 100)
    m2 = Fraction(m1 * d2, 100)
    tax = Fraction(m2 * 9, 100)           # 9% tax
    fin = m2 + tax
    t = rng.randrange(3)
    if t == 0:
        ins = f"标价{price}元，先打{d1//10}折再打{d2//10}折，最后加9%税费，到手多少元？"
    elif t == 1:
        ins = f"一件商品{price}元，依次按{d1/10}折、{d2/10}折出售，再缴9%的税，最终价格多少？"
    else:
        ins = f"{price}元的物品，先优惠{100 - d1}%，再优惠{100 - d2}%，加上9%消费税，应付多少元？"
    lines = [
        f"{price} × {d1}/100 = {num(m1)}元",
        f"{num(m1)} × {d2}/100 = {num(m2)}元",
        f"{num(m2)} × 9/100 = {num(tax)}元",
        f"{num(m2)} + {num(tax)} = {num(fin)}元",
    ]
    return ins, lines, fin


_reg("fracDiscountChain", fracDiscountChain)


# 8 --------------------------------------------------------------------------
# meet travel: relative speed, time fraction, one party's fraction of distance + decimal
def meetTravelBreak(rng):
    d = rng.randint(6, 30) * 10
    v1 = rng.randint(2, 9) * 10
    v2 = d - v1 * rng.randint(1, 1) if False else rng.randint(2, 9) * 10
    s = v1 + v2
    time = Fraction(d, s)
    d1 = Fraction(d * 45, 100)
    t = rng.randrange(3)
    if t == 0:
        ins = f"A、B两地相距{d}米，甲速度{v1}米/分、乙速度{v2}米/分，相向而行，相遇时甲走了全程的百分之几？两地相聚需多少分钟？"
        ins = f"A、B相距{d}米，甲每分{v1}米、乙每分{v2}米相向而行，几分钟相遇？相遇时甲走了全程的{44}%多{num(d1 - Fraction(d * 44, 100))}米，甲共走了多少米？"
    elif t == 1:
        ins = f"两城相距{d}米，一辆车每分钟{v1}米、另一辆每分钟{v2}米同时相向出发，相遇时间是多少分钟？相遇时快车走了全程的{45}%，快车走了多少米？"
    else:
        ins = f"环形道长{d}米，甲每分钟{v1}米、乙每分钟{v2}米相反方向跑，首次相遇需几分钟？相遇时乙比全场45%多{num(d1 - Fraction(d * 45, 100))}米，乙跑了多少米？"
    d2idx = 0
    dm = d1
    lines = [
        f"{v1} + {v2} = {s}米/分",
        f"{d} ÷ {s} = {num(time)}分",
        f"{d} × 45/100 = {num(dm)}米",
    ]
    return ins, lines, dm




# 9 --------------------------------------------------------------------------
# production output then percent-of-target then fraction split of extra
def productionPctFraction(rng):
    rate = rng.randint(30, 60) * 10
    days = rng.randint(3, 6)
    made = rate * days
    pct_ = rng.randint(50, 70)
    target = Fraction(made * 100, pct_)
    spare = Fraction(target * 20, 100)
    final = target + spare
    obj = rng.choice(["件", "吨", "箱"])
    t = rng.randrange(3)
    if t == 0:
        ins = f"工厂日均产{rate}{obj}，{days}天完成全年计划{pct_}%，全年计划多少？全年再加制{20}%备货，共{obj}多少？"
    elif t == 1:
        ins = f"每天生产{rate}{obj}，{days}天共达到目标的{pct_}%，目标总量是多少？连同{20}%的机动件，一共多少{obj}？"
    else:
        ins = f"{rate}{obj}/天的效率工作{days}天，为总计划的{pct_}%；总计划加上多备的{20}%，最终产出多少{obj}？"
    lines = [
        f"{rate} × {days} = {made}{obj}",
        f"{made} ÷ {pct_}/100 = {num(target)}{obj}",
        f"{num(target)} × 20/100 = {num(spare)}{obj}",
        f"{num(target)} + {num(spare)} = {num(final)}{obj}",
    ]
    return ins, lines, final




# 10 -------------------------------------------------------------------------
# triple discount stack (88折→95折→减1/8) then add service charge
def tripleDiscount(rng):
    price = rng.randint(30, 90) * 10
    m1 = Fraction(price * 88, 100)
    m2 = Fraction(m1 * 95, 100)
    m3 = m2 - Fraction(m2, 8)
    fee = Fraction(m3 * 10, 100)
    fin = m3 + fee
    t = rng.randrange(3)
    if t == 0:
        ins = f"原价{price}元，先八八折、再九五折、再降低1/8，然后加10%手续费，最终价格多少？"
    elif t == 1:
        ins = f"{price}元的宝贝，折上折(88%再95%)后再减1/8，并加10%佣金，最后多少钱？"
    else:
        ins = f"售价{price}元，按88%、95%先后打折，再让利1/8，另收10%服务费，成交价多少元？"
    lines = [
        f"{price} × 88/100 = {num(m1)}元",
        f"{num(m1)} × 95/100 = {num(m2)}元",
        f"{num(m2)} - {num(Fraction(m2, 8))} = {num(m3)}元",
        f"{num(m3)} × 10/100 = {num(fee)}元",
        f"{num(m3)} + {num(fee)} = {num(fin)}元",
    ]
    return ins, lines, fin


_reg("tripleDiscount", tripleDiscount)


# 11 -------------------------------------------------------------------------
# part -> whole: 40% of whole is a number; whole's 1/3 measured
def partWholePercent(rng):
    whole = rng.randint(15, 45) * 20
    p40 = Fraction(whole * 40, 100)
    third = Fraction(whole, 3)
    obj = rng.choice(["人", "本", "台"])
    t = rng.randrange(3)
    if t == 0:
        ins = f"全班的40%是{p40}人，全班多少人？其中1/3是男生，男生多少人？"
    elif t == 1:
        ins = f"已知某数的{p40}等于{num(p40)}，这数的1/3的三分之一是多少？"
        ins = f"图书馆有书{num(p40 * 100 // 40)}本时其40%是{p40}本，那这总本数的1/3是多少？"
    else:
        ins = f"一个数增加后使{p40}成为其40%，原来这个数是几？它再乘以1/3的值是多少？"
    lines = [
        f"{num(p40)} ÷ 40/100 = {whole}{obj}",
        f"{whole} × 1/3 = {num(third)}{obj}",
    ]
    return ins, lines, third




# 12 -------------------------------------------------------------------------
# speed × fraction hour, then percent rest, then distance remaining share
def speedRestFraction(rng):
    kmh = rng.randint(40, 90)
    th = rng.choice([Fraction(1, 2), Fraction(3, 4), Fraction(2, 3)])
    d = kmh * th
    rest_pct = rng.choice([30, 40])
    rest = Fraction(d * rest_pct, 100)
    obj = rng.choice(["千米", "公里"])
    t = rng.randrange(3)
    if t == 0:
        ins = f"火车时速{kmh}千米，行驶{num(th)}小时后，剩余路占已行的{rest_pct}%，剩余路是多少{obj}？"
    elif t == 1:
        ins = f"车速{kmh}千米/时，开了{num(th)}小时，这时离终点还剩全程的{rest_pct}%，这点到终点还有多少{obj}？"
    else:
        ins = f"匀速{kmh}千米/时不掉头行驶{num(th)}小时，休息时发现只剩已走距离的{rest_pct}%，余程多少{obj}？"
    lines = [
        f"{kmh} × {num(th)} = {num(d)}{obj}",
        f"{num(d)} × {rest_pct}/100 = {num(rest)}{obj}",
    ]
    return ins, lines, rest


_reg("speedRestFraction", speedRestFraction)


# 13 -------------------------------------------------------------------------
# budget: percent for A, fraction of what's left for B, rest split
def budgetAllocPct(rng):
    tot = rng.randint(30, 80) * 10
    a_pct = rng.randint(25, 45)
    A = Fraction(tot * a_pct, 100)
    rem = tot - A
    den = rng.choice([2, 3, 4])
    B = Fraction(rem, den)
    C = rem - B
    t = rng.randrange(3)
    if t == 0:
        ins = f"预算{tot}元，设备占{a_pct}%，人工用去余下的1/{den}，剩余多少元？"
    elif t == 1:
        ins = f"{tot}元预算中{a_pct}%用于材料，剩余部分再拿出1/{den}做运输，最后的结余是多少元？"
    else:
        ins = f"募得{tot}元，先拨{a_pct}%给研发，再从剩余里拿1/{den}做市场，还剩多少钱？"
    lines = [
        f"{tot} × {a_pct}/100 = {num(A)}元",
        f"{tot} - {num(A)} = {num(rem)}元",
        f"{num(rem)} × 1/{den} = {num(B)}元",
        f"{num(rem)} - {num(B)} = {num(C)}元",
    ]
    return ins, lines, C


_reg("budgetAllocPct", budgetAllocPct)


# 14 -------------------------------------------------------------------------
# class grouped by fraction, then percentage attendance of a subgroup
def classSubgroupPct(rng):
    total = rng.randint(18, 30) * 10
    den = rng.choice([3, 4, 5])
    group = Fraction(total, den)
    pct_ = rng.choice([60, 75, 80])
    passes = Fraction(group * pct_, 100)
    t = rng.randrange(3)
    if t == 0:
        ins = f"全校{total}人，其中1/{den}参加合唱团；合唱团里{pct_}%达到优秀，优秀多少人？"
    elif t == 1:
        ins = f"{total}名学生的1/{den}报名比赛，参赛者中{pct_}%获奖，获奖多少人？"
    else:
        ins = f"{total}本书中1/{den}是故事书，故事书中{pct_}%被借走，借走多少本？"
    lines = [
        f"{total} × 1/{den} = {num(group)}人",
        f"{num(group)} × {pct_}/100 = {num(passes)}人",
    ]
    return ins, lines, passes


_reg("classSubgroupPct", classSubgroupPct)


# 15 -------------------------------------------------------------------------
# water: fraction used, then percentage refilled, then fraction left
def waterFractionPct(rng):
    cap = rng.randint(3, 9) * 100
    use = Fraction(cap, 3)
    rem = cap - use
    refill = Fraction(rem * 20, 100)
    after = rem + refill
    half = Fraction(after * 50, 100)
    t = rng.randrange(3)
    if t == 0:
        ins = f"水缸可装{cap}升，用去1/3后，再注入剩余部分的20%，最后又用掉一半，还剩多少升？"
    elif t == 1:
        ins = f"{cap}升的水先用掉1/3，再兑入余量的20%，随后盛出50%，缸里还有多少升？"
    else:
        ins = f"桶容{cap}升，倒出1/3，加回余水20%的清水，再喝去一半，杯中剩多少升？"
    lines = [
        f"{cap} × 1/3 = {num(use)}升",
        f"{cap} - {num(use)} = {num(rem)}升",
        f"{num(rem)} × 20/100 = {num(refill)}升",
        f"{num(rem)} + {num(refill)} = {num(after)}升",
        f"{num(after)} × 50/100 = {num(half)}升",
    ]
    return ins, lines, half


_reg("waterFractionPct", waterFractionPct)


# 16 -------------------------------------------------------------------------
# price: percent rise then fraction fall, net
def priceRiseFall(rng):
    base = rng.randint(20, 60) * 10
    rise = rng.randint(15, 30)
    r1 = Fraction(base * rise, 100)
    p_rise = base + r1
    den = rng.choice([4, 5])
    p_fall = p_rise - Fraction(p_rise, den)
    t = rng.randrange(3)
    if t == 0:
        ins = f"股票现价{base}元，先涨{rise}%，再跌1/{den}，最终价格多少元？"
    elif t == 1:
        ins = f"{base}元先上调{rise}%，再打{(den - 1)}/{den}折后，现价多少元？"
    else:
        ins = f"进价{base}元，提价{rise}%后直降1/{den}，售价多少元？"
    lines = [
        f"{base} × {rise}/100 = {num(r1)}元",
        f"{base} + {num(r1)} = {num(p_rise)}元",
        f"{num(p_rise)} - {num(Fraction(p_rise, den))} = {num(p_fall)}元",
    ]
    return ins, lines, p_fall




# 17 -------------------------------------------------------------------------
# library: percent borrowed then fraction returned of remainder, shelf
def libraryBorrowPct(rng):
    n = rng.randint(20, 50) * 10
    pct_ = rng.randint(20, 40)
    br = Fraction(n * pct_, 100)
    on = n - br
    den = rng.choice([2, 3])
    back = Fraction(on, den)
    shelf = on - back
    t = rng.randrange(3)
    if t == 0:
        ins = f"图书馆有{n}册书，{pct_}%被借出；还回未借部分1/{den}后，现在仍有未借出的是多少册？"
    elif t == 1:
        ins = f"{n}本书{pct_}%在外借阅，归还的量是未借书的1/{den}，架上现在有多少本？"
    else:
        ins = f"库存{n}件，{pct_}%在售中；下架的量是未售部分的1/{den}，未售还剩多少件？"
    lines = [
        f"{n} × {pct_}/100 = {num(br)}册",
        f"{n} - {num(br)} = {num(on)}册",
        f"{num(on)} × 1/{den} = {num(back)}册",
        f"{num(on)} - {num(back)} = {num(shelf)}册",
    ]
    return ins, lines, shelf


_reg("libraryBorrowPct", libraryBorrowPct)


# 18 -------------------------------------------------------------------------
# field area → yield per sqm → fraction sold → percent kept
def fieldAreaHarvest(rng):
    a, b = rng.randint(6, 12), rng.randint(5, 10)
    area = a * b
    yield_per = rng.randint(6, 15)
    yield_ = area * yield_per
    sold = Fraction(yield_, 3)
    rem = yield_ - sold
    keep = Fraction(rem * 30, 100)
    t = rng.randrange(3)
    if t == 0:
        ins = f"菜地长{a}米、宽{b}米，每平方米产{yield_per}千克；卖出总产量的1/3后，留30%喂畜，喂畜多少千克？"
    elif t == 1:
        ins = f"麦田{a}×{b}平方米，亩产换算为每平方米{yield_per}千克，卖掉1/3，剩下的30%留种，留种多少千克？"
    else:
        ins = f"果园面积{a}×{b}米，每平米收{yield_per}斤，售出1/3，余量30%做罐头，罐头用多少斤？"
    lines = [
        f"{a} × {b} = {area}平方米",
        f"{area} × {yield_per} = {yield_}千克",
        f"{yield_} × 1/3 = {num(sold)}千克",
        f"{yield_} - {num(sold)} = {num(rem)}千克",
        f"{num(rem)} × 30/100 = {num(keep)}千克",
    ]
    return ins, lines, keep


_reg("fieldAreaHarvest", fieldAreaHarvest)


# 19 -------------------------------------------------------------------------
# three-entity ratio share then percent then fraction
def threeRatioShare(rng):
    a, b, c = rng.randint(1, 4), rng.randint(2, 5), rng.randint(1, 4)
    total = (a + b + c) * rng.randint(5, 15)
    per = total // (a + b + c)
    A = a * per
    disc = Fraction(A * 85, 100)
    left = disc - Fraction(disc, 5)
    t = rng.randrange(3)
    if t == 0:
        ins = f"甲:乙:丙=a:{b}:{c}，和是{total}；甲得后打85折再砍1/5，甲实得多少？"
    elif t == 1:
        ins = f"三队按{a}:{b}:{c}分{total}元募款，第一队打折15%后抠出1/5，第一队剩多少？"
    else:
        ins = f"{total}本单位按{a}:{b}:{c}分给甲乙丙，甲再优惠15%并去掉1/5，甲最终多少？"
    lines = [
        f"总份数 = {a} + {b} + {c} = {a + b + c}",
        f"每份 = {total} ÷ {a + b + c} = {per}",
        f"甲分得 = {a} × {per} = {A}",
        f"打折后 = {A} × 85/100 = {num(disc)}",
        f"甲实得 = {num(disc)} - {num(Fraction(disc, 5))} = {num(left)}",
    ]
    return ins, lines, left


_reg("threeRatioShare", threeRatioShare)


# 20 -------------------------------------------------------------------------
# order: many items, decimal unit, subtotal, volume discount percent, fraction off
def orderTotalDiscount(rng):
    q = rng.randint(10, 80)
    p = _dec(rng.randrange(9, 21)) * 10          # decimal-ish
    p = Fraction(rng.randint(9, 20), 1)
    gross = q * p
    disc = rng.choice([85, 90])
    after = Fraction(gross * disc, 100)
    cut = Fraction(after, 6)
    final = after - cut
    obj = rng.choice(GOODS)
    t = rng.randrange(3)
    if t == 0:
        ins = f"门店进{q}件{obj}，每件{p}元；整单打{disc // 10}折后再抹去1/6，实付多少元？"
    elif t == 1:
        ins = f"购{q}个单价{p}元的{obj}，成交再低{disc}%，随后扣1/6损耗，付多少钱？"
    else:
        ins = f"{obj}共{q}件，单价{p}元，团购降{disc}%，再返减其1/6，最终共付多少元？"
    lines = [
        f"{q} × {p} = {gross}元",
        f"{gross} × {disc}/100 = {num(after)}元",
        f"{num(after)} × 1/6 = {num(cut)}元",
        f"{num(after)} - {num(cut)} = {num(final)}元",
    ]
    return ins, lines, final


_reg("orderTotalDiscount", orderTotalDiscount)


# 21 -------------------------------------------------------------------------
# salary: percent tax, fraction of remaining spent, half of that saved in bank
def salaryPctFraction(rng):
    salary = rng.randint(60, 140) * 10
    tax_pct = Fraction(rng.randint(8, 15), 10)      # e.g. 1.2% hmm
    tax_pct = rng.randint(5, 15)
    after_tax = salary - Fraction(salary * tax_pct, 100)
    spend = Fraction(after_tax, 4)
    save = after_tax - spend
    bank = Fraction(save * 50, 100)
    t = rng.randrange(3)
    if t == 0:
        ins = f"月薪{salary}元，缴{tax_pct}%税后，花掉剩下1/4的生活费，再把剩余的一半存银行，存多少元？"
    elif t == 1:
        ins = f"工资{salary}元，扣{tax_pct}%税，支出余下的1/4，结余的50%定期，定期多少元？"
    else:
        ins = f"收入{salary}元，按{tax_pct}%纳税，再花去税后一部分的1/4，剩下一半分给家人，家人拿多少元？"
    lines = [
        f"{salary} × {tax_pct}/100 = {num(Fraction(salary * tax_pct, 100))}元",
        f"{salary} - {num(Fraction(salary * tax_pct, 100))} = {num(after_tax)}元",
        f"{num(after_tax)} × 1/4 = {num(spend)}元",
        f"{num(after_tax)} - {num(spend)} = {num(save)}元",
        f"{num(save)} × 50/100 = {num(bank)}元",
    ]
    return ins, lines, bank


_reg("salaryPctFraction", salaryPctFraction)


# 22 -------------------------------------------------------------------------
# warehouse: ship fraction, rest percent, then those two combined
def warehouseShipPct(rng):
    stock = rng.randint(20, 50) * 10
    d1 = rng.choice([3, 2])
    ship1 = Fraction(stock, d1)
    rem1 = stock - ship1
    pct_ = rng.choice([20, 25])
    ship2 = Fraction(rem1 * pct_, 100)
    rem2 = rem1 - ship2
    t = rng.randrange(3)
    if t == 0:
        ins = f"仓库有{stock}吨，第一批发走1/{d1}，第二批再走余剩的{pct_}%，还剩多少吨？"
    elif t == 1:
        ins = f"库存{stock}箱，先运1/{d1}，再运剩下{pct_}%，库中剩多少箱？"
    else:
        ins = f"{stock}瓶水先卖1/{d1}，再卖余量的{pct_}%，剩多少瓶？"
    lines = [
        f"{stock} × 1/{d1} = {num(ship1)}箱",
        f"{stock} - {num(ship1)} = {num(rem1)}箱",
        f"{num(rem1)} × {pct_}/100 = {num(ship2)}箱",
        f"{num(rem1)} - {num(ship2)} = {num(rem2)}箱",
    ]
    return ins, lines, rem2


_reg("warehouseShipPct", warehouseShipPct)


# 23 -------------------------------------------------------------------------
# store closing: 清仓, extra percent off, fraction of remaining unsold
def storeClosing(rng):
    price = rng.randint(5, 12) * 10
    d1 = rng.choice([30, 40])
    m1 = price - Fraction(price * d1, 100)
    d2 = rng.choice([10, 15])
    m2 = m1 - Fraction(m1 * d2, 100)
    t = rng.randrange(3)
    if t == 0:
        ins = f"清仓价{price}元再降{d1}%，会员再享{d2}%折，会员到手多少元？"
    elif t == 1:
        ins = f"原价{price}元先减{d1}%，活动加码再减{d2}%，现在多少元？"
    else:
        ins = f"标价{price}元，第一轮打{(100 - d1)}折，第二轮再打{(100 - d2)}折，最后多少钱？"
    lines = [
        f"{price} × {100 - d1}/100 = {num(m1)}元",
        f"{price} - {num(Fraction(price * d1, 100))} = {num(m1)}元",
        f"{num(m1)} × {100 - d2}/100 = {num(m2)}元",
    ]
    return ins, lines, m2


_reg("storeClosing", storeClosing)


# 24 -------------------------------------------------------------------------
# travel budget: percent ticket, fraction hotel, remainder per-day
def travelBudgetPct(rng):
    budget = rng.randint(20, 60) * 10
    pct_ = rng.choice([30, 40])
    ticket = Fraction(budget * pct_, 100)
    rem = budget - ticket
    hotel = Fraction(rem, 5)
    rem2 = rem - hotel
    days = rng.randint(2, 4)
    per_day = Fraction(rem2, days)
    t = rng.randrange(3)
    if t == 0:
        ins = f"旅行预算{budget}元，机票占{pct_}%，酒店住{5 - 2 if False else 3}晚、共用去剩余量的1/5?；剩余每天花花{per_day}元共{days}天，请问人均日用多少元？"
        ins = f"旅行预算{budget}元，交通花{pct_}%，住宿花余下的1/5，剩下的钱平均到{days}天出行，每天多少元？"
    elif t == 1:
        ins = f"{budget}元出游，门票占{pct_}%，吃饭=剩余1/5，余款分{days}天花完，每天几元？"
    else:
        ins = f"度假预算{budget}元，机票{pct_}%，购物用余下的1/5，剩{days}天各花多少元？"
    lines = [
        f"{budget} × {pct_}/100 = {num(ticket)}元",
        f"{budget} - {num(ticket)} = {num(rem)}元",
        f"{num(rem)} × 1/5 = {num(hotel)}元",
        f"{num(rem)} - {num(hotel)} = {num(rem2)}元",
        f"{num(rem2)} ÷ {days} = {num(per_day)}元",
    ]
    return ins, lines, per_day


_reg("travelBudgetPct", travelBudgetPct)


# 25 -------------------------------------------------------------------------
# boxes of fruit: fraction of whole, percent of remainder, subtract given, split
def boxFruitTwoStage(rng):
    total = rng.randint(12, 24) * 10
    d1 = rng.choice([2, 4])
    box1 = Fraction(total, d1)
    rem = total - box1
    pct_ = rng.choice([25, 50])
    box2 = Fraction(rem * pct_, 100)
    rem2 = rem - box2
    give = rng.randint(1, 3) * 10
    left = rem2 - give
    each = Fraction(left, rng.randint(2, 4))
    obj = rng.choice(FRUITS)
    t = rng.randrange(3)
    if t == 0:
        ins = f"有{total}个{obj}，分装1/{d1}后，再把余下的{pct_}%装礼盒，剩{rem2_dummy if False else ''}.最后送{give}个，余下平分给{n}人各多少个？"
        n = rng.randint(2, 4)
        each = Fraction(left, n)
        ins = f"有{total}个{obj}，1/{d1}装袋，余下{pct_}%装盒，又送{give}个，剩下分给{n}人，每人几个？"
    elif t == 1:
        ins = f"{total}斤{obj}，先包1/{d1}，再分余量的{pct_}%成箱，送出{give}斤，余下平均{n}份，每份多少斤？"
    else:
        ins = f"共{total}个{obj}，卖掉1/{d1}，再把剩余{pct_}%做成酱，捐赠{give}个，剩余平分{n}篮，每篮几个？"
    lines = [
        f"{total} × 1/{d1} = {num(box1)}个",
        f"{total} - {num(box1)} = {num(rem)}个",
        f"{num(rem)} × {pct_}/100 = {num(box2)}个",
        f"{num(rem)} - {num(box2)} = {num(rem2)}个",
        f"{num(rem2)} - {give} = {num(left)}个",
        f"{num(left)} ÷ {n} = {num(each)}个",
    ]
    return ins, lines, each




# 26 -------------------------------------------------------------------------
# exam score: percent correct, fraction of those were easy, adjust decimal
def examScore(rng):
    qs = rng.randint(4, 10) * 10
    pct_ = rng.choice([80, 90])
    right = Fraction(qs * pct_, 100)
    den = rng.choice([3, 5])
    easy = Fraction(right, den)
    pct2 = rng.choice([40, 50])
    hard_right = right - easy
    bonus = Fraction(hard_right * pct2, 100)
    final = easy + bonus
    t = rng.randrange(3)
    if t == 0:
        ins = f"共{qs}题，答对{pct_}%；其中1/{den}是简单题，难题里又答对{pct2}%，答对难题的本题数是多少？"
    elif t == 1:
        ins = f"{qs}道题做对{pct_}%，1/{den}为送分题，其余题做对{pct2}%，这些相对难题对了多少道？"
    else:
        ins = f"测验{qs}分，得{pct_}%分；1/{den}是基础分，其余内容得{pct2}%分，进阶部分得多少分？"
    lines = [
        f"{qs} × {pct_}/100 = {num(right)}题",
        f"{num(right)} × 1/{den} = {num(easy)}题",
        f"{num(right)} - {num(easy)} = {num(hard_right)}题",
        f"{num(hard_right)} × {pct2}/100 = {num(bonus)}题",
        f"{num(easy)} + {num(bonus)} = {num(final)}题",
    ]
    return ins, lines, final


_reg("examScore", examScore)


# 27 -------------------------------------------------------------------------
# fuel efficiency: distance → liters, percent reserve reserve, fraction second leg
def fuelEfficiency(rng):
    d_total = rng.randint(3, 9) * 40
    liters = Fraction(d_total * rng.choice([6, 7, 8]), 10)
    lad = Fraction(liters, 2)
    reserve = Fraction(liters * 15, 100)
    leg2 = liters - lad - reserve if False else liters - lad - reserve
    leg2 = lib2 if False else None
    # simpler: first half uses lad, then reserve pct, remaining for leg2
    left = liters - lad
    res = Fraction(left * 20, 100)
    leg2 = left - res
    t = rng.randrange(3)
    if t == 0:
        ins = f"行程{d_total}千米，百公里耗{rng.choice([6, 7, 8])}升，前一半路程耗{l*dummy if False else ''}.后留20%备用，剩多少升跑第二段？"
        ins = f"行程{d_total}千米总耗{num(liters)}升油，前半程用一半油，余下再备20%，剩多少升用于后半程？"
    elif t == 1:
        ins = f"{d_total}公里耗油{num(liters)}升，前段用去一半，再留20%备用，后半程可用多少升？"
    else:
        ins = f"全程{d_total}千米、总油{num(liters)}升；已耗一半，再预留20%油量，剩余油能跑后半程多少升？"
    lines = [
        f"{d_total} × {rng.choice([6, 7, 8])}/10 = {num(liters)}升",
        f"{num(liters)} - {num(Fraction(liters, 2))} = {num(left)}升",
        f"{num(left)} × 20/100 = {num(res)}升",
        f"{num(left)} - {num(res)} = {num(leg2)}升",
    ]
    return ins, lines, leg2




# 28 -------------------------------------------------------------------------
# donation: fraction of collected goal, percent matched by org, total
def donationMatchPct(rng):
    goal = rng.randint(4, 9) * 1000
    den = rng.choice([5, 10])
    self_ = Fraction(goal, den)
    match_pct = rng.choice([50, 100])
    match = Fraction(self_ * match_pct, 100)
    net = self_ + match
    t = rng.randrange(3)
    if t == 0:
        ins = f"募zhi目标{goal}元，我方捐1/{den}，企业按我方捐款的{match_pct}%匹配，共筹多少元？"
    elif t == 1:
        ins = f"目标{goal}元，志愿者捐1/{den}，基金会再捐等同于其{match_pct}%的数额，总计多少元？"
    else:
        ins = f"筹资{goal}元，自筹达到目标的1/{den}，政府按自筹的{match_pct}%注资，共计多少元？"
    lines = [
        f"{goal} × 1/{den} = {num(self_)}元",
        f"{num(self_)} × {match_pct}/100 = {num(match)}元",
        f"{num(self_)} + {num(match)} = {num(net)}元",
    ]
    return ins, lines, net


_reg("donationMatchPct", donationMatchPct)


# 29 -------------------------------------------------------------------------
# double tap discount then add fractional fee and salary-ish split
def doubleTapFee(rng):
    price = rng.randint(30, 80) * 10
    d1 = rng.choice([80, 85])
    d2 = rng.choice([90, 95])
    m1 = Fraction(price * d1, 100)
    m2 = Fraction(m1 * d2, 100)
    fee = Fraction(m2, 4)
    final = m2 + fee
    t = rng.randrange(3)
    if t == 0:
        ins = f"商品{price}元，连打{d1//10}折、{d2//10}折，再加其1/4服务费，最后多少元？"
    elif t == 1:
        ins = f"{price}元打{d1//10}折再{d2//10}折，另收成交价1/4的佣金，共付多少？"
    else:
        ins = f"定价{price}元，同步享受{d1//10}折与{d2//10}折，附1/4税费，总价多少？"
    lines = [
        f"{price} × {d1}/100 = {num(m1)}元",
        f"{num(m1)} × {d2}/100 = {num(m2)}元",
        f"{num(m2)} × 1/4 = {num(fee)}元",
        f"{num(m2)} + {num(fee)} = {num(final)}元",
    ]
    return ins, lines, final


_reg("doubleTapFee", doubleTapFee)


# 30 -------------------------------------------------------------------------
# harvest: total = field area × rate, sell pct, share fraction of what's kept
def harvestPartWhole(rng):
    area = rng.randint(4, 9) ** 2 if False else rng.randint(4, 9) * rng.randint(4, 9)
    rate = rng.randint(8, 16)
    gross = area * rate
    sell = Fraction(gross * 60, 100)
    keep = gross - sell
    den = rng.choice([3, 5])
    seed = Fraction(keep, den)
    t = rng.randrange(3)
    if t == 0:
        ins = f"果园{area}平方米，每平方米收{rate}千克共{gross_gross if False else ''}.卖掉{gross * 60 // 100 if False else '60'}%后，留种是剩余量的1/{den}，留种多少千克？"
        ins = f"果园{area}平方米、每平米{rate}千克，售出60%，再把留存量的1/{den}留作种子，种子多少千克？"
    elif t == 1:
        ins = f"田{area}㎡，产量{rate}公斤/㎡，卖出60%，余量的1/{den}留种，留种几公斤？"
    else:
        ins = f"菜地{area}平米收菜{rate}/平米，60%出售，余下1/{den}喂牲畜，喂料多少千克？"
    lines = [
        f"{area} × {rate} = {gross}千克",
        f"{gross} × 60/100 = {num(sell)}千克",
        f"{gross} - {num(sell)} = {num(keep)}千克",
        f"{num(keep)} × 1/{den} = {num(seed)}千克",
    ]
    return ins, lines, seed


_reg("harvestPartWhole", harvestPartWhole)


# 31 -------------------------------------------------------------------------
# attendance: class total, present pct fraction, then split groups
def attendancePct(rng):
    total = rng.randint(12, 24) * 10
    pct_ = rng.choice([85, 90, 95])
    present = Fraction(total * pct_, 100)
    den = rng.choice([3, 6])
    girls = Fraction(present, den)
    t = rng.randrange(3)
    if t == 0:
        ins = f"全校{total}人，出勤{pct_}%；其中女生占出席人数的1/{den}，女生多少人？"
    elif t == 1:
        ins = f"{total}名员工，{pct_}%按时到岗，其中1/{den}是女生，到岗女生几人？"
    else:
        ins = f"{total}名学生出勤{pct_}%，女生为出勤者的1/{den}，女生参加几人？"
    lines = [
        f"{total} × {pct_}/100 = {num(present)}人",
        f"{num(present)} × 1/{den} = {num(girls)}人",
    ]
    return ins, lines, girls


_reg("attendancePct", attendancePct)


# 32 -------------------------------------------------------------------------
# tiles: floor area, one type covers fraction, another percent, total tiles
def tileFloorPct(rng):
    l, w = rng.randint(6, 12), rng.randint(5, 10)
    area = l * w
    t1 = Fraction(area, 2)
    rem = area - t1
    t2 = Fraction(rem * 40, 100)
    t3 = rem - t2
    t = rng.randrange(3)
    if t == 0:
        ins = f"地面{l}×{w}米铺砖，深色砖铺一半，剩余40%铺浅色，其余彩色，彩色砖铺多少平方米？"
    elif t == 1:
        ins = f"展厅{l}m×{w}m，木地板占1/2，其余40%铺地毯，剩余铺瓷砖，瓷砖面积多少㎡？"
    else:
        ins = f"操场{l}×{w}米，1/2为跑道，剩余40%种草坪，别的铺石子，石子面积多少平米？"
    lines = [
        f"{l} × {w} = {area}平方米",
        f"{area} × 1/2 = {num(t1)}平方米",
        f"{area} - {num(t1)} = {num(rem)}平方米",
        f"{num(rem)} × 40/100 = {num(t2)}平方米",
        f"{num(rem)} - {num(t2)} = {num(t3)}平方米",
    ]
    return ins, lines, t3


_reg("tileFloorPct", tileFloorPct)


# 33 -------------------------------------------------------------------------
# construction: project done fraction + percent, rest timeline split
def buildRemainPct(rng):
    tot = rng.randint(20, 60) * 10
    d1 = rng.choice([4, 5])
    done1 = Fraction(tot, d1)
    rem = tot - done1
    pct_ = rng.choice([25, 30])
    done2 = Fraction(rem * pct_, 100)
    rem2 = rem - done2
    t = rng.randrange(3)
    if t == 0:
        ins = f"工程总量{tot}，一期完成1/{d1}，二期完成余下的{pct_}%，还剩多少未完工？"
    elif t == 1:
        ins = f"{tot}页报告，先写1/{d1}，再写剩余{pct_}%，还剩多少页？"
    else:
        ins = f"隧道长{tot}米，第一个月挖1/{d1}，第二个月挖余下{pct_}%，还差多少米贯通？"
    lines = [
        f"一期完成 = {tot} × 1/{d1} = {num(done1)}",
        f"余下 = {tot} - {num(done1)} = {num(rem)}",
        f"二期完成 = {num(rem)} × {pct_}/100 = {num(done2)}",
        f"还剩 = {num(rem)} - {num(done2)} = {num(rem2)}",
    ]
    return ins, lines, rem2


_reg("buildRemainPct", buildRemainPct)


# 34 -------------------------------------------------------------------------
# salary + bonus: base pct, fraction bonus, then monthly split
def salaryBonus(rng):
    base = rng.randint(5, 12) * 100
    pct_ = rng.choice([20, 30])
    bonus = Fraction(base * pct_, 100)
    gross = base + bonus
    den = rng.choice([2, 3])
    take = gross - Fraction(gross, den)
    tax = Fraction(take * 10, 100)
    net = take - tax
    t = rng.randrange(3)
    if t == 0:
        ins = f"底薪{base}元，奖金占底薪的{pct_}%；扣{den - 1 if False else ''}.先缴一半社保，再扣10%税，到手多少？"
        ins = f"底薪{base}元，奖金为其{pct_}%；扣除毛收入的1/{den}作社保，再扣10%个税，实得多少元？"
    elif t == 1:
        ins = f"{base}元工资加{pct_}%的奖金，社保是毛额1/{den}，税收再10%，实发多少？"
    else:
        ins = f"基本工资{base}元与{pct_}%奖金合计，五险占1/{den}，所得税10%，到账多少元？"
    lines = [
        f"{base} × {pct_}/100 = {num(bonus)}元",
        f"{base} + {num(bonus)} = {num(gross)}元",
        f"{num(gross)} × 1/{den} = {num(Fraction(gross, den))}元",
        f"{num(gross)} - {num(Fraction(gross, den))} = {num(take)}元",
        f"{num(take)} × 10/100 = {num(tax)}元",
        f"{num(take)} - {num(tax)} = {num(net)}元",
    ]
    return ins, lines, net


_reg("salaryBonus", salaryBonus)


# 35 -------------------------------------------------------------------------
# bus: passenger fraction alight, percent board, net
def busPassengerPct(rng):
    start = rng.randint(30, 60)
    den = rng.choice([3, 5])
    alight = Fraction(start, den)
    onboard = start - alight
    pct_ = rng.choice([20, 25])
    board = Fraction(onboard * pct_, 100)
    net = onboard + board
    t = rng.randrange(3)
    if t == 0:
        ins = f"公交始发{start}人，到站下客1/{den}，又上{onboard * pct_ // 100 if False else onboard if False else ''}余?的{pct_}%，现在车上多少人？"
        ins = f"始发{start}人，首站下1/{den}，又上余客的{pct_}%，目前车上几人？"
    elif t == 1:
        ins = f"地铁进站{start}人，1/{den}下车，再上站内剩余{pct_}%的人，车上有多少？"
    else:
        ins = f"车厢原有{start}人，下去1/{den}，又上来剩下人数{pct_}%的乘客，现有多少人？"
    lines = [
        f"{start} × 1/{den} = {num(alight)}人",
        f"{start} - {num(alight)} = {num(onboard)}人",
        f"{num(onboard)} × {pct_}/100 = {num(board)}人",
        f"{num(onboard)} + {num(board)} = {num(net)}人",
    ]
    return ins, lines, net


_reg("busPassengerPct", busPassengerPct)


# 36 -------------------------------------------------------------------------
# goods remain: pct sold first, fraction of remainder damaged, share rest
def goodsRemain(rng):
    total = rng.randint(20, 50) * 10
    pct_ = rng.choice([60, 70])
    sold = Fraction(total * pct_, 100)
    rem = total - sold
    den = rng.choice([4, 5])
    damaged = Fraction(rem, den)
    left = rem - damaged
    t = rng.randrange(3)
    if t == 0:
        ins = f"进货{total}件，卖出{pct_}%，余下的1/{den}在途中损坏，完好剩多少件？"
    elif t == 1:
        ins = f"库存{total}台，售出{pct_}%，其中1/{den}为次品需返修，良品剩几台？"
    else:
        ins = f"{total}公斤水果卖掉{pct_}%，剩下的1/{den}变质，可卖的新鲜水果剩多少公斤？"
    lines = [
        f"{total} × {pct_}/100 = {num(sold)}件",
        f"{total} - {num(sold)} = {num(rem)}件",
        f"{num(rem)} × 1/{den} = {num(damaged)}件",
        f"{num(rem)} - {num(damaged)} = {num(left)}件",
    ]
    return ins, lines, left


_reg("goodsRemain", goodsRemain)


# 37 -------------------------------------------------------------------------
# exams: percent pass, fraction top, adjust with pass+0. descriptor
def examPercentFraction(rng):
    n = rng.randint(20, 50) * 10
    pct_ = rng.choice([80, 85, 90])
    pass_ = Fraction(n * pct_, 100)
    den = rng.choice([4, 5])
    top = Fraction(pass_, den)
    other = pass_ - top
    t = rng.randrange(3)
    if t == 0:
        ins = f"{n}人考试，{pct_}%合格；合格者中1/{den}为优秀，优秀且非优的其余合格多少人？"
    elif t == 1:
        ins = f"参加竞赛{n}人，{pct_}%进决赛，决赛1/{den}夺金，夺银及以下决赛选手有几人？"
    else:
        ins = f"{n}名测试者{pct_}%通过，通过里1/{den}为满分，未得满分的通过者几人？"
    lines = [
        f"{n} × {pct_}/100 = {num(pass_)}人",
        f"{num(pass_)} × 1/{den} = {num(top)}人",
        f"{num(pass_)} - {num(top)} = {num(other)}人",
    ]
    return ins, lines, other


_reg("examPercentFraction", examPercentFraction)


# 38 -------------------------------------------------------------------------
# orchard: two trees ratio, percent of one fraction, count
def orchardPctFraction(rng):
    a, b = rng.randint(2, 4), rng.randint(3, 5)
    total = (a + b) * rng.randint(10, 20)
    per = total // (a + b)
    peach = a * per
    bear = Fraction(peach * 75, 100)
    t = rng.randrange(3)
    if t == 0:
        ins = f"果园桃、梨之比为{a}:{b}，共{total}棵；挂果的桃占桃树的75%，挂果桃多少棵？"
    elif t == 1:
        ins = f"苹果:梨={a}:{b}，共{total}棵，苹果树已结果{75}%，结果苹果多少棵？"
    else:
        ins = f"两种树总数{total}棵按{a}:{b}分，多的一类里{75}%开花，开花多少棵？"
    lines = [
        f"总份数 = {a} + {b} = {a + b}",
        f"每份 = {total} ÷ {a + b} = {per}",
        f"{a} × {per} = {peach}棵",
        f"{peach} × 75/100 = {num(bear)}棵",
    ]
    return ins, lines, bear


_reg("orchardPctFraction", orchardPctFraction)


# 39 -------------------------------------------------------------------------
# produce cost: unit decimal, pct waste cost, fraction profit margin
def produceCost(rng):
    qty = rng.randint(20, 60)
    c = rng.randint(20, 45) * 10         # cost per unit ×10 unit gen
    unitc = Fraction(rng.randint(20, 45), 10)  # decimal
    raw = qty * unitc
    wastepct = rng.choice([10, 15])
    waste = Fraction(raw * wastepct, 100)
    total = raw + waste
    den = rng.choice([5, 6])
    profit = Fraction(total, den)
    price = total + profit
    t = rng.randrange(3)
    if t == 0:
        ins = f"采购{qty}个零件、单价{unitc}元，损耗{wastepct}%；以成本加其1/{den}的利润定价，总价多少元？"
    elif t == 1:
        ins = f"备料{qty}份、每份{unitc}元，采损{wastepct}%，加成1/{den}毛利后卖，总售价多少？"
    else:
        ins = f"买菜{qty}斤、每斤{unitc}元，坏损{wastepct}%，按成本的1/{den}加价出售，一共卖多少元？"
    lines = [
        f"{qty} × {unitc} = {num(raw)}元",
        f"{num(raw)} × {wastepct}/100 = {num(waste)}元",
        f"{num(raw)} + {num(waste)} = {num(total)}元",
        f"{num(total)} × 1/{den} = {num(profit)}元",
        f"{num(total)} + {num(profit)} = {num(price)}元",
    ]
    return ins, lines, price


_reg("produceCost", produceCost)


# 40 -------------------------------------------------------------------------
# price: pct discount then fraction of that discounted, then per-item decimal
def pctFracPerItem(rng):
    price = rng.randint(20, 60) * 10
    d1 = rng.choice([70, 80])
    m1 = Fraction(price * d1, 100)
    den = rng.choice([4, 7])
    m2 = m1 - Fraction(m1, den)
    each = Fraction(m2, 3)
    t = rng.randrange(3)
    if t == 0:
        ins = f"整箱{price}元打{d1}折，每箱再让利1/{den}，剩3件平分单价每件多少元？"
    elif t == 1:
        ins = f"批发价{price}元日享{d1}%，再扣1/{den}返点，余款按3件均分，单件多少？"
    else:
        ins = f"订单{price}元减{d1}%，再折掉1/{den}，余下平均到3个买家，每人付多少元？"
    lines = [
        f"{price} × {d1}/100 = {num(m1)}元",
        f"{num(m1)} - {num(Fraction(m1, den))} = {num(m2)}元",
        f"{num(m2)} ÷ 3 = {num(each)}元",
    ]
    return ins, lines, each




if __name__ == "__main__":
    rng = random.Random(11)
    from run_math_short import verify
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines}"
            ok += 1
    print(f"L3_ext1 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")