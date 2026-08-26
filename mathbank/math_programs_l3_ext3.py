#!/usr/bin/env python3
"""L3 ext3: 62 distinct 5-7 step families (percent/fraction/ratio/decimal/geometry).

Every program: fn(rng) -> (instruction, lines, ans); >=3 equation lines;
>=4 phrasings; all non-integer arithmetic via Fraction, rendered with num().
Verified against run_math_short.verify.
"""
import random
import re
from fractions import Fraction
from mathcommon import (ANIMALS, FOOD, FRUITS, GOODS, NAMES, PLACE, STATIONERY,
                         UNIT_FRUIT, UNIT_N, UNIT_ZHI, num)

PROGRAMS = []


def _reg(name, fn):
    def wrapped(rng):
        ins, lines, ans = fn(rng)
        labels = _LABELS.get(name, ())
        out, i = [], 0
        for ln in lines:
            parts = ln.split("=")
            if len(parts) == 2 and not re.search(r"[一-鿿]", parts[1]):
                out.append(f"{labels[i]} = {ln}")
                i += 1
            else:
                out.append(ln)
        return ins, out, ans
    PROGRAMS.append(("L3", name, wrapped))


# Chinese label per bare line (a line whose RHS carries no unit/Chinese), in order.
_LABELS = {
    "simple_interest_rate": ["每年利息", "年利率"],
    "reverse_discount": ["折扣"],
    "discount_find_rate": ["最多折扣"],
    "buy_n_get_m_free": ["每组件数", "组数", "需付钱数"],
    "manjian_discount": ["满减次数"],
    "discount_loss_gain": ["折扣差"],
    "equivalent_double_discount": ["第一次折扣", "第二次折扣", "最终折扣"],
    "rebate_effective": ["实际比例"],
    "population_pct_diff": ["两倍占比", "多出的百分比", "扩大100倍", "全镇人数"],
    "ratio_transfer": ["份数差", "两倍的x", "每份数量", "甲原有的数量"],
    "digit_ratio_number": ["份数和", "每份", "十位数字", "个位数字", "这个两位数"],
    "three_ratio_diff": ["份数和", "份数差", "每份", "最大数"],
    "continued_ratio": ["甲的份数", "乙的份数", "丙的份数", "份数和", "每份", "甲数"],
    "fraction_add_same": ["两外项积", "两内项积", "积的差", "份数差", "这个数"],
    "fraction_value_find": ["份数和", "每份", "分子"],
    "ratio_speed_meet": ["份数差", "每份路程", "份数和", "两地距离"],
    "triangle_ratio_side": ["份数和", "每份长度", "最长边"],
    "cuboid_ratio_volume": ["长宽高之和", "每份长度", "长", "宽", "高", "体积"],
    "conc_ratio_water": ["份数和", "每份重量", "含盐率"],
    "three_invest_weighted": ["甲的权重", "乙的权重", "丙的权重", "权重和"],
    "fraction_give_equal": ["份数差", "两倍的x", "乙的份数倍", "乙原有的数量"],
    "pct_transfer_equal": ["乙的百分比", "转移的钱", "甲剩下的钱", "乙得到后的钱", "甲原有的钱"],
    "fraction_equal_remainder": ["乙筐剩下的", "甲筐剩下的a倍", "甲筐原有的"],
    "fraction_three_people": ["分母积", "丙的份数", "总价的份数倍", "物品总价"],
    "work_b_alone": ["合作效率", "甲的效率", "乙的效率", "乙单独做的天数"],
    "work_fraction_completion": ["甲的效率", "乙的效率", "合作效率", "合作天数"],
    "work_efficiency_ratio": ["合作效率", "份数和", "每份效率", "甲的效率", "甲单独做的天数"],
    "work_three_pair": ["甲乙效率和", "乙丙效率和", "甲丙效率和", "两倍三人效率和", "三人效率和", "三人合作天数"],
    "work_rest_day": ["乙的效率", "乙完成的量", "剩余工程量", "甲的效率", "合作效率", "合作天数", "总天数"],
    "work_machines": ["一台机器的效率", "机器台数", "总效率", "需要的时间"],
    "travel_meet_rest": ["乙先走的路程", "剩余路程", "速度和", "相遇时间", "总时间"],
    "travel_two_leg_dist": ["第一段时间", "第二段时间", "总路程", "总时间", "平均速度"],
    "boat_roundtrip_time": ["顺水速度", "逆水速度", "顺水时间", "逆水时间", "往返总时间"],
    "boat_find_speed": ["顺水速度", "逆水速度", "速度和", "静水速度"],
    "train_pole_bridge": ["多用的秒数", "火车速度", "火车长度"],
    "pool_capacity_frac": ["容量份数差", "注入量的d倍", "再乘e", "水池容量"],
    "travel_meet_fraction": ["乙行的份数", "甲行的路程", "乙行的路程", "乙行路程的b倍", "两地距离"],
    "average_add_one": ["原来的总数", "新的总数", "新的个数", "新的平均数"],
    "average_remove_one": ["5个数的和", "4个数的和", "去掉的数"],
    "average_find_count": ["前几次的总分", "现在的总分", "这次的成绩", "测验次数"],
    "circle_area_cost": ["半径的平方", "花坛面积"],
    "circle_laps": ["半径", "一圈的长度", "一共跑的米数"],
    "circle_ring": ["外圆半径平方", "内圆半径平方", "半径平方差", "环岛面积"],
    "cylinder_volume": ["半径平方", "底面积", "容积"],
    "cube_surface_cost": ["一个面的面积", "表面积"],
    "cuboid_edge_surface": ["长宽高之和", "棱长总和", "上下底面面积", "前后两面面积", "左右两面面积", "三对面面积和", "表面积"],
    "volume_rock": ["水面上升高度", "底面积", "石头体积"],
    "volume_pour": ["甲水箱底面积", "水的体积", "乙水箱底面积", "水深"],
    "stair_carpet": ["水平总长", "垂直总长", "地毯长度"],
    "planting_trees": ["一旁的间隔数", "一旁的棵数", "总棵数"],
    "sawing_time": ["锯的次数", "锯一次的时间", "锯的次数", "需要的时间"],
    "stairs_time": ["要走的层数", "每层的秒数", "要走的层数", "需要的秒数"],
    "clock_strike": ["间隔数", "每个间隔的秒数", "间隔数", "需要的秒数"],
    "oil_bucket": ["用去的重量", "油的总重量", "桶的重量"],
    "conc_add_solute": ["原有盐的重量", "现在盐的重量", "现在盐水的重量", "含盐率"],
    "conc_mix_find_amount": ["需提高的浓度", "多出的浓度", "盐的差量倍", "乙杯盐水的重量"],
    "conc_evaporate_to_target": ["原有盐的重量", "盐的100倍", "蒸发后盐水的重量", "蒸发掉的水"],
    "sprout_replant": ["发芽的种子数", "发芽总数", "种子总数", "总发芽率"],
    "juice_mix_ratio": ["第一杯份数和", "第一杯果汁重量", "第二杯份数和", "第二杯果汁重量", "果汁总重量", "糖水总重量", "混合后浓度"],
    "reverse_three_stage": ["第三天前的d3倍", "第三天前的量", "第二天前的d2倍", "第二天前的量", "原来的d1倍", "原来的总量"],
    "fraction_book_reading": ["已看的占比", "剩下的占比", "剩下页数的d倍", "全书页数"],
    "class_two_fractions": ["数学小组人数", "剩下的人数", "英语小组人数", "都没参加的人数"],
    "ratio_age_future": ["年龄份数和", "每份年龄", "父亲年龄", "儿子年龄", "需要的年数"],
    "fraction_rope_diff": ["第一根用去的", "第二根用去的", "第一根剩下的", "第二根剩下的", "剩下的差"],
}


# 1. simple interest: principal + interest -> total
def simple_interest(rng):
    P = rng.randint(2, 20) * 500
    r = rng.randint(2, 6)
    t = rng.randint(2, 5)
    per_year = Fraction(P * r, 100)
    interest = per_year * t
    total = P + interest
    ins = rng.choice([
        f"把{P}元存入银行，年利率{r}%，存{t}年，到期后本金和利息一共多少元？",
        f"妈妈把{P}元钱存入银行，定期{t}年，年利率{r}%，到期时一共能取回多少元？",
        f"银行一年期年利率{r}%，爸爸存入{P}元，存{t}年后本息共多少元？",
        f"小红把压岁钱{P}元存入银行，年利率{r}%，{t}年后她的账户上共有多少元？",
    ])
    lines = [
        f"{P} × {r}/100 = {num(per_year)}元",
        f"{num(per_year)} × {t} = {num(interest)}元",
        f"{P} + {num(interest)} = {num(total)}元",
    ]
    return ins, lines, total


_reg("simple_interest", simple_interest)


# 2. find annual interest rate from interest earned
def simple_interest_rate(rng):
    P = rng.choice([1000, 2000, 3000, 5000, 8000])
    t = rng.randint(2, 5)
    r = rng.randint(2, 6)
    I = Fraction(P * t * r, 100)
    per_year = Fraction(I, t)
    rate = Fraction(per_year, P)
    ins = rng.choice([
        f"把{P}元存入银行，{t}年后得到利息{num(I)}元，年利率是多少？",
        f"爸爸存入银行{P}元，定期{t}年，到期获利息{num(I)}元，年利率是百分之几？",
        f"一笔{P}元的存款，存{t}年得利息{num(I)}元，银行年利率是多少？",
        f"妈妈存了{P}元钱，{t}年后利息是{num(I)}元，年利率是百分之几？",
    ])
    lines = [
        f"{num(I)} ÷ {t} = {num(per_year)}元",
        f"{num(per_year)} ÷ {P} = {num(rate)}",
        f"{num(rate)} × 100 = {r}",
    ]
    return ins, lines, r


_reg("simple_interest_rate", simple_interest_rate)


# 3. discounted price given -> original price
def reverse_discount(rng):
    p = rng.choice([5, 6, 7, 8, 9])
    k = rng.randint(10, 80)
    X = p * k
    orig = 10 * k
    saved = orig - X
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}打{p}折后售价是{X}元，原价是多少元？",
        f"商店促销，{obj}按原价的{p}折出售，卖{X}元，原价多少元？",
        f"一台电器打{p}折后的价格是{X}元，这台电器原价多少元？",
        f"某商品现价{X}元，是按原价打{p}折出售的，原价是多少元？",
    ])
    lines = [
        f"{p} ÷ 10 = {num(Fraction(p, 10))}",
        f"{orig} - {X} = {saved}元",
        f"{X} ÷ ({num(Fraction(p, 10))}) = {orig}元",
    ]
    return ins, lines, orig


_reg("reverse_discount", reverse_discount)


# 4. max discount rate that keeps a target profit margin
def discount_find_rate(rng):
    pairs = [(50, 20), (50, 5), (60, 12), (40, 12), (40, 26), (30, 4), (25, 10)]
    p, r = rng.choice(pairs)
    C = rng.randint(5, 30) * 10
    price = Fraction(C * (100 + p), 100)
    target = Fraction(C * (100 + r), 100)
    rate = Fraction(target * 10, price)
    ins = rng.choice([
        f"一件商品成本是{C}元，按成本加价{p}%定价，要保持{r}%的利润，最多可以打几折？",
        f"某电器成本{C}元，商家先加价{p}%标价，促销时要保证{r}%的利润率，最低能打几折？",
        f"一件商品成本{C}元，定价时加价{p}%，打折出售仍要赚{r}%，最多打几折？",
        f"某商品成本{C}元，按加价{p}%定价，为了保证{r}%的利润，最多能按几折出售？",
    ])
    lines = [
        f"{C} × ({100 + p}/100) = {num(price)}元",
        f"{C} × ({100 + r}/100) = {num(target)}元",
        f"{num(target)} ÷ ({num(price)}) × 10 = {num(rate)}",
    ]
    return ins, lines, rate


_reg("discount_find_rate", discount_find_rate)


# 5. buy n get m free: cost of taking home total pieces
def buy_n_get_m_free(rng):
    n = rng.randint(2, 4)
    m = rng.randint(1, 3)
    g = rng.randint(2, 6)
    price = rng.randint(5, 20)
    k = n + m
    total = k * g
    paid = n * g
    cost = paid * price
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店促销：买{n}件{obj}送{m}件。小明想买回{total}件，每件{price}元，一共要付多少元？",
        f"文具店搞活动，买{n}支送{m}支，老师需要{total}支铅笔，每支{price}元，要付多少元？",
        f"超市里{obj}买{n}送{m}，小红要得到{total}件，每件{price}元，她需付多少元？",
        f"一种{obj}买{n}件赠{m}件，要买够{total}件，每件{price}元，共花多少元？",
    ])
    lines = [
        f"{n} + {m} = {k}",
        f"{total} ÷ {k} = {g}",
        f"{n} × {g} = {paid}",
        f"{paid} × {price} = {cost}元",
    ]
    return ins, lines, cost


_reg("buy_n_get_m_free", buy_n_get_m_free)


# 6. full-reduction: spend M get C off, total T -> pay
def manjian_discount(rng):
    M = rng.choice([100, 150, 200, 250, 300, 400])
    n = rng.randint(2, 5)
    C = rng.choice([10, 20, 30, 50, 80])
    for _ in range(50):
        if C < M:
            break
        C = rng.choice([10, 20, 30, 50, 80])
    T = M * n
    off = n * C
    pay = T - off
    ins = rng.choice([
        f"商场促销：每满{M}元减{C}元。妈妈买了总价{T}元的商品，实际要付多少元？",
        f"超市活动满{M}元减{C}元，爸爸买了{T}元的日用品，应付多少元？",
        f"一家店规定消费满{M}元减{C}元，买{T}元的商品实际花多少元？",
        f"商场满减活动：每满{M}元减{C}元，小红买了{T}元的文具，实际付多少元？",
    ])
    lines = [
        f"{T} ÷ {M} = {n}",
        f"{n} × {C} = {off}元",
        f"{T} - {off} = {pay}元",
    ]
    return ins, lines, pay


_reg("manjian_discount", manjian_discount)


# 7. two items, one gains p% one loses p% -> overall loss
def two_items_profit_loss(rng):
    pr = rng.choice([(20, 1), (20, 2), (20, 3), (20, 4), (20, 5), (20, 6),
                     (25, 3), (25, 6), (40, 3), (40, 6), (50, 1), (50, 2),
                     (50, 3), (50, 4), (10, 9)])
    p, rr = pr
    P = (100 + p) * rr
    cost1 = 100 * rr
    cost2 = Fraction(100 * P, 100 - p)
    loss = cost1 + cost2 - 2 * P
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店卖出两件{obj}，各卖{P}元，一件赚了{p}%，另一件亏了{p}%，商店总体亏了多少元？",
        f"两件{obj}都卖{P}元，一件盈利{p}%，一件亏损{p}%，合起来亏了多少元？",
        f"老板卖两台{obj}，每台售价都是{P}元，一台赚{p}%，一台赔{p}%，总共亏多少元？",
        f"两种{obj}各卖{P}元，其中一种赚{p}%，另一种亏{p}%，卖这两件亏了多少元？",
    ])
    lines = [
        f"{P} ÷ ({100 + p}/100) = {num(cost1)}元",
        f"{P} ÷ ({100 - p}/100) = {num(cost2)}元",
        f"{num(cost1)} + {num(cost2)} = {num(cost1 + cost2)}元",
        f"{num(cost1 + cost2)} - {2 * P} = {num(loss)}元",
    ]
    return ins, lines, loss


_reg("two_items_profit_loss", two_items_profit_loss)


# 8. two discount schemes, one loses one gains -> cost price
def discount_loss_gain(rng):
    p = rng.choice([6, 7, 8])
    q = p + rng.choice([1, 2])
    c = rng.randint(1, 10) * 10
    e = rng.randint(1, 10) * 10
    for _ in range(50):
        if (c + e) * 10 % (q - p) == 0:
            break
        c = rng.randint(1, 10) * 10
        e = rng.randint(1, 10) * 10
    diff = q - p
    P = Fraction((c + e) * 10, diff)
    sell_p = Fraction(P * p, 10)
    cost = sell_p + c
    ins = rng.choice([
        f"一件商品打{p}折出售亏{c}元，打{q}折出售赚{e}元，这件商品的成本是多少元？",
        f"某商品按定价打{p}折卖亏{c}元，打{q}折卖赚{e}元，成本价是多少元？",
        f"一台电器打{p}折出售赔{c}元，打{q}折出售盈利{e}元，它的成本是多少元？",
        f"一件商品定价的{p}折比成本低{c}元，定价的{q}折比成本高{e}元，成本是多少元？",
    ])
    lines = [
        f"{q} - {p} = {diff}",
        f"{c} + {e} = {c + e}元",
        f"{c + e} × 10 = {(c + e) * 10}元",
        f"{(c + e) * 10} ÷ {diff} = {num(P)}元",
        f"{num(P)} × {p}/10 = {num(sell_p)}元",
        f"{num(sell_p)} + {c} = {num(cost)}元",
    ]
    return ins, lines, cost


_reg("discount_loss_gain", discount_loss_gain)


# 9. two stacked discounts -> equivalent single fraction of original
def equivalent_double_discount(rng):
    p = rng.choice([6, 7, 8, 9])
    q = rng.choice([6, 7, 8, 9])
    fp = Fraction(p, 10)
    fq = Fraction(q, 10)
    rate = fp * fq
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}先打{p}折，再打{q}折，最终价格相当于原价的几分之几？",
        f"某{obj}先按{p}折出售，顾客持会员卡再打{q}折，实际付的钱是原价的几分之几？",
        f"一台{obj}先降价到{p}折，再优惠打{q}折，现价是原价的几分之几？",
        f"商店的{obj}先打{p}折，再打{q}折促销，最终售价是原价的几分之几？",
    ])
    lines = [
        f"{p} ÷ 10 = {num(fp)}",
        f"{q} ÷ 10 = {num(fq)}",
        f"{num(fp)} × {num(fq)} = {num(rate)}",
    ]
    return ins, lines, rate


_reg("equivalent_double_discount", equivalent_double_discount)


# 10. rebate voucher -> effective fraction of value
def rebate_effective(rng):
    M = rng.choice([100, 150, 200, 250, 300, 400, 500, 600, 800, 1000])
    p = rng.choice([10, 20, 25, 30, 40])
    voucher = Fraction(M * p, 100)
    value = M + voucher
    rate = Fraction(M, value)
    ins = rng.choice([
        f"商场每消费{M}元返还{p}%的购物券，小明花了{M}元并用券购物，实际花费相当于得到商品总价值的几分之几？",
        f"超市满{M}元送{p}%购物券，妈妈花了{M}元，券也用完了，她的实际花费是所得商品价值的几分之几？",
        f"商店促销：每付{M}元返{p}%购物券，顾客花掉{M}元并花完券，实际支付相当于商品价值的几分之几？",
        f"某店每消费{M}元赠{p}%的券，小红花了{M}元，券全部使用，她实际花的钱是商品总价的几分之几？",
    ])
    lines = [
        f"{M} × {p}/100 = {num(voucher)}元",
        f"{M} + {num(voucher)} = {num(value)}元",
        f"{M} ÷ ({num(value)}) = {num(rate)}",
    ]
    return ins, lines, rate


_reg("rebate_effective", rebate_effective)


# 11. compound growth: same entity grows p% for two months
def compound_growth(rng):
    a = rng.randint(2, 20) * 100
    p = rng.choice([5, 10, 15, 20, 25])
    inc1 = Fraction(a * p, 100)
    m1 = a + inc1
    inc2 = Fraction(m1 * p, 100)
    m2 = m1 + inc2
    unit = rng.choice(["件", "吨", "元", "千克"])
    ins = rng.choice([
        f"工厂本月生产{a}{unit}产品，以后每月比上月增长{p}%，两个月后每月生产多少{unit}？",
        f"某店本月营业额{a}元，预计每月增长{p}%，两个月后的营业额是多少元？",
        f"小明家本月用电{a}度，以后每月节约增长{p}%，两个月后用电多少度？",
        f"养殖场本月出栏{a}{unit}，计划每月增产{p}%，两个月后每月出栏多少{unit}？",
    ])
    lines = [
        f"{a} × {p}/100 = {num(inc1)}{unit}",
        f"{a} + {num(inc1)} = {num(m1)}{unit}",
        f"{num(m1)} × {p}/100 = {num(inc2)}{unit}",
        f"{num(m1)} + {num(inc2)} = {num(m2)}{unit}",
    ]
    return ins, lines, m2


_reg("compound_growth", compound_growth)


# 12. two salaries both raised p% -> new gap
def salary_raise_diff(rng):
    d = rng.randint(5, 30) * 100
    b = rng.randint(30, 80) * 100
    a = b + d
    p = rng.choice([5, 10, 15, 20])
    inc = Fraction(d * p, 100)
    newdiff = d + inc
    ins = rng.choice([
        f"甲月薪{a}元，乙月薪{b}元，两人工资都上涨{p}%后，甲比乙多多少元？",
        f"哥哥月工资{a}元，弟弟月工资{b}元，都涨薪{p}%后，两人工资相差多少元？",
        f"甲厂人均工资{a}元，乙厂人均工资{b}元，两厂都上调{p}%后，相差多少元？",
        f"爸爸月薪{a}元，妈妈月薪{b}元，两人同时加薪{p}%后，爸爸比妈妈多多少元？",
    ])
    lines = [
        f"{a} - {b} = {d}元",
        f"{d} × {p}/100 = {num(inc)}元",
        f"{d} + {num(inc)} = {num(newdiff)}元",
    ]
    return ins, lines, newdiff


_reg("salary_raise_diff", salary_raise_diff)


# 13. population: urban p%, urban exceeds rural by x -> total
def population_pct_diff(rng):
    p = rng.choice([55, 60, 65, 70, 75])
    k = rng.randint(2, 20)
    x = (2 * p - 100) * k
    total = 100 * k
    ins = rng.choice([
        f"某镇人口中城镇人口占{p}%，城镇人口比农村人口多{x}万人，全镇共有多少万人？",
        f"某校学生中走读生占{p}%，走读生比住宿生多{x}人，全校共有多少人？",
        f"一批产品中合格品占{p}%，合格品比不合格品多{x}件，这批产品共多少件？",
        f"某村村民中参加医保的占{p}%，参保的比未参保的多{x}人，全村共有多少人？",
    ])
    lines = [
        f"2 × {p} = {2 * p}",
        f"{2 * p} - 100 = {2 * p - 100}",
        f"{x} × 100 = {x * 100}",
        f"{x * 100} ÷ {2 * p - 100} = {total}",
    ]
    return ins, lines, total


_reg("population_pct_diff", population_pct_diff)


# 14. decimal unit price x integer qty -> change from 100
def decimal_price_change(rng):
    a = rng.randrange(5, 19, 2)
    p = Fraction(a, 2)
    q = rng.randint(3, 12)
    for _ in range(50):
        if p * q <= 100:
            break
        q = rng.randint(3, 12)
    total = p * q
    change = 100 - total
    obj = rng.choice(GOODS)
    name = rng.choice(NAMES)
    ins = rng.choice([
        f"{name}买了{q}个单价{num(p)}元的{obj}，付给售货员100元，应找回多少元？",
        f"妈妈买{q}千克{num(p)}元一千克的苹果，付100元，找回多少元？",
        f"文具店的笔记本每本{num(p)}元，小红买{q}本，给100元，应找回多少元？",
        f"一种{obj}每个{num(p)}元，买{q}个，付100元，找回多少元？",
    ])
    lines = [
        f"50 + 50 = 100元",
        f"{num(p)} × {q} = {num(total)}元",
        f"100 - {num(total)} = {num(change)}元",
    ]
    return ins, lines, change


_reg("decimal_price_change", decimal_price_change)


# 15. rational table dimensions -> perimeter, area, cost
def rational_table(rng):
    l = rng.choice([Fraction(3, 2), Fraction(5, 2), Fraction(7, 2),
                    Fraction(4, 3), Fraction(5, 3), Fraction(7, 3)])
    w = rng.choice([Fraction(3, 2), Fraction(5, 2), Fraction(4, 3),
                    Fraction(5, 3), Fraction(7, 4), Fraction(9, 4)])
    c = rng.randint(10, 50)
    peri = 2 * (l + w)
    area = l * w
    total = area * c
    ins = rng.choice([
        f"一张长方形桌面长{num(l)}米、宽{num(w)}米，每平方米木料{c}元，做这张桌面需要多少元？",
        f"一块长方形布长{num(l)}米、宽{num(w)}米，每平方米布{c}元，这块布值多少元？",
        f"长方形地毯长{num(l)}米、宽{num(w)}米，每平方米{c}元，铺满要多少元？",
        f"一块长方形玻璃长{num(l)}米、宽{num(w)}米，每平方米{c}元，共需多少元？",
    ])
    lines = [
        f"{num(l)} + {num(w)} = {num(l + w)}米",
        f"2 × {num(l + w)} = {num(peri)}米",
        f"{num(l)} × {num(w)} = {num(area)}平方米",
        f"{num(area)} × {c} = {num(total)}元",
    ]
    return ins, lines, total


_reg("rational_table", rational_table)


# 16. km/h -> m/s then distance in t seconds
def speed_kmh_to_ms(rng):
    v = rng.choice([36, 54, 72, 90, 108])
    t = rng.randint(5, 30)
    m_h = v * 1000
    rate = Fraction(m_h, 3600)
    d = rate * t
    ins = rng.choice([
        f"一辆汽车每小时行{v}千米，照这样的速度，{t}秒能行多少米？",
        f"火车的速度是每小时{v}千米，{t}秒行驶多少米？",
        f"小明骑车每小时行{v}千米，{t}秒能骑多少米？",
        f"一架飞机每秒飞行的速度是每小时{v}千米，{t}秒飞多少米？",
    ])
    lines = [
        f"{v} × 1000 = {m_h}米",
        f"{m_h} ÷ 3600 = {num(rate)}米",
        f"{num(rate)} × {t} = {num(d)}米",
    ]
    return ins, lines, d


_reg("speed_kmh_to_ms", speed_kmh_to_ms)


# 17. ratio a:b, A gives B x so they equal -> A's original amount
def ratio_transfer(rng):
    a = rng.randint(3, 9)
    b = rng.randint(2, a - 1)
    d = a - b
    k = rng.randint(2, 12)
    x = d * k
    per = 2 * k
    A = a * per
    obj = rng.choice(["元", "本书", "颗糖", "张邮票"])
    ins = rng.choice([
        f"甲、乙两人的钱数比是{a}:{b}，甲给乙{x}{obj}后两人钱数相等，甲原来有多少{obj}？",
        f"甲、乙两数的比是{a}:{b}，甲数减去{x}后与乙数相等，甲数是多少？",
        f"哥哥和弟弟的邮票数比是{a}:{b}，哥哥给弟弟{x}张后两人一样多，哥哥原来有多少张？",
        f"甲、乙两堆货物的比是{a}:{b}，从甲堆运{x}吨到乙堆后两堆相等，甲堆原有多少吨？",
    ])
    lines = [
        f"{a} - {b} = {d}",
        f"{x} × 2 = {2 * x}",
        f"{2 * x} ÷ {d} = {per}",
        f"{a} × {per} = {A}",
    ]
    return ins, lines, A


_reg("ratio_transfer", ratio_transfer)


# 18. two-digit number: tens:units ratio + digit sum -> the number
def digit_ratio_number(rng):
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    for _ in range(50):
        if a != b:
            break
        b = rng.randint(1, 9)
    per = rng.choice([1, 2, 3])
    for _ in range(50):
        if max(a, b) * per <= 9:
            break
        per = rng.choice([1, 2, 3])
    S = (a + b) * per
    tens = a * per
    units = b * per
    number = tens * 10 + units
    ins = rng.choice([
        f"一个两位数，十位数字与个位数字的比是{a}:{b}，两个数字的和是{S}，这个两位数是多少？",
        f"一个两位数，个位数字和十位数字的比是{b}:{a}，数字和是{S}，这个数是多少？",
        f"小明想了一个两位数，十位与个位数字之比为{a}:{b}，数字之和为{S}，这个数是几？",
        f"一个两位数，十位数字是个位数字的{a}/{b}倍，两个数字相加得{S}，这个两位数是多少？",
    ])
    lines = [
        f"{a} + {b} = {a + b}",
        f"{S} ÷ {a + b} = {per}",
        f"{a} × {per} = {tens}",
        f"{b} × {per} = {units}",
        f"{tens} × 10 + {units} = {number}",
    ]
    return ins, lines, number


_reg("digit_ratio_number", digit_ratio_number)


# 19. three numbers in ratio, max-minus-min difference given -> max
def three_ratio_diff(rng):
    a = rng.randint(1, 4)
    b = rng.randint(1, 5)
    c = rng.randint(2, 6)
    for _ in range(50):
        if not (a == b == c):
            break
        b = rng.randint(1, 5)
    lo = min(a, b, c)
    hi = max(a, b, c)
    diff = hi - lo
    k = rng.randint(3, 15)
    D = diff * k
    maxv = hi * k
    ins = rng.choice([
        f"甲、乙、丙三个数的比是{a}:{b}:{c}，最大数比最小数多{D}，最大数是多少？",
        f"三个班分得的图书本数比是{a}:{b}:{c}，最多的班比最少的班多{D}本，最多的班分多少本？",
        f"甲、乙、丙三人的体重比是{a}:{b}:{c}，最重的比最轻的重{D}千克，最重的是多少千克？",
        f"三个数的比为{a}:{b}:{c}，其中最大数与最小数相差{D}，最大数是几？",
    ])
    lines = [
        f"{a} + {b} + {c} = {a + b + c}",
        f"{hi} - {lo} = {diff}",
        f"{D} ÷ {diff} = {k}",
        f"{hi} × {k} = {maxv}",
    ]
    return ins, lines, maxv


_reg("three_ratio_diff", three_ratio_diff)


# 20. continued ratio A:B and B:C -> combined, total given -> A
def continued_ratio(rng):
    pairs = [(2, 3), (3, 4), (4, 5), (3, 5), (2, 5), (5, 6)]
    b, c = rng.choice(pairs)
    a = rng.randint(1, 3)
    d = rng.randint(1, 3)
    ac = a * c
    bc = b * c
    bd = b * d
    s = ac + bc + bd
    k = rng.randint(2, 6)
    T = s * k
    A = ac * k
    ins = rng.choice([
        f"甲、乙的钱数比是{a}:{b}，乙、丙的钱数比是{c}:{d}，三人共有{T}元，甲有多少元？",
        f"甲:乙={a}:{b}，乙:丙={c}:{d}，三个数的和是{T}，甲数是多少？",
        f"三个组的人数比中，一组与二组的比是{a}:{b}，二组与三组的比是{c}:{d}，三组共{T}人，一组有多少人？",
        f"甲、乙、丙三人藏书，甲与乙的本数比是{a}:{b}，乙与丙的本数比是{c}:{d}，三人共藏书{T}本，甲有多少本？",
    ])
    lines = [
        f"{a} × {c} = {ac}",
        f"{b} × {c} = {bc}",
        f"{b} × {d} = {bd}",
        f"{ac} + {bc} + {bd} = {s}",
        f"{T} ÷ {s} = {k}",
        f"{ac} × {k} = {A}",
    ]
    return ins, lines, A


_reg("continued_ratio", continued_ratio)


# 21. fraction a/b, add same x to numerator and denominator -> equals c/d, find x
def fraction_add_same(rng):
    a = rng.randint(2, 8)
    b = rng.randint(2 * a + 1, 2 * a + 9)
    c, d = 1, 2
    bc = b * c
    ad = a * d
    numer = bc - ad
    den = d - c
    x = Fraction(numer, den)
    ins = rng.choice([
        f"分数{a}/{b}的分子和分母同时加上同一个数后等于{c}/{d}，这个数是多少？",
        f"把{a}/{b}的分子、分母加上同一个数，约分后得到{c}/{d}，加上的数是几？",
        f"一个分数是{a}/{b}，分子分母都加上同一个数后变成{c}/{d}，求这个数。",
        f"分数{a}/{b}的分子与分母同时加上几后，结果等于{c}/{d}？",
    ])
    lines = [
        f"{b} × {c} = {bc}",
        f"{a} × {d} = {ad}",
        f"{bc} - {ad} = {numer}",
        f"{d} - {c} = {den}",
        f"{numer} ÷ {den} = {num(x)}",
    ]
    return ins, lines, x


_reg("fraction_add_same", fraction_add_same)


# 22. numerator+denominator sum S, reduced form a/b -> numerator
def fraction_value_find(rng):
    a = rng.randint(2, 7)
    b = rng.randint(a + 1, 9)
    k = rng.randint(2, 9)
    S = (a + b) * k
    numer = a * k
    ins = rng.choice([
        f"一个分数的分子与分母之和是{S}，约分后等于{a}/{b}，这个分数的分子是多少？",
        f"一个分数约分后是{a}/{b}，原来分子与分母的和是{S}，原来的分子是几？",
        f"某分数的分子加分母等于{S}，化成最简分数是{a}/{b}，分子是多少？",
        f"一个分数的分子和分母相加得{S}，约分后为{a}/{b}，它的分子是多少？",
    ])
    lines = [
        f"{a} + {b} = {a + b}",
        f"{S} ÷ {a + b} = {k}",
        f"{a} × {k} = {numer}",
    ]
    return ins, lines, numer


_reg("fraction_value_find", fraction_value_find)


# 23. speed ratio + distance difference at meeting -> total distance
def ratio_speed_meet(rng):
    a = rng.randint(2, 6)
    b = rng.randint(a + 1, 9)
    k = rng.randint(5, 30)
    x = (b - a) * k
    per = k
    total = (a + b) * per
    ins = rng.choice([
        f"甲、乙两车的速度比是{a}:{b}，两车同时相向出发，相遇时乙车比甲车多行{x}千米，两地相距多少千米？",
        f"甲、乙两人速度比为{a}:{b}，同时从两地相向而行，相遇时甲比乙少走{x}米，两地相距多少米？",
        f"快船和慢船的速度比是{a}:{b}，两船同时相向开出，相遇时快船比慢船多行{x}千米，两港相距多少千米？",
        f"甲、乙步行速度比是{a}:{b}，两人同时出发相向而行，相遇时乙比甲多走{x}米，全程多少米？",
    ])
    lines = [
        f"{b} - {a} = {b - a}",
        f"{x} ÷ {b - a} = {per}",
        f"{a} + {b} = {a + b}",
        f"{a + b} × {per} = {total}",
    ]
    return ins, lines, total


_reg("ratio_speed_meet", ratio_speed_meet)


# 24. rectangle length:width ratio + perimeter -> area
def rectangle_ratio_area(rng):
    a = rng.randint(3, 7)
    b = rng.randint(2, a - 1)
    k = rng.randint(2, 8)
    C = 2 * (a + b) * k
    per = k
    l = a * per
    w = b * per
    area = l * w
    ins = rng.choice([
        f"一块长方形菜地，长与宽的比是{a}:{b}，周长是{C}米，这块菜地的面积是多少平方米？",
        f"长方形操场长和宽的比是{a}:{b}，周长{C}米，面积是多少平方米？",
        f"一个长方形花坛，长宽比为{a}:{b}，周长是{C}米，它的面积是多少平方米？",
        f"一块长方形地的周长是{C}米，长与宽之比为{a}:{b}，面积是多少平方米？",
    ])
    lines = [
        f"{C} ÷ 2 = {C // 2}米",
        f"{C // 2} ÷ {a + b} = {per}米",
        f"{a} × {per} = {l}米",
        f"{b} × {per} = {w}米",
        f"{l} × {w} = {area}平方米",
    ]
    return ins, lines, area


_reg("rectangle_ratio_area", rectangle_ratio_area)


# 25. triangle side ratio + perimeter -> longest side
def triangle_ratio_side(rng):
    a = rng.randint(2, 5)
    b = rng.randint(2, 6)
    c = rng.randint(2, 7)
    s = a + b + c
    k = rng.randint(2, 9)
    P = s * k
    hi = max(a, b, c)
    side = hi * k
    ins = rng.choice([
        f"一个三角形三条边的长度比是{a}:{b}:{c}，周长是{P}厘米，最长的边是多少厘米？",
        f"用一根长{P}厘米的铁丝围成一个三角形，三边长度比为{a}:{b}:{c}，最长边多少厘米？",
        f"三角形三边之比是{a}:{b}:{c}，周长为{P}厘米，最长的边长多少厘米？",
        f"一块三角形菜地，三条边的比是{a}:{b}:{c}，周长是{P}米，最长的边是多少米？",
    ])
    lines = [
        f"{a} + {b} + {c} = {s}",
        f"{P} ÷ {s} = {k}",
        f"{hi} × {k} = {side}",
    ]
    return ins, lines, side


_reg("triangle_ratio_side", triangle_ratio_side)


# 26. cuboid edge ratio + total edge length -> volume
def cuboid_ratio_volume(rng):
    a = rng.randint(2, 5)
    b = rng.randint(1, 4)
    c = rng.randint(1, 3)
    s = a + b + c
    k = rng.randint(2, 6)
    L = 4 * s * k
    la = a * k
    lb = b * k
    lc = c * k
    vol = la * lb * lc
    ins = rng.choice([
        f"一个长方体长、宽、高的比是{a}:{b}:{c}，所有棱长的和是{L}分米，它的体积是多少立方分米？",
        f"长方体的长宽高之比为{a}:{b}:{c}，棱长总和是{L}厘米，体积是多少立方厘米？",
        f"一个长方体模型，长宽高的比是{a}:{b}:{c}，棱长之和为{L}分米，体积是多少？",
        f"长方体长、宽、高的比是{a}:{b}:{c}，全部棱长共{L}米，体积是多少立方米？",
    ])
    lines = [
        f"{L} ÷ 4 = {L // 4}",
        f"{L // 4} ÷ {s} = {k}",
        f"{a} × {k} = {la}",
        f"{b} × {k} = {lb}",
        f"{c} × {k} = {lc}",
        f"{la} × {lb} × {lc} = {vol}",
    ]
    return ins, lines, vol


_reg("cuboid_ratio_volume", cuboid_ratio_volume)


# 27. salt:water ratio -> salt amount, then add water -> concentration
def conc_ratio_water(rng):
    a = rng.randint(1, 5)
    b = rng.randint(2, 8)
    k = rng.randint(10, 60)
    s = (a + b) * k
    salt = a * k
    x = rng.randint(2, 20) * 10
    total = s + x
    pct = Fraction(salt * 100, total)
    ins = rng.choice([
        f"一杯盐水中盐与水的比是{a}:{b}，盐水共{s}克，再加入{x}克水后，盐占盐水的百分之几？",
        f"一种盐水里盐和水的比为{a}:{b}，共{s}克，加入{x}克水后，含盐率是多少？",
        f"盐水中盐与水的比是{a}:{b}，重{s}克，又加入{x}克水，这时盐占盐水的百分之几？",
        f"一瓶盐水重{s}克，其中盐与水的比是{a}:{b}，再加入{x}克水，含盐率变为多少？",
    ])
    lines = [
        f"{a} + {b} = {a + b}",
        f"{s} ÷ {a + b} = {k}",
        f"{a} × {k} = {salt}克",
        f"{s} + {x} = {total}克",
        f"{salt} ÷ {total} × 100 = {num(pct)}",
    ]
    return ins, lines, pct


_reg("conc_ratio_water", conc_ratio_water)


# 28. three investors weighted by money x time -> one share
def three_invest_weighted(rng):
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    c = rng.randint(2, 9)
    m1 = rng.randint(2, 6)
    m2 = rng.randint(2, 6)
    m3 = rng.randint(2, 6)
    wa = a * m1
    wb = b * m2
    wc = c * m3
    ws = wa + wb + wc
    k = rng.randint(2, 15)
    profit = ws * k
    per = k
    share = wa * per
    ins = rng.choice([
        f"甲投资{a}万元{m1}个月，乙投资{b}万元{m2}个月，丙投资{c}万元{m3}个月，共获利{profit}万元，按投资金额乘时间分配，甲分得多少万元？",
        f"三人合伙，甲出{a}万元经营{m1}个月，乙出{b}万元经营{m2}个月，丙出{c}万元经营{m3}个月，盈利{profit}万元按出资乘时间分配，甲得多少万元？",
        f"甲、乙、丙分别投入{a}、{b}、{c}万元，时间分别为{m1}、{m2}、{m3}个月，年终盈利{profit}万元，按资金乘时间的比分配，甲分得多少万元？",
        f"甲投资{a}万元用了{m1}个月，乙投资{b}万元用了{m2}个月，丙投资{c}万元用了{m3}个月，共赢利{profit}万元，按投资乘时间分配，甲应得多少万元？",
    ])
    lines = [
        f"{a} × {m1} = {wa}",
        f"{b} × {m2} = {wb}",
        f"{c} × {m3} = {wc}",
        f"{wa} + {wb} + {wc} = {ws}",
        f"{profit} ÷ {ws} = {per}万元",
        f"{wa} × {per} = {share}万元",
    ]
    return ins, lines, share


_reg("three_invest_weighted", three_invest_weighted)


# 29. A is a/b of B, B gives A x then equal -> B's original
def fraction_give_equal(rng):
    a = rng.randint(2, 5)
    b = rng.randint(a + 1, 9)
    d = b - a
    k = rng.randint(2, 12)
    x = d * k
    B = 2 * k * b
    ins = rng.choice([
        f"甲的钱数是乙的{a}/{b}，乙给甲{x}元后两人钱数相等，乙原来有多少元？",
        f"甲堆货物是乙堆的{a}/{b}，从乙堆运{x}吨到甲堆后两堆相等，乙堆原有多少吨？",
        f"小明的邮票是小红的{a}/{b}，小红给小明{x}张后两人一样多，小红原来有多少张？",
        f"甲仓存粮是乙仓的{a}/{b}，乙仓运{x}吨到甲仓后两仓相等，乙仓原来存粮多少吨？",
    ])
    lines = [
        f"{b} - {a} = {d}",
        f"{x} × 2 = {2 * x}",
        f"{2 * x} × {b} = {2 * x * b}",
        f"{2 * x * b} ÷ {d} = {B}",
    ]
    return ins, lines, B


_reg("fraction_give_equal", fraction_give_equal)


# 30. A gives B p% of own money, then equal; B has b -> A's original
def pct_transfer_equal(rng):
    p = rng.choice([10, 15, 20, 25, 30, 35, 40])
    k = rng.randint(2, 20)
    b = (100 - 2 * p) * k
    A = 100 * k
    transfer = A * p // 100
    a_after = A - transfer
    b_after = b + transfer
    ins = rng.choice([
        f"甲把自己钱数的{p}%给乙后，两人钱数相等，已知乙原有{b}元，甲原来有多少元？",
        f"从甲仓运出{p}%的粮食到乙仓后两仓相等，乙仓原有{b}吨，甲仓原有多少吨？",
        f"小明把自己邮票的{p}%送给小红后两人邮票数相等，小红原有{b}张，小明原来有多少张？",
        f"甲拿出自己钱的{p}%给乙，两人的钱就一样多，乙原有{b}元，甲原有多少元？",
    ])
    lines = [
        f"100 - {2 * p} = {100 - 2 * p}",
        f"{A} × {p}/100 = {transfer}",
        f"{A} - {transfer} = {a_after}",
        f"{b} + {transfer} = {b_after}",
        f"{b} ÷ ({100 - 2 * p}/100) = {A}",
    ]
    return ins, lines, A


_reg("pct_transfer_equal", pct_transfer_equal)


# 31. two baskets sell different fractions, remainders equal; B=y -> A
def fraction_equal_remainder(rng):
    a = rng.randint(3, 6)
    b = rng.randint(3, 7)
    for _ in range(50):
        if a != b:
            break
        b = rng.randint(3, 7)
    k = rng.randint(2, 10)
    y = b * (a - 1) * k
    rem_b = (a - 1) * (b - 1) * k
    A = a * (b - 1) * k
    ins = rng.choice([
        f"甲、乙两筐苹果，甲筐卖出1/{a}，乙筐卖出1/{b}后，两筐剩下的苹果相等，乙筐原有{y}千克，甲筐原有多少千克？",
        f"两堆煤，甲堆用去1/{a}，乙堆用去1/{b}后余下的相等，乙堆原有{y}吨，甲堆原有多少吨？",
        f"甲、乙各有一笔钱，甲花掉1/{a}，乙花掉1/{b}后两人剩下的相等，乙原有{y}元，甲原有多少元？",
        f"书店有两种书，甲种卖出1/{a}，乙种卖出1/{b}后剩下的本数相等，乙种原有{y}本，甲种原有多少本？",
    ])
    lines = [
        f"{y} × ({b} - 1)/{b} = {num(Fraction(y * (b - 1), b))}",
        f"{num(Fraction(y * (b - 1), b))} × {a} = {rem_b * a}",
        f"{rem_b * a} ÷ ({a} - 1) = {A}",
    ]
    return ins, lines, A


_reg("fraction_equal_remainder", fraction_equal_remainder)


# 32. three people buy together: A pays 1/a, B pays 1/b, C pays x -> total
def fraction_three_people(rng):
    a = rng.choice([3, 4, 5, 6])
    b = rng.choice([3, 4, 5, 6])
    for _ in range(50):
        if a * b - a - b > 0:
            break
        b = rng.choice([3, 4, 5, 6])
    rest = a * b - a - b
    k = rng.randint(2, 12)
    x = rest * k
    total = a * b * k
    ins = rng.choice([
        f"甲、乙、丙三人合买一件物品，甲出总价的1/{a}，乙出总价的1/{b}，丙出了{x}元，这件物品多少元？",
        f"三人合买一台机器，甲付1/{a}，乙付1/{b}，丙付{x}元，这台机器共多少元？",
        f"一批货物由三人合运，甲运了1/{a}，乙运了1/{b}，丙运了{x}吨，这批货物共多少吨？",
        f"甲、乙、丙合修一段路，甲修1/{a}，乙修1/{b}，丙修{x}米，这段路长多少米？",
    ])
    lines = [
        f"{a} × {b} = {a * b}",
        f"{a * b} - {a} - {b} = {rest}",
        f"{x} × {a * b} = {x * a * b}",
        f"{x * a * b} ÷ {rest} = {total}",
    ]
    return ins, lines, total


_reg("fraction_three_people", fraction_three_people)


# 33. A+B together t days, A alone a days -> B alone
def work_b_alone(rng):
    a = rng.randint(8, 20)
    t = rng.randint(3, a - 2)
    rt = Fraction(1, t)
    ra = Fraction(1, a)
    rb = rt - ra
    b = Fraction(1, rb)
    ins = rng.choice([
        f"一项工程，甲、乙两队合作{t}天完成，甲队单独做{a}天完成，乙队单独做多少天完成？",
        f"修一条路，甲、乙合修{t}天完成，甲单独修{a}天完成，乙单独修要多少天？",
        f"一批零件，师徒合作{t}天完成，师傅单独做{a}天完成，徒弟单独做要多少天？",
        f"一个水池，甲、乙两管同开{t}小时注满，单开甲管{a}小时注满，单开乙管多少小时注满？",
    ])
    lines = [
        f"1 ÷ {t} = {num(rt)}",
        f"1 ÷ {a} = {num(ra)}",
        f"{num(rt)} - {num(ra)} = {num(rb)}",
        f"1 ÷ ({num(rb)}) = {num(b)}",
    ]
    return ins, lines, b


_reg("work_b_alone", work_b_alone)


# 34. A alone a, B alone b -> time to finish fraction f together
def work_fraction_completion(rng):
    a = rng.choice([8, 10, 12, 15, 20])
    b = rng.choice([6, 8, 10, 12])
    f = rng.choice([Fraction(1, 2), Fraction(2, 3), Fraction(3, 4)])
    ra = Fraction(1, a)
    rb = Fraction(1, b)
    rate = ra + rb
    t = Fraction(f, rate)
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成，乙单独做{b}天完成，两队合作完成全部工程的{num(f)}需要多少天？",
        f"修一条路，甲队单独修{a}天完成，乙队单独修{b}天完成，两队合修完成{num(f)}要多少天？",
        f"打一份稿件，甲单独打{a}小时完成，乙单独打{b}小时完成，两人合打完成{num(f)}需多少小时？",
        f"一批货物，甲车单独运{a}天运完，乙车单独运{b}天运完，两车合运这批货物的{num(f)}要多少天？",
    ])
    lines = [
        f"1 ÷ {a} = {num(ra)}",
        f"1 ÷ {b} = {num(rb)}",
        f"{num(ra)} + {num(rb)} = {num(rate)}",
        f"{num(f)} ÷ ({num(rate)}) = {num(t)}",
    ]
    return ins, lines, t


_reg("work_fraction_completion", work_fraction_completion)


# 35. efficiency ratio a:b, together t days -> A alone
def work_efficiency_ratio(rng):
    triples = []
    for aa in range(2, 7):
        for bb in range(1, 7):
            if bb == aa:
                continue
            for tt in (4, 5, 6, 8, 10, 12, 15):
                if tt * bb % aa == 0:
                    triples.append((aa, bb, tt))
    a, b, t = rng.choice(triples)
    together = Fraction(1, t)
    s = a + b
    per = Fraction(together, s)
    ra = a * per
    days = Fraction(1, ra)
    ins = rng.choice([
        f"一项工程，甲、乙两队工作效率的比是{a}:{b}，两队合作{t}天完成，甲队单独做多少天完成？",
        f"加工一批零件，师徒效率比为{a}:{b}，两人合作{t}天完成，师傅单独做要多少天？",
        f"修一条路，甲、乙两队的效率比是{a}:{b}，合修{t}天完成，甲队单独修需多少天？",
        f"一个水池，甲、乙两管注水效率比为{a}:{b}，同开{t}小时注满，单开甲管多少小时注满？",
    ])
    lines = [
        f"1 ÷ {t} = {num(together)}",
        f"{a} + {b} = {s}",
        f"{num(together)} ÷ {s} = {num(per)}",
        f"{a} × {num(per)} = {num(ra)}",
        f"1 ÷ ({num(ra)}) = {num(days)}",
    ]
    return ins, lines, days


_reg("work_efficiency_ratio", work_efficiency_ratio)


# 36. pairwise cooperation times -> three together time
def work_three_pair(rng):
    times = [6, 8, 10, 12, 15, 20]
    t1 = rng.choice(times)
    t2 = rng.choice(times)
    t3 = rng.choice(times)
    for _ in range(50):
        if t1 != t2 and t2 != t3 and t1 != t3:
            break
        t2 = rng.choice(times)
        t3 = rng.choice(times)
    r1 = Fraction(1, t1)
    r2 = Fraction(1, t2)
    r3 = Fraction(1, t3)
    s = r1 + r2 + r3
    half = Fraction(s, 2)
    t = Fraction(1, half)
    ins = rng.choice([
        f"一项工程，甲、乙合作{t1}天完成，乙、丙合作{t2}天完成，甲、丙合作{t3}天完成，三人合作多少天完成？",
        f"修一条路，甲乙合修{t1}天完成，乙丙合修{t2}天完成，甲丙合修{t3}天完成，三队合修多少天完成？",
        f"一批零件，师徒合作{t1}天完成，师傅与另一徒弟合作{t2}天完成，两个徒弟合作{t3}天完成，三人合作多少天完成？",
        f"一个水池，甲乙两管同开{t1}小时注满，乙丙同开{t2}小时注满，甲丙同开{t3}小时注满，三管同开多少小时注满？",
    ])
    lines = [
        f"1 ÷ {t1} = {num(r1)}",
        f"1 ÷ {t2} = {num(r2)}",
        f"1 ÷ {t3} = {num(r3)}",
        f"{num(r1)} + {num(r2)} + {num(r3)} = {num(s)}",
        f"{num(s)} ÷ 2 = {num(half)}",
        f"1 ÷ ({num(half)}) = {num(t)}",
    ]
    return ins, lines, t


_reg("work_three_pair", work_three_pair)


# 37. A alone a, B alone b, A rests c days -> total time
def work_rest_day(rng):
    a = rng.choice([10, 12, 15, 20, 24])
    b = rng.choice([6, 8, 10, 12])
    c = rng.choice([1, 2, 3])
    for _ in range(50):
        if c < b:
            break
        c = rng.choice([1, 2, 3])
    rb = Fraction(1, b)
    done = rb * c
    rem = 1 - done
    ra = Fraction(1, a)
    rate = ra + rb
    t = Fraction(rem, rate)
    total = t + c
    ins = rng.choice([
        f"一项工程，甲单独做{a}天完成，乙单独做{b}天完成，两队合作期间甲休息了{c}天，完成这项工程共需多少天？",
        f"修一条路，甲队单独修{a}天完成，乙队单独修{b}天完成，合修时甲队中途休息{c}天，修完共需多少天？",
        f"一批零件，师傅单独做{a}天完成，徒弟单独做{b}天完成，合作中师傅休息{c}天，完成任务共需多少天？",
        f"一个水池，甲管单独注满需{a}小时，乙管需{b}小时，两管同开期间甲管停了{c}小时，注满共需多少小时？",
    ])
    lines = [
        f"1 ÷ {b} = {num(rb)}",
        f"{num(rb)} × {c} = {num(done)}",
        f"1 - {num(done)} = {num(rem)}",
        f"1 ÷ {a} = {num(ra)}",
        f"{num(ra)} + {num(rb)} = {num(rate)}",
        f"{num(rem)} ÷ ({num(rate)}) = {num(t)}",
        f"{num(t)} + {c} = {num(total)}",
    ]
    return ins, lines, total


_reg("work_rest_day", work_rest_day)


# 38. one machine a hours, add b identical machines -> time
def work_machines(rng):
    b = rng.choice([1, 2, 3, 4, 5, 7])
    k = rng.randint(2, 10)
    a = (1 + b) * k
    rate = Fraction(1, a)
    n = 1 + b
    nrate = rate * n
    t = Fraction(1, nrate)
    ins = rng.choice([
        f"一台机器加工一批零件要{a}小时完成，增加{b}台同样的机器后，多少小时可以完成？",
        f"一台收割机收完一片麦子要{a}小时，再增加{b}台同样的收割机，几小时能收完？",
        f"一个工人完成一项任务要{a}小时，增加{b}个效率相同的工人后，多少小时完成？",
        f"一台抽水机抽完一池水要{a}小时，增加{b}台同样的抽水机，几小时可以抽完？",
    ])
    lines = [
        f"1 ÷ {a} = {num(rate)}",
        f"1 + {b} = {n}",
        f"{num(rate)} × {n} = {num(nrate)}",
        f"1 ÷ ({num(nrate)}) = {num(t)}",
    ]
    return ins, lines, t


_reg("work_machines", work_machines)


# 39. meet with one party resting t0 hours -> total time
def travel_meet_rest(rng):
    v1 = rng.randint(40, 80)
    v2 = rng.randint(40, 80)
    t0 = rng.choice([1, 2])
    k = rng.randint(2, 6)
    D = v2 * t0 + (v1 + v2) * k
    head = v2 * t0
    rem = D - head
    s = v1 + v2
    t = Fraction(rem, s)
    total = t + t0
    ins = rng.choice([
        f"甲、乙两地相距{D}千米，甲车每小时行{v1}千米，乙车每小时行{v2}千米，两车相向而行，途中甲车休息了{t0}小时，从出发到相遇共需多少小时？",
        f"两地相距{D}千米，小明每小时走{v1}千米，小红每小时走{v2}千米，两人同时相向出发，小明中途休息{t0}小时，几小时后相遇？",
        f"A、B两城相距{D}千米，快车每小时行{v1}千米，慢车每小时行{v2}千米，同时相向开出，慢车中途停留{t0}小时，几小时后相遇？",
        f"甲、乙两人相距{D}千米，甲每分钟走{v1}米，乙每分钟走{v2}米，相向而行，甲途中休息{t0}分钟，多久后相遇？",
    ])
    lines = [
        f"{v2} × {t0} = {head}",
        f"{D} - {head} = {rem}",
        f"{v1} + {v2} = {s}",
        f"{rem} ÷ {s} = {num(t)}",
        f"{num(t)} + {t0} = {num(total)}",
    ]
    return ins, lines, total


_reg("travel_meet_rest", travel_meet_rest)


# 40. two legs of different distances and speeds -> average speed
def travel_two_leg_dist(rng):
    v1 = rng.randint(30, 60)
    v2 = rng.randint(50, 90)
    k1 = rng.randint(2, 5)
    k2 = rng.randint(2, 5)
    D1 = v1 * k1
    D2 = v2 * k2
    t1 = k1
    t2 = k2
    total = D1 + D2
    tt = t1 + t2
    avg = Fraction(total, tt)
    ins = rng.choice([
        f"一辆汽车先以每小时{v1}千米的速度行驶{D1}千米，又以每小时{v2}千米的速度行驶{D2}千米，全程的平均速度是多少千米/时？",
        f"小明骑车先走{D1}千米，速度是每小时{v1}千米；再走{D2}千米，速度是每小时{v2}千米，他全程的平均速度是多少？",
        f"一列火车先以每小时{v1}千米的速度行了{D1}千米，再以每小时{v2}千米的速度行了{D2}千米，平均每小时行多少千米？",
        f"一艘船先以每小时{v1}千米的速度航行{D1}千米，又以每小时{v2}千米的速度航行{D2}千米，全程平均速度是多少千米/时？",
    ])
    lines = [
        f"{D1} ÷ {v1} = {t1}",
        f"{D2} ÷ {v2} = {t2}",
        f"{D1} + {D2} = {total}",
        f"{t1} + {t2} = {tt}",
        f"{total} ÷ {tt} = {num(avg)}",
    ]
    return ins, lines, avg


_reg("travel_two_leg_dist", travel_two_leg_dist)


# 41. boat round trip between two ports -> total time
def boat_roundtrip_time(rng):
    pairs = [(15, 5, 20), (18, 6, 24), (25, 5, 60), (21, 7, 28),
              (16, 4, 60), (24, 6, 90), (12, 4, 16), (20, 5, 75),
              (27, 9, 36), (14, 7, 21), (10, 5, 15), (30, 6, 72),
              (18, 9, 27), (15, 3, 36), (22, 11, 33), (16, 8, 24),
              (12, 6, 18), (24, 8, 32)]
    b, c, lcm = rng.choice(pairs)
    k = rng.randint(1, 3)
    D = lcm * k
    down = b + c
    up = b - c
    t1 = Fraction(D, down)
    t2 = Fraction(D, up)
    total = t1 + t2
    ins = rng.choice([
        f"一艘船在静水中每小时行{b}千米，水流速度每小时{c}千米，往返相距{D}千米的两港一次共需多少小时？",
        f"轮船在静水中速度为每小时{b}千米，水速每小时{c}千米，往返{D}千米需要多少小时？",
        f"一条河水流速度每小时{c}千米，船在静水中每小时行{b}千米，往返相距{D}千米的两个码头共需多少小时？",
        f"船在静水中每小时行{b}千米，水流每小时{c}千米，往返航程{D}千米共要多少小时？",
    ])
    lines = [
        f"{b} + {c} = {down}",
        f"{b} - {c} = {up}",
        f"{D} ÷ {down} = {num(t1)}",
        f"{D} ÷ {up} = {num(t2)}",
        f"{num(t1)} + {num(t2)} = {num(total)}",
    ]
    return ins, lines, total


_reg("boat_roundtrip_time", boat_roundtrip_time)


# 42. downstream/upstream times for same distance -> still-water speed
def boat_find_speed(rng):
    t1 = rng.choice([3, 4, 5])
    t2 = rng.choice([5, 6, 8])
    for _ in range(50):
        if t2 > t1:
            break
        t2 = rng.choice([5, 6, 8])
    import math
    lcm = t1 * t2 // math.gcd(t1, t2)
    k = rng.randint(2, 8)
    D = lcm * k
    vd = Fraction(D, t1)
    vu = Fraction(D, t2)
    b = Fraction(vd + vu, 2)
    ins = rng.choice([
        f"一艘船顺水航行{D}千米用了{t1}小时，逆水航行同样的距离用了{t2}小时，船在静水中的速度是多少千米/时？",
        f"轮船顺水行{D}千米需{t1}小时，逆水行{D}千米需{t2}小时，求轮船在静水中的速度。",
        f"一条船顺流航行{D}千米要{t1}小时，逆流航行{D}千米要{t2}小时，船在静水中每小时行多少千米？",
        f"甲、乙两港相距{D}千米，一艘船顺水而行用{t1}小时，逆水而行用{t2}小时，船在静水中的速度是多少？",
    ])
    lines = [
        f"{D} ÷ {t1} = {num(vd)}",
        f"{D} ÷ {t2} = {num(vu)}",
        f"{num(vd)} + {num(vu)} = {num(vd + vu)}",
        f"{num(vd + vu)} ÷ 2 = {num(b)}",
    ]
    return ins, lines, b


_reg("boat_find_speed", boat_find_speed)


# 43. train passes a signal in t1 s, a bridge in t2 s -> train length
def train_pole_bridge(rng):
    t1 = rng.choice([5, 10, 15, 20])
    d = rng.choice([5, 10, 15, 20])
    t2 = t1 + d
    v = rng.choice([10, 15, 20, 25, 30])
    B = d * v
    L = v * t1
    ins = rng.choice([
        f"一列火车通过一个信号灯用了{t1}秒，以同样的速度通过一座长{B}米的大桥用了{t2}秒，这列火车长多少米？",
        f"火车通过一根电线杆用{t1}秒，通过长{B}米的隧道用{t2}秒，火车每秒行{v}米，火车长多少米？",
        f"一列火车以每秒{v}米的速度行驶，通过一个路标用了{t1}秒，通过一座长{B}米的桥用了{t2}秒，火车长多少米？",
        f"火车经过一个信号灯用了{t1}秒，穿过长{B}米的山洞用了{t2}秒，车速是每秒{v}米，火车长多少米？",
    ])
    lines = [
        f"{t2} - {t1} = {d}",
        f"{B} ÷ {d} = {v}",
        f"{v} × {t1} = {L}",
    ]
    return ins, lines, L


_reg("train_pole_bridge", train_pole_bridge)


# 44. pool at 1/d, add x liters reaches 1/e -> capacity
def pool_capacity_frac(rng):
    d = rng.choice([3, 4, 5, 6])
    e = rng.choice([2, 3, 4])
    for _ in range(50):
        if e < d:
            break
        e = rng.choice([2, 3, 4])
    k = rng.randint(10, 60)
    x = (d - e) * k
    cap = d * e * k
    ins = rng.choice([
        f"一个水池装了1/{d}的水，再注入{x}升水后正好装到1/{e}，这个水池的容量是多少升？",
        f"水箱里有水是容量的1/{d}，加入{x}升后达到容量的1/{e}，水箱能装多少升？",
        f"一个水桶装了1/{d}桶水，再倒入{x}升就装到1/{e}，这个水桶能装多少升？",
        f"水池蓄水占容量的1/{d}，再放进{x}升水后占容量的1/{e}，水池容量是多少升？",
    ])
    lines = [
        f"{d} - {e} = {d - e}",
        f"{x} × {d} = {x * d}",
        f"{x * d} × {e} = {x * d * e}",
        f"{x * d * e} ÷ {d - e} = {cap}",
    ]
    return ins, lines, cap


_reg("pool_capacity_frac", pool_capacity_frac)


# 45. at meeting A traveled a/b of total, B traveled x -> total
def travel_meet_fraction(rng):
    a = rng.randint(2, 4)
    b = rng.randint(a + 1, 7)
    k = rng.randint(10, 60)
    x = (b - a) * k
    total = b * k
    jia = a * k
    ins = rng.choice([
        f"甲、乙两车同时从两地相向而行，相遇时甲车行了全程的{a}/{b}，乙车行了{x}千米，两地相距多少千米？",
        f"小明和小红从两地同时出发相向而行，相遇时小明走了全程的{a}/{b}，小红走了{x}米，全程多少米？",
        f"两艘轮船从两港同时相向开出，相遇时快船行了全程的{a}/{b}，慢船行了{x}千米，两港相距多少千米？",
        f"甲、乙两人相向而行，相遇时甲走了全程的{a}/{b}，乙走了{x}千米，全程是多少千米？",
    ])
    lines = [
        f"{b} - {a} = {b - a}",
        f"{total} × {a}/{b} = {jia}",
        f"{total} - {jia} = {x}",
        f"{x} × {b} = {x * b}",
        f"{x * b} ÷ {b - a} = {total}",
    ]
    return ins, lines, total


_reg("travel_meet_fraction", travel_meet_fraction)


# 46. average of n numbers, add one more -> new average
def average_add_one(rng):
    n = rng.randint(3, 6)
    a = rng.randint(70, 95)
    b = rng.randint(a + 1, a + 6)
    x = (n + 1) * b - n * a
    total = n * a
    newtotal = total + x
    avg = b
    ins = rng.choice([
        f"{n}个数的平均数是{a}，再加上{x}后，这{n + 1}个数的平均数是多少？",
        f"小明前{n}次数学测验平均{a}分，这次考了{x}分，他{n + 1}次测验的平均分是多少？",
        f"{n}筐苹果平均每筐{a}千克，又运来一筐{x}千克，现在平均每筐多少千克？",
        f"一组{n}人的平均身高是{a}厘米，新加入一名身高{x}厘米的同学后，平均身高是多少厘米？",
    ])
    lines = [
        f"{n} × {a} = {total}",
        f"{total} + {x} = {newtotal}",
        f"{n} + 1 = {n + 1}",
        f"{newtotal} ÷ {n + 1} = {avg}",
    ]
    return ins, lines, avg


_reg("average_add_one", average_add_one)


# 47. average of 5 numbers, remove one, average of remaining 4 -> removed
def average_remove_one(rng):
    a = rng.randint(70, 95)
    b = rng.randint(70, 95)
    for _ in range(50):
        if 5 * a > 4 * b:
            break
        b = rng.randint(70, 95)
    sum5 = 5 * a
    sum4 = 4 * b
    removed = sum5 - sum4
    ins = rng.choice([
        f"5个数的平均数是{a}，去掉一个数后，余下4个数的平均数是{b}，去掉的数是多少？",
        f"小明5次测验平均{a}分，去掉一次成绩后，剩下4次平均{b}分，去掉的那次是多少分？",
        f"5筐水果平均每筐{a}千克，卖出一筐后，剩下4筐平均每筐{b}千克，卖出的那筐重多少千克？",
        f"5个同学的平均体重是{a}千克，走了一个同学后，剩下4人的平均体重是{b}千克，走了的同学体重多少千克？",
    ])
    lines = [
        f"5 × {a} = {sum5}",
        f"4 × {b} = {sum4}",
        f"{sum5} - {sum4} = {removed}",
    ]
    return ins, lines, removed


_reg("average_remove_one", average_remove_one)


# 48. average rises after this test -> which test number is this
def average_find_count(rng):
    n = rng.randint(2, 6)
    a = rng.randint(80, 92)
    b = rng.randint(a + 2, a + 8)
    x = (n + 1) * b - n * a
    s1 = n * a
    s2 = (n + 1) * b
    ins = rng.choice([
        f"小明前几次数学测验的平均分是{a}分，这次考了{x}分后，平均分提高到{b}分，这次是第几次测验？",
        f"小红前几次跳绳平均每次跳{a}下，这次跳了{x}下，平均成绩提高到{b}下，这次是第几次？",
        f"小华前几次英语测验平均{a}分，这次得{x}分后平均分变成{b}分，这是他第几次测验？",
        f"一名运动员前几次训练平均成绩是{a}秒，这次用了{x}秒后平均成绩提高到{b}秒，这是第几次训练？",
    ])
    lines = [
        f"{n} × {a} = {s1}",
        f"{n + 1} × {b} = {s2}",
        f"{s2} - {s1} = {x}",
        f"{n} + 1 = {n + 1}",
    ]
    return ins, lines, n + 1


_reg("average_find_count", average_find_count)


# 49. circle area x unit cost (pi = 3.14)
def circle_area_cost(rng):
    r = rng.choice([10, 20, 30, 40, 50])
    c = rng.randint(10, 50)
    r2 = r * r
    area = Fraction(314, 100) * r2
    total = area * c
    ins = rng.choice([
        f"一个圆形花坛的半径是{r}米，每平方米种花需要{c}元（π取3.14），种满花坛一共需要多少元？",
        f"圆形草坪半径{r}米，每平方米草皮{c}元（π取3.14），铺满草皮要多少元？",
        f"一个圆形水池半径{r}米，池底每平方米铺砖{c}元（π取3.14），铺完池底共需多少元？",
        f"一块圆形菜地半径{r}米，围起来种菜每平方米投入{c}元（π取3.14），共需多少元？",
    ])
    lines = [
        f"{r} × {r} = {r2}",
        f"3.14 × {r2} = {num(area)}",
        f"{num(area)} × {c} = {num(total)}元",
    ]
    return ins, lines, total


_reg("circle_area_cost", circle_area_cost)


# 50. circle circumference x laps
def circle_laps(rng):
    d = rng.choice([10, 20, 25, 30, 40, 50, 60, 75, 80, 100])
    k = rng.randint(2, 8)
    r = Fraction(d, 2)
    C = 2 * Fraction(314, 100) * r
    total = C * k
    ins = rng.choice([
        f"一个圆形花坛的直径是{d}米，小明沿花坛跑了{k}圈，一共跑了多少米？",
        f"圆形水池直径{d}米，绕水池走{k}圈是多少米？",
        f"一个圆形操场直径{d}米，小红跑了{k}圈，她跑了多少米？",
        f"圆形广场直径{d}米，沿广场散步{k}圈，一共走了多少米？",
    ])
    lines = [
        f"{d} ÷ 2 = {num(r)}",
        f"2 × 3.14 × {num(r)} = {num(C)}",
        f"{num(C)} × {k} = {num(total)}",
    ]
    return ins, lines, total


_reg("circle_laps", circle_laps)


# 51. ring area (outer circle minus inner circle)
def circle_ring(rng):
    R = rng.choice([20, 30, 40, 50, 60, 80])
    r = rng.choice([10, 20, 30, 40])
    for _ in range(50):
        if r < R:
            break
        r = rng.choice([10, 20, 30, 40])
    c = rng.randint(10, 30)
    R2 = R * R
    r2 = r * r
    diff = R2 - r2
    area = Fraction(314, 100) * diff
    total = area * c
    ins = rng.choice([
        f"一个圆形环岛，外半径{R}米，中间花坛半径{r}米（π取3.14），环岛每平方米绿化{c}元，绿化共需多少元？",
        f"圆形水池外半径{R}米，池内小岛半径{r}米（π取3.14），水面每平方米养护{c}元，共需多少元？",
        f"一个圆环，外圆半径{R}米，内圆半径{r}米（π取3.14），每平方米刷漆{c}元，刷漆共需多少元？",
        f"圆形广场半径{R}米，中央喷泉半径{r}米（π取3.14），其余部分每平方米铺砖{c}元，共需多少元？",
    ])
    lines = [
        f"{R} × {R} = {R2}",
        f"{r} × {r} = {r2}",
        f"{R2} - {r2} = {diff}",
        f"3.14 × {diff} = {num(area)}",
        f"{num(area)} × {c} = {num(total)}元",
    ]
    return ins, lines, total


_reg("circle_ring", circle_ring)


# 52. cylinder volume (pi = 3.14)
def cylinder_volume(rng):
    r = rng.choice([10, 20, 30, 40])
    h = rng.randint(2, 12)
    r2 = r * r
    base = Fraction(314, 100) * r2
    vol = base * h
    ins = rng.choice([
        f"一个圆柱形水池，底面半径{r}米，深{h}米（π取3.14），这个水池能装多少立方米的水？",
        f"圆柱形粮仓底面半径{r}米，高{h}米（π取3.14），粮仓的容积是多少立方米？",
        f"一个圆柱底面半径{r}米，高{h}米（π取3.14），它的体积是多少立方米？",
        f"圆柱形水桶底面半径{r}米，桶高{h}米（π取3.14），最多能装多少立方米的水？",
    ])
    lines = [
        f"{r} × {r} = {r2}",
        f"3.14 × {r2} = {num(base)}",
        f"{num(base)} × {h} = {num(vol)}",
    ]
    return ins, lines, vol


_reg("cylinder_volume", cylinder_volume)


# 53. cube surface area x cost
def cube_surface_cost(rng):
    a = rng.randint(3, 12)
    c = rng.randint(10, 50)
    a2 = a * a
    surf = 6 * a2
    total = surf * c
    ins = rng.choice([
        f"一个正方体木箱棱长{a}米，每平方米油漆{c}元，油漆表面共需多少元？",
        f"正方体水池棱长{a}米，池底和四壁贴瓷砖，每平方米{c}元，共需多少元？",
        f"一个正方体展台棱长{a}米，表面贴饰面板，每平方米{c}元，共需多少元？",
        f"正方体礼盒棱长{a}米，包装表面每平方米花纸{c}元，包装需要多少元？",
    ])
    lines = [
        f"{a} × {a} = {a2}",
        f"6 × {a2} = {surf}",
        f"{surf} × {c} = {total}元",
    ]
    return ins, lines, total


_reg("cube_surface_cost", cube_surface_cost)


# 54. cuboid: total edge length -> surface area
def cuboid_edge_surface(rng):
    a = rng.randint(2, 10)
    b = rng.randint(2, 10)
    c = rng.randint(2, 10)
    s = a + b + c
    edge = 4 * s
    ab = a * b
    bc = b * c
    ac = a * c
    pairs = ab + bc + ac
    surf = 2 * pairs
    ins = rng.choice([
        f"一个长方体长{a}米、宽{b}米、高{c}米，它的表面积是多少平方米？",
        f"长方体礼盒长{a}厘米、宽{b}厘米、高{c}厘米，表面积是多少平方厘米？",
        f"一个长方体木箱长{a}米、宽{b}米、高{c}米，做这个木箱至少需要多少平方米木板？",
        f"长方体水池长{a}米、宽{b}米、深{c}米，池底和四壁的面积是多少平方米？",
    ])
    lines = [
        f"{a} + {b} + {c} = {s}",
        f"4 × {s} = {edge}",
        f"{a} × {b} = {ab}",
        f"{b} × {c} = {bc}",
        f"{a} × {c} = {ac}",
        f"{ab} + {bc} + {ac} = {pairs}",
        f"2 × {pairs} = {surf}",
    ]
    return ins, lines, surf


_reg("cuboid_edge_surface", cuboid_edge_surface)


# 55. water displacement: rock volume
def volume_rock(rng):
    a = rng.randint(4, 10)
    b = rng.randint(3, 8)
    h1 = rng.randint(1, 4)
    rise = rng.randint(1, 3)
    h2 = h1 + rise
    base = a * b
    vol = base * rise
    ins = rng.choice([
        f"一个长方体水箱长{a}分米、宽{b}分米，水深{h1}分米，放入一块石头后水面上升到{h2}分米，石头的体积是多少立方分米？",
        f"长方体玻璃缸长{a}厘米、宽{b}厘米，原来水深{h1}厘米，放入石块后水深{h2}厘米，石块体积是多少立方厘米？",
        f"一个长方体容器长{a}分米、宽{b}分米，水面高{h1}分米，放入一块石头完全浸没后水面升到{h2}分米，石头体积是多少？",
        f"水箱长{a}厘米、宽{b}厘米，水深{h1}厘米，放进一块石头后水深变为{h2}厘米，这块石头的体积是多少立方厘米？",
    ])
    lines = [
        f"{h2} - {h1} = {rise}",
        f"{a} × {b} = {base}",
        f"{base} × {rise} = {vol}",
    ]
    return ins, lines, vol


_reg("volume_rock", volume_rock)


# 56. pour water from cuboid A into cube B -> depth
def volume_pour(rng):
    s = rng.choice([2, 3, 4, 5])
    a = s * rng.choice([1, 2, 3])
    b = s
    h1 = rng.randint(2, 6)
    base = a * b
    vol = base * h1
    s2 = s * s
    depth = Fraction(vol, s2)
    ins = rng.choice([
        f"甲水箱长{a}分米、宽{b}分米、水深{h1}分米，把水全部倒入棱长{s}分米的正方体乙水箱，乙水箱水深多少分米？",
        f"一个长方体容器长{a}厘米、宽{b}厘米，水深{h1}厘米，将水倒入棱长{s}厘米的正方体容器，水深多少厘米？",
        f"甲缸长{a}分米、宽{b}分米，装水深{h1}分米，把水全部倒进棱长{s}分米的正方体缸，水深多少分米？",
        f"长方体水箱长{a}厘米、宽{b}厘米、水深{h1}厘米，把水倒入棱长{s}厘米的正方体水箱，水面高多少厘米？",
    ])
    lines = [
        f"{a} × {b} = {base}",
        f"{base} × {h1} = {vol}",
        f"{s} × {s} = {s2}",
        f"{vol} ÷ {s2} = {num(depth)}",
    ]
    return ins, lines, depth


_reg("volume_pour", volume_pour)


# 57. stair carpet length
def stair_carpet(rng):
    n = rng.randint(4, 12)
    w = Fraction(rng.randint(2, 5), 10)
    h = Fraction(rng.randint(2, 6), 10)
    horiz = n * w
    vert = n * h
    total = horiz + vert
    ins = rng.choice([
        f"一段楼梯有{n}级台阶，每级宽{num(w)}米、高{num(h)}米，铺满楼梯至少需要多少米长的地毯？",
        f"商场楼梯共{n}级，每级台阶宽{num(w)}米、高{num(h)}米，铺地毯至少要多少米？",
        f"一栋楼的楼梯有{n}级，每级宽{num(w)}米、高{num(h)}米，给楼梯铺地毯，地毯至少长多少米？",
        f"公园台阶共{n}级，每级宽{num(w)}米、高{num(h)}米，从下到上铺地毯需要多少米？",
    ])
    lines = [
        f"{n} × {num(w)} = {num(horiz)}",
        f"{n} × {num(h)} = {num(vert)}",
        f"{num(horiz)} + {num(vert)} = {num(total)}",
    ]
    return ins, lines, total


_reg("stair_carpet", stair_carpet)


# 58. planting trees along a road, both sides, both ends
def planting_trees(rng):
    d = rng.choice([3, 4, 5, 6, 8])
    g = rng.randint(10, 40)
    L = d * g
    side = g + 1
    total = 2 * side
    ins = rng.choice([
        f"在一条长{L}米的公路两旁植树，每隔{d}米栽一棵，两端都栽，一共需要多少棵树苗？",
        f"一条路长{L}米，在路的两边每隔{d}米种一棵树，两头都种，共需多少棵树？",
        f"校园小路长{L}米，两侧每隔{d}米摆一盆花，两端都摆，一共要摆多少盆？",
        f"在长{L}米的跑道两旁插彩旗，每隔{d}米插一面，两端都插，共需多少面彩旗？",
    ])
    lines = [
        f"{L} ÷ {d} = {g}",
        f"{g} + 1 = {side}",
        f"{side} × 2 = {total}",
    ]
    return ins, lines, total


_reg("planting_trees", planting_trees)


# 59. sawing wood: time per cut
def sawing_time(rng):
    n = rng.randint(3, 6)
    k = rng.choice([2, 3, 4, 5])
    t = (n - 1) * k
    m = rng.randint(n + 1, n + 5)
    cuts2 = m - 1
    total = k * cuts2
    obj = rng.choice(["木头", "钢管", "竹竿", "绳子"])
    ins = rng.choice([
        f"把一根{obj}锯成{n}段需要{t}分钟，以同样的速度锯成{m}段需要多少分钟？",
        f"一根{obj}锯成{n}段用了{t}分钟，照这样计算，锯成{m}段要多少分钟？",
        f"工人把{obj}锯成{n}段花了{t}分钟，以同样的速度，锯成{m}段需要几分钟？",
        f"把一根{obj}平均锯成{n}段需{t}分钟，锯成{m}段需多少分钟？",
    ])
    lines = [
        f"{n} - 1 = {n - 1}",
        f"{t} ÷ {n - 1} = {k}",
        f"{m} - 1 = {cuts2}",
        f"{k} × {cuts2} = {total}",
    ]
    return ins, lines, total


_reg("sawing_time", sawing_time)


# 60. climbing stairs at constant speed
def stairs_time(rng):
    a = rng.randint(3, 8)
    k = rng.choice([5, 6, 9, 10, 12, 15])
    t = (a - 1) * k
    b = rng.randint(a + 1, a + 6)
    total = k * (b - 1)
    ins = rng.choice([
        f"小明从1楼走到{a}楼用了{t}秒，以同样的速度，他从1楼走到{b}楼需要多少秒？",
        f"小红从一楼上到{a}楼用了{t}秒，照这样的速度，上到{b}楼要用多少秒？",
        f"爸爸从1楼爬到{a}楼用了{t}秒，以同样的速度爬到{b}楼需要多少秒？",
        f"小丽从一楼走到{a}楼用了{t}秒，她以同样的速度继续走到{b}楼，还要多少秒？",
    ])
    lines = [
        f"{a} - 1 = {a - 1}",
        f"{t} ÷ {a - 1} = {k}",
        f"{b} - 1 = {b - 1}",
        f"{k} × {b - 1} = {total}",
    ]
    return ins, lines, total


_reg("stairs_time", stairs_time)


# 61. clock striking: intervals between strikes
def clock_strike(rng):
    a = rng.randint(3, 6)
    k = rng.choice([1, 2, 3, 4])
    t = (a - 1) * k
    b = rng.randint(a + 1, 12)
    total = k * (b - 1)
    ins = rng.choice([
        f"一座钟敲{a}下用了{t}秒，以同样的间隔，敲{b}下要用多少秒？",
        f"大钟敲{a}响用了{t}秒，照这样计算，敲{b}响需要多少秒？",
        f"时钟报时敲{a}下用了{t}秒，以同样的速度，敲{b}下用多少秒？",
        f"一座钟楼敲{a}下用{t}秒，敲{b}下需要多少秒？",
    ])
    lines = [
        f"{a} - 1 = {a - 1}",
        f"{t} ÷ {a - 1} = {k}",
        f"{b} - 1 = {b - 1}",
        f"{k} × {b - 1} = {total}",
    ]
    return ins, lines, total


_reg("clock_strike", clock_strike)


# 62. oil with bucket: use 1/d, weights before/after -> bucket weight
def oil_bucket(rng):
    d = rng.choice([3, 4, 5])
    k = rng.randint(2, 6)
    bucket = rng.randint(1, 8)
    A = k * d + bucket
    B = A - k
    oil = k * d
    ins = rng.choice([
        f"一桶油连桶重{A}千克，用去1/{d}后连桶重{B}千克，桶重多少千克？",
        f"一瓶油连瓶重{A}千克，用去1/{d}后连瓶重{B}千克，瓶重多少千克？",
        f"一桶水连桶重{A}千克，倒出1/{d}后连桶重{B}千克，桶重多少千克？",
        f"一箱苹果连箱重{A}千克，卖出1/{d}后连箱重{B}千克，箱子重多少千克？",
    ])
    lines = [
        f"{A} - {B} = {k}",
        f"{k} × {d} = {oil}",
        f"{A} - {oil} = {bucket}",
    ]
    return ins, lines, bucket


_reg("oil_bucket", oil_bucket)


# 63. add solute to a solution -> new concentration
def conc_add_solute(rng):
    s = rng.choice([100, 200, 300])
    c = rng.choice([5, 10, 15, 20])
    x = rng.choice([10, 20, 30, 50])
    salt = Fraction(s * c, 100)
    newsalt = salt + x
    total = s + x
    pct = Fraction(newsalt * 100, total)
    ins = rng.choice([
        f"有{s}克含盐{c}%的盐水，再加入{x}克盐，这时盐占盐水的百分之几？",
        f"一杯{s}克的盐水含盐{c}%，加入{x}克盐完全溶解后，含盐率是多少？",
        f"把{x}克盐加入{s}克含盐{c}%的盐水中，新盐水的含盐率是多少？",
        f"现有{s}克含盐{c}%的盐水，再加入{x}克盐，这时盐水的浓度是多少？",
    ])
    lines = [
        f"{s} × {c}/100 = {num(salt)}",
        f"{num(salt)} + {x} = {num(newsalt)}",
        f"{s} + {x} = {total}",
        f"{num(newsalt)} ÷ {total} × 100 = {num(pct)}",
    ]
    return ins, lines, pct


_reg("conc_add_solute", conc_add_solute)


# 64. mix two concentrations to a target -> amount of the second
def conc_mix_find_amount(rng):
    c1 = rng.choice([5, 10, 15])
    c2 = rng.choice([25, 30, 40])
    c = rng.choice([v for v in range(c1 + 5, c2, 5)])
    gap = c2 - c
    need = c - c1
    k = rng.randint(10, 40)
    s1 = gap * k
    x = need * k
    ins = rng.choice([
        f"甲杯有{s1}克含盐{c1}%的盐水，乙杯是含盐{c2}%的盐水，把两杯混合成含盐{c}%的盐水，需要乙杯盐水多少克？",
        f"用{s1}克含盐{c1}%的盐水和含盐{c2}%的盐水配成含盐{c}%的盐水，需要浓度{c2}%的盐水多少克？",
        f"现有{s1}克浓度{c1}%的盐水，要兑成浓度{c}%的盐水，需加入浓度{c2}%的盐水多少克？",
        f"把{s1}克含盐{c1}%的盐水与一些含盐{c2}%的盐水混合，得到含盐{c}%的盐水，加入的盐水是多少克？",
    ])
    lines = [
        f"{c} - {c1} = {need}",
        f"{c2} - {c} = {gap}",
        f"{s1} × {need} = {s1 * need}",
        f"{s1 * need} ÷ {gap} = {x}",
    ]
    return ins, lines, x


_reg("conc_mix_find_amount", conc_mix_find_amount)


# 65. evaporate water to reach a target concentration -> amount evaporated
def conc_evaporate_to_target(rng):
    pairs = [(20, 25), (20, 30), (20, 40), (10, 30), (10, 40), (15, 40)]
    c, c2 = rng.choice(pairs)
    k = rng.randint(2, 10)
    s = c2 * k
    salt = Fraction(s * c, 100)
    newsol = Fraction(salt * 100, c2)
    w = s - newsol
    ins = rng.choice([
        f"有{s}克含盐{c}%的盐水，蒸发掉一部分水后浓度变为{c2}%，蒸发掉多少克水？",
        f"一杯{s}克的盐水含盐{c}%，加热蒸发后浓度变成{c2}%，蒸发了多少克水？",
        f"把{s}克含盐{c}%的盐水晒成含盐{c2}%的盐水，需要蒸发多少克水？",
        f"现有{s}克浓度{c}%的盐水，要使浓度变为{c2}%，需蒸发掉多少克水？",
    ])
    lines = [
        f"{s} × {c}/100 = {num(salt)}",
        f"{num(salt)} × 100 = {num(salt * 100)}",
        f"{num(salt * 100)} ÷ {c2} = {num(newsol)}",
        f"{s} - {num(newsol)} = {num(w)}",
    ]
    return ins, lines, w


_reg("conc_evaporate_to_target", conc_evaporate_to_target)


# 66. first batch sprout rate + full-germination replant -> overall rate
def sprout_replant(rng):
    a = rng.choice([100, 200, 300])
    p = rng.choice([80, 85, 90, 95])
    b = rng.choice([10, 20, 30, 50])
    germ = Fraction(a * p, 100)
    tg = germ + b
    ts = a + b
    rate = Fraction(tg * 100, ts)
    ins = rng.choice([
        f"种下{a}粒种子，发芽率是{p}%，后来又补种{b}粒且全部发芽，这些种子的总发芽率是多少？",
        f"第一批播了{a}粒种子，发芽率{p}%，第二批播的{b}粒全部发芽，总的发芽率是多少？",
        f"小明种了{a}粒花种，发芽率为{p}%，他又补种{b}粒全部发芽，总发芽率是多少？",
        f"试验田播下{a}粒种子，发芽率{p}%，补播的{b}粒全部发芽，这批种子的发芽率是多少？",
    ])
    lines = [
        f"{a} × {p}/100 = {num(germ)}",
        f"{num(germ)} + {b} = {num(tg)}",
        f"{a} + {b} = {ts}",
        f"{num(tg)} ÷ {ts} × 100 = {num(rate)}",
    ]
    return ins, lines, rate


_reg("sprout_replant", sprout_replant)


# 67. two juices with different juice:water ratios -> mixed concentration
def juice_mix_ratio(rng):
    a = rng.randint(1, 3)
    b = rng.randint(2, 4)
    k1 = rng.randint(10, 40)
    s1 = (a + b) * k1
    c = rng.randint(1, 3)
    d = rng.randint(2, 4)
    k2 = rng.randint(10, 40)
    s2 = (c + d) * k2
    j1 = a * k1
    j2 = c * k2
    juice = j1 + j2
    total = s1 + s2
    pct = Fraction(juice * 100, total)
    ins = rng.choice([
        f"第一杯{s1}克果汁中果汁与水的比是{a}:{b}，第二杯{s2}克果汁中果汁与水的比是{c}:{d}，两杯混合后果汁占百分之几？",
        f"甲杯{s1}克糖水，糖与水的比是{a}:{b}；乙杯{s2}克糖水，糖与水的比是{c}:{d}，混合后糖占糖水的百分之几？",
        f"两瓶果汁分别重{s1}克和{s2}克，果汁与水的比分别为{a}:{b}和{c}:{d}，混合后果汁的浓度是多少？",
        f"第一杯{s1}克盐水中盐与水的比是{a}:{b}，第二杯{s2}克盐水中盐与水的比是{c}:{d}，混合后盐占盐水的百分之几？",
    ])
    lines = [
        f"{a} + {b} = {a + b}",
        f"{s1} × {a}/{a + b} = {j1}",
        f"{c} + {d} = {c + d}",
        f"{s2} × {c}/{c + d} = {j2}",
        f"{j1} + {j2} = {juice}",
        f"{s1} + {s2} = {total}",
        f"{juice} ÷ {total} × 100 = {num(pct)}",
    ]
    return ins, lines, pct


_reg("juice_mix_ratio", juice_mix_ratio)


# 68. three-stage fraction use, remaining given -> original total
def reverse_three_stage(rng):
    d1 = rng.choice([2, 3, 4])
    d2 = rng.choice([2, 3, 4])
    d3 = rng.choice([2, 3, 4])
    k = rng.randint(2, 10)
    R = (d1 - 1) * (d2 - 1) * (d3 - 1) * k
    r2 = Fraction(R * d3, d3 - 1)
    r1 = Fraction(r2 * d2, d2 - 1)
    total = Fraction(r1 * d1, d1 - 1)
    unit = rng.choice(["吨", "本", "千克", "升"])
    ins = rng.choice([
        f"仓库有一批货物，第一天运走1/{d1}，第二天运走余下的1/{d2}，第三天运走余下的1/{d3}，还剩{R}{unit}，这批货物原来有多少{unit}？",
        f"一根绳子，第一次剪去1/{d1}，第二次剪去剩下的1/{d2}，第三次剪去剩下的1/{d3}，还剩{R}米，绳子原来长多少米？",
        f"食堂有一批大米，第一周吃了1/{d1}，第二周吃了余下的1/{d2}，第三周吃了余下的1/{d3}，还剩{R}{unit}，原来有多少{unit}？",
        f"书店有一批书，第一天卖出1/{d1}，第二天卖出余下的1/{d2}，第三天卖出余下的1/{d3}，还剩{R}本，这批书原来有多少本？",
    ])
    lines = [
        f"{R} × {d3} = {R * d3}",
        f"{R * d3} ÷ ({d3} - 1) = {num(r2)}",
        f"{num(r2)} × {d2} = {num(r2 * d2)}",
        f"{num(r2 * d2)} ÷ ({d2} - 1) = {num(r1)}",
        f"{num(r1)} × {d1} = {num(r1 * d1)}",
        f"{num(r1 * d1)} ÷ ({d1} - 1) = {num(total)}",
    ]
    return ins, lines, total


_reg("reverse_three_stage", reverse_three_stage)


# 69. read 1/d per day for k days, R pages left -> total pages
def fraction_book_reading(rng):
    d = rng.choice([4, 5, 6, 7, 8])
    k = rng.randint(1, d - 2)
    m = rng.randint(5, 30)
    R = (d - k) * m
    total = d * m
    read = Fraction(k, d)
    ins = rng.choice([
        f"小明看一本书，每天看全书的1/{d}，看了{k}天后还剩{R}页，这本书共有多少页？",
        f"小红读一本故事书，每天读全书的1/{d}，读了{k}天还剩{R}页，这本书共多少页？",
        f"小华看一本书，每天看1/{d}，{k}天后还剩{R}页没看，全书多少页？",
        f"一本小说，每天看全书的1/{d}，看了{k}天还剩{R}页，这本书一共有多少页？",
    ])
    lines = [
        f"{k} × 1/{d} = {num(read)}",
        f"1 - {num(read)} = {num(1 - read)}",
        f"{R} × {d} = {R * d}",
        f"{R * d} ÷ ({d} - {k}) = {total}",
    ]
    return ins, lines, total


_reg("fraction_book_reading", fraction_book_reading)


# 70. class: 1/d in math club, 1/e of the rest in English club -> neither
def class_two_fractions(rng):
    d = rng.choice([3, 4, 5, 6])
    e = rng.choice([2, 3, 4])
    k = rng.randint(2, 10)
    a = d * e * k
    math = e * k
    rem = a - math
    eng = k * (d - 1)
    neither = rem - eng
    ins = rng.choice([
        f"全班有{a}人，其中1/{d}参加数学兴趣小组，剩下的1/{e}参加英语兴趣小组，两个小组都没参加的有多少人？",
        f"图书馆有{a}本书，1/{d}是故事书，余下的1/{e}是科技书，其余的是漫画书，漫画书有多少本？",
        f"果园有{a}棵果树，1/{d}是苹果树，余下的1/{e}是梨树，其余的是桃树，桃树有多少棵？",
        f"学校运来{a}棵树苗，1/{d}分给六年级，余下的1/{e}分给五年级，其余的分给四年级，四年级分多少棵？",
    ])
    lines = [
        f"{a} × 1/{d} = {math}",
        f"{a} - {math} = {rem}",
        f"{rem} × 1/{e} = {eng}",
        f"{rem} - {eng} = {neither}",
    ]
    return ins, lines, neither


_reg("class_two_fractions", class_two_fractions)


# 71. father:son age sum + ratio -> years until father is 2x son
def ratio_age_future(rng):
    pairs = [(5, 1), (5, 2), (6, 1), (6, 2), (7, 2), (7, 3), (8, 3)]
    a, b = rng.choice(pairs)
    k = rng.randint(3, 10)
    S = (a + b) * k
    father = a * k
    son = b * k
    t = father - 2 * son
    ins = rng.choice([
        f"今年父子年龄和是{S}岁，父亲与儿子的年龄比是{a}:{b}，多少年后父亲的年龄正好是儿子的2倍？",
        f"今年爸爸和小明的年龄和是{S}岁，年龄比是{a}:{b}，几年后爸爸的年龄是小明的2倍？",
        f"母子今年年龄和为{S}岁，年龄比是{a}:{b}，多少年后母亲的年龄是儿子的2倍？",
        f"今年爷孙年龄和是{S}岁，年龄比是{a}:{b}，几年后爷爷的年龄是孙子的2倍？",
    ])
    lines = [
        f"{a} + {b} = {a + b}",
        f"{S} ÷ {a + b} = {k}",
        f"{a} × {k} = {father}",
        f"{b} × {k} = {son}",
        f"{father} - 2 × {son} = {t}",
    ]
    return ins, lines, t


_reg("ratio_age_future", ratio_age_future)


# 72. two equal ropes, different fractions used -> remainder difference
def fraction_rope_diff(rng):
    a = rng.choice([3, 4, 5])
    b = rng.choice([4, 5, 6])
    for _ in range(50):
        if b > a:
            break
        b = rng.choice([4, 5, 6])
    k = rng.randint(2, 10)
    L = a * b * k
    used1 = b * k
    used2 = a * k
    rem1 = L - used1
    rem2 = L - used2
    diff = rem2 - rem1
    ins = rng.choice([
        f"两根同样长的绳子都长{L}米，第一根用去1/{a}，第二根用去1/{b}，第二根剩下的比第一根剩下的长多少米？",
        f"两根彩带都长{L}米，第一根剪去1/{a}，第二根剪去1/{b}，剩下的相差多少米？",
        f"两捆一样长的电线都长{L}米，第一捆用去1/{a}，第二捆用去1/{b}，哪捆剩下的多？多多少米？",
        f"两根铁丝都长{L}米，第一根截去1/{a}，第二根截去1/{b}，剩下的长度相差多少米？",
    ])
    lines = [
        f"{L} ÷ {a} = {used1}",
        f"{L} ÷ {b} = {used2}",
        f"{L} - {used1} = {rem1}",
        f"{L} - {used2} = {rem2}",
        f"{rem2} - {rem1} = {diff}",
    ]
    return ins, lines, diff


_reg("fraction_rope_diff", fraction_rope_diff)


if __name__ == "__main__":
    rng = random.Random(3)
    from run_math_short import verify
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines} {ans}"
            ok += 1
    print(f"L3 ext3 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
