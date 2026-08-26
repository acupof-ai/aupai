#!/usr/bin/env python3
"""L3 ext5: distinct 5-7 step families (decimals, fractions, percents, ratios).

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
_LABELS = {}


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


def _d(f):
    """Exact decimal rendering for terminating Fractions, else n/d fallback."""
    f = Fraction(f)
    if f.denominator == 1:
        return str(f.numerator)
    n, d = abs(f.numerator), f.denominator
    sign = "-" if f.numerator < 0 else ""
    for k in range(1, 7):
        if (10 ** k) % d == 0:
            v = n * (10 ** k // d)
            s = str(v).zfill(k + 1)
            return f"{sign}{s[:-k]}.{s[-k:]}"
    return num(f)


# 1. decimal weight x bags, sell p%, rest sold at c yuan/kg -> money
def decimal_weight_total(rng):
    w = Fraction(rng.randint(15, 48), 10)
    n = rng.randint(8, 40)
    p = rng.choice([10, 20, 25, 30, 40])
    c = rng.randint(3, 12)
    total = w * n
    sold = Fraction(total * p, 100)
    left = total - sold
    money = left * c
    obj = rng.choice(["大米", "面粉", "苹果", "土豆", "洋葱", "胡萝卜"])
    ins = rng.choice([
        f"超市运来{n}袋{obj}，每袋重{_d(w)}千克，卖出{p}%后，剩下的{obj}每千克{c}元，还能卖多少元？",
        f"粮店运进{n}袋{obj}，每袋{_d(w)}千克，卖掉{p}%后，余下的每千克{c}元出售，还能卖多少元？",
        f"食堂买了{n}袋{obj}，每袋重{_d(w)}千克，用掉{p}%后，剩下的按每千克{c}元卖出，还能卖多少元？",
        f"批发市场有{n}袋{obj}，每袋{_d(w)}千克，卖出{p}%后，剩余的每千克{c}元，还能卖多少元？",
    ])
    lines = [
        f"{_d(w)} × {n} = {_d(total)}千克",
        f"{_d(total)} × {p}/100 = {_d(sold)}千克",
        f"{_d(total)} - {_d(sold)} = {_d(left)}千克",
        f"{_d(left)} × {c} = {_d(money)}元",
    ]
    return ins, lines, money


_reg("decimal_weight_total", decimal_weight_total)


# 2. decimal rope: cut fraction of total, then fixed meters, rest split into b pieces
def decimal_length_split(rng):
    L = Fraction(rng.randint(200, 500), 10)
    d = rng.choice([2, 4, 5])
    a = Fraction(rng.randint(20, 60), 10)
    for _ in range(50):
        if a < L - L / d:
            break
        a = Fraction(rng.randint(20, 60), 10)
    b = rng.choice([2, 4, 5])
    first = L / d
    rem = L - first
    left = rem - a
    piece = left / b
    obj = rng.choice(["绳子", "彩带", "铁丝", "电线", "布条"])
    ins = rng.choice([
        f"一根{obj}长{_d(L)}米，第一次剪去全长的1/{d}，第二次剪去{_d(a)}米，剩下的平均分成{b}段，每段长多少米？",
        f"一根{obj}长{_d(L)}米，先剪去全长的1/{d}，又剪去{_d(a)}米，余下的平均分成{b}段，每段多少米？",
        f"一根{obj}长{_d(L)}米，第一次用去全长的1/{d}，第二次用去{_d(a)}米，剩下的平均分成{b}段，每段长多少米？",
        f"一根{_d(L)}米长的{obj}，剪去全长的1/{d}后又剪去{_d(a)}米，剩下的平均分成{b}段，每段长多少米？",
    ])
    lines = [
        f"{_d(L)} × 1/{d} = {_d(first)}米",
        f"{_d(L)} - {_d(first)} = {_d(rem)}米",
        f"{_d(rem)} - {_d(a)} = {_d(left)}米",
        f"{_d(left)} ÷ {b} = {_d(piece)}米",
    ]
    return ins, lines, piece


_reg("decimal_length_split", decimal_length_split)


# 3. decimal speed x fraction time, then 1/d of total left -> total distance
def decimal_speed_distance(rng):
    v = Fraction(rng.randint(80, 200), 10)
    t = Fraction(rng.randint(15, 45), 10)
    d = rng.choice([3, 5, 6])
    gone = v * t
    total = gone * d / (d - 1)
    left = total - gone
    who = rng.choice(["小明", "小红", "小华", "小刚", "爸爸", "妈妈"])
    ins = rng.choice([
        f"{who}骑车每小时行{_d(v)}千米，行了{_d(t)}小时后，还剩全程的1/{d}没走，全程多少千米？",
        f"{who}骑自行车每小时走{_d(v)}千米，骑了{_d(t)}小时后，还有全程的1/{d}没走完，全程多少千米？",
        f"{who}骑车的速度是每小时{_d(v)}千米，行{_d(t)}小时后，还剩全程的1/{d}，全程多少千米？",
        f"{who}每小时骑车行{_d(v)}千米，行了{_d(t)}小时，正好走了全程的（1-1/{d}），全程多少千米？",
    ])
    lines = [
        f"{_d(v)} × {_d(t)} = {_d(gone)}千米",
        f"剩下的占比 = 1 - 1/{d} = {num(Fraction(d - 1, d))}",
        f"{_d(gone)} ÷ ({num(Fraction(d - 1, d))}) = {_d(total)}千米",
    ]
    return ins, lines, total


_reg("decimal_speed_distance", decimal_speed_distance)


# 4. four decimal jump results -> average -> gap to record
def decimal_average_jump(rng):
    a = Fraction(rng.randint(120, 200), 100)
    b = Fraction(rng.randint(120, 200), 100)
    c = Fraction(rng.randint(120, 200), 100)
    e = Fraction(rng.randint(120, 200), 100)
    d = Fraction(rng.randint(210, 260), 100)
    s = a + b + c + e
    avg = s / 4
    diff = d - avg
    who = rng.choice(["小明", "小红", "小华", "小刚", "小丽", "小军"])
    event = rng.choice(["跳远", "立定跳远", "三级跳远"])
    ins = rng.choice([
        f"{who}四次{event}的成绩分别是{_d(a)}米、{_d(b)}米、{_d(c)}米、{_d(e)}米，平均成绩是多少米？比学校纪录{_d(d)}米少多少米？",
        f"{who}练{event}，四次成绩为{_d(a)}米、{_d(b)}米、{_d(c)}米、{_d(e)}米，平均每次跳多少米？和{_d(d)}米的纪录相差多少米？",
        f"在{event}测试中，{who}四次分别跳了{_d(a)}米、{_d(b)}米、{_d(c)}米、{_d(e)}米，平均成绩是多少米？比{_d(d)}米的纪录少多少米？",
        f"{who}四次{event}成绩为{_d(a)}米、{_d(b)}米、{_d(c)}米、{_d(e)}米，平均成绩多少米？离{_d(d)}米的纪录还差多少米？",
    ])
    lines = [
        f"{_d(a)} + {_d(b)} = {_d(a + b)}米",
        f"{_d(a + b)} + {_d(c)} = {_d(a + b + c)}米",
        f"{_d(a + b + c)} + {_d(e)} = {_d(s)}米",
        f"{_d(s)} ÷ 4 = {_d(avg)}米",
        f"{_d(d)} - {_d(avg)} = {_d(diff)}米",
    ]
    return ins, lines, diff


_reg("decimal_average_jump", decimal_average_jump)


# 5. four people split a decimal bill, three pay fixed amounts -> fourth over/under average
def decimal_money_split(rng):
    A = Fraction(rng.randint(800, 2000), 10)
    a = Fraction(rng.randint(100, 500), 10)
    b = Fraction(rng.randint(100, 500), 10)
    c = Fraction(rng.randint(100, 500), 10)
    for _ in range(50):
        if a + b + c < A:
            break
        A = Fraction(rng.randint(800, 2000), 10)
    d_pay = A - a - b - c
    avg = A / 4
    diff = d_pay - avg
    obj = rng.choice(["礼物", "蛋糕", "水果篮", "书籍", "文具"])
    who = rng.choice(["甲、乙、丙、丁四人", "小红、小明、小华、小刚四人", "四个同学", "四位朋友"])
    ins = rng.choice([
        f"{who}合买一件{obj}共用去{_d(A)}元，甲付{_d(a)}元，乙付{_d(b)}元，丙付{_d(c)}元，其余的丁付，丁比平均每人多付多少元？",
        f"{who}合买{obj}花了{_d(A)}元，其中三人分别付了{_d(a)}元、{_d(b)}元、{_d(c)}元，剩下的由第四人付，他比平均每人多付多少元？",
        f"一件{obj}售价{_d(A)}元，{who}凑钱购买，三人各付{_d(a)}元、{_d(b)}元、{_d(c)}元，余下的第四人付清，他比平均每人多付多少元？",
        f"{who}一起买了{_d(A)}元的{obj}，甲出{_d(a)}元，乙出{_d(b)}元，丙出{_d(c)}元，剩下的丁出，丁比平均每人多出多少元？",
    ])
    lines = [
        f"{_d(a)} + {_d(b)} = {_d(a + b)}元",
        f"{_d(a + b)} + {_d(c)} = {_d(a + b + c)}元",
        f"{_d(A)} - {_d(a + b + c)} = {_d(d_pay)}元",
        f"{_d(A)} ÷ 4 = {_d(avg)}元",
        f"{_d(d_pay)} - {_d(avg)} = {_d(diff)}元",
    ]
    return ins, lines, diff


_reg("decimal_money_split", decimal_money_split)


# 6. unit-price compare with unit conversion (1.5L bottle vs 500mL bottle)
def decimal_unit_price_compare(rng):
    k = rng.randint(4, 15)
    x = 3 * k
    y = rng.randint(k + 1, k + 12)
    per_big = 2 * k
    per_small = 2 * y
    diff = per_small - per_big
    obj = rng.choice(["洗发水", "牛奶", "果汁", "食用油", "洗衣液", "沐浴露"])
    ins = rng.choice([
        f"大瓶{obj}1.5升售价{x}元，小瓶500毫升售价{y}元，买哪种便宜？每升便宜多少元？",
        f"一种{obj}大瓶装1.5升卖{x}元，小瓶装500毫升卖{y}元，哪种便宜？每升便宜多少元？",
        f"超市里{obj}有1.5升装{x}元和500毫升装{y}元两种，哪种更便宜？每升便宜多少元？",
        f"妈妈买{obj}，1.5升的标价{x}元，500毫升的标价{y}元，哪种便宜？每升便宜多少元？",
    ])
    lines = [
        f"500 ÷ 1000 = 1/2升",
        f"{y} ÷ (1/2) = {per_small}元",
        f"{x} ÷ (3/2) = {per_big}元",
        f"{per_small} - {per_big} = {diff}元",
    ]
    return ins, lines, diff


_reg("decimal_unit_price_compare", decimal_unit_price_compare)


# 7. decimal machine rate x hours x days, qualified p% -> qualified count
def decimal_machine_output(rng):
    rate = Fraction(rng.randint(12, 36), 2)
    hours = Fraction(rng.randint(60, 95), 10)
    days = rng.randint(3, 7)
    p = rng.choice([90, 92, 95, 96, 98])
    per_day = rate * hours
    total = per_day * days
    good = Fraction(total * p, 100)
    bad = total - good
    obj = rng.choice(["零件", "螺丝", "玩具", "文具", "口罩"])
    ins = rng.choice([
        f"一台机器每小时加工{_d(rate)}个{obj}，每天工作{_d(hours)}小时，{days}天共加工多少个？合格率是{p}%，合格的有多少个？",
        f"某车间每小时生产{_d(rate)}个{obj}，每天开工{_d(hours)}小时，{days}天一共生产多少个？其中合格率{p}%，合格产品多少个？",
        f"一台设备每小时做{_d(rate)}个{obj}，每天运转{_d(hours)}小时，{days}天共做多少个？合格率为{p}%，合格的有多少个？",
        f"工厂的机器每小时生产{_d(rate)}个{obj}，每天工作{_d(hours)}小时，{days}天生产总数是多少？合格率{p}%，合格品多少个？",
    ])
    lines = [
        f"{_d(rate)} × {_d(hours)} = {_d(per_day)}个",
        f"{_d(per_day)} × {days} = {_d(total)}个",
        f"{_d(total)} × {p}/100 = {_d(good)}个",
    ]
    return ins, lines, good


_reg("decimal_machine_output", decimal_machine_output)


# 8. decimal pages/day x days, then more pages/day -> remaining days
def decimal_book_pages(rng):
    x = rng.randint(150, 300)
    per = Fraction(rng.randint(12, 24), 2)
    c = rng.randint(4, 7)
    d = rng.randint(2, 6)
    read = per * c
    left = x - read
    new_per = per + d
    days_left = left / new_per
    who = rng.choice(["小明", "小红", "小华", "小丽", "小军"])
    ins = rng.choice([
        f"一本书共{x}页，{who}每天看{_d(per)}页，看了{c}天后，每天多看{d}页，剩下的还要看多少天？",
        f"{who}看一本{x}页的书，原计划每天看{_d(per)}页，看了{c}天后，每天比原来多看{d}页，还需多少天看完？",
        f"一本{x}页的故事书，{who}每天读{_d(per)}页，读了{c}天，之后每天多读{d}页，剩下的还要读几天？",
        f"{who}要看完{x}页的书，每天看{_d(per)}页，看了{c}天后加快速度，每天多看{d}页，还要几天看完？",
    ])
    lines = [
        f"{_d(per)} × {c} = {_d(read)}页",
        f"{x} - {_d(read)} = {_d(left)}页",
        f"{_d(per)} + {d} = {_d(new_per)}页",
        f"{_d(left)} ÷ ({_d(new_per)}) = {num(days_left)}天",
    ]
    return ins, lines, days_left


_reg("decimal_book_pages", decimal_book_pages)


# 9. second item half price: buy n items -> total, compare with original
def second_half_price(rng):
    p = rng.randint(8, 40)
    n = rng.randint(2, 6) * 2
    half = Fraction(p, 2)
    m = n // 2
    pair = p + half
    total = pair * m
    orig = p * n
    saved = orig - total
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店促销：第二件半价。{obj}每件{p}元，小明买{n}件，一共要付多少元？",
        f"超市里{obj}搞活动，买一件{p}元，第二件半价，小红买{n}件，共付多少元？",
        f"一种{obj}售价{p}元，活动期间第二件半价，买{n}件需要多少元？",
        f"{obj}原价每件{p}元，现第二件半价，老师买{n}件，实际付多少元？",
    ])
    lines = [
        f"{p} ÷ 2 = {num(half)}元",
        f"{p} + {num(half)} = {num(pair)}元",
        f"{n} ÷ 2 = {m}组",
        f"{num(pair)} × {m} = {num(total)}元",
    ]
    return ins, lines, total


_reg("second_half_price", second_half_price)


# 10. group buy: m people pay T total, each saves x -> original price per person
def groupbuy_original_price(rng):
    m = rng.randint(5, 20)
    g = rng.randint(20, 80)
    T = m * g
    x = rng.randint(5, 30)
    saved = m * x
    orig_total = T + saved
    orig = g + x
    obj = rng.choice(["电影票", "自助餐券", "游乐园门票", "健身卡", "蛋糕券"])
    ins = rng.choice([
        f"{m}人团购{obj}共付{T}元，每人比单独买便宜{x}元，单独买每张多少元？",
        f"团购{obj}，{m}人一共付了{T}元，已知每人比原价便宜{x}元，原价每张多少元？",
        f"某{obj}原价每张若干元，{m}人团购共花{T}元，每人省了{x}元，原价是多少元？",
        f"{m}个同学合买{obj}，共付{T}元，比每人单独买一共少花{x * m}元，每张原价多少元？",
    ])
    lines = [
        f"{T} ÷ {m} = {g}元",
        f"{m} × {x} = {saved}元",
        f"{T} + {saved} = {orig_total}元",
        f"{orig_total} ÷ {m} = {orig}元",
    ]
    return ins, lines, orig


_reg("groupbuy_original_price", groupbuy_original_price)


# 11. phone plans: monthly fee + per-minute vs per-minute only -> breakeven minutes
def phone_plan_breakeven(rng):
    t0 = rng.randint(50, 300)
    diff = Fraction(rng.randint(1, 3), 10)
    x = Fraction(rng.randint(1, 3), 10)
    y = x + diff
    a = t0 * diff
    costA = a + x * t0
    costB = y * t0
    ins = rng.choice([
        f"手机套餐A：月租{_d(a)}元，通话每分钟{_d(x)}元；套餐B：无月租，每分钟{_d(y)}元。通话多少分钟时两种套餐费用相同？",
        f"话费方案一：每月{_d(a)}元月租，每分钟通话{_d(x)}元；方案二：免月租，每分钟{_d(y)}元。每月通话多少分钟，两种方案费用一样？",
        f"两种手机卡：A卡月租{_d(a)}元，每分钟{_d(x)}元；B卡无月租，每分钟{_d(y)}元。通话多少分钟时两卡费用相同？",
        f"通讯公司有两种套餐：甲套餐月租{_d(a)}元、每分钟{_d(x)}元，乙套餐无月租、每分钟{_d(y)}元。通话多少分钟两种套餐费用相等？",
    ])
    lines = [
        f"{_d(y)} - {_d(x)} = {_d(diff)}元",
        f"{_d(diff)} × {t0} = {_d(a)}元",
        f"{_d(a)} ÷ ({_d(diff)}) = {t0}分",
    ]
    return ins, lines, t0


_reg("phone_plan_breakeven", phone_plan_breakeven)


# 12. taxi fare: flagfall + distance + waiting time -> total
def taxi_fare(rng):
    a = rng.choice([8, 9, 10, 11, 12, 13])
    b = rng.choice([2, 3])
    c = rng.randint(2, 4)
    d = rng.randint(b + 2, b + 12)
    t = rng.randint(3, 10)
    w = Fraction(rng.randint(1, 3), 10)
    over = d - b
    extra = over * c
    wait_fee = t * w
    total = a + extra + wait_fee
    who = rng.choice(["小明", "爸爸", "妈妈", "小红", "叔叔"])
    ins = rng.choice([
        f"出租车起步价{a}元（含{b}千米），超过部分每千米{c}元，{who}乘车行了{d}千米，中途等待{t}分钟，每分钟{_d(w)}元，共付车费多少元？",
        f"某市出租车起步价{a}元，可行{b}千米，超出后每千米{c}元。{who}坐车行{d}千米，等候{t}分钟，每分钟{_d(w)}元，应付多少元？",
        f"出租车收费：起步{a}元（{b}千米内），超出每千米{c}元，等待每分钟{_d(w)}元。{who}行了{d}千米，等待{t}分钟，车费多少元？",
        f"{who}乘出租车，起步价{a}元含{b}千米，超过的路程每千米{c}元，路上等待{t}分钟，每分钟{_d(w)}元，共行{d}千米，应付多少元？",
    ])
    lines = [
        f"{d} - {b} = {over}千米",
        f"{over} × {c} = {extra}元",
        f"{t} × {_d(w)} = {_d(wait_fee)}元",
        f"{a} + {extra} + {_d(wait_fee)} = {_d(total)}元",
    ]
    return ins, lines, total


_reg("taxi_fare", taxi_fare)


# 13. spoilage pricing: cost c, loss 1/d, want p% profit -> selling price per kg
def spoilage_pricing(rng):
    for _ in range(50):
        c = rng.randint(4, 12)
        d = rng.choice([10, 20, 25])
        p = rng.choice([20, 25, 40, 50])
        n = rng.randint(50, 200)
        total_cost = c * n
        sold_kg = Fraction(n * (d - 1), d)
        revenue = Fraction(total_cost * (100 + p), 100)
        price = revenue / sold_kg
        if price.denominator == 1:
            break
    obj = rng.choice(["苹果", "香蕉", "葡萄", "草莓", "桃子", "梨"])
    ins = rng.choice([
        f"水果店以每千克{c}元购进{n}千克{obj}，运输中损耗1/{d}，要想获利{p}%，每千克应卖多少元？",
        f"商家购进{n}千克{obj}，进价每千克{c}元，运输损耗了1/{d}，要赚{p}%的利润，售价应是每千克多少元？",
        f"一批{obj}共{n}千克，进价{c}元/千克，损耗1/{d}后，要获利{p}%，每千克售价多少元？",
        f"超市进了{n}千克{obj}，每千克{c}元，预计损耗1/{d}，要保证{p}%的利润，每千克应定价多少元？",
    ])
    lines = [
        f"{c} × {n} = {total_cost}元",
        f"{n} × {d - 1}/{d} = {num(sold_kg)}千克",
        f"{total_cost} × ({100 + p}/100) = {num(revenue)}元",
        f"{num(revenue)} ÷ ({num(sold_kg)}) = {num(price)}元",
    ]
    return ins, lines, price


_reg("spoilage_pricing", spoilage_pricing)


# 14. member price is p% cheaper, member price X -> original price
def reverse_pct_decrease(rng):
    p = rng.choice([10, 15, 20, 25, 30, 40])
    k = rng.randint(2, 30)
    X = (100 - p) * k
    orig = 100 * k
    saved = orig - X
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"商店的{obj}凭会员卡便宜{p}%，会员价是{X}元，原价是多少元？",
        f"一件{obj}按会员价出售比原价少{p}%，会员价{X}元，原价多少元？",
        f"超市会员购买{obj}可省{p}%，小红付了{X}元，这件{obj}原价多少元？",
        f"某{obj}原价若干元，会员卡可减{p}%，小明用会员卡花了{X}元，原价是多少元？",
    ])
    lines = [
        f"会员价占比 = 100 - {p} = {100 - p}",
        f"{X} × {p} = {X * p}元",
        f"{X * p} ÷ {100 - p} = {saved}元",
        f"{X} + {saved} = {orig}元",
    ]
    return ins, lines, orig


_reg("reverse_pct_decrease", reverse_pct_decrease)


# 15. sold at X with p% profit -> cost; then sold at q discount -> profit/loss
def profit_cost_reverse(rng):
    p = rng.choice([10, 20, 25, 40, 50])
    k = rng.randint(2, 30)
    X = (100 + p) * k
    cost = 100 * k
    profit = X - cost
    q = rng.choice([5, 6, 7, 8, 9])
    sell2 = Fraction(cost * q, 10)
    diff2 = cost - sell2
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}售价{X}元，赚了{p}%。若按成本的{q}折出售，要亏多少元？",
        f"某{obj}卖{X}元，利润率是{p}%，如果按成本打{q}折出售，亏多少元？",
        f"商店以{X}元卖出一件{obj}，赚了{p}%，这件{obj}按成本的{q}折出售要亏多少元？",
        f"一件{obj}现价{X}元，正好赚{p}%，若按成本价的{q}折出售，亏多少元？",
    ])
    lines = [
        f"{X} ÷ ({100 + p}/100) = {cost}元",
        f"{cost} × {q}/10 = {num(sell2)}元",
        f"{cost} - {num(sell2)} = {num(diff2)}元",
    ]
    return ins, lines, diff2


_reg("profit_cost_reverse", profit_cost_reverse)


# 16. cost C, sold at X -> markup percent; then sold at d discount -> profit
def markup_rate_find(rng):
    C = rng.randint(2, 20) * 20
    p = rng.choice([20, 25, 50, 50, 50])
    profit = C * p // 100
    X = C + profit
    d = rng.choice([7, 8, 9]) if p == 50 else 9
    sell_d = Fraction(X * d, 10)
    profit_d = sell_d - C
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}成本是{C}元，售价是{X}元，加价了百分之几？若按售价打{d}折出售，每件还能赚多少元？",
        f"某{obj}进价{C}元，卖{X}元，利润率是百分之几？按售价的{d}折出售，每件赚多少元？",
        f"商店进一件{obj}花了{C}元，以{X}元卖出，加价百分之几？若打{d}折出售，还能赚多少元？",
        f"一件{obj}成本{C}元，标价{X}元，标价比成本高百分之几？按标价打{d}折卖出，每件赚多少元？",
    ])
    lines = [
        f"{X} - {C} = {profit}元",
        f"{profit} ÷ {C} = {num(Fraction(profit, C))}",
        f"{num(Fraction(profit, C))} × 100 = {p}",
        f"{X} × {d}/10 = {num(sell_d)}元",
        f"{num(sell_d)} - {C} = {num(profit_d)}元",
    ]
    return ins, lines, profit_d


_reg("markup_rate_find", markup_rate_find)


# 17. group ticket p% cheaper, m people -> total saved
def group_ticket_save(rng):
    a = rng.randint(20, 80)
    p = rng.choice([10, 20, 25, 40])
    m = rng.randint(10, 45)
    save_each = Fraction(a * p, 100)
    group = a - save_each
    orig_total = a * m
    group_total = group * m
    save_total = orig_total - group_total
    place = rng.choice(["动物园", "植物园", "游乐园", "博物馆", "水族馆", "滑雪场"])
    ins = rng.choice([
        f"{place}门票每张{a}元，{m}人以上可购团体票，每张便宜{p}%，{m}人买团体票共省多少元？",
        f"某{place}门票原价{a}元，团体票优惠{p}%，{m}人一起买团体票，一共少花多少元？",
        f"{place}单人票{a}元，购团体票每张便宜{p}%，{m}人买团体票比买单人票省多少元？",
        f"同学们去{place}，门票每张{a}元，{m}人买团体票每张便宜{p}%，共节省多少元？",
    ])
    lines = [
        f"{a} × {p}/100 = {num(save_each)}元",
        f"{a} - {num(save_each)} = {num(group)}元",
        f"{a} × {m} = {orig_total}元",
        f"{num(group)} × {m} = {num(group_total)}元",
        f"{orig_total} - {num(group_total)} = {num(save_total)}元",
    ]
    return ins, lines, save_total


_reg("group_ticket_save", group_ticket_save)


# 18. coal x tons, burn y kg/day for z days -> fraction left
def coal_burn_fraction(rng):
    x = rng.randint(4, 12)
    y = rng.randint(50, 250)
    z = rng.randint(3, 8)
    burned_kg = y * z
    burned_t = Fraction(burned_kg, 1000)
    left_t = x - burned_t
    frac = left_t / x
    obj = rng.choice(["煤", "大米", "面粉", "饲料", "化肥"])
    unit = rng.choice(["吨", "吨", "吨", "千克"])
    ins = rng.choice([
        f"一堆{obj}重{x}吨，每天烧{y}千克，烧了{z}天，还剩这堆{obj}的几分之几？",
        f"食堂运来{x}吨{obj}，每天用去{y}千克，{z}天后还剩几分之几？",
        f"仓库有{x}吨{obj}，每天消耗{y}千克，用了{z}天，剩下的占原来的几分之几？",
        f"一批{obj}共{x}吨，每天烧{y}千克，烧了{z}天，还剩几分之几？",
    ])
    lines = [
        f"{y} × {z} = {burned_kg}千克",
        f"{burned_kg} ÷ 1000 = {num(burned_t)}吨",
        f"{x} - {num(burned_t)} = {num(left_t)}吨",
        f"{num(left_t)} ÷ {x} = {num(frac)}",
    ]
    return ins, lines, frac


_reg("coal_burn_fraction", coal_burn_fraction)


# 19. 甲的 a/b = 乙的 c/d, sum S -> 甲
def frac_equal_ratio_split(rng):
    a = rng.randint(2, 3)
    b = rng.randint(3, 5)
    c = rng.randint(2, 3)
    d = rng.randint(3, 5)
    for _ in range(50):
        if a * d != b * c:
            break
        c = rng.randint(2, 3)
        d = rng.randint(3, 5)
    p1 = b * c
    p2 = a * d
    k = rng.randint(2, 20)
    S = (p1 + p2) * k
    jia = p1 * k
    obj = rng.choice(["钱", "本书", "颗糖", "张邮票", "朵花"])
    who = rng.choice([("甲", "乙"), ("小明", "小红"), ("哥哥", "弟弟"), ("姐姐", "妹妹")])
    w1, w2 = who
    ins = rng.choice([
        f"{w1}的{a}/{b}等于{w2}的{c}/{d}，两人共有{S}{obj}，{w1}有多少{obj}？",
        f"{w1}钱数的{a}/{b}与{w2}钱数的{c}/{d}相等，两人共{S}{obj}，{w1}有多少{obj}？",
        f"{w1}的{a}/{b}和{w2}的{c}/{d}一样多，两人一共有{S}{obj}，{w1}有多少{obj}？",
        f"{w1}的{a}/{b}等于{w2}的{c}/{d}，两人合计{S}{obj}，{w1}有多少{obj}？",
    ])
    lines = [
        f"{b} × {c} = {p1}",
        f"{a} × {d} = {p2}",
        f"{p1} + {p2} = {p1 + p2}",
        f"{S} ÷ {p1 + p2} = {k}",
        f"{p1} × {k} = {jia}{obj}",
    ]
    return ins, lines, jia


_reg("frac_equal_ratio_split", frac_equal_ratio_split)


# 20. day1 1/a, day2 1/b of remainder, two days total X pages -> total pages
def fraction_sum_denominators(rng):
    a = rng.randint(3, 6)
    b = rng.randint(2, 4)
    k = rng.randint(2, 20)
    X = (b + a - 1) * k
    total = a * b * k
    who = rng.choice(["小明", "小红", "小华", "小丽", "小军"])
    obj = rng.choice(["故事书", "童话书", "科技书", "漫画书", "小说"])
    ins = rng.choice([
        f"{who}看一本{obj}，第一天看了全书的1/{a}，第二天看了余下的1/{b}，两天共看了{X}页，这本书共多少页？",
        f"一本{obj}，第一天读了1/{a}，第二天读了剩下的1/{b}，两天一共读了{X}页，全书多少页？",
        f"{who}读一本{obj}，第一天看全书的1/{a}，第二天看余下的1/{b}，两天共看{X}页，这本书有多少页？",
        f"一本{obj}共若干页，第一天看了1/{a}，第二天看了余下的1/{b}，两天合计看了{X}页，全书多少页？",
    ])
    lines = [
        f"1 - 1/{a} = {num(Fraction(a - 1, a))}",
        f"{num(Fraction(a - 1, a))} × 1/{b} = {num(Fraction(a - 1, a * b))}",
        f"1/{a} + {num(Fraction(a - 1, a * b))} = {num(Fraction(b + a - 1, a * b))}",
        f"{X} ÷ ({num(Fraction(b + a - 1, a * b))}) = {total}页",
    ]
    return ins, lines, total


_reg("fraction_sum_denominators", fraction_sum_denominators)


# 21. used 1/a, remainder exceeds used by x meters -> original length
def fraction_remaining_compare(rng):
    a = rng.randint(3, 6)
    k = rng.randint(2, 20)
    x = (a - 2) * k
    L = a * k
    used = k
    obj = rng.choice(["绳子", "彩带", "铁丝", "电线", "布条"])
    ins = rng.choice([
        f"一根{obj}，用去全长的1/{a}，剩下的比用去的多{x}米，这根{obj}原来长多少米？",
        f"一根{obj}长若干米，用去1/{a}后，剩下的比用去的多{x}米，原来长多少米？",
        f"一根{obj}，剪去全长的1/{a}，余下的比剪去的长{x}米，这根{obj}长多少米？",
        f"一根{obj}用去它的1/{a}，剩下的部分比用去的多{x}米，{obj}原来长多少米？",
    ])
    lines = [
        f"1 - 1/{a} = {num(Fraction(a - 1, a))}",
        f"{num(Fraction(a - 1, a))} - 1/{a} = {num(Fraction(a - 2, a))}",
        f"{x} ÷ ({num(Fraction(a - 2, a))}) = {L}米",
    ]
    return ins, lines, L


_reg("fraction_remaining_compare", fraction_remaining_compare)


# 22. A has A yuan, gives 1/a then x yuan twice, then equal -> B's original
def fraction_give_twice(rng):
    a = rng.randint(3, 5)
    k = rng.randint(10, 40)
    A = a * k
    x = rng.randint(2, 10)
    for _ in range(50):
        if k * (a - 2) > 2 * x:
            break
        x = rng.randint(2, 10)
    gave = k
    B = A - 2 * gave - 2 * x
    obj = rng.choice(["元", "本书", "颗糖", "张邮票"])
    who = rng.choice([("甲", "乙"), ("小明", "小红"), ("哥哥", "弟弟"), ("姐姐", "妹妹")])
    w1, w2 = who
    ins = rng.choice([
        f"{w1}有{A}{obj}，先拿出自己的1/{a}给{w2}，又给{w2}{x}{obj}，这时两人一样多，{w2}原来有多少{obj}？",
        f"{w1}有{A}{obj}，第一次给{w2}自己的1/{a}，第二次又给{w2}{x}{obj}，两人相等，{w2}原有多少{obj}？",
        f"{w1}原有{A}{obj}，把自己的1/{a}给{w2}后，再给{w2}{x}{obj}，两人正好一样多，{w2}原来有多少{obj}？",
        f"{w1}有{A}{obj}，给{w2}自己的1/{a}，又给{w2}{x}{obj}，两人{obj}数相等，{w2}原来有多少{obj}？",
    ])
    lines = [
        f"{A} × 1/{a} = {gave}{obj}",
        f"{gave} × 2 = {2 * gave}{obj}",
        f"{x} × 2 = {2 * x}{obj}",
        f"{A} - {2 * gave} - {2 * x} = {B}{obj}",
    ]
    return ins, lines, B


_reg("fraction_give_twice", fraction_give_twice)


# 23. three ropes of mixed-number lengths, total minus used d meters
def fraction_mixed_total(rng):
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    c = rng.randint(2, 6)
    d = rng.randint(2, 6)
    for _ in range(50):
        if d < a + b + c + 1:
            break
        d = rng.randint(2, 6)
    total = a + b + c + 1
    left = total - d
    obj = rng.choice(["绳子", "彩带", "铁丝", "电线", "布条"])
    ins = rng.choice([
        f"三根{obj}分别长{a}又1/2米、{b}又1/3米、{c}又1/6米，三根一共长多少米？用去{d}米后还剩多少米？",
        f"三根{obj}的长度是{a}又1/2米、{b}又1/3米、{c}又1/6米，共长多少米？用掉{d}米，还剩多少米？",
        f"有三根{obj}，分别长{a}又1/2米、{b}又1/3米、{c}又1/6米，一共长多少米？剪去{d}米，还剩多少米？",
        f"三根{obj}长{a}又1/2米、{b}又1/3米、{c}又1/6米，总长多少米？用去{d}米后剩多少米？",
    ])
    lines = [
        f"1/2 + 1/3 = {num(Fraction(5, 6))}",
        f"{num(Fraction(5, 6))} + 1/6 = 1",
        f"{a} + {b} + {c} = {a + b + c}米",
        f"{a + b + c} + 1 = {total}米",
        f"{total} - {d} = {left}米",
    ]
    return ins, lines, left


_reg("fraction_mixed_total", fraction_mixed_total)


# 24. 甲是乙的1/a, 乙是丙的1/b, sum S -> 丙
def nested_fraction_three(rng):
    a = rng.randint(2, 4)
    b = rng.randint(2, 4)
    k = rng.randint(2, 15)
    S = (a * b + b + 1) * k
    bing = a * b * k
    obj = rng.choice(["本书", "元钱", "颗糖", "张邮票", "朵花"])
    who = rng.choice([("甲", "乙", "丙"), ("小明", "小红", "小华"), ("哥哥", "姐姐", "弟弟")])
    w1, w2, w3 = who
    ins = rng.choice([
        f"{w1}是{w2}的1/{a}，{w2}是{w3}的1/{b}，三人共有{S}{obj}，{w3}有多少{obj}？",
        f"{w1}的钱数是{w2}的1/{a}，{w2}的钱数是{w3}的1/{b}，三人共{S}{obj}，{w3}有多少{obj}？",
        f"{w1}等于{w2}的1/{a}，{w2}等于{w3}的1/{b}，三个数的和是{S}，{w3}是多少？",
        f"{w1}是{w2}的1/{a}，{w2}是{w3}的1/{b}，三人合计{S}{obj}，{w3}有多少{obj}？",
    ])
    lines = [
        f"{a} × {b} = {a * b}",
        f"{a * b} + {b} + 1 = {a * b + b + 1}",
        f"{S} × {a * b} = {S * a * b}",
        f"{S * a * b} ÷ {a * b + b + 1} = {bing}{obj}",
    ]
    return ins, lines, bing


_reg("nested_fraction_three", nested_fraction_three)


# 25. two equal ropes, cut 1/a and 1/b, remainders differ by x -> length
def fraction_rope_cut_compare(rng):
    a = rng.randint(3, 5)
    b = rng.randint(a + 1, a + 4)
    k = rng.randint(2, 15)
    x = (b - a) * k
    L = a * b * k
    obj = rng.choice(["绳子", "彩带", "铁丝", "电线", "布条"])
    ins = rng.choice([
        f"两根同样长的{obj}，第一根剪去全长的1/{a}，第二根剪去全长的1/{b}，剩下的相差{x}米，原来每根长多少米？",
        f"两根{obj}一样长，第一根用去1/{a}，第二根用去1/{b}，余下的相差{x}米，每根原来长多少米？",
        f"两根同样长的{obj}，一根剪去1/{a}，另一根剪去1/{b}，剩下的长度差{x}米，原来每根长多少米？",
        f"有两根等长的{obj}，第一根剪去全长的1/{a}，第二根剪去全长的1/{b}，第二根比第一根剩下的长{x}米，每根长多少米？",
    ])
    lines = [
        f"1 - 1/{a} = {num(Fraction(a - 1, a))}",
        f"1 - 1/{b} = {num(Fraction(b - 1, b))}",
        f"{num(Fraction(b - 1, b))} - {num(Fraction(a - 1, a))} = {num(Fraction(b - a, a * b))}",
        f"{x} ÷ ({num(Fraction(b - a, a * b))}) = {L}米",
    ]
    return ins, lines, L


_reg("fraction_rope_cut_compare", fraction_rope_cut_compare)


# 26. day1 1/a, day2 1/b of remainder, day1 exceeds day2 by x pages -> total
def fraction_reading_compare(rng):
    a = rng.randint(3, 5)
    b = rng.randint(a, a + 3)
    k = rng.randint(2, 15)
    x = (b - a + 1) * k
    total = a * b * k
    who = rng.choice(["小明", "小红", "小华", "小丽", "小军"])
    obj = rng.choice(["故事书", "童话书", "科技书", "漫画书", "小说"])
    ins = rng.choice([
        f"{who}看一本{obj}，第一天看了全书的1/{a}，第二天看了余下的1/{b}，第一天比第二天多看了{x}页，这本书共多少页？",
        f"一本{obj}，第一天读了1/{a}，第二天读了剩下的1/{b}，第一天比第二天多读{x}页，全书多少页？",
        f"{who}读一本{obj}，第一天看全书的1/{a}，第二天看余下的1/{b}，第一天比第二天多看{x}页，这本书有多少页？",
        f"一本{obj}，第一天看了1/{a}，第二天看了余下的1/{b}，第二天比第一天少看{x}页，全书多少页？",
    ])
    lines = [
        f"1 - 1/{a} = {num(Fraction(a - 1, a))}",
        f"{num(Fraction(a - 1, a))} × 1/{b} = {num(Fraction(a - 1, a * b))}",
        f"1/{a} - {num(Fraction(a - 1, a * b))} = {num(Fraction(b - a + 1, a * b))}",
        f"{x} ÷ ({num(Fraction(b - a + 1, a * b))}) = {total}页",
    ]
    return ins, lines, total


_reg("fraction_reading_compare", fraction_reading_compare)


# 27. three people share S: 甲 1/a, 乙 1/b of remainder -> 丙
def fraction_money_nested(rng):
    a = rng.randint(3, 6)
    b = rng.randint(2, 4)
    k = rng.randint(2, 15)
    S = a * b * k
    jia = b * k
    rem = S - jia
    yi = k * (a - 1)
    bing = rem - yi
    obj = rng.choice(["元", "本书", "颗糖", "张邮票"])
    who = rng.choice([("甲", "乙", "丙"), ("小明", "小红", "小华"), ("哥哥", "姐姐", "弟弟")])
    w1, w2, w3 = who
    ins = rng.choice([
        f"三人共有{S}{obj}，{w1}占总数的1/{a}，{w2}占剩下的1/{b}，其余的是{w3}的，{w3}有多少{obj}？",
        f"{w1}、{w2}、{w3}共有{S}{obj}，{w1}分得总数的1/{a}，{w2}分得余下的1/{b}，{w3}分得多少{obj}？",
        f"一笔钱共{S}{obj}，{w1}得1/{a}，{w2}得余下的1/{b}，剩下的归{w3}，{w3}得多少{obj}？",
        f"三人分{S}{obj}，{w1}先拿总数的1/{a}，{w2}再拿余下的1/{b}，{w3}拿到多少{obj}？",
    ])
    lines = [
        f"{S} × 1/{a} = {jia}{obj}",
        f"{S} - {jia} = {rem}{obj}",
        f"{rem} × 1/{b} = {yi}{obj}",
        f"{rem} - {yi} = {bing}{obj}",
    ]
    return ins, lines, bing


_reg("fraction_money_nested", fraction_money_nested)


# 28. three cups: 乙 is 1/b of 甲, 丙 is c/d of 乙 -> total
def fraction_water_three_cups(rng):
    b = rng.randint(2, 4)
    d = rng.randint(2, 5)
    c = rng.randint(1, d - 1)
    k = rng.randint(5, 30)
    a = b * d * k
    yi = d * k
    bing = c * k
    total = a + yi + bing
    obj = rng.choice(["水", "牛奶", "果汁", "豆浆", "盐水"])
    who = rng.choice(["甲、乙、丙三个杯子", "三个杯子", "甲杯、乙杯、丙杯"])
    ins = rng.choice([
        f"{who}里共有{obj}若干，甲杯有{a}克，乙杯是甲杯的1/{b}，丙杯是乙杯的{c}/{d}，三杯共多少克？",
        f"甲杯有{a}克{obj}，乙杯的{obj}是甲杯的1/{b}，丙杯是乙杯的{c}/{d}，三杯一共多少克？",
        f"三杯{obj}，甲杯{a}克，乙杯相当于甲杯的1/{b}，丙杯相当于乙杯的{c}/{d}，共多少克？",
        f"甲杯装{a}克{obj}，乙杯装的是甲杯的1/{b}，丙杯装的是乙杯的{c}/{d}，三杯总共多少克？",
    ])
    lines = [
        f"{a} × 1/{b} = {yi}克",
        f"{yi} × {c}/{d} = {bing}克",
        f"{a} + {yi} = {a + yi}克",
        f"{a + yi} + {bing} = {total}克",
    ]
    return ins, lines, total


_reg("fraction_water_three_cups", fraction_water_three_cups)


# 29. 甲是乙的 p%, 乙是丙的 q%, sum S -> 丙
def pct_nested_three(rng):
    pairs = [(20, 50), (25, 40), (40, 25), (50, 20), (20, 20), (40, 40),
             (50, 50), (10, 50), (10, 20), (25, 80), (40, 50), (50, 60)]
    p, q = rng.choice(pairs)
    k = rng.choice([2, 3, 4, 5, 8, 10])
    bing = 100 * k
    yi = q * k
    jia = p * q * k // 100
    S = bing + yi + jia
    obj = rng.choice(["本书", "元钱", "颗糖", "张邮票", "朵花"])
    who = rng.choice([("甲", "乙", "丙"), ("小明", "小红", "小华"), ("哥哥", "姐姐", "弟弟")])
    w1, w2, w3 = who
    ins = rng.choice([
        f"{w1}是{w2}的{p}%，{w2}是{w3}的{q}%，三人共有{S}{obj}，{w3}有多少{obj}？",
        f"{w1}的钱数是{w2}的{p}%，{w2}的钱数是{w3}的{q}%，三人共{S}{obj}，{w3}有多少{obj}？",
        f"{w1}等于{w2}的{p}%，{w2}等于{w3}的{q}%，三个数的和是{S}，{w3}是多少？",
        f"{w1}是{w2}的{p}%，{w2}是{w3}的{q}%，三人合计{S}{obj}，{w3}有多少{obj}？",
    ])
    lines = [
        f"{p} × {q} = {p * q}",
        f"10000 + 100 × {q} + {p * q} = {10000 + 100 * q + p * q}",
        f"{S} × 10000 = {S * 10000}",
        f"{S * 10000} ÷ {10000 + 100 * q + p * q} = {bing}{obj}",
    ]
    return ins, lines, bing


_reg("pct_nested_three", pct_nested_three)


# 30. day1 sell p1%, day2 sell p2% of remainder, left X -> original
def pct_of_remainder_reverse(rng):
    pairs = [(10, 20), (10, 40), (20, 20), (20, 50), (25, 20), (25, 60),
             (40, 25), (40, 50), (50, 20), (50, 60), (60, 25), (75, 40)]
    p1, p2 = rng.choice(pairs)
    k = rng.choice([1, 2, 3, 5])
    total = 100 * k
    X = (100 - p1) * (100 - p2) * k // 100
    unit = rng.choice(["吨", "千克", "本", "升", "件"])
    obj = rng.choice(["货物", "水果", "图书", "饮料", "商品", "大米"])
    ins = rng.choice([
        f"商店有一批{obj}，第一天卖出{p1}%，第二天卖出余下的{p2}%，还剩{X}{unit}，这批{obj}原来有多少{unit}？",
        f"仓库运来一批{obj}，第一天运走{p1}%，第二天运走余下的{p2}%，还剩{X}{unit}，这批{obj}共多少{unit}？",
        f"一批{obj}，第一次卖掉{p1}%，第二次卖掉剩下的{p2}%，还剩{X}{unit}，原来有多少{unit}？",
        f"书店新进一批{obj}，第一天卖出总数的{p1}%，第二天卖出余下的{p2}%，还剩{X}{unit}，新进多少{unit}？",
    ])
    lines = [
        f"100 - {p2} = {100 - p2}",
        f"{100 - p1} × {100 - p2} = {(100 - p1) * (100 - p2)}",
        f"{X} × 10000 = {X * 10000}",
        f"{X * 10000} ÷ {(100 - p1) * (100 - p2)} = {total}{unit}",
    ]
    return ins, lines, total


_reg("pct_of_remainder_reverse", pct_of_remainder_reverse)


_LABELS["markup_rate_find"] = ["加价比例", "加价百分比"]
_LABELS["coal_burn_fraction"] = ["剩下的占比"]
_LABELS["frac_equal_ratio_split"] = ["甲的份数", "乙的份数", "份数和", "每份"]
_LABELS["fraction_sum_denominators"] = ["第一天后剩下的", "第二天看的", "两天共看的占比"]
_LABELS["fraction_remaining_compare"] = ["剩下的占比", "剩下比用去多的占比"]
_LABELS["fraction_mixed_total"] = ["分数部分的和", "分数部分合计"]
_LABELS["nested_fraction_three"] = ["分母的积", "份数和", "总和扩大的倍数"]
_LABELS["fraction_rope_cut_compare"] = ["第一根剩下的", "第二根剩下的", "剩下的占比差"]
_LABELS["fraction_reading_compare"] = ["第一天后剩下的", "第二天看的", "第一天比第二天多的占比"]
_LABELS["pct_nested_three"] = ["百分数的积", "份数和", "总和扩大的倍数"]
_LABELS["pct_of_remainder_reverse"] = ["第二天后剩下的百分比", "剩下的百分比积", "剩下的量扩大的倍数"]


# 31. A is p% more than B, B is q% more than C, A given -> C
def pct_compare_chain(rng):
    pairs = [(10, 20), (10, 40), (10, 50), (20, 25), (20, 50), (20, 75),
             (25, 40), (25, 60), (40, 50), (40, 75), (50, 20), (50, 40),
             (50, 60), (50, 80)]
    p, q = rng.choice(pairs)
    k = rng.choice([1, 2, 3])
    A = (100 + p) * (100 + q) * k // 100
    yi = (100 + q) * k
    bing = 100 * k
    obj = rng.choice(["", "本书", "元钱", "颗糖", "张邮票"])
    who = rng.choice([("甲", "乙", "丙"), ("小明", "小红", "小华"), ("哥哥", "姐姐", "弟弟")])
    w1, w2, w3 = who
    ins = rng.choice([
        f"{w1}比{w2}多{p}%，{w2}比{w3}多{q}%，{w1}有{A}{obj}，{w3}有多少{obj}？",
        f"{w1}的{obj}比{w2}多{p}%，{w2}的{obj}比{w3}多{q}%，{w1}有{A}{obj}，{w3}有多少{obj}？",
        f"{w1}比{w2}大{p}%，{w2}比{w3}大{q}%，已知{w1}是{A}{obj}，{w3}是多少{obj}？",
        f"{w1}的钱比{w2}多{p}%，{w2}的钱比{w3}多{q}%，{w1}有{A}{obj}，{w3}有多少{obj}？",
    ])
    lines = [
        f"{A} × 100 = {A * 100}",
        f"{A * 100} ÷ {100 + p} = {yi}",
        f"{yi} × 100 = {yi * 100}",
        f"{yi * 100} ÷ {100 + q} = {bing}{obj}",
    ]
    return ins, lines, bing


_LABELS["pct_compare_chain"] = ["甲扩大100倍", "乙数", "乙扩大100倍", "丙数"]
_reg("pct_compare_chain", pct_compare_chain)


# 32. price P, up p1%, down p2%, up p3% -> final price
def pct_three_changes(rng):
    P = rng.randint(2, 10) * 100
    p1 = rng.choice([10, 20, 25, 40, 50])
    p2 = rng.choice([10, 20, 25, 40, 50])
    p3 = rng.choice([10, 20, 25, 40, 50])
    m1 = Fraction(P * (100 + p1), 100)
    m2 = Fraction(m1 * (100 - p2), 100)
    m3 = Fraction(m2 * (100 + p3), 100)
    obj = rng.choice(GOODS)
    ins = rng.choice([
        f"一件{obj}原价{P}元，一月涨价{p1}%，二月降价{p2}%，三月又涨价{p3}%，现价多少元？",
        f"某{obj}原价{P}元，先涨{p1}%，再降{p2}%，又涨{p3}%，现在售价多少元？",
        f"一件{obj}的价格是{P}元，第一个月上调{p1}%，第二个月下调{p2}%，第三个月再上调{p3}%，现价多少元？",
        f"{obj}原价{P}元，经历涨价{p1}%、降价{p2}%、涨价{p3}%三次调整后，售价多少元？",
    ])
    lines = [
        f"{P} × ({100 + p1}/100) = {num(m1)}元",
        f"{num(m1)} × ({100 - p2}/100) = {num(m2)}元",
        f"{num(m2)} × ({100 + p3}/100) = {num(m3)}元",
    ]
    return ins, lines, m3


_reg("pct_three_changes", pct_three_changes)


# 33. pass rate p%, failing x people -> total people
def pct_pass_reverse(rng):
    p = rng.choice([80, 85, 90, 92, 95, 96, 98])
    k = rng.randint(1, 10)
    x = (100 - p) * k
    total = 100 * k
    obj = rng.choice(["学生", "考生", "选手", "学员"])
    ins = rng.choice([
        f"一次考试及格率是{p}%，不及格的有{x}人，参加考试的共多少人？",
        f"某年级数学测验及格率为{p}%，不及格{x}人，全年级有多少{obj}？",
        f"一场竞赛的及格率是{p}%，已知不及格{x}人，共有多少{obj}参加？",
        f"某次测试及格率{p}%，不及格人数是{x}人，参加测试的有多少人？",
    ])
    lines = [
        f"不及格的百分比 = 100 - {p} = {100 - p}",
        f"{x} × 100 = {x * 100}人",
        f"{x * 100} ÷ {100 - p} = {total}人",
    ]
    return ins, lines, total


_reg("pct_pass_reverse", pct_pass_reverse)


# 34. ratio a:b, difference d -> product
def ratio_product_find(rng):
    a = rng.randint(3, 7)
    b = rng.randint(2, a - 1)
    k = rng.randint(2, 12)
    d = (a - b) * k
    jia = a * k
    yi = b * k
    P = jia * yi
    obj = rng.choice(["元", "本书", "颗糖", "张邮票"])
    who = rng.choice([("甲", "乙"), ("小明", "小红"), ("哥哥", "弟弟"), ("姐姐", "妹妹")])
    w1, w2 = who
    ins = rng.choice([
        f"{w1}、{w2}两人{obj}数的比是{a}:{b}，{w1}比{w2}多{d}{obj}，两人{obj}数的积是多少？",
        f"{w1}与{w2}的{obj}数比为{a}:{b}，{w1}比{w2}多{d}{obj}，两人{obj}数相乘是多少？",
        f"{w1}、{w2}的{obj}数之比是{a}:{b}，{w2}比{w1}少{d}{obj}，两人{obj}数的积是多少？",
        f"{w1}和{w2}的{obj}数比是{a}:{b}，{w1}比{w2}多{d}{obj}，他们{obj}数的乘积是多少？",
    ])
    lines = [
        f"份数差 = {a} - {b} = {a - b}",
        f"每份 = {d} ÷ {a - b} = {k}",
        f"{w1} = {a} × {k} = {jia}{obj}",
        f"{w2} = {b} × {k} = {yi}{obj}",
        f"{jia} × {yi} = {P}",
    ]
    return ins, lines, P


_LABELS["ratio_product_find"] = ["两数的积"]
_reg("ratio_product_find", ratio_product_find)


# 35. ratio a:b, add x to both -> ratio c:d -> 甲
def ratio_change_after_add(rng):
    tuples = [(5, 3, 3, 2), (7, 5, 4, 3), (7, 4, 5, 3), (9, 7, 5, 4),
              (13, 8, 8, 5), (9, 5, 7, 4), (10, 7, 7, 5)]
    a, b, c, d = rng.choice(tuples)
    x = rng.randint(2, 10)
    k = x * (c - d)
    jia = a * k
    obj = rng.choice(["元", "本书", "颗糖", "张邮票"])
    who = rng.choice([("甲", "乙"), ("小明", "小红"), ("哥哥", "弟弟"), ("姐姐", "妹妹")])
    w1, w2 = who
    ins = rng.choice([
        f"{w1}、{w2}两人{obj}数的比是{a}:{b}，两人各增加{x}{obj}后，比变成{c}:{d}，{w1}原来有多少{obj}？",
        f"{w1}与{w2}的{obj}数比为{a}:{b}，{w1}、{w2}各添{x}{obj}后，比是{c}:{d}，{w1}原有多少{obj}？",
        f"{w1}、{w2}的{obj}数之比是{a}:{b}，两人都增加{x}{obj}后，比变为{c}:{d}，{w1}原来有多少{obj}？",
        f"{w1}和{w2}的{obj}数比是{a}:{b}，各加上{x}{obj}后比是{c}:{d}，{w1}原有多少{obj}？",
    ])
    lines = [
        f"{a} × {d} = {a * d}",
        f"{b} × {c} = {b * c}",
        f"{a * d} - {b * c} = {a * d - b * c}",
        f"{x} × {c - d} = {x * (c - d)}",
        f"{x * (c - d)} ÷ {a * d - b * c} = {k}",
        f"{a} × {k} = {jia}{obj}",
    ]
    return ins, lines, jia


_LABELS["ratio_change_after_add"] = ["甲的分母积", "乙的分母积", "份数差", "变化的份数", "每份"]
_reg("ratio_change_after_add", ratio_change_after_add)


# 36. ratio a:b, 甲 gives 乙 x -> ratio c:d -> 甲 original
def ratio_fraction_transfer(rng):
    tuples = [(5, 3, 3, 2), (7, 5, 4, 3), (7, 4, 5, 3), (9, 7, 5, 4),
              (13, 8, 8, 5), (9, 5, 7, 4), (10, 7, 7, 5)]
    a, b, c, d = rng.choice(tuples)
    x = rng.randint(2, 10)
    T = x * (a + b) * (c + d)
    jia = x * a * (c + d)
    obj = rng.choice(["元", "本书", "颗糖", "张邮票"])
    who = rng.choice([("甲", "乙"), ("小明", "小红"), ("哥哥", "弟弟"), ("姐姐", "妹妹")])
    w1, w2 = who
    ins = rng.choice([
        f"{w1}、{w2}两人{obj}数的比是{a}:{b}，{w1}给{w2}{x}{obj}后，比变成{c}:{d}，{w1}原来有多少{obj}？",
        f"{w1}与{w2}的{obj}数比为{a}:{b}，{w1}拿出{x}{obj}给{w2}后，比是{c}:{d}，{w1}原有多少{obj}？",
        f"{w1}、{w2}的{obj}数之比是{a}:{b}，{w1}给{w2}{x}{obj}后，两人比变为{c}:{d}，{w1}原来有多少{obj}？",
        f"{w1}和{w2}的{obj}数比是{a}:{b}，{w1}给{w2}{x}{obj}后比是{c}:{d}，{w1}原有多少{obj}？",
    ])
    lines = [
        f"{a} + {b} = {a + b}",
        f"{c} + {d} = {c + d}",
        f"{a + b} × {c + d} = {(a + b) * (c + d)}",
        f"{(a + b) * (c + d)} × {x} = {T}{obj}",
        f"{T} × {a} = {T * a}{obj}",
        f"{T * a} ÷ {a + b} = {jia}{obj}",
    ]
    return ins, lines, jia


_LABELS["ratio_fraction_transfer"] = ["原来的份数和", "后来的份数和", "份数积"]
_reg("ratio_fraction_transfer", ratio_fraction_transfer)


# 37. radius ratio a:b, big circle area A -> small circle area
def ratio_circle_area(rng):
    a = rng.randint(3, 6)
    b = rng.randint(2, a - 1)
    k = rng.randint(2, 10)
    A = a * a * k
    small = b * b * k
    ins = rng.choice([
        f"大、小两个圆半径的比是{a}:{b}，大圆面积是{A}平方米，小圆面积是多少平方米？",
        f"两个圆的半径比为{a}:{b}，大圆面积{A}平方米，小圆面积多少平方米？",
        f"大圆与小圆的半径比是{a}:{b}，已知大圆面积是{A}平方米，小圆面积是多少平方米？",
        f"甲、乙两圆半径比是{a}:{b}，甲圆面积为{A}平方米，乙圆面积是多少平方米？",
    ])
    lines = [
        f"{a} × {a} = {a * a}",
        f"{b} × {b} = {b * b}",
        f"{A} × {b * b} = {A * b * b}",
        f"{A * b * b} ÷ {a * a} = {small}平方米",
    ]
    return ins, lines, small


_LABELS["ratio_circle_area"] = ["大圆半径的平方", "小圆半径的平方", "扩大后的面积"]
_reg("ratio_circle_area", ratio_circle_area)


# 38. edge ratio a:b, big cube volume V -> small cube volume
def ratio_cube_volume(rng):
    a = rng.randint(3, 6)
    b = rng.randint(2, a - 1)
    k = rng.randint(1, 8)
    V = a ** 3 * k
    small = b ** 3 * k
    ins = rng.choice([
        f"大、小两个正方体棱长的比是{a}:{b}，大正方体体积是{V}立方分米，小正方体体积是多少立方分米？",
        f"两个正方体棱长比为{a}:{b}，大正方体体积{V}立方分米，小正方体体积是多少立方分米？",
        f"大正方体与小正方体的棱长比是{a}:{b}，大正方体体积是{V}立方分米，小正方体体积是多少立方分米？",
        f"甲、乙两个正方体棱长比是{a}:{b}，甲的体积是{V}立方分米，乙的体积是多少立方分米？",
    ])
    lines = [
        f"{a} × {a} × {a} = {a ** 3}",
        f"{b} × {b} × {b} = {b ** 3}",
        f"{V} × {b ** 3} = {V * b ** 3}",
        f"{V * b ** 3} ÷ {a ** 3} = {small}立方分米",
    ]
    return ins, lines, small


_LABELS["ratio_cube_volume"] = ["大正方体棱长的立方", "小正方体棱长的立方", "扩大后的体积"]
_reg("ratio_cube_volume", ratio_cube_volume)


# 39. two equal-weight alloys with ratios a:b and c:d, same sum -> mixed ratio
def alloy_mix_ratio(rng):
    s = rng.randint(5, 10)
    a = rng.randint(1, s - 1)
    c = rng.randint(1, s - 1)
    for _ in range(50):
        if a != c:
            break
        c = rng.randint(1, s - 1)
    b = s - a
    d = s - c
    gold = a + c
    copper = 2 * s - gold
    ratio = Fraction(gold, copper)
    ins = rng.choice([
        f"两块同样重的合金，金与铜的比分别是{a}:{b}和{c}:{d}，熔合后金与铜的比是多少？",
        f"两块合金一样重，第一块金铜比{a}:{b}，第二块金铜比{c}:{d}，熔在一起后金铜比是多少？",
        f"有两块等重的合金，含金铜比分别为{a}:{b}和{c}:{d}，熔合后新合金的金铜比是多少？",
        f"两块重量相同的合金，金铜比各为{a}:{b}和{c}:{d}，熔化混合后金与铜的比是多少？",
    ])
    lines = [
        f"{a} + {b} = {s}",
        f"{c} + {d} = {s}",
        f"{a} + {c} = {gold}",
        f"{s} × 2 = {2 * s}",
        f"{2 * s} - {gold} = {copper}",
        f"{gold} ÷ {copper} = {num(ratio)}",
    ]
    return ins, lines, ratio


_LABELS["alloy_mix_ratio"] = ["第一块的份数和", "第二块的份数和", "金的总份数", "两块的总份数", "铜的总份数", "金铜比"]
_reg("alloy_mix_ratio", alloy_mix_ratio)


# 40. ratio a:b, 甲 deposits x -> ratio c:b (乙 unchanged) -> 乙
def ratio_future_money(rng):
    a = rng.randint(2, 5)
    c = rng.randint(a + 1, a + 3)
    b = rng.randint(2, 8)
    k = rng.randint(2, 15)
    x = (c - a) * k
    yi = b * k
    jia_old = a * k
    jia_new = jia_old + x
    obj = rng.choice(["元", "本书", "颗糖", "张邮票"])
    who = rng.choice([("甲", "乙"), ("小明", "小红"), ("哥哥", "弟弟"), ("姐姐", "妹妹")])
    w1, w2 = who
    ins = rng.choice([
        f"{w1}、{w2}两人{obj}数的比是{a}:{b}，{w1}又存入{x}{obj}后，比变成{c}:{b}，{w2}原来有多少{obj}？",
        f"{w1}与{w2}的{obj}数比为{a}:{b}，{w1}增加{x}{obj}后，两人比是{c}:{b}，{w2}原有多少{obj}？",
        f"{w1}、{w2}的{obj}数之比是{a}:{b}，{w1}又得到{x}{obj}后，比变为{c}:{b}，{w2}原来有多少{obj}？",
        f"{w1}和{w2}的{obj}数比是{a}:{b}，{w1}添了{x}{obj}后比是{c}:{b}，{w2}原有多少{obj}？",
    ])
    lines = [
        f"份数差 = {c} - {a} = {c - a}",
        f"每份 = {x} ÷ {c - a} = {k}",
        f"{w1}原有的 = {a} × {k} = {jia_old}{obj}",
        f"{w1}现有的 = {jia_old} + {x} = {jia_new}{obj}",
        f"{w2} = {b} × {k} = {yi}{obj}",
    ]
    return ins, lines, yi


_reg("ratio_future_money", ratio_future_money)


# 41. sugar:water a:b, add x grams water -> ratio a:c -> original total
def ratio_sugar_water(rng):
    a = rng.randint(1, 5)
    b = rng.randint(2, 6)
    c = rng.randint(b + 1, b + 3)
    k = rng.randint(2, 15)
    x = (c - b) * k
    total = (a + b) * k
    ins = rng.choice([
        f"一杯糖水中糖与水的比是{a}:{b}，加入{x}克水后，糖与水的比变成{a}:{c}，这杯糖水原来有多少克？",
        f"糖水中糖和水的比为{a}:{b}，加水{x}克后比变为{a}:{c}，原来糖水多少克？",
        f"一杯糖水糖与水之比是{a}:{b}，加入{x}克水后，糖与水之比是{a}:{c}，原来这杯糖水多少克？",
        f"糖水中糖与水的比是{a}:{b}，再加入{x}克水，比变成{a}:{c}，这杯糖水原有多少克？",
    ])
    lines = [
        f"份数差 = {c} - {b} = {c - b}",
        f"每份 = {x} ÷ {c - b} = {k}",
        f"原来的份数和 = {a} + {b} = {a + b}",
        f"{a + b} × {k} = {total}克",
    ]
    return ins, lines, total


_reg("ratio_sugar_water", ratio_sugar_water)


# 42. black:white a:b, remove x black -> ratio c:b -> original black
def ratio_black_white(rng):
    a = rng.randint(3, 7)
    c = rng.randint(2, a - 1)
    b = rng.randint(2, 6)
    k = rng.randint(2, 15)
    x = (a - c) * k
    black = a * k
    total = (a + b) * k
    obj = rng.choice(["棋子", "玻璃球", "卡片", "石子"])
    ins = rng.choice([
        f"一堆{obj}中黑、白两种的比是{a}:{b}，取走{x}颗黑色的后，比变成{c}:{b}，黑色的原有多少颗？",
        f"黑、白{obj}的比为{a}:{b}，拿走{x}颗黑{obj}后比变为{c}:{b}，黑{obj}原来有多少颗？",
        f"一盒{obj}黑与白之比是{a}:{b}，取出{x}颗黑{obj}后，比是{c}:{b}，黑{obj}原有多少颗？",
        f"黑色和白色{obj}的比是{a}:{b}，取走{x}颗黑色后比变成{c}:{b}，黑色{obj}原有多少颗？",
    ])
    lines = [
        f"份数差 = {a} - {c} = {a - c}",
        f"每份 = {x} ÷ {a - c} = {k}",
        f"总份数 = {a} + {b} = {a + b}",
        f"{a + b} × {k} = {total}颗",
        f"{a} × {k} = {black}颗",
    ]
    return ins, lines, black


_reg("ratio_black_white", ratio_black_white)


# 43. read:unread a:b, read x more pages -> ratio c:d -> total pages
def ratio_reading_pages(rng):
    tuples = [(2, 3, 3, 4), (3, 4, 4, 5), (4, 5, 5, 6), (5, 6, 6, 7),
              (2, 5, 3, 7), (3, 5, 5, 8), (2, 7, 3, 10)]
    a, b, c, d = rng.choice(tuples)
    x = rng.randint(2, 8)
    k = x * (c + d)
    total = (a + b) * k
    who = rng.choice(["小明", "小红", "小华", "小丽", "小军"])
    ins = rng.choice([
        f"{who}看一本书，已读与未读的页数比是{a}:{b}，再读{x}页后，比变成{c}:{d}，这本书共多少页？",
        f"一本书已读和未读的比为{a}:{b}，再读{x}页后比变为{c}:{d}，全书多少页？",
        f"{who}读一本书，已读页数与未读页数之比是{a}:{b}，又读了{x}页后，比是{c}:{d}，这本书有多少页？",
        f"一本故事书已读与未读的比是{a}:{b}，再读{x}页，已读与未读的比变成{c}:{d}，全书共多少页？",
    ])
    lines = [
        f"{b} × {c} = {b * c}",
        f"{a} × {d} = {a * d}",
        f"{b * c} - {a * d} = {b * c - a * d}",
        f"{x} × {c + d} = {x * (c + d)}",
        f"{x * (c + d)} ÷ {b * c - a * d} = {k}",
        f"{a + b} × {k} = {total}页",
    ]
    return ins, lines, total


_LABELS["ratio_reading_pages"] = ["内项积", "外项积", "份数差", "变化的份数", "每份"]
_reg("ratio_reading_pages", ratio_reading_pages)


# 44. speed up by 1/n, arrive t hours early -> distance
def speed_fraction_early(rng):
    n = rng.randint(3, 6)
    t = rng.randint(1, 3)
    v = rng.choice([40, 50, 60, 70, 80, 90])
    t_orig = t * (n + 1)
    dist = v * t_orig
    who = rng.choice(["小明", "爸爸", "司机", "一辆汽车", "小红"])
    ins = rng.choice([
        f"{who}从甲地到乙地，若车速提高1/{n}，可提前{t}小时到达，原速度是每小时{v}千米，甲乙两地相距多少千米？",
        f"一辆车从A地到B地，速度提高1/{n}后，提前{t}小时到达，已知原速为每小时{v}千米，两地距离多少千米？",
        f"{who}开车去某地，把车速提高1/{n}，就比原定时间早到{t}小时，原速每小时{v}千米，路程是多少千米？",
        f"从甲地到乙地，车速提高1/{n}可提前{t}小时到达，若每小时行{v}千米，甲乙两地相距多少千米？",
    ])
    lines = [
        f"1 + 1/{n} = {num(Fraction(n + 1, n))}",
        f"1 ÷ ({num(Fraction(n + 1, n))}) = {num(Fraction(n, n + 1))}",
        f"1 - {num(Fraction(n, n + 1))} = {num(Fraction(1, n + 1))}",
        f"{t} ÷ (1/{n + 1}) = {t_orig}小时",
        f"{v} × {t_orig} = {dist}千米",
    ]
    return ins, lines, dist


_LABELS["speed_fraction_early"] = ["提速后的倍数", "时间的比例", "提前的占比"]
_reg("speed_fraction_early", speed_fraction_early)


# 45. two people walk toward each other, dog runs back and forth -> dog distance
def dog_between_meet(rng):
    v1 = rng.randint(40, 90)
    v2 = rng.randint(40, 90)
    k = rng.randint(2, 4)
    D = (v1 + v2) * k
    vd = rng.randint(100, 200)
    dog = vd * k
    who = rng.choice([("甲", "乙"), ("小明", "小红"), ("哥哥", "弟弟")])
    w1, w2 = who
    ins = rng.choice([
        f"{w1}、{w2}两人从相距{D}米的两地相向而行，{w1}每分钟走{v1}米，{w2}每分钟走{v2}米，一只狗以每分钟{vd}米的速度在两人之间往返跑，两人相遇时狗跑了多少米？",
        f"两地相距{D}米，{w1}、{w2}同时相向出发，速度分别为每分钟{v1}米、{v2}米，一只狗每分钟{vd}米在两人间来回跑，相遇时狗跑了多少米？",
        f"{w1}和{w2}相距{D}米，同时相向而行，{w1}每分走{v1}米，{w2}每分走{v2}米，一条狗每分跑{vd}米在两人中间往返，到两人相遇时狗共跑多少米？",
        f"甲、乙两地相距{D}米，{w1}、{w2}两人同时相向而行，速度各为每分{v1}米、{v2}米，一只狗以每分{vd}米的速度往返于两人之间，相遇时狗跑了多少米？",
    ])
    lines = [
        f"{v1} + {v2} = {v1 + v2}米",
        f"{D} ÷ {v1 + v2} = {k}分",
        f"{vd} × {k} = {dog}米",
    ]
    return ins, lines, dog


_reg("dog_between_meet", dog_between_meet)


# 46. train passes a walker: same direction and opposite direction times
def train_pass_walker(rng):
    v1 = rng.randint(15, 30)
    v2 = rng.randint(1, 3)
    k = rng.randint(1, 2)
    L = (v1 * v1 - v2 * v2) * k
    t1 = Fraction(L, v1 - v2)
    t2 = Fraction(L, v1 + v2)
    ins = rng.choice([
        f"一列火车长{L}米，每秒行{v1}米，一个人在铁路旁每秒走{v2}米，火车从身后开来，经过这个人需要多少秒？若迎面走来呢？",
        f"火车长{L}米，速度每秒{v1}米，路旁一人每秒行{v2}米，火车同向追上这个人到完全超过需要多少秒？相向而行呢？",
        f"一列长{L}米的火车以每秒{v1}米行驶，一人每秒走{v2}米，火车从他身边同向通过要多少秒？相向通过要多少秒？",
        f"火车长{L}米，每秒{v1}米，一人每秒{v2}米沿铁路行走，火车从后面超过他需多少秒？迎面相遇需多少秒？",
    ])
    lines = [
        f"{v1} - {v2} = {v1 - v2}米",
        f"{v1} + {v2} = {v1 + v2}米",
        f"{L} ÷ {v1 + v2} = {num(t2)}秒",
        f"{L} ÷ {v1 - v2} = {num(t1)}秒",
    ]
    return ins, lines, t1


_reg("train_pass_walker", train_pass_walker)


# 47. plane speed is k times train, plane saves t hours -> distance
def plane_train_time_diff(rng):
    k = rng.choice([3, 4, 5, 6, 8])
    m = rng.randint(1, 3)
    u = rng.choice([100, 150, 200, 250])
    v = k * u
    t = (k - 1) * m
    dist = v * m
    ins = rng.choice([
        f"飞机每小时飞行{v}千米，是火车速度的{k}倍，从甲城到乙城飞机比火车少用{t}小时，甲乙两城相距多少千米？",
        f"飞机速度是每小时{v}千米，正好是火车的{k}倍，乘飞机比坐火车少用{t}小时，两地相距多少千米？",
        f"从A市到B市，飞机每小时{v}千米，速度是火车的{k}倍，飞机比火车快{t}小时，A、B两市相距多少千米？",
        f"飞机时速{v}千米，是火车时速的{k}倍，飞过去比坐火车少花{t}小时，两地距离多少千米？",
    ])
    lines = [
        f"{v} ÷ {k} = {u}千米",
        f"倍数差 = {k} - 1 = {k - 1}",
        f"{t} ÷ {k - 1} = {m}小时",
        f"{v} × {m} = {dist}千米",
    ]
    return ins, lines, dist


_reg("plane_train_time_diff", plane_train_time_diff)


# 48. circular track: same direction lap time, opposite direction meet time
def circle_track(rng):
    pairs = [(a, b) for a in range(5, 13) for b in range(1, a)]
    v1, v2 = rng.choice(pairs)
    t1 = Fraction(400, v1 - v2)
    t2 = Fraction(400, v1 + v2)
    ins = rng.choice([
        f"环形跑道一圈400米，甲每秒跑{v1}米，乙每秒跑{v2}米，两人同时同地同向出发，多少秒后甲第一次追上乙？若背向而行，多少秒相遇？",
        f"400米的环形跑道上，甲每秒{v1}米，乙每秒{v2}米，同时同地同向跑，甲几秒后追上乙？反向跑几秒相遇？",
        f"两人在400米环形跑道上跑步，甲每秒{v1}米，乙每秒{v2}米，同时同地同向出发，甲第一次追上乙要多少秒？背向跑多久相遇？",
        f"环形跑道长400米，甲、乙速度分别为每秒{v1}米、{v2}米，同时同地同向而行，甲多少秒追上乙？相向而行多少秒相遇？",
    ])
    lines = [
        f"{v1} - {v2} = {v1 - v2}米",
        f"{v1} + {v2} = {v1 + v2}米",
        f"400 ÷ {v1 + v2} = {num(t2)}秒",
        f"400 ÷ {v1 - v2} = {num(t1)}秒",
    ]
    return ins, lines, t1


_reg("circle_track", circle_track)


# 49. book x pages, plan a/day, actually 1/n more per day -> days saved
def reading_plan_early(rng):
    a = rng.choice([10, 12, 15, 20, 24, 30])
    n = rng.randint(2, 5)
    k = rng.randint(2, 5)
    x = a * (n + 1) * k
    plan = (n + 1) * k
    extra = Fraction(a, n)
    actual_per = a + extra
    actual_days = n * k
    early = k
    who = rng.choice(["小明", "小红", "小华", "小丽", "小军"])
    ins = rng.choice([
        f"一本书共{x}页，{who}计划每天看{a}页，实际每天比计划多看1/{n}，可以提前几天看完？",
        f"{who}看一本{x}页的书，原计划每天看{a}页，实际每天多看1/{n}，提前几天看完？",
        f"一本{x}页的书，计划每天读{a}页，实际每天读的比计划多1/{n}，可提前多少天读完？",
        f"{who}要看完{x}页的书，计划每天看{a}页，实际每天比计划多1/{n}，能提前几天？",
    ])
    lines = [
        f"{x} ÷ {a} = {plan}天",
        f"{a} × 1/{n} = {num(extra)}页",
        f"{a} + {num(extra)} = {num(actual_per)}页",
        f"{x} ÷ ({num(actual_per)}) = {actual_days}天",
        f"{plan} - {actual_days} = {early}天",
    ]
    return ins, lines, early


_reg("reading_plan_early", reading_plan_early)


# 50. A a days, B b days, cooperate, wage W split by work -> A's share
def work_wage_split(rng):
    a = rng.choice([6, 8, 10, 12, 15, 20])
    b = rng.choice([4, 6, 8, 10, 12])
    for _ in range(50):
        if a != b:
            break
        b = rng.choice([4, 6, 8, 10, 12])
    k = rng.randint(10, 50)
    W = (a + b) * k
    jia = b * k
    who = rng.choice([("甲", "乙"), ("一队", "二队"), ("张师傅", "李师傅"), ("小明", "小红")])
    w1, w2 = who
    ins = rng.choice([
        f"一项工程，{w1}单独做{a}天完成，{w2}单独做{b}天完成，两人合作完成后共得工资{W}元，按工作量分配，{w1}应得多少元？",
        f"{w1}单独完成一项工程要{a}天，{w2}单独做要{b}天，合作完成得工资{W}元，按工作量分，{w1}分得多少元？",
        f"一批零件，{w1}单独做{a}天完成，{w2}单独做{b}天完成，合做完成后得工钱{W}元，按工作量分配，{w1}得多少元？",
        f"一项工程{w1}独做{a}天完成，{w2}独做{b}天完成，合作完工后工资{W}元，按各自工作量分，{w1}应得多少元？",
    ])
    lines = [
        f"1 ÷ {a} = 1/{a}",
        f"1 ÷ {b} = 1/{b}",
        f"1/{a} + 1/{b} = {num(Fraction(a + b, a * b))}",
        f"{W} × 1/{a} = {num(Fraction(W, a))}元",
        f"{num(Fraction(W, a))} ÷ ({num(Fraction(a + b, a * b))}) = {jia}元",
    ]
    return ins, lines, jia


_LABELS["work_wage_split"] = ["甲的效率", "乙的效率", "效率和"]
_reg("work_wage_split", work_wage_split)


# 51. fill pipe a hours, with drain open t hours to empty full pool -> drain alone
def pipe_drain_find_rate(rng):
    pairs = [(a, t) for a in (3, 4, 5, 6, 8, 10, 12)
             for t in (4, 5, 6, 8, 10, 12, 15, 20, 24, 30)]
    a, t = rng.choice(pairs)
    rd = Fraction(1, a) + Fraction(1, t)
    x = Fraction(1, rd)
    ins = rng.choice([
        f"一个水池，进水管{a}小时可注满，若同时打开排水管，{t}小时可将满池水排空，排水管单独开几小时排空满池水？",
        f"水池有一个进水管和一个排水管，单开进水管{a}小时注满，两管同开{t}小时排空满池水，排水管单开几小时排空？",
        f"一水池进水管{a}小时注满，进水管和排水管同时开，{t}小时能把满池水放完，排水管单独开要几小时？",
        f"单开进水管{a}小时可把空池注满，同时打开排水管，{t}小时可把满池水排空，排水管单独开几小时排空满池水？",
    ])
    lines = [
        f"1 ÷ {a} = 1/{a}",
        f"1 ÷ {t} = 1/{t}",
        f"1/{a} + 1/{t} = {num(Fraction(a + t, a * t))}",
        f"1 ÷ ({num(Fraction(a + t, a * t))}) = {num(x)}小时",
    ]
    return ins, lines, x


_LABELS["pipe_drain_find_rate"] = ["进水效率", "每小时净排水", "排水管效率"]
_reg("pipe_drain_find_rate", pipe_drain_find_rate)


# 52. fence against a wall: L meters of fence, length:width a:b -> area
def fence_wall_area(rng):
    a = rng.randint(3, 7)
    b = rng.randint(2, a - 1)
    m = rng.randint(2, 8)
    L = (a + 2 * b) * m
    length = a * m
    width = b * m
    area = length * width
    obj = rng.choice(["菜地", "花圃", "菜园", "苗圃", "草坪"])
    ins = rng.choice([
        f"用篱笆一面靠墙围一个长方形{obj}，篱笆共长{L}米，长与宽的比是{a}:{b}（长平行于墙），{obj}的面积是多少平方米？",
        f"靠墙围一块长方形{obj}，篱笆长{L}米，长和宽的比为{a}:{b}，长的一边靠墙，面积是多少平方米？",
        f"用{L}米篱笆一面靠墙围长方形{obj}，长与宽之比是{a}:{b}（靠墙的一边为长），{obj}面积多少平方米？",
        f"一个长方形{obj}一边靠墙，其余三边用篱笆，篱笆共{L}米，长宽比是{a}:{b}，面积是多少平方米？",
    ])
    lines = [
        f"篱笆的份数 = {a} + 2 × {b} = {a + 2 * b}",
        f"{L} ÷ {a + 2 * b} = {m}米",
        f"{a} × {m} = {length}米",
        f"{b} × {m} = {width}米",
        f"{length} × {width} = {area}平方米",
    ]
    return ins, lines, area


_reg("fence_wall_area", fence_wall_area)


# 53. square side a, largest inscribed circle -> leftover area
def square_inscribed_circle(rng):
    a = rng.choice([4, 6, 8, 10, 12, 14, 16, 20])
    r = a // 2
    circle = Fraction(314, 100) * r * r
    square = a * a
    left = square - circle
    obj = rng.choice(["正方形纸", "正方形铁皮", "正方形木板", "正方形布"])
    ins = rng.choice([
        f"在边长{a}米的{obj}上剪一个最大的圆，剩下的面积是多少平方米？",
        f"一块边长{a}米的{obj}，剪下一个最大的圆，边角料的面积是多少平方米？",
        f"边长{a}米的{obj}中画一个最大的圆，圆以外的面积是多少平方米？",
        f"从边长{a}米的{obj}上截下最大的圆，剩余部分面积多少平方米？",
        f"{obj}边长{a}米，剪去最大的圆后，剩下多少平方米？",
        f"边长{a}米的{obj}剪出最大圆，余下面积多少平方米？",
    ])
    lines = [
        f"{a} ÷ 2 = {r}米",
        f"半径的平方 = {r} × {r} = {r * r}",
        f"3.14 × {r * r} = {num(circle)}平方米",
        f"{a} × {a} = {square}平方米",
        f"{square} - {num(circle)} = {num(left)}平方米",
    ]
    return ins, lines, left


_reg("square_inscribed_circle", square_inscribed_circle)


# 54. sector: radius r, angle n degrees -> area
def sector_area(rng):
    pairs = [(r, n) for r in (6, 10, 12, 15, 20, 30, 40)
             for n in (45, 60, 90, 120, 180, 240, 270)]
    r, n = rng.choice(pairs)
    circle = Fraction(314, 100) * r * r
    frac = Fraction(n, 360)
    sector = circle * frac
    arc = 2 * Fraction(314, 100) * r * frac
    ins = rng.choice([
        f"一个扇形半径是{r}米，圆心角是{n}度，它的面积是多少平方米？",
        f"扇形的半径为{r}米，圆心角{n}度，面积是多少平方米？",
        f"圆心角{n}度、半径{r}米的扇形，面积是多少平方米？",
        f"一个扇形，半径{r}米，圆心角{n}度，求它的面积。",
    ])
    lines = [
        f"3.14 × {r} × {r} = {num(circle)}平方米",
        f"圆心角占比 = {n} ÷ 360 = {num(frac)}",
        f"2 × 3.14 × {r} = {num(2 * Fraction(314, 100) * r)}米",
        f"{num(circle)} × {num(frac)} = {num(sector)}平方米",
    ]
    return ins, lines, sector


_reg("sector_area", sector_area)


# 55. rhombus diagonals d1, d2 -> area -> cost
def rhombus_area_cost(rng):
    d1 = rng.choice([8, 10, 12, 16, 20, 24])
    d2 = rng.choice([6, 8, 10, 12, 16, 20])
    for _ in range(50):
        if d1 != d2:
            break
        d2 = rng.choice([6, 8, 10, 12, 16, 20])
    area = Fraction(d1 * d2, 2)
    c = rng.randint(10, 50)
    cost = area * c
    obj = rng.choice(["菱形花坛", "菱形地砖", "菱形草坪", "菱形铁片"])
    ins = rng.choice([
        f"一个{obj}的两条对角线分别长{d1}米和{d2}米，它的面积是多少平方米？每平方米{c}元，共需多少元？",
        f"菱形的两条对角线是{d1}米和{d2}米，面积是多少平方米？铺每平方米{c}元的材料要多少元？",
        f"一块{obj}，对角线长{d1}米和{d2}米，面积多少平方米？每平方米造价{c}元，一共多少元？",
        f"菱形两条对角线分别为{d1}米、{d2}米，它的面积是多少平方米？每平方米{c}元，总价多少元？",
    ])
    lines = [
        f"对角线的积 = {d1} × {d2} = {d1 * d2}",
        f"{d1 * d2} ÷ 2 = {num(area)}平方米",
        f"{num(area)} × {c} = {num(cost)}元",
    ]
    return ins, lines, cost


_reg("rhombus_area_cost", rhombus_area_cost)


# 56. L-shaped field: big a x b minus corner c x d -> area -> harvest
def l_shape_harvest(rng):
    a = rng.randint(10, 20)
    b = rng.randint(8, 16)
    c = rng.randint(3, a - 3)
    d = rng.randint(3, b - 3)
    x = rng.randint(2, 8)
    big = a * b
    small = c * d
    area = big - small
    harvest = area * x
    obj = rng.choice(["菜地", "花圃", "玉米地", "麦田", "菜园"])
    ins = rng.choice([
        f"一块L形{obj}，可以看成一个长{a}米、宽{b}米的长方形缺了一个长{c}米、宽{d}米的角，{obj}的面积是多少平方米？每平方米收{x}千克，共收多少千克？",
        f"一块{obj}呈L形，大长方形长{a}米宽{b}米，凹进去的部分长{c}米宽{d}米，面积是多少平方米？每平方米产{x}千克，总产量多少千克？",
        f"L形{obj}的尺寸是：外长{a}米、外宽{b}米，缺口长{c}米、宽{d}米，这块{obj}面积多少平方米？每平方米收{x}千克，一共收多少千克？",
        f"一块L形{obj}，由长{a}米宽{b}米的长方形去掉长{c}米宽{d}米的小长方形得到，面积是多少平方米？每平方米收{x}千克，共收多少千克？",
    ])
    lines = [
        f"{a} × {b} = {big}平方米",
        f"{c} × {d} = {small}平方米",
        f"{big} - {small} = {area}平方米",
        f"{area} × {x} = {harvest}千克",
    ]
    return ins, lines, harvest


_reg("l_shape_harvest", l_shape_harvest)


# 57. map scale: d cm on map, scale 1:n -> actual km, car speed v -> hours
def map_scale_distance(rng):
    triples = [(d, n, v) for d in (4, 5, 6, 8, 10, 12)
               for n in (500000, 1000000, 2000000, 4000000)
               for v in (40, 50, 60, 80, 100)]
    d, n, v = rng.choice(triples)
    cm = d * n
    km = Fraction(cm, 100000)
    m = km / v
    ins = rng.choice([
        f"在比例尺1:{n}的地图上，量得两地距离{d}厘米，一辆汽车每小时行{v}千米，从一地到另一地需要多少小时？",
        f"地图比例尺是1:{n}，量得甲乙两地相距{d}厘米，汽车以每小时{v}千米的速度行驶，几小时到达？",
        f"一幅地图的比例尺为1:{n}，图上两地距离{d}厘米，开车每小时{v}千米，需要多少小时？",
        f"在1:{n}的地图上，两地相距{d}厘米，若汽车每小时行{v}千米，多长时间能到达？",
    ])
    lines = [
        f"{d} × {n} = {cm}厘米",
        f"{cm} ÷ 100000 = {num(km)}千米",
        f"{num(km)} ÷ {v} = {num(m)}小时",
    ]
    return ins, lines, m


_reg("map_scale_distance", map_scale_distance)


# 58. square diagonal d -> area -> harvest -> money
def square_diagonal_area(rng):
    d = rng.choice([4, 6, 8, 10, 12, 14, 16, 20])
    area = Fraction(d * d, 2)
    x = rng.randint(2, 8)
    y = rng.randint(2, 6)
    kg = area * x
    money = kg * y
    obj = rng.choice(["菜地", "花圃", "草坪", "苗圃"])
    ins = rng.choice([
        f"一块正方形{obj}的对角线长{d}米，它的面积是多少平方米？每平方米收菜{x}千克，每千克卖{y}元，这块{obj}的菜共可卖多少元？",
        f"正方形{obj}对角线长{d}米，面积是多少平方米？每平方米产{x}千克，每千克{y}元，总收入多少元？",
        f"一块正方形{obj}，对角线长{d}米，面积多少平方米？每平方米收{x}千克菜，每千克{y}元，一共卖多少元？",
        f"正方形{obj}的对角线是{d}米，它的面积是多少平方米？每平方米收{x}千克，每千克售价{y}元，共收入多少元？",
    ])
    lines = [
        f"对角线的平方 = {d} × {d} = {d * d}",
        f"{d * d} ÷ 2 = {num(area)}平方米",
        f"{num(area)} × {x} = {num(kg)}千克",
        f"{num(kg)} × {y} = {num(money)}元",
    ]
    return ins, lines, money


_reg("square_diagonal_area", square_diagonal_area)


# 59. clock angle at H:30
def clock_angle(rng):
    H = rng.randint(1, 11)
    hour = 30 * H + 15
    minute = 180
    diff = abs(hour - minute)
    angle = min(diff, 360 - diff)
    who = rng.choice(["小明", "小红", "小华", "小丽", "小军"])
    ins = rng.choice([
        f"{H}点30分时，钟面上时针和分针的夹角是多少度？",
        f"钟面上{H}时30分，时针与分针所成的较小角是多少度？",
        f"{who}看到钟面显示{H}:30，这时时针和分针的夹角是多少度？",
        f"下午{H}点30分，钟面上时针与分针的夹角是多少度？",
        f"{who}问：{H}点30分的时候，时针和分针的夹角是多少度？",
        f"晚上{H}点30分，钟面上时针与分针所成的较小角是多少度？",
        f"{who}在{H}点30分看钟，时针和分针的夹角是多少度？",
        f"{who}想知道{H}时30分时针与分针的夹角是多少度？",
    ])
    lines = [
        f"360 ÷ 12 = 30度",
        f"30 × {H} = {30 * H}度",
        f"时针每分走的度数 = 30 ÷ 60 = 1/2",
        f"1/2 × 30 = 15度",
        f"{30 * H} + 15 = {hour}度",
        f"360 ÷ 2 = 180度",
        (f"180 - {hour} = {angle}度" if hour <= 180 else f"{hour} - 180 = {angle}度"),
    ]
    return ins, lines, angle


_reg("clock_angle", clock_angle)


# 60. clock hands overlap after H o'clock
def clock_overlap(rng):
    H = rng.randint(1, 11)
    gap = Fraction(H, 12)
    speed_diff = Fraction(11, 12)
    t_hours = gap / speed_diff
    t_min = t_hours * 60
    who = rng.choice(["小明", "小红", "小华", "小丽", "小军"])
    ins = rng.choice([
        f"{H}点整以后，时针和分针第一次重合是几点几分？（结果用分数表示）",
        f"钟面上{H}时整，再过多少分钟时针与分针第一次重合？",
        f"{H}点后，分针和时针第一次重合在什么时刻？（分钟用分数表示）",
        f"从{H}点整开始，经过多少分钟时针与分针第一次重合？",
        f"{who}问：{H}点整以后，时针和分针第一次重合是几点几分？",
        f"{who}看到钟面上{H}时整，再过多少分钟时针与分针第一次重合？",
        f"{who}想知道{H}点后分针和时针第一次重合在什么时刻？",
        f"{who}从{H}点整开始计时，经过多少分钟时针与分针第一次重合？",
    ])
    lines = [
        f"1 ÷ 12 = 1/12",
        f"1 - 1/12 = 11/12",
        f"{H} ÷ 12 = {num(gap)}",
        f"{num(gap)} ÷ (11/12) = {num(t_hours)}小时",
        f"{num(t_hours)} × 60 = {num(t_min)}分",
    ]
    return ins, lines, t_min


_LABELS["clock_overlap"] = ["时针速度", "速度差", "初始间隔"]
_reg("clock_overlap", clock_overlap)


# 61. five Wednesdays in a month, date sum S -> first Wednesday
def calendar_five_weekdays(rng):
    mid = rng.choice([15, 16, 17])
    S = 5 * mid
    first = mid - 14
    last = mid + 14
    day = rng.choice(["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"])
    ins = rng.choice([
        f"某月有5个{day}，它们的日期数相加的和是{S}，这个月第一个{day}是几号？",
        f"一个月里有5个{day}，这5个日期的和是{S}，第一个{day}是几号？",
        f"某月的日历上有5个{day}，日期之和为{S}，问第一个{day}是几号？",
        f"已知某月有5个{day}，且它们的日期和是{S}，这个月第一个{day}是几号？",
        f"某月有5个{day}，5个日期数之和是{S}，第一个{day}是几号？",
        f"一个月中出现了5次{day}，日期和为{S}，第一个{day}是几号？",
        f"某月的5个{day}日期相加得{S}，这个月第一个{day}是几号？",
        f"日历上某月有5个{day}，它们的日期和是{S}，第一个{day}是几号？",
        f"某月里共有5个{day}，日期数之和为{S}，第一个{day}是几号？",
        f"查日历发现某月有5个{day}，日期相加得{S}，第一个{day}是几号？",
        f"某月的5个{day}分别是哪几号？它们的和是{S}，第一个{day}是几号？",
    ])
    lines = [
        f"{S} ÷ 5 = {mid}号",
        f"7 × 4 = 28天",
        f"{mid} + 14 = {last}号",
        f"{mid} - 14 = {first}号",
    ]
    return ins, lines, first


_reg("calendar_five_weekdays", calendar_five_weekdays)


# 62. page numbering uses N digits -> number of pages
def page_numbering(rng):
    k = rng.randint(10, 300)
    N = 189 + 3 * k
    pages = 99 + k
    ins = rng.choice([
        f"给一本书编页码，一共用了{N}个数字，这本书共有多少页？",
        f"一本故事书的页码共用了{N}个数字，这本书有多少页？",
        f"编一本书的页码用了{N}个数字，这本书共多少页？",
        f"一本书从第1页开始编页码，共用了{N}个数字，这本书共有多少页？",
    ])
    lines = [
        f"9 × 1 = 9个",
        f"90 × 2 = 180个",
        f"9 + 180 = 189个",
        f"{N} - 189 = {N - 189}个",
        f"{N - 189} ÷ 3 = {k}页",
        f"99 + {k} = {pages}页",
    ]
    return ins, lines, pages


_reg("page_numbering", page_numbering)


# 63. three-digit number: tens digit b, units is c times hundreds, digit sum S -> number
def three_digit_ratio(rng):
    c = rng.choice([2, 3])
    x = rng.randint(1, 9 // c)
    b = rng.randint(1, 9)
    S = b + (c + 1) * x
    ge = c * x
    number = x * 100 + b * 10 + ge
    ins = rng.choice([
        f"一个三位数，十位数字是{b}，个位数字是百位数字的{c}倍，三个数位上的数字之和是{S}，这个三位数是多少？",
        f"一个三位数，十位上是{b}，个位是百位的{c}倍，各位数字之和为{S}，求这个三位数。",
        f"三位数的十位数字是{b}，个位数字是百位数字的{c}倍，三个数字的和是{S}，这个数是多少？",
        f"一个三位数，十位为{b}，个位数字是百位数字的{c}倍，数位上数字之和是{S}，这个三位数是多少？",
    ])
    lines = [
        f"份数和 = {c} + 1 = {c + 1}",
        f"百位与个位的和 = {S} - {b} = {S - b}",
        f"百位数字 = {S - b} ÷ {c + 1} = {x}",
        f"个位数字 = {c} × {x} = {ge}",
        f"三位数 = {x} × 100 + {b} × 10 + {ge} = {number}",
    ]
    return ins, lines, number


_reg("three_digit_ratio", three_digit_ratio)


# 64. a boys, b girls, every boy-girl pair shakes hands, t seconds each -> total minutes
def handshakes_mixed(rng):
    a = rng.randint(3, 10)
    b = rng.randint(3, 10)
    t = rng.choice([2, 3, 4, 5, 6])
    pairs = a * b
    sec = pairs * t
    minutes = Fraction(sec, 60)
    ins = rng.choice([
        f"一次聚会有{a}个男生和{b}个女生，每个男生和每个女生各握手一次，每次握手{t}秒，一共握手多少分钟？",
        f"联欢会上有{a}名男生、{b}名女生，每位男生与每位女生握手一次，每次{t}秒，握手总时间是多少分钟？",
        f"小组里有{a}个男生和{b}个女生，男女生之间两两握手一次，每次握手用时{t}秒，全部握手共多少分钟？",
        f"一次活动中{a}个男生和{b}个女生相互认识，每个男生和每个女生握一次手，每次{t}秒，共需多少分钟？",
    ])
    lines = [
        f"{a} × {b} = {pairs}次",
        f"{pairs} × {t} = {sec}秒",
        f"{sec} ÷ 60 = {num(minutes)}分",
    ]
    return ins, lines, minutes


_reg("handshakes_mixed", handshakes_mixed)


# 65. a red, b blue balls -> probability of red; add x red -> new probability
def probability_add(rng):
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    x = rng.randint(1, 9)
    total = a + b
    p1 = Fraction(a, total)
    new_red = a + x
    new_total = total + x
    p2 = Fraction(new_red, new_total)
    obj = rng.choice(["球", "卡片", "棋子", "弹珠"])
    color = rng.choice(["红", "蓝", "白", "黄"])
    ins = rng.choice([
        f"袋子里有{a}个{color}{obj}和{b}个其他颜色的{obj}，任意摸一个，摸到{color}{obj}的可能性是几分之几？再放入{x}个{color}{obj}后呢？",
        f"盒中有{a}个{color}{obj}、{b}个别的{obj}，摸出{color}{obj}的可能性是多少？再加入{x}个{color}{obj}，可能性变为多少？",
        f"口袋里装了{a}个{color}{obj}和{b}个其他{obj}，摸到{color}{obj}的可能性是几分之几？若再放入{x}个{color}{obj}，可能性是多少？",
        f"袋子中有{a}个{color}{obj}和{b}个别的{obj}，随便拿一个是{color}{obj}的可能性为几分之几？再添{x}个{color}{obj}后呢？",
    ])
    lines = [
        f"{a} + {b} = {total}个",
        f"原来的可能性 = {a} ÷ {total} = {num(p1)}",
        f"{a} + {x} = {new_red}个",
        f"{total} + {x} = {new_total}个",
        f"后来的可能性 = {new_red} ÷ {new_total} = {num(p2)}",
    ]
    return ins, lines, p2


_reg("probability_add", probability_add)


# 66. rows of a with surplus b, rows of c with shortage d -> people
def surplus_shortage(rng):
    for _ in range(50):
        a = rng.randint(4, 9)
        c = rng.randint(a + 1, a + 4)
        b = rng.randint(2, 12)
        d = rng.randint(2, 12)
        if (b + d) % (c - a) == 0:
            break
    rows = (b + d) // (c - a)
    people = a * rows + b
    obj = rng.choice(["同学", "小朋友", "学生", "队员"])
    ins = rng.choice([
        f"{obj}们排队，每行站{a}人则多{b}人，每行站{c}人则少{d}人，共有多少{obj}？",
        f"一批{obj}排队，每行{a}人多出{b}人，每行{c}人还差{d}人，这批{obj}有多少人？",
        f"{obj}做操，每行{a}人余{b}人，每行{c}人缺{d}人，一共有多少{obj}？",
        f"老师安排{obj}排队，每行站{a}人多{b}人，每行站{c}人少{d}人，共有多少人？",
    ])
    lines = [
        f"{b} + {d} = {b + d}人",
        f"每行的差 = {c} - {a} = {c - a}",
        f"{b + d} ÷ {c - a} = {rows}行",
        f"{a} × {rows} + {b} = {people}人",
    ]
    return ins, lines, people


_reg("surplus_shortage", surplus_shortage)


# 67. chickens and rabbits: H heads, L legs -> rabbits
def chicken_rabbit(rng):
    H = rng.randint(8, 30)
    chicken = rng.randint(2, H - 2)
    rabbit = H - chicken
    L = 2 * chicken + 4 * rabbit
    ins = rng.choice([
        f"鸡兔同笼，共有{H}个头、{L}条腿，兔有多少只？",
        f"笼子里有鸡和兔，共{H}个头、{L}条腿，兔子有多少只？",
        f"鸡和兔关在一个笼子里，数头{H}个，数腿{L}条，兔有多少只？",
        f"鸡兔同笼，头共{H}个，腿共{L}条，兔子有多少只？",
    ])
    lines = [
        f"{H} × 4 = {H * 4}条",
        f"{H * 4} - {L} = {H * 4 - L}条",
        f"4 - 2 = 2条",
        f"{H * 4 - L} ÷ 2 = {chicken}只",
        f"{H} - {chicken} = {rabbit}只",
    ]
    return ins, lines, rabbit


_reg("chicken_rabbit", chicken_rabbit)


# 68. quiz: c questions, right +a, wrong -b, score d -> right count
def quiz_scoring(rng):
    a = rng.choice([3, 5, 8, 10])
    b = rng.choice([1, 2, 3, 5])
    c = rng.randint(8, 20)
    wrong = rng.randint(1, max(1, (a * c - 1) // (a + b)))
    right = c - wrong
    d = a * right - b * wrong
    ins = rng.choice([
        f"一次竞赛共{c}题，答对一题得{a}分，答错一题扣{b}分，小明得了{d}分，他答对了几题？",
        f"数学竞赛有{c}道题，答对加{a}分，答错减{b}分，小红得了{d}分，她答对多少题？",
        f"抢答赛共{c}题，答对一题得{a}分，答错一题倒扣{b}分，小华得{d}分，他答对了几题？",
        f"一份试卷共{c}题，答对每题{a}分，答错每题扣{b}分，小丽得{d}分，她答对了几题？",
    ])
    lines = [
        f"{c} × {a} = {c * a}分",
        f"{c * a} - {d} = {c * a - d}分",
        f"{a} + {b} = {a + b}分",
        f"{c * a - d} ÷ {a + b} = {wrong}题",
        f"{c} - {wrong} = {right}题",
    ]
    return ins, lines, right


_reg("quiz_scoring", quiz_scoring)


# 69. rope measure well depth: fold in half surplus x, fold in thirds shortage y -> depth
def rope_measure_well(rng):
    x = rng.randint(1, 8)
    y = rng.randint(1, 8)
    L = 6 * (x + y)
    half = L // 2
    depth = half - x
    ins = rng.choice([
        f"用一根绳子测井深，把绳子对折来量，井外余{x}米；把绳子三折来量，离井口还差{y}米，井深多少米？",
        f"用绳测井深，对折量井外余{x}米，三折量少{y}米，井深多少米？",
        f"一根绳子测井深，对折后井外剩{x}米，三折后还差{y}米到井口，井深多少米？",
        f"测井深：绳对折余{x}米，绳三折缺{y}米，井深多少米？",
    ])
    lines = [
        f"份数差 = 1/2 - 1/3 = 1/6",
        f"{x} + {y} = {x + y}米",
        f"{x + y} ÷ (1/6) = {L}米",
        f"{L} ÷ 2 = {half}米",
        f"{half} - {x} = {depth}米",
    ]
    return ins, lines, depth


_reg("rope_measure_well", rope_measure_well)


# 70. rope folded n times, each segment x meters, cut p% -> left
def rope_folded(rng):
    n = rng.choice([2, 3, 4])
    x = rng.randint(2, 12)
    p = rng.choice([10, 20, 25, 40, 50])
    seg = 2 ** n
    L = seg * x
    cut = Fraction(L * p, 100)
    left = L - cut
    obj = rng.choice(["绳子", "彩带", "铁丝", "电线", "布条"])
    ins = rng.choice([
        f"一根{obj}对折{n}次后，每段长{x}米，这根{obj}原来长多少米？剪去全长的{p}%后还剩多少米？",
        f"一根{obj}对折{n}次后量得每段{x}米，{obj}原长多少米？用去{p}%后还剩多少米？",
        f"把一根{obj}对折{n}次，这时每段长{x}米，这根{obj}长多少米？剪去{p}%后剩多少米？",
        f"一根{obj}对折{n}次后每段{x}米，原来长多少米？剪掉{p}%后还剩多少米？",
    ])
    lines = [
        f"对折{n}次后的段数 = 2{' × 2' * (n - 1)} = {seg}段",
        f"{seg} × {x} = {L}米",
        f"{L} × {p}/100 = {num(cut)}米",
        f"{L} - {num(cut)} = {num(left)}米",
    ]
    return ins, lines, left


_reg("rope_folded", rope_folded)


# 71. paper 0.1mm thick, folded n times -> thickness in cm
def paper_fold_thickness(rng):
    n = rng.choice([3, 4, 5, 6, 7, 8])
    t = rng.choice([Fraction(1, 10), Fraction(1, 5), Fraction(1, 20)])
    obj = rng.choice(["纸", "卡纸", "纸片", "彩纸"])
    thick = t * (2 ** n)
    cm = thick / 10
    ins = rng.choice([
        f"一张厚{_d(t)}毫米的{obj}，把它对折{n}次后，厚度是多少毫米？合多少厘米？",
        f"一张厚度{_d(t)}毫米的{obj}，对折{n}次后有多厚？折合多少厘米？",
        f"{obj}厚{_d(t)}毫米，对折{n}次后，厚度达到多少毫米？是多少厘米？",
        f"把一张{_d(t)}毫米厚的{obj}对折{n}次，厚多少毫米？合多少厘米？",
        f"一张{obj}的厚度是{_d(t)}毫米，对折{n}次后厚多少毫米？合多少厘米？",
        f"厚度{_d(t)}毫米的{obj}对折{n}次，厚度变为多少毫米？是多少厘米？",
    ])
    lines = [
        f"对折{n}次后的层数 = 2{' × 2' * (n - 1)} = {2 ** n}层",
        f"{_d(t)} × {2 ** n} = {_d(thick)}毫米",
        f"{_d(thick)} ÷ 10 = {_d(cm)}厘米",
    ]
    return ins, lines, cm


_reg("paper_fold_thickness", paper_fold_thickness)


# 72. bacteria double every hour, after n hours x -> original count
def bacteria_double(rng):
    n = rng.choice([3, 4, 5, 6])
    k = rng.randint(1, 12)
    x = (2 ** n) * 100 * k
    orig = 100 * k
    ins = rng.choice([
        f"一种细菌每小时数量翻倍，{n}小时后达到{x}个，原来有多少个？",
        f"某种细菌每小时增加一倍，{n}小时后有{x}个，原来有多少个？",
        f"细菌的数量每小时翻一番，{n}小时后是{x}个，原来有多少个？",
        f"一种细菌每小时数量变为原来的2倍，{n}小时后达到{x}个，原来有多少个？",
    ])
    lines = [
        f"{n}小时后的倍数 = 2{' × 2' * (n - 1)} = {2 ** n}倍",
        f"{x} ÷ {2 ** n} = {orig}个",
    ]
    return ins, lines, orig


_reg("bacteria_double", bacteria_double)


# 73. water lily doubles daily, full on day x with S area -> day of 1/4 pool
def water_lily(rng):
    x = rng.randint(8, 20)
    S = rng.choice([100, 200, 400, 500, 800, 1000])
    half = Fraction(S, 2)
    quarter = Fraction(S, 4)
    ins = rng.choice([
        f"池塘里的睡莲每天面积翻倍，第{x}天长满整个池塘（面积{S}平方米），第几天长满池塘的1/4？",
        f"睡莲的面积每天增加一倍，第{x}天铺满{S}平方米的池塘，池塘的1/4是第几天铺满的？",
        f"一种水生植物每天面积翻倍，第{x}天覆盖{S}平方米的池塘，它在第几天覆盖池塘的1/4？",
        f"池塘中睡莲每天长大一倍，第{x}天铺满全池{S}平方米，问第几天铺满池塘的1/4？",
    ])
    lines = [
        f"{S} ÷ 2 = {num(half)}平方米",
        f"{num(half)} ÷ 2 = {num(quarter)}平方米",
        f"{num(quarter)} × 4 = {S}平方米",
        f"{x} - 1 = {x - 1}天",
        f"{x - 1} - 1 = {x - 2}天",
    ]
    return ins, lines, x - 2


_reg("water_lily", water_lily)


# 74. average of three numbers is a, two of them b and c -> third
def average_find_third(rng):
    a = rng.randint(70, 95)
    b = rng.randint(60, 100)
    c = rng.randint(60, 100)
    third = 3 * a - b - c
    for _ in range(50):
        if third > 0:
            break
        b = rng.randint(60, 100)
        c = rng.randint(60, 100)
        third = 3 * a - b - c
    obj = rng.choice(["数", "成绩", "身高", "体重"])
    who = rng.choice(["小明", "小红", "小华", "小丽"])
    ins = rng.choice([
        f"三个数的平均数是{a}，其中两个数是{b}和{c}，第三个数是多少？",
        f"{who}三次测验的平均分是{a}分，前两次分别是{b}分和{c}分，第三次是多少分？",
        f"三个{obj}的平均数为{a}，已知两个分别是{b}和{c}，第三个是多少？",
        f"甲、乙、丙三个数的平均数是{a}，甲是{b}，乙是{c}，丙是多少？",
    ])
    lines = [
        f"三个数的和 = {a} × 3 = {3 * a}",
        f"两数之和 = {b} + {c} = {b + c}",
        f"第三个数 = {3 * a} - {b + c} = {third}",
    ]
    return ins, lines, third


_reg("average_find_third", average_find_third)


# 75. two groups: m people avg a, n people avg b -> overall average
def average_two_groups(rng):
    m = rng.randint(3, 12)
    n = rng.randint(3, 12)
    a = rng.randint(70, 95)
    b = rng.randint(70, 95)
    s1 = m * a
    s2 = n * b
    total = s1 + s2
    count = m + n
    avg = Fraction(total, count)
    obj = rng.choice(["学生", "同学", "选手", "工人"])
    ins = rng.choice([
        f"甲组有{m}个{obj}，平均成绩{a}分；乙组有{n}个{obj}，平均成绩{b}分，两组合起来平均成绩是多少分？",
        f"第一组{m}人平均{a}分，第二组{n}人平均{b}分，全体的平均分是多少？",
        f"甲班{m}名{obj}平均身高{a}厘米，乙班{n}名{obj}平均身高{b}厘米，两班平均身高多少厘米？",
        f"车间里男工{m}人平均日产{a}个零件，女工{n}人平均日产{b}个零件，全车间平均每人日产多少个？",
    ])
    lines = [
        f"{m} × {a} = {s1}分",
        f"{n} × {b} = {s2}分",
        f"{s1} + {s2} = {total}分",
        f"{m} + {n} = {count}人",
        f"{total} ÷ {count} = {num(avg)}分",
    ]
    return ins, lines, avg


_reg("average_two_groups", average_two_groups)


if __name__ == "__main__":
    rng = random.Random(3)
    from run_math_short import verify
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines}"
            ok += 1
    print(f"L3 ext5 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
