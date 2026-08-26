#!/usr/bin/env python3
"""L2 ext2: 53 verified 3-4 step word problems.

Families: compound unit conversions (week/day, hour/min, km/m, kg/g, ton/kg,
yuan/jiao), perimeters/areas with derived sides, times-more/less, compare-then-
combine chains, money (tickets, membership, discount, save/spend, split, equal-
give), rate/work, meeting speed, remainders (group-fill, max-buy), ages, elapsed
time, interval counting (stairs, tree line, lanterns), library flow, sum-and-
difference, sale chains, average targets, distribution, volume conversion.
All lines are independently verified by run_math_short.verify.
"""
import random
from fractions import Fraction
from mathcommon import (ANIMALS, FRUITS, GOODS, NAMES, PLACE, STATIONERY,
                        UNIT_N, num)

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L2", name, fn))


# 1. 星期↔天: w周零d天 = w*7+d 天
def weeks_days(rng):
    w = rng.randint(2, 6)
    d = rng.randint(1, 6)
    total = w * 7 + d
    t = rng.randrange(3)
    ins = [
        f"假期有{w}个星期零{d}天，一共是多少天？",
        f"一项工程做了{w}周零{d}天，一共用了多少天？",
        f"小明在奶奶家住了{w}个星期多{d}天，他一共住了多少天？",
    ][t]
    lines = [
        f"{w} × 7 = {w * 7}天",
        f"{w * 7} + {d} = {total}天",
    ]
    return ins, lines, total


_reg("weeks_days", weeks_days)


# 2. 天↔时 比较: d天 vs h小时, 差多少
def days_hours_compare(rng):
    d = rng.randint(2, 5)
    h = rng.randint(1, 23)
    dh = d * 24
    diff = dh - h
    t = rng.randrange(3)
    ins = [
        f"{d}天和{h}小时，哪个时间长？长多少小时？",
        f"一台机器连续运转{d}天，另一台运转{h}小时，前者比后者多多少小时？",
        f"小明旅游用了{d}天，小红旅游用了{h}小时，小明比小红多多少小时？",
    ][t]
    lines = [
        f"{d} × 24 = {dh}时",
        f"{dh} - {h} = {diff}时",
    ]
    return ins, lines, diff


_reg("days_hours_compare", days_hours_compare)


# 3. 时↔分: h小时m分 = h*60+m 分
def hours_minutes(rng):
    h = rng.randint(2, 6)
    m = rng.randint(1, 59)
    total = h * 60 + m
    t = rng.randrange(3)
    ins = [
        f"{h}小时{m}分一共是多少分钟？",
        f"一场报告进行了{h}小时{m}分，合多少分钟？",
        f"小明做作业用了{h}小时{m}分，一共是多少分钟？",
    ][t]
    lines = [
        f"{h} × 60 = {h * 60}分",
        f"{h * 60} + {m} = {total}分",
    ]
    return ins, lines, total


_reg("hours_minutes", hours_minutes)


# 4. 分↔秒 比较: m分 vs s秒, 差多少
def minutes_seconds_compare(rng):
    m = rng.randint(2, 5)
    s = rng.randint(1, 59)
    ms = m * 60
    diff = ms - s
    t = rng.randrange(3)
    ins = [
        f"{m}分钟和{s}秒，哪个时间长？长多少秒？",
        f"小红跑一圈用{m}分钟，小丽用{s}秒，小红比小丽多用多少秒？",
        f"一节课{m}分钟，课间休息{s}秒，上课比休息长多少秒？",
    ][t]
    lines = [
        f"{m} × 60 = {ms}秒",
        f"{ms} - {s} = {diff}秒",
    ]
    return ins, lines, diff


_reg("minutes_seconds_compare", minutes_seconds_compare)


# 5. 千米↔米: km千米m米 = km*1000+m 米
def km_m_add(rng):
    km = rng.randint(2, 8)
    m = rng.randint(1, 999)
    total = km * 1000 + m
    t = rng.randrange(3)
    ins = [
        f"从学校到体育馆是{km}千米{m}米，合多少米？",
        f"一条公路长{km}千米{m}米，一共是多少米？",
        f"小明骑车行了{km}千米{m}米，他行了多少米？",
    ][t]
    lines = [
        f"{km} × 1000 = {km * 1000}米",
        f"{km * 1000} + {m} = {total}米",
    ]
    return ins, lines, total


_reg("km_m_add", km_m_add)


# 6. 米↔厘米 比较: m米 vs cm厘米, 差多少
def m_cm_compare(rng):
    m = rng.randint(2, 5)
    mc = m * 100
    extra = rng.randint(10, 150)
    cm = mc + extra
    t = rng.randrange(3)
    ins = [
        f"红绳长{m}米，蓝绳长{cm}厘米，蓝绳比红绳长多少厘米？",
        f"小明身高{m}米，哥哥身高{cm}厘米，哥哥比小明高多少厘米？",
        f"一根竹竿长{m}米，另一根长{cm}厘米，两根相差多少厘米？",
    ][t]
    lines = [
        f"{m} × 100 = {mc}厘米",
        f"{cm} - {mc} = {extra}厘米",
    ]
    return ins, lines, extra


_reg("m_cm_compare", m_cm_compare)


# 7. 千克↔克: kg千克g克 = kg*1000+g 克
def kg_g_add(rng):
    kg = rng.randint(2, 8)
    g = rng.randint(1, 999)
    total = kg * 1000 + g
    t = rng.randrange(3)
    ins = [
        f"一袋大米重{kg}千克{g}克，合多少克？",
        f"小明的体重是{kg}千克{g}克，一共是多少克？",
        f"一筐水果重{kg}千克{g}克，共重多少克？",
    ][t]
    lines = [
        f"{kg} × 1000 = {kg * 1000}克",
        f"{kg * 1000} + {g} = {total}克",
    ]
    return ins, lines, total


_reg("kg_g_add", kg_g_add)


# 8. 吨↔千克 比较: t吨 vs kg千克, 差多少
def ton_kg_compare(rng):
    ton = rng.randint(2, 6)
    kg = rng.randint(100, 900)
    tk = ton * 1000
    diff = tk - kg
    t = rng.randrange(3)
    ins = [
        f"一头大象重{ton}吨，一头牛重{kg}千克，大象比牛重多少千克？",
        f"一辆卡车装货{ton}吨，一辆三轮车装货{kg}千克，卡车比三轮车多装多少千克？",
        f"一堆煤重{ton}吨，一堆沙子重{kg}千克，煤比沙子重多少千克？",
    ][t]
    lines = [
        f"{ton} × 1000 = {tk}千克",
        f"{tk} - {kg} = {diff}千克",
    ]
    return ins, lines, diff


_reg("ton_kg_compare", ton_kg_compare)


# 9. 元↔角: y元j角 = y*10+j 角
def yuan_jiao_add(rng):
    y = rng.randint(2, 9)
    j = rng.randint(1, 9)
    total = y * 10 + j
    t = rng.randrange(3)
    ins = [
        f"{y}元{j}角一共是多少角？",
        f"小明有{y}元{j}角零花钱，合多少角？",
        f"一本书售价{y}元{j}角，是多少角？",
    ][t]
    lines = [
        f"{y} × 10 = {y * 10}角",
        f"{y * 10} + {j} = {total}角",
    ]
    return ins, lines, total


_reg("yuan_jiao_add", yuan_jiao_add)


# 10. 正方形周长 × 圈数
def square_walk_rounds(rng):
    a = rng.randint(5, 30)
    rounds = rng.randint(3, 5)
    per = a * 4
    total = per * rounds
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"正方形花坛边长{a}米，{n1}绕它走了{rounds}圈，一共走了多少米？",
        f"一个正方形操场边长{a}米，小明跑了{rounds}圈，共跑多少米？",
        f"正方形池塘边长{a}米，沿池边走{rounds}圈，一共走多少米？",
    ][t]
    lines = [
        f"{a} × 4 = {per}米",
        f"{per} × {rounds} = {total}米",
    ]
    return ins, lines, total


_reg("square_walk_rounds", square_walk_rounds)


# 11. 长方形周长: 长已知, 宽是长的一半
def rect_perimeter_half(rng):
    a = rng.randint(6, 40) * 2
    w = a // 2
    p = (a + w) * 2
    t = rng.randrange(3)
    ins = [
        f"一块长方形菜地长{a}米，宽是长的一半，周长是多少米？",
        f"长方形的长是{a}米，宽正好是长的一半，它的周长是多少米？",
        f"教室地面是长方形，长{a}米，宽是长的一半，周长是多少米？",
    ][t]
    lines = [
        f"宽 = {a} ÷ 2 = {w}米",
        f"周长 = ({a} + {w}) × 2 = {p}米",
    ]
    return ins, lines, p


_reg("rect_perimeter_half", rect_perimeter_half)


# 12. 长方形面积 × 每平方米人数
def rect_area_people(rng):
    a = rng.randint(6, 15)
    b = rng.randint(5, 12)
    n = rng.randint(3, 6)
    area = a * b
    total = area * n
    t = rng.randrange(3)
    ins = [
        f"教室长{a}米、宽{b}米，每平方米站{n}人，一共可以站多少人？",
        f"一块长方形场地长{a}米、宽{b}米，每平方米放{n}把椅子，共放多少把？",
        f"长方形草坪长{a}米、宽{b}米，每平方米种{n}棵树苗，一共种多少棵？",
    ][t]
    lines = [
        f"面积 = {a} × {b} = {area}平方米",
        f"{area} × {n} = {total}人",
    ]
    return ins, lines, total


_reg("rect_area_people", rect_area_people)


# 13. 正方形面积 × 每平方米棵数
def square_area_flowers(rng):
    a = rng.randint(4, 12)
    b = rng.randint(3, 8)
    area = a * a
    total = area * b
    t = rng.randrange(3)
    ins = [
        f"正方形花圃边长{a}米，每平方米种{b}棵花，一共种多少棵？",
        f"一个正方形鱼池边长{a}米，每平方米养{b}条鱼，共养多少条？",
        f"正方形广场边长{a}米，每平方米铺{b}块地砖，共铺多少块？",
    ][t]
    lines = [
        f"面积 = {a} × {a} = {area}平方米",
        f"{area} × {b} = {total}棵",
    ]
    return ins, lines, total


_reg("square_area_flowers", square_area_flowers)


# 14. 正方形篱笆费用
def fence_cost(rng):
    a = rng.randint(5, 20)
    b = rng.randint(3, 15)
    per = a * 4
    cost = per * b
    t = rng.randrange(3)
    ins = [
        f"正方形菜地边长{a}米，四周围一圈篱笆，每米篱笆{b}元，一共要多少元？",
        f"给边长{a}米的正方形花园围栅栏，每米{b}元，共需多少元？",
        f"正方形鱼塘边长{a}米，沿塘边修护栏，每米{b}元，一共多少元？",
    ][t]
    lines = [
        f"{a} × 4 = {per}米",
        f"{per} × {b} = {cost}元",
    ]
    return ins, lines, cost


_reg("fence_cost", fence_cost)


# 15. 铁丝围长方形求宽
def wire_rect_width(rng):
    long = rng.randint(5, 15)
    half = long + rng.randint(2, 10)
    wire = half * 2
    width = half - long
    t = rng.randrange(3)
    ins = [
        f"用一根长{wire}厘米的铁丝围成一个长方形，长是{long}厘米，宽是多少厘米？",
        f"一根铁丝长{wire}厘米，正好围成一个长{long}厘米的长方形，宽是多少厘米？",
        f"长方形的周长是{wire}厘米，长是{long}厘米，宽是多少厘米？",
    ][t]
    lines = [
        f"{wire} ÷ 2 = {half}厘米",
        f"{half} - {long} = {width}厘米",
    ]
    return ins, lines, width


_reg("wire_rect_width", wire_rect_width)


# 16. 几倍多几
def times_more_fruit(rng):
    a = rng.randint(5, 20)
    k = rng.randint(3, 5)
    extra = rng.randint(3, 9)
    f1, f2 = rng.sample(FRUITS, 2)
    n1 = rng.choice(NAMES)
    base = a * k
    total = base + extra
    t = rng.randrange(3)
    ins = [
        f"{n1}摘了{a}个{f1}，摘的{f2}是{f1}的{k}倍多{extra}个，{f2}有多少个？",
        f"果园里有{a}棵{f1}树，{f2}树的棵数是{f1}树的{k}倍多{extra}棵，{f2}树有多少棵？",
        f"妈妈买了{a}个{f1}，买的{f2}比{f1}的{k}倍还多{extra}个，{f2}有多少个？",
    ][t]
    lines = [
        f"{a} × {k} = {base}个",
        f"{base} + {extra} = {total}个",
    ]
    return ins, lines, total


_reg("times_more_fruit", times_more_fruit)


# 17. 几倍少几
def times_less_fruit(rng):
    a = rng.randint(5, 20)
    k = rng.randint(3, 5)
    less = rng.randint(2, a * k - 5)
    f1, f2 = rng.sample(FRUITS, 2)
    n1 = rng.choice(NAMES)
    base = a * k
    total = base - less
    t = rng.randrange(3)
    ins = [
        f"{n1}买了{a}个{f1}，买的{f2}是{f1}的{k}倍少{less}个，{f2}有多少个？",
        f"养殖场有{a}只一种家禽，另一种的只数是它的{k}倍少{less}只，另一种有多少只？",
        f"红花有{a}朵，黄花的朵数是红花的{k}倍少{less}朵，黄花有多少朵？",
    ][t]
    lines = [
        f"{a} × {k} = {base}个",
        f"{base} - {less} = {total}个",
    ]
    return ins, lines, total


_reg("times_less_fruit", times_less_fruit)


# 18. 几倍求总和
def twice_total_fruit(rng):
    a = rng.randint(5, 30)
    f1, f2 = rng.sample(FRUITS, 2)
    n1 = rng.choice(NAMES)
    twice = a * 2
    total = a + twice
    t = rng.randrange(3)
    ins = [
        f"{n1}买了{a}个{f1}，买的{f2}是{f1}的2倍，两种水果一共买了多少个？",
        f"食堂运来{a}筐{f1}，运来的{f2}是{f1}的2倍，一共运来多少筐？",
        f"水果店上午卖{a}斤{f1}，卖的{f2}是{f1}的2倍，这天两种共卖多少斤？",
    ][t]
    lines = [
        f"{a} × 2 = {twice}个",
        f"{a} + {twice} = {total}个",
    ]
    return ins, lines, total


_reg("twice_total_fruit", twice_total_fruit)


# 19. 三连比较: 多几再少几
def three_animal_chain(rng):
    a = rng.randint(10, 40)
    b = rng.randint(3, 15)
    c = rng.randint(2, a + b - 5)
    a1, a2, a3 = rng.sample(ANIMALS, 3)
    second = a + b
    third = second - c
    t = rng.randrange(3)
    ins = [
        f"农场有{a}只{a1}，{a2}比{a1}多{b}只，{a3}比{a2}少{c}只，{a3}有多少只？",
        f"动物园里{a1}有{a}只，{a2}比{a1}多{b}只，{a3}比{a2}少{c}只，{a3}有多少只？",
        f"养殖场养了{a}只{a1}，{a2}的只数比{a1}多{b}只，{a3}比{a2}少{c}只，{a3}有多少只？",
    ][t]
    lines = [
        f"{a2} = {a} + {b} = {second}只",
        f"{a3} = {second} - {c} = {third}只",
    ]
    return ins, lines, third


_reg("three_animal_chain", three_animal_chain)


# 20. 几倍后再少几
def fish_times_less(rng):
    a = rng.randint(10, 40)
    c = rng.randint(3, a * 2 - 5)
    twice = a * 2
    ans = twice - c
    t = rng.randrange(3)
    ins = [
        f"池塘里有{a}条鲤鱼，草鱼是鲤鱼的2倍，鲫鱼比草鱼少{c}条，鲫鱼有多少条？",
        f"鱼缸里有{a}条红金鱼，黑金鱼是红金鱼的2倍，花金鱼比黑金鱼少{c}条，花金鱼有多少条？",
        f"河里有{a}只鸭子，鹅的只数是鸭子的2倍，鸡比鹅少{c}只，鸡有多少只？",
    ][t]
    lines = [
        f"{a} × 2 = {twice}条",
        f"{twice} - {c} = {ans}条",
    ]
    return ins, lines, ans


_reg("fish_times_less", fish_times_less)


# 21. 成人票儿童票
def tickets_family(rng):
    adult = rng.randint(10, 60)
    child = rng.randint(5, adult - 1)
    na = rng.randint(2, 4)
    nc = rng.randint(2, 4)
    ta = adult * na
    tc = child * nc
    total = ta + tc
    place = rng.choice(PLACE)
    t = rng.randrange(3)
    ins = [
        f"{place}成人票每张{adult}元，儿童票每张{child}元，{na}个成人和{nc}个儿童买票共需多少元？",
        f"去{place}玩，成人票{adult}元一张，儿童票{child}元一张，{na}个大人带{nc}个孩子一共要多少元？",
        f"{place}的门票成人每张{adult}元、儿童每张{child}元，{na}个成人和{nc}个儿童买门票共花多少元？",
    ][t]
    lines = [
        f"{adult} × {na} = {ta}元",
        f"{child} × {nc} = {tc}元",
        f"{ta} + {tc} = {total}元",
    ]
    return ins, lines, total


_reg("tickets_family", tickets_family)


# 22. 办卡+按次付费
def membership_cost(rng):
    card = rng.randint(50, 200)
    per = rng.randint(10, 40)
    times = rng.randint(3, 12)
    pt = per * times
    total = card + pt
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"健身卡办卡{card}元，每次游泳再付{per}元，{n1}去了{times}次，一共花了多少元？",
        f"游泳馆办卡{card}元，每次入场{per}元，{n1}办卡后去了{times}次，共花多少元？",
        f"办一张借书卡{card}元，每借一本书再付{per}元，{n1}借了{times}本，一共花了多少元？",
    ][t]
    lines = [
        f"{per} × {times} = {pt}元",
        f"{card} + {pt} = {total}元",
    ]
    return ins, lines, total


_reg("membership_cost", membership_cost)


# 23. 降价后购买
def discount_buy(rng):
    price = rng.randint(8, 30)
    off = rng.randint(2, price - 3)
    qty = rng.randint(3, 8)
    now = price - off
    total = now * qty
    obj = rng.choice(STATIONERY + FRUITS)
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{obj}原来每个{price}元，现在每个便宜{off}元，{n1}买{qty}个要多少元？",
        f"{obj}原价每个{price}元，降价{off}元后，买{qty}个一共多少元？",
        f"每个{obj}售价{price}元，促销时每个少{off}元，买{qty}个要付多少元？",
    ][t]
    lines = [
        f"现价 = {price} - {off} = {now}元",
        f"{now} × {qty} = {total}元",
    ]
    return ins, lines, total


_reg("discount_buy", discount_buy)


# 24. 存钱后花费
def save_spend(rng):
    per = rng.randint(5, 30)
    weeks = rng.randint(3, 8)
    saved = per * weeks
    spend = rng.randint(10, saved - 5)
    left = saved - spend
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}每周存{per}元零花钱，存了{weeks}周，买书花了{spend}元，还剩多少元？",
        f"{n1}每周攒{per}元，攒了{weeks}周，买文具用掉{spend}元，还剩多少元？",
        f"{n1}把每周的{per}元压岁钱存起来，存了{weeks}周，捐给灾区{spend}元，还剩多少元？",
    ][t]
    lines = [
        f"{per} × {weeks} = {saved}元",
        f"{saved} - {spend} = {left}元",
    ]
    return ins, lines, left


_reg("save_spend", save_spend)


# 25. 两人共有, 求差
def money_split_diff(rng):
    b = rng.randint(10, 40)
    a = b * 2 + rng.randint(2, 20)
    yi = a - b
    diff = yi - b
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}和{n2}共有{a}元，{n1}有{b}元，{n2}比{n1}多多少元？",
        f"甲乙两人共有{a}元，甲有{b}元，乙比甲多多少元？",
        f"兄妹俩一共存了{a}元，哥哥存了{b}元，妹妹比哥哥多存多少元？",
    ][t]
    lines = [
        f"乙 = {a} - {b} = {yi}元",
        f"{yi} - {b} = {diff}元",
    ]
    return ins, lines, diff


_reg("money_split_diff", money_split_diff)


# 26. 给完一样多, 求原有
def money_equal_give(rng):
    a = rng.randint(10, 50)
    b = rng.randint(3, 15)
    give = b * 2
    yi = a + give
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}有{a}元，{n2}给{n1}{b}元后两人的钱一样多，{n2}原来有多少元？",
        f"甲有{a}元，乙给甲{b}元后两人钱数相等，乙原来有多少元？",
        f"小红有{a}元，小明给小红{b}元后两人的钱同样多，小明原来有多少元？",
    ][t]
    lines = [
        f"{b} × 2 = {give}元",
        f"{a} + {give} = {yi}元",
    ]
    return ins, lines, yi


_reg("money_equal_give", money_equal_give)


# 27. 已做+未做=总数
def work_total(rng):
    per = rng.randint(5, 30)
    days = rng.randint(3, 8)
    done = per * days
    left = rng.randint(10, 100)
    total = done + left
    t = rng.randrange(3)
    ins = [
        f"工人每天做{per}个零件，做了{days}天，还剩{left}个没做，这批零件共多少个？",
        f"小明每天折{per}只纸鹤，折了{days}天，还差{left}只就完成，他计划折多少只？",
        f"工厂每天生产{per}台机器，生产了{days}天，还剩{left}台没生产，这批订单共多少台？",
    ][t]
    lines = [
        f"{per} × {days} = {done}个",
        f"{done} + {left} = {total}个",
    ]
    return ins, lines, total


_reg("work_total", work_total)


# 28. 总数-已做=剩下
def work_remaining(rng):
    per = rng.randint(5, 20)
    days = rng.randint(3, 8)
    done = per * days
    total = done + rng.randint(10, 80)
    left = total - done
    t = rng.randrange(3)
    ins = [
        f"一批零件共{total}个，每天做{per}个，做了{days}天，还剩多少个？",
        f"一本书共{total}页，每天看{per}页，看了{days}天，还剩多少页没看？",
        f"工地要运{total}吨水泥，每天运{per}吨，运了{days}天，还剩多少吨？",
    ][t]
    lines = [
        f"{per} × {days} = {done}个",
        f"{total} - {done} = {left}个",
    ]
    return ins, lines, left


_reg("work_remaining", work_remaining)


# 29. 计划总量÷实际每天=实际天数
def read_plan_days(rng):
    plan = rng.randint(6, 15) * 2
    D = rng.randint(5, 9)
    total = plan * D
    cands = [d for d in range(2, D) if total % d == 0]
    d = rng.choice(cands)
    actual = total // d
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"一本书{n1}计划每天看{plan}页，{D}天看完。实际每天看{actual}页，实际多少天看完？",
        f"一批货物计划每天运{plan}吨，{D}天运完。实际每天运{actual}吨，实际几天运完？",
        f"做一批手工计划每天做{plan}个，{D}天完成。实际每天做{actual}个，实际几天完成？",
    ][t]
    lines = [
        f"{plan} × {D} = {total}页",
        f"{total} ÷ {actual} = {d}天",
    ]
    return ins, lines, d


_reg("read_plan_days", read_plan_days)


# 30. 来回游
def swim_laps(rng):
    a = rng.randint(20, 50)
    laps = rng.randint(3, 8)
    total = a * 2 * laps
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"游泳池长{a}米，{n1}游了{laps}个来回，一共游了多少米？",
        f"泳道长{a}米，小明游了{laps}个来回，他共游了多少米？",
        f"小河宽{a}米，小船划了{laps}个来回，一共划了多少米？",
    ][t]
    lines = [
        f"{a} × 2 = {a * 2}米",
        f"{a * 2} × {laps} = {total}米",
    ]
    return ins, lines, total


_reg("swim_laps", swim_laps)


# 31. 相向而行相遇路程
def two_cars_meet(rng):
    v1 = rng.randint(40, 80)
    v2 = rng.randint(40, 80)
    h = rng.randint(2, 4)
    s = v1 + v2
    total = s * h
    t = rng.randrange(3)
    ins = [
        f"甲乙两车同时从两地相向而行，甲车每小时行{v1}千米，乙车每小时行{v2}千米，{h}小时后相遇。两地相距多少千米？",
        f"两艘轮船同时从两港相对开出，甲船每小时行{v1}千米，乙船每小时行{v2}千米，{h}小时后相遇。两港相距多少千米？",
        f"小明和小红同时从两地相向走来，小明每分钟走{v1}米，小红每分钟走{v2}米，{h}分钟后相遇。两地相距多少米？",
    ][t]
    lines = [
        f"{v1} + {v2} = {s}千米",
        f"{s} × {h} = {total}千米",
    ]
    return ins, lines, total


_reg("two_cars_meet", two_cars_meet)


# 32. 快车慢车各行几小时共行
def speed_compare_total(rng):
    v = rng.randint(20, 40)
    v2 = v * 2
    h = rng.randint(2, 4)
    total = (v + v2) * h
    t = rng.randrange(3)
    ins = [
        f"甲车每小时行{v}千米，乙车速度是甲车的2倍，两车各行了{h}小时，一共行了多少千米？",
        f"小明每小时走{v}千米，爸爸骑车的速度是小明的2倍，两人各走了{h}小时，一共行多少千米？",
        f"货车每小时行{v}千米，客车速度是货车的2倍，两车各行驶{h}小时，共行驶多少千米？",
    ][t]
    lines = [
        f"{v} × 2 = {v2}千米",
        f"({v} + {v2}) × {h} = {total}千米",
    ]
    return ins, lines, total


_reg("speed_compare_total", speed_compare_total)


# 33. 分组后再来几人又能组一组
def group_need_more(rng):
    k = rng.randint(4, 9)
    full = rng.randint(3, 7)
    left = rng.randint(1, k - 1)
    n = k * full + left
    need = k - left
    t = rng.randrange(3)
    ins = [
        f"{n}个同学跳绳，每{k}人一组，分完组后，至少再来几人又能组成一组？",
        f"二（1）班有{n}人，每{k}人一组做游戏，分组后至少再来几人又能分一组？",
        f"有{n}名同学，每{k}人一组，分完后至少再来几名同学又能组成一组？",
    ][t]
    lines = [
        f"{k} × {full} = {k * full}人",
        f"{n} - {k * full} = {left}人",
        f"{k} - {left} = {need}人",
    ]
    return ins, lines, need


_reg("group_need_more", group_need_more)


# 34. 尽量多买后剩多少
def money_max_buy_change(rng):
    for _ in range(50):
        money = rng.randint(30, 100)
        price = rng.randint(4, 12)
        left = money % price
        if left > 0:
            break
    qty = money // price
    spent = qty * price
    obj = rng.choice(STATIONERY)
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}带了{money}元去买{obj}，每个{price}元。他尽量多买，买完后还剩多少元？",
        f"{n1}有{money}元，{obj}每个{price}元，最多买几个后还剩多少元？",
        f"文具店里{obj}每个{price}元，{n1}带{money}元尽量多买，还剩多少元？",
    ][t]
    lines = [
        f"{price} × {qty} = {spent}元",
        f"{money} - {spent} = {left}元",
    ]
    return ins, lines, left


_reg("money_max_buy_change", money_max_buy_change)


# 35. 年龄和求未来
def age_sum_future(rng):
    son = rng.randint(6, 15)
    dad = son + rng.randint(24, 38)
    s = son + dad
    yrs = rng.randint(3, 10)
    dad_future = dad + yrs
    t = rng.randrange(3)
    ins = [
        f"今年父子俩的年龄和是{s}岁，儿子今年{son}岁。{yrs}年后父亲多少岁？",
        f"今年父女俩的年龄和是{s}岁，女儿今年{son}岁。{yrs}年后父亲多少岁？",
        f"今年母子俩的年龄和是{s}岁，儿子今年{son}岁。{yrs}年后母亲多少岁？",
    ][t]
    lines = [
        f"父亲 = {s} - {son} = {dad}岁",
        f"{dad} + {yrs} = {dad_future}岁",
    ]
    return ins, lines, dad_future


_reg("age_sum_future", age_sum_future)


# 36. 年龄差求过去
def mom_age_past(rng):
    girl = rng.randint(7, 15)
    diff = rng.randint(24, 35)
    mom = girl + diff
    yrs = rng.randint(3, 10)
    mom_past = mom - yrs
    t = rng.randrange(3)
    ins = [
        f"妈妈比小红大{diff}岁，小红今年{girl}岁。{yrs}年前妈妈多少岁？",
        f"爸爸比小明大{diff}岁，小明今年{girl}岁。{yrs}年前爸爸多少岁？",
        f"老师比小丽大{diff}岁，小丽今年{girl}岁。{yrs}年前老师多少岁？",
    ][t]
    lines = [
        f"{girl} + {diff} = {mom}岁",
        f"{mom} - {yrs} = {mom_past}岁",
    ]
    return ins, lines, mom_past


_reg("mom_age_past", mom_age_past)


# 37. 全家年龄和
def family_age_sum(rng):
    dad = rng.randint(32, 45)
    diff = rng.randint(1, 6)
    mom = dad - diff
    kid = rng.randint(6, 15)
    parents = dad + mom
    total = parents + kid
    t = rng.randrange(3)
    ins = [
        f"爸爸今年{dad}岁，妈妈比爸爸小{diff}岁，孩子今年{kid}岁。全家年龄和是多少岁？",
        f"爷爷今年{dad}岁，奶奶比爷爷小{diff}岁，孙子今年{kid}岁。三人年龄和是多少岁？",
        f"爸爸今年{dad}岁，妈妈比爸爸小{diff}岁，小明今年{kid}岁。一家人年龄和是多少岁？",
    ][t]
    lines = [
        f"妈妈 = {dad} - {diff} = {mom}岁",
        f"父母和 = {dad} + {mom} = {parents}岁",
        f"{parents} + {kid} = {total}岁",
    ]
    return ins, lines, total


_reg("family_age_sum", family_age_sum)


# 38. 经过时间(同小时段内)
def elapsed_minutes(rng):
    for _ in range(50):
        h1 = rng.randint(1, 8)
        m1 = rng.randint(5, 55)
        dur = rng.randint(30, 240)
        end = h1 * 60 + m1 + dur
        h2, m2 = end // 60, end % 60
        if m2 >= m1 and h2 <= 12:
            break
    base = (h2 - h1) * 60
    t = rng.randrange(3)
    ins = [
        f"电影从{h1}时{m1}分开始，{h2}时{m2}分结束，这场电影放映了多少分钟？",
        f"一节课从{h1}时{m1}分开始，{h2}时{m2}分下课，这节课有多少分钟？",
        f"会议从{h1}时{m1}分开到{h2}时{m2}分，一共开了多少分钟？",
    ][t]
    lines = [
        f"({h2} - {h1}) × 60 = {base}分",
        f"{base} + {m2} - {m1} = {dur}分",
    ]
    return ins, lines, dur


_reg("elapsed_minutes", elapsed_minutes)


# 39. 经过时间(跨半点)
def elapsed_hours(rng):
    h1 = rng.randint(7, 10)
    m1 = rng.randint(20, 55)
    h2 = h1 + rng.randint(2, 4)
    m2 = rng.randint(5, m1 - 1)
    base = (h2 - h1) * 60
    dur = base - m1 + m2
    t = rng.randrange(3)
    ins = [
        f"一列火车{h1}时{m1}分从甲站开出，{h2}时{m2}分到达乙站，路上用了多少分钟？",
        f"小明{h1}时{m1}分从家出发，{h2}时{m2}分到学校，路上走了多少分钟？",
        f"汽车{h1}时{m1}分从县城出发，{h2}时{m2}分到达省城，行了多少分钟？",
    ][t]
    lines = [
        f"({h2} - {h1}) × 60 = {base}分",
        f"{base} - {m1} + {m2} = {dur}分",
    ]
    return ins, lines, dur


_reg("elapsed_hours", elapsed_hours)


# 40. 楼层台阶
def stairs_floors(rng):
    floors = rng.randint(3, 8)
    per = rng.randint(10, 24)
    gaps = floors - 1
    total = gaps * per
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}从1楼走到{floors}楼，每层楼有{per}级台阶，一共走了多少级台阶？",
        f"小明家住{floors}楼，每层楼有{per}级台阶，他从1楼走到家要走多少级台阶？",
        f"小红从1楼上到{floors}楼，每层有{per}级台阶，她共上了多少级台阶？",
    ][t]
    lines = [
        f"{floors} - 1 = {gaps}层",
        f"{gaps} × {per} = {total}级",
    ]
    return ins, lines, total


_reg("stairs_floors", stairs_floors)


# 41. 直线栽树(两端都栽)
def tree_line_plant(rng):
    gap = rng.randint(3, 8)
    seg = rng.randint(3, 8)
    length = gap * seg
    trees = seg + 1
    t = rng.randrange(3)
    ins = [
        f"在一条长{length}米的小路一边栽树，每隔{gap}米栽一棵（两端都栽），一共要栽多少棵？",
        f"一条马路长{length}米，在路的一边从头到尾每隔{gap}米种一棵树，共种多少棵？",
        f"公园小路长{length}米，沿一侧每隔{gap}米放一把长椅（两端都放），共放多少把？",
    ][t]
    lines = [
        f"{length} ÷ {gap} = {seg}段",
        f"{seg} + 1 = {trees}棵",
    ]
    return ins, lines, trees


_reg("tree_line_plant", tree_line_plant)


# 42. 走廊两边挂灯笼
def lanterns_both_sides(rng):
    gap = rng.randint(3, 8)
    seg = rng.randint(3, 8)
    length = gap * seg
    one_side = seg + 1
    total = one_side * 2
    t = rng.randrange(3)
    ins = [
        f"走廊长{length}米，每隔{gap}米挂一盏灯笼（两端都挂），两边都挂，一共要挂多少盏？",
        f"一条路长{length}米，在路的两旁每隔{gap}米插一面彩旗（两端都插），共插多少面？",
        f"小桥长{length}米，桥的两边每隔{gap}米有一根栏杆柱（两端都有），共多少根？",
    ][t]
    lines = [
        f"{length} ÷ {gap} = {seg}段",
        f"{seg} + 1 = {one_side}盏",
        f"{one_side} × 2 = {total}盏",
    ]
    return ins, lines, total


_reg("lanterns_both_sides", lanterns_both_sides)


# 43. 借出还回
def borrow_return(rng):
    total = rng.randint(80, 200)
    out = rng.randint(20, total - 30)
    back = rng.randint(5, out - 5)
    after_out = total - out
    now = after_out + back
    place = rng.choice(PLACE)
    t = rng.randrange(3)
    ins = [
        f"图书馆有{total}本书，借走{out}本，还回来{back}本，现在有多少本？",
        f"图书角有{total}本书，同学们借走{out}本，又还回{back}本，现在有多少本？",
        f"{place}原来有{total}本书，借出{out}本后还回{back}本，现在有多少本？",
    ][t]
    lines = [
        f"{total} - {out} = {after_out}本",
        f"{after_out} + {back} = {now}本",
    ]
    return ins, lines, now


_reg("borrow_return", borrow_return)


# 44. 书架放书
def shelves_books(rng):
    ns = rng.randint(3, 5)
    nl = rng.randint(3, 5)
    per = rng.randint(20, 60)
    layers = ns * nl
    total = layers * per
    t = rng.randrange(3)
    ins = [
        f"图书馆有{ns}个书架，每个书架有{nl}层，每层放{per}本书，一共能放多少本？",
        f"教室图书角有{ns}个书架，每个{nl}层，每层摆{per}本书，共摆多少本？",
        f"阅览室有{ns}个新书架，每个书架{nl}层，每层放{per}本，一共可放多少本？",
    ][t]
    lines = [
        f"{ns} × {nl} = {layers}层",
        f"{layers} × {per} = {total}本",
    ]
    return ins, lines, total


_reg("shelves_books", shelves_books)


# 45. 和差问题求大数
def sum_diff_big(rng):
    small = rng.randint(10, 40)
    diff = rng.randint(3, 15)
    big = small + diff
    s = big + small
    ans = (s + diff) // 2
    t = rng.randrange(3)
    ins = [
        f"甲乙两数的和是{s}，差是{diff}，甲数比乙数大。甲数是多少？",
        f"两个数的和是{s}，它们的差是{diff}，其中较大的数是多少？",
        f"兄妹俩共有{s}元，哥哥比妹妹多{diff}元，哥哥有多少元？",
    ][t]
    who = ["甲数", "较大的数", "哥哥"][t]
    unit = "元" if t == 2 else ""
    lines = [
        f"和加差 = {s} + {diff} = {s + diff}{unit}",
        f"{who} = {s + diff} ÷ 2 = {ans}{unit}",
    ]
    return ins, lines, ans


_reg("sum_diff_big", sum_diff_big)


# 46. 三天销量: 第三天=前两天总和
def sale_three_day(rng):
    d1 = rng.randint(20, 60)
    b = rng.randint(5, 20)
    d2 = d1 + b
    d3 = d1 + d2
    f1 = rng.choice(FRUITS)
    t = rng.randrange(3)
    ins = [
        f"水果店三天卖完一批{f1}。第一天卖{d1}筐，第二天比第一天多卖{b}筐，第三天卖的等于前两天的总和。第三天卖了多少筐？",
        f"超市三天运进一批{f1}：第一天{d1}筐，第二天比第一天多{b}筐，第三天运的是前两天的总和。第三天运多少筐？",
        f"果园三天摘完一批{f1}：第一天摘{d1}筐，第二天比第一天多摘{b}筐，第三天摘的正好是前两天的总和。第三天摘多少筐？",
    ][t]
    lines = [
        f"第二天 = {d1} + {b} = {d2}筐",
        f"第三天 = {d1} + {d2} = {d3}筐",
    ]
    return ins, lines, d3


_reg("sale_three_day", sale_three_day)


# 47. 平均分目标: 下一次考多少
def next_score_target(rng):
    k = rng.randint(3, 5)
    avg = rng.randint(80, 90)
    bump = rng.randint(1, (100 - avg) // (k + 1))
    target = avg + bump
    total = avg * k
    new_total = target * (k + 1)
    need = new_total - total
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}前{k}次测验的平均分是{avg}分，想让{k + 1}次测验的平均分达到{target}分，下一次要考多少分？",
        f"小红前{k}次考试平均{avg}分，她希望{k + 1}次考试的平均分是{target}分，下次要考多少分？",
        f"小明前{k}次测验平均{avg}分，若{k + 1}次测验的平均分要达到{target}分，下一次需考多少分？",
    ][t]
    lines = [
        f"{avg} × {k} = {total}分",
        f"{target} × {k + 1} = {new_total}分",
        f"{new_total} - {total} = {need}分",
    ]
    return ins, lines, need


_reg("next_score_target", next_score_target)


# 48. 前两次平均+第三次=总分
def two_avg_total(rng):
    avg2 = rng.randint(70, 95)
    third = rng.randint(60, 100)
    first_two = avg2 * 2
    total = first_two + third
    n1 = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n1}前两次测验的平均分是{avg2}分，第三次考了{third}分，三次测验一共多少分？",
        f"小红前两次数学测验平均{avg2}分，第三次得{third}分，三次共得多少分？",
        f"小明前两次跳绳平均每次{avg2}下，第三次跳了{third}下，三次一共跳多少下？",
    ][t]
    lines = [
        f"{avg2} × 2 = {first_two}分",
        f"{first_two} + {third} = {total}分",
    ]
    return ins, lines, total


_reg("two_avg_total", two_avg_total)


# 49. 分组求每组女生
def class_groups_girls(rng):
    girls_per = rng.randint(2, 6)
    boys_per = rng.randint(3, 8)
    per = girls_per + boys_per
    ng = rng.randint(3, 6)
    total = per * ng
    t = rng.randrange(3)
    ins = [
        f"全班{total}人，正好分成{ng}组，每组男生{boys_per}人，每组女生多少人？",
        f"二（2）班{total}人，分成{ng}个小组，每组有男生{boys_per}人，每组有女生多少人？",
        f"同学们{total}人去植树，分成{ng}组，每组男生{boys_per}人，每组女生几人？",
    ][t]
    lines = [
        f"{total} ÷ {ng} = {per}人",
        f"{per} - {boys_per} = {girls_per}人",
    ]
    return ins, lines, girls_per


_reg("class_groups_girls", class_groups_girls)


# 50. 买纸发纸剩多少
def paper_distribute(rng):
    for _ in range(50):
        per_pack = rng.randint(20, 50)
        packs = rng.randint(3, 6)
        total_paper = per_pack * packs
        n_people = rng.randint(3, 6)
        per_person = rng.randint(5, 15)
        given = n_people * per_person
        if given < total_paper:
            break
    left = total_paper - given
    t = rng.randrange(3)
    ins = [
        f"学校买来{packs}包打印纸，每包{per_pack}张，发给{n_people}个办公室，每个办公室{per_person}张，还剩多少张？",
        f"商店运来{packs}包本子，每包{per_pack}本，卖给{n_people}个班，每班{per_person}本，还剩多少本？",
        f"仓库有{packs}包彩纸，每包{per_pack}张，发给{n_people}个年级，每个年级{per_person}张，还剩多少张？",
    ][t]
    lines = [
        f"{per_pack} × {packs} = {total_paper}张",
        f"{n_people} × {per_person} = {given}张",
        f"{total_paper} - {given} = {left}张",
    ]
    return ins, lines, left


_reg("paper_distribute", paper_distribute)


# 51. 毫升与升
def juice_liters(rng):
    bottles = rng.choice([4, 5, 8])
    k = rng.randint(1, 2)
    ml = 1000 * k // bottles
    total_ml = ml * bottles
    liters = total_ml // 1000
    obj = rng.choice(["果汁", "牛奶", "酱油", "矿泉水"])
    t = rng.randrange(3)
    ins = [
        f"一瓶{obj}{ml}毫升，{bottles}瓶{obj}共多少毫升？合多少升？",
        f"每瓶{obj}{ml}毫升，买{bottles}瓶一共多少毫升？是多少升？",
        f"一桶{obj}{ml}毫升，{bottles}桶共多少毫升？合多少升？",
    ][t]
    lines = [
        f"{ml} × {bottles} = {total_ml}毫升",
        f"{total_ml} ÷ 1000 = {liters}升",
    ]
    return ins, lines, liters


_reg("juice_liters", juice_liters)


# 52. 梨树比苹果树多, 桃树是梨树2倍
def orchard_trees(rng):
    apple = rng.randint(20, 60)
    b = rng.randint(5, 20)
    pear = apple + b
    peach = pear * 2
    t = rng.randrange(3)
    ins = [
        f"果园有苹果树{apple}棵，梨树比苹果树多{b}棵，桃树的棵数是梨树的2倍，桃树有多少棵？",
        f"山上有松树{apple}棵，柏树比松树多{b}棵，杨树的棵数是柏树的2倍，杨树有多少棵？",
        f"养殖场有小鸡{apple}只，小鸭比小鸡多{b}只，小鹅的只数是小鸭的2倍，小鹅有多少只？",
    ][t]
    lines = [
        f"梨树 = {apple} + {b} = {pear}棵",
        f"{pear} × 2 = {peach}棵",
    ]
    return ins, lines, peach


_reg("orchard_trees", orchard_trees)


# 53. 第二筐少几, 两筐共重
def fruit_weight_compare(rng):
    a = rng.randint(20, 60)
    b = rng.randint(3, a - 5)
    second = a - b
    total = a + second
    f1 = rng.choice(FRUITS)
    t = rng.randrange(3)
    ins = [
        f"第一筐{f1}重{a}千克，第二筐比第一筐少{b}千克，两筐{f1}一共重多少千克？",
        f"一袋米重{a}千克，另一袋比这袋少{b}千克，两袋米一共重多少千克？",
        f"第一筐{f1}重{a}千克，第二筐比第一筐轻{b}千克，两筐一共重多少千克？",
    ][t]
    lines = [
        f"第二筐 = {a} - {b} = {second}千克",
        f"两筐和 = {a} + {second} = {total}千克",
    ]
    return ins, lines, total


_reg("fruit_weight_compare", fruit_weight_compare)


if __name__ == "__main__":
    rng = random.Random(2)
    from run_math_short import verify
    ok = 0
    for _lvl, name, fn in PROGRAMS:
        for _ in range(40):
            ins, lines, ans = fn(rng)
            out, good = verify(ins, lines, ans)
            assert good, f"{name} FAILED: {ins!r} {lines}"
            ok += 1
    print(f"L2 ext2 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
