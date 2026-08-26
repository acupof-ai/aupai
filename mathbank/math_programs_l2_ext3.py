#!/usr/bin/env python3
"""L2 ext3: 60 verified 3-4 step word problems (new quantitative structures).

Families: half-then-more chains, missing-value averages, area->perimeter and
area->width geometry, unit-price-then-buy, fraction parts (read/road/rope/
class), sum-multiple / difference-multiple, chicken-rabbit and variants,
cooperation and round-up division, utility/taxi bills, interval rates (clock
strikes, closed-circle planting, square formations), arithmetic sequences,
pour-to-equal, reverse operations, digits, pair-sum weights, profit,
train-bridge, downstream, fraction-of-fraction, buy-3-get-1, and more.
All lines are independently verified by run_math_short.verify.
"""
import random
from fractions import Fraction
from math import gcd
from mathcommon import (ANIMALS, FOOD, FRUITS, GOODS, NAMES, PLACE,
                        STATIONERY, num)

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L2", name, fn))


# 1. 一半再多几: a/2 + c
def half_then_more(rng):
    a = rng.randint(10, 40) * 2
    c = rng.randint(3, 15)
    h = a // 2
    ans = h + c
    n1, n2, n3 = rng.sample(NAMES, 3)
    obj = rng.choice(["纸鹤", "星星", "小船", "花", "风车"])
    t = rng.randrange(4)
    ins = [
        f"{n1}折了{a}只{obj}，{n2}折的是{n1}的一半，{n3}比{n2}多折{c}只，{n3}折了多少只？",
        f"{n1}做了{a}个{obj}，{n2}做的个数是{n1}的一半，{n3}比{n2}多做{c}个，{n3}做了多少个？",
        f"{n1}剪了{a}朵{obj}，{n2}剪的是{n1}的一半，{n3}比{n2}多剪{c}朵，{n3}剪了多少朵？",
        f"{n1}画了{a}个{obj}，{n2}画的是{n1}的一半，{n3}比{n2}多画{c}个，{n3}画了多少个？",
    ][t]
    lines = [
        f"{a} ÷ 2 = {h}只",
        f"{h} + {c} = {ans}只",
    ]
    return ins, lines, ans


_reg("half_then_more", half_then_more)


# 2. 平均数求第三个数
def avg_missing_third(rng):
    for _ in range(50):
        avg = rng.randint(78, 92)
        b = rng.randint(60, 95)
        c = rng.randint(60, 95)
        total = avg * 3
        ans = total - b - c
        if 60 <= ans <= 100:
            break
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"{n1}三次测验的平均分是{avg}分，前两次分别得{b}分和{c}分，第三次得多少分？",
        f"小红三次考试平均{avg}分，前两次考了{b}分和{c}分，第三次考了多少分？",
        f"水果店三天平均每天卖{avg}筐苹果，前两天卖了{b}筐和{c}筐，第三天卖多少筐？",
        f"三个数的平均数是{avg}，其中两个数是{b}和{c}，第三个数是多少？",
    ][t]
    lines = [
        f"总分 = {avg} × 3 = {total}",
        f"第三次 = {total} - {b} - {c} = {ans}",
    ]
    return ins, lines, ans


_reg("avg_missing_third", avg_missing_third)


# 3. 正方形面积 -> 周长
def square_area_to_perimeter(rng):
    s = rng.randint(4, 12)
    area = s * s
    per = s * 4
    obj = rng.choice(["花坛", "池塘", "广场", "操场", "菜地", "花圃"])
    t = rng.randrange(4)
    ins = [
        f"一个正方形{obj}的面积是{area}平方米，它的周长是多少米？",
        f"正方形{obj}的面积为{area}平方米，周长是多少米？",
        f"一块正方形{obj}占地{area}平方米，绕它走一圈是多少米？",
        f"正方形{obj}面积是{area}平方米，四周边长一共多少米？",
    ][t]
    lines = [
        f"{s} × {s} = {area}平方米",
        f"{s} × 4 = {per}米",
    ]
    return ins, lines, per


_reg("square_area_to_perimeter", square_area_to_perimeter)


# 4. 长方形面积 -> 宽 -> 周长
def rect_area_find_width(rng):
    L = rng.randint(6, 14)
    w = rng.randint(3, 9)
    A = L * w
    per = (L + w) * 2
    obj = rng.choice(["菜地", "草坪", "场地", "花圃", "操场"])
    t = rng.randrange(4)
    ins = [
        f"一块长方形{obj}的面积是{A}平方米，长是{L}米，它的周长是多少米？",
        f"长方形{obj}面积为{A}平方米，长{L}米，周长是多少米？",
        f"一个长方形{obj}占地{A}平方米，量得长是{L}米，它的周长是多少米？",
        f"长方形{obj}的面积是{A}平方米，长是{L}米，四周围一圈共多少米？",
    ][t]
    lines = [
        f"{A} ÷ {L} = {w}米",
        f"({L} + {w}) × 2 = {per}米",
    ]
    return ins, lines, per


_reg("rect_area_find_width", rect_area_find_width)


# 5. 单价 -> 再买
def unit_price_then_buy(rng):
    a = rng.randint(2, 4)
    p = rng.randint(5, 20)
    b = a * p
    c = rng.randint(2, 6)
    ans = p * c
    f1 = rng.choice(FRUITS)
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"妈妈买{a}千克{f1}花了{b}元，照这样计算，买{c}千克要多少元？",
        f"{n1}买{a}千克{f1}用了{b}元，每千克多少元？买{c}千克共要多少元？",
        f"水果店{a}千克{f1}售价{b}元，买{c}千克应付多少元？",
        f"{a}千克{f1}卖{b}元，照这样的价格，买{c}千克需要多少元？",
    ][t]
    lines = [
        f"{b} ÷ {a} = {p}元",
        f"{p} × {c} = {ans}元",
    ]
    return ins, lines, ans


_reg("unit_price_then_buy", unit_price_then_buy)


# 6. 看了 1/n，求剩下
def fraction_read_remaining(rng):
    n = rng.randint(3, 5)
    a = rng.randint(4, 20) * n
    part = a // n
    ans = a - part
    obj = rng.choice(["故事书", "童话书", "漫画书", "科技书"])
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"一本{obj}共{a}页，{n1}看了全书的 1/{n}，还剩多少页没看？",
        f"{n1}看一本{a}页的{obj}，已经看了 1/{n}，还剩多少页？",
        f"一本{obj}有{a}页，第一天看了全书的 1/{n}，还剩多少页？",
        f"一本{a}页的{obj}，小红看了 1/{n}，还剩多少页没有看？",
    ][t]
    lines = [
        f"{a} ÷ {n} = {part}页",
        f"{a} - {part} = {ans}页",
    ]
    return ins, lines, ans


_reg("fraction_read_remaining", fraction_read_remaining)


# 7. 两天各修 1/4 和 1/3，共修
def fraction_two_days_sum(rng):
    a = rng.randint(2, 12) * 12
    x = a // 4
    y = a // 3
    ans = x + y
    obj = rng.choice(["公路", "水渠", "跑道", "围墙"])
    t = rng.randrange(4)
    ins = [
        f"修路队修一条长{a}米的{obj}，第一天修了 1/4，第二天修了 1/3，两天共修多少米？",
        f"一条{obj}长{a}米，第一天修了全长的 1/4，第二天修了全长的 1/3，两天一共修多少米？",
        f"工程队修{a}米长的{obj}，第一天修 1/4，第二天修 1/3，两天共修多少米？",
        f"一条{a}米的{obj}，第一天修了 1/4，第二天修了 1/3，两天合计修了多少米？",
    ][t]
    lines = [
        f"{a} ÷ 4 = {x}米",
        f"{a} ÷ 3 = {y}米",
        f"{x} + {y} = {ans}米",
    ]
    return ins, lines, ans


_reg("fraction_two_days_sum", fraction_two_days_sum)


# 8. 一半多几
def fraction_half_more(rng):
    a = rng.randint(10, 40) * 2
    b = rng.randint(3, a // 2 - 1)
    h = a // 2
    ans = h + b
    obj = rng.choice(FRUITS)
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"一筐{obj}有{a}个，第一天卖出一半多{b}个，第一天卖出多少个？",
        f"水果店运来{a}个{obj}，第一天卖出一半还多{b}个，第一天卖了多少个？",
        f"{n1}家有{a}个{obj}，第一天吃掉一半多{b}个，第一天吃了多少个？",
        f"一堆{obj}共{a}个，第一天运走一半多{b}个，第一天运走多少个？",
    ][t]
    lines = [
        f"{a} ÷ 2 = {h}个",
        f"{h} + {b} = {ans}个",
    ]
    return ins, lines, ans


_reg("fraction_half_more", fraction_half_more)


# 9. 和倍问题求大数
def sum_multiple_big(rng):
    k = rng.randint(2, 4)
    yi = rng.randint(10, 40)
    s = yi * (k + 1)
    jia = yi * k
    t = rng.randrange(4)
    ins = [
        f"果园里桃树和梨树共{s}棵，桃树的棵数是梨树的{k}倍，桃树有多少棵？",
        f"小明和小红共有{s}元，小明的钱是小红的{k}倍，小明有多少元？",
        f"书架上故事书和科技书共{s}本，故事书是科技书的{k}倍，故事书有多少本？",
        f"养鸡场公鸡和母鸡共{s}只，母鸡只数是公鸡的{k}倍，母鸡有多少只？",
    ][t]
    lines = [
        f"{s} ÷ {k + 1} = {yi}棵",
        f"{yi} × {k} = {jia}棵",
    ]
    return ins, lines, jia


_reg("sum_multiple_big", sum_multiple_big)


# 10. 差倍问题求大数
def diff_multiple_big(rng):
    k = rng.randint(3, 5)
    yi = rng.randint(10, 30)
    d = yi * (k - 1)
    jia = yi * k
    t = rng.randrange(4)
    ins = [
        f"甲比乙多{d}元，甲的钱数正好是乙的{k}倍，甲有多少元？",
        f"苹果比梨多{d}筐，苹果的筐数是梨的{k}倍，苹果有多少筐？",
        f"哥哥比弟弟多{d}张邮票，哥哥的邮票是弟弟的{k}倍，哥哥有多少张？",
        f"红花比黄花多{d}朵，红花朵数是黄花的{k}倍，红花有多少朵？",
    ][t]
    lines = [
        f"{d} ÷ {k - 1} = {yi}元",
        f"{yi} × {k} = {jia}元",
    ]
    return ins, lines, jia


_reg("diff_multiple_big", diff_multiple_big)


# 11. 鸡兔同笼求兔
def chicken_rabbit(rng):
    h = rng.randint(8, 20)
    rabbits = rng.randint(2, h - 2)
    l = 2 * h + 2 * rabbits
    t = rng.randrange(4)
    ins = [
        f"鸡兔同笼，共有{h}个头、{l}条腿，兔有多少只？",
        f"笼子里鸡和兔共{h}只，腿一共有{l}条，兔有多少只？",
        f"鸡和兔关在一个笼子里，共{h}个头、{l}条腿，兔子有多少只？",
        f"鸡兔同笼，数头共{h}个，数腿共{l}条，兔有多少只？",
    ][t]
    lines = [
        f"{h} × 2 = {2 * h}条",
        f"{l} - {2 * h} = {2 * rabbits}条",
        f"{2 * rabbits} ÷ 2 = {rabbits}只",
    ]
    return ins, lines, rabbits


_reg("chicken_rabbit", chicken_rabbit)


# 12. 合作几天完成
def coop_work_days(rng):
    a = rng.randint(10, 30)
    b = rng.randint(10, 30)
    days = rng.randint(3, 8)
    N = (a + b) * days
    t = rng.randrange(4)
    ins = [
        f"一批零件共{N}个，甲每天做{a}个，乙每天做{b}个，两人合作几天完成？",
        f"要加工{N}个零件，师傅每天做{a}个，徒弟每天做{b}个，两人合做几天完成？",
        f"一条路长{N}米，甲队每天修{a}米，乙队每天修{b}米，两队合修几天修完？",
        f"仓库有{N}吨货物，甲车每天运{a}吨，乙车每天运{b}吨，两车合运几天运完？",
    ][t]
    lines = [
        f"{a} + {b} = {a + b}个",
        f"{N} ÷ {a + b} = {days}天",
    ]
    return ins, lines, days


_reg("coop_work_days", coop_work_days)


# 13. 电表读数算电费
def meter_reading_cost(rng):
    a = rng.randint(100, 500)
    b = a + rng.randint(50, 300)
    c = rng.randint(3, 9)
    used = b - a
    ans = used * c
    t = rng.randrange(4)
    ins = [
        f"小明家上月电表读数是{a}度，本月读数是{b}度，每度电{c}角，本月电费多少角？",
        f"电表上月底读数{a}度，这月底读数{b}度，每度电{c}角，这个月应交电费多少角？",
        f"小红家上月抄表{a}度，本月抄表{b}度，电价每度{c}角，本月电费多少角？",
        f"学校电表上月读数{a}度，本月读数{b}度，每度电{c}角，本月用电多少角？",
    ][t]
    lines = [
        f"{b} - {a} = {used}度",
        f"{used} × {c} = {ans}角",
    ]
    return ins, lines, ans


_reg("meter_reading_cost", meter_reading_cost)


# 14. 出租车计费
def taxi_fare(rng):
    a = rng.randint(5, 14)
    c = rng.randint(4, 12)
    b = rng.randint(2, 5)
    extra = c - 3
    add = extra * b
    ans = a + add
    t = rng.randrange(4)
    ins = [
        f"出租车起步价{a}元（3千米以内），超出部分每千米{b}元，行{c}千米要多少元？",
        f"某市出租车起步价{a}元，3千米后每千米收{b}元，小明坐车行了{c}千米，应付多少元？",
        f"出租车3千米内{a}元，超过3千米每千米{b}元，小红乘车{c}千米，需付多少元？",
        f"打车起步价{a}元（含3千米），以后每千米{b}元，行驶{c}千米共多少元？",
    ][t]
    lines = [
        f"{c} - 3 = {extra}千米",
        f"{extra} × {b} = {add}元",
        f"{a} + {add} = {ans}元",
    ]
    return ins, lines, ans


_reg("taxi_fare", taxi_fare)


# 15. 租船进一法
def boat_rental_roundup(rng):
    b = rng.randint(4, 9)
    q = rng.randint(3, 7)
    r = rng.randint(1, b - 1)
    a = b * q + r
    ans = q + 1
    t = rng.randrange(4)
    ins = [
        f"{a}个同学去划船，每条船坐{b}人，至少要租几条船？",
        f"二（1）班{a}人去公园划船，每条船限坐{b}人，至少需要几条船？",
        f"{a}名师生过河，每条船最多坐{b}人，至少要租几条船？",
        f"同学们去划船，共{a}人，每条船坐{b}人，最少要租几条船？",
    ][t]
    lines = [
        f"{b} × {q} = {b * q}人",
        f"{a} - {b * q} = {r}人",
        f"{q} + 1 = {ans}条",
    ]
    return ins, lines, ans


_reg("boat_rental_roundup", boat_rental_roundup)


# 16. 敲钟间隔
def clock_strike_interval(rng):
    a = rng.randint(3, 5)
    iv = rng.randint(2, 8)
    t = (a - 1) * iv
    b = rng.randint(6, 9)
    ans = iv * (b - 1)
    p = rng.randrange(4)
    ins = [
        f"时钟{a}点敲{a}下，用{t}秒敲完，照这样计算，{b}点敲{b}下用多少秒？",
        f"一座钟敲{a}下用了{t}秒，敲{b}下要用多少秒？",
        f"时钟报时，敲{a}下用{t}秒，那么敲{b}下需要多少秒？",
        f"广场的钟{a}点敲{a}下，{t}秒敲完，{b}点敲{b}下，多少秒敲完？",
    ][p]
    lines = [
        f"{t} ÷ ({a} - 1) = {iv}秒",
        f"{iv} × ({b} - 1) = {ans}秒",
    ]
    return ins, lines, ans


_reg("clock_strike_interval", clock_strike_interval)


# 17. 封闭图形栽树
def plant_circle_closed(rng):
    b = rng.randint(3, 8)
    k = rng.randint(3, 8)
    a = b * k
    ans = a // b
    obj = rng.choice(["树", "花", "灯柱", "长椅", "彩旗"])
    t = rng.randrange(4)
    ins = [
        f"圆形花坛周长{a}米，每隔{b}米放一个{obj}，一共要放多少个？",
        f"一个圆形池塘周长{a}米，沿池边每隔{b}米栽一棵{obj}，共栽多少棵？",
        f"圆形操场一周长{a}米，每隔{b}米插一面{obj}，一共要插多少面？",
        f"在周长{a}米的圆形广场边上，每隔{b}米放一个{obj}，需要多少个？",
    ][t]
    lines = [
        f"{b} × {ans} = {a}米",
        f"{a} ÷ {b} = {ans}个",
    ]
    return ins, lines, ans


_reg("plant_circle_closed", plant_circle_closed)


# 18. 方阵最外层人数
def square_formation_outer(rng):
    a = rng.randint(5, 15)
    ans = (a - 1) * 4
    obj, unit = rng.choice([("同学", "人"), ("棋子", "枚"), ("花盆", "盆"), ("士兵", "人")])
    t = rng.randrange(4)
    ins = [
        f"{obj}排成一个{a}行{a}列的方阵，最外层一共有多少{unit}？",
        f"用{obj}摆成{a}行{a}列的方阵，最外层有多少{unit}？",
        f"同学们组成{a}行{a}列的方阵做操，最外层一圈有多少{unit}？",
        f"一个{a}行{a}列的{obj}方阵，最外面一层共多少{unit}？",
    ][t]
    lines = [
        f"每边段数 = {a} - 1 = {a - 1}",
        f"{a - 1} × 4 = {ans}{unit}",
    ]
    return ins, lines, ans


_reg("square_formation_outer", square_formation_outer)


# 19. 等差数列第3项
def arith_seq_third(rng):
    a = rng.randint(10, 40)
    b = rng.randint(3, 12)
    s2 = a + b
    ans = a + 2 * b
    obj = rng.choice([("页", "看"), ("个", "写"), ("道", "做"), ("只", "折")])
    unit, verb = obj
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"{n1}第一天{verb}{a}{unit}，以后每天比前一天多{verb}{b}{unit}，第3天{verb}多少{unit}？",
        f"小红第一天{verb}{a}{unit}，从第二天起每天比前一天多{verb}{b}{unit}，第3天{verb}多少{unit}？",
        f"小明练习写字，第一天{a}{unit}，以后每天比前一天多{verb}{b}{unit}，第3天{verb}多少{unit}？",
        f"小华第一天{verb}{a}{unit}，以后每天都比前一天多{b}{unit}，第3天{verb}了多少{unit}？",
    ][t]
    lines = [
        f"第2天 = {a} + {b} = {s2}{unit}",
        f"第3天 = {s2} + {b} = {ans}{unit}",
    ]
    return ins, lines, ans


_reg("arith_seq_third", arith_seq_third)


# 20. 等差数列3天总和
def arith_seq_sum3(rng):
    a = rng.randint(10, 30)
    b = rng.randint(3, 10)
    s2 = a + b
    s3 = a + 2 * b
    ans = a + s2 + s3
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"{n1}第一天写{a}个大字，以后每天比前一天多写{b}个，3天一共写多少个？",
        f"小明第一天跑{a}米，以后每天比前一天多跑{b}米，3天共跑多少米？",
        f"小红第一天背{a}个单词，以后每天比前一天多背{b}个，3天一共背多少个？",
        f"小军第一天看{a}页书，以后每天比前一天多看{b}页，3天共看多少页？",
    ][t]
    lines = [
        f"第2天 = {a} + {b} = {s2}个",
        f"第3天 = {s2} + {b} = {s3}个",
        f"总和 = {a} + {s2} + {s3} = {ans}个",
    ]
    return ins, lines, ans


_reg("arith_seq_sum3", arith_seq_sum3)


# 21. 倒多少使两者相等
def pour_equal(rng):
    a = rng.randint(20, 60)
    b = a - rng.randint(1, 15) * 2
    diff = a - b
    ans = diff // 2
    obj, unit = rng.choice([("果汁", "毫升"), ("油", "千克"), ("苹果", "个"), ("书", "本")])
    t = rng.randrange(4)
    ins = [
        f"甲杯有{a}{unit}{obj}，乙杯有{b}{unit}，甲倒给乙多少{unit}后两杯同样多？",
        f"甲筐有{a}{unit}{obj}，乙筐有{b}{unit}，从甲筐拿多少{unit}到乙筐，两筐一样多？",
        f"小明有{a}{unit}{obj}，小红有{b}{unit}，小明给小红多少{unit}后两人同样多？",
        f"甲桶有{a}{unit}{obj}，乙桶有{b}{unit}，甲桶倒多少{unit}给乙桶，两桶相等？",
    ][t]
    lines = [
        f"{a} - {b} = {diff}{unit}",
        f"{diff} ÷ 2 = {ans}{unit}",
    ]
    return ins, lines, ans


_reg("pour_equal", pour_equal)


# 22. 剪去同样长后成2倍
def rope_cut_equal(rng):
    b = rng.randint(10, 25)
    a = rng.randint(b + 1, 2 * b - 1)
    ans = 2 * b - a
    t = rng.randrange(4)
    ins = [
        f"甲绳长{a}米，乙绳长{b}米，剪去同样长的一段后，甲绳剩下的正好是乙绳剩下的2倍，剪去多少米？",
        f"两根绳子分别长{a}米和{b}米，剪去同样长后，长绳剩下的是短绳剩下的2倍，各剪去多少米？",
        f"甲绳{a}米、乙绳{b}米，同时剪去相同的长度后，甲绳剩下的长度是乙绳的2倍，剪去了多少米？",
        f"一根绳子长{a}米，另一根长{b}米，剪去同样长后，第一根剩下的是第二根的2倍，剪去多少米？",
    ][t]
    lines = [
        f"{b} × 2 = {2 * b}米",
        f"{2 * b} - {a} = {ans}米",
    ]
    return ins, lines, ans


_reg("rope_cut_equal", rope_cut_equal)


# 23. 两种水果求单价
def money_two_fruits_price(rng):
    d = rng.randint(3, 10)
    aa = rng.randint(2, 5)
    ad = aa * d
    bb = rng.randint(2, 5)
    ans = rng.randint(2, 8)
    c = ad + bb * ans
    f1, f2 = rng.sample(FRUITS, 2)
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"买{aa}千克{f1}和{bb}千克{f2}共花{c}元，{f1}每千克{d}元，{f2}每千克多少元？",
        f"{n1}买{aa}千克{f1}、{bb}千克{f2}，一共用去{c}元，已知{f1}每千克{d}元，{f2}每千克多少元？",
        f"妈妈买{aa}千克{f1}和{bb}千克{f2}，付{c}元，{f1}每千克{d}元，{f2}每千克多少元？",
        f"水果店{aa}千克{f1}加{bb}千克{f2}共{c}元，{f1}每千克{d}元，{f2}每千克多少元？",
    ][t]
    lines = [
        f"{aa} × {d} = {ad}元",
        f"{c} - {ad} = {bb * ans}元",
        f"{bb * ans} ÷ {bb} = {ans}元",
    ]
    return ins, lines, ans


_reg("money_two_fruits_price", money_two_fruits_price)


# 24. 往返求返回速度
def round_trip_speed(rng):
    for _ in range(50):
        a = rng.randint(30, 60)
        bb = rng.randint(2, 4)
        c = rng.choice([2, 3, 4, 6])
        if (a * bb) % c == 0:
            break
    d = a * bb
    ans = d // c
    t = rng.randrange(4)
    ins = [
        f"汽车从甲地到乙地每小时行{a}千米，行了{bb}小时；返回时用了{c}小时，返回时每小时行多少千米？",
        f"小明骑车去公园每小时行{a}千米，{bb}小时到达，原路返回用了{c}小时，返回每小时行多少千米？",
        f"一艘轮船从甲港到乙港每小时行{a}千米，{bb}小时到达，返回用{c}小时，返回速度是多少？",
        f"货车送货每小时行{a}千米，{bb}小时到达，空车返回用{c}小时，返回时每小时行多少千米？",
    ][t]
    lines = [
        f"{a} × {bb} = {d}千米",
        f"{d} ÷ {c} = {ans}千米",
    ]
    return ins, lines, ans


_reg("round_trip_speed", round_trip_speed)


# 25. 离中点还有多少
def midpoint_distance(rng):
    a = rng.randint(20, 80) * 2
    b = rng.randint(5, a // 2 - 2)
    half = a // 2
    ans = half - b
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"一条路长{a}米，{n1}已经走了{b}米，离中点还有多少米？",
        f"小红跑{a}米，已经跑了{b}米，再跑多少米正好到中点？",
        f"一段公路长{a}米，修了{b}米后，离中点还有多少米？",
        f"小明看一本{a}页的书，已经看{b}页，再看多少页正好看完一半？",
    ][t]
    lines = [
        f"{a} ÷ 2 = {half}米",
        f"{half} - {b} = {ans}米",
    ]
    return ins, lines, ans


_reg("midpoint_distance", midpoint_distance)


# 26. 编页码用多少数字
def page_digit_count(rng):
    a = rng.randint(20, 99)
    p = a - 9
    ans = p * 2 + 9
    t = rng.randrange(4)
    ins = [
        f"一本书有{a}页，编页码一共要用多少个数字？",
        f"一本故事书共{a}页，排页码时一共需要多少个数字？",
        f"给一本{a}页的书编页码，共要用多少个数字？",
        f"一本书{a}页，从第1页到最后一页，编页码共用多少个数字？",
    ][t]
    lines = [
        f"{a} - 9 = {p}页",
        f"{p} × 2 + 9 = {ans}个",
    ]
    return ins, lines, ans


_reg("page_digit_count", page_digit_count)


# 27. 两位数构造
def two_digit_number(rng):
    a = rng.randint(1, 9)
    b = rng.randint(0, 9)
    ans = 10 * a + b
    t = rng.randrange(4)
    ins = [
        f"一个两位数，十位数字是{a}，个位数字是{b}，这个数是多少？",
        f"一个数，十位上是{a}，个位上是{b}，这个数是多少？",
        f"一个两位数，十位上是{a}，个位上是{b}，这个数是多少？",
        f"一个两位数，个位是{b}，十位是{a}，这个数是多少？",
    ][t]
    lines = [
        f"十位 = 10 × {a} = {10 * a}",
        f"这个数 = {10 * a} + {b} = {ans}",
    ]
    return ins, lines, ans


_reg("two_digit_number", two_digit_number)


# 28. 数字调换后的差
def reverse_digits_diff(rng):
    a = rng.randint(2, 9)
    b = rng.randint(1, a - 1)
    big = 10 * a + b
    small = 10 * b + a
    ans = big - small
    t = rng.randrange(6)
    ins = [
        f"一个两位数，十位是{a}、个位是{b}，把数字调换位置后，新数比原数小多少？",
        f"一个两位数，十位数字{a}，个位数字{b}，交换两个数字的位置，得到的新数比原数少多少？",
        f"有一个两位数，十位上是{a}，个位上是{b}，把十位和个位颠倒后，新数比原数小多少？",
        f"一个两位数，十位是{a}、个位是{b}，调换数字顺序后，原数比新数大多少？",
        f"一个两位数，十位数字{a}、个位数字{b}，倒过来写得到的新数比原数少多少？",
        f"有一个两位数，十位上是{a}，个位上是{b}，交换数位后，原数比新数多多少？",
    ][t]
    lines = [
        f"原数 = 10 × {a} + {b} = {big}",
        f"新数 = 10 × {b} + {a} = {small}",
        f"相差 = {big} - {small} = {ans}",
    ]
    return ins, lines, ans


_reg("reverse_digits_diff", reverse_digits_diff)


# 29. 有余数除法求被除数
def reverse_remainder_dividend(rng):
    bb = rng.randint(3, 9)
    q = rng.randint(3, 9)
    r = rng.randint(1, bb - 1)
    ans = bb * q + r
    t = rng.randrange(4)
    ins = [
        f"一个数除以{bb}，商是{q}，余数是{r}，这个数是多少？",
        f"某数除以{bb}得商{q}余{r}，求这个数。",
        f"一个除法算式，除数是{bb}，商{q}，余数{r}，被除数是多少？",
        f"一个数除以{bb}，商{q}余{r}，这个数是几？",
    ][t]
    lines = [
        f"商乘除数 = {bb} × {q} = {bb * q}",
        f"被除数 = {bb * q} + {r} = {ans}",
    ]
    return ins, lines, ans


_reg("reverse_remainder_dividend", reverse_remainder_dividend)


# 30. 逆推求原数
def mystery_number_reverse(rng):
    for _ in range(50):
        a = rng.randint(2, 6)
        bb = rng.randint(3, 6)
        c = rng.randint(2, 12)
        d = rng.randint(20, 80)
        if (d + c) % bb == 0 and (d + c) // bb > a:
            break
    s = d + c
    prod = s // bb
    ans = prod - a
    t = rng.randrange(4)
    ins = [
        f"一个数加上{a}，再乘{bb}，然后减去{c}，结果是{d}，这个数是多少？",
        f"某数加上{a}，乘{bb}，减去{c}，得{d}，求这个数。",
        f"一个数先加{a}，再乘{bb}，再减{c}，最后得到{d}，这个数是多少？",
        f"小明想了一个数，把它加上{a}，乘{bb}，减去{c}，结果是{d}，这个数是多少？",
    ][t]
    lines = [
        f"和 = {d} + {c} = {s}",
        f"积 = {s} ÷ {bb} = {prod}",
        f"原数 = {prod} - {a} = {ans}",
    ]
    return ins, lines, ans


_reg("mystery_number_reverse", mystery_number_reverse)


# 31. 半价购买
def half_price_buy(rng):
    a = rng.randint(10, 40) * 2
    bb = rng.randint(3, 10)
    h = a // 2
    ans = h * bb
    obj = rng.choice(STATIONERY + FRUITS)
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"{obj}原价每个{a}元，店庆半价出售，{n1}买{bb}个要多少元？",
        f"商店促销，{obj}每个{a}元，现在半价销售，买{bb}个共多少元？",
        f"{obj}原来每个{a}元，打折后半价，{n1}买{bb}个应付多少元？",
        f"文具店{obj}每个{a}元，活动期间半价，买{bb}个要花多少元？",
    ][t]
    lines = [
        f"{a} ÷ 2 = {h}元",
        f"{h} × {bb} = {ans}元",
    ]
    return ins, lines, ans


_reg("half_price_buy", half_price_buy)


# 32. 按打计价
def dozen_price(rng):
    a = rng.choice([24, 36, 48, 60, 72, 96])
    bb = rng.randint(3, 11)
    each = a // 12
    ans = each * bb
    obj = rng.choice(["鸡蛋", "铅笔", "乒乓球", "羽毛球", "本子"])
    t = rng.randrange(4)
    ins = [
        f"一打{obj}共{a}元（1打=12个），买{bb}个要多少元？",
        f"{obj}每打{a}元（12个），买{bb}个应付多少元？",
        f"商店里一打{obj}卖{a}元，照这样算，买{bb}个需要多少元？",
        f"{obj}12个共{a}元，买{bb}个要多少元？",
    ][t]
    lines = [
        f"{a} ÷ 12 = {each}元",
        f"{each} × {bb} = {ans}元",
    ]
    return ins, lines, ans


_reg("dozen_price", dozen_price)


# 33. 两两和求一个
def three_weights_pair_sums(rng):
    for _ in range(50):
        jia = rng.randint(20, 50)
        yi = rng.randint(20, 50)
        bing = rng.randint(20, 50)
        a = jia + yi
        bb = yi + bing
        c = jia + bing
        if (a + c - bb) % 2 == 0:
            break
    diff = a + c - bb
    ans = diff // 2
    t = rng.randrange(4)
    ins = [
        f"甲乙两人共重{a}千克，乙丙两人共重{bb}千克，甲丙两人共重{c}千克，甲重多少千克？",
        f"小明和小红共{a}千克，小红和小丽共{bb}千克，小明和小丽共{c}千克，小明重多少千克？",
        f"三筐水果，甲乙两筐共{a}千克，乙丙两筐共{bb}千克，甲丙两筐共{c}千克，甲筐重多少千克？",
        f"甲乙共有{a}元，乙丙共有{bb}元，甲丙共有{c}元，甲有多少元？",
    ][t]
    lines = [
        f"甲的两倍 = {a} + {c} = {a + c}",
        f"减去乙丙 = {a + c} - {bb} = {diff}",
        f"甲 = {diff} ÷ 2 = {ans}千克",
    ]
    return ins, lines, ans


_reg("three_weights_pair_sums", three_weights_pair_sums)


# 34. 每件利润乘数量
def profit_per_item(rng):
    a = rng.randint(20, 60)
    bb = a + rng.randint(5, 30)
    c = rng.randint(3, 10)
    pf = bb - a
    ans = pf * c
    obj = rng.choice(GOODS)
    t = rng.randrange(4)
    ins = [
        f"商店进一批{obj}，每个进价{a}元，售价{bb}元，卖出{c}个共赚多少元？",
        f"一件{obj}成本{a}元，卖{bb}元，商店卖出{c}件，一共盈利多少元？",
        f"文具店每个{obj}进价{a}元、售价{bb}元，卖{c}个能赚多少元？",
        f"老板进了一批{obj}，每个{a}元，以每个{bb}元卖出，卖出{c}个共赚多少元？",
    ][t]
    lines = [
        f"{bb} - {a} = {pf}元",
        f"{pf} × {c} = {ans}元",
    ]
    return ins, lines, ans


_reg("profit_per_item", profit_per_item)


# 35. 火车过桥时间
def train_bridge_time(rng):
    for _ in range(50):
        a = rng.randint(100, 300)
        bb = rng.randint(200, 800)
        vs = [v for v in (10, 20, 25, 50) if (a + bb) % v == 0]
        if vs:
            v = rng.choice(vs)
            break
    length = a + bb
    ans = length // v
    t = rng.randrange(4)
    ins = [
        f"一列火车长{a}米，以每秒{v}米的速度通过一座长{bb}米的大桥，完全通过需要多少秒？",
        f"火车长{a}米，每秒行{v}米，通过一座{bb}米长的桥，从车头上桥到车尾离桥要多少秒？",
        f"一列长{a}米的火车，以每秒{v}米的速度穿过长{bb}米的隧道，需要多少秒？",
        f"火车以每秒{v}米的速度通过一座长{bb}米的桥，火车自身长{a}米，完全通过需多少秒？",
    ][t]
    lines = [
        f"{a} + {bb} = {length}米",
        f"{length} ÷ {v} = {ans}秒",
    ]
    return ins, lines, ans


_reg("train_bridge_time", train_bridge_time)


# 36. 顺水航行
def downstream_speed(rng):
    a = rng.randint(20, 50)
    bb = rng.randint(2, 8)
    c = rng.randint(2, 5)
    speed = a + bb
    ans = speed * c
    t = rng.randrange(4)
    ins = [
        f"轮船在静水中每小时行{a}千米，水流速度每小时{bb}千米，顺水航行{c}小时行多少千米？",
        f"一艘船静水速度每小时{a}千米，水速每小时{bb}千米，顺水{c}小时能行多少千米？",
        f"小船在静水中每小时划{a}千米，河水每小时流{bb}千米，顺水划{c}小时行多少千米？",
        f"轮船静水时速{a}千米，水流时速{bb}千米，顺水航行{c}小时，共行多少千米？",
    ][t]
    lines = [
        f"{a} + {bb} = {speed}千米",
        f"{speed} × {c} = {ans}千米",
    ]
    return ins, lines, ans


_reg("downstream_speed", downstream_speed)


# 37. 每周做5天，几周做完
def work_week_days(rng):
    bb = rng.randint(3, 8)
    w = rng.randint(2, 8)
    a = 5 * bb * w
    per_week = 5 * bb
    t = rng.randrange(4)
    ins = [
        f"一批零件共{a}个，工人每天做{bb}个，每周工作5天，做完这批零件需要几周？",
        f"要加工{a}个零件，每天做{bb}个，一周做5天，几周可以做完？",
        f"工厂每天生产{bb}台机器，每周生产5天，完成{a}台的订单需要几周？",
        f"小明每天背{bb}个单词，每周背5天，背完{a}个单词需要几周？",
    ][t]
    lines = [
        f"5 × {bb} = {per_week}个",
        f"{a} ÷ {per_week} = {w}周",
    ]
    return ins, lines, w


_reg("work_week_days", work_week_days)


# 38. 两种分组都余r，至少多少人
def lcm_remainder_people(rng):
    for _ in range(50):
        k1 = rng.randint(3, 6)
        k2 = rng.randint(4, 12)
        if gcd(k1, k2) == 1:
            break
    r = rng.randint(1, min(3, k1 - 1))
    prod = k1 * k2
    ans = prod + r
    t = rng.randrange(4)
    ins = [
        f"同学们排队，每{k1}人一组多{r}人，每{k2}人一组也多{r}人，至少有多少人？",
        f"一批同学，分成{k1}人一组剩{r}人，分成{k2}人一组也剩{r}人，至少有多少人？",
        f"小朋友排队，每{k1}人一行多{r}人，每{k2}人一行也多{r}人，最少有多少人？",
        f"学生分组活动，每{k1}人一组余{r}人，每{k2}人一组也余{r}人，至少有多少人？",
    ][t]
    lines = [
        f"{k1} × {k2} = {prod}人",
        f"{prod} + {r} = {ans}人",
    ]
    return ins, lines, ans


_reg("lcm_remainder_people", lcm_remainder_people)


# 39. 三角形求第三角
def triangle_angle(rng):
    a = rng.randint(30, 80)
    bb = rng.randint(30, 150 - a)
    s = a + bb
    ans = 180 - s
    t = rng.randrange(4)
    ins = [
        f"三角形中两个角分别是{a}°和{bb}°，第三个角是多少度？",
        f"一个三角形，两个内角分别为{a}°和{bb}°，另一个角是多少度？",
        f"三角形的两个角是{a}°和{bb}°，第三个角是多少度？",
        f"已知三角形两个内角分别是{a}°和{bb}°，求第三个角的度数。",
    ][t]
    lines = [
        f"两角和 = {a} + {bb} = {s}°",
        f"第三个角 = 180 - {s} = {ans}°",
    ]
    return ins, lines, ans


_reg("triangle_angle", triangle_angle)


# 40. 等腰三角形求底角
def isosceles_base_angle(rng):
    a = rng.randint(10, 60) * 2
    rest = 180 - a
    ans = rest // 2
    t = rng.randrange(4)
    ins = [
        f"等腰三角形的顶角是{a}°，它的一个底角是多少度？",
        f"一个等腰三角形，顶角{a}°，底角是多少度？",
        f"等腰三角形顶角为{a}°，每个底角是多少度？",
        f"一个等腰三角形的顶角是{a}°，它的底角是多少度？",
    ][t]
    lines = [
        f"两底角和 = 180 - {a} = {rest}°",
        f"一个底角 = {rest} ÷ 2 = {ans}°",
    ]
    return ins, lines, ans


_reg("isosceles_base_angle", isosceles_base_angle)


# 41. 测验得分求答对数
def test_scoring(rng):
    for _ in range(50):
        a = rng.randint(3, 5)
        bb = rng.randint(1, 3)
        c = rng.randint(10, 20)
        x = rng.randint(5, c - 2)
        d = a * x - bb * (c - x)
        if d > 0:
            break
    gap = d + bb * c
    ab = a + bb
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"一次测验共{c}题，答对一题得{a}分，答错一题扣{bb}分，{n1}得了{d}分，他答对了几题？",
        f"数学竞赛共{c}道题，答对一道得{a}分，答错一道扣{bb}分，小明得{d}分，他答对几道？",
        f"知识竞赛{c}题，答对一题加{a}分，答错一题扣{bb}分，小红得了{d}分，她答对了几题？",
        f"一次考试有{c}道题，答对得{a}分，答错扣{bb}分，小华得了{d}分，他答对了多少题？",
    ][t]
    lines = [
        f"{d} + {bb} × {c} = {gap}分",
        f"{a} + {bb} = {ab}分",
        f"{gap} ÷ {ab} = {x}题",
    ]
    return ins, lines, x


_reg("test_scoring", test_scoring)


# 42. 进水管出水管同时开
def pool_fill_with_drain(rng):
    a = rng.randint(10, 30)
    bb = rng.randint(2, a - 2)
    days = rng.randint(2, 6)
    c = (a - bb) * days
    net = a - bb
    t = rng.randrange(4)
    ins = [
        f"水池容量{c}立方米，进水管每小时注水{a}立方米，出水管每小时放水{bb}立方米，两管同时开，几小时注满？",
        f"一个水池能装{c}立方米水，进水管每小时进{a}立方米，出水管每小时出{bb}立方米，同时开放几小时注满？",
        f"泳池蓄水{c}立方米，A管每小时注{a}立方米，B管每小时排{bb}立方米，两管齐开几小时注满？",
        f"水箱容积{c}立方米，进水管每小时{a}立方米，出水管每小时{bb}立方米，同时打开几小时能注满？",
    ][t]
    lines = [
        f"{a} - {bb} = {net}立方米",
        f"{c} ÷ {net} = {days}小时",
    ]
    return ins, lines, days


_reg("pool_fill_with_drain", pool_fill_with_drain)


# 43. 余下的几天看完，每天看多少
def read_plan_adjust(rng):
    for _ in range(50):
        a = rng.randint(100, 300)
        bb = rng.randint(5, 15)
        c = rng.randint(5, 15)
        d = rng.randint(2, 6)
        done = bb * c
        left = a - done
        if left > 0 and left % d == 0:
            break
    ans = left // d
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"一本书{a}页，{n1}已经看了{bb}天，每天看{c}页，余下的要在{d}天内看完，平均每天看多少页？",
        f"一本故事书{a}页，小红看了{bb}天，每天{c}页，剩下的计划{d}天看完，每天应看多少页？",
        f"小明看一本{a}页的书，已经看了{bb}天，每天看{c}页，余下的{d}天看完，平均每天看几页？",
        f"一本书共{a}页，小军前{bb}天每天看{c}页，剩下的要{d}天看完，每天要看多少页？",
    ][t]
    lines = [
        f"{bb} × {c} = {done}页",
        f"{a} - {done} = {left}页",
        f"{left} ÷ {d} = {ans}页",
    ]
    return ins, lines, ans


_reg("read_plan_adjust", read_plan_adjust)


# 44. 两组加权平均
def weighted_average_two_groups(rng):
    a = rng.randint(3, 8)
    bb = rng.randint(70, 95)
    c = rng.randint(3, 8)
    d = rng.randint(70, 95)
    s1 = a * bb
    s2 = c * d
    total = a + c
    ans = Fraction(s1 + s2, total)
    t = rng.randrange(4)
    ins = [
        f"甲组{a}人平均每人跳{bb}下，乙组{c}人平均每人跳{d}下，两组同学平均每人跳多少下？",
        f"一班{a}人平均身高{bb}厘米，二班{c}人平均身高{d}厘米，两班同学平均身高多少厘米？",
        f"男生{a}人平均体重{bb}千克，女生{c}人平均体重{d}千克，全班平均体重多少千克？",
        f"甲组{a}人平均每人得{bb}分，乙组{c}人平均每人得{d}分，两组平均每人得多少分？",
    ][t]
    lines = [
        f"{a} × {bb} = {s1}下",
        f"{c} × {d} = {s2}下",
        f"({s1} + {s2}) ÷ {total} = {num(ans)}下",
    ]
    return ins, lines, ans


_reg("weighted_average_two_groups", weighted_average_two_groups)


# 45. 爷爷年龄是孙子的几倍多几
def grandpa_age_multiple_more(rng):
    bb = rng.randint(5, 9)
    c = rng.randint(1, 5)
    sun = rng.randint(6, 12)
    a = sun * bb + c
    rest = a - c
    ans = rest // bb
    t = rng.randrange(4)
    ins = [
        f"爷爷今年{a}岁，比孙子年龄的{bb}倍还多{c}岁，孙子今年多少岁？",
        f"爸爸今年{a}岁，比小明年龄的{bb}倍多{c}岁，小明今年多少岁？",
        f"奶奶今年{a}岁，比小红年龄的{bb}倍还多{c}岁，小红今年多少岁？",
        f"老师今年{a}岁，比小华年龄的{bb}倍多{c}岁，小华今年多少岁？",
    ][t]
    lines = [
        f"{a} - {c} = {rest}岁",
        f"{rest} ÷ {bb} = {ans}岁",
    ]
    return ins, lines, ans


_reg("grandpa_age_multiple_more", grandpa_age_multiple_more)


# 46. 两积比较
def compare_two_products(rng):
    for _ in range(50):
        a, bb = rng.randint(10, 30), rng.randint(3, 8)
        c, d = rng.randint(10, 30), rng.randint(2, 6)
        p1, p2 = a * bb, c * d
        if p1 != p2:
            break
    big, small = max(p1, p2), min(p1, p2)
    ans = big - small
    t = rng.randrange(4)
    ins = [
        f"商店运来{a}箱苹果每箱{bb}千克，又运来{c}箱梨每箱{d}千克，苹果和梨相差多少千克？",
        f"食堂买{a}袋大米每袋{bb}千克，又买{c}袋面粉每袋{d}千克，大米和面粉相差多少千克？",
        f"水果店运来{a}筐橘子每筐{bb}千克，又运来{c}筐香蕉每筐{d}千克，两种水果相差多少千克？",
        f"工地运来{a}车砖每车{bb}块，又运来{c}车瓦每车{d}块，砖和瓦相差多少块？",
    ][t]
    lines = [
        f"{a} × {bb} = {p1}千克",
        f"{c} × {d} = {p2}千克",
        f"{big} - {small} = {ans}千克",
    ]
    return ins, lines, ans


_reg("compare_two_products", compare_two_products)


# 47. 用去 1/n，剩下比用去多多少
def fraction_used_vs_left(rng):
    n = rng.randint(3, 5)
    a = rng.randint(5, 20) * n
    part = a // n
    both = part * 2
    ans = a - both
    obj = rng.choice(["公路", "水渠", "跑道", "绳子"])
    t = rng.randrange(4)
    ins = [
        f"一条{obj}长{a}米，用去了 1/{n}，剩下的比用去的多多少米？",
        f"一根{obj}长{a}米，用去 1/{n}，剩下的比用去的多多少米？",
        f"修一条长{a}米的{obj}，已经修了 1/{n}，剩下的比已修的多多少米？",
        f"一条{obj}全长{a}米，用去 1/{n}后，剩下的比用去的多多少米？",
    ][t]
    lines = [
        f"{a} ÷ {n} = {part}米",
        f"{part} × 2 = {both}米",
        f"{a} - {both} = {ans}米",
    ]
    return ins, lines, ans


_reg("fraction_used_vs_left", fraction_used_vs_left)


# 48. 三个连续自然数求中间
def consecutive_numbers_middle(rng):
    m = rng.randint(4, 40)
    a = 3 * m
    t = rng.randrange(5)
    ins = [
        f"三个连续自然数的和是{a}，中间的数是多少？",
        f"三个连续整数的和是{a}，正中间的数是多少？",
        f"小明翻开书，看到的三个连续页码之和是{a}，中间的页码是多少？",
        f"三个连续自然数相加得{a}，中间那个数是几？",
        f"三个连续自然数的和为{a}，排在中间的是多少？",
    ][t]
    lines = [
        f"三数和 = {m - 1} + {m} + {m + 1} = {a}",
        f"中间数 = {a} ÷ 3 = {m}",
    ]
    return ins, lines, m


_reg("consecutive_numbers_middle", consecutive_numbers_middle)


# 49. 三个连续偶数求最大
def consecutive_even_largest(rng):
    m = rng.randint(4, 40) * 2
    a = 3 * m
    ans = m + 2
    t = rng.randrange(5)
    ins = [
        f"三个连续偶数的和是{a}，最大的偶数是多少？",
        f"三个连续偶数相加得{a}，其中最大的是几？",
        f"三个连续偶数的和为{a}，最大的一个是多少？",
        f"有三个连续偶数，它们的和是{a}，最大的偶数是多少？",
        f"三个连续偶数的和是{a}，排在最后的是多少？",
    ][t]
    lines = [
        f"中间偶数 = {a} ÷ 3 = {m}",
        f"最大的偶数 = {m} + 2 = {ans}",
    ]
    return ins, lines, ans


_reg("consecutive_even_largest", consecutive_even_largest)


# 50. 三位数构造
def three_digit_number(rng):
    a = rng.randint(1, 9)
    bb = rng.randint(0, 9)
    c = rng.randint(0, 9)
    h = 100 * a
    hb = h + 10 * bb
    ans = hb + c
    t = rng.randrange(4)
    ins = [
        f"一个三位数，百位是{a}、十位是{bb}、个位是{c}，这个数是多少？",
        f"一个数，百位上是{a}，十位上是{bb}，个位上是{c}，这个数是多少？",
        f"有一个三位数，百位数字{a}，十位数字{bb}，个位数字{c}，这个数是多少？",
        f"一个三位数，个位是{c}，十位是{bb}，百位是{a}，这个数是多少？",
    ][t]
    lines = [
        f"百位 = 100 × {a} = {h}",
        f"前两位 = {h} + 10 × {bb} = {hb}",
        f"这个数 = {hb} + {c} = {ans}",
    ]
    return ins, lines, ans


_reg("three_digit_number", three_digit_number)


# 51. 又买来一些，甲架再分多少
def shelves_redistribute(rng):
    for _ in range(50):
        a = rng.randint(20, 60)
        bb = rng.randint(20, 60)
        c = rng.randint(5, 30)
        total = a + bb + c
        if total % 2 == 0 and bb + c > a:
            break
    each = total // 2
    ans = each - a
    t = rng.randrange(4)
    ins = [
        f"甲书架有{a}本书，乙书架有{bb}本，又买来{c}本，甲书架应再分多少本，两个书架的书才同样多？",
        f"一班有{a}本图书，二班有{bb}本，学校又买来{c}本，一班再分多少本两班就一样多？",
        f"甲筐有{a}个苹果，乙筐有{bb}个，又摘来{c}个，甲筐再放多少个两筐相等？",
        f"小明有{a}元，小红有{bb}元，妈妈又拿出{c}元，小明再得多少元两人才同样多？",
    ][t]
    lines = [
        f"{a} + {bb} + {c} = {total}本",
        f"{total} ÷ 2 = {each}本",
        f"{each} - {a} = {ans}本",
    ]
    return ins, lines, ans


_reg("shelves_redistribute", shelves_redistribute)


# 52. 几年后父亲年龄是儿子2倍
def age_future_multiple(rng):
    bb = rng.randint(6, 12)
    a = rng.randint(2 * bb + 2, 2 * bb + 20)
    twice = 2 * bb
    ans = a - twice
    t = rng.randrange(4)
    ins = [
        f"父亲今年{a}岁，儿子今年{bb}岁，几年后父亲的年龄正好是儿子的2倍？",
        f"爸爸今年{a}岁，小明今年{bb}岁，再过几年爸爸的年龄是小明的2倍？",
        f"妈妈今年{a}岁，女儿今年{bb}岁，几年后妈妈的年龄是女儿的2倍？",
        f"哥哥今年{a}岁，弟弟今年{bb}岁，几年后哥哥的年龄正好是弟弟的2倍？",
    ][t]
    lines = [
        f"{bb} × 2 = {twice}岁",
        f"{a} - {twice} = {ans}年",
    ]
    return ins, lines, ans


_reg("age_future_multiple", age_future_multiple)


# 53. 同向追及
def distance_chase(rng):
    v1 = rng.randint(30, 70)
    v2 = rng.randint(20, v1 - 5)
    t = rng.randint(2, 6)
    a = (v1 - v2) * t
    diff = v1 - v2
    p = rng.randrange(4)
    ins = [
        f"甲乙两人相距{a}米，甲每分钟走{v1}米，乙每分钟走{v2}米，两人同时同向而行，甲几分钟追上乙？",
        f"小明和小红相距{a}米，小明每分钟跑{v1}米，小红每分钟跑{v2}米，两人同时同向出发，小明几分钟追上小红？",
        f"快车在慢车后面{a}千米，快车每小时行{v1}千米，慢车每小时行{v2}千米，快车几小时追上慢车？",
        f"弟弟在哥哥前面{a}米，哥哥每秒跑{v1}米，弟弟每秒跑{v2}米，哥哥几秒追上弟弟？",
    ][p]
    lines = [
        f"{v1} - {v2} = {diff}米",
        f"{a} ÷ {diff} = {t}分钟",
    ]
    return ins, lines, t


_reg("distance_chase", distance_chase)


# 54. 一半的一半
def fraction_of_fraction(rng):
    a = rng.randint(4, 20) * 4
    half = a // 2
    ans = half // 2
    obj, unit = rng.choice([("一本书", "页"), ("一根绳子", "米"), ("一桶油", "千克"), ("一袋米", "千克")])
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"{obj}共{a}{unit}，第一天用去一半，第二天用去剩下的一半，第二天用去多少{unit}？",
        f"{n1}看{obj}，共{a}{unit}，第一天看了一半，第二天看了余下的一半，第二天看多少{unit}？",
        f"{obj}有{a}{unit}，第一次用去一半，第二次用去余下的一半，第二次用去多少{unit}？",
        f"{obj}一共{a}{unit}，第一天用去一半，第二天用去剩下的一半，第二天用了多少{unit}？",
    ][t]
    lines = [
        f"{a} ÷ 2 = {half}{unit}",
        f"{half} ÷ 2 = {ans}{unit}",
    ]
    return ins, lines, ans


_reg("fraction_of_fraction", fraction_of_fraction)


# 55. 买3送1
def buy3get1_pay(rng):
    groups = rng.randint(2, 8)
    a = groups * 4
    p = rng.randint(3, 12)
    pay = groups * 3
    ans = pay * p
    obj = rng.choice(GOODS)
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"文具店促销：买3个送1个。{n1}买{a}个{obj}，每个{p}元，一共要付多少元？",
        f"商店做活动，{obj}买3送1，小明买{a}个，每个{p}元，应付多少元？",
        f"超市{obj}搞促销，买3个送1个，小红买{a}个，每个{p}元，共付多少元？",
        f"{obj}每个{p}元，店庆买3送1，{n1}买{a}个要花多少元？",
    ][t]
    lines = [
        f"3 + 1 = 4个",
        f"{a} ÷ 4 × 3 = {pay}个",
        f"{pay} × {p} = {ans}元",
    ]
    return ins, lines, ans


_reg("buy3get1_pay", buy3get1_pay)


# 56. 剪几次成几段
def rope_segments(rng):
    bb = rng.randint(2, 6)
    seg = rng.randint(2, 10)
    a = (bb + 1) * seg
    segs = bb + 1
    ans = a // segs
    t = rng.randrange(4)
    ins = [
        f"一根绳子长{a}米，剪{bb}次后剪成同样长的小段，每段长多少米？",
        f"把一根{a}米长的绳子剪{bb}次，剪成相等的小段，每段多少米？",
        f"一根彩带长{a}米，剪了{bb}次，剪成同样长的短带，每根短带长多少米？",
        f"一根{a}米的铁丝剪{bb}次，分成同样长的小段，每段长多少米？",
    ][t]
    lines = [
        f"{bb} + 1 = {segs}段",
        f"{a} ÷ {segs} = {ans}米",
    ]
    return ins, lines, ans


_reg("rope_segments", rope_segments)


# 57. 给完还多，求原有
def money_give_twice(rng):
    for _ in range(50):
        a = rng.randint(30, 80)
        bb = rng.randint(3, 12)
        c = rng.randint(2, 10)
        if a - 2 * bb - c > 0:
            break
    after = a - bb
    yi_after = after - c
    ans = yi_after - bb
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(4)
    ins = [
        f"{n1}有{a}元，他给{n2}{bb}元后，还比{n2}多{c}元，{n2}原来有多少元？",
        f"小明有{a}元，给小红{bb}元后，还比小红多{c}元，小红原来有多少元？",
        f"哥哥有{a}元，给弟弟{bb}元后，仍比弟弟多{c}元，弟弟原来有多少元？",
        f"{n1}有{a}元，拿{bb}元给{n2}，这时还比{n2}多{c}元，{n2}原有多少元？",
    ][t]
    lines = [
        f"{a} - {bb} = {after}元",
        f"{after} - {c} = {yi_after}元",
        f"{yi_after} - {bb} = {ans}元",
    ]
    return ins, lines, ans


_reg("money_give_twice", money_give_twice)


# 58. 几倍后合作
def factory_compare_coop(rng):
    a = rng.randint(10, 30)
    bb = rng.randint(2, 4)
    c = rng.randint(3, 8)
    eb = a * bb
    ans = (a + eb) * c
    t = rng.randrange(4)
    ins = [
        f"甲厂每天生产{a}个零件，乙厂每天产量是甲厂的{bb}倍，两厂合作{c}天共生产多少个？",
        f"小明每天折{a}只纸鹤，小红每天折的是小明的{bb}倍，两人合作{c}天共折多少只？",
        f"甲队每天修{a}米路，乙队每天修的是甲队的{bb}倍，两队合修{c}天共修多少米？",
        f"公鸡每天吃{a}克饲料，母鸡吃的是公鸡的{bb}倍，{c}天共吃多少克？",
    ][t]
    lines = [
        f"{a} × {bb} = {eb}个",
        f"({a} + {eb}) × {c} = {ans}个",
    ]
    return ins, lines, ans


_reg("factory_compare_coop", factory_compare_coop)


# 59. 2倍多几，求两天总和
def read_twice_plus(rng):
    a = rng.randint(10, 30)
    bb = rng.randint(3, 12)
    twice = 2 * a
    second = twice + bb
    ans = a + second
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"{n1}第一天写{a}个大字，第二天写的比第一天的2倍还多{bb}个，两天共写多少个？",
        f"小红第一天看{a}页书，第二天看的比第一天的2倍多{bb}页，两天一共看多少页？",
        f"小明第一天折{a}只纸鹤，第二天折的比第一天的2倍还多{bb}只，两天共折多少只？",
        f"小华第一天跑{a}米，第二天跑的比第一天的2倍多{bb}米，两天共跑多少米？",
    ][t]
    lines = [
        f"{a} × 2 = {twice}个",
        f"{twice} + {bb} = {second}个",
        f"{a} + {second} = {ans}个",
    ]
    return ins, lines, ans


_reg("read_twice_plus", read_twice_plus)


# 60. 每班分几根，还差多少
def divide_classes_shortfall(rng):
    for _ in range(50):
        bb = rng.randint(3, 6)
        c = rng.randint(10, 30)
        a = rng.randint(bb * c - 20, bb * c - 1)
        if a > 0:
            break
    need = bb * c
    ans = need - a
    t = rng.randrange(4)
    ins = [
        f"学校买来{a}根跳绳，分给{bb}个班，每班{c}根，还差多少根？",
        f"幼儿园有{a}个苹果，分给{bb}个班，每班{c}个，还缺多少个？",
        f"老师买来{a}本练习本，发给{bb}个班，每班{c}本，还差多少本？",
        f"学校运来{a}棵树苗，分给{bb}个班，每班{c}棵，还少多少棵？",
    ][t]
    lines = [
        f"{bb} × {c} = {need}根",
        f"{need} - {a} = {ans}根",
    ]
    return ins, lines, ans


_reg("divide_classes_shortfall", divide_classes_shortfall)


# 61. 地砖块数
def tile_floor_count(rng):
    a = rng.randint(4, 10)
    bb = rng.randint(3, 9)
    area = a * bb
    ans = area * 4
    t = rng.randrange(4)
    ins = [
        f"一间教室长{a}米、宽{bb}米，用边长5分米的正方形地砖铺地，需要多少块？",
        f"一个房间长{a}米、宽{bb}米，用边长5分米的方砖铺地，共需多少块？",
        f"客厅长{a}米、宽{bb}米，铺边长5分米的正方形地砖，需要多少块？",
        f"一间会议室长{a}米、宽{bb}米，用边长5分米的地砖铺地，一共需要多少块？",
    ][t]
    lines = [
        f"5 × 5 = 25平方分米",
        f"{a} × {bb} = {area}平方米",
        f"{area} ÷ 0.25 = {ans}块",
    ]
    return ins, lines, ans


_reg("tile_floor_count", tile_floor_count)


# 62. 带的钱不够，还差
def money_afford_shortage(rng):
    a = rng.randint(3, 8)
    bb = rng.randint(5, 15)
    c = rng.randint(5, a * bb - 3)
    cost = a * bb
    ans = cost - c
    obj = rng.choice(STATIONERY + FRUITS)
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"{n1}想买{a}个{obj}，每个{bb}元，他带了{c}元，还差多少元？",
        f"一个{obj}{bb}元，小明买{a}个，只带了{c}元，还差多少元？",
        f"小红想买{a}本{obj}，每本{bb}元，她有{c}元，还缺多少元？",
        f"{obj}每个{bb}元，买{a}个，小华带了{c}元，还差多少元？",
    ][t]
    lines = [
        f"{a} × {bb} = {cost}元",
        f"{cost} - {c} = {ans}元",
    ]
    return ins, lines, ans


_reg("money_afford_shortage", money_afford_shortage)


# 63. 买完还剩，原有多少
def money_surplus(rng):
    a = rng.randint(3, 8)
    bb = rng.randint(5, 15)
    c = rng.randint(5, 40)
    cost = a * bb
    ans = cost + c
    obj = rng.choice(STATIONERY + FRUITS)
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"{n1}买{a}个{obj}，每个{bb}元，还剩{c}元，他原来有多少元？",
        f"小明买{a}本{obj}，每本{bb}元，买完后还剩{c}元，小明原来有多少元？",
        f"小红买{a}千克{obj}，每千克{bb}元，买完还剩{c}元，她原来有多少元？",
        f"妈妈买{a}个{obj}花去一些钱，每个{bb}元，还剩{c}元，妈妈原来有多少元？",
    ][t]
    lines = [
        f"{a} × {bb} = {cost}元",
        f"{cost} + {c} = {ans}元",
    ]
    return ins, lines, ans


_reg("money_surplus", money_surplus)


# 64. 甲是乙的几倍，甲比乙多多少
def savings_compare(rng):
    for _ in range(50):
        bb = rng.randint(2, 5)
        yi = rng.randint(10, 40)
        a = yi * bb
        break
    ans = a - yi
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(4)
    ins = [
        f"{n1}存了{a}元，是{n2}的{bb}倍，{n1}比{n2}多存多少元？",
        f"小明有{a}元，是小红的{bb}倍，小明比小红多多少元？",
        f"果园有苹果树{a}棵，是梨树的{bb}倍，苹果树比梨树多多少棵？",
        f"公鸡有{a}只，是母鸡的{bb}倍，公鸡比母鸡多多少只？",
    ][t]
    lines = [
        f"{a} ÷ {bb} = {yi}元",
        f"{a} - {yi} = {ans}元",
    ]
    return ins, lines, ans


_reg("savings_compare", savings_compare)


# 65. 两天各修 1/3 和 1/4，求剩下
def road_repair_fraction(rng):
    a = rng.randint(2, 15) * 12
    x = a // 3
    y = a // 4
    ans = a - x - y
    obj = rng.choice(["公路", "水渠", "跑道", "围墙"])
    t = rng.randrange(4)
    ins = [
        f"修一条长{a}米的{obj}，第一天修了 1/3，第二天修了 1/4，还剩多少米没修？",
        f"一条{obj}长{a}米，第一天修全长的 1/3，第二天修全长的 1/4，还剩多少米？",
        f"工程队修{a}米的{obj}，第一天修 1/3，第二天修 1/4，还剩多少米？",
        f"一条{a}米长的{obj}，第一天修了 1/3，第二天修了 1/4，还剩多少米？",
    ][t]
    lines = [
        f"{a} ÷ 3 = {x}米",
        f"{a} ÷ 4 = {y}米",
        f"{a} - {x} - {y} = {ans}米",
    ]
    return ins, lines, ans


_reg("road_repair_fraction", road_repair_fraction)


# 66. 用去 1/n 又用去 b，求剩
def rope_fraction_cut(rng):
    for _ in range(50):
        n = rng.randint(3, 5)
        a = rng.randint(5, 20) * n
        b = rng.randint(3, a - a // n - 3)
        if a - a // n - b > 0:
            break
    part = a // n
    ans = a - part - b
    obj, unit = rng.choice([("绳子", "米"), ("公路", "米"), ("水管", "米"), ("彩带", "米")])
    t = rng.randrange(4)
    ins = [
        f"一根{obj}长{a}米，先用去 1/{n}，又用去{b}米，还剩多少米？",
        f"一条{obj}长{a}米，第一次用去 1/{n}，第二次用去{b}米，还剩多少米？",
        f"一根{obj}共{a}米，用去全长的 1/{n}后，又用去{b}米，还剩多少米？",
        f"{obj}长{a}米，先剪去 1/{n}，再剪去{b}米，还剩多少米？",
    ][t]
    lines = [
        f"{a} ÷ {n} = {part}米",
        f"{a} - {part} - {b} = {ans}米",
    ]
    return ins, lines, ans


_reg("rope_fraction_cut", rope_fraction_cut)


# 67. 连桶称重
def oil_barrel(rng):
    a = rng.randint(20, 40)
    b = rng.randint(a // 2 + 1, a - 1)
    ans = 2 * b - a
    t = rng.randrange(4)
    ins = [
        f"一桶油连桶重{a}千克，用去一半油后连桶重{b}千克，桶重多少千克？",
        f"一筐苹果连筐重{a}千克，卖出一半苹果后连筐重{b}千克，筐重多少千克？",
        f"一桶水连桶重{a}千克，倒出一半水后连桶重{b}千克，桶重多少千克？",
        f"一袋米连袋重{a}千克，吃掉一半米后连袋重{b}千克，袋子重多少千克？",
    ][t]
    lines = [
        f"{b} × 2 = {2 * b}千克",
        f"{2 * b} - {a} = {ans}千克",
    ]
    return ins, lines, ans


_reg("oil_barrel", oil_barrel)


# 68. 鸡比兔多，共腿多少
def chicken_rabbit_extra(rng):
    a = rng.randint(3, 10)
    rabbits = rng.randint(3, 12)
    b = 6 * rabbits + 2 * a
    rest = b - 2 * a
    ans = rest // 6
    t = rng.randrange(4)
    ins = [
        f"鸡比兔多{a}只，鸡和兔共有{b}条腿，兔有多少只？",
        f"养殖场鸡比兔多{a}只，数腿共{b}条，兔有多少只？",
        f"笼子里鸡比兔多{a}只，腿一共{b}条，兔有多少只？",
        f"鸡兔同笼，鸡比兔多{a}只，共有{b}条腿，兔有多少只？",
    ][t]
    lines = [
        f"{a} × 2 = {2 * a}条",
        f"{b} - {2 * a} = {rest}条",
        f"{rest} ÷ 6 = {ans}只",
    ]
    return ins, lines, ans


_reg("chicken_rabbit_extra", chicken_rabbit_extra)


# 69. 几年后两人年龄和
def age_brothers_sum_future(rng):
    for _ in range(50):
        a = rng.randint(8, 15)
        b = rng.randint(5, 12)
        c = rng.randint(a + b + 2, a + b + 20)
        if (c - a - b) % 2 == 0:
            break
    now = a + b
    gap = c - now
    ans = gap // 2
    t = rng.randrange(4)
    ins = [
        f"哥哥今年{a}岁，弟弟今年{b}岁，几年后两人的年龄和是{c}岁？",
        f"小明{a}岁、小红{b}岁，几年后两人的年龄和为{c}岁？",
        f"姐姐{a}岁、妹妹{b}岁，再过几年两人年龄和是{c}岁？",
        f"哥哥{a}岁，弟弟{b}岁，当两人年龄和是{c}岁时，是几年后？",
    ][t]
    lines = [
        f"{a} + {b} = {now}岁",
        f"{c} - {now} = {gap}岁",
        f"{gap} ÷ 2 = {ans}年",
    ]
    return ins, lines, ans


_reg("age_brothers_sum_future", age_brothers_sum_future)


# 70. 妈妈年龄等于兄妹和
def mother_kids_catchup(rng):
    b = rng.randint(6, 12)
    c = rng.randint(4, 10)
    a = rng.randint(b + c + 5, b + c + 15)
    kids = b + c
    ans = a - kids
    t = rng.randrange(4)
    ins = [
        f"妈妈今年{a}岁，哥哥{b}岁、妹妹{c}岁，几年后妈妈的年龄正好等于兄妹俩的年龄和？",
        f"爸爸{a}岁，小明{b}岁、小红{c}岁，几年后爸爸的年龄等于两个孩子的年龄和？",
        f"老师{a}岁，两个学生分别{b}岁和{c}岁，几年后老师的年龄等于两个学生年龄和？",
        f"奶奶{a}岁，孙子{b}岁、孙女{c}岁，几年后奶奶的年龄等于孙辈两人年龄和？",
    ][t]
    lines = [
        f"{b} + {c} = {kids}岁",
        f"{a} - {kids} = {ans}年",
    ]
    return ins, lines, ans


_reg("mother_kids_catchup", mother_kids_catchup)


# 71. 数字和与差求两位数
def two_digit_sum_digits(rng):
    for _ in range(50):
        a = rng.randint(5, 17)
        b = rng.randint(1, a - 1)
        if (a + b) % 2 == 0 and (a + b) // 2 <= 9 and (a - b) // 2 >= 1:
            break
    big = (a + b) // 2
    small = a - big
    ans = 10 * big + small
    t = rng.randrange(6)
    ins = [
        f"一个两位数，十位数字与个位数字的和是{a}，差是{b}，这个数是多少？",
        f"一个两位数，两个数字之和是{a}，之差是{b}，这个两位数是多少？",
        f"有一个两位数，十位数字加个位数字得{a}，十位数字减个位数字得{b}，这个数是多少？",
        f"一个两位数，数字和为{a}，数字差为{b}，求这个两位数。",
        f"一个两位数，十位数字与个位数字相加得{a}，相减得{b}，这个两位数是多少？",
        f"有一个两位数，两个数字的和是{a}、差是{b}，这个数是多少？",
    ][t]
    lines = [
        f"十位数字 = ({a} + {b}) ÷ 2 = {big}",
        f"个位数字 = {a} - {big} = {small}",
        f"这个数 = 10 × {big} + {small} = {ans}",
    ]
    return ins, lines, ans


_reg("two_digit_sum_digits", two_digit_sum_digits)


# 72. 每千米油费
def car_oil_cost_per_km(rng):
    for _ in range(50):
        b = rng.randint(2, 10)
        c = rng.randint(5, 12)
        cost = b * c
        divs = [d for d in range(1, 6) if cost % d == 0]
        if divs:
            ans = rng.choice(divs)
            a = cost // ans
            break
    t = rng.randrange(4)
    ins = [
        f"一辆汽车行{a}千米耗油{b}升，每升汽油{c}元，平均每千米油费多少元？",
        f"小汽车行驶{a}千米用了{b}升汽油，汽油每升{c}元，每千米的油费是多少元？",
        f"一辆货车行{a}千米耗油{b}升，油价每升{c}元，平均每千米耗油费多少元？",
        f"汽车跑{a}千米用{b}升油，每升油{c}元，每千米的油费是多少元？",
    ][t]
    lines = [
        f"{b} × {c} = {cost}元",
        f"{cost} ÷ {a} = {ans}元",
    ]
    return ins, lines, ans


_reg("car_oil_cost_per_km", car_oil_cost_per_km)


# 73. 双面打印用纸
def printing_double_sided(rng):
    for _ in range(50):
        a = rng.randint(10, 60)
        b = rng.randint(2, 8)
        if (a * b) % 2 == 0:
            break
    pages = a * b
    ans = pages // 2
    t = rng.randrange(4)
    ins = [
        f"一份{a}页的文件要印{b}份，双面打印，一共需要多少张纸？",
        f"学校打印{a}页的资料{b}份，双面印刷，共需多少张纸？",
        f"一本{a}页的手册印{b}本，双面打印，需要多少张纸？",
        f"打印一份{a}页的稿件{b}份，正反面都印，一共用多少张纸？",
    ][t]
    lines = [
        f"{a} × {b} = {pages}页",
        f"{pages} ÷ 2 = {ans}张",
    ]
    return ins, lines, ans


_reg("printing_double_sided", printing_double_sided)


# 74. 三筐求第三筐
def fruit_baskets_third(rng):
    b = rng.randint(10, 40)
    c = rng.randint(10, 40)
    a = b + c + rng.randint(5, 30)
    two = b + c
    ans = a - two
    f1 = rng.choice(FRUITS)
    t = rng.randrange(4)
    ins = [
        f"三筐{f1}共{a}千克，第一筐{b}千克，第二筐{c}千克，第三筐多少千克？",
        f"水果店运来三筐{f1}，一共{a}千克，第一筐{b}千克、第二筐{c}千克，第三筐多少千克？",
        f"三筐{f1}重{a}千克，第一筐重{b}千克，第二筐重{c}千克，第三筐重多少千克？",
        f"三筐{f1}共{a}千克，已知前两筐分别重{b}千克和{c}千克，第三筐重多少千克？",
    ][t]
    lines = [
        f"{b} + {c} = {two}千克",
        f"{a} - {two} = {ans}千克",
    ]
    return ins, lines, ans


_reg("fruit_baskets_third", fruit_baskets_third)


# 75. 剩下的2分钟走完
def walk_rest_speed(rng):
    for _ in range(50):
        a = rng.randint(200, 800)
        b = rng.randint(50, a - 100)
        if (a - b) % 2 == 0:
            break
    left = a - b
    ans = left // 2
    n1 = rng.choice(NAMES)
    t = rng.randrange(4)
    ins = [
        f"{n1}家到学校共{a}米，他走了{b}米后，剩下的路2分钟走完，平均每分钟走多少米？",
        f"一条路长{a}米，小红走了{b}米，余下的2分钟走完，每分钟走多少米？",
        f"小明跑{a}米，已经跑了{b}米，剩下的要在2分钟内跑完，每分钟跑多少米？",
        f"从家到图书馆{a}米，小华走了{b}米，剩下的2分钟走到，每分钟走多少米？",
    ][t]
    lines = [
        f"{a} - {b} = {left}米",
        f"{left} ÷ 2 = {ans}米",
    ]
    return ins, lines, ans


_reg("walk_rest_speed", walk_rest_speed)


# 76. 三种动物腿数
def legs_three_animals(rng):
    a = rng.randint(3, 12)
    b = rng.randint(3, 12)
    c = rng.randint(2, 8)
    legs2 = (a + b) * 2
    legs_c = c * 4
    ans = legs2 + legs_c
    birds_pool = ["小鸡", "小鸭", "小鸟", "鸽子", "麻雀", "燕子"]
    four = ["兔子", "小羊", "猴子", "熊猫", "小猫", "小狗"]
    a1, a2 = rng.sample(birds_pool, 2)
    a3 = rng.choice(four)
    t = rng.randrange(4)
    ins = [
        f"养殖场有{a}只{a1}、{b}只{a2}和{c}只{a3}，一共有多少条腿？",
        f"动物园里{a1}有{a}只，{a2}有{b}只，{a3}有{c}只，这些动物共有多少条腿？",
        f"农场养了{a}只{a1}、{b}只{a2}、{c}只{a3}，一共有多少条腿？",
        f"院子里有{a}只{a1}、{b}只{a2}和{c}只{a3}，数腿一共多少条？",
    ][t]
    lines = [
        f"({a} + {b}) × 2 = {legs2}条",
        f"{c} × 4 = {legs_c}条",
        f"{legs2} + {legs_c} = {ans}条",
    ]
    return ins, lines, ans


_reg("legs_three_animals", legs_three_animals)


# 77. 分类求连环画
def classify_books(rng):
    for _ in range(50):
        b = rng.randint(10, 40)
        c = rng.randint(2, 4)
        a = rng.randint(b + b * c + 5, b + b * c + 40)
        break
    keji = b * c
    both = b + keji
    ans = a - both
    t = rng.randrange(4)
    ins = [
        f"书架上有{a}本书，其中故事书{b}本，科技书是故事书的{c}倍，其余是连环画，连环画有多少本？",
        f"图书角共{a}本书，故事书{b}本，科技书是故事书的{c}倍，剩下的是连环画，连环画多少本？",
        f"图书馆买来{a}本书，故事书{b}本，科技书是故事书的{c}倍，其余是连环画，连环画有多少本？",
        f"书架上共{a}本书，故事书{b}本，科技书的本数是故事书的{c}倍，其余是连环画，连环画多少本？",
    ][t]
    lines = [
        f"{b} × {c} = {keji}本",
        f"{b} + {keji} = {both}本",
        f"{a} - {both} = {ans}本",
    ]
    return ins, lines, ans


_reg("classify_books", classify_books)


# 78. 乙是甲的几倍多几，求总和
def two_ropes_total(rng):
    a = rng.randint(10, 30)
    b = rng.randint(2, 4)
    c = rng.randint(3, 15)
    times = a * b
    yi = times + c
    ans = a + yi
    t = rng.randrange(4)
    ins = [
        f"甲绳长{a}米，乙绳比甲绳的{b}倍还多{c}米，两根绳子一共长多少米？",
        f"第一根绳子长{a}米，第二根比第一根的{b}倍多{c}米，两根共长多少米？",
        f"甲修路队修{a}米，乙队修的比甲队的{b}倍多{c}米，两队共修多少米？",
        f"小明折{a}只纸鹤，小红折的比小明的{b}倍多{c}只，两人共折多少只？",
    ][t]
    lines = [
        f"{a} × {b} = {times}米",
        f"{times} + {c} = {yi}米",
        f"{a} + {yi} = {ans}米",
    ]
    return ins, lines, ans


_reg("two_ropes_total", two_ropes_total)


# 79. 乙比甲多，合作总量
def workers_compare_coop(rng):
    a = rng.randint(10, 30)
    b = rng.randint(3, 15)
    c = rng.randint(3, 8)
    yi = a + b
    both = a + yi
    ans = both * c
    t = rng.randrange(4)
    ins = [
        f"甲每天做{a}个零件，乙每天比甲多做{b}个，两人合作{c}天共做多少个？",
        f"小明每天折{a}只纸鹤，小红每天比小明多折{b}只，两人合作{c}天共折多少只？",
        f"甲队每天修{a}米，乙队每天比甲队多修{b}米，两队合修{c}天共修多少米？",
        f"师傅每天做{a}个，徒弟每天比师傅多做{b}个，师徒合作{c}天共做多少个？",
    ][t]
    lines = [
        f"{a} + {b} = {yi}个",
        f"{a} + {yi} = {both}个",
        f"{both} × {c} = {ans}个",
    ]
    return ins, lines, ans


_reg("workers_compare_coop", workers_compare_coop)


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
    print(f"L2 ext3 selfcheck OK: {len(PROGRAMS)} programs, {ok} instances verified")
