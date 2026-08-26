#!/usr/bin/env python3
"""L1 (extended) programs: single arithmetic operation, 1-2 steps. 40 families.

Distinct from math_programs_l1.py (buy_one / sum_two / ...). Each fn(rng)
returns (instruction, lines[list[str]], ans:int|Fraction). Three phrasings per
family, shared pools from mathcommon, L1 clean ranges, division always exact.
Every line `X op Y = Z`; the last line's value must equal num(ans).
"""

import random
from fractions import Fraction
from mathcommon import (
    ANIMALS, FOOD, GOODS, NAMES, STATIONERY, UNIT_N, UNIT_ZHI,
    num,
)

PROGRAMS = []


def _reg(name, fn):
    PROGRAMS.append(("L1", name, fn))


# 1. pear_per_box: x ÷ boxes = each -----------------------------------------
def pear_per_box(rng):
    per = rng.randint(2, 9)
    boxes = rng.randint(2, 6)
    total = per * boxes
    obj = rng.choice(["苹果", "梨", "橘子", "桃"])
    t = rng.randrange(3)
    ins = [
        f"把{total}个{obj}平均装进{boxes}个盒子，每个盒子放几个？",
        f"{total}个{obj}，平均分到{boxes}个盘子里，每盘几个？",
        f"有{total}个{obj}，要平均分给{boxes}个同学，每人几个？",
    ][t]
    lines = [f"每盒 = {total} ÷ {boxes} = {per}个"]
    return ins, lines, per


_reg("pear_per_box", pear_per_box)


# 2. rows_seats: rows × seats = total ---------------------------------------
def rows_seats(rng):
    rows = rng.randint(2, 9)
    seats = rng.randint(3, 15)
    t = rng.randrange(3)
    ins = [
        f"礼堂有{rows}排座位，每排{seats}座，一共多少个座位？",
        f"每排摆{seats}张椅子，摆满{rows}排，共有多少张椅子？",
        f"电影院{rows}排，每排{seats}个座位，座位总数是多少？",
    ][t]
    lines = [f"座位总数 = {rows} × {seats} = {rows * seats}个"]
    return ins, lines, rows * seats


_reg("rows_seats", rows_seats)


# 3. stamp_pages: pages × stamps = total ------------------------------------
def stamp_pages(rng):
    pages = rng.randint(3, 12)
    per = rng.randint(3, 10)
    obj = rng.choice(["邮票", "贴纸", "书签"])
    t = rng.randrange(3)
    ins = [
        f"集邮册每页能贴{per}张{obj}，贴满{pages}页，一共多少张？",
        f"{pages}页，每页{per}张{obj}，总共能放多少张？",
        f"每页{per}张{obj}，共{pages}页，{obj}一共有多少张？",
    ][t]
    lines = [f"{obj}总数 = {per} × {pages} = {per * pages}张"]
    return ins, lines, per * pages


_reg("stamp_pages", stamp_pages)


# 4. cut_rope: length ÷ piece = number --------------------------------------
def cut_rope(rng):
    piece = rng.randint(2, 6)
    n = rng.randint(3, 10)
    length = piece * n
    obj = rng.choice(["绳子", "绸带", "铁丝"])
    t = rng.randrange(3)
    ins = [
        f"一根{length}米的{obj}，剪成每段{piece}米，能剪几段？",
        f"把{length}米长的{obj}平均分成{piece}米一段，共分几段？",
        f"{obj}长{length}米，每{piece}米剪一刀，一共几段？",
    ][t]
    lines = [f"段数 = {length} ÷ {piece} = {n}段"]
    return ins, lines, n


_reg("cut_rope", cut_rope)


# 5. tricycle_wheels: n × 3 = wheels ----------------------------------------
def tricycle_wheels(rng):
    n = rng.randint(2, 12)
    obj = rng.choice(["三轮车", "三轮摩托"])
    t = rng.randrange(3)
    ins = [
        f"操场边停了{n}辆{obj}，一共多少个轮子？",
        f"每辆{obj}有3个轮子，{n}辆共有几个轮子？",
        f"停车场有{n}辆{obj}，轮子一共有多少个？",
    ][t]
    lines = [f"轮子总数 = {n} × 3 = {n * 3}个"]
    return ins, lines, n * 3


_reg("tricycle_wheels", tricycle_wheels)


# 6. stair_climb: floors × steps = total ------------------------------------
def stair_climb(rng):
    steps = rng.randint(5, 15)
    floors = rng.randint(2, 6)
    t = rng.randrange(3)
    ins = [
        f"每层楼有{steps}级台阶，走完{floors}层要爬多少级台阶？",
        f"{floors}层楼，每层{steps}级，一共多少级台阶？",
        f"上到第{floors}层，每层{steps}级台阶，共爬了几级？",
    ][t]
    lines = [f"台阶总数 = {floors} × {steps} = {floors * steps}级"]
    return ins, lines, floors * steps


_reg("stair_climb", stair_climb)


# 7. candy_days: total ÷ per day = days -------------------------------------
def candy_days(rng):
    per = rng.randint(2, 6)
    days = rng.randint(2, 8)
    total = per * days
    t = rng.randrange(3)
    ins = [
        f"一袋里有{total}颗糖，每天吃{per}颗，可以吃几天？",
        f"{total}颗糖果，每天吃{per}颗，够吃多少天？",
        f"一共有{total}颗糖，计划每天吃{per}颗，能吃几天？",
    ][t]
    lines = [f"天数 = {total} ÷ {per} = {days}天"]
    return ins, lines, days


_reg("candy_days", candy_days)


# 8. triple_deposit: 3 × v = total ------------------------------------------
def triple_deposit(rng):
    v = rng.randint(5, 60)
    n = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n}每月存{v}元，存够3个月一共存了多少元？",
        f"一个月存{v}元，3个月共存多少元？",
        f"{n}连续存了3个月，每月{v}元，一共多少元？",
    ][t]
    lines = [f"一共 = {v} × 3 = {v * 3}元"]
    return ins, lines, v * 3


_reg("triple_deposit", triple_deposit)


# 9. lap_distance: laps × per-lap = total -----------------------------------
def lap_distance(rng):
    per = rng.randint(20, 60)
    laps = rng.randint(2, 4)
    n = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"操场一圈{per}米，{n}跑了{laps}圈，一共多少米？",
        f"每跑一圈是{per}米，跑了{laps}圈，共跑多少米？",
        f"操场一圈{per}米，绕操场{laps}圈，路程是多少米？",
    ][t]
    lines = [f"路程 = {per} × {laps} = {per * laps}米"]
    return ins, lines, per * laps


_reg("lap_distance", lap_distance)


# 10. water_bottles: boxes × per box = total ---------------------------------
def water_bottles(rng):
    per = rng.randint(6, 12)
    boxes = rng.randint(2, 6)
    obj = rng.choice(["矿泉水", "酸奶", "牛奶"])
    t = rng.randrange(3)
    ins = [
        f"每箱{obj}有{per}瓶，搬了{boxes}箱，一共有多少瓶？",
        f"{boxes}箱{obj}，每箱{per}瓶，总共几瓶？",
        f"每箱装{per}瓶，{boxes}箱共装多少瓶？",
    ][t]
    lines = [f"总瓶数 = {per} × {boxes} = {per * boxes}瓶"]
    return ins, lines, per * boxes


_reg("water_bottles", water_bottles)


# 11. age_gap: older - younger = difference ----------------------------------
def age_gap(rng):
    younger = rng.randint(5, 15)
    gap = rng.randint(2, 20)
    older = younger + gap
    n1, n2 = rng.sample(NAMES, 2)
    t = rng.randrange(3)
    ins = [
        f"{n1}今年{older}岁，{n2}今年{younger}岁，两人相差几岁？",
        f"哥哥{older}岁，弟弟{younger}岁，哥哥比弟弟大几岁？",
        f"{n1}{older}岁，{n2}{younger}岁，年龄差是多少岁？",
    ][t]
    lines = [f"相差 = {older} - {younger} = {gap}岁"]
    return ins, lines, gap


_reg("age_gap", age_gap)


# 12. exceed_by: base + more = second ----------------------------------------
def exceed_by(rng):
    base = rng.randint(5, 50)
    more = rng.randint(2, 20)
    total = base + more
    n1, n2 = rng.sample(NAMES, 2)
    obj = rng.choice(["颗糖", "本书", "个球", "张贴纸"])
    t = rng.randrange(3)
    ins = [
        f"{n1}有{base}{obj}，{n2}比{n1}多{more}{obj}，{n2}有多少{obj}？",
        f"甲有{base}个，乙比甲多{more}个，乙有几个？",
        f"{n1}捡了{base}{obj}，比{obj}多{more}{obj}的{n2}有几个？",
    ][t]
    lines = [f"{'乙有' if t == 1 else n2 + '有'} = {base} + {more} = {total}{'个' if t == 1 else obj[0]}"]
    return ins, lines, total


_reg("exceed_by", exceed_by)


# 13. short_by: base - less = second -----------------------------------------
def short_by(rng):
    base = rng.randint(10, 60)
    less = rng.randint(2, min(25, base - 1))
    result = base - less
    n1, n2 = rng.sample(NAMES, 2)
    obj = rng.choice(["个", "本", "张", "枝"])
    t = rng.randrange(3)
    ins = [
        f"{n1}有{base}{obj}，{n2}比{n1}少{less}{obj}，{n2}有几个？",
        f"甲做了{base}道题，乙比甲少做{less}道，乙做了几道？",
        f"{n1}有{base}{obj}，比{n2}多{less}{obj}，{n2}有多少{obj}？",
    ][t]
    lines = [f"{'乙做了' if t == 1 else n2 + '有'} = {base} - {less} = {result}{'道' if t == 1 else obj}"]
    return ins, lines, result


_reg("short_by", short_by)


# 14. sell_left: stock - sold = left ------------------------------------------
def sell_left(rng):
    stock = rng.randint(20, 90)
    sold = rng.randint(3, stock - 3)
    left = stock - sold
    n = rng.choice(NAMES)
    obj = rng.choice(["个苹果", "条鱼", "本本子", "瓶汽水"])
    t = rng.randrange(3)
    ins = [
        f"店里有{stock}{obj}，卖掉了{sold}{obj}，还剩多少？",
        f"{n}进了{stock}{obj}，上午卖出{sold}{obj}，还有多少？",
        f"原来有{stock}{obj}，卖出{sold}{obj}后剩几个？",
    ][t]
    lines = [f"剩下 = {stock} - {sold} = {left}{obj[0]}"]
    return ins, lines, left


_reg("sell_left", sell_left)


# 15. ticket_total: per ticket × people = total ------------------------------
def ticket_total(rng):
    per = rng.choice([2, 3, 5])
    people = rng.randint(2, 8)
    t = rng.randrange(3)
    ins = [
        f"每人门票{per}元，{people}人一共要花多少元？",
        f"门票每张{per}元，一家{people}口人买票共多少元？",
        f"{people}个人去玩，每人{per}元，共需多少元？",
    ][t]
    lines = [f"一共 = {people} × {per} = {people * per}元"]
    return ins, lines, people * per


_reg("ticket_total", ticket_total)


# 16. orchard_rows: rows × per row = total -----------------------------------
def orchard_rows(rng):
    rows = rng.randint(3, 8)
    per = rng.randint(4, 12)
    tree = rng.choice(["苹果树", "桃树", "梨树"])
    t = rng.randrange(3)
    ins = [
        f"果园有{rows}行{tree}，每行{per}棵，一共有多少棵？",
        f"每行种{per}棵，种了{rows}行，共多少棵树？",
        f"{rows}行{tree}，每行{per}棵，总共几棵？",
    ][t]
    lines = [f"总棵数 = {rows} × {per} = {rows * per}棵"]
    return ins, lines, rows * per


_reg("orchard_rows", orchard_rows)


# 17. bookshelf_total: shelves × per shelf = total ---------------------------
def bookshelf_total(rng):
    shelves = rng.randint(3, 8)
    per = rng.randint(4, 12)
    t = rng.randrange(3)
    ins = [
        f"书柜有{shelves}层，每层放{per}本书，一共能放多少本？",
        f"每层放{per}本书，摆满{shelves}层，共多少本？",
        f"{shelves}层书架，每层{per}本，能放多少本书？",
    ][t]
    lines = [f"总本数 = {shelves} × {per} = {shelves * per}本"]
    return ins, lines, shelves * per


_reg("bookshelf_total", bookshelf_total)


# 18. hours_seconds: hours × 3600 = seconds (small hours) --------------------
def hours_seconds(rng):
    hours = rng.randint(1, 3)
    t = rng.randrange(3)
    ins = [
        f"{hours}小时等于多少秒？",
        f"1小时有3600秒，{hours}小时是多少秒？",
        f"{hours}个小时一共有多少秒？",
    ][t]
    lines = [f"一共 = {hours} × 3600 = {hours * 3600}秒"]
    return ins, lines, hours * 3600


_reg("hours_seconds", hours_seconds)


# 19. score_award: per item × count = total ----------------------------------
def score_award(rng):
    per = rng.randint(2, 5)
    count = rng.randint(4, 10)
    t = rng.randrange(3)
    ins = [
        f"答对一题得{per}分，答对{count}题一共得多少分？",
        f"每答对1题加{per}分，答对{count}题共得几分？",
        f"一道题{per}分，{count}道题全对得多少分？",
    ][t]
    lines = [f"一共 = {per} × {count} = {per * count}分"]
    return ins, lines, per * count


_reg("score_award", score_award)


# 20. dose_days: total ÷ per day = days ---------------------------------------
def dose_days(rng):
    per = rng.randint(2, 4)
    days = rng.randint(3, 8)
    total = per * days
    t = rng.randrange(3)
    ins = [
        f"一盒有{total}片药，每天吃{per}片，能吃几天？",
        f"{total}片药，每天吃{per}片，一共可以吃多少天？",
        f"药共{total}片，每天{per}片，能吃几天？",
    ][t]
    lines = [f"天数 = {total} ÷ {per} = {days}天"]
    return ins, lines, days


_reg("dose_days", dose_days)


# 21. shirt_buttons: buttons × shirts = total ---------------------------------
def shirt_buttons(rng):
    per = rng.randint(3, 6)
    shirts = rng.randint(3, 10)
    obj = rng.choice(["衬衫", "外套", "毛衣"])
    t = rng.randrange(3)
    ins = [
        f"每件{obj}有{per}颗扣子，{shirts}件一共几颗扣子？",
        f"{shirts}件{obj}，每件{per}颗纽扣，共多少颗？",
        f"一件衣服缝{per}颗扣子，{shirts}件共多少颗？",
    ][t]
    lines = [f"扣子总数 = {per} × {shirts} = {per * shirts}颗"]
    return ins, lines, per * shirts


_reg("shirt_buttons", shirt_buttons)


# 22. square_perimeter: 4 × side = perimeter ---------------------------------
def square_perimeter(rng):
    side = rng.randint(5, 30)
    obj = rng.choice(["方形花坛", "正方形贺卡"])
    t = rng.randrange(3)
    ins = [
        f"{obj}边长{side}厘米，它的周长是多少厘米？",
        f"正方形边长{side}厘米，一周长是多少？",
        f"一个正方形花坛边长{side}米，四周的长度是多少米？",
    ][t]
    lines = [f"周长 = {side} × 4 = {side * 4}厘米"]
    return ins, lines, side * 4


_reg("square_perimeter", square_perimeter)


# 23. books_per_day: total ÷ days = each day ----------------------------------
def books_per_day(rng):
    days = rng.randint(3, 7)
    per = rng.randint(3, 10)
    total = per * days
    t = rng.randrange(3)
    ins = [
        f"{total}页故事书，计划{days}天看完，平均每天看多少页？",
        f"{days}天读完{total}页书，每天读几页？",
        f"共{total}页，{days}天看完，每天要看多少页？",
    ][t]
    lines = [f"每天看 = {total} ÷ {days} = {per}页"]
    return ins, lines, per


_reg("books_per_day", books_per_day)


# 24. divide_money: total ÷ people = each -------------------------------------
def divide_money(rng):
    people = rng.randint(2, 6)
    per = rng.randint(2, 8)
    total = people * per
    t = rng.randrange(3)
    ins = [
        f"{total}元零花钱平均分给{people}个小朋友，每人得多少元？",
        f"把{total}元平均分成{people}份，每份多少元？",
        f"{total}元，{people}人平分，每人几元？",
    ][t]
    lines = [f"每人 = {total} ÷ {people} = {per}元"]
    return ins, lines, per


_reg("divide_money", divide_money)


# 25. rows_count_students: total ÷ per row = rows -----------------------------
def rows_count_students(rng):
    per = rng.randint(4, 12)
    rows = rng.randint(2, 6)
    total = per * rows
    kids = rng.choice(["小朋友", "同学"])
    t = rng.randrange(3)
    ins = [
        f"{total}名{kids}排成每排{per}人的队伍，能排几排？",
        f"每排坐{per}人，{total}人一共坐满几排？",
        f"{total}个{kids}，每{per}人一排，有几排？",
    ][t]
    lines = [f"排数 = {total} ÷ {per} = {rows}排"]
    return ins, lines, rows


_reg("rows_count_students", rows_count_students)


# 26. pencils_per_box: total ÷ boxes = each -----------------------------------
def pencils_per_box(rng):
    per = rng.randint(3, 8)
    boxes = rng.randint(2, 5)
    total = per * boxes
    n = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"把{total}支铅笔平均放进{boxes}个笔盒，每盒几支？",
        f"{total}支铅笔平均分到{boxes}个盒子，每盒多少支？",
        f"{total}支铅笔，平均装进{boxes}盒，每盒几支？",
    ][t]
    lines = [f"每盒 = {total} ÷ {boxes} = {per}支"]
    return ins, lines, per


_reg("pencils_per_box", pencils_per_box)


# 27. fishper_aquarium: tanks × per tank = total ------------------------------
def fishper_aquarium(rng):
    tanks = rng.randint(2, 6)
    per = rng.randint(3, 8)
    obj = rng.choice(["小鱼", "金鱼"])
    t = rng.randrange(3)
    ins = [
        f"有{tanks}个鱼缸，每个鱼缸养{per}条{obj}，一共多少条？",
        f"每缸{per}条，{tanks}缸共多少条？",
        f"{tanks}个缸，每缸{per}条鱼，总共有几条？",
    ][t]
    lines = [f"总条数 = {tanks} × {per} = {tanks * per}条"]
    return ins, lines, tanks * per


_reg("fishper_aquarium", fishper_aquarium)


# 28. minutes_to_hours reads backwards: minutes ÷ 60 = hours ------------------
def minutes_to_hours(rng):
    hours = rng.randint(1, 4)
    minutes = hours * 60
    t = rng.randrange(3)
    ins = [
        f"{minutes}分钟等于多少小时？",
        f"1小时有60分钟，{minutes}分钟是几小时？",
        f"{minutes}分钟能换算成多少小时？",
    ][t]
    lines = [f"小时数 = {minutes} ÷ 60 = {hours}小时"]
    return ins, lines, hours


_reg("minutes_to_hours", minutes_to_hours)


# 29. book_covers: per stack × stacks = total ---------------------------------
def book_covers(rng):
    stacks = rng.randint(2, 5)
    per = rng.randint(10, 30)
    total = stacks * per
    t = rng.randrange(3)
    ins = [
        f"每摞课本有{per}本，摆了{stacks}摞，一共有多少本？",
        f"{stacks}摞书，每摞{per}本，共多少本？",
        f"每叠{per}本，共{stacks}叠，课本总数是多少？",
    ][t]
    lines = [f"总本数 = {stacks} × {per} = {total}本"]
    return ins, lines, total


_reg("book_covers", book_covers)


# 30. money_left_spent: had - spent = left ------------------------------------
def money_left_spent(rng):
    had = rng.randint(15, 90)
    spent = rng.randint(3, had - 3)
    left = had - spent
    n = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n}有{had}元，买零食用了{spent}元，还剩多少元？",
        f"妈妈给{n}{had}元，花了{spent}元，剩几元？",
        f"{n}带{had}元出门，花掉{spent}元，还剩多少？",
    ][t]
    lines = [f"剩下 = {had} - {spent} = {left}元"]
    return ins, lines, left


_reg("money_left_spent", money_left_spent)


# 31. photo_album: pages × per page = total -----------------------------------
def photo_album(rng):
    pages = rng.randint(4, 10)
    per = rng.randint(3, 8)
    obj = rng.choice(["照片", "卡片", "明信片"])
    t = rng.randrange(3)
    ins = [
        f"相册每页放{per}张{obj}，放满{pages}页，一共有多少张？",
        f"{pages}页，每页{per}张，共放多少张？",
        f"每页贴{per}张，共{pages}页，能贴多少张？",
    ][t]
    lines = [f"{obj}总数 = {pages} × {per} = {pages * per}张"]
    return ins, lines, pages * per


_reg("photo_album", photo_album)


# 32. step_count_blocks: steps × per step = total -----------------------------
def step_count_blocks(rng):
    per = rng.randint(2, 5)
    steps = rng.randint(5, 20)
    n = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n}每走一步跨{per}格，走了{steps}步，共走多少格？",
        f"一步跳{per}格，跳{steps}步是多少格？",
        f"每次跳{per}格，连跳{steps}次，一共几格？",
    ][t]
    lines = [f"总格数 = {per} × {steps} = {per * steps}格"]
    return ins, lines, per * steps


_reg("step_count_blocks", step_count_blocks)


# 33. share_friends: total ÷ friends = each -----------------------------------
def share_friends(rng):
    friends = rng.randint(2, 6)
    per = rng.randint(2, 8)
    total = friends * per
    n = rng.choice(NAMES)
    t = rng.randrange(3)
    ins = [
        f"{n}有{total}颗弹珠，平均分给{friends}个朋友，每人几颗？",
        f"{total}颗珠子，{friends}人平分，每人得几颗？",
        f"一共{total}颗糖，分给{friends}个小伙伴，每人几颗？",
    ][t]
    lines = [f"每人 = {total} ÷ {friends} = {per}颗"]
    return ins, lines, per


_reg("share_friends", share_friends)


# 34. desks_rows: per row × rows = total --------------------------------------
def desks_rows(rng):
    per = rng.randint(3, 6)
    rows = rng.randint(5, 12)
    total = per * rows
    t = rng.randrange(3)
    ins = [
        f"每排课桌{per}张，摆了{rows}排，教室一共有多少张课桌？",
        f"{rows}排凳子，每排{per}张，共有多少张？",
        f"每行{per}张桌子，共{rows}行，桌子总数是多少？",
    ][t]
    lines = [f"课桌总数 = {per} × {rows} = {total}张"]
    return ins, lines, total


_reg("desks_rows", desks_rows)


# 35. candies_jars: jars × per jar = total ------------------------------------
def candies_jars(rng):
    jars = rng.randint(2, 6)
    per = rng.randint(5, 12)
    total = jars * per
    t = rng.randrange(3)
    ins = [
        f"有{jars}个罐子，每个罐子装{per}颗糖，一共多少颗？",
        f"每罐{per}颗，{jars}罐共多少颗？",
        f"{jars}个瓶，每瓶{per}颗糖，总共有几颗？",
    ][t]
    lines = [f"糖总数 = {jars} × {per} = {total}颗"]
    return ins, lines, total


_reg("candies_jars", candies_jars)


# 36. leftover_from: pool - taken = left --------------------------------------
def leftover_from(rng):
    had = rng.randint(20, 80)
    taken = rng.randint(3, had - 3)
    left = had - taken
    n = rng.choice(NAMES)
    obj = rng.choice(["个大饼", "条绳子", "根黄瓜", "个排球"])
    t = rng.randrange(3)
    ins = [
        f"仓库有{had}{obj}，取走{taken}{obj}，还剩多少？",
        f"原来有{had}{obj}，借出{taken}{obj}，剩下几个？",
        f"{n}拿来{had}{obj}，用掉{taken}个，还剩多少？",
    ][t]
    lines = [f"剩下 = {had} - {taken} = {left}{obj[0]}"]
    return ins, lines, left


_reg("leftover_from", leftover_from)


# 37. dinner_plates: people × plates = total ----------------------------------
def dinner_plates(rng):
    people = rng.randint(2, 8)
    per = rng.randint(2, 4)
    total = people * per
    obj = rng.choice(["筷子", "碗", "盘子"])
    t = rng.randrange(3)
    ins = [
        f"每个人摆{per}双{obj}，{people}个人一共摆多少双？",
        f"每人用{per}个{obj}，{people}人共用几个？",
        f"一家{people}口人，每人{per}双筷子，共几双？",
    ][t]
    lines = [f"一共 = {people} × {per} = {total}{'个' if t == 1 else '双'}"]
    return ins, lines, total


_reg("dinner_plates", dinner_plates)


# 38. divide_teams: total ÷ per team = teams ----------------------------------
def divide_teams(rng):
    per = rng.randint(3, 8)
    teams = rng.randint(2, 6)
    total = per * teams
    kids = rng.choice(["小朋友", "队员"])
    t = rng.randrange(3)
    ins = [
        f"{total}名{kids}平均分成每队{per}人，能分几队？",
        f"每队{per}人，{total}人一共是几队？",
        f"{total}个同学，每{per}人一队，能排几队？",
    ][t]
    lines = [f"队数 = {total} ÷ {per} = {teams}队"]
    return ins, lines, teams


_reg("divide_teams", divide_teams)


# 39. window_panes: windows × per window = total ------------------------------
def window_panes(rng):
    windows = rng.randint(2, 6)
    per = rng.randint(3, 8)
    total = windows * per
    t = rng.randrange(3)
    ins = [
        f"教学楼有{windows}扇窗，每扇窗{per}块玻璃，一共多少块玻璃？",
        f"每扇窗户{per}块玻璃，{windows}扇共多少块？",
        f"{windows}扇窗，每扇{per}格玻璃，总共有几格？",
    ][t]
    lines = [f"玻璃总数 = {windows} × {per} = {total}块"]
    return ins, lines, total


_reg("window_panes", window_panes)


# 40. weekly_sell: per day × days = total -------------------------------------
def weekly_sell(rng):
    per = rng.randint(5, 20)
    days = rng.choice([3, 5, 7])
    total = per * days
    obj = rng.choice(["个面包", "杯奶茶", "份盒饭"])
    measure = obj[0]
    item = obj[1:]
    t = rng.randrange(3)
    ins = [
        f"烘焙店每天卖出{per}{obj}，卖{days}天一共卖出多少？",
        f"每天卖{per}{measure}{item}，{days}天共卖出多少？",
        f"小卖部一天卖{per}{obj}，{days}天共卖多少？",
    ][t]
    lines = [f"一共卖出 = {per} × {days} = {total}{measure}"]
    return ins, lines, total


_reg("weekly_sell", weekly_sell)


if __name__ == "__main__":
    from run_math_short import verify
    rng = random.Random(1)
    bad = 0
    for level, name, fn in PROGRAMS:
        for _ in range(40):
            try:
                ins, lines, ans = fn(rng)
            except Exception as e:
                print(f"{name}: EXC {e!r}")
                bad += 1
                continue
            out, ok = verify(ins, lines, ans)
            if not ok:
                print(f"{name}: FAIL {lines!r} ans={ans}")
                bad += 1
    print(f"selfcheck {'PASSED' if bad == 0 else f'FAILED ({bad} errors)'}")