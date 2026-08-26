#!/usr/bin/env python3
"""Shared helpers + pools for the math program bank + runtime. No policy here."""
import re
from fractions import Fraction

NAMES = ["小明", "小红", "小华", "小刚", "小丽", "小军", "小雅", "小宇", "小雨",
         "小斌", "小丽", "小萌", "小杰", "小睿", "小彤", "小宇", "小梁", "小沛"]
FRUITS = ["苹果", "香蕉", "西瓜", "橙子", "葡萄", "芒果", "桃子", "甜瓜", "橘子", "猕猴桃"]
STATIONERY = ["笔记本", "铅笔", "橡皮", "钢笔", "尺子", "书包", "铅笔盒", "日记本", "蜡笔", "水彩笔"]
FOOD = ["面包", "奶茶", "包子", "饺子", "鸡蛋", "煎饼", "烤肠", "豆浆", "饭团", "三明治"]
GOODS = FRUITS + STATIONERY + FOOD
ANIMALS = ["小鸡", "小猫", "小狗", "小鸭", "兔子", "绵羊", "山羊", "小马", "猴子", "熊猫"]
PLACE = ["学校", "超市", "公园", "果园", "图书馆", "体育馆", "菜市场", "玩具店", "文具店", "花店"]
UNIT_FRUIT = "斤"
UNIT_N = "个"
UNIT_ZHI = "支"


def num(x):
    """Exact string for int/Fraction: integers drop '.0', fractions show n/d."""
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def frac(x):
    return Fraction(x)


def pct(f):
    """Fraction -> clean percent string (exact where integral, else 1 decimal)."""
    v = f.numerator / f.denominator * 100
    return f"{round(v)}%" if abs(v - round(v)) < 1e-9 else f"{v:.1f}%"


def eval_lhs(s):
    """Safely evaluate an equation LHS: numeric literals + () + - * / only."""
    s = (s.replace("×", "*").replace("÷", "/").replace("−", "-")
         .replace("（", "(").replace("）", ")"))
    if not re.fullmatch(r"[0-9()+\-*/.\s]+", s):
        raise ValueError(f"unsafe lhs: {s!r}")
    return float(eval(s))