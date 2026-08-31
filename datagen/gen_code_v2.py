#!/usr/bin/env python3
"""code-500-v2: clean Chinese Python eval, 10 families x 50 problems.

Every family is absent from gen_code.py's 30 templates (the code-500 carve
source) and from the v5 Evol-Instruct addon. Outputs are integer-safe or
explicitly rounded (the scorer is exact line match); every output is
deterministic (sorted, no bare set/dict repr).

Provenance: this file IS the generator. data/eval/code_holdout_v2_500.jsonl
is its only output. The census + verbatim probe (t51) verify cleanliness
before the set is used.

Usage: python datagen/gen_code_v2.py [n_per_family]  (default 500//30+1=17)

restartable: regenerates deterministically from rng = random.Random(2026) in
seconds; output data/eval/code_holdout_v2_500.jsonl is re-creatable at any
time by re-running this script.
"""
import hashlib
import json
import os
import random
import subprocess
import sys

rng = random.Random(2026)

N = lambda lo, hi: rng.randint(lo, hi)  # noqa


def _ints(k, lo=1, hi=99):
    return [N(lo, hi) for _ in range(k)]


def code_for(name):
    if name == "isqrt":
        n = N(100, 9999)
        code = f"import math\nprint(math.isqrt({n}))"
        pr = f"用 math.isqrt 求 {n} 的整数平方根"
    elif name == "comb":
        n, k = N(5, 20), N(2, 5)
        code = f"import math\nprint(math.comb({n}, {k}))"
        pr = f"用 math.comb 求组合数 C({n},{k})"
    elif name == "gcd":
        a, b = N(12, 500), N(12, 500)
        code = f"import math\nprint(math.gcd({a}, {b}))"
        pr = f"用 math.gcd 求 {a} 和 {b} 的最大公约数"
    elif name == "lcm":
        a, b = N(4, 60), N(4, 60)
        code = f"import math\nprint(math.lcm({a}, {b}))"
        pr = f"用 math.lcm 求 {a} 和 {b} 的最小公倍数"
    elif name == "factorial":
        n = N(4, 60)
        code = f"import math\nprint(math.factorial({n}))"
        pr = f"用 math.factorial 求 {n} 的阶乘"
    elif name == "ceil_div":
        a, b = N(10, 200), N(3, 12)
        code = f"import math\nprint(math.ceil({a} / {b}))"
        pr = f"用 math.ceil 求 {a}÷{b} 的上取整结果"
    elif name == "floor_div":
        a, b = N(10, 200), N(3, 12)
        code = f"import math\nprint(math.floor({a} / {b}))"
        pr = f"用 math.floor 求 {a}÷{b} 的下取整结果"
    elif name == "counter_items":
        vals = sorted(_ints(12, 1, 8))
        code = f"from collections import Counter\nc = Counter({vals})\nfor k in sorted(c):\n    print(k, c[k])"
        pr = f"用 Counter 统计列表 {vals} 中每个元素出现的次数，按元素从小到大输出"
    elif name == "counter_most":
        vals = sorted(_ints(15, 1, 6))
        code = f"from collections import Counter\nc = Counter({vals})\nfor v, n in c.most_common(3):\n    print(v, n)"
        pr = f"用 Counter 求列表 {vals} 中出现次数最多的 3 个元素及其次数"
    elif name == "counter_str":
        s = "".join(rng.choice("abcde") for _ in range(15))
        code = f'from collections import Counter\nc = Counter("{s}")\nfor k in sorted(c):\n    print(k, c[k])'
        pr = f'用 Counter 统计字符串 "{s}" 中各字符出现次数，按字母序输出'
    elif name == "set_union":
        a, b = sorted(set(_ints(8, 1, 30))), sorted(set(_ints(8, 1, 30)))
        code = f"a = set({a})\nb = set({b})\nprint(sorted(a | b))"
        pr = f"求集合 {set(a)} 和 {set(b)} 的并集，排序输出"
    elif name == "set_inter":
        a, b = sorted(set(_ints(8, 1, 30))), sorted(set(_ints(8, 1, 30)))
        code = f"a = set({a})\nb = set({b})\nprint(sorted(a & b))"
        pr = f"求集合 {set(a)} 和 {set(b)} 的交集，排序输出"
    elif name == "set_diff":
        a, b = sorted(set(_ints(8, 1, 30))), sorted(set(_ints(8, 1, 30)))
        code = f"a = set({a})\nb = set({b})\nprint(sorted(a - b))"
        pr = f"求集合 {set(a)} 减 {set(b)} 的差集，排序输出"
    elif name == "set_symdiff":
        a, b = sorted(set(_ints(8, 1, 30))), sorted(set(_ints(8, 1, 30)))
        code = f"a = set({a})\nb = set({b})\nprint(sorted(a ^ b))"
        pr = f"求集合 {set(a)} 和 {set(b)} 的对称差集，排序输出"
    elif name == "enum_zip":
        a, b = _ints(5, 1, 50), _ints(5, 1, 50)
        code = (f"a = {a}\nb = {b}\n"
                f"for i, (x, y) in enumerate(zip(a, b)):\n    print(i, x + y)")
        pr = f"用 enumerate 和 zip 同时遍历列表 {a} 和 {b}，输出下标与对应元素之和"
    elif name == "enum_index":
        vals = _ints(6, 1, 99)
        code = f"vals = {vals}\nfor i, v in enumerate(vals):\n    print(i, v)"
        pr = f"用 enumerate 输出列表 {vals} 每个元素的下标和值"
    elif name == "sort_abs":
        vals = _ints(8, -50, 50)
        code = f"vals = {vals}\nprint(sorted(vals, key=abs))"
        pr = f"将列表 {vals} 按绝对值从小到大排序"
    elif name == "sort_len":
        words = ["".join(rng.choice("abcdef") for _ in range(N(2, 7))) for _ in range(6)]
        code = f"words = {words}\nprint(sorted(words, key=len))"
        pr = f"将字符串列表 {words} 按长度从小到大排序"
    elif name == "sort_second":
        pairs = [(N(1, 9), N(1, 9)) for _ in range(5)]
        code = f"pairs = {pairs}\nprint(sorted(pairs, key=lambda x: x[1]))"
        pr = f"将数对列表 {pairs} 按每个数对的第二个元素排序"
    elif name == "sort_digit":
        vals = _ints(8, 10, 99)
        code = f"vals = {vals}\nprint(sorted(vals, key=lambda x: x % 10))"
        pr = f"将列表 {vals} 按个位数字从小到大排序"
    elif name == "str_chain":
        words = [rng.choice(["Hello", "World", "Python", "Code", "Test"]) for _ in range(4)]
        s = "  " + ", ".join(words) + "  "
        old, new = rng.choice(["Hello", "World", "Python"]), rng.choice(["Hi", "OK", "Go"])
        code = (f's = "{s}"\n'
                f'print(s.strip().replace("{old}", "{new}").upper().split(", "))')
        pr = (f'对字符串 "{s}" 依次执行：去首尾空格、把 "{old}" 替换为 "{new}"、'
              f"转大写、按逗号空格分割，输出结果")
    elif name == "str_join":
        words = ["".join(rng.choice("abcdef") for _ in range(N(3, 6))) for _ in range(5)]
        code = f"words = {words}\nprint(\"-\".join(words))"
        pr = f"用连字符 - 把字符串列表 {words} 拼接成一个字符串"
    elif name == "nested_leaves":
        d = {"a": {"x": N(1, 9), "y": N(1, 9)}, "b": {"z": N(1, 9)}}
        code = (f"d = {d}\n"
                f"def leaves(d):\n"
                f"    out = []\n"
                f"    for k in sorted(d):\n"
                f"        v = d[k]\n"
                f"        if isinstance(v, dict):\n"
                f"            out.extend(leaves(v))\n"
                f"        else:\n"
                f"            out.append(v)\n"
                f"    return out\n"
                f"print(leaves(d))")
        pr = f"提取嵌套字典 {d} 中的所有叶子值（非字典的值），按 key 排序遍历，输出值列表"
    elif name == "nested_get":
        d = {"u": {"v": {"w": N(10, 99)}}, "x": {"y": N(10, 99)}}
        path = ["u", "v", "w"]
        code = (f"d = {d}\npath = {path}\n"
                f"cur = d\nfor k in path:\n    cur = cur[k]\nprint(cur)")
        pr = f"按路径 {path} 逐层访问嵌套字典 {d}，输出最终值"
    elif name == "flatten_one":
        nested = [_ints(3, 1, 9) for _ in range(4)]
        code = f"nested = {nested}\nprint([x for sub in nested for x in sub])"
        pr = f"将二维列表 {nested} 展平一层，输出一维列表"
    elif name == "flatten_deep":
        v1, v2, v3, v4, v5 = _ints(5, 1, 9)
        nested = [[v1, [v2, v3]], [[v4], v5]]
        code = (f"nested = {nested}\n"
                f"def flat(d):\n"
                f"    out = []\n"
                f"    for x in d:\n"
                f"        if isinstance(x, list):\n"
                f"            out.extend(flat(x))\n"
                f"        else:\n"
                f"            out.append(x)\n"
                f"    return out\n"
                f"print(flat(nested))")
        pr = f"将嵌套列表 {nested} 完全展平（不限层数），输出一维列表"
    elif name == "running_sum":
        vals = _ints(6, 1, 20)
        code = (f"vals = {vals}\n"
                f"acc = 0\nfor v in vals:\n    acc += v\n    print(acc)")
        pr = f"计算列表 {vals} 的累计和，逐项输出"
    elif name == "running_max":
        vals = _ints(7, 1, 50)
        code = (f"vals = {vals}\n"
                f"m = vals[0]\nfor v in vals:\n    m = max(m, v)\n    print(m)")
        pr = f"逐项输出列表 {vals} 的前缀最大值"
    elif name == "pair_sum":
        vals = sorted(_ints(8, 1, 20))
        t = N(10, 30)
        code = (f"vals = {vals}\nt = {t}\n"
                f"pairs = []\n"
                f"for i in range(len(vals)):\n"
                f"    for j in range(i + 1, len(vals)):\n"
                f"        if vals[i] + vals[j] == t:\n"
                f"            pairs.append((vals[i], vals[j]))\n"
                f"print(pairs)")
        pr = f"在有序列表 {vals} 中找出所有和为 {t} 的数对（i<j），输出数对列表"
    elif name == "pair_count":
        vals = sorted(_ints(8, 1, 15))
        t = N(8, 20)
        code = (f"vals = {vals}\nt = {t}\n"
                f"cnt = 0\n"
                f"for i in range(len(vals)):\n"
                f"    for j in range(i + 1, len(vals)):\n"
                f"        if vals[i] + vals[j] == t:\n"
                f"            cnt += 1\n"
                f"print(cnt)")
        pr = f"统计有序列表 {vals} 中和为 {t} 的数对个数（i<j）"
    else:
        raise KeyError(name)
    return pr, code


NAMES = [
    "isqrt", "comb", "gcd", "lcm", "factorial", "ceil_div", "floor_div",
    "counter_items", "counter_most", "counter_str",
    "set_union", "set_inter", "set_diff", "set_symdiff",
    "enum_zip", "enum_index",
    "sort_abs", "sort_len", "sort_second", "sort_digit",
    "str_chain", "str_join",
    "nested_leaves", "nested_get",
    "flatten_one", "flatten_deep",
    "running_sum", "running_max",
    "pair_sum", "pair_count",
]


def run(code):
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=10)
        if p.returncode != 0:
            return None, p.stderr.strip()[:200]
        return p.stdout.strip(), None
    except Exception as e:
        return None, str(e)


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 500 // len(NAMES) + 1
    rows = []
    seen = set()
    stat = {}
    for nm in NAMES:
        got = 0
        attempts = 0
        while got < per:
            attempts += 1
            if attempts > per * 50:
                raise RuntimeError(f"{nm}: cannot generate {per} unique codes after {attempts} attempts")
            pr, code = code_for(nm)
            if code in seen:
                continue
            out, err = run(code)
            if err or out is None:
                stat[nm] = stat.get(nm, 0) + 1
                continue
            seen.add(code)
            rows.append({
                "instruction": pr,
                "reference_code": code,
                "expected_output": out,
                "source": "gen_code_v2",
                "family": nm,
                "sha1": hashlib.sha1(pr.encode()).hexdigest()[:12],
            })
            got += 1
    rng.shuffle(rows)
    rows = rows[:500]
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "eval", "code_holdout_v2_500.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows ({per} per family x {len(NAMES)} families) to {out_path}")
    if stat:
        print("generation failures:", stat)


if __name__ == "__main__":
    main()
