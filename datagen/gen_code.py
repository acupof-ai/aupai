#!/usr/bin/env python3
"""Procedural Chinese Python basics tutorials. Code correctness verified by running
each snippet; the shown 运行输出 is the real captured stdout, so it is never wrong.

Templates: variables, arithmetic, for/while, if/elif, list/dict operations, string,
functions, simple algorithms (factorial, fib, gcd, prime, sort). Each → one jsonl
row {"instruction": 中文任务, "output": "```python ... ```\n运行输出: ...\n\n讲解: ..."}.
Deterministic seed, dedup by code.
"""

import ast
import json
import random
import subprocess
import sys

rng = random.Random(2024)


def code_for(name):
    n = lambda lo, hi: rng.randint(lo, hi)  # noqa
    if name == "print":
        w = rng.choice(["你好,Python!", "欢迎学习编程", "1 + 1 = ?", "坚持就是胜利"])
        code = f'print("{w}")'
        pr = f"用 print 输出一句话:{w}"
        exp = lambda o: o.strip().startswith(w[:3])
    elif name == "var":
        a, b = n(3, 50), n(2, 40)
        code = f"a = {a}\nb = {b}\nprint(a + b)\nprint(a - b)"
        pr = f"定义变量 a={a}、b={b},输出它们的和与差"
        exp = lambda o: o.strip() == f"{a + b}\n{a - b}"
    elif name == "for_sum":
        m = n(5, 30)
        code = f"s = 0\nfor x in range(1, {m}+1):\n    s += x\nprint(s)"
        pr = f"用 for 循环求 1 到 {m} 的和"
        exp = lambda o: o.strip() == str(m * (m + 1) // 2)
    elif name == "while_cnt":
        m = n(4, 10)
        code = f"i = {m}\nwhile i >= 1:\n    print(i)\n    i -= 1"
        pr = f"用 while 循环从 {m} 倒数打印到 1"
        exp = lambda o: o.strip() == "\n".join(map(str, range(m, 0, -1)))
    elif name == "grade":
        sc = n(0, 100)
        if sc >= 90:
            g = "优秀"
        elif sc >= 75:
            g = "良好"
        elif sc >= 60:
            g = "及格"
        else:
            g = "不及格"
        code = f"score = {sc}\nif score >= 90:\n    print('优秀')\nelif score >= 75:\n    print('良好')\nelif score >= 60:\n    print('及格')\nelse:\n    print('不及格')"
        pr = f"分数为{sc},用 if/elif/else 输出对应成绩等级"
        exp = lambda o: o.strip() == g
    elif name == "listmax":
        lo = n(1, 30)
        hi = n(lo, 50)
        vals = [n(lo, hi) for _ in range(n(4, 7))]
        code = f"x = {vals}\nprint(max(x))\nprint(min(x))"
        pr = f"列表为 {vals},用 max 和 min 求最大值与最小值"
        exp = lambda o: o.strip() == f"{max(vals)}\n{min(vals)}"
    elif name == "listsum":
        vals = [n(1, 100) for _ in range(n(4, 7))]
        s, ln = sum(vals), len(vals)
        code = f"x = {vals}\nprint(sum(x))\nprint(sum(x) / len(x))"
        pr = f"列表为 {vals},求元素总和与平均值"
        exp = lambda o: o.strip() == f"{s}\n{s / ln}"
    elif name == "revstr":
        w = rng.choice(["hello", "python", "algorithm", "data", "study"])
        code = f"s = '{w}'\nprint(s[::-1])"
        pr = f"把字符串 {w} 反转输出"
        exp = lambda o: o.strip() == w[::-1]
    elif name == "countv":
        w = rng.choice(["banana", "apple", "computer", "education", "opinion"])
        code = f"s = '{w}'\ncnt = 0\nfor ch in s:\n    if ch in 'aeiou':\n        cnt += 1\nprint(cnt)"
        pr = f"统计字符串 {w} 中元音字母(a/e/i/o/u)的个数"
        exp = lambda o: o.strip() == str(sum(1 for ch in w if ch in "aeiou"))
    elif name == "evenodd":
        vals = [n(1, 20) for _ in range(n(5, 8))]
        evens = [v for v in vals if v % 2 == 0]
        code = f"x = {vals}\neven = []\nfor v in x:\n    if v % 2 == 0:\n        even.append(v)\nprint(even)"
        pr = f"从列表 {vals} 中筛选出所有偶数"
        exp = lambda o: o.strip() == str(evens)
    elif name == "dictcnt":
        w = rng.choice(["aabbbbccc", "programming", "helloworld", "banana"])
        code = f"s = '{w}'\ncnt = {{}}\nfor ch in s:\n    cnt[ch] = cnt.get(ch, 0) + 1\nprint(cnt)"
        pr = f"用字典统计字符串 {w} 中各字符出现的次数"
        exp = lambda o: ast.literal_eval(o.strip()) == {c: w.count(c) for c in w}
    elif name == "fact":
        m = n(3, 10)
        code = f"result = 1\nfor i in range(1, {m}+1):\n    result *= i\nprint(result)"
        pr = f"用循环求 {m} 的阶乘"
        import math

        exp = lambda o: o.strip() == str(math.factorial(m))
    elif name == "fib":
        k = n(5, 10)
        code = "a, b = 0, 1\nfor _ in range(%d):\n    print(a)\n    a, b = b, a + b" % k
        seq = [0, 1]
        while len(seq) < k:
            seq.append(seq[-1] + seq[-2])
        pr = f"生成斐波那契数列的前 {k} 项"
        exp = lambda o: o.strip() == "\n".join(map(str, seq[:k]))
    elif name == "gcd":
        a, b = n(36, 180), n(24, 150)
        x, y = a, b
        while y:
            x, y = y, x % y
        code = f"def gcd(m, n):\n    while n:\n        m, n = n, m % n\n    return m\nprint(gcd({a}, {b}))"
        pr = f"用辗转相除法求 {a} 和 {b} 的最大公约数"
        exp = lambda o: o.strip() == str(x)
    elif name == "prime":
        m = n(2, 100)
        isp = all(m % i for i in range(2, int(m**0.5) + 1)) if m > 1 else False
        res = "是素数" if isp else "不是素数"
        code = f"""import math
n = {m}
if n < 2:
    print('不是素数')
else:
    ok = True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            ok = False
            break
    print('是素数' if ok else '不是素数')"""
        pr = f"判断 {m} 是否为素数"
        exp = lambda o: o.strip() == res
    elif name == "primes_n":
        m = n(10, 40)
        ps = [x for x in range(2, m + 1) if all(x % i for i in range(2, int(x**0.5) + 1))]
        code = f"""def is_prime(x):
    if x < 2:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True
print([x for x in range(2, {m}+1) if is_prime(x)])"""
        pr = f"输出 2 到 {m} 之间所有的素数"
        exp = lambda o: o.strip() == str(ps)
    elif name == "func_max":
        a, b = n(1, 90), n(1, 90)
        code = f"def bigger(x, y):\n    return x if x > y else y\nprint(bigger({a}, {b}))"
        pr = f"定义函数 bigger 返回 {a} 和 {b} 中的较大者并调用"
        exp = lambda o: o.strip() == str(max(a, b))
    elif name == "func_area":
        r = n(1, 20)
        import math

        a = round(math.pi * r * r, 2)
        code = f"import math\ndef area(r):\n    return math.pi * r * r\nprint(round(area({r}), 2))"
        pr = f"定义函数 area 计算半径为 {r} 的圆面积并保留两位小数输出"
        exp = lambda o: abs(float(o.strip()) - a) < 0.01
    elif name == "func_rev":
        vals = [n(1, 9) for _ in range(n(3, 6))]
        code = f"def reverse(x):\n    return x[::-1]\nprint(reverse({vals}))"
        pr = f"定义函数 reverse 把列表 {vals} 倒序返回并输出"
        exp = lambda o: o.strip() == str(vals[::-1])
    elif name == "bubble":
        vals = [n(1, 50) for _ in range(n(5, 8))]
        code = (
            f"x = {vals}\nfor i in range(len(x)):\n"
            "    for j in range(0, len(x)-1-i):\n"
            "        if x[j] > x[j+1]:\n            x[j], x[j+1] = x[j+1], x[j]\nprint(x)"
        )
        pr = f"用冒泡排序把 {vals} 从小到大排序"
        exp = lambda o: o.strip() == str(sorted(vals))
    elif name == "listcomp":
        m = n(5, 12)
        code = f"squares = [x * x for x in range(1, {m}+1)]\nprint(squares)"
        pr = f"用列表推导式生成 1 到 {m} 的平方列表"
        exp = lambda o: o.strip() == str([x * x for x in range(1, m + 1)])
    elif name == "fmtstr":
        name_, age = rng.choice(["小明", "小红", "阿强", "丽丽"]), n(12, 30)
        code = f'name = "{name_}"\nage = {age}\nprint(f"我叫{{name}},今年{{age}}岁")'
        pr = f"用 f-string 输出『我叫{name_},今年{age}岁』"
        exp = lambda o: o.strip() == f"我叫{name_},今年{age}岁"
    elif name == "indexof":
        vals = [n(1, 20) for _ in range(n(4, 7))]
        t = vals[n(0, len(vals) - 1)]
        code = f"x = {vals}\nprint(x.index({t}))"
        pr = f"查找 {t} 在列表 {vals} 中第一次出现的位置"
        exp = lambda o: o.strip() == str(vals.index(t))
    elif name == "dedup":
        vals = [n(1, 6) for _ in range(n(5, 9))]
        code = f"x = {vals}\nprint(list(dict.fromkeys(x)))\nprint(len(set(x)))"
        pr = f"去掉列表 {vals} 中的重复元素(去重)并输出去重后长度"
        exp = lambda o: o.strip() == f"{list(dict.fromkeys(vals))}\n{len(set(vals))}"
    elif name == "matrix":
        a = [n(1, 9) for _ in range(3)]
        b = [n(1, 9) for _ in range(3)]
        code = f"a = {a}\nb = {b}\nprint([x + y for x, y in zip(a, b)])"
        pr = f"求向量 {a} 与 {b} 对应元素之和的新向量"
        exp = lambda o: o.strip() == str([x + y for x, y in zip(a, b)])
    elif name == "setmem":
        s = rng.sample(range(1, 50), n(4, 7))
        m = rng.choice(s + [rng.randint(1, 50)])
        code = f"s = {set(s)}\nprint({m} in s)"
        pr = f"判断 {m} 是否在集合 {set(s)} 中"
        exp = lambda o: o.strip() == str(m in set(s))
    elif name == "slice":
        w = rng.choice(["abcdefg", "hijklmn", "program", "science", "python"])
        i, j = n(1, 2), n(4, len(w))
        if i >= j:
            i, j = 1, len(w) - 1
        code = f"s = '{w}'\nprint(s[{i}:{j}])"
        pr = f"取字符串 {w} 从下标 {i} 到 {j} 的子串"
        exp = lambda o: o.strip() == w[i:j]
    elif name == "sum_even":
        vals = [n(1, 30) for _ in range(n(5, 9))]
        code = f"x = {vals}\ntotal = 0\nfor v in x:\n    if v % 2 == 0:\n        total += v\nprint(total)"
        pr = f"求列表 {vals} 中所有偶数的和"
        exp = lambda o: o.strip() == str(sum(v for v in vals if v % 2 == 0))
    elif name == "count_gt":
        vals = [n(1, 40) for _ in range(n(5, 9))]
        thr = n(15, 25)
        code = f"x = {vals}\ncount = 0\nfor v in x:\n    if v > {thr}:\n        count += 1\nprint(count)"
        pr = f"统计列表 {vals} 中大于 {thr} 的元素个数"
        exp = lambda o: o.strip() == str(sum(1 for v in vals if v > thr))
    elif name == "phone":
        name_, num = rng.choice(["张三", "李四", "王五", "赵六"]), str(n(13000000000, 13999999999))
        code = (
            f'phone = {{"张三": "13800000001", "李四": "13800000002", "{name_}": "{num}"}}\n'
            f'print(phone.get("{name_}", "查无此人"))'
        )
        pr = f"用字典实现通讯录,查询 {name_} 的号码"
        exp = lambda o: o.strip() == num
    else:
        raise KeyError(name)

    return pr, code, exp


def run(code):
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        if p.returncode != 0:
            return None, p.stderr.strip()[:200]
        return p.stdout.strip(), None
    except Exception as e:
        return None, str(e)


def main():
    N = int(sys.argv[1])
    NAMES = [
        "print",
        "var",
        "for_sum",
        "while_cnt",
        "grade",
        "listmax",
        "listsum",
        "revstr",
        "countv",
        "evenodd",
        "dictcnt",
        "fact",
        "fib",
        "gcd",
        "prime",
        "primes_n",
        "func_max",
        "func_area",
        "func_rev",
        "bubble",
        "listcomp",
        "fmtstr",
        "indexof",
        "dedup",
        "matrix",
        "setmem",
        "slice",
        "sum_even",
        "count_gt",
        "phone",
    ]
    rows = []
    seen = set()
    stat = {}
    while len(rows) < N:
        nm = rng.choice(NAMES)
        pr, code, exp = code_for(nm)
        if code in seen:
            continue
        out, err = run(code)
        if err:
            stat[nm] = stat.get(nm, 0) + 1
            continue
        if not exp(out):
            stat[nm] = stat.get(nm, 0) + 1
            continue
        seen.add(code)
        expl = {
            "print": "print 是 Python 最基础的输出函数。把内容放进一对引号里交给 print,即可在屏幕打印该内容。",
            "var": "变量用于存值;先用等号赋值,才能参与 + - * / 等算术运算。要点:变量使用前必须先赋值。",
            "for_sum": "range(1, n+1) 生成 1 到 n 的整数序列;累加器 s 每轮加上当前数,循环结束即得总和。这是最典型的累加套路。",
            "while_cnt": "while 在条件成立时反复执行循环体;循环体里必须让条件逐渐变假(这里 i 每次减 1),否则会死循环。",
            "grade": "if/elif/else 从上到下逐个检查;命中一个分支就结束判断并执行对应输出,分数归入正确档位。",
            "listmax": "max() 和 min() 是内置函数,分别返回列表中的最大与最小元素,无需自己遍历。",
            "listsum": "sum() 求列表元素总和,len() 求元素个数,avg = 总和 ÷ 个数,即平均值。",
            "revstr": "切片 s[::-1] 以步长 -1 从末尾向开头取值,得到原字符串的倒序,是反转最简洁的写法。",
            "countv": "遍历字符串的每个字符,若落在元音集合 'aeiou' 中则计数加一。注意小写元音五个: a e i o u。",
            "evenodd": "一个整数是偶数当且仅当 x % 2 == 0(对 2 取模余 0)。据此把偶数挑出放进新列表。",
            "dictcnt": "字典以字符为键、次数为值;dict.get(k, 0) 取不到时返回默认 0,再 +1 写回,实现跨越式计数。",
            "fact": "阶乘 n! = 1×2×…×n;用循环从 1 连乘到 n,result 初值置 1,避免从 0 开始把结果清零。",
            "fib": "斐波那契每项是前两项之和;a, b = b, a+b 一行同时更新两个变量并滚动前进,产生下一项。",
            "gcd": "辗转相除:两数相除取余,余数为 0 时除数为大公约数;否则把 (除数, 余数) 继续相除,循环至余 0。",
            "prime": "素数是大于 1 且只有 1 和它本身两个因数的数;从 2 试除到 √n 即可,若都不整除则为素数。",
            "primes_n": "双层循环:外层是候选数,内层逐个试除判断素数;把是素数的数收集进列表。判断到 √x 即可截断。",
            "func_max": "def 定义函数,return 把结果交回调用处;封装好逻辑后,传不同实参即可复用同一段计算。",
            "func_area": "圆面积公式 S = πr²;用 math.pi 提高精度,round(值, 2) 保留两位小数输出。",
            "func_rev": "切片 [::-1] 直接生成倒序副本;把这一逻辑包进函数,任意列表都能复用。",
            "bubble": "冒泡排序:相邻比较、大的后移;每趟外循环把当前最大数送到数组末尾,内循环范围随之缩短。",
            "listcomp": "列表推导式 [表达式 for x in 序列] 一行同时完成 生成、变换与收集,比手写循环更简洁。",
            "fmtstr": "f-string 在字符串前加 f,花括号 {} 内写变量或表达式,自动把值嵌入字符串,最直观的格式化方式。",
            "indexof": "list.index(x) 返回元素 x 首次出现的下标。若元素不存在会抛 ValueError,通常先用 in 判断。",
            "dedup": "set 是无序不重复的集合;len(set(x)) 直接得到去重后的个数。若要保序去重,可用 dict.fromkeys。",
            "matrix": "zip(a, b) 把两个序列按位置配对,列表推导式对每一对求和,得到对应元素之和的新序列。",
            "setmem": "集合用于快速判断成员:用 in 查询平均只需 O(1) 时间,比在列表里线性查找快得多。",
            "slice": "切片 s[i:j] 从下标 i 取到 j(不含 j);下标从 0 开始,i 或 j 缺省表示默认到首或到末。",
            "sum_even": "遍历列表,凡偶数(x % 2 == 0)就加进累加器 total,循环结束 total 即所有偶数之和。",
            "count_gt": "逐个元素与阈值比较,大于阈值的计数加一;循环完即得到满足条件的元素个数。",
            "phone": "字典按名字存号码;dict.get(key, 默认值) 取键时若不存在返回默认值,实现友好查询而非报错。",
        }[nm]
        rows.append(
            {
                "instruction": pr,
                "output": f"```python\n{code}\n```\n\n运行输出:\n```\n{out}\n```\n\n讲解:{expl}",
            }
        )
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tutorials | template failures={dict(stat)}", file=sys.stderr)


if __name__ == "__main__":
    main()
