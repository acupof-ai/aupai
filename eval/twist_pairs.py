#!/usr/bin/env python3
# restartable: pure; rewrites eval/twist_pairs.jsonl from the table below on every run.
"""Base/twist pairs: a common idiom (base) and the same function with one spec change (twist)
whose tests the base idiom fails. A model that passes base and fails twist recognises the task
type without reading the spec. Scored by eval/probe_cases.py --data eval/twist_pairs.jsonl.

    python3 eval/twist_pairs.py --selftest   # each reference passes its own tests,
                                             # and the base reference fails the twist tests
    python3 eval/twist_pairs.py              # writes eval/twist_pairs.jsonl
"""
import argparse
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twist_pairs.jsonl")

# (name, base_sig, base_doc, base_ref, base_tests, twist_sig, twist_doc, twist_ref, twist_tests)
PAIRS = [
    ("close", "has_close_elements(numbers, threshold)",
     "Return True if any two numbers in the list are closer to each other than threshold.",
     "return any(abs(a-b)<threshold for i,a in enumerate(numbers) for b in numbers[i+1:])",
     ["([1.0, 2.0, 3.0], 0.5) == False", "([1.0, 2.8, 3.0], 0.3) == True"],
     "all_far_apart(numbers, threshold)",
     "Return True if every pair of numbers in the list is at least threshold apart.",
     "return all(abs(a-b)>=threshold for i,a in enumerate(numbers) for b in numbers[i+1:])",
     ["([1.0, 2.0, 3.0], 0.5) == True", "([1.0, 2.8, 3.0], 0.3) == False"]),
    ("prime", "is_prime(n)", "Return True if n is a prime number, otherwise False.",
     "return n>1 and all(n%i for i in range(2,int(n**0.5)+1))",
     ["(2) == True", "(1) == False", "(9) == False", "(13) == True"],
     "is_composite(n)",
     "Return True if n is a composite number (greater than 1 and not prime), otherwise False.",
     "return n>1 and not all(n%i for i in range(2,int(n**0.5)+1))",
     ["(2) == False", "(1) == False", "(9) == True", "(13) == False"]),
    ("fib", "fib(n)", "Return the n-th Fibonacci number, with fib(0) = 0 and fib(1) = 1.",
     "a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a",
     ["(0) == 0", "(1) == 1", "(10) == 55"],
     "lucas(n)", "Return the n-th Lucas number, with lucas(0) = 2 and lucas(1) = 1; "
     "each later number is the sum of the previous two.",
     "a,b=2,1\n    for _ in range(n): a,b=b,a+b\n    return a",
     ["(0) == 2", "(1) == 1", "(5) == 11"]),
    ("sum_n", "sum_to_n(n)", "Return the sum of all integers from 1 to n.",
     "return n*(n+1)//2", ["(1) == 1", "(10) == 55"],
     "sum_even_to_n(n)", "Return the sum of the even integers from 1 to n.",
     "return sum(i for i in range(1,n+1) if i%2==0)", ["(1) == 0", "(10) == 30"]),
    ("positive", "get_positive(l)", "Return only the positive numbers in the list, in order.",
     "return [x for x in l if x>0]", ["([-1, 2, 0, 5]) == [2, 5]"],
     "get_non_positive(l)", "Return only the numbers in the list that are zero or negative, in order.",
     "return [x for x in l if x<=0]", ["([-1, 2, 0, 5]) == [-1, 0]"]),
    ("strlen", "strlen(s)", "Return the length of the string s.",
     "return len(s)", ["('abc') == 3", "('a b') == 3"],
     "count_non_space(s)", "Return the number of characters in s that are not spaces.",
     "return sum(c!=' ' for c in s)", ["('abc') == 3", "('a b') == 2"]),
    ("unique", "unique(l)", "Return the sorted unique elements of the list.",
     "return sorted(set(l))", ["([3, 1, 3, 2]) == [1, 2, 3]"],
     "unique_in_order(l)",
     "Return the unique elements of the list in the order they first appear (do not sort).",
     "seen=[]\n    [seen.append(x) for x in l if x not in seen]\n    return seen",
     ["([3, 1, 3, 2]) == [3, 1, 2]"]),
    ("incr", "incr_list(l)", "Return the list with every element increased by 1.",
     "return [x+1 for x in l]", ["([1, 2, 3]) == [2, 3, 4]"],
     "decr_list_by_two(l)", "Return the list with every element decreased by 2.",
     "return [x-2 for x in l]", ["([1, 2, 3]) == [-1, 0, 1]"]),
    ("gcd", "gcd(a, b)", "Return the greatest common divisor of two positive integers a and b.",
     "import math\n    return math.gcd(a,b)", ["(12, 18) == 6", "(7, 5) == 1"],
     "lcm(a, b)", "Return the least common multiple of two positive integers a and b.",
     "import math\n    return a*b//math.gcd(a,b)", ["(12, 18) == 36", "(7, 5) == 35"]),
    ("reverse", "reverse(s)", "Return the string s reversed.",
     "return s[::-1]", ["('abc def') == 'fed cba'"],
     "reverse_each_word(s)",
     "Reverse the letters of each word in s but keep the words in their original order; "
     "words are separated by single spaces.",
     "return ' '.join(w[::-1] for w in s.split(' '))", ["('abc def') == 'cba fed'"]),
    ("max", "max_element(l)", "Return the largest element of a non-empty list.",
     "return max(l)", ["([1, 5, 3]) == 5"],
     "second_largest(l)", "Return the second largest distinct value in the list.",
     "return sorted(set(l))[-2]", ["([1, 5, 3, 5]) == 3"]),
    ("vowels", "count_vowels(s)", "Return the number of vowels (a, e, i, o, u, either case) in s.",
     "return sum(c in 'aeiouAEIOU' for c in s)", ["('Hello World') == 3"],
     "count_consonants(s)",
     "Return the number of letters in s that are not vowels (vowels are a, e, i, o, u, either case).",
     "return sum(c.isalpha() and c not in 'aeiouAEIOU' for c in s)", ["('Hello World') == 7"]),
]


def row(tid, sig, doc, tests):
    name = sig.split("(")[0]
    return {"task_id": tid, "entry_point": name,
            "prompt": f"def {sig}:\n    \"\"\"{doc}\"\"\"\n",
            "test": "def check(candidate):\n" + "".join(f"    assert candidate{t}\n" for t in tests)}


def rows():
    out = []
    for p in PAIRS:
        out.append(row(f"base/{p[0]}", p[1], p[2], p[4]))
        out.append(row(f"twist/{p[0]}", p[5], p[6], p[8]))
    return out


def passes(r, body):
    src = r["prompt"] + "    " + body + "\n" + r["test"] + f"\ncheck({r['entry_point']})\n"
    try:
        exec(src, {"__name__": "__main__"})
        return True
    except BaseException:
        return False


def selftest():
    rs = rows()
    for i, p in enumerate(PAIRS):
        b, t = rs[2 * i], rs[2 * i + 1]
        assert passes(b, p[3]), f"{p[0]}: base ref fails base tests"
        assert passes(t, p[7]), f"{p[0]}: twist ref fails twist tests"
        tb = dict(t, prompt=t["prompt"].replace(t["entry_point"], b["entry_point"]),
                  test=t["test"], entry_point=b["entry_point"])
        assert not passes(tb, p[3]), f"{p[0]}: base idiom passes the twist tests; twist does not discriminate"
    print(f"selftest ok: {len(PAIRS)} pairs, every twist rejects its base idiom")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
    else:
        with open(OUT, "w", encoding="utf-8") as f:
            for r in rows():
                f.write(json.dumps(r) + "\n")
        print(f"wrote {OUT}: {2 * len(PAIRS)} rows")
