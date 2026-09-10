#!/usr/bin/env python3
"""Generate synthetic Python exercises (phi-1 CodeExercises shape) with the 27B teacher.

Task 0e-1, docs/standards/p1_data_recipe.md:69. Emits JSONL, one candidate per line:
  {pair_id, topic, variation, sample, prompt, output, tests, exec_pass,
   decontam_hit, dup, kept, discard_reason, est_tokens, model, ts}
prompt+output+tests is the training text; judge() is the in-process exec shape
copied from eval/humaneval_gen.py (6s SIGALRM). Every candidate is written with
its verdict so the discard rate is computed from the file, not asserted.

Acceptance: decontamination against HumanEval and MBPP that FAILS LOUDLY on a
missing benchmark file and runs a planted known-positive control in the main
path (not just selftest); PAIRED samples (two per topic+variation seed) so the
50-sample spot check can see self-repetition.

Runs on the pod against the tileRL serve (localhost:8010/8011, OpenAI-shaped).
restartable: yes -- append-only output, resume skips pair_ids already on disk.
"""
import argparse
import contextlib
import io
import json
import os
import random
import re
import signal
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMANEVAL = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP = os.path.join(ROOT, "data", "eval", "mbpp_holdouts.jsonl")
PORTS = [8010, 8011]
MODEL = "qwen38-27b"
EXEC_TIMEOUT_S = 6
MAX_TOKENS = 2048
CHARS_PER_TOKEN = 3.6  # code domains measured 0.278 tok/byte (p1_data_recipe.md); estimate only

SYSTEM = ("You are a Python exercise author for a university introductory programming "
          "textbook. You write short, self-contained exercises: a function signature "
          "with a docstring, a correct solution body, and executable tests.")

PROMPT_TEMPLATE = """Write ONE Python exercise on the topic: @TOPIC@
Constraint for this exercise: @VARIATION@

Respond with exactly one JSON object, no prose, in this shape:
{"prompt": "def f(...):\\n    \\"\\"\\"<one-line docstring>\\n\\"\\"\\"\\n", "output": "    <solution body>\\n", "tests": "def check():\\n    assert f(...) == ...\\n"}

Rules:
- prompt is the signature plus docstring, ending with a newline after the closing triple quotes.
- output is the function body only, every line indented by 4 spaces.
- tests defines check() with 2-4 assert statements covering normal and edge cases; do not call check().
- Pure functions only: no file, network, process, or unseeded randomness.
- Keep the whole exercise under 40 lines.

Example:
{"prompt": "def reverse_words(s):\\n    \\"\\"\\"Reverse the order of space-separated words in s.\\"\\"\\"\\n", "output": "    return \\" \\".join(reversed(s.split()))\\n", "tests": "def check():\\n    assert reverse_words(\\"hello world\\") == \\"world hello\\"\\n    assert reverse_words(\\"a\\") == \\"a\\"\\n"}"""

VARIATIONS = [
    "using a dictionary",
    "using a set",
    "with input validation raising ValueError",
    "handling empty and single-element inputs",
    "optimized for time complexity",
    "using recursion",
    "using iteration only",
    "with type hints",
    "returning a new value rather than mutating the input",
    "with a worked example in the docstring",
]

TOPICS = """bubble sort
insertion sort
merge sort
quicksort
heapsort
selection sort
counting sort
sort strings by length
sort tuples by their second element
kth smallest element
merge two sorted lists
binary search
linear search
find a local peak in a list
first occurrence of a value
last occurrence of a value
count occurrences of a value
search in a rotated sorted list
binary search on a monotonic predicate
reverse a string
palindrome check
anagram check
word count in a sentence
character frequency
longest word
capitalize every word
snake_case to camelCase
camelCase to snake_case
run-length encoding
run-length decoding
count substrings
collapse repeated whitespace
valid parentheses
roman numeral to integer
integer to roman numeral
string to integer
string rotation check
longest common prefix
edit distance
prime check
sieve of Eratosthenes
prime factorization
greatest common divisor
least common multiple
factorial
fibonacci number
integer power
integer square root
perfect number check
sum of digits
reverse digits
Pascal triangle row
binomial coefficient
matrix transpose
matrix multiplication
vector dot product
mean median mode
variance
Euclidean distance
Manhattan distance
collinear points
shoelace polygon area
quadratic equation roots
base conversion
is leap year
days between two dates
Collatz step count
happy number
narcissistic number
coin change number of ways
coin change minimum coins
0/1 knapsack
longest common subsequence
longest increasing subsequence
unique paths in a grid
climbing stairs
minimum path sum
house robber
maximum subarray sum
subset sum
partition into equal sums
word break
decode ways
best time to buy and sell stock
jump game reachability
gas station tour
binary tree depth
binary tree size
binary tree mirror
inorder traversal
preorder traversal
postorder traversal
binary search tree insert
binary search tree search
validate binary search tree
lowest common ancestor
level order traversal
binary tree diameter
binary heap insert
binary heap extract minimum
heapify a list
graph breadth-first search
graph depth-first search
connected components
cycle detection in an undirected graph
topological sort
shortest path in an unweighted graph
Dijkstra shortest path
union-find
two sum
three sum count
find duplicates
remove duplicates in place
intersection of two lists
union of two lists
missing number
single number
majority element
group anagrams
top k frequent elements
sort by parity
move zeros to the end
rotate a list
flatten a nested list
chunk a list
cumulative sum
sliding window sum
sliding window maximum
prefix range sum
spiral matrix traversal
rotate a matrix 90 degrees
matrix diagonal sum
matrix saddle point
reshape a matrix
identity matrix check
symmetric matrix check
matrix trace
parse a CSV line
tokenize an arithmetic expression
infix to postfix
evaluate postfix
validate an IPv4 address
parse a URL query string
format a table row
wrap text to a width
number to words
words to number
format a duration in seconds
slugify a title
acronym from a phrase
password strength score
Luhn checksum
ISBN-10 checksum
Hamming distance
Soundex encoding
stack class
queue class
circular buffer
LRU cache
counter class
Vector2D class
Fraction class
polynomial class
Date class
binary search tree class
trie class
adjacency-list graph class
priority queue class
FizzBuzz
one tick of Conway's Game of Life
rock paper scissors winner
Monte Carlo pi estimate
random walk final distance
reservoir sampling
Fisher-Yates shuffle
Boyer-Moore majority vote
weekday from a date
add business days
quarter of a year
ISO week number
format an ISO date
parse an ISO date
month length
timezone offset in hours
next occurrence of a weekday""".split("\n")


class _TO(Exception):
    pass


def _alarm(*_a):
    raise _TO()


signal.signal(signal.SIGALRM, _alarm)


def judge(rec):
    """prompt + output + tests + check(); pass iff clean exit. Same shape as humaneval_gen.judge."""
    src = rec["prompt"] + rec["output"] + "\n" + rec["tests"] + "\ncheck()\n"
    g = {"__name__": "__main__"}
    signal.alarm(EXEC_TIMEOUT_S)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(src, g)
        return True
    except BaseException:
        return False
    finally:
        signal.alarm(0)


def parse_exercise(content):
    if not content:
        return None
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        rec = json.loads(s[i : j + 1], strict=False)  # strict=False: teachers emit raw newlines in strings
    except Exception:
        return None
    if not all(isinstance(rec.get(k), str) for k in ("prompt", "output", "tests")):
        return None
    if "def " not in rec["prompt"] or "check" not in rec["tests"]:
        return None
    return {k: rec[k] for k in ("prompt", "output", "tests")}


def teacher(topic, variation, port, reasoning, retries=2):
    prompt = PROMPT_TEMPLATE.replace("@TOPIC@", topic).replace("@VARIATION@", variation)
    msg = {"model": MODEL,
           "messages": [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt}],
           "max_tokens": MAX_TOKENS, "temperature": 0.9}
    if reasoning != "default":
        msg["reasoning_effort"] = reasoning
    body = json.dumps(msg).encode()
    err = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                content = json.loads(r.read())["choices"][0]["message"]["content"]
            rec = parse_exercise(content)
            if rec:
                return rec, None
            err = "parse failed"
        except Exception as e:
            err = str(e)[:120]
    return None, err


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def load_benchmarks(humaneval=HUMANEVAL, mbpp=MBPP):
    """[(source, task_id, key_text, match_text, key_min)] -- RAISES on a missing file."""
    bench = []
    for path, source, key_min in ((humaneval, "humaneval", 100), (mbpp, "mbpp", 40)):
        if not os.path.exists(path):
            raise SystemExit(f"decontam benchmark missing: {path} -- refusing to run on an unmeasured set")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if source == "humaneval":
                bench.append((source, r["task_id"], _norm(r["prompt"]),
                              _norm(r["prompt"] + r.get("canonical_solution", "")), key_min))
            else:
                bench.append((source, r["task_id"], _norm(r.get("text", "")),
                              _norm(r.get("code", "")), key_min))
    return bench


def decontam(rec, bench):
    """Exact (normalized prompt+output equals a benchmark row) or containment
    (benchmark key text, >= key_min normalized chars, inside the exercise prompt)."""
    body = _norm(rec["prompt"] + rec["output"])
    doc = _norm(rec["prompt"])
    for source, task_id, key, match, key_min in bench:
        if match and body == match:
            return [source, task_id, "exact"]
        if key and len(key) >= key_min and key in doc:
            return [source, task_id, "containment"]
    return None


def planted_control(bench, humaneval=HUMANEVAL):
    """A real HumanEval problem rebuilt as an exercise record MUST be caught. Main path."""
    if not os.path.exists(humaneval):
        raise SystemExit(f"planted control cannot read {humaneval}")
    r = json.loads(open(humaneval, encoding="utf-8").readline())
    rec = {"prompt": r["prompt"], "output": r["canonical_solution"], "tests": ""}
    hit = decontam(rec, bench)
    if not hit or hit[2] != "exact":
        raise SystemExit(f"planted control {r['task_id']} NOT caught (hit={hit}) -- "
                         "decontamination is not measuring; refusing to run")
    return r["task_id"]


def alive_ports():
    up = []
    for p in PORTS:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    up.append(p)
        except Exception:
            pass
    if not up:
        raise SystemExit(f"no teacher serve answering on {PORTS}")
    return up


def est_tokens(rec):
    return int(len((rec["prompt"] + rec["output"] + rec["tests"]).encode()) / CHARS_PER_TOKEN)


def gen_pair(pair_id, topic, variation, port, reasoning):
    return pair_id, topic, variation, [teacher(topic, variation, port, reasoning) for _ in (0, 1)]


def read_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["pair_id"])
            except Exception:
                pass
    return done


def run(args):
    ports = alive_ports()
    bench = load_benchmarks()
    planted = planted_control(bench)
    print(f"decontam OK: {len(bench)} benchmark rows, planted control {planted} caught; "
          f"serve on ports {ports}", flush=True)

    topics = [l.strip() for l in open(args.seeds, encoding="utf-8") if l.strip()] if args.seeds else TOPICS
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = read_done(out)
    seen = set()
    stats = {"kept": 0, "exec_fail": 0, "parse_fail": 0, "dup": 0, "decontam": 0}
    kept_tok = 0
    t0 = time.time()
    rng = random.Random(args.seed)

    def pairs():
        for rnd in range(args.max_rounds):
            order = list(range(len(topics)))
            rng.shuffle(order)
            for ti in order:
                vi = (ti + rnd) % len(VARIATIONS)
                pair_id = f"t{ti:04d}-v{vi:02d}-r{rnd:03d}"
                if pair_id in done:
                    continue
                yield pair_id, topics[ti], VARIATIONS[vi]

    def write(rec):
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    submitted = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        pair_iter = pairs()
        for _ in range(args.workers * 2):
            try:
                pid, topic, var = next(pair_iter)
            except StopIteration:
                break
            futs[pool.submit(gen_pair, pid, topic, var, ports[submitted % len(ports)], args.reasoning)] = pid
            submitted += 1
        while futs:
            done_fut = next(as_completed(futs))
            del futs[done_fut]
            pair_id, topic, variation, items = done_fut.result()
            completed += 1
            for sample, (rec, err) in enumerate(items):
                if rec is None:
                    stats["parse_fail"] += 1
                    write({"pair_id": pair_id, "topic": topic, "variation": variation,
                           "sample": sample, "kept": False, "discard_reason": f"teacher:{err}",
                           "model": MODEL, "ts": _ts()})
                    continue
                ok = judge(rec)
                hit = decontam(rec, bench)
                sig = _norm(rec["prompt"] + rec["output"])
                is_dup = sig in seen
                if ok:
                    seen.add(sig)
                kept = ok and not hit and not is_dup
                if kept:
                    stats["kept"] += 1
                    kept_tok += est_tokens(rec)
                elif not ok:
                    stats["exec_fail"] += 1
                elif hit:
                    stats["decontam"] += 1
                elif is_dup:
                    stats["dup"] += 1
                write({"pair_id": pair_id, "topic": topic, "variation": variation, "sample": sample,
                       "prompt": rec["prompt"], "output": rec["output"], "tests": rec["tests"],
                       "exec_pass": ok, "decontam_hit": hit, "dup": is_dup, "kept": kept,
                       "discard_reason": None if kept else ("exec" if not ok else "decontam" if hit else "dup"),
                       "est_tokens": est_tokens(rec), "model": MODEL, "ts": _ts()})
            if completed % 64 == 0:
                el = time.time() - t0
                print(f"[{el/60:.1f}m] pairs={completed} kept={stats['kept']} "
                      f"exec_fail={stats['exec_fail']} parse_fail={stats['parse_fail']} "
                      f"dup={stats['dup']} decontam={stats['decontam']} "
                      f"kept_tokens~{kept_tok/1e6:.1f}M rate={stats['kept']/max(completed*2,1):.2f}/item",
                      flush=True)
            if args.smoke and completed >= args.smoke:
                print(f"smoke: {completed} pairs done", flush=True)
                break
            if not args.smoke and kept_tok >= args.target_tokens:
                print(f"target {args.target_tokens:.0f} est tokens reached", flush=True)
                break
            try:
                pid, topic, var = next(pair_iter)
                futs[pool.submit(gen_pair, pid, topic, var, ports[submitted % len(ports)], args.reasoning)] = pid
                submitted += 1
            except StopIteration:
                pass

    total = sum(stats.values())
    print(f"DONE pairs={completed} candidates={total} kept={stats['kept']} "
          f"discard_rate={1 - stats['kept']/max(total,1):.3f} kept_tokens~{kept_tok/1e6:.1f}M "
          f"({(time.time()-t0)/60:.1f}m)", flush=True)


def _ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def selftest():
    good = {"prompt": "def f(x):\n    \"\"\"Return x+1.\"\"\"\n", "output": "    return x + 1\n",
            "tests": "def check():\n    assert f(1) == 2\n    assert f(-1) == 0\n"}
    assert judge(good), "known-good record must pass"
    assert not judge({**good, "output": "    return x + 2\n"}), "wrong body must fail"
    assert not judge({"prompt": "def f():\n    \"\"\"Loop forever.\"\"\"\n", "output": "    while True:\n        pass\n",
                      "tests": "def check():\n    assert f() is None\n"}), "timeout must fail"
    assert parse_exercise('```json\n{"prompt": "def a():\n    \'\'\'d\'\'\'\n", "output": "    pass\n", '
                          '"tests": "def check():\n    assert True\n"}\n```'), "fenced JSON must parse"
    assert parse_exercise("no json here") is None, "garbage must return None"
    bench = [("humaneval", "HE/0", "def has_close_elements(numbers, threshold)",
              "def has_close_elements(numbers, threshold): for idx, elem", 100),
             ("mbpp", "mbpp-0", "Write a function to find the longest chain", "class Pair(object):", 40)]
    assert decontam({"prompt": "def has_close_elements(numbers, threshold): ",
                     "output": "for idx, elem"}, bench) == ["humaneval", "HE/0", "exact"], "exact copy must hit"
    assert decontam({"prompt": "def f():\n    \"\"\"Write a function to find the longest chain of pairs.\"\"\"\n",
                     "output": "    pass\n"}, bench) == ["mbpp", "mbpp-0", "containment"], "containment must hit"
    assert decontam({"prompt": "def f():\n    \"\"\"Add two numbers.\"\"\"\n", "output": "    return a + b\n"},
                    bench) is None, "clean exercise must not hit"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        he = os.path.join(d, "he.jsonl")
        open(he, "w").write(json.dumps({"task_id": "HumanEval/0",
                                        "prompt": "def has_close_elements(numbers, threshold):\n",
                                        "canonical_solution": "    for idx, elem in enumerate(numbers):\n"}) + "\n")
        open(os.path.join(d, "mbpp.jsonl"), "w").write(json.dumps(
            {"task_id": "mbpp-0", "text": "x", "code": "y"}) + "\n")
        b = load_benchmarks(he, os.path.join(d, "mbpp.jsonl"))
        assert planted_control(b, he) == "HumanEval/0", "planted control must be caught in the main path"
        try:
            load_benchmarks(os.path.join(d, "absent.jsonl"), os.path.join(d, "mbpp.jsonl"))
            raise AssertionError("missing benchmark must raise")
        except SystemExit:
            pass
    print("selftest OK: judge pass/fail/timeout, parse, decontam exact/containment/clean, "
          "planted control, loud missing-benchmark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/aupai/data/p1/exercises.jsonl")
    ap.add_argument("--seeds", help="topic file (one per line); defaults to the built-in list")
    ap.add_argument("--target-tokens", type=float, default=1.8e8)
    ap.add_argument("--max-rounds", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reasoning", default="none", choices=["none", "low", "medium", "high", "default"])
    ap.add_argument("--smoke", type=int, default=0, help="N pairs to a smoke file, then exit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.smoke:
        args.out = args.out.replace(".jsonl", "_smoke.jsonl")
        args.target_tokens = float("inf")
    run(args)


if __name__ == "__main__":
    main()
