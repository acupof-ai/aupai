"""Problem-type classifier for HumanEval-shaped code data.

ONE classifier, used on every dataset in this study (HumanEval, SFT A sources, the RL
pool, pretrain code). It is deliberately mechanical and transparent: every label is a
weighted set of literal signal patterns, the score is a count of matches, and the
assignment is top-1 plus top-2 when it is close. There is no model in the loop, so the
same input always yields the same type, and a reader can see exactly which signal fired.

The taxonomy is defined by WHAT THE SOLUTION MUST DO, not by the topic the prompt
mentions. A task that describes a fruit basket but is solved with a subtraction is
`arithmetic_basics`; a task about planets solved by an index lookup is `collection_transform`.

Why weighted signals and not first-match: the two errors this repo has already made
twice this week were both criteria that decided their own answer (a threshold calibrated
on its own subject, a regex that trimmed a path before testing it). First-match on
ordered keywords has the same shape -- one early pattern swallows a whole class. Weighted
scoring with an audit is the version that can be shown wrong.

Usage:
    python3 scripts/problem_types.py --selftest
    python3 scripts/problem_types.py --humaneval data/eval/humaneval/humaneval_164.jsonl
"""
# restartable: read-only single pass over already-materialized jsonl rows; the cost is IO,
# and an interrupt loses only the rows not yet classified because there is no cumulative state.
import argparse
import json
import re

# Label -> (human definition, [(pattern, weight), ...]).
# Patterns are matched case-insensitively against the classification text.
# A pattern is a regex; plain words are word-bounded automatically by `\b`-free
# substring matching on the normalized text (lowercased, whitespace collapsed).
TYPES = {
    "bracket_nesting": (
        "Balance and depth of parentheses/brackets; the answer is a property of the "
        "nesting structure, not of the characters as text.",
        [(r"\bparenthes", 3), (r"\bbracket", 3), (r"\bnesting", 3), (r"\bbalanced\b", 2),
         (r"open and clos", 2), (r"\bparen_string\b", 4), (r"opening bracket", 3),
         (r"deepest level", 3)]),
    "string_parse": (
        "Input is a formatted string; the work is extracting structured values from it "
        "(numbers, tokens, records, validity).",
        [(r"\bparse\b", 3), (r"represented? (as|by) a string", 3), (r"string represent", 3),
         (r"special ascii format", 4), (r"delimited", 3), (r"separated by (commas|spaces|spaces or commas)", 3),
         (r"valid date|date is valid|validat", 4), (r"file'?s? name is considered", 4), (r"sentences are delimited", 4),
         (r"space-delimited string", 4), (r"hexadecimal number as a string", 4),
         (r"words separated by", 3), (r"music_string", 4)]),
    "dynamic_programming": (
        "A sequence or optimum defined by a recurrence or by overlapping subproblems "
        "(including path/subarray optimisation).",
        [(r"\bfib4\b", 5), (r"\bfibfib\b", 5), (r"\btribonacci", 5), (r"\bfibonacci", 4),
         (r"defined as follows", 2), (r"sub-?array", 4), (r"\bcollatz\b", 3),
         (r"minimum sum", 3), (r"number of triples", 3), (r"minimum number of (elements|changes)", 3),
         (r"\bminpath\b", 5), (r"each cell of the grid", 3), (r"table\b", 1)]),
    "number_theory": (
        "Integer arithmetic where the subject is divisibility, primality, factors or "
        "digit properties of a number.",
        [(r"\bprime\b", 4), (r"\bdivisor", 4), (r"\bdivides\b", 4), (r"\bfactor", 4), (r"\bgcd\b", 5),
         (r"greatest common divisor", 5), (r"modulo", 3), (r"divisible", 3),
         (r"\beven digit", 3), (r"\bodd digit", 3), (r"product of the odd", 3),
         (r"sum of its digits", 3), (r"unit digits", 3), (r"perfect square", 2),
         (r"\bcube\b", 2), (r"simple power", 3), (r"multiplication of", 3)]),
    "base_and_bits": (
        "Changing or reading a number's representation: base conversion, binary, roman "
        "numerals, bit patterns.",
        [(r"\bbinary\b", 4), (r"change (numerical )?base", 5), (r"roman numeral", 5),
         (r"\bhexadecimal\b", 4), (r"\bmd5\b", 4), (r"number of ones in their binary", 5),
         (r"binary (representation|format)", 4), (r"\bbit\b", 2)]),
    "float_geometry": (
        "Real-number arithmetic: rounding, areas, distances, linear rescaling, "
        "floating-point decomposition.",
        [(r"\bfloat", 3), (r"\bround", 3), (r"\barea\b", 4), (r"\btriangle\b", 4),
         (r"nearest integer", 4), (r"linear transform", 4), (r"deviation", 4),
         (r"decimal part", 3), (r"right-angled", 4), (r"\bclosest integer\b", 5),
         (r"rounded to", 3), (r"squared", 2), (r"\bmean\b", 3), (r"\baverage\b", 3)]),
    "sorting_select": (
        "The answer is an order statistic or an ordering: sort by a key, k-th element, "
        "min/max/median, closest pair, top-k.",
        [(r"\bsort", 4), (r"sorted order", 4), (r"\bmedian\b", 5), (r"\bk-th\b", 3),
         (r"\bmaximum\b", 1), (r"\bminimum\b", 1), (r"\blargest\b", 2), (r"\bsmallest\b", 2),
         (r"closest to each other", 4), (r"2nd smallest", 5), (r"n-th", 2),
         (r"ascending order", 3), (r"next odd number", 2), (r"top\b", 1),
         (r"\bunique elements\b", 2)]),
    "counting_histogram": (
        "The answer is a count or a frequency aggregate over a collection.",
        [(r"\bcount\b", 3), (r"how many (times|distinct|characters)", 4),
         (r"number of times", 4), (r"\bfrequenc", 4), (r"\bhistogram\b", 5),
         (r"most repetition", 4), (r"dictionary of", 3), (r"number of elements", 2),
         (r"number of (even|odd|upper|lower)", 3), (r"appears?\b.*\bdivisible", 2),
         (r"number of distinct", 3)]),
    "string_transform": (
        "Character-level work on a string: change case, reverse, replace, remove, "
        "encode/decode, shift, concatenate.",
        [(r"flip\b", 3), (r"\breverse\b", 2), (r"\bremove_vowels\b", 5), (r"\bencod", 3),
         (r"\bdecod", 3), (r"replace .* with", 3), (r"swap case", 3), (r"\bxor\b", 3),
         (r"\bcipher|encrypt", 3), (r"substring", 2), (r"same characters", 4), (r"\bpalinfrome|\bpalindrome", 3),
         (r"words_in_sentence", 3), (r"vowels", 2), (r"\bstring\b", 1),
         (r"upper characters", 2), (r"\bprefixes\b", 2), (r"alphabet", 2)]),
    "collection_transform": (
        "A collection (list, array, dict, nested list) in, and out by a position-wise, "
        "membership or per-element rule: filter, map, dedupe, flatten, index arithmetic. "
        "This is the generic collection class -- it wins when no more specific type fires.",
        [(r"\bfilter\b", 4), (r"return only", 4), (r"\ball numbers in the\b", 3),
         (r"\bdictionary\b", 3), (r"\bpolynomial\b", 3), (r"\bcoefficients\b", 3),
         (r"total number of chars", 3), (r"\blist of\b", 1), (r"\barray\b", 1), (r"\belements?\b", 1),
         (r"remove (all )?elements", 4), (r"\bindex(es)?\b", 2), (r"\bodd indicies|even indicies|indices\b", 3),
         (r"\bconcatenate\b", 3), (r"\bcommon elements\b", 4), (r"\bduplicate", 3),
         (r"\bmonotonic", 4), (r"\bsubset\b", 2), (r"\bmatrix\b", 2), (r"\brow\b", 2),
         (r"\bcolumn\b", 2), (r"non-negative integer nodes", 2), (r"\bnested list", 3),
         (r"\bkeys? (are|is)\b", 3), (r"\bderivative\b", 3)]),
    "scenario_simulation": (
        "A described world or process that has to be modelled; the code simulates it "
        "rather than computing a stated formula.",
        [(r"\bcars?\b", 3), (r"\bwells?\b", 3), (r"\bbucket", 3), (r"\bplanets?\b", 4),
         (r"\bcarrots?\b", 3), (r"\bwill fly\b", 5), (r"\bbank account", 4),
         (r"\bbranch of a tree\b", 4), (r"\bclass name\b", 3), (r"\bextensions\b", 3),
         (r"\bhungry rabbit", 5), (r"\bpile\b", 3), (r"\bbasket of fruit", 5),
         (r"\bexchanged?\b.*\belements", 3), (r"\brotate\b", 2), (r"event is finally known", 5),
         (r"\bboredom", 4), (r"guess", 2), (r"balance of the account", 4)]),
    "arithmetic_basics": (
        "Short numeric/boolean arithmetic that fits none of the above: sums, products, "
        "sign checks, simple predicates.",
        [(r"\badd\b", 2), (r"\bsum\b", 2), (r"\bproduct\b", 2), (r"sum of", 2),
         (r"\bdifference\b", 2), (r"\bmultiply", 3), (r"\bequal to the sum\b", 3),
         (r"less then 100", 2), (r"\bpositive integer n\b", 1), (r"\bnumber x\b", 1),
         (r"\brange \[", 3), (r"even integer", 3), (r"\binclusive\b", 2)]),
}

# Ties are broken by this order: the more specific class wins over the more generic.
PRIORITY = ["bracket_nesting", "string_parse", "dynamic_programming", "number_theory",
            "base_and_bits", "float_geometry", "sorting_select", "counting_histogram",
            "string_transform", "collection_transform", "scenario_simulation", "arithmetic_basics"]


def norm(text):
    return re.sub(r"\s+", " ", (text or "").lower())


def score(text):
    t = norm(text)
    out = {}
    for label, (_, sigs) in TYPES.items():
        s = 0
        hits = []
        for pat, w in sigs:
            n = len(re.findall(pat, t))
            if n:
                s += w * min(n, 3)          # saturate: repetition is not extra evidence
                hits.append(pat)
        if s:
            out[label] = (s, hits)
    return out


def classify(text, top_n=2, close=0.6):
    """Return (primary, secondary_or_None, scores). Secondary is kept only when its score
    is within `close` of the primary's -- a distant second is noise, not a real dual type."""
    sc = score(text)
    if not sc:
        return "unclassified", None, {}
    order = sorted(sc, key=lambda k: (-sc[k][0], PRIORITY.index(k)))
    primary = order[0]
    secondary = None
    if len(order) > 1 and sc[order[1]][0] >= close * sc[primary][0]:
        secondary = order[1]
    return primary, secondary, {k: v[0] for k, v in sc.items()}


def classification_text(problem):
    """HumanEval: the problem statement is the docstring; the skill is also visible in the
    canonical solution, so both go in. The solution is weighted the same as the statement
    because the type is a property of the work, not of the narrative."""
    return problem.get("prompt", "") + "\n" + problem.get("canonical_solution", "")


def load_humaneval(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def run_humaneval(path):
    probs = load_humaneval(path)
    out = []
    for p in probs:
        prim, sec, sc = classify(classification_text(p))
        out.append({"task_id": p["task_id"], "type": prim, "type2": sec,
                    "entry_point": p["entry_point"], "scores": sc})
    return out


def _selftest():
    """Every label must be reachable, and the two known-hard rows must land where a human
    reading puts them. A taxonomy whose rules never fire is a rubric nobody follows."""
    reachable = set()
    cases = [
        ("Input to this function is a string containing multiple groups of nested parentheses.",
         "bracket_nesting"),
        ("fibfib(0) == 0, fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3)",
         "dynamic_programming"),
        ("Return true if a given number is prime, and false otherwise.", "number_theory"),
        ("Change numerical base of input number x to base.", "base_and_bits"),
        ("Given length of a side and high return area for a triangle.", "float_geometry"),
        ("Return median of elements in the list l.", "sorting_select"),
        ("Given a string, find out how many distinct characters does it consist of",
         "counting_histogram"),
        ("flip lowercase characters to uppercase and uppercase to lowercase", "string_transform"),
        ("Filter an input list of strings only for ones that contain given substring",
         "collection_transform"),
        ("You're given a list of deposit and withdrawal operations on a bank account",
         "scenario_simulation"),
        ("Add two numbers x and y", "arithmetic_basics"),
        ("Input to this function is a string representing musical notes in a special ASCII "
         "format.", "string_parse"),
    ]
    for text, want in cases:
        got, sec, _ = classify(text)
        reachable.add(got)
        assert got == want, f"selftest: {text[:50]!r} -> {got}, want {want}"
    missing = set(TYPES) - reachable
    assert not missing, f"selftest: labels no case reaches: {sorted(missing)}"
    # A label with no signals can never fire; catch that structurally, not by example.
    for label, (_, sigs) in TYPES.items():
        assert sigs, f"selftest: {label} has no signals"
    # Order must be a total order over the same key set, or tie-breaks are undefined.
    assert sorted(PRIORITY) == sorted(TYPES), "selftest: PRIORITY and TYPES disagree"
    print(f"selftest OK: {len(TYPES)} labels, all reachable, all with signals")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--humaneval")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.humaneval:
        ap.error("--humaneval or --selftest")
    rows = run_humaneval(a.humaneval)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    c = collections.Counter(r["type"] for r in rows)
    for k, v in c.most_common():
        print(f"{v:4}  {k}")
    print(f"{len(rows):4}  TOTAL")


if __name__ == "__main__":
    main()
