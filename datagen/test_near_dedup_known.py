#!/usr/bin/env python3
"""3b-8: near-dedup known-answer test (host-side, no GPU, no /data00).

Validates the capture logic in near_dedup_postpass under known answers:
  POSITIVE  a near-dup code pair (renamed identifiers, different comments,
            replaced literals) MUST be captured as near-dup (Jaccard >= 0.5)
  NEGATIVE  two genuinely different functions MUST NOT be captured
Also asserts the normaliser's invariants (numbers->#, ident->placeholder).
Runs on constructed code fixtures, so the sample's text content does not matter.
Exit 0 = all known answers hold; nonzero = a known answer broke.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import near_dedup_postpass as ND  # noqa: E402

# ---- fixtures ----
A = (
    "def bubble_sort(seq):\n"
    "    # sort in place\n"
    "    for i in range(len(seq)):\n"
    "        for j in range(len(seq) - 1):\n"
    "            if seq[j] > seq[j + 1]:\n"
    "                seq[j], seq[j + 1] = seq[j + 1], seq[j]\n"
    "    return seq\n"
)
# near-dup of A: only identifiers + the comment differ; structure is byte-identical
A_PRIME = (
    "def bubble(arr):\n"
    "    # bubblesort\n"
    "    for x in range(len(arr)):\n"
    "        for y in range(len(arr) - 1):\n"
    "            if arr[y] > arr[y + 1]:\n"
    "                arr[y], arr[y + 1] = arr[y + 1], arr[y]\n"
    "    return arr\n"
)
B = (
    "def quick_sort(seq):\n"
    "    if len(seq) <= 1:\n"
    "        return seq\n"
    "    pivot = seq[len(seq) // 2]\n"
    "    left = [x for x in seq if x < pivot]\n"
    "    middle = [x for x in seq if x == pivot]\n"
    "    right = [x for x in seq if x > pivot]\n"
    "    return quick_sort(left) + middle + quick_sort(right)\n"
)
C = (
    "def fib(n):\n"
    "    a, b = 0, 1\n"
    "    for _ in range(n):\n"
    "        a, b = b, a + b\n"
    "    return a\n"
)

THRESHOLD = 0.5


def jac(x, y):
    return ND.jaccard(ND.word_shingles(ND.normalise_code(x)),
                      ND.word_shingles(ND.normalise_code(y)))


def main():
    failures = []

    # normaliser invariants
    na = ND.normalise_code(A)
    if "bubble_sort" in na or "seq" in na:
        failures.append("normaliser kept an identifier (expected placeholder)")
    if any(ch.isdigit() for ch in na):
        failures.append("normaliser kept a literal digit")

    # POSITIVE: A vs A_PRIME is a near-dup and must reach threshold
    s_aa = jac(A, A_PRIME)
    if not (s_aa >= THRESHOLD):
        failures.append(f"A~A' Jaccard {s_aa:.3f} < {THRESHOLD} (near-dup NOT captured)")

    # NEGATIVE: A vs B / A vs C are different and must stay below threshold
    for other, name in ((B, "B"), (C, "C")):
        s = jac(A, other)
        if s >= THRESHOLD:
            failures.append(f"A vs {name} Jaccard {s:.3f} >= {THRESHOLD} (distinct falsely captured)")

    # TRUE normaliser test: byte-identical text with only every identifier renamed.
    # normalise must collapse them to the SAME text -> the rename is a near-dup.
    A_IDENT = A.replace("seq", "items")
    s_ident = jac(A, A_IDENT)
    same_norm = ND.normalise_code(A) == ND.normalise_code(A_IDENT)
    if not (same_norm and s_ident >= THRESHOLD):
        failures.append(
            f"identifier-only rename NOT near-dup: ident_Jaccard={s_ident:.3f} same_norm={same_norm}")

    # cluster-level: A and A_PRIME union together, B and C stay singletons
    docset = [A, A_PRIME, B, C]
    parent = list(range(len(docset)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i in range(len(docset)):
        for j in range(i + 1, len(docset)):
            if jac(docset[i], docset[j]) >= THRESHOLD:
                union(i, j)
    if find(0) != find(1):
        failures.append("cluster: A and A_PRIME should be one cluster")
    if find(0) == find(2) or find(0) == find(3) or find(2) == find(3):
        failures.append("cluster: distinct docs merged")

    json_result = {
        "known_pos_captured": round(s_aa, 3),
        "known_neg_A_B": round(jac(A, B), 3),
        "known_neg_A_C": round(jac(A, C), 3),
        "cluster": {"A_Aprime_union": find(0) == find(1),
                     "singletons_distinct": find(0) != find(2) and find(2) != find(3)},
        "n_failures": len(failures),
        "verdict": "PASS" if not failures else "FAIL",
    }
    import json
    print(json.dumps(json_result, ensure_ascii=False))
    for f in failures:
        print("  FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())