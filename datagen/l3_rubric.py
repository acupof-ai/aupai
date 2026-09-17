"""Multi-dimensional teacher rubrics and loud-failure parsing for the L3 label funnel.

Why not one quality score: docs/lessons/data_quality_methods.md — a single scalar
drifts with style/register and is untraceable. Nemotron-CC style: independent
dimensions, scored separately, kept separate. Two rubrics because code exercises
and natural-language prose answer different value questions.

Each dimension is an integer 1..5 judged INDEPENDENTLY. The parser REFUSES (raises)
on a missing dimension, a non-integer, or an out-of-range value rather than coercing
- the project's repeated failure class is a silent garbage label (harness
loud-failure convention; facts_well_formed-style hard validation).
"""

import json
import re

RUBRIC_VERSION = "l3rubric-v1.0"

# Each dimension: key, one-line judging instruction, and the 1/5 anchors. Scores are
# independent: a correct but trivial answer can be high on factual, low on complexity.
CODE_RUBRIC = {
    "kind": "code",
    "dimensions": {
        "content_quality": (
            "How clean, complete, and well-structured is the code/problem statement? "
            "1 = broken/truncated/unreadable; 5 = production-quality, clear, self-contained."
        ),
        "factual_correctness": (
            "Would a correct implementation actually solve the stated problem, with no "
            "wrong API usage or logical error? 1 = clearly wrong; 5 = correct on inspection."
        ),
        "complexity": (
            "Technical/depth level worth a model's attention (algorithms, edge cases, "
            "multi-step reasoning). 1 = trivial getter/hello-world; 5 = genuinely hard."
        ),
        "educational_or_code_value": (
            "Value as pretraining/SFT material for coding ability (reusable idioms, "
            "correct patterns, instructive tests). 1 = none; 5 = highly valuable."
        ),
    },
}

NL_RUBRIC = {
    "kind": "natural_language",
    "dimensions": {
        "content_quality": (
            "Coherence, fluency, organization, non-boilerplate. 1 = spam/SEO/gibberish; "
            "5 = coherent, well-written substantive prose."
        ),
        "factual_correctness": (
            "Are verifiable claims true and internally consistent? 1 = false/misleading; "
            "5 = accurate (mark 3 for unverifiable/opinion without checkable claims)."
        ),
        "complexity": (
            "Information density and reasoning depth. 1 = thin/repetitive; 5 = dense, nuanced, multi-idea."
        ),
        "educational_or_code_value": (
            "General educational/explanatory value for building language/reasoning "
            "(not necessarily code). 1 = none; 5 = strongly instructive/reference-worthy."
        ),
    },
}

_DIMS = ("content_quality", "factual_correctness", "complexity", "educational_or_code_value")
MIN_SCORE, MAX_SCORE = 1, 5

# A teacher prompt cannot carry an unbounded chunk. Inputs longer than this are scored on
# only the first TRUNC_CHARS characters; the label is then bound to that prefix and the
# caller must mark the ledger row truncated=True so a whole-chunk trainer excludes it.
TRUNC_CHARS = 6000


def _is_code_content(text: str) -> bool:
    """Cheap routing between the two rubrics without a model. L3 code rows are
    problem statements that embed code; a dominant code-fence/def/import signal or a
    Python prompt 'Write a ... function' picks the code rubric. Conservative: only
    strong signals route to code, otherwise natural_language."""
    t = text
    code_marks = (
        t.count("```")
        + t.count("\ndef ")
        + t.count("\nclass ")
        + t.count("import ")
        + t.count("return ")
        + t.count("    ")
    )
    asks_code = bool(re.search(r"\bwrite\b.{0,60}\b(function|class|program|script)\b", t[:400], re.I | re.S))
    return asks_code or code_marks >= 4


def select_rubric(text: str) -> dict:
    return CODE_RUBRIC if _is_code_content(text) else NL_RUBRIC


def build_prompt(text: str, rubric: dict):
    """Return (prompt, truncated). The model must answer with ONLY a JSON object; the
    parser's job is to enforce shape, not to salvage prose.

    A document longer than TRUNC_CHARS is scored on only its first TRUNC_CHARS characters.
    Such a label covers a prefix, not the whole chunk, so truncated is returned True and an
    explicit notice is put in the prompt itself: the teacher is told exactly what it saw and
    that the scores apply only to that part. Callers thread the bool onto the ledger row."""
    truncated = len(text) > TRUNC_CHARS
    body = text[:TRUNC_CHARS]
    lines = [
        "You are labeling data for a coding-model pretraining corpus.",
        f"Rubric kind: {rubric['kind']}. Score FOUR dimensions INDEPENDENTLY, each an "
        f"integer {MIN_SCORE}-{MAX_SCORE}. Do not let one score influence another.",
        "",
        "Dimensions:",
    ]
    for i, dim in enumerate(_DIMS, 1):
        lines.append(f"{i}. {dim}: {rubric['dimensions'][dim]}")
    lines += [
        "",
        "Respond with ONLY a JSON object on one line, no prose, no code fence:",
        json.dumps({d: f"<{MIN_SCORE}-{MAX_SCORE} int>" for d in _DIMS}),
        "",
    ]
    if truncated:
        lines += [
            f"NOTE: the document below is longer than {TRUNC_CHARS} characters; only its "
            f"FIRST {TRUNC_CHARS} characters are shown. Score ONLY the shown prefix and do "
            "not infer anything about the unseen remainder.",
            "",
        ]
    lines += ["Document:", body]
    return "\n".join(lines), truncated


def parse_scores(raw: str, rubric: dict) -> dict:
    """Parse the teacher's reply into {dim:int}. Loud failure: the first malformed or
    out-of-range/missing dimension raises with the raw tail, so a bad teacher never
    silently becomes a training label."""
    if not raw or not isinstance(raw, str):
        raise ValueError("empty teacher response")
    s = raw.strip()
    # tolerate a lone ```json fence but nothing else; extract the first {...} block.
    fence = re.search(r"\{.*\}", s, re.S)
    if not fence:
        raise ValueError(f"no JSON object in response: {s[:200]!r}")
    try:
        obj = json.loads(fence.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"response JSON invalid ({e}): {s[:200]!r}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"response is not a JSON object: {s[:200]!r}")
    out = {}
    for dim in _DIMS:
        if dim not in obj:
            raise ValueError(f"missing dimension {dim!r}: {s[:200]!r}")
        v = obj[dim]
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"dimension {dim!r} not an int ({v!r}): {s[:200]!r}")
        if not (MIN_SCORE <= v <= MAX_SCORE):
            raise ValueError(f"dimension {dim!r} out of range {v}: {s[:200]!r}")
        out[dim] = v
    extra = set(obj) - set(_DIMS)
    if extra:
        raise ValueError(f"unexpected extra keys {sorted(extra)}: {s[:200]!r}")
    return out


def _selftest() -> int:
    # Known answers for the two funnel contracts fb hardened 2026-09-17:
    #  A. truncation: build_prompt flags a >TRUNC_CHARS document and a prefix-only document
    #     EXACTLY at the boundary, in both rubric kinds; the prompt tells the teacher what it
    #     saw. A whole-chunk trainer excludes truncated rows, so a missed flag silently feeds a
    #     prefix label as if it covered the whole chunk.
    #  B. parse_scores refuses every malformed teacher reply (a silent garbage label is the
    #     funnel's repeated failure class): missing dim, non-int, bool, out-of-range, extra
    #     key, no-JSON, empty.
    short = "x" * TRUNC_CHARS
    over = "x" * (TRUNC_CHARS + 1)

    def trunc_flag(text, rubric):
        _prompt, truncated = build_prompt(text, rubric)
        return truncated

    # exact boundary, both rubric kinds: == limit is whole, limit+1 is a prefix
    for rub in (CODE_RUBRIC, NL_RUBRIC):
        assert trunc_flag(short, rub) is False, f"{rub['kind']}: at-limit must be whole"
        assert trunc_flag(over, rub) is True, f"{rub['kind']}: over-limit must flag prefix"
        p_full, _ = build_prompt(short, rub)
        p_tr, t_tr = build_prompt(over, rub)
        assert t_tr is True and "FIRST" in p_tr and "longer than" in p_tr, \
            f"{rub['kind']}: truncated prompt must tell the teacher it saw only a prefix"
        assert "FIRST" not in p_full, f"{rub['kind']}: whole doc must not carry the prefix notice"
        # the shown body is capped at TRUNC_CHARS (the long doc's tail is never sent)
        assert p_tr.endswith("x" * 400), "prompt body is the capped prefix"

    valid = json.dumps({d: 3 for d in _DIMS})
    good = parse_scores(valid, NL_RUBRIC)
    assert good == {d: 3 for d in _DIMS} and set(good) == set(_DIMS)
    # a lone ```json fence is tolerated, nothing else
    assert parse_scores("```json\n" + valid + "\n```", NL_RUBRIC) == good

    def must_raise(label, raw):
        try:
            parse_scores(raw, NL_RUBRIC)
        except ValueError:
            return
        raise AssertionError(f"{label}: malformed teacher reply must refuse, got accepted")

    must_raise("empty", "")
    must_raise("no-json", "the answer is unclear")
    for dim in _DIMS:
        must_raise(f"missing {dim}", json.dumps({d: 3 for d in _DIMS if d != dim}))
        must_raise(f"non-int {dim}", json.dumps({**good, dim: 3.0}))
        must_raise(f"bool {dim}", json.dumps({**good, dim: True}))
        must_raise(f"string {dim}", json.dumps({**good, dim: "3"}))
        must_raise(f"zero {dim}", json.dumps({**good, dim: 0}))
        must_raise(f"six {dim}", json.dumps({**good, dim: 6}))
    must_raise("extra-key", json.dumps({**good, "note": "x"}))
    must_raise("bad-json", '{"content_quality": 3,')

    # rubric routing: a code-marked problem statement selects the code rubric, plain prose NL
    code_stmt = "def solve(x):\n    return x\n\nimport os\nclass A:\n    pass\n"
    assert select_rubric(code_stmt)["kind"] == "code", "strong code signal must route to code"
    assert select_rubric("an ordinary paragraph about history with no code marks")["kind"] \
        == "natural_language"
    print("selftest ok: truncation flags at TRUNC_CHARS both rubrics with prefix notice; "
          "parse_scores accepts the one valid shape and refuses missing/non-int/bool/"
          "out-of-range/extra/no-json/empty; rubric routing code vs NL")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
