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


def build_prompt(text: str, rubric: dict) -> str:
    """Return the user prompt. The model must answer with ONLY a JSON object; the
    parser's job is to enforce shape, not to salvage prose."""
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
        "Document:",
        text[:6000],
    ]
    return "\n".join(lines)


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
