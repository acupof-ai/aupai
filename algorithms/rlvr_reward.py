#!/usr/bin/env python3
"""Verifiable \\boxed{} reward for Chinese math RLVR.

Pure stdlib — importable without torch/GPU. Comparison is type-split on the
gold answer: integers and fractions compare exactly, decimals get an absolute
tolerance (math answers are exact quantities, not measurements — the FP
window must not grow with magnitude). Unbalanced \\boxed{} is refused, not
truncated-tolerated: bad GTs are rejected at the data boundary
(rlvr_data.load_problems), where a refusal is loud.
"""

import re
from fractions import Fraction


def extract_boxed(text):
    """Extract content of the LAST \\boxed{...} with balanced braces.
    Unbalanced -> None (truncation is a data error, refused at load)."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None  # unbalanced — refuse, do not take the rest


def normalize_answer(s):
    """Normalize a LaTeX answer for comparison."""
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)  # \frac{a}{b}
    s = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\{?([^{}\\]*)", r"\1", s)
    s = s.replace("\\", "").replace("{", "").replace("}", "").replace("$", "")
    s = re.sub(r"[\s,，、]+", "", s)
    s = s.rstrip("。.,，")
    s = re.sub(
        r"(?:平方厘米|平方米|立方米|平方千米|厘米|分米|毫米|千米|小时|分钟|米|元|角|分|个|只|条|本|"
        r"把|张|件|台|辆|座|道|题|岁|天|年|月|点|时|人|次|棵|支|双|块|名|位|组|班|排|筐|盒|箱|"
        r"袋|包|瓶|杯|碗|盘|锅|桶|盆|柜|桌|椅|床|门|窗|墙|地|周|日|度|%)+$",
        "",
        s,
    )
    return s or None


def _typed(s):
    """Parse a normalized answer into (kind, value):
    'i' int, 'f' Fraction, 'd' float, 's' str. None if empty."""
    if not s:
        return None
    if re.fullmatch(r"-?\d+", s):
        return ("i", int(s))
    m = re.fullmatch(r"\(?(-?\d+)\)?/\(?(-?\d+)\)?", s)
    if m and int(m.group(2)) != 0:
        return ("f", Fraction(int(m.group(1)), int(m.group(2))))
    try:
        return ("d", float(s))
    except ValueError:
        return ("s", s)


def to_number(s):
    """Parse a normalized answer to float; None if not numeric.
    Kept for build_math's numeric-answer filter (Fraction -> float)."""
    t = _typed(s)
    if t is None or t[0] == "s":
        return None
    return float(t[1])


def reward_fn(gen_text, gt_raw):
    """0/1 verifiable reward: extract \\boxed{} from the response, normalize,
    then compare type-split — integer/fraction gold exactly, decimal gold
    within an absolute 1e-4 tolerance."""
    pred = extract_boxed(gen_text)
    if pred is None:
        return 0.0
    p, g = normalize_answer(pred), normalize_answer(gt_raw)
    if p is None or g is None:
        return 0.0
    if p == g:
        return 1.0
    pt, gt = _typed(p), _typed(g)
    if pt is None or gt is None or "s" in (pt[0], gt[0]):
        return 0.0
    if gt[0] == "d":  # decimal gold: absolute tolerance, magnitude-independent
        return 1.0 if abs(pt[1] - gt[1]) <= 1e-4 else 0.0
    return 1.0 if pt[1] == gt[1] else 0.0  # int/fraction gold: exact
