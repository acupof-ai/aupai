#!/usr/bin/env python3
"""Verifiable \\boxed{} reward for Chinese math RLVR.

Pure stdlib — importable without torch/GPU. Tolerates the truncated GT
strings in rlvr_math.jsonl (e.g. '\\dfrac{10', '23 \\text{点').
"""

import re


def extract_boxed(text):
    """Extract content of the LAST \\boxed{...} with balanced braces."""
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
    return text[j + 1 :]  # unbalanced (truncated) — take the rest


def normalize_answer(s):
    """Normalize a LaTeX answer for comparison."""
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)  # \frac{a}{b}
    s = re.sub(r"\\[dt]?frac\{([^{}]*)", r"\1", s)  # truncated \frac{a
    s = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\{?([^{}\\]*)", r"\1", s)
    s = s.replace("\\", "").replace("{", "").replace("}", "").replace("$", "")
    s = re.sub(r"[\s,，、]+", "", s)
    s = s.rstrip("。.,，")
    s = re.sub(
        r"(?:平方厘米|平方米|立方米|平方千米|厘米|分米|毫米|千米|小时|分钟|米|元|角|分|个|只|条|本|"
        r"把|张|件|台|辆|座|道|题|岁|天|年|月|点|时|人|次|棵|支|双|对|块|名|位|组|班|排|筐|盒|箱|"
        r"袋|包|瓶|杯|碗|盘|锅|桶|盆|柜|桌|椅|床|门|窗|墙|地|周|日|度|%)+$",
        "",
        s,
    )
    return s or None


def to_number(s):
    """Parse float or a/b fraction; None if not numeric."""
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def reward_fn(gen_text, gt_raw):
    """0/1 verifiable reward: extract \\boxed{} from the response, normalize,
    exact or numeric (1e-4 relative tolerance) match against ground truth."""
    pred = extract_boxed(gen_text)
    if pred is None:
        return 0.0
    p, g = normalize_answer(pred), normalize_answer(gt_raw)
    if p is None or g is None:
        return 0.0
    if p == g:
        return 1.0
    pn, gn = to_number(p), to_number(g)
    if pn is not None and gn is not None and abs(pn - gn) <= 1e-4 * max(1.0, abs(gn)):
        return 1.0
    return 0.0
