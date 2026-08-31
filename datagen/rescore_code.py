#!/usr/bin/env python3
"""Re-score a code preds file from (gen, expected) alone.

code_zh.py computes ok by executing the code extracted from the FULL generation,
but the preds file used to store only gen[-300:]: a correct-then-degenerate row
scored ok=True while its stored tail showed only the degeneration, and no
re-scorer over the file could reproduce the number (t28). This script is that
re-scorer: it extracts and executes exactly as eval/code_zh.py does, and reports
whether the file's ok count reproduces. Exit 0 on reproduction, 1 otherwise.

    python datagen/rescore_code.py data/eval/preds_code_ckpt_sft_p324_v3.pt.jsonl
    python datagen/rescore_code.py --selftest
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.code_zh import extract_code, score_code  # noqa: E402


def rescore(path):
    """(stats, None) if the file carries what ok was computed from; (None, err) otherwise."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if rows and "expected" not in rows[0]:
        return None, (
            "rows lack 'expected' -- file predates full-gen storage (t28); "
            "ok cannot be reproduced from the truncated gen"
        )
    n = agree = stored_ok = rescored_ok = 0
    for r in rows:
        code = extract_code(r.get("gen", ""))
        ok = False
        if code is not None:
            ok, _, _ = score_code(code, r["expected"])
        n += 1
        stored_ok += int(r.get("ok", False))
        rescored_ok += int(ok)
        agree += int(ok == bool(r.get("ok", False)))
    return {
        "n": n,
        "stored_ok": stored_ok,
        "rescored_ok": rescored_ok,
        "row_agree": agree,
        "reproduces": stored_ok == rescored_ok,
    }, None


def selftest():
    """The guard: a truncated gen must NOT re-score to the same ok. Deliberately
    stores a full gen that scores, truncates it the way code_zh.py used to, and
    asserts the re-score disagrees."""
    import tempfile

    gen = "```python\nprint(2 + 2)\n```\n" + "lorem ipsum " * 100  # correct front, long tail
    expected = "4"
    code = extract_code(gen)
    assert code is not None, "selftest oracle broken: fence not found"
    ok_full, _, _ = score_code(code, expected)
    assert ok_full, "selftest oracle broken: the fenced code should score"
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"gen": gen, "expected": expected, "ok": True}) + "\n")
        path = f.name
    try:
        # truncate the stored gen the way code_zh.py did before t28
        rows = [json.loads(l) for l in open(path)]
        rows[0]["gen"] = rows[0]["gen"][-300:]
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        r, err = rescore(path)
        assert err is None, err
        assert not r["reproduces"], (
            "guard did not fire: a truncated gen re-scored to the same ok -- "
            "the file does not carry what ok was computed from"
        )
        print(f"rescore_code selftest OK (full gen ok={ok_full}; truncated gen disagrees: {r})")
    finally:
        os.unlink(path)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
        return
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    r, err = rescore(sys.argv[1])
    if err:
        print(f"CANNOT RE-SCORE: {err}")
        sys.exit(1)
    print(
        f"n={r['n']} stored_ok={r['stored_ok']} rescored_ok={r['rescored_ok']} "
        f"row_agree={r['row_agree']}/{r['n']} reproduces={r['reproduces']}"
    )
    sys.exit(0 if r["reproduces"] else 1)


if __name__ == "__main__":
    main()
