#!/usr/bin/env python3
"""Executable-filter YIELD prototype on code_rp1t (fb P0 2026-09-01). Phi-1's whole
result was a filter that keeps code which actually runs. Over a ~20K doc sample:
  - language distribution (Python detection by source/url extension + content heuristic)
  - of PYTHON docs: fraction that PARSE (ast.parse), and of those, fraction that RUN
    (exec in a subprocess sandbox, stdlib-only, time-limited, non-zero-exit/hang = fail)
  - surviving tokens at each stage

The pipeline runs detached on the pod (code_rp1t shards + no parse/run deps beyond
python3 ast/subprocess). This is a PROTOTYPE over a sample, not the full filter -- the
yield decides whether the 73.6B code budget becomes 30B or 3B."""
import subprocess
import sys
import json
import glob
import os

SAMPLE = 20_000
SB_TIMEOUT = 3  # a run that exceeds this is a hang, not executable; most docs fail fast on import


def detect_lang(content, source):
    s = (source or "").lower()
    for ext, lang in ((".py", "py"), (".pyx", "py"), (".js", "js"), (".ts", "ts"), (".tsx", "ts"),
                      (".java", "java"), (".c", "c"), (".h", "c"), (".cpp", "cpp"), (".cc", "cpp"),
                      (".hpp", "cpp"), (".go", "go"), (".rb", "rb"), (".php", "php"), (".rs", "rs"),
                      (".sh", "sh"), (".html", "html"), (".css", "css"), (".sql", "sql")):
        if s.endswith(ext) or ("/" + ext[1:] + "/") in s:
            return lang
    # python content heuristic: def/import/print at line starts
    head = content[:4000]
    n_def = head.count("\n    def ") + head.count("\ndef ")
    n_import = head.count("import ") + head.count("from ")
    if n_import >= 2 or (n_def >= 1 and n_import >= 1):
        return "py"
    return "other"


def parses_py(code):
    try:
        compile(code, "<doc>", "exec")
        return True
    except SyntaxError:
        return False


def runs_py(code, timeout=SB_TIMEOUT):
    """True if `code` executes to exit 0 in a subprocess sandbox within `timeout`.
    stdlib-only globals, stdout/stderr discarded, resource-limited."""
    # wrap: run in a subprocess so a hang cannot stall the measurement
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def main():
    try:
        from tokenizers import Tokenizer

        tk = Tokenizer.from_file("/work/aupai/data/tokenizer.json")
    except Exception:
        tk = None
    shards = sorted(glob.glob("/work/aupai/data/corpus/code_rp1t/*.jsonl"))
    lang = {}
    py_docs = []
    sampled = 0
    for shard in shards:
        for line in open(shard, encoding="utf-8"):
            if sampled >= SAMPLE:
                break
            d = json.loads(line)
            c = d.get("content") or ""
            if not c:
                continue
            l = detect_lang(c, d.get("source"))
            lang[l] = lang.get(l, 0) + 1
            if l == "py" and len(c) < 200_000:  # skip pathological giants for the run probe
                py_docs.append(c)
            sampled += 1
        if sampled >= SAMPLE:
            break
    n_py = len(py_docs)
    ok_parse = [c for c in py_docs if parses_py(c)]
    ok_run = [c for c in ok_parse if runs_py(c)]
    surviving_chars = sum(len(c) for c in ok_run)
    surviving_tokens = sum(len(tk.encode(c).ids) for c in ok_run) if tk else None
    out = {
        "sample": sampled,
        "language_dist": {k: lang[k] for k in sorted(lang)},
        "python_docs": n_py,
        "parse_ok": len(ok_parse),
        "parse_yield_of_python": (len(ok_parse) / n_py) if n_py else 0.0,
        "run_ok": len(ok_run),
        "run_yield_of_python": (len(ok_run) / n_py) if n_py else 0.0,
        "run_yield_of_parsed": (len(ok_run) / len(ok_parse)) if ok_parse else 0.0,
        "surviving_chars_py": surviving_chars,
        "surviving_tokens_py": surviving_tokens,
        "tokenizer": "data/tokenizer.json" if tk else "none (chars only)",
        "run_config": {"sandbox": "subprocess python3 -c, stdlib-only, exit0=run",
                       "timeout_s": SB_TIMEOUT},
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()