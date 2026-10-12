#!/usr/bin/env python3
"""E5: basic code, the small-sample gate (NOT the full HumanEval benchmark).

Contract: docs/standards/eval_ladder_v1_1009.md section 5, from the v42 Engram
handover doc section 6, E5 row:

    syntax compiles; function signature correct; unit test runs; HumanEval small
    sample; no irrelevant explanation and no secret-style text; base uses
    continuation, SFT uses ChatML.
    Gate: syntax, unit test and small-sample accuracy pass at the same time.

Scope. This is a fixed small sample (default N=8, hard cap 20) of
`data/eval/humaneval/humaneval_164.jsonl`, the first N task ids after sorting by
the numeric part of `HumanEval/<k>`. Never random, never sharded, never the full
164: the full denominators are E6's (`eval/ladder/e6_bench.py`, owner w3), and a
stage that silently grew to 164 would produce a number nobody can compare. The
sample size, the task ids and the file's sha256 go into the seal and the summary
so a later run is comparable.

Subject. /work/aupai/ckpt_v42s2.pt, an un-SFT Engram base. The protocol is base
continuation: the model is handed the HumanEval prompt verbatim, which ends at
the closing docstring, and continues the body. No ChatML prefix is ever built
here -- `<|im_start|>` occurs 0 times in the pretraining corpus (AGENTS.md, Chat
format), so a ChatML prefix on a base checkpoint measures the response to an
unseen prefix (facts/base_eval.json#be.l1_fewshot_p324: +38.2pt of pure format
effect). SFT checkpoints get ChatML from E2; E5 grades base continuations only,
and E6 is where a ChatML arm belongs.

The six judgments, each with its own fail_class (the negative controls in
control_selftest each land on the one they are built to trip):

  G1 syntax        `ast.parse` of the code the model produced. Not a text
                   pattern, not "does it look like code": `parse_module()`. A
                   completion that does not parse can still LOOK right, and a
                   HumanEval continuation is a function body, so the parse is
                   done on a file where the prompt's def header is restored
                   (PROTOCOL NOTE below), never on the bare text.
  G2 signature     a module-level `def <entry_point>` must exist and its
                   parameter shape (names, order, defaults, *args/**kwargs) must
                   equal the prompt's own def. A body-only continuation inherits
                   the prompt's signature and passes by construction; an answer
                   that re-declares the function must re-declare it correctly.
                   Parameter names are reported but are NOT part of the gate:
                   HumanEval tests call positionally, so a rename cannot change
                   the score and gating on it would grade style.
  G3 unit test     the task's own asserts, executed with the candidate function
                   under them, in a CHILD PROCESS (`subprocess`, `-I`), with a
                   wall timeout, an RLIMIT_CPU backstop, RLIMIT_AS, RLIMIT_FSIZE,
                   RLIMIT_NPROC and a default SIGALRM. Timeout and crash are both
                   FAIL (`timeout`, `crash`) -- an infinite loop must not hang the
                   evaluation, and a segfault is not a pass. Being a child process
                   is the point: generated code that calls `sys.exit`,
                   `os._exit`, `signal.alarm` or `faulthandler` cannot take the
                   evaluator with it, and the evaluator's own exit code stays a
                   statement about the evaluator.
  G4 explanation   no irrelevant long explanation. Three sub-rules, all on the
                   GENERATED segment only (the prompt's docstring is not the
                   model's text): prose-line share <= 0.40 of non-blank lines,
                   preamble before the definition <= 2 lines / 200 chars, and no
                   32x-repeated consecutive-line run. A prose line is one that no
                   parse arm accepts -- not a comment, not inside an open bracket,
                   not a block header, and not a dedented statement (`_is_code_line`).
                   "cannot compile" is the criterion, not a keyword list.
  G5 secret-style  no secret-shaped text: base64/hex runs >= 40 chars above an
                   entropy floor, or a key prefix (sk-, ghp_, AKIA, xox, PEM,
                   JWT). Handover doc section 4.1: the v6 smoke generated
                   secret-style text. Canary literals are exempt only for the
                   samples this module itself plants (see CANARIES), so the
                   detector can be driven on a known answer without a detector
                   that skips whatever it is shown. An exemption keyed on the
                   TASK's own canonical solution/test was considered and
                   rejected as the first rule: it waves through a secret landing
                   next to a test vector.
  G6 termination   sampled long enough to tell a stopped generation from a capped
                   one, and the empty-completion rate is reported. This is a
                   reported metric, not a gate: at a 512-token cap a legitimately
                   long answer and a runaway look the same, so gating on it would
                   charge a real answer to a stop-behaviour defect. E1 owns
                   stop behaviour; it is recorded here because a "0/N" is
                   uninterpretable without it.

Fail classes, one per judgment, plus the empty case: EMPTY_RESPONSE (the model
produced nothing, which the handover doc lists as a symptom of its own),
SYNTAX_ERROR, SIGNATURE_MISMATCH, UNIT_TEST_FAIL, UNIT_TEST_TIMEOUT,
UNIT_TEST_CRASH, EXPLANATION, SECRET_STYLE. An item can carry several; the
summary's `fail_class` field is the first of them and the histogram counts all.

Gate: gates = {syntax, unit, sample} with

    syntax_ok    = G1
    unit_ok      = passed_count / n >= 1/3
    sample_pass  = passed_clean / n >= 0.25   (passed_clean: the whole test suite
                   passed with no irrelevant-explanation and no secret-style text
                   on the same item)

Both rates are stated as fractions in the summary; the denominators are in the
summary and the seal, so a rate can be recomputed. The passed_count/unit gate and
the passed_clean/sample gate are deliberately two numbers: an item whose asserts
pass but whose text also carries a key is counted by the first and not the second.

PROTOCOL NOTE (the one place a naive rewrite breaks the gate). HumanEval
`canonical_solution` is the INDENTED BODY after the prompt's docstring, so
`ast.parse(body)` raises IndentationError for all 164 tasks (measured). The
completion is therefore stitched back under the prompt's own def header before
G1 parses it: from the prompt's completed signature to the last triple quote is
removed, everything before that is kept, and the body is appended. That yields
`from typing import List` + the def header + the body, which is exactly the file
the canonical solution is. The prefix matters: without it an annotated `-> bool`
definition parses fine but the body's `List[...]` name does not resolve at
runtime -- and the same completion text is what G3 executes.

Model output gets the same treatment as the canonical solution, on purpose: a
by-name answer (the model re-declares `def has_close_elements(...)`) is extracted
and judged against the prompt's module prefix plus the task's test, which is
`humaneval_gen.judge_by_name`'s protocol, not a new one.

Evidence -- this module is not the metric of record for HumanEval; it is the
ladder's gate. The baseline it must be read against is in the handover doc
section 4.2: Engram step54000 44/156 clean = 28.21%, SFT v3 47/156 = 30.13% on
the full 164 with rstrip-nl base continuation.

Usage:
    CUDA_VISIBLE_DEVICES="" python3 eval/eval_ladder.py selftest --only e5
    python3 eval/eval_ladder.py e5 --ckpt ckpt_v42s2.pt --device cuda:2 --n 8
    python3 eval/eval_ladder.py e5 --ckpt ckpt_v42s2.pt --device cuda:2 --control
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _repo_root():
    """The repository root, found by looking for a marker rather than by depth.

    Stage modules live at `eval/ladder/<name>.py`, so `parents[2]` is the root
    when this file sits in a checkout. A copy staged elsewhere (a scratch dir, a
    second worktree) would silently resolve `data/...` against the wrong tree and
    refuse a sample that exists, so the marker wins: the first ancestor holding
    both `eval/ladder_common.py` and `data` is the root.
    """
    for cand in pathlib.Path(__file__).resolve().parents:
        if (cand / "eval" / "ladder_common.py").is_file() and (cand / "data").is_dir():
            return cand
    return ROOT


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from eval import ladder_common as lc  # noqa: E402

#: The HumanEval source of record. Pod path from the handover doc section 3.3
#: (`/work/aupai/data/eval/humaneval/humaneval_164.jsonl`). Read, never written.
DATA_DEFAULT = str(ROOT / "data" / "eval" / "humaneval" / "humaneval_164.jsonl")

MAX_SAMPLE = 20          # the contract's cap: this is a small-sample gate, not E6
DEFAULT_N = 8
DEFAULT_MAX_NEW = 512
TASK_TIMEOUT_S = 6.0     # per item, generation-test execution
RUN_MEM_MB = 2048        # RLIMIT_AS for a scored candidate
RUN_NPROC = 256

#: Explanation thresholds (G4). Fractions, all on the generated segment.
PROSE_SHARE_MAX = 0.40
PREAMBLE_LINES_MAX = 2
PREAMBLE_CHARS_MAX = 200
REPEAT_RUN_MAX = 32

#: Secret-style thresholds (G5).
SECRET_MIN_LEN = 40      # 32 would flag HumanEval/162's md5 test vectors
B64_ENTROPY_MIN = 4.0
HEX_ENTROPY_MIN = 3.0
#: Secret-shaped literals this module plants in the negative controls. NOT exempt
#: from the detector -- these are the secrets the controls require to be caught.
CTRL_SECRET_B64 = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2"
CTRL_SECRET_HEX = "3f786850e387550fdab836ed7e6dc881de23001b1c2d3e4f"
#: The exemption whitelist: the literal(s) `secret_report` waves through, so this
#: module's own control can carry a known secret-shaped value without failing the
#: gate it is testing. The exemption is an EXACT match on the matched candidate,
#: never a prefix -- a prefix exemption would hide the canary with one character
#: appended, and control_selftest asserts that direction. Deliberately different
#: strings from CTRL_SECRET_*, so no negative-control secret is waved through.
CANARIES = ("TG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQgY29uc2VjdGV0dXIgYWRpcGlzY2luZyBlbGl0",
            "deadbeefcafebabe0123456789abcdef00112233445566")
BASE64_RE = re.compile(rf"[A-Za-z0-9+/]{{{SECRET_MIN_LEN},}}={{0,2}}")
HEX_RE = re.compile(rf"\b[0-9a-fA-F]{{{SECRET_MIN_LEN},}}\b")
KEY_PREFIX_RES = tuple(re.compile(p) for p in (
    r"sk-[A-Za-z0-9_\-]{16,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"eyJ[A-Za-z0-9_\-]{20,}\.",
))

#: The negative-control completions, at module scope so the selftest and any
#: later reader see the same strings. Only used by --control / control_selftest.
NEG_CONSTANT_FALSE = "    return False\n"
NEG_SYNTAX = "    for idx, elem in enumerate(numbers)\n        return True\n"
NEG_PROSE = ("This function is a classic O(n^2) pairwise comparison problem. "
             "The idea is to read every element, compare it against every other "
             "element, and return True as soon as one pair is closer than the "
             "threshold. Note that the comparison must skip the identity pair, "
             "because an element is not closer to itself than any positive "
             "threshold. A production implementation would sort first, then "
             "compare adjacent elements, which is O(n log n) and easier to "
             "reason about.\n")
NEG_SECRET = (f'    api_key = "{CTRL_SECRET_B64}"\n'
              "    return len(numbers) > 0\n")
#: A valid by-name answer that also carries a key literal. Isolates the secret
#: judgment from the signature judgment: the definition is correct, so a
#: signature failure here would mean the two judgments are conflated.
NEG_SECRET_CODE = ('def has_close_elements(numbers, threshold):\n'
                   f'    token = "{CTRL_SECRET_HEX}"\n'
                   '    for idx, elem in enumerate(numbers):\n'
                   '        for idx2, elem2 in enumerate(numbers):\n'
                   '            if idx != idx2 and abs(elem - elem2) < threshold:\n'
                   '                return True\n'
                   '    return False\n')
NEG_SIGNATURE = ("def has_close_element(numbers):\n"
                 "    return False\n")
#: Right function name, wrong parameter count: the gate is the shape, not the name.
NEG_SIGNATURE_PARAMS = ("def has_close_elements(numbers, threshold, extra=None):\n"
                        "    return False\n")
NEG_INFINITE = ("    while True:\n"
                "        pass\n")


class Task:
    """One HumanEval sample: the four fields E5 actually reads."""

    __slots__ = ("task_id", "prompt", "canonical_solution", "entry_point", "test")

    def __init__(self, row):
        self.task_id = str(row.get("task_id", "")).strip()
        self.prompt = row.get("prompt", "") or ""
        self.canonical_solution = row.get("canonical_solution", "") or ""
        self.entry_point = str(row.get("entry_point", "")).strip()
        self.test = row.get("test", "") or ""


def item_id(task_id):
    """Sort key: the numeric part of HumanEval/<k>, so /9 precedes /10.

    Sorting the raw string puts /10 before /9 and would silently change which
    tasks the first N are. A task_id without a slash sorts last and is refused by
    load_tasks rather than quietly ordered.
    """
    tail = str(task_id).rsplit("/", 1)[-1]
    return (0, int(tail), "") if tail.isdigit() else (1, 0, str(task_id))


def load_tasks(path, n):
    """The first N tasks by numeric id, from the fixed file. Never random.

    The file is not carried in this repository: it is an eval set, and a
    committed copy would be one `git add` away from the pretraining corpus. The
    pod has it at the handover doc's path; a missing file is a REFUSE naming the
    path, never an empty sample scored as 0/N.
    """
    lc.require(1 <= n <= MAX_SAMPLE,
               f"--n {n} outside 1..{MAX_SAMPLE}; E5 is the small-sample gate and "
               f"the full 164 belongs to E6")
    p = pathlib.Path(path)
    lc.require(p.is_file(), f"HumanEval file not found: {p} (pod: "
                            f"/work/aupai/data/eval/humaneval/humaneval_164.jsonl)")
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    lc.require(bool(rows), f"{p} has no rows")
    tasks = [Task(r) for r in rows]
    bad = [t.task_id for t in tasks if not t.task_id or not t.entry_point or not t.test]
    lc.require(not bad, f"rows missing task_id/entry_point/test: {bad[:5]}")
    ids = [t.task_id for t in tasks]
    lc.require(len(set(ids)) == len(ids), f"duplicate task ids in {p}")
    tasks.sort(key=lambda t: item_id(t.task_id))
    out = tasks[:n]
    lc.require(len(out) == n, f"{p} holds {len(tasks)} tasks, asked for {n}")
    return out


def count_tasks(path):
    """Rows in the source file, for the summary's denominator provenance."""
    p = pathlib.Path(path)
    lc.require(p.is_file(), f"HumanEval file not found: {p}")
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def ckpt_identity(ckpt):
    """step and vocab_id from the CHECKPOINT FILE, the way E0 reads them.

    `ladder_common.load_for_eval` reports the step through `getattr(cfg, "step", 0)`,
    and this checkpoint stores its step at the top level of the checkpoint dict with
    no `cfg.step` -- so that field reads 0 for a step-54051 checkpoint (measured
    2026-10-09, E5's first live run: the summary said step 0). A summary that names
    the wrong step is worse than one that names none, so the step is read here from
    the file and its source is recorded beside it.

    The cfg fallback reads a cfg that IS a dict, through `.get()`. It does not fall
    back to an object's `__dict__`: harness check `getattr_cfg_names_exist` refuses
    `getattr(cfg, "<name>", <non-None>)` because a missing name returns the default
    as if it were a measurement, and this checkpoint needs no such arm.
    """
    import torch

    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    step, source = None, "absent"
    for key in ("step", "total_steps"):
        if ck.get(key) is not None:
            step, source = int(ck[key]), f"checkpoint['{key}']"
            break
    if step is None:
        cfg = ck.get("cfg")
        if isinstance(cfg, dict):
            for key in ("step", "total_steps"):
                if cfg.get(key) is not None:
                    step, source = int(cfg[key]), f"checkpoint cfg['{key}']"
                    break
    return {"step": step, "step_source": source, "ckpt_vocab_id": ck.get("vocab_id")}


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# G1/G2: parsing the produced code
# ---------------------------------------------------------------------------

def complete_source(prompt, completion):
    """Prompt + continuation as one compilable file.

    A completion that opens its own `def` is appended to the prompt's imports
    only; a body continuation is stitched under the prompt's def header. The two
    cases are decided by parsing, not by a regex guessing intent.
    """
    comp = completion or ""
    try:
        tree = ast.parse(comp)
    except SyntaxError:
        tree = None
    if tree is not None and any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body):
        # by-name answer: keep the prompt's module prefix and drop its partial def,
        # which is what humaneval_gen.judge_by_name does.
        m = re.search(r"(?m)^def\s+\w+\s*\(", prompt)
        return prompt[: m.start()] if m else "", comp
    header = _completed_def(prompt)
    return header, comp


def _completed_def(prompt):
    """The prompt with the docstring removed: module imports plus the def header.

    HumanEval prompt = imports + `def f(...) -> T:` + indented docstring. Removing
    from the def to the last triple quote leaves exactly the header, so appending
    an indented body yields a compilable function. Returns the prompt unchanged
    when it has no docstring or no def, which keeps a bare-`def` prompt working.
    """
    m = re.search(r"(?m)^def\s+\w+\s*\(", prompt)
    if not m:
        return prompt
    quotes = [i for i in (prompt.rfind('"""'), prompt.rfind("'''")) if i > m.start()]
    if not quotes:
        return prompt
    end = max(quotes) + 3
    if prompt[: m.start()].strip() == "" and end >= len(prompt.rstrip()):
        return prompt
    return prompt[:end] + ("\n" if not prompt[:end].endswith("\n") else "")


def parse_module(src):
    """(tree, None) or (None, SyntaxError). The single definition of G1."""
    try:
        return ast.parse(src), None
    except SyntaxError as exc:
        return None, exc


def find_definition(tree, entry_point):
    """The module-level def named entry_point, or None. Nested defs do not count."""
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return node
    return None


def signature_of(node):
    """Runtime-relevant parameter shape: names, order, defaults, *args/**kwargs.

    Annotations are excluded: they do not change what a positional call binds.
    """
    args = node.args
    positional = [a.arg for a in list(args.posonlyargs) + list(args.args)]
    ndef = len(args.defaults)
    params = [(name, i >= len(positional) - ndef) for i, name in enumerate(positional)]
    return {
        "name": node.name,
        "params": params,
        "vararg": args.vararg.arg if args.vararg else None,
        "kwarg": args.kwarg.arg if args.kwarg else None,
        "kwonly": [a.arg for a in args.kwonlyargs],
    }


def signature_from_prompt(prompt, entry_point=None):
    """The prompt's own def, as the expected shape. None if the prompt is bare.

    Looked up BY NAME, not by position: a HumanEval prompt can carry a helper def
    before the one to complete (HumanEval/10 has is_palindrome, /32 has poly, /38
    has encode_cyclic). Taking the first def compared the answer against the
    helper and reported SIGNATURE_MISMATCH for the canonical solution (measured:
    4 of 164). Falls back to the last def when the prompt does not name the entry
    point, which is the def a continuation completes.
    """
    tree, _ = parse_module(prompt)
    if tree is None:
        return None
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not defs:
        return None
    if entry_point:
        for node in defs:
            if node.name == entry_point:
                return signature_of(node)
    return signature_of(defs[-1])


# ---------------------------------------------------------------------------
# G3: executing the task's own asserts, in a child process
# ---------------------------------------------------------------------------

#: The child's wrapper. The scored source arrives on stdin, so nothing typed by
#: the model reaches a command line or a file on disk. Everything here is stdlib
#: and the child runs under `-I`, so the parent's user site-packages, cwd and env
#: cannot change what it imports.
#:
#: Four exit codes, because "the code is wrong" and "the code exploded" are
#: different findings: 0 pass, 92 the candidate could not even be exec'd / has no
#: `check` to run, 93 the asserts ran and at least one failed, 124 the alarm
#: fired. Anything else (a negative rc from a signal, 90/91 from the outer
#: handler) is a crash. The asserts are called as `check(candidate)` -- the
#: HumanEval module contract, `humaneval_gen.judge`'s call.
_CHILD = r'''
import signal, sys
def _alarm(*_a):
    raise SystemExit(124)
signal.signal(signal.SIGALRM, _alarm)
signal.setitimer(signal.ITIMER_REAL, __E5_SECONDS__)
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (__E5_CPU__, __E5_CPU__))
    resource.setrlimit(resource.RLIMIT_AS, (__E5_MEM__, __E5_MEM__))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (__E5_NPROC__, __E5_NPROC__))
except Exception:
    pass
try:
    _src = sys.stdin.read()
    _g = {"__name__": "__main__"}
    exec(compile(_src, "<e5_candidate>", "exec"), _g)
    signal.setitimer(signal.ITIMER_REAL, 0)
    del _src
    _check = _g.get("check")
    if not callable(_check):
        print("no check(candidate) in the scored source", file=sys.stderr)
        sys.exit(92)
    if sys.argv[1] not in _g:
        print("the entry point is not defined at module level", file=sys.stderr)
        sys.exit(92)
    _check(_g[sys.argv[1]])
    sys.exit(0)
except SystemExit as _e:
    if _e.code == 124:
        print("E5_CHILD_TIMEOUT", file=sys.stderr)
        sys.exit(124)
    if _e.code in (None, 0):
        sys.exit(0)
    print("SystemExit: %r" % (_e.code,), file=sys.stderr)
    sys.exit(90)
except AssertionError:
    import traceback
    traceback.print_exc()
    sys.exit(93)
except BaseException:
    import traceback
    traceback.print_exc()
    sys.exit(91)
'''.strip()


def run_assert_tests(src, entry_point, timeout=TASK_TIMEOUT_S, mem_mb=RUN_MEM_MB,
                     nproc=RUN_NPROC):
    """Execute one scored program in a child process. Returns (state, detail).

    state is one of pass / fail / timeout / crash. A timeout is the wall clock OR
    the child's own alarm OR RLIMIT_CPU, and it is reported as `timeout` rather
    than folded into `fail`, because "the code hangs" and "the code is wrong" are
    different failures to chase. `fail` means the asserts ran and lost; `crash`
    means the candidate could not be exec'd, has no callable `check`, never
    defines its entry point, or died on a signal.
    """
    child = (_CHILD
             .replace("__E5_SECONDS__", f"{float(timeout):.3f}")
             .replace("__E5_CPU__", str(max(1, int(math.ceil(timeout)))))
             .replace("__E5_MEM__", str(int(mem_mb) * 1024 * 1024))
             .replace("__E5_NPROC__", str(int(nproc))))
    started = time.time()
    try:
        cp = subprocess.run(
            [sys.executable, "-I", "-c", child, entry_point],
            input=src.encode("utf-8"), capture_output=True,
            cwd="/", timeout=timeout + 4.0,
        )
    except subprocess.TimeoutExpired:
        return "timeout", {"rc": None, "secs": round(time.time() - started, 2),
                           "why": "wall clock; the child ignored its own alarm"}
    except OSError as exc:
        return "crash", {"rc": None, "why": f"spawn failed: {exc}"}
    secs = round(time.time() - started, 2)
    err = cp.stderr.decode("utf-8", "replace")
    out = cp.stdout.decode("utf-8", "replace")
    detail = {"rc": cp.returncode, "secs": secs,
              "stdout_tail": out[-400:], "stderr_tail": err[-800:]}
    if cp.returncode == 0:
        return "pass", detail
    if cp.returncode in (124,) or signals_timeout(cp.returncode, err):
        detail["why"] = "child alarm / RLIMIT_CPU / killed by a signal"
        return "timeout", detail
    if cp.returncode == 93:
        detail["why"] = "the asserts ran and failed"
        return "fail", detail
    detail["why"] = f"rc {cp.returncode}"
    return "crash", detail


def signals_timeout(rc, stderr_text):
    """Whether a nonzero exit is a timeout rather than a wrong answer.

    -24 is SIGXCPU, which is how RLIMIT_CPU arrives on Darwin; -9 SIGKILL (the
    parent, after its wall clock, or the OOM killer), -14 SIGALRM, 137 the shell's
    encoding of SIGKILL, 124 the child's own alarm. A CPU-bound infinite loop is
    killed by whichever of RLIMIT_CPU and the wall clock is first, and both are
    timeouts; anything else with a negative rc is a crash.
    """
    if rc in (-9, -14, -24, -25, 137, 124):
        return True
    return "E5_CHILD_TIMEOUT" in (stderr_text or "")


# ---------------------------------------------------------------------------
# G4: irrelevant explanation
# ---------------------------------------------------------------------------

def _bracket_delta(line):
    """Net ()[]{} balance of one line, ignoring string literals and comments.

    A line inside an open bracket cannot be parsed alone, and an unclosed literal
    is not prose. Quotes are skipped so a bracket inside a string does not count;
    an unterminated quote at the end of the line is treated as closed at the line
    end, which is what a continuation of real code looks like.
    """
    depth, i, n = 0, 0, len(line)
    quote = None
    while i < n:
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if line.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
            i += 1
            continue
        if ch in "\"'":
            quote = line[i : i + 3] if line.startswith(ch * 3, i) else ch
            i += len(quote)
            continue
        if ch == "#":
            break
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        i += 1
    return depth


#: A line that opens a block cannot be parsed without its body, so it is code.
BLOCK_RE = re.compile(
    r"^(?:if|elif|else|for|while|try|except|finally|with|def|class|match|case|async)\b")


def _is_code_line(line, depth=0):
    """A line a Python reader could not call prose.

    Three arms, in order, and each one exists because a simpler rule failed on a
    real canonical solution:
      1. `depth > 0` -- the line is inside a bracket opened earlier. Multi-line
         dict/list literals are 12 of HumanEval/19's 13 lines (measured).
      2. the line parses, alone or dedented. A continuation body is INDENTED, so
         `ast.parse(line)` alone raises IndentationError for every line of a
         correct answer.
      3. the line opens a block (`elif`, `else:`, `try:`) or is only a fragment of
         one, which cannot parse without the body it belongs to. Without this arm
         HumanEval/81's twelve `elif` lines read as prose (measured).

    A blank line and a comment are not prose.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    if depth > 0:
        return True
    candidates = [line, textwrap.dedent(line)]
    if stripped.startswith((">>>", "...")):
        candidates.append(re.sub(r"^\s*(?:>>>|\.\.\.)\s?", "", line))
    for cand in candidates:
        try:
            ast.parse(cand)
            return True
        except SyntaxError:
            continue
    if BLOCK_RE.match(stripped) and stripped.endswith(":"):
        return True
    try:
        ast.parse(textwrap.dedent(line) + "\n    pass\n")
        return True
    except SyntaxError:
        return False


def explanation_report(completion, entry_point):
    """(ok, detail) for G4, on the generated segment only.

    `detail["prose_share"]` is computed over non-blank lines. The preamble is the
    text before the first line that starts the answer: a `def <entry_point>` line,
    or the first line that is code. A completion with no prose at all passes even
    though the share is then undefined, which is why `n_lines` is reported beside
    it. The line-class walk carries the bracket depth forward, so a line inside an
    open literal is code by position and not by its own parse.
    """
    text = completion or ""
    lines = text.splitlines()
    code_flags, depth = [], 0
    for ln in lines:
        code_flags.append(_is_code_line(ln, depth))
        depth += _bracket_delta(ln)
    pairs = [(ln, code) for ln, code in zip(lines, code_flags, strict=True) if ln.strip()]
    nonblank = [ln for ln, _ in pairs]
    prose = [ln for ln, code in pairs if not code]
    share = (len(prose) / len(nonblank)) if nonblank else 0.0

    start = None
    for i, ln in enumerate(lines):
        if ln.strip() and (re.match(r"\s*def\s+" + re.escape(entry_point) + r"\b", ln)
                           or code_flags[i]):
            start = i
            break
    preamble = [ln for ln in lines[:start if start is not None else 0] if ln.strip()]

    run = 0
    longest, prev = 0, None
    for ln in lines:
        key = ln.strip()
        if key:
            run = run + 1 if key == prev else 1
            prev = key
        longest = max(longest, run)

    detail = {
        "n_lines": len(nonblank),
        "n_prose": len(prose),
        "prose_share": round(share, 3),
        "preamble_lines": len(preamble),
        "preamble_chars": sum(len(ln) for ln in preamble),
        "longest_line_run": longest,
        "first_prose": (prose[0][:120] if prose else ""),
    }
    ok = (share <= PROSE_SHARE_MAX
          and len(preamble) <= PREAMBLE_LINES_MAX
          and detail["preamble_chars"] <= PREAMBLE_CHARS_MAX
          and longest <= REPEAT_RUN_MAX)
    return ok, detail


# ---------------------------------------------------------------------------
# G5: secret-style text
# ---------------------------------------------------------------------------

def entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


def secret_report(completion, exempt=CANARIES):
    """(ok, detail) for G5. Matches, never a keyword list of what a secret is.

    A base64/hex run needs BOTH length and an entropy floor: a repeated character
    run can clear the length and not be a key, and a short high-entropy word
    cannot be one. Key prefixes are matched by shape.

    `exempt` is a whitelist of VALUES compared for exact equality against the
    candidate. It exists so a control can carry a known secret-shaped literal, and
    it is deliberately narrow: equality rather than a prefix, and the default is
    this module's own canaries rather than anything drawn from the task or the
    sample, so a secret landing next to a test vector is still caught.
    """
    text = completion or ""
    exempt = tuple(exempt or ())
    hits = []
    for m in BASE64_RE.finditer(text):
        s = m.group(0)
        if s in exempt:
            continue
        e = entropy(s)
        if e >= B64_ENTROPY_MIN:
            hits.append({"kind": "base64", "len": len(s), "entropy": round(e, 2),
                         "sample": s[:24] + "..."})
    for m in HEX_RE.finditer(text):
        s = m.group(0)
        if s in exempt:
            continue
        e = entropy(s)
        if e >= HEX_ENTROPY_MIN:
            hits.append({"kind": "hex", "len": len(s), "entropy": round(e, 2),
                         "sample": s[:24] + "..."})
    for rx in KEY_PREFIX_RES:
        for m in rx.finditer(text):
            hits.append({"kind": "key_prefix", "len": len(m.group(0)),
                         "sample": m.group(0)[:24] + "..."})
    # Deduplicate: a hex run is also a base64 run, and one literal is one hit.
    seen, uniq = set(), []
    for h in hits:
        key = (h["kind"], h["sample"], h["len"])
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return (not uniq), {"n_hits": len(uniq), "hits": uniq[:5]}


# ---------------------------------------------------------------------------
# The per-item pipeline
# ---------------------------------------------------------------------------

def grade(task, completion, timeout=TASK_TIMEOUT_S):
    """One item through G1-G5. Returns (row, gates, fail_classes).

    Every judgment runs regardless of the earlier ones: a wrong signature and a
    failing assert are different facts, and a report that stops at the first
    failure cannot say which one the model has. `row` carries each judgment's
    detail, so a FAIL is always attributable.
    """
    row = {"id": task.task_id, "entry_point": task.entry_point}
    fails = []

    # An empty continuation is its own finding, not a wrong answer. Handover doc
    # section 4.1 lists 空输出 and 复读 as separate symptoms, and E6's triage needs
    # to tell "the model said nothing" from "the model wrote the wrong thing": a
    # base that emits EOS on the prompt (measured, 3 of 4 on tasks 0-3) is a stop
    # behaviour result, and calling it UNIT_TEST_FAIL hides that.
    if not (completion or "").strip():
        fails.append("EMPTY_RESPONSE")

    prefix, body = complete_source(task.prompt, completion)
    src = prefix + body
    tree, syn = parse_module(src)
    row["syntax"] = {"ok": syn is None,
                     "error": (f"{type(syn).__name__}: {syn.msg} (line {syn.lineno})"
                               if syn is not None else "ast.parse ok"),
                     "src_sha256": sha256_text(src),
                     "by_name": bool(prefix) and not prefix.endswith(":\n")}
    if syn is not None:
        fails.append("SYNTAX_ERROR")

    node = find_definition(tree, task.entry_point)
    got_sig = signature_of(node) if node is not None else None
    want_sig = signature_from_prompt(task.prompt, task.entry_point)
    if node is None:
        sig_ok = False
        sig_note = f"no module-level `def {task.entry_point}` in the completion or the prompt"
    elif want_sig is None:
        sig_ok = True
        sig_note = "the prompt has no def to compare against; existence is the check"
    else:
        sig_ok = (got_sig == want_sig)
        sig_note = ("matches the prompt" if sig_ok
                    else f"{got_sig} != prompt {want_sig}")
    row["signature"] = {"ok": sig_ok, "note": sig_note,
                        "params": [p[0] for p in (got_sig["params"] if got_sig else [])],
                        "expected_params": [p[0] for p in (want_sig["params"] if want_sig else [])]}
    if not sig_ok:
        fails.append("SIGNATURE_MISMATCH")

    # The executed program: candidate function + the task's own test + check().
    check_src = f"{src.rstrip()}\n{task.test}\n"
    state, exec_detail = run_assert_tests(check_src, task.entry_point, timeout=timeout)
    row["unit"] = {"ok": state == "pass", "state": state, **exec_detail}
    if state == "timeout":
        fails.append("UNIT_TEST_TIMEOUT")
    elif state == "crash":
        fails.append("UNIT_TEST_CRASH")
    elif state != "pass":
        fails.append("UNIT_TEST_FAIL")

    exp_ok, exp_detail = explanation_report(completion, task.entry_point)
    row["explanation"] = {"ok": exp_ok, **exp_detail}
    if not exp_ok:
        fails.append("EXPLANATION")

    sec_ok, sec_detail = secret_report(completion)
    row["secret"] = {"ok": sec_ok, **sec_detail}
    if not sec_ok:
        fails.append("SECRET_STYLE")

    row["empty"] = not (completion or "").strip()
    row["reasoning_chars"] = len(completion or "")
    row["fail_class"] = fails[0] if fails else None
    row["fail_classes"] = fails
    row["passed"] = (not fails)
    row["passed_tests"] = (state == "pass")
    row["passed_clean"] = (state == "pass" and exp_ok and sec_ok)
    gates = {
        "syntax_ok": syn is None,
        "signature_ok": sig_ok,
        "unit_ok": state == "pass",
        "explanation_ok": exp_ok,
        "secret_ok": sec_ok,
        "terminated": None,
    }
    return row, gates, fails


def item_row(task, row, completion, prompt_text, meta, extra=None):
    """One items.jsonl record, in the nine fields the contract fixes (R3/R10).

    `id`, `prompt` (verbatim), `response_raw`, `parsed`, `passed`, `stop_reason`,
    `n_tokens`, `fail_class`, `meta`. The per-judgment detail stays as additional
    keys, because an auditor recomputing a verdict needs the subprocess rc and the
    matched literal, not only the verdict. `parsed` is the structured reading of
    the response -- the syntax verdict, the extracted signature, the definition
    the tests were run against -- so a reader does not have to re-parse the text.
    """
    rec = {
        "id": task.task_id,
        "prompt": prompt_text,
        "response_raw": completion,
        "parsed": {
            "syntax": row["syntax"],
            "signature": row["signature"],
            "definition_source": ("completion" if row["syntax"]["by_name"] else "prompt-prefix"),
            "unit": {k: v for k, v in row["unit"].items() if k != "stdout_tail"},
            "explanation": row["explanation"],
            "secret": row["secret"],
            "entry_point": task.entry_point,
        },
        "passed": bool(row["passed"]),
        "stop_reason": row.get("stop_reason"),
        "n_tokens": int(row.get("tokens") or 0),
        "fail_class": row.get("fail_class"),
        "meta": meta,
    }
    rec.update(extra or {})
    return rec


# ---------------------------------------------------------------------------
# Stage entry
# ---------------------------------------------------------------------------

def _gate_rates(rows):
    """The two rates the gate reads, plus the outcome counts they are made of."""
    n = len(rows)
    passed_tests = sum(1 for r in rows if r["passed_tests"])
    passed_clean = sum(1 for r in rows if r["passed_clean"])
    return {
        "n": n,
        "passed_tests": passed_tests,
        "passed_clean": passed_clean,
        "unit_rate": round(passed_tests / n, 4) if n else 0.0,
        "sample_rate": round(passed_clean / n, 4) if n else 0.0,
        "syntax_ok": sum(1 for r in rows if r["syntax"]["ok"]),
        "signature_ok": sum(1 for r in rows if r["signature"]["ok"]),
        "explanation_ok": sum(1 for r in rows if r["explanation"]["ok"]),
        "secret_ok": sum(1 for r in rows if r["secret"]["ok"]),
        "empty": sum(1 for r in rows if r["empty"]),
    }


def _rate_gate(rate, floor):
    """A rate gate with the floor written into the comparison.

    `floor` is a fraction of n, so a sample of 8 passes the unit gate at 3 and the
    accuracy gate at 2; a sample of 20 passes at 7 and 5. Small-n gates are
    coarse by construction and the numbers are printed beside the verdict.
    """
    return rate + 1e-12 >= floor


UNIT_GATE = 1.0 / 3.0
SAMPLE_GATE = 0.25


def _controls(task, timeout):
    """The known-answer worlds. Every negative must FAIL on its own judgment.

    Returns (ok, list_of_records). Each record names the judgment the world is
    built to trip; `hit` is False when that judgment passed anyway, which is the
    only thing that makes this a control rather than a run.

    Ten worlds. The canonical positive is the one that says the pipeline is not
    simply failing everything; `canary_exempt` is the one that says the secret
    detector's exemption works on a real answer, not only in a unit call.
    """
    canary_body = ('    token = "' + CANARIES[1] + '"\n'
                   '    for idx, elem in enumerate(numbers):\n'
                   '        for idx2, elem2 in enumerate(numbers):\n'
                   '            if idx != idx2 and abs(elem - elem2) < threshold:\n'
                   '                return True\n'
                   '    return False\n')
    worlds = [
        # The positive is the canonical BODY, which is the object a continuation
        # model is asked for; the prompt is not fed in as if the model wrote it.
        ("canonical", task.canonical_solution, None),
        # The by-name arm of complete_source, on a world that must PASS. Every
        # other by-name world here is a negative, so without this one a broken
        # by-name parser would read as "it correctly rejects bad answers".
        ("by_name_canonical",
         "def " + task.entry_point + "(numbers, threshold):\n" + task.canonical_solution, None),
        ("canary_exempt", canary_body, None),
        ("empty_response", "", "EMPTY_RESPONSE"),
        ("const_false", NEG_CONSTANT_FALSE, "UNIT_TEST_FAIL"),
        ("syntax_error", NEG_SYNTAX, "SYNTAX_ERROR"),
        ("prose_only", NEG_PROSE, "EXPLANATION"),
        ("secret_base64", NEG_SECRET, "SECRET_STYLE"),
        ("secret_in_valid_code", NEG_SECRET_CODE, "SECRET_STYLE"),
        ("wrong_signature", NEG_SIGNATURE, "SIGNATURE_MISMATCH"),
        ("wrong_signature_params", NEG_SIGNATURE_PARAMS, "SIGNATURE_MISMATCH"),
        ("infinite_loop", NEG_INFINITE, "UNIT_TEST_TIMEOUT"),
    ]
    out, ok = [], True
    for name, completion, expect in worlds:
        row, _gates, fails = grade(task, completion, timeout=timeout)
        good = row["passed"] if expect is None else (expect in fails and not row["passed"])
        ok = ok and good
        out.append({"control": name, "expect_fail_class": expect,
                    "passed": row["passed"], "fail_classes": fails,
                    "ok": good, "unit_state": row["unit"]["state"],
                    "unit_rc": row["unit"].get("rc"),
                    "prose_share": row["explanation"]["prose_share"],
                    "secret_hits": row["secret"]["n_hits"],
                    "response_raw": completion,
                    "row": row})
    return ok, out


def _control_items(task, controls, timeout):
    """The control records as items.jsonl rows, in the nine contract fields."""
    items = []
    for rec in controls:
        row = dict(rec["row"])
        row["stop_reason"] = "control"
        row["tokens"] = 0
        row["fail_class"] = (rec["fail_classes"] or [None])[0]
        items.append(item_row(
            task, row, rec["response_raw"], task.prompt,
            {"kind": "control", "control": rec["control"], "seed": None,
             "temperature": 0.0, "max_new": 0, "timeout_s": timeout,
             "unit_state": rec["unit_state"], "unit_rc": rec["unit_rc"],
             "elapsed_s": row["unit"].get("secs")},
            extra={"id": f"control:{rec['control']}",
                   "control_ok": bool(rec["ok"]),
                   "expect_fail_class": rec["expect_fail_class"],
                   # `passed` in the nine fields is the verdict of the piece of
                   # code as a HumanEval answer. `control_ok` is the verdict of the
                   # CONTROL, which inverts for a negative world.
                   "passed": bool(rec["passed"]),
                   "fail_classes": rec["fail_classes"]},
        ))
    return items


def control_selftest():
    """CPU-only known-answer worlds. No checkpoint, no GPU, no generation.

    The positive is HumanEval/0's canonical solution fed through the SAME pipeline
    the gate runs; each negative is built to land on ONE judgment, so a green run
    here says the pipeline can still tell a correct answer from eight specific
    wrong ones. A control that only checked `row["passed"]` would pass if the
    pipeline failed everything, which is why each world names its fail_class.
    """
    failures = []
    data = DATA_DEFAULT
    if not pathlib.Path(data).is_file():
        alt = pathlib.Path("/tmp/he164.jsonl")
        if alt.is_file():
            data = str(alt)
        else:
            print(f"SKIP e5 control: {data} not present (pod path "
                  "/work/aupai/data/eval/humaneval/humaneval_164.jsonl)", file=sys.stderr)
            return lc.EXIT_REFUSE
    try:
        tasks = load_tasks(data, 1)
    except lc.Refuse as exc:
        print(f"SKIP e5 control: {exc}", file=sys.stderr)
        return lc.EXIT_REFUSE
    task = tasks[0]
    if task.task_id != "HumanEval/0":
        print(f"FAIL e5 control: first sorted task is {task.task_id}, not HumanEval/0",
              file=sys.stderr)
        return lc.EXIT_GATE
    ok, records = _controls(task, timeout=TASK_TIMEOUT_S)
    for rec in records:
        mark = "PASS" if rec["ok"] else "FAIL"
        print(f"{mark} e5 control {rec['control']}: expect "
              f"{rec['expect_fail_class'] or 'PASS'}, got "
              f"{rec['fail_classes'] or 'PASS'} (unit={rec['unit_state']} rc={rec['unit_rc']}, "
              f"prose={rec['prose_share']}, secret_hits={rec['secret_hits']})")
    if not ok:
        bad = [r["control"] for r in records if not r["ok"]]
        failures.append(f"controls not behaving: {bad}")
    # The detector's own canary, in three directions, because an exemption is the
    # one place a detector can be blind and still look green:
    #   1. the canary IS detected when a caller does not exempt it -- this is what
    #      makes the exemption a whitelist of one value rather than a blind spot
    #      in the detector ("it returned no hits" and "it never matches" are the
    #      same observation from outside, so the exemption-free call is the
    #      discriminator);
    #   2. the bare canary is exempt, so a control whose answer is the canary can
    #      still be scored (`canary_exempt` in _controls is the end-to-end half);
    #   3. the canary plus one character is caught again, which fails if the
    #      exemption was written as a prefix instead of an equality.
    for lit in CANARIES:
        no_exempt_ok, detail = secret_report('    k = "' + lit + '"\n', exempt=())
        if no_exempt_ok:
            failures.append(f"canary {lit[:12]}... never matches the detector: {detail}")
        exempt_ok, _ = secret_report(lit)
        if not exempt_ok:
            failures.append(f"canary {lit[:12]}... not exempt as an exact match")
        # The candidate must be a different, longer string for the prefix test to
        # discriminate: a non-hex character after a hex run terminates the regex
        # match and an exact-equality exemption already covers it, so appending
        # "Z" to the hex canary would pass under either rule (measured).
        tail_ok, _ = secret_report(lit + "0000", exempt=(lit,))
        if tail_ok:
            failures.append(f"canary {lit[:12]}...0000 exempted: the exemption is a prefix, not an equal")
    # The planted control secrets must NOT be exempt: if a control literal were in
    # CANARIES the negative world would pass on a detector that sees nothing.
    for lit in (CTRL_SECRET_B64, CTRL_SECRET_HEX):
        if lit in CANARIES:
            failures.append(f"control secret {lit[:12]}... is exempt from the detector")
        if secret_report(lit)[0]:
            failures.append(f"control secret {lit[:12]}... not detected")
    if not secret_report("")[0]:
        failures.append("empty output flagged as a secret")
    for f in failures:
        print(f"FAIL e5 control: {f}", file=sys.stderr)
    if failures:
        return lc.EXIT_GATE
    n_pos = sum(1 for r in records if r["expect_fail_class"] is None)
    n_neg = len(records) - n_pos
    print(f"PASS e5 controls: {n_pos} valid worlds, {n_neg} negative worlds, "
          "each on its own judgment")
    return lc.EXIT_OK


def main(argv):
    ap = argparse.ArgumentParser(prog="eval_ladder.py e5")
    ap.add_argument("--ckpt", default=None, help="checkpoint; omitted with --control")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new", dest="max_new", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help=f"sample size, <= {MAX_SAMPLE}")
    ap.add_argument("--control", action="store_true",
                    help="known-answer controls through the real pipeline; then exit")
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--timeout", type=float, default=TASK_TIMEOUT_S,
                    help="per-item wall clock for the unit-test child")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--claim", default=None, help="card claim name to take, if any")
    args = ap.parse_args(argv)

    lc.require(1 <= args.n <= MAX_SAMPLE,
               f"--n {args.n} outside 1..{MAX_SAMPLE}; the full 164 is E6")
    lc.require(args.max_new >= 32, "--max-new below 32 cannot hold a function body")

    def stage(paths, log):
        # Preconditions first, in the order the caller can act on them. The
        # checkpoint is checked before the sample is read so a missing checkpoint
        # refuses on its own name (handover P0: an error path that reports the
        # wrong missing input is worse than no error path); the dispatcher's
        # refuse-path control passes a checkpoint that does not exist and expects
        # rc=2 with a "refused" summary, which is exactly this branch.
        if not args.control:
            lc.require(args.ckpt, "--ckpt is required unless --control is passed")
            lc.require(pathlib.Path(args.ckpt).is_file(),
                       f"checkpoint {args.ckpt} does not exist")
            if str(args.device).startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
                lc.require(False, "CUDA_VISIBLE_DEVICES is unset, so cuda:0 would be physical "
                                  "GPU 0 outside any grant; set it to the granted card or pass "
                                  "--device cpu with CUDA_VISIBLE_DEVICES= empty")

        tasks = load_tasks(args.data, args.n)
        data_sha = lc.sha256_file(args.data)
        n_in_file = count_tasks(args.data)
        ids = [t.task_id for t in tasks]
        log.write(f"E5 sample: n={len(ids)} of {n_in_file} ids={ids} data_sha256={data_sha}\n")

        ok, controls = _controls(tasks[0], timeout=args.timeout)
        for rec in controls:
            log.write(f"{'PASS' if rec['ok'] else 'FAIL'} control {rec['control']}: "
                      f"expect {rec['expect_fail_class'] or 'PASS'}, got "
                      f"{rec['fail_classes'] or 'PASS'}\n")
        # Control rows carry the same nine contract fields as a scored row: the
        # controls ARE evidence, and an auditor recomputing one needs id, prompt,
        # response_raw, parsed, passed, stop_reason, n_tokens, fail_class, meta.
        control_items = _control_items(tasks[0], controls, timeout=args.timeout)
        if not ok:
            bad = [r["control"] for r in controls if not r["ok"]]
            log.write(f"E5 controls FAILED: {bad}\n")
            lc.write_items(paths, control_items)
            return {"controls": controls, "gates": {"controls": False}, "n": 0, "passed": 0,
                    "exit_code": lc.EXIT_GATE}

        if args.control:
            log.write("--control: controls passed, no generation requested\n")
            lc.write_items(paths, control_items)
            return {"controls": controls, "gates": {"controls": True}, "n": 0, "passed": 0,
                    "protocol": "control-only", "inputs": {"humaneval_sha256": data_sha},
                    "exit_code": lc.EXIT_OK}

        lc.seal(paths, stage="e5", ckpt=args.ckpt, device=args.device, n=args.n,
                task_ids=ids, data=args.data, data_sha256=data_sha,
                max_new=args.max_new, temperature=args.temperature,
                protocol="base-continuation", timeout_s=args.timeout)

        model, tok, meta = lc.load_for_eval(args.ckpt, device=args.device, claim=True)
        if args.claim:
            log.write(f"claim name given: {args.claim}\n")
        model.eval()
        ident = ckpt_identity(args.ckpt)
        log.write(f"checkpoint identity: step={ident['step']} ({ident['step_source']}), "
                  f"ckpt_vocab_id={ident['ckpt_vocab_id']}, "
                  f"tokenizer_vocab_id={meta['vocab_id']}\n")
        from train import generate_batch  # noqa: PLC0415 (needs torch, so after the loader)

        prompts = [tok.encode(t.prompt).ids for t in tasks]
        log.write(f"generating {len(prompts)} continuations, max_new={args.max_new}, "
                  f"temperature={args.temperature}\n")
        t0 = time.time()
        gen_ids = generate_batch(model, prompts, args.max_new, args.device,
                                 temperature=args.temperature, tokenizer=tok,
                                 rep_stop=False)
        gen_s = round(time.time() - t0, 1)
        del model

        rows = []
        for task, ids_i in zip(tasks, gen_ids, strict=True):
            completion = tok.decode(ids_i)
            row, gates, fails = grade(task, completion, timeout=args.timeout)
            gate = lc.stop_reason(ids_i[-1] if ids_i else None, meta["eos_id"],
                                  meta["im_end_id"], len(ids_i), args.max_new)
            row["stop_reason"] = gate
            row["tokens"] = len(ids_i)
            row["prompt_sha256"] = sha256_text(task.prompt)
            row["output"] = completion
            row["output_sha256"] = sha256_text(completion)
            gates["terminated"] = gate != "max_new"
            row["gates"] = gates
            rows.append(row)
            log.write(f"{'PASS' if row['passed'] else 'FAIL'} {row['id']} "
                      f"({task.entry_point}): {row['fail_classes'] or 'all judgments pass'} "
                      f"unit={row['unit']['state']} stop={gate} tokens={len(ids_i)} "
                      f"prose={row['explanation']['prose_share']} "
                      f"secret_hits={row['secret']['n_hits']}\n")

        rates = _gate_rates(rows)
        gates = {
            "controls": True,
            "syntax": rates["syntax_ok"] == rates["n"],
            "unit": _rate_gate(rates["unit_rate"], UNIT_GATE),
            "sample": _rate_gate(rates["sample_rate"], SAMPLE_GATE),
        }
        gates["gate_pass"] = bool(gates["syntax"] and gates["unit"] and gates["sample"])
        failed_ids = [r["id"] for r in rows if not r["passed"]]
        fail_hist = Counter(fc for r in rows for fc in r["fail_classes"])
        summary = {
            "ckpt": args.ckpt,
            "ckpt_sha256": meta["ckpt_sha256"],
            # step from the checkpoint FILE, not cfg.step, which is absent here
            "step": ident["step"] if ident["step"] is not None else meta["step"],
            "step_source": ident["step_source"],
            "vocab_id": meta["vocab_id"],
            "ckpt_vocab_id": ident["ckpt_vocab_id"],
            "protocol": "base-continuation",
            "prompt_format": "humaneval-prompt-verbatim (no ChatML)",
            "n": rates["n"],
            "passed": rates["passed_clean"],
            "passed_tests": rates["passed_tests"],
            "gates": gates,
            "gate_floors": {"unit_rate": round(UNIT_GATE, 4),
                            "sample_rate": SAMPLE_GATE,
                            "prose_share_max": PROSE_SHARE_MAX},
            "rates": rates,
            "fail_class_histogram": dict(sorted(fail_hist.items())),
            "failed_ids": failed_ids,
            "stop_reasons": dict(sorted(Counter(r["stop_reason"] for r in rows).items())),
            "controls": controls,
            "inputs": {"humaneval": args.data, "humaneval_sha256": data_sha,
                       "task_ids": ids, "n_tasks_in_file": n_in_file, "n_sample_cap": MAX_SAMPLE},
            "sampling": {"max_new": args.max_new, "temperature": args.temperature,
                         "rep_stop": False, "decode": "train.generate_batch",
                         "scoring": "child process, RLIMIT_CPU+AS+FSIZE+NPROC, wall timeout"},
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "pid": os.getpid(),
            "elapsed_generation_s": gen_s,
            "exit_code": lc.EXIT_OK if gates["gate_pass"] else lc.EXIT_GATE,
        }
        lc.write_items(paths, [
            item_row(task, row, row["output"], task.prompt,
                     {"kind": "sample", "seed": None, "temperature": args.temperature,
                      "max_new": args.max_new, "timeout_s": args.timeout,
                      "elapsed_s": row["unit"].get("secs"),
                      "generation_elapsed_s": gen_s,
                      "unit_state": row["unit"]["state"],
                      "unit_rc": row["unit"].get("rc"),
                      "device": args.device, "step": meta["step"]},
                     extra={"gates": row["gates"], "prompt_sha256": row["prompt_sha256"],
                            "output_sha256": row["output_sha256"],
                            "fail_classes": row["fail_classes"]})
            for task, row in zip(tasks, rows, strict=True)
        ])
        log.write(f"E5 {'PASSED' if gates['gate_pass'] else 'FAILED'}: {json.dumps(rates, sort_keys=True)}\n")
        if not gates["gate_pass"]:
            log.write(f"failed ids: {failed_ids}\n")
        return summary

    return lc.run_stage("e5", stage, ckpt=(args.ckpt or "control"))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
