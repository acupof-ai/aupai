#!/usr/bin/env python3
"""The single place this project's progress is checked, recorded, and advanced.

Why this exists. Over one night three experiments produced three write-ups and zero
runs of `scripts/eval_hard.sh`, the metric of record -- so "which checkpoint is best"
was unanswerable while the conclusions read as settled. Separately, `Cfg.mix`
defaulted to the v2 mix (88% unfiltered web) for days, guarded only inside a wrapper
script that the documented entry point does not call. Both are the same failure: a
claim resting on a measurement nobody took, with nothing in the system that could
say so out loud.

So the rule this file enforces is:

    A STAGE IS NOT DONE BECAUSE IT PRODUCED A FILE. IT IS DONE BECAUSE THE
    MEASUREMENT THAT WOULD FALSIFY IT EXISTS AND IS RECORDED.

And the rule this file enforces on ITSELF, because four agents writing guards for
this repo in one afternoon all shipped the same defect:

    A CHECK WITHOUT A FAILING CASE IS NOT A CHECK. Every entry in CHECKS carries a
    `broken()` that builds a world where the condition is violated, and --selftest
    asserts the check reports FAIL on it. A check that cannot be made to fail is
    reported as UNTESTED and counts as a failure of the harness, not a pass of the
    thing checked.

That rule is not decoration. The defects it exists to stop, all found by review of
code written for this repo today:
  - `"web" in domains` PASSing because `domains` parsed to `[]`
  - a guard living entirely inside `if use_mix:`, so a MISSING mix file skipped it
  - a selftest that verified a guard's logic while deleting its only call site went
    undetected
  - `eval_hard.sh <ckpt>` passing the checkpoint positionally, so every score it
    produced was dropped and its checkpoints listed as never measured

    python scripts/harness.py            # check + status
    python scripts/harness.py check      # invariants only; exit 1 on any failure
    python scripts/harness.py ledger     # provenance and score, one row per checkpoint
    python scripts/harness.py gaps       # what is NOT measured, stated out loud
    python scripts/harness.py --selftest # every check must fail on its broken world
"""

import ast
import glob
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DATA = os.path.join(ROOT, "data")
CORPUS = os.path.join(DATA, "corpus")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


# --------------------------------------------------------------------------- facts


def cfg_default(field):
    """Read a Cfg field from train.py by AST, without importing torch.

    Importing train.py pulls in torch, fla and liger; this file has to run in CI on
    CPU and on a laptop with none of them."""
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "Cfg":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and stmt.targets[0].id == field:
                    return ast.literal_eval(stmt.value)
    return None


def read_mix(path):
    """(domains, error). An unreadable or shapeless mix returns an error rather than
    an empty dict -- `"web" in {}` is False, which would report a passing guard for a
    mix nobody could parse."""
    if not os.path.exists(path):
        return None, f"{os.path.relpath(path, ROOT)} does not exist"
    try:
        obj = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return None, f"unparseable: {e}"
    doms = obj.get("domains")
    if not isinstance(doms, dict) or not doms:
        return None, "no non-empty 'domains' map (schema drift, or an empty mix)"
    return list(doms), None


def experiments():
    p = os.path.join(ROOT, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


CKPT_RE = re.compile(r"\bckpt_[A-Za-z0-9_.-]+?\.pt\b")
# The number must be the one carrying the % sign. "math-hard 37/1032 = 3.6%" holds
# three numbers and only the last is the score; a class excluding digits cannot even
# reach it (the fraction blocks the way) and returns nothing at all, which is worse
# than a wrong score because it reads as "never measured".
SCORE_RE = re.compile(r"math-hard.{0,40}?(\d+(?:\.\d+)?)\s*%", re.S)


def score_from(text):
    m = SCORE_RE.search(text or "")
    return float(m.group(1)) if m else None


def recorded_scores():
    """checkpoint -> (math-hard %, where it came from), plus the scores that matched
    no checkpoint at all.

    Attribution has to cover `scripts/eval_hard.sh <ckpt>`, which takes the
    checkpoint POSITIONALLY. Matching only `--out ckpt_X.pt` dropped every score the
    metric of record ever produced -- including k6_fone's 1.7% -- and then listed
    those checkpoints as never measured, in the section whose whole job is to make a
    missing measurement visible."""
    scores, orphans = {}, []
    for row in experiments():
        s = score_from(str(row.get("result", "")))
        if s is None:
            continue
        cmd = str(row.get("cmd", ""))
        run = str(row.get("name", "?"))
        # Priority matters, and every level of it is a real form in this log:
        #   --out X.pt          sft_math.py names its output
        #   --name X            train.py derives ckpt_<name>.pt from it
        #   (empty cmd)         a summary row; its own name IS the checkpoint
        #   positional X.pt     scripts/eval_hard.sh takes the checkpoint positionally
        # INPUTS must be excluded or the score lands on the wrong checkpoint: rl_direct
        # resumed ckpt_k4_11b_lr05.pt and scored the RL OUTPUT, so a naive positional
        # match credited k4 with a number k4 did not produce.
        inputs = set(re.findall(r"--(?:resume|sft_path|tokenizer|ckpt)\s+(\S+)", cmd))
        cand = None
        m = re.search(r"--out\s+(ckpt_[A-Za-z0-9_.-]+)\.pt", cmd)
        if m:
            cand = m.group(1)
        elif m := re.search(r"--name\s+([A-Za-z0-9_.-]+)", cmd):
            cand = f"ckpt_{m.group(1)}"
        elif not cmd.strip():
            cand = f"ckpt_{run}"
        else:
            free = [n for n in CKPT_RE.findall(cmd) if n not in inputs]
            if len(free) == 1:
                cand = free[0][: -len(".pt")]
        if cand is None:
            orphans.append((run, s, cmd[:60]))
            continue
        scores.setdefault(cand, (s, run))
    return scores, orphans


def local_tokenizers():
    """path -> fingerprint, for every data/tokenizer*.json that loads."""
    out = {}
    try:
        from tokenizers import Tokenizer

        from loader import vocab_fingerprint
    except Exception:
        return out
    for p in sorted(glob.glob(os.path.join(DATA, "tokenizer*.json"))):
        try:
            out[os.path.basename(p)] = vocab_fingerprint(Tokenizer.from_file(p))
        except Exception:
            pass
    return out


# -------------------------------------------------------------------------- checks
#
# Each check is (name, asserts, incident, run, broken). `run(root)` returns
# (state, evidence). `broken()` yields a temporary root where the condition is
# violated; --selftest asserts run() returns FAIL there.


def _tmp_repo(mix_obj=None, tokenizer_src=None):
    """A throwaway tree shaped like the repo, for a check to fail against."""
    import shutil
    import tempfile

    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "data", "corpus"), exist_ok=True)
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    if mix_obj is not None:
        json.dump(mix_obj, open(os.path.join(d, "data", "mix_test.json"), "w"))
    if tokenizer_src and os.path.exists(tokenizer_src):
        shutil.copy(tokenizer_src, os.path.join(d, "data", "tokenizer.json"))
    return d


def _tiny_tokenizer_json(eos_id=1, with_num=True):
    """A minimal WordLevel tokenizer that is VALID but LOSSY.

    An absent tokenizer only exercises the SKIP path, and a check that has never
    been seen to FAIL is indistinguishable from `return PASS`. This gives the
    round-trip and pinned-id checks something real to reject."""
    vocab = {"<unk>": 0, "<eos>": eos_id, "a": 2, "b": 3}
    if with_num:
        vocab["[NUM]"] = 4
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }


def _broken_tokenizer(eos_id=1, with_num=True):
    d = _tmp_repo()
    json.dump(
        _tiny_tokenizer_json(eos_id, with_num),
        open(os.path.join(d, "data", "tokenizer.json"), "w"),
    )
    return d


def _broken_stale_run():
    d = _tmp_repo()
    with open(os.path.join(d, "runs", "experiments.jsonl"), "w") as f:
        f.write(json.dumps({"name": "killed_job", "status": "running", "date": "2020-01-01 00:00:00"}) + "\n")
    return d


def check_mix_not_unfiltered(root):
    mix = cfg_default("mix")
    if not mix:
        return SKIP, 'Cfg.mix is empty (flat corpus chosen on purpose via --mix "")'
    doms, err = read_mix(os.path.join(root, mix))
    if err:
        # NOT a pass. An unreadable mix means the guard could not run, and the whole
        # point of this harness is that "could not check" never reads as "checked".
        return FAIL, f"cannot read the default mix: {err}"
    if "web" in doms:
        return FAIL, f"{mix} names domain 'web' (the unfiltered 2,991,648-doc corpus)"
    return PASS, f"{mix} domains={doms}"


def _broken_mix():
    return _tmp_repo({"total_tokens": 1e9, "domains": {"web": {"weight": 1.0}}})


def check_mix_shards(root):
    mix = cfg_default("mix")
    if not mix:
        return SKIP, "no mix configured"
    doms, err = read_mix(os.path.join(root, mix))
    if err:
        return FAIL, f"cannot read the default mix: {err}"
    corpus = os.path.join(root, "data", "corpus")
    if not os.path.isdir(corpus):
        return SKIP, "data/corpus/ is not present (laptop checkout, not the pod)"
    missing = [d for d in doms if not glob.glob(os.path.join(corpus, d, "*.jsonl"))]
    if missing:
        return FAIL, f"no shards for {missing}"
    return PASS, f"all {len(doms)} domains have shards"


def check_tokenizer_roundtrip(root):
    p = os.path.join(root, "data", "tokenizer.json")
    if not os.path.exists(p):
        return SKIP, "data/tokenizer.json not present"
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(p)
    except Exception as e:
        return FAIL, f"tokenizer will not load: {e}"
    probe = "a\x00b\tc 中文 42"
    got = tok.decode(tok.encode(probe, add_special_tokens=False).ids)
    if got != probe:
        return FAIL, f"round-trip lost bytes: {probe!r} -> {got!r}"
    return PASS, "NUL, tab, hanzi and digits survive"


def check_pinned_ids(root):
    p = os.path.join(root, "data", "tokenizer.json")
    if not os.path.exists(p):
        return SKIP, "data/tokenizer.json not present"
    try:
        from tokenizers import Tokenizer

        import loader

        v = Tokenizer.from_file(p).get_vocab()
    except Exception as e:
        return FAIL, f"cannot read vocabulary: {e}"
    eos, num = v.get("<eos>"), v.get("[NUM]")
    want_num = cfg_default("num_id")
    if eos != loader.EOS_ID:
        return FAIL, f"<eos> is {eos}, four files hardcode {loader.EOS_ID}"
    if num != want_num:
        return FAIL, f"[NUM] is {num}, Cfg.num_id is {want_num}"
    return PASS, f"<eos>={eos} [NUM]={num}"


def check_no_stale_running(root):
    p = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return SKIP, "runs/experiments.jsonl not present"
    rows = []
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "running":
            continue
        try:
            t = time.mktime(time.strptime(str(r.get("date", ""))[:19], "%Y-%m-%d %H:%M:%S"))
        except Exception:
            continue
        age_h = (time.time() - t) / 3600
        if age_h > 24:
            rows.append(f"{r.get('name', '?')} {age_h:.0f}h")
    if rows:
        return FAIL, f"{len(rows)} killed mid-run and never closed: {', '.join(rows[:6])}"
    return PASS, "no run has been 'running' for over a day"


def check_guard_on_path(root):
    """The guard's logic being right is not the property that failed; its being ON THE
    PATH is. train.py's own import-time selftest asserts this, and this check makes
    the same assertion visible here so removing it is reported rather than merely
    raising somewhere in CI."""
    src_path = os.path.join(root, "train.py")
    if not os.path.exists(src_path):
        return SKIP, "train.py not present"
    tree = ast.parse(open(src_path, encoding="utf-8").read())
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if fn is None:
        return FAIL, "train.py has no main()"
    called = {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    if "_assert_mix_domains" not in called:
        return FAIL, "main() does not call _assert_mix_domains; run_ddp.sh is unguarded"
    return PASS, "main() calls _assert_mix_domains"


def _broken_guard():
    d = _tmp_repo()
    open(os.path.join(d, "train.py"), "w").write("def main():\n    pass\n")
    return d


CHECKS = [
    (
        "mix_not_unfiltered",
        "the mix train.py defaults to does not name 'web'",
        "the v2 mix gave 88% weight to the unfiltered corpus and Cfg.mix pointed at it by default",
        check_mix_not_unfiltered,
        _broken_mix,
    ),
    (
        "mix_shards_present",
        "every domain in the default mix has shards on disk",
        "a domain with no shards is only caught after the other domains are tokenized",
        check_mix_shards,
        _broken_mix,
    ),
    (
        "tokenizer_roundtrip",
        "data/tokenizer.json decodes back to the exact input bytes",
        "the k5 vocabulary silently dropped NUL and tab",
        check_tokenizer_roundtrip,
        _broken_tokenizer,
    ),
    (
        "pinned_ids",
        "<eos> is loader.EOS_ID and [NUM] is Cfg.num_id",
        "four files hardcode these ids and a vocabulary rebuild moves them silently",
        check_pinned_ids,
        lambda: _broken_tokenizer(eos_id=5),
    ),
    (
        "no_stale_running",
        "no experiments.jsonl row has been 'running' for over 24h",
        "a killed job wrote its checkpoint, never ran its eval, and left the row open",
        check_no_stale_running,
        _broken_stale_run,
    ),
    (
        "guard_on_path",
        "train.py main() actually calls the mix guard",
        "the guard lived in a wrapper while the documented entry point bypassed it",
        check_guard_on_path,
        _broken_guard,
    ),
]


# -------------------------------------------------------------------------- stages
#
# A stage is done when its POSTCONDITION -- the measurement that could falsify it --
# exists. Never when its artifact exists.

STAGES = [
    (
        "tokenizer",
        ["tokenizer_roundtrip", "pinned_ids"],
        "a tokenizer_<name>.json pinned per live checkpoint",
    ),
    ("corpus", ["mix_not_unfiltered", "mix_shards_present"], "contamination scan recorded for every source"),
    ("pretrain", ["guard_on_path", "no_stale_running"], "checkpoint carries vocab_id; val loss recorded"),
    ("sft", ["pinned_ids"], "pack fingerprint == checkpoint vocab_id; loss-mask test passes"),
    ("eval", [], "math-hard recorded in runs/experiments.jsonl"),
]


# ------------------------------------------------------------------------- reports


def run_checks(root=ROOT, quiet=False):
    results = []
    for name, asserts, incident, fn, _broken in CHECKS:
        try:
            state, evidence = fn(root)
        except Exception as e:  # a check that crashes is a failed check, never a pass
            state, evidence = FAIL, f"the check itself raised: {type(e).__name__}: {e}"
        results.append((name, state, evidence, asserts, incident))
        if not quiet:
            print(f"  [{state:^4}] {name:<22} {evidence}")
            if state == FAIL:
                print(f"         asserts: {asserts}")
                print(f"         prevents: {incident}")
    return results


def ledger():
    scores, orphans = recorded_scores()
    toks = local_tokenizers()
    names = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(ROOT, "ckpt_*.pt"))}
    for row in experiments():
        names.update(n[:-3] for n in CKPT_RE.findall(str(row.get("cmd", ""))))
    rows = []
    for n in sorted(names):
        on_disk = os.path.exists(os.path.join(ROOT, f"{n}.pt"))
        s, src = scores.get(n, (None, None))
        rows.append((n, on_disk, s, src))
    print(f"  {'checkpoint':<26}{'on disk':>8}{'math-hard':>11}   source of the score")
    for n, on_disk, s, src in rows:
        sc = f"{s:.1f}%" if s is not None else "-"
        print(f"  {n:<26}{'yes' if on_disk else 'record':>8}{sc:>11}   {src or ''}")
    if orphans:
        # Named out loud: a score that matched no checkpoint was previously dropped in
        # silence, which is how five real measurements became "never measured".
        print(f"\n  {len(orphans)} recorded score(s) matched NO checkpoint name:")
        for name, s, cmd in orphans:
            print(f"    {name}: {s:.1f}%   cmd={cmd!r}")
    if toks:
        print("\n  local tokenizers:")
        for k, v in toks.items():
            print(f"    {k:<26}{v}")
    return rows


def gaps():
    scores, _orphans = recorded_scores()
    names = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(ROOT, "ckpt_*.pt"))}
    for row in experiments():
        names.update(n[: -len(".pt")] for n in CKPT_RE.findall(str(row.get("cmd", ""))))
    rows = [(n, True, scores.get(n, (None, None))[0], None) for n in sorted(names)]
    unmeasured = [n for n, _d, s, _src in rows if s is None]
    print(f"  {len(unmeasured)} checkpoint(s) with NO math-hard on record:")
    print("    " + ", ".join(unmeasured) if unmeasured else "    (none)")
    md = os.path.join(ROOT, "EXPERIMENTS.md")
    if os.path.exists(md):
        markers = (
            "not controlled",
            "never was",
            "still untested",
            "no benefit measurement",
            "cannot resolve",
            "unexplained",
            "has never been",
        )
        hits = [
            (i, ln.strip())
            for i, ln in enumerate(open(md, encoding="utf-8"), 1)
            if any(m in ln.lower() for m in markers)
        ]
        print(f"\n  {len(hits)} claim(s) EXPERIMENTS.md marks as uncontrolled or unmeasured:")
        for i, ln in hits[:12]:
            print(f"    EXPERIMENTS.md:{i}  {ln[:96]}")


def stages():
    res = {n: s for n, s, _e, _a, _i in run_checks(quiet=True)}
    scores, _ = recorded_scores()
    print(f"  {'stage':<12}{'gates':>26}   postcondition")
    for name, gates, post in STAGES:
        bad = [g for g in gates if res.get(g) == FAIL]
        detail = f"BLOCKED: {','.join(bad)}" if bad else f"{len(gates)} gate(s) pass"
        print(f"  {name:<12}{detail:>26}   {post}")
    print(f"\n  eval postcondition: {len(scores)} checkpoint(s) carry a math-hard score.")


# ------------------------------------------------------------------------ selftest


def _demo():
    """Every check must FAIL on a world where its condition is violated.

    A check nobody has ever seen fail is indistinguishable from `return PASS`, and
    that is not hypothetical here: reviewers mutated four separately-written guards
    for this repo today and every one of them still reported success."""
    import shutil

    assert score_from("math-hard 37/1032 = 3.6%") == 3.6, "took the numerator, not the percentage"
    assert score_from("math-hard deferred to the bench stage") is None, "invented a score"
    assert score_from("math-hard 1.7% (18/1032) vs k5 1.9%") == 1.7

    # the positional eval form must attribute, or the metric of record vanishes
    saved = os.path.join(ROOT, "runs", "experiments.jsonl")
    if os.path.exists(saved):
        s, _o = recorded_scores()
        assert s, "no scores attributed at all from a non-empty experiments.jsonl"

    doms, err = read_mix(os.path.join(DATA, "mix_v3.json"))
    if not err:
        assert "web" not in doms and "web_hq" in doms, doms
    # an empty domains map is an ERROR, never a quiet pass
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        json.dump({"domains": {}}, open(p, "w"))
        _d, e = read_mix(p)
        assert e, "an empty domains map read as a valid mix"

    untested = []
    for name, _a, _i, fn, broken in CHECKS:
        root = broken()
        try:
            state, evidence = fn(root)
            if state != FAIL:
                untested.append(f"{name} reported {state} on its broken world ({evidence})")
        except Exception as e:
            untested.append(f"{name} raised instead of reporting FAIL: {e}")
        finally:
            shutil.rmtree(root, ignore_errors=True)
    assert not untested, "checks that cannot be made to fail:\n  " + "\n  ".join(untested)
    print(f"harness self-test OK ({len(CHECKS)} checks each verified to FAIL on a broken world)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args else "all"
    if cmd in ("all", "check"):
        print("INVARIANTS  (a check that cannot run is a FAILURE, never a pass)")
        res = run_checks()
        bad = [n for n, s, *_ in res if s == FAIL]
    else:
        res, bad = [], []
    if cmd in ("all", "ledger"):
        print("\nLEDGER  (provenance and score on one line)")
        ledger()
    if cmd in ("all", "gaps"):
        print("\nGAPS  (stated out loud, never inferred from an absence)")
        gaps()
    if cmd in ("all", "stages"):
        print("\nSTAGES  (a stage is done when its falsifying measurement exists)")
        stages()
    if bad:
        print(f"\n{len(bad)} invariant(s) FAILED: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        sys.exit(main())
