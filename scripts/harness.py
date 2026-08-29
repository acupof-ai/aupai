#!/usr/bin/env python3
"""The single place this project's progress is checked, recorded, and advanced.

Two rules:

    A STAGE IS NOT DONE BECAUSE IT PRODUCED A FILE. IT IS DONE BECAUSE THE
    MEASUREMENT THAT WOULD FALSIFY IT EXISTS AND IS RECORDED.

    A CHECK WITHOUT A FAILING CASE IS NOT A CHECK. Every entry in CHECKS carries a
    `broken()` world; --selftest asserts the check reports FAIL on it. Reviewers
    mutated four separately-written guards in this repo and all four still PASSed.

    python scripts/harness.py            # check + status
    python scripts/harness.py check      # invariants only; exit 1 on any failure
    python scripts/harness.py ledger     # provenance and score, one row per checkpoint
    python scripts/harness.py gaps       # what is NOT measured, stated out loud
    python scripts/harness.py measure    # ...then GO MEASURE IT (full matrix, records itself)
    python scripts/harness.py --selftest # every check must fail on its broken world
"""

import argparse
import ast
import functools
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DATA = os.path.join(ROOT, "data")
SAMPLE_DOMAIN = "sample"  # the only corpus directory a git checkout ships

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


# --------------------------------------------------------------------------- facts


@functools.lru_cache(maxsize=None)
def cfg_default(field):
    """Read a Cfg field from train.py by AST: importing train.py pulls torch/fla/liger,
    and this file must run on CPU-only CI and laptops.

    Raises on a field it cannot read. Returning None let a one-token edit retire two
    checks: annotating `mix = "..."` as `mix: str = "..."` makes it an ast.AnnAssign,
    cfg_default returned None, and both mix checks reported SKIP with the text 'chosen
    on purpose' -- an intent nobody expressed -- while main() exited 0."""
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "Cfg":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and stmt.targets[0].id == field:
                    return ast.literal_eval(stmt.value)
                if isinstance(stmt, ast.AnnAssign) and getattr(stmt.target, "id", None) == field:
                    return ast.literal_eval(stmt.value)
    raise KeyError(f"train.py has no Cfg.{field}; the check that reads it cannot run")


def read_mix(path):
    """(domains, error). Never an empty dict: `"web" in {}` is False, so an unparseable
    mix would report a passing guard."""
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


@functools.lru_cache(maxsize=None)
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
# Must take the number carrying the %: "math-hard 37/1032 = 3.6%" holds three numbers and
# only the last is the score. `[^%]` stops the window bleeding into the NEXT metric --
# "math-hard deferred to bench stage; math-500 44.0%" scored 44.0 as a math-hard result.
SCORE_RE = re.compile(r"math-hard[^%]{0,40}?(\d+(?:\.\d+)?)\s*%")


def score_from(text):
    m = SCORE_RE.search(text or "")
    return float(m.group(1)) if m else None


def recorded_scores():
    """checkpoint -> (math-hard %, source), plus scores that matched no checkpoint.

    Must cover `scripts/eval_hard.sh <ckpt>`, which takes the checkpoint POSITIONALLY:
    matching only `--out ckpt_X.pt` dropped every score eval_hard.sh ever produced
    (k6_fone's 1.7% included) and listed those checkpoints as never measured."""
    scores, orphans = {}, []
    for row in experiments():
        s = score_from(str(row.get("result", "")))
        if s is None:
            continue
        cmd = str(row.get("cmd", ""))
        run = str(row.get("name", "?"))
        # Priority order; every form below is real in this log. INPUTS are excluded or
        # the score lands on the wrong checkpoint: rl_direct resumed ckpt_k4_11b_lr05.pt
        # and scored the RL output, crediting k4 with a number k4 did not produce.
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


def checkpoint_names(scores):
    """Every checkpoint this repo knows about: on disk, named in a command, OR carrying a
    score. The last source was missing and it silently dropped two real measurements --
    ckpt_rl_k4 4.1% and ckpt_sft_v5_hard 3.1% -- because `--name X` attributes a score to
    ckpt_X without `ckpt_X.pt` ever appearing in a command. 4.1% is HIGHER than the 3.6%
    the ledger was calling the best on record, so the one place progress is read from was
    hiding the top of its own table."""
    names = {os.path.basename(p)[: -len(".pt")] for p in glob.glob(os.path.join(ROOT, "ckpt_*.pt"))}
    for row in experiments():
        names.update(n[: -len(".pt")] for n in CKPT_RE.findall(str(row.get("cmd", ""))))
    return sorted(names | set(scores))


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
# Each check is (name, asserts, incident, run, broken). `run(root)` -> (state, evidence);
# `broken()` -> a temp root violating the condition, where run() must report FAIL.


def _tmp_repo(mix_obj=None):
    """A throwaway tree shaped like the repo, for a check to fail against.

    The mix goes at cfg_default("mix") -- the path the checks actually read. Writing it to
    a made-up data/mix_test.json instead meant both mix checks FAILed on their
    file-does-not-exist branch and their real logic was never once executed by --selftest."""
    import tempfile

    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "data", "corpus"), exist_ok=True)
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    if mix_obj is not None:
        p = os.path.join(d, cfg_default("mix"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(mix_obj, open(p, "w"))
    return d


def _tiny_tokenizer_json(eos_id=1, with_num=True):
    """A minimal WordLevel tokenizer that is VALID but LOSSY, so the round-trip and
    pinned-id checks have something real to reject (an absent file only hits SKIP)."""
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
    """The row is built by the REAL logger, not hand-written. The old broken world invented
    a `date` key to match the check's own bug, so both agreed on a schema exp.py has never
    written and the selftest passed on a check that could not fire."""
    import subprocess

    d = _tmp_repo()
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "exp.py"),
            "--root",
            d,
            "start",
            "--name",
            "killed_job",
            "--cmd",
            "x",
        ],
        check=True,
        capture_output=True,
    )
    p = os.path.join(d, "runs", "experiments.jsonl")
    rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    assert rows and rows[0]["status"] == "running", "exp.py start no longer opens a running row"
    rows[0]["started"] = "2020-01-01 00:00"
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    return d


def check_mix_not_unfiltered(root):
    doms, err = read_mix(os.path.join(root, cfg_default("mix")))
    if err:
        # NOT a pass: "could not check" must never read as "checked".
        return FAIL, f"cannot read the default mix: {err}"
    if "web" in doms:
        return FAIL, "the default mix names domain 'web' (the unfiltered 2,991,648-doc corpus)"
    return PASS, f"domains={doms}"


def _broken_mix():
    """Names 'web' AND has one domain resolving with the other absent -- the second half is
    what makes check_mix_shards report FAIL rather than the checkout SKIP."""
    d = _tmp_repo({"total_tokens": 1e9, "domains": {"web": {"weight": 0.5}, "gone": {"weight": 0.5}}})
    os.makedirs(os.path.join(d, "data", "corpus", "web"))
    open(os.path.join(d, "data", "corpus", "web", "a.jsonl"), "w").write("{}\n")
    return d


def check_mix_shards(root):
    doms, err = read_mix(os.path.join(root, cfg_default("mix")))
    if err:
        return FAIL, f"cannot read the default mix: {err}"
    corpus = os.path.join(root, "data", "corpus")
    missing = [d for d in doms if not glob.glob(os.path.join(corpus, d, "*.jsonl"))]
    if not missing:
        return PASS, f"all {len(doms)} domains have shards"
    # "No domain resolves" is not evidence of a checkout: a pod whose corpus was wiped,
    # unmounted or renamed resolves zero domains too, and that is the catastrophe this
    # check exists for. Discriminate on whether a corpus EXISTS at all -- a git checkout
    # ships data/corpus/sample and only that -- so the question becomes "is there a corpus
    # here that the mix fails to point at", which is the real one.
    other = [d for d in glob.glob(os.path.join(corpus, "*")) if os.path.basename(d) != SAMPLE_DOMAIN]
    if len(missing) == len(doms) and not any(glob.glob(os.path.join(d, "*.jsonl")) for d in other):
        return SKIP, "data/corpus holds only the shipped sample: this is a checkout, not the pod"
    return FAIL, f"no shards for {missing}"


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


MAX_TRACKED_MB = 5


def check_no_oversized_blob(root):
    """gitignore does not apply to an already-tracked path, so the pattern list was never the
    guard it reads as. Moving a file into data/_quarantine/ un-ignored it and committed 40MB;
    data/sft/*.jsonl was ignored while .parquet and .json were not. Both were patched after
    the fact. This fires on the NEXT one."""
    import subprocess

    p = subprocess.run(["git", "-C", root, "ls-tree", "-r", "-l", "HEAD"], capture_output=True, text=True)
    if p.returncode:
        return SKIP, "not a git repository (the pod checkout is not one)"
    big = []
    for ln in p.stdout.splitlines():
        f = ln.split(maxsplit=4)
        if len(f) == 5 and f[1] == "blob" and f[3].isdigit() and int(f[3]) > MAX_TRACKED_MB * 2**20:
            big.append(f"{f[4]} ({int(f[3]) / 2**20:.0f}MB)")
    if big:
        return FAIL, f"{len(big)} tracked blob(s) over {MAX_TRACKED_MB}MB: {', '.join(big[:4])}"
    return PASS, f"no tracked blob over {MAX_TRACKED_MB}MB"


def _broken_blob():
    """A real blob through real git plumbing. Synthesising an ls-tree line would repeat the
    mistake that left no_stale_running reading a key exp.py never wrote."""
    import subprocess

    d = _tmp_repo()
    big = os.path.join(d, "big.jsonl")
    with open(big, "wb") as f:
        f.write(b"x" * ((MAX_TRACKED_MB + 1) * 2**20))
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-f", "big.jsonl"],
        ["git", "commit", "-qm", "big"],
    ):
        subprocess.run(cmd, cwd=d, check=True, capture_output=True)
    return d


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
        # The field is `started`, which is what scripts/exp.py writes, in %Y-%m-%d %H:%M.
        # This read `date` -- a key exp.py has never emitted -- so every row raised into a
        # bare `except: continue` and the check returned PASS having examined ZERO rows,
        # with five runs up to three days stale. An unreadable date is now a FAIL: a check
        # that cannot see its subject must not report on it.
        try:
            t = time.mktime(time.strptime(str(r.get("started", "")), "%Y-%m-%d %H:%M"))
        except Exception:
            return FAIL, f"row {r.get('name', '?')!r} has no readable `started`: {r.get('started')!r}"
        age_h = (time.time() - t) / 3600
        if age_h > 24:
            rows.append(f"{r.get('name', '?')} {age_h:.0f}h")
    if rows:
        return FAIL, f"{len(rows)} killed mid-run and never closed: {', '.join(rows[:6])}"
    return PASS, "no run has been 'running' for over a day"


def check_guard_on_path(root):
    """The guard's logic was never what failed; its being ON THE PATH was. Reported here
    so deleting the call site shows up as a FAIL, not just a raise somewhere in CI."""
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
        "no_oversized_blob",
        f"no file over {MAX_TRACKED_MB}MB is tracked by git",
        "gitignore does not apply to already-tracked paths, so moving a file out of an ignored "
        "directory committed 40MB, and data/sft ignored only *.jsonl while the data arrived as "
        "*.parquet",
        check_no_oversized_blob,
        _broken_blob,
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
# A stage is done when its POSTCONDITION exists, never when its artifact does.

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
    print(f"  {'checkpoint':<26}{'on disk':>8}{'math-hard':>11}   source of the score")
    for n in checkpoint_names(scores):
        on_disk = os.path.exists(os.path.join(ROOT, f"{n}.pt"))
        s, src = scores.get(n, (None, None))
        sc = f"{s:.1f}%" if s is not None else "-"
        print(f"  {n:<26}{'yes' if on_disk else 'record':>8}{sc:>11}   {src or ''}")
    if orphans:
        # Named out loud: dropping unmatched scores silently is how five real
        # measurements became "never measured".
        print(f"\n  {len(orphans)} recorded score(s) matched NO checkpoint name:")
        for name, s, cmd in orphans:
            print(f"    {name}: {s:.1f}%   cmd={cmd!r}")
    if toks:
        print("\n  local tokenizers:")
        for k, v in toks.items():
            print(f"    {k:<26}{v}")


def gaps():
    scores, _orphans = recorded_scores()
    unmeasured = [n for n in checkpoint_names(scores) if n not in scores]
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


def measure(only=None, ngpu=None, tokenizer=None, dry=False, full=False):
    """CLOSE the gaps instead of reporting them.

    `gaps` naming a checkpoint as unmeasured, over and over, is not progress -- somebody
    still has to type the command, and on this project that somebody produced three
    write-ups and zero runs of the metric of record in one night. This runs the FULL
    matrix (scripts/eval_all.sh: math-hard, math-500, the MC suite, and the digit head for
    a FoNE checkpoint) on every checkpoint that is on disk and has no score, and writes the
    result back through scripts/exp.py so the ledger picks it up on the next read.

    Needs GPUs and the checkpoints, i.e. the pod. A checkpoint whose vocabulary does not
    match the tokenizer is recorded as a FAILURE, not skipped: eval_all.sh stops on that
    mismatch by design and an unrecorded stop is how a gap becomes permanent."""
    import subprocess

    scores, _ = recorded_scores()
    todo = [
        n
        for n in checkpoint_names(scores)
        if n not in scores and os.path.exists(os.path.join(ROOT, f"{n}.pt"))
    ]
    if only:
        todo = [n for n in todo if only in n]
    # Newest first, so the checkpoints anyone is waiting on land before the archaeology.
    # NOT capped: the whole point is that gaps stops listing the same names forever, and
    # math-hard alone is ~5 min per checkpoint, so even 38 of them is one idle night.
    todo.sort(key=lambda n: os.path.getmtime(os.path.join(ROOT, f"{n}.pt")), reverse=True)
    # gaps counts every unscored name; this can only close the ones whose weights exist. Say
    # which ones it cannot, or an empty todo reads as "nothing left" over gaps' remainder.
    absent = [n for n in checkpoint_names(scores) if n not in scores and n not in todo]
    if absent:
        print(
            f"  {len(absent)} unscored checkpoint(s) NOT on disk, so not closable here: {', '.join(absent)}"
        )
    if not todo:
        print("  nothing to measure: every checkpoint whose weights are here carries a score")
        return 0
    print(f"  {len(todo)} checkpoint(s) on disk with no score: {', '.join(todo)}")
    if dry:
        return 0
    env = {**os.environ, "NGPU": str(ngpu)} if ngpu else None
    for n in todo:
        ck = f"{n}.pt"
        # math-hard alone by default: it is the metric of record, the only thing score_from
        # reads, and the only thing that closes a gaps entry. The MC suite is ~30% of the
        # matrix's runtime and eval_all.sh's own comment says it sits at the chance line.
        if full:
            cmd = ["bash", os.path.join(HERE, "eval_all.sh"), ck] + ([tokenizer] if tokenizer else [])
        else:
            cmd = ["bash", os.path.join(HERE, "eval_hard.sh"), ck, str(ngpu or 6)]
        print(f"\n  === {ck} ===", flush=True)
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        # eval_all.sh writes runs/evalall_<ckpt>.log unconditionally, so a crash after the
        # math-hard stage still leaves the lines that DID land; re-capturing stdout loses them.
        log = os.path.join(ROOT, "runs", f"evalall_{n}.log")
        out = open(log, encoding="utf-8").read() if full and os.path.exists(log) else p.stdout + p.stderr
        # eval_all.sh:91's own extractor. Matching only "TOTAL" dropped the digit head and the
        # arithmetic rate -- the two things the matrix exists to report BESIDE the score.
        keep = re.compile(r"TOTAL|whole-number exact|^Average|% wrong|STOP:")
        hits = [ln.strip() for ln in out.splitlines() if keep.search(ln)]
        result = " | ".join(hits) if hits else f"eval produced no summary line (rc={p.returncode})"
        # `ok` has to mean what ledger and gaps mean by "measured", which is score_from() being
        # able to read a math-hard number out of this string. Deciding it on "some TOTAL line
        # appeared" lets measure record a gap as closed while gaps still lists it -- exactly the
        # failure this command exists to prevent.
        status = "ok" if p.returncode == 0 and score_from(result) is not None else "fail"
        print(f"  {result}")

        # start THEN done, not done alone: exp.py's done appends a row with cmd="" when no
        # running row matches, and recorded_scores attributes an empty-cmd row as
        # f"ckpt_{name}" -- so a done-only row here would score `ckpt_ckpt_k8` and leave the
        # gap open. The start row is also where the ledger reads provenance from.
        def exp(*argv):
            subprocess.run([sys.executable, os.path.join(HERE, "exp.py"), *argv], cwd=ROOT, check=True)

        exp("start", "--name", n, "--cmd", " ".join(cmd), "--hypothesis", "harness measure")
        exp("done", "--name", n, "--status", status, "--result", result)
    print(f"\n  measured {len(todo)}; re-run `harness.py gaps` to see what is left")
    return 0


def stages(res=None):
    res = {n: s for n, s, _e, _a, _i in (res or run_checks(quiet=True))}
    scores, _ = recorded_scores()
    print(f"  {'stage':<12}{'gates':>26}   postcondition")
    for name, gates, post in STAGES:
        bad = [g for g in gates if res.get(g) == FAIL]
        detail = f"BLOCKED: {','.join(bad)}" if bad else f"{len(gates)} gate(s) pass"
        print(f"  {name:<12}{detail:>26}   {post}")
    print(f"\n  eval postcondition: {len(scores)} checkpoint(s) carry a math-hard score.")


# ------------------------------------------------------------------------ selftest


def _demo():
    """Every check must FAIL on a world where its condition is violated."""
    import shutil

    assert score_from("math-hard 37/1032 = 3.6%") == 3.6, "took the numerator, not the percentage"
    assert score_from("math-hard deferred to the bench stage") is None, "invented a score"
    assert score_from("math-hard 1.7% (18/1032) vs k5 1.9%") == 1.7

    saved = os.path.join(ROOT, "runs", "experiments.jsonl")
    if os.path.exists(saved):
        s, _o = recorded_scores()
        assert s, "no scores attributed at all from a non-empty experiments.jsonl"

    doms, err = read_mix(os.path.join(DATA, "mix_v3.json"))
    if not err:
        assert "web" not in doms and "web_hq" in doms, doms
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
    # argparse, not a hand-rolled scan: two parsers (a bare-flag filter for the subcommand
    # and an index lookup for the values) let `harness.py --ngpu 7 measure` resolve cmd="7",
    # match no branch, print nothing and exit 0. A silent no-op is the failure mode this
    # file exists to prevent, and `choices` turns it into an error for free.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "cmd", nargs="?", default="all", choices=["all", "check", "ledger", "gaps", "measure", "stages"]
    )
    ap.add_argument("--only", help="measure: substring filter on the checkpoint name")
    ap.add_argument("--ngpu", help="measure: shards for eval_all.sh")
    ap.add_argument("--tokenizer", help="measure: the vocabulary these checkpoints were trained on")
    ap.add_argument("--dry", action="store_true", help="measure: list what would run")
    ap.add_argument("--full", action="store_true", help="measure: the whole matrix, not just math-hard")
    ap.add_argument("--selftest", action="store_true", help="every check must FAIL on its broken world")
    a = ap.parse_args()
    if a.selftest:
        return _demo() or 0
    cmd = a.cmd
    res = []
    if cmd in ("all", "check"):
        print("INVARIANTS  (a check that cannot run is a FAILURE, never a pass)")
        res = run_checks()
        bad = [n for n, s, *_ in res if s == FAIL]
    else:
        bad = []
    if cmd in ("all", "ledger"):
        print("\nLEDGER  (provenance and score on one line)")
        ledger()
    if cmd in ("all", "gaps"):
        print("\nGAPS  (stated out loud, never inferred from an absence)")
        gaps()
    if cmd == "measure":
        return measure(only=a.only, ngpu=a.ngpu, tokenizer=a.tokenizer, dry=a.dry, full=a.full)
    if cmd in ("all", "stages"):
        print("\nSTAGES  (a stage is done when its falsifying measurement exists)")
        stages(res)
    if bad:
        print(f"\n{len(bad)} invariant(s) FAILED: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
