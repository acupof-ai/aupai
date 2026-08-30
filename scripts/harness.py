#!/usr/bin/env python3
"""The single place this project's progress is checked, recorded, and advanced.

Two rules:
- A stage is done when the measurement that would falsify it exists and is recorded, not when it produced a file.
- A check without a failing case is not a check: every CHECKS entry carries broken(), and --selftest asserts FAIL on it.

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
import corpus_fingerprint as cfp  # noqa: E402
import pod_drift  # noqa: E402
DATA = os.path.join(ROOT, "data")
SAMPLE_DOMAIN = "sample"  # the only corpus directory a git checkout ships

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


# --------------------------------------------------------------------------- facts


@functools.lru_cache(maxsize=None)
def cfg_default(field):
    """Read a Cfg field from train.py by AST -- importing train.py pulls torch, and this
    file must run on CPU-only CI. Raises on a field it cannot read: returning None once
    let a one-token annotation edit retire two checks while main() exited 0."""
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
# only the last is the score. `[^%]` stops the window bleeding into the next metric.
SCORE_RE = re.compile(r"math-hard[^%]{0,40}?(\d+(?:\.\d+)?)\s*%")


def score_from(text):
    m = SCORE_RE.search(text or "")
    return float(m.group(1)) if m else None


def produced_checkpoint(cmd, run_name):
    """The checkpoint a run's cmd produced, or None. Priority: --out, then --name,
    then a single free ckpt_*.pt in the cmd. INPUTS are excluded: rl_direct resumed
    ckpt_k4 and scored its own output, and crediting k4 with that score is the
    loudest wrong-attribution bug this ledger had."""
    inputs = set(re.findall(r"--(?:resume|sft_path|tokenizer|ckpt)\s+(\S+)", cmd))
    m = re.search(r"--out\s+(ckpt_[A-Za-z0-9_.-]+)\.pt", cmd)
    if m:
        return m.group(1)
    if m := re.search(r"--name\s+([A-Za-z0-9_.-]+)", cmd):
        return f"ckpt_{m.group(1)}"
    if not cmd.strip():
        return f"ckpt_{run_name}"
    free = [n for n in CKPT_RE.findall(cmd) if n not in inputs]
    if len(free) == 1:
        return free[0][: -len(".pt")]
    return None


def recorded_scores():
    """checkpoint -> (math-hard %, source), plus scores that matched no checkpoint.
    eval_hard.sh takes the checkpoint positionally (matching only --out dropped every
    score it produced); inputs are excluded, or resuming ckpt_A credits A with the
    output's score."""
    scores, orphans = {}, []
    for row in experiments():
        s = score_from(str(row.get("result", "")))
        if s is None:
            continue
        cmd = str(row.get("cmd", ""))
        run = str(row.get("name", "?"))
        cand = produced_checkpoint(cmd, run)
        if cand is None:
            orphans.append((run, s, cmd[:60]))
            continue
        scores.setdefault(cand, (s, run))
    return scores, orphans


def checkpoint_names(scores):
    """Every checkpoint this repo knows about: on disk, named in a command, OR carrying a
    score. The last source once silently dropped the top of the ledger's own table."""
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
    """A throwaway tree shaped like the repo, for a check to fail against. The mix goes at
    cfg_default("mix") -- the path the checks actually read, not a made-up one."""
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
    """The row is built by the REAL logger, not hand-written -- a hand-written row shares
    the check's own schema assumptions."""
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


def _gpu_present():
    """Whether this machine can train. The strict branch of mix_shards_present guards the
    pod; a dev box with a partial corpus is normal. HARNESS_GPU_PRESENT=1/0 overrides -- the
    selftest forces 1 so the broken world exercises the strict branch."""
    forced = os.environ.get("HARNESS_GPU_PRESENT")
    if forced is not None:
        return forced == "1"
    return bool(glob.glob("/dev/nvidia[0-9]*"))


def check_mix_shards(root):
    doms, err = read_mix(os.path.join(root, cfg_default("mix")))
    if err:
        return FAIL, f"cannot read the default mix: {err}"
    corpus = os.path.join(root, "data", "corpus")
    missing = [d for d in doms if not glob.glob(os.path.join(corpus, d, "*.jsonl"))]
    if not missing:
        return PASS, f"all {len(doms)} domains have shards"
    # Strictness follows the ability to train: a GPU box with a missing domain is about to
    # tokenize on missing data; dev boxes ship no corpus, and a permanent red is no signal.
    if not _gpu_present():
        return SKIP, f"no GPU on this machine: {len(missing)}/{len(doms)} domains lack shards (not the pod)"
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
    """gitignore does not cover already-tracked paths, so the pattern list was never the
    guard. This fires on the next one."""
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
    """A real blob through real git plumbing -- a synthesised ls-tree line shares the
    check's own assumptions."""
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
        # The field is `started`, in exp.py's %Y-%m-%d %H:%M format. An unreadable date is
        # a FAIL: a check that cannot see its subject must not report on it.
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


def check_no_ghost_running(root):
    # no_stale_running's blind spot: a run that FINISHED but was never recorded stays
    # 'running' for up to 24h. On the pod, a running row older than 2h with no live
    # process is a ghost -- close it with exp.py done. Pod-only: processes live there.
    if not pod_drift.is_pod(root):
        return SKIP, "dev checkout; process state lives on the pod"
    p = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return SKIP, "runs/experiments.jsonl not present"
    import subprocess

    ghosts = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "running":
                continue
            try:
                t = time.mktime(time.strptime(str(r.get("started", "")), "%Y-%m-%d %H:%M"))
            except Exception:
                return FAIL, f"row {r.get('name', '?')!r} has no readable `started`: {r.get('started')!r}"
            if (time.time() - t) / 3600 < 2:
                continue  # grace: a launched run takes time to appear in ps
            name = r.get("name", "")
            if name and not subprocess.run(["pgrep", "-f", name], capture_output=True, text=True).stdout.strip():
                ghosts.append(f"{name} (started {r.get('started')})")
    if ghosts:
        return FAIL, f"running rows with no live process: {', '.join(ghosts[:6])}; close with exp.py done"
    return PASS, "every running row has a live process"


def check_guard_on_path(root):
    """Deleting the guard's call site must show up as a FAIL, not just a raise in CI."""
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


# --------------------------------------------------------------------------- facts
#
# Measurements live in facts/*.json, never in prose. A fact carries its measurement
# config -- a value without one is this project's repeated failure class.

FACTS_DIR = os.path.join(ROOT, "facts")
FACT_REQUIRED = {"id", "value", "measured", "source", "config", "uncertainty", "status"}
FACT_STATUS = {"measured", "recorded", "unmeasured", "retracted"}
FACT_NEEDS_CLAIM = {"unmeasured", "retracted"}
FACT_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
FACT_SOURCE_PATH = re.compile(r"(?:scripts|docs|eval|datagen|filters|mathbank|algorithms|workflows)/[\w./-]+")


def check_facts_well_formed(root):
    facts_dir = os.path.join(root, "facts")
    if not os.path.isdir(facts_dir):
        return FAIL, "facts/ does not exist -- measurements have nowhere to carry their config"
    files = sorted(glob.glob(os.path.join(facts_dir, "*.json")))
    if not files:
        return FAIL, "facts/ holds no *.json"
    errors, ids, entries = [], {}, []
    for p in files:
        fn = os.path.basename(p)
        try:
            lst = json.load(open(p, encoding="utf-8"))["facts"]
            assert isinstance(lst, list) and lst
        except Exception as e:
            errors.append(f"{fn}: no readable non-empty 'facts' list ({e})")
            continue
        for e in lst:
            if not isinstance(e, dict):
                errors.append(f"{fn}: entry is not an object")
                continue
            tag = f"{fn}#{e.get('id', '?')}"
            if missing := FACT_REQUIRED - e.keys():
                errors.append(f"{tag}: missing {sorted(missing)}")
                continue
            if e["status"] not in FACT_STATUS:
                errors.append(f"{tag}: bad status {e['status']!r}")
            if not isinstance(e["config"], dict) or not e["config"]:
                errors.append(f"{tag}: config must be a non-empty object")
            if not FACT_DATE_RE.fullmatch(str(e["measured"])):
                errors.append(f"{tag}: measured must be YYYY-MM-DD, got {e['measured']!r}")
            if e["status"] in FACT_NEEDS_CLAIM:
                for k in ("claim", "audit", "refuted_by"):
                    if not e.get(k):
                        errors.append(f"{tag}: {e['status']} fact needs {k}")
            if e["id"] in ids:
                errors.append(f"duplicate id {e['id']!r} in {fn} and {ids[e['id']]}")
            ids[e["id"]] = fn
            for m in FACT_SOURCE_PATH.findall(str(e["source"])):
                if not os.path.exists(os.path.join(root, m)):
                    errors.append(f"{tag}: source path {m} does not exist")
            entries.append((fn, e))
    agents = os.path.join(root, "AGENTS.md")
    prose = open(agents, encoding="utf-8").read() if os.path.exists(agents) else ""
    for fn, e in entries:
        for phrase in e.get("guard_phrases", []):
            if phrase in prose:
                errors.append(f"AGENTS.md asserts {phrase!r}, recorded as {e['status']} in {fn}#{e['id']}")
    for p in files:
        if os.path.basename(p) not in prose:
            errors.append(f"AGENTS.md never mentions {os.path.basename(p)} -- an orphan fact file")
    for m in re.findall(r"facts/[\w.-]+\.json", prose):
        if not os.path.exists(os.path.join(root, m)):
            errors.append(f"AGENTS.md cites {m}, which does not exist")
    if errors:
        head = "; ".join(errors[:5])
        return FAIL, head + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else "")
    return PASS, f"{len(entries)} facts in {len(files)} files, every entry carries its config"


def _broken_facts():
    """The REAL facts files and REAL AGENTS.md, with one entry's config deleted. A
    hand-written file would share the check's own assumptions."""
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "facts"))
    for f in glob.glob(os.path.join(FACTS_DIR, "*.json")):
        shutil.copy(f, os.path.join(d, "facts"))
    obj = json.load(open(os.path.join(d, "facts", "tokenizer.json"), encoding="utf-8"))
    del obj["facts"][0]["config"]
    json.dump(obj, open(os.path.join(d, "facts", "tokenizer.json"), "w"))
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    return d


ENTRY_SCRIPT_RE = re.compile(r"(?:scripts|eval|datagen|mathbank)/[\w.-]+\.(?:sh|py)|run_ddp\.sh")


def check_entrypoints_ran(root):
    """A cited script that does not exist is FAIL -- the doc is rotten. A command tried and
    never ok is WARN -- a to-do fixed by running it. Zero log matches is skipped: never
    tried is not tried and failed (wrappers log the inner command)."""
    agents = os.path.join(root, "AGENTS.md")
    if not os.path.exists(agents):
        return SKIP, "AGENTS.md not present"
    log = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(log):
        return SKIP, "runs/experiments.jsonl not present"
    rows = []
    for line in open(log, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    if not rows:
        # An empty log is the post-reset state, not a clean bill: zero matches must read
        # as "never tried", never as PASS.
        return SKIP, "runs/experiments.jsonl has no rows"
    missing, stale = [], []
    for line in open(agents, encoding="utf-8"):
        if "|" not in line or not ENTRY_SCRIPT_RE.search(line):
            continue
        # Task-cell tokens catch attempts logged under an inner command (the wrapper is
        # invisible to the log).
        task_tokens = {t for t in re.split(r"[^a-z0-9]+", line.split("|")[1].lower()) if len(t) >= 5}
        for s in sorted(set(re.findall(r"[\w/.-]+\.(?:sh|py)", line))):
            if not os.path.exists(os.path.join(root, s)):
                missing.append(s)
                continue
            matched = [
                r
                for r in rows
                if s in str(r.get("cmd", ""))
                or any(
                    t in str(r.get("name", "")).lower() or t in str(r.get("cmd", "")).lower()
                    for t in task_tokens
                )
            ]
            if matched and not any(r.get("status") == "ok" for r in matched):
                latest = matched[-1]
                finding = " ".join(str(latest.get("finding", "")).split())[:120]
                stale.append(
                    f"{s}: {len(matched)} run(s), never ok, latest={latest.get('status')!r}"
                    + (f" -- {finding}" if finding else "")
                )
    if missing:
        return FAIL, f"entry-point table cites script(s) not in the repo: {missing}"
    if stale:
        return WARN, "; ".join(stale[:4])
    return PASS, "every tried entry-point command has at least one ok run"


def check_entrypoints_table_present(root):
    """The entry-point table is the doc's contract with the repo. Zero script-citing rows
    is the cfg_default failure shape: two corpus invariants reported SKIP 'chosen on
    purpose' and check exited 0 -- an empty list silences the guard. FAIL, never SKIP."""
    agents = os.path.join(root, "AGENTS.md")
    if not os.path.exists(agents):
        return SKIP, "AGENTS.md not present"
    n = sum(1 for line in open(agents, encoding="utf-8") if "|" in line and ENTRY_SCRIPT_RE.search(line))
    if n == 0:
        return FAIL, "no entry-point row cites a script -- an empty list silences the guard (cfg_default shape)"
    return PASS, f"{n} entry-point row(s) cite scripts"


def _broken_entrypoint():
    """The REAL AGENTS.md with one table row added citing a script that does not exist -- the
    FAIL tier. The WARN tier is live in the real repo (run_ablation.sh), so it needs no
    synthetic world. The log row is written by the REAL logger with --root d, so the check
    runs instead of SKIPping on an absent log."""
    import shutil, subprocess

    d = _tmp_repo()
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    with open(os.path.join(d, "AGENTS.md"), "a") as f:
        f.write("| Ghost | `python scripts/ghost_command.sh` |\n")
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "exp.py"),
            "--root",
            d,
            "start",
            "--name",
            "broken_world",
            "--cmd",
            "./run_ddp.sh --mix x",
        ],
        check=True,
        capture_output=True,
    )
    return d


def _broken_entrypoints_table():
    """The REAL AGENTS.md with every script-citing table row deleted -- the check must
    FAIL, not SKIP. The doc carries several script-citing tables, so deleting only the
    entry-point block would leave the count above zero."""
    d = _tmp_repo()
    lines = open(os.path.join(ROOT, "AGENTS.md"), encoding="utf-8").read().splitlines(keepends=True)
    out = [ln for ln in lines if not ("|" in ln and ENTRY_SCRIPT_RE.search(ln))]
    with open(os.path.join(d, "AGENTS.md"), "w", encoding="utf-8") as f:
        f.writelines(out)
    return d


# --------------------------------------------------------------------------- docs layout
#
# docs/ has three subdirectories and zero .md files at its root. Research docs carry
# question/status/source frontmatter; lessons cite facts by facts/<file>.json#<id>.

DOCS_SUBDIRS = ("lessons", "audits")
FRONTMATTER_KEYS = ("question", "status", "source")
FRONTMATTER_STATUS = ("measured", "recorded", "open", "retracted")
FACT_REF_RE = re.compile(r"facts/([\w.-]+)\.json#([\w.]+)")
CMD_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\n(.*?)```", re.S)
CMD_PATH_RE = re.compile(r"(?<![\w.-])([\w./-]+\.(?:sh|py))(?![\w.-])")


def _frontmatter(path):
    text = open(path, encoding="utf-8").read()
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None
    fields = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()
    return fields


def check_docs_root_clean(root):
    docs = os.path.join(root, "docs")
    if not os.path.isdir(docs):
        return FAIL, "docs/ missing"
    stray = sorted(f for f in os.listdir(docs) if f.endswith(".md") and os.path.isfile(os.path.join(docs, f)))
    if stray:
        return FAIL, f"docs/ root holds .md files: {stray[:5]} -- classify into lessons/, audits/, standards/"
    return PASS, "docs/ root holds no .md files"


def check_lessons_frontmatter(root):
    docs = os.path.join(root, "docs")
    if not os.path.isdir(docs):
        return FAIL, "docs/ missing"
    problems, n = [], 0
    for sub in DOCS_SUBDIRS:
        d = os.path.join(docs, sub)
        if not os.path.isdir(d):
            problems.append(f"docs/{sub}/ missing")
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".md") or f.startswith("README"):
                continue
            n += 1
            fm = _frontmatter(os.path.join(d, f))
            if fm is None:
                problems.append(f"docs/{sub}/{f}: no frontmatter")
                continue
            missing = [k for k in FRONTMATTER_KEYS if not fm.get(k)]
            if missing:
                problems.append(f"docs/{sub}/{f}: missing {missing}")
            elif fm["status"] not in FRONTMATTER_STATUS:
                problems.append(f"docs/{sub}/{f}: bad status {fm['status']!r}")
    if problems:
        return FAIL, "; ".join(problems[:5])
    if n == 0:
        return FAIL, "no lesson/audit files found -- an empty list silences the guard"
    return PASS, f"{n} research docs carry question/status/source"


def check_fact_refs(root):
    facts_dir = os.path.join(root, "facts")
    if not os.path.isdir(facts_dir):
        return FAIL, "facts/ missing"
    index = {}
    for f in glob.glob(os.path.join(facts_dir, "*.json")):
        try:
            obj = json.load(open(f, encoding="utf-8"))
            index[os.path.basename(f)] = {e["id"]: e for e in obj.get("facts", [])}
        except Exception as e:
            return FAIL, f"cannot parse {f}: {e}"
    bad, retracted, n = [], [], 0
    for sub in DOCS_SUBDIRS:
        d = os.path.join(root, "docs", sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".md"):
                continue
            for m in FACT_REF_RE.finditer(open(os.path.join(d, f), encoding="utf-8").read()):
                n += 1
                fname, fid = m.group(1) + ".json", m.group(2)
                if fname not in index:
                    bad.append(f"docs/{sub}/{f}: facts/{fname}.json does not exist")
                elif fid not in index[fname]:
                    bad.append(f"docs/{sub}/{f}: {fid} not in facts/{fname}.json")
                elif index[fname][fid].get("status") == "retracted":
                    retracted.append(f"docs/{sub}/{f} cites retracted {fname}#{fid}")
    if bad:
        return FAIL, "; ".join(bad[:5])
    if retracted:
        return WARN, f"{n} citation(s); " + "; ".join(retracted[:4])
    return PASS, f"{n} fact citation(s) all resolve"


def _broken_docs_root():
    """The REAL docs tree plus one stray .md at the root -- the rule is zero .md files
    directly under docs/, so any new root file FAILs until classified."""
    import shutil

    d = _tmp_repo()
    shutil.copytree(os.path.join(ROOT, "docs"), os.path.join(d, "docs"))
    open(os.path.join(d, "docs", "stray.md"), "w").write("# stray\n")
    return d


def _broken_lessons_fm():
    """The REAL docs tree with kept_methods.md's frontmatter stripped -- the check must
    FAIL on the missing fields, not on a hand-written file sharing the check's own
    assumptions."""
    import shutil

    d = _tmp_repo()
    shutil.copytree(os.path.join(ROOT, "docs"), os.path.join(d, "docs"))
    p = os.path.join(d, "docs", "lessons", "kept_methods.md")
    text = open(p, encoding="utf-8").read()
    if text.startswith("---\n"):
        text = text[text.find("\n---", 4) + 4 :].lstrip("\n")
    open(p, "w", encoding="utf-8").write(text)
    return d


def _broken_fact_ref():
    """The REAL lessons and facts trees, with one citation to a nonexistent fact appended
    to a real lesson."""
    import shutil

    d = _tmp_repo()
    shutil.copytree(os.path.join(ROOT, "docs"), os.path.join(d, "docs"))
    shutil.copytree(os.path.join(ROOT, "facts"), os.path.join(d, "facts"))
    with open(os.path.join(d, "docs", "lessons", "kept_methods.md"), "a", encoding="utf-8") as f:
        f.write("\n\nSee facts/tokenizer.json#tok.does_not_exist.\n")
    return d


def check_doc_commands(root):
    """Every .sh/.py cited in an AGENTS.md command block exists. A documented command
    that does not run is worse than none. Only fenced blocks are scanned; prose citations
    and data files are not (a missing data file is caught at runtime by the mix guard)."""
    agents = os.path.join(root, "AGENTS.md")
    if not os.path.exists(agents):
        return SKIP, "AGENTS.md not present"
    missing = set()
    for block in CMD_BLOCK_RE.findall(open(agents, encoding="utf-8").read()):
        for tok in CMD_PATH_RE.findall(block):
            if not os.path.exists(os.path.join(root, tok)):
                missing.add(tok)
    if missing:
        return FAIL, f"command block cites script(s) not in the repo: {sorted(missing)[:5]}"
    return PASS, "every script cited in a command block exists"


def _broken_doc_commands():
    """The REAL AGENTS.md with one fenced block appended citing a script that does not
    exist -- the FAIL tier."""
    import shutil

    d = _tmp_repo()
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    with open(os.path.join(d, "AGENTS.md"), "a", encoding="utf-8") as f:
        f.write("\n```bash\npython scripts/nonexistent_command.sh --flag\n```\n")
    return d


def check_score_matrix(root):
    """Every status=ok training run has a score-matrix record for the checkpoint it
    produced. 'Trained but not scored' must be impossible: an ok row with no matrix
    record is a FAIL, not a gap. Eval/measure rows produce no checkpoint and are
    exempt. The matrix is runs/score_matrix.jsonl, one record per scored checkpoint."""
    log = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(log):
        return SKIP, "runs/experiments.jsonl not present"
    rows = []
    for line in open(log, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    if not rows:
        return SKIP, "experiments.jsonl has no rows"
    scored = set()
    matrix = os.path.join(root, "runs", "score_matrix.jsonl")
    if os.path.exists(matrix):
        for line in open(matrix, encoding="utf-8"):
            try:
                scored.add(json.loads(line).get("ckpt"))
            except Exception:
                pass
    missing = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        cmd = str(r.get("cmd", ""))
        if not any(t in cmd for t in ("train.py", "sft_math", "rlvr", "run_ddp.sh", "run_sft.sh")):
            continue
        cand = produced_checkpoint(cmd, str(r.get("name", "?")))
        if cand and f"{cand}.pt" not in scored:
            missing.append(cand)
    if missing:
        return FAIL, f"ok training run(s) with no score-matrix record: {sorted(set(missing))[:5]}"
    return PASS, "every ok training run has a score-matrix record"


def _broken_score_matrix():
    """A REAL ok training row, written by the real exp.py, with no score-matrix
    record -- the FAIL tier."""
    import subprocess

    d = _tmp_repo()
    for argv in (
        ["start", "--name", "x", "--cmd", "./run_ddp.sh --name x"],
        ["done", "--name", "x", "--status", "ok", "--result", "done"],
    ):
        subprocess.run(
            [sys.executable, os.path.join(HERE, "exp.py"), "--root", d, *argv],
            check=True, capture_output=True,
        )
    return d


def _provenance_fingerprints(path, domains):
    """{domain: fingerprint} parsed from data/PROVENANCE.md. A domain block is a heading
    whose text contains the domain as a whole token; its fingerprint is a
    `fingerprint: <16-hex>` line in that section. Only mix domains are attributed, so a
    fingerprint under an unrelated heading ('SFT-math candidates') cannot hijack 'math'."""
    if not os.path.isfile(path):
        return {}
    domset = {d.lower() for d in domains}
    out, section = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            section = m.group(1)
            continue
        fp = re.search(r"fingerprint[:\s=]+([0-9a-f]{16})", line, re.I)
        if fp and section:
            for tok in re.findall(r"[A-Za-z0-9_]+", section):
                if tok.lower() in domset:
                    out[tok.lower()] = fp.group(1)
    return out


def check_corpus_fp(root):
    """Every domain the default mix names must (a) carry a build-time fingerprint
    (build_corpus.py stamps build_corpus_stats.json) matching the live directory, and
    (b) have a provenance block in data/PROVENANCE.md whose recorded fingerprint also
    matches. A missing stamp or a missing block is FAIL, not SKIP: an unstamped or
    unrecorded domain cannot be distinguished from a swapped-in one -- the voided 0.2b
    run trained on CCI3 shards under web_hq's name, and fineweb2 web_hq was lost with no
    record of how it was built. Domains with no directory on this machine are
    mix_shards_present's beat, not this one."""
    doms, err = read_mix(os.path.join(root, cfg_default("mix")))
    if err:
        return FAIL, f"cannot read the default mix: {err}"
    corpus = os.path.join(root, "data", "corpus")
    present = [d for d in doms if os.path.isdir(os.path.join(corpus, d))]
    if not present:
        return SKIP, "no mix domain has a directory on this machine"
    prov = _provenance_fingerprints(os.path.join(root, "data", "PROVENANCE.md"), doms)
    problems, ok = [], 0
    for dom in present:
        stats = os.path.join(corpus, dom, "build_corpus_stats.json")
        try:
            with open(stats, encoding="utf-8") as f:
                stamped = json.load(f).get("fingerprint")
        except Exception:
            stamped = None
        live = cfp.fp_domain(dom, corpus)
        dom_ok = True
        if not stamped:
            problems.append(f"{dom}: no build-time fingerprint")
            dom_ok = False
        elif live != stamped:
            problems.append(f"{dom}: stamped {stamped} != live {live}")
            dom_ok = False
        if dom not in prov:
            problems.append(f"{dom}: no PROVENANCE.md block")
            dom_ok = False
        elif prov[dom] != live:
            problems.append(f"{dom}: PROVENANCE.md {prov[dom]} != live {live}")
            dom_ok = False
        if dom_ok:
            ok += 1
    if problems:
        return FAIL, f"{ok}/{len(present)} match; " + "; ".join(problems[:3])
    return PASS, f"{ok}/{len(present)} mix domains match their build-time and PROVENANCE.md fingerprints"


def check_pod_drift(root):
    # The pod is not a git repo: its files must match the committed manifest. CI gates the
    # manifest against HEAD. A dev checkout skips both -- uncommitted changes are normal there.
    if pod_drift.is_pod(root):
        ok, evidence = pod_drift.check_pod(root)
        return (PASS if ok else FAIL), evidence
    if os.environ.get("CI") == "true":
        ok, evidence = pod_drift.check_head(root)
        return (PASS if ok else FAIL), evidence
    return SKIP, "dev checkout; CI gates manifest freshness, the pod gates file drift"


def _broken_ghost_running():
    """The REAL experiment log plus a fake running row whose process cannot exist: the
    pod-only ghost check must see it. The 2h grace is passed by backdating the row."""
    import shutil

    d = _tmp_repo()
    shutil.copy(os.path.join(ROOT, "runs", "experiments.jsonl"), os.path.join(d, "runs", "experiments.jsonl"))
    with open(os.path.join(d, "runs", "experiments.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"started": "2026-08-29 00:00", "name": "ghost_run_xyz", "status": "running"}) + "\n")
    return d


def _broken_pod_drift():
    """The REAL manifest plus one REAL scoped file, mutated: the pod gate must see the
    mismatch. The CI branch cannot be exercised here -- the selftest world has no .git."""
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "scripts"))
    shutil.copy(
        os.path.join(ROOT, "data", "pod_head_manifest.txt"),
        os.path.join(d, "data", "pod_head_manifest.txt"),
    )
    shutil.copy(os.path.join(ROOT, "scripts", "harness.py"), os.path.join(d, "scripts", "harness.py"))
    with open(os.path.join(d, "scripts", "harness.py"), "a", encoding="utf-8") as f:
        f.write("\n# broken world drift\n")
    return d


def _broken_corpus_fp():
    """The REAL default mix copied into the broken world, with two of its real-named domains
    present: one stamped correctly and then drifted, one carrying no stamp at all. Both tiers
    must report FAIL, and the evidence must carry the denominator -- '1 domain(s) match' once
    read as all-green when it was 1 of 7."""
    import shutil

    d = _tmp_repo()
    mix_rel = cfg_default("mix")
    shutil.copy(os.path.join(ROOT, mix_rel), os.path.join(d, mix_rel))
    doms, _ = read_mix(os.path.join(ROOT, mix_rel))
    corpus = os.path.join(d, "data", "corpus")
    real = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", doms[0], "*.jsonl")))
    real = real or sorted(glob.glob(os.path.join(ROOT, "data", "**", "*.jsonl"), recursive=True))
    drifted = os.path.join(corpus, doms[0])
    os.makedirs(drifted)
    shutil.copy(real[0], os.path.join(drifted, "real_shard.jsonl"))
    with open(os.path.join(drifted, "build_corpus_stats.json"), "w") as f:
        json.dump({"fingerprint": cfp.fp_domain(doms[0], corpus)}, f)
    with open(os.path.join(drifted, "real_shard.jsonl"), "a", encoding="utf-8") as f:
        f.write('{"question": "broken world drift", "output": "1"}\n')
    unstamped = os.path.join(corpus, doms[1])
    os.makedirs(unstamped)
    shutil.copy(real[0], os.path.join(unstamped, "real_shard.jsonl"))
    # A PROVENANCE.md at the real path: doms[0]'s block records a WRONG fingerprint
    # (the mismatched-block tier), doms[1] has no block at all.
    with open(os.path.join(d, "data", "PROVENANCE.md"), "w") as f:
        f.write(f"# provenance\n\n## {doms[0]}\n\nfingerprint: 0000000000000000\n")
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
        "gitignore does not cover already-tracked paths; a 40MB file committed once because of it",
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
        "no_ghost_running",
        "a running row older than 2h has a live process (pod only)",
        "a finished-but-unrecorded run looked alive for up to 24h under no_stale_running alone",
        check_no_ghost_running,
        _broken_ghost_running,
    ),
    (
        "guard_on_path",
        "train.py main() actually calls the mix guard",
        "the guard lived in a wrapper while the documented entry point bypassed it",
        check_guard_on_path,
        _broken_guard,
    ),
    (
        "facts_well_formed",
        "every facts/*.json entry carries its measurement config, and AGENTS.md asserts no guarded phrase",
        "a value without its measurement config is the project's repeated failure class",
        check_facts_well_formed,
        _broken_facts,
    ),
    (
        "entrypoints_ran",
        "every script the entry-point table cites exists (FAIL); every tried one has an ok run (WARN)",
        "run_ablation.sh shipped as the AttnRes A/B entry while its rows read killed and OOM-fail",
        check_entrypoints_ran,
        _broken_entrypoint,
    ),
    (
        "entrypoints_table_present",
        "AGENTS.md contains at least one entry-point row citing a script",
        "cfg_default: two corpus invariants reported SKIP 'chosen on purpose' and check exited 0 -- "
        "an empty list silences the guard",
        check_entrypoints_table_present,
        _broken_entrypoints_table,
    ),
    (
        "docs_root_clean",
        "zero .md files directly under docs/ -- research, audits, standards live in subdirs",
        "docs/ was flat with audit_*/data_recipe*/exp_* mixed at the root, no rule and no check",
        check_docs_root_clean,
        _broken_docs_root,
    ),
    (
        "lessons_have_frontmatter",
        "every docs/lessons|audits/*.md (README excepted) carries question/status/source",
        "research docs carried no machine-checkable contract; a doc could answer no question and cite nothing",
        check_lessons_frontmatter,
        _broken_lessons_fm,
    ),
    (
        "fact_refs_resolve",
        "every facts/<file>.json#<id> citation resolves; citing a retracted fact WARNs",
        "the citation regex dropped .json from the path, so this check passed on zero real citations until its fix on 2026-08-30 (36 citations now resolve)",
        check_fact_refs,
        _broken_fact_ref,
    ),
    (
        "corpus_fp_matches",
        "every domain the default mix names carries a build-time fingerprint matching its live directory; a missing stamp is FAIL, not SKIP",
        "the voided 0.2b run trained on CCI3 shards under web_hq's name and no fingerprint said so -- an unstamped domain cannot be distinguished from a swapped-in one",
        check_corpus_fp,
        _broken_corpus_fp,
    ),
    (
        "pod_drift",
        "pod files match the committed manifest; in CI, the manifest matches HEAD",
        "the pod ran 142 files behind HEAD and its harness had never run the full check set -- training happened under rules the repo no longer had",
        check_pod_drift,
        _broken_pod_drift,
    ),
    (
        "doc_commands_exist",
        "every .sh/.py cited in an AGENTS.md command block exists",
        "a documented command that does not run is worse than none",
        check_doc_commands,
        _broken_doc_commands,
    ),
    (
        "score_matrix_present",
        "every status=ok training run has a score-matrix record for its checkpoint",
        "a base checkpoint reads zero on every generative eval, and an unscored ok run is invisible -- the matrix is the only score that moves on a base",
        check_score_matrix,
        _broken_score_matrix,
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
    ("pretrain", ["guard_on_path", "no_stale_running", "score_matrix_present"], "checkpoint carries vocab_id; val loss recorded"),
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
            if state in (FAIL, WARN):
                print(f"         asserts: {asserts}")
            if state == FAIL:
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
        # Dropping unmatched scores silently turns real measurements into "never measured".
        print(f"\n  {len(orphans)} recorded score(s) matched NO checkpoint name:")
        for name, s, cmd in orphans:
            print(f"    {name}: {s:.1f}%   cmd={cmd!r}")
    if toks:
        print("\n  local tokenizers:")
        for k, v in toks.items():
            print(f"    {k:<26}{v}")


def gaps():
    """An unmeasured checkpoint whose weights are gone is not a gap, it is history.

    Listing the two together made `gaps` nag about names nobody can ever score, which is
    how a to-do list stops being read. Only the ones whose weights are here are actionable,
    and `measure` closes exactly those."""
    scores, _orphans = recorded_scores()
    unmeasured = [n for n in checkpoint_names(scores) if n not in scores]
    here = [n for n in unmeasured if os.path.exists(os.path.join(ROOT, f"{n}.pt"))]
    gone = [n for n in unmeasured if n not in here]
    print(f"  {len(here)} checkpoint(s) with weights here and NO math-hard -- run `harness.py measure`:")
    print("    " + (", ".join(here) if here else "(none)"))
    if gone:
        print(
            f"\n  {len(gone)} unmeasured checkpoint(s) whose weights are GONE. Not a to-do: they"
            "\n  were deleted, and EXPERIMENTS.md is now the whole of what is known about them."
        )
        print("    " + ", ".join(gone))
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
    mismatch by design; an unrecorded stop makes a gap permanent."""
    import subprocess

    scores, _ = recorded_scores()
    todo = [
        n
        for n in checkpoint_names(scores)
        if n not in scores and os.path.exists(os.path.join(ROOT, f"{n}.pt"))
    ]
    if only:
        todo = [n for n in todo if only in n]
    # Newest first. NOT capped: gaps must stop listing the same names forever, and
    # math-hard alone is ~5 min per checkpoint.
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
    if os.path.exists(saved) and os.path.getsize(saved):
        # Only when the log carries math-hard-shaped results: a fresh log (the 0830v1 reset
        # wiped it) legitimately has none, and an empty parse of an empty-of-scores log is
        # not a parser regression. A ceval percentage in a row does not count -- recorded_scores
        # is the math-hard ledger.
        rows = [json.loads(l) for l in open(saved, encoding="utf-8") if l.strip()]
        if any(SCORE_RE.search(str(r.get("result", ""))) for r in rows):
            s, _o = recorded_scores()
            assert s, "math-hard-shaped results exist but none attributed: score_from stopped parsing"

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        json.dump({"domains": {}}, open(p, "w"))
        _d, e = read_mix(p)
        assert e, "an empty domains map read as a valid mix"

    # mix_shards_present's strict branch runs only on a GPU box; force it so the broken
    # world exercises the branch that runs on the pod.
    os.environ["HARNESS_GPU_PRESENT"] = "1"
    # A broken world must hold at least one file at a path the real repo also contains. A
    # world hand-written from the check's own assumptions cannot -- scan_math_contamination's
    # self-check wrote its own rows with its own field names and never saw a real corpus
    # shape. no_oversized_blob is the exception: its artifact is a >MAX file that cannot
    # exist in the repo by design; its reality comes from real git plumbing, not a repo file.
    # Known ceiling: this catches worlds built on made-up paths, not worlds that mutate one
    # real file and hand-write the rest -- the latter is a code-review property, not a tree one.
    synthetic_world = {"no_oversized_blob"}
    untested = []
    for name, _a, _i, fn, broken in CHECKS:
        root = broken()
        try:
            if name not in synthetic_world and not any(
                os.path.exists(os.path.join(ROOT, os.path.relpath(os.path.join(dp, f), root)))
                for dp, _dn, fns in os.walk(root)
                for f in fns
            ):
                untested.append(f"{name}: broken world holds no file at a repo-real path -- hand-written?")
                continue
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
    # argparse with choices, not a hand-rolled scan: a bare-flag filter once resolved
    # cmd="7", matched no branch, printed nothing and exited 0 -- a silent no-op, the
    # failure mode this file exists to prevent.
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
        warns = [n for n, s, *_ in res if s == WARN]
    else:
        bad, warns = [], []
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
    if warns:
        print(f"\n{len(warns)} non-blocking warning(s) (to-dos, not failures): {', '.join(warns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
