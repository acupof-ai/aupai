#!/usr/bin/env python3
"""The single place this project's progress is checked, recorded, and advanced.

Two rules:
- A stage is done when the measurement that would falsify it exists and is recorded, not when it produced a file.
- A check without a failing case is not a check: every CHECKS entry carries broken(), and --selftest asserts FAIL on it.

python scripts/harness.py            # check + status
python scripts/harness.py check      # invariants only; exit 1 on any failure
python scripts/harness.py run <step> # the only verb that executes; refuses while check is red
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
import subprocess
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

# --------------------------------------------------------------------------- workspace root
# One root, configured once. AUPAI_ROOT resolves to an absolute path; every data
# location the pipeline steps write derives from it. Default: the repo root.
# The harness refuses a pipeline step whose data path escapes the root, and
# check_root_durable verifies the root is not on a Kubernetes emptyDir.
#
# /work is a Kubernetes emptyDir on the host's root disk -- a pod deletion erases
# it. The durable NVMe drives (/data00-/data03, ~11 TB) are on the host but NOT
# mounted inside the container, so the check detects known-ephemeral mounts
# rather than comparing against durable ones. When the migration mounts /data00
# inside the container, add it to a durable list and invert the check.
EPHEMERAL_MOUNTS = ("/work",)


def aupai_root():
    """Resolve AUPAI_ROOT to an absolute path. Default: the repo root."""
    env = os.environ.get("AUPAI_ROOT")
    return os.path.abspath(env) if env else ROOT


def check_root_durable(root):
    """AUPAI_ROOT must not be on a Kubernetes emptyDir. A pod deletion destroys
    everything on /work; the durable NVMe drives are not visible inside the
    container, so this detects known-ephemeral mounts rather than comparing
    against durable ones. Reports FAIL on the pod today (root is /work/aupai)."""
    env = os.environ.get("AUPAI_ROOT")
    aupai = os.path.abspath(env) if env else root
    # Selftest override: a .ephemeral_mounts file in the root names the ephemeral list.
    ef = os.path.join(root, ".ephemeral_mounts")
    if os.path.exists(ef):
        ephemeral = [l.strip() for l in open(ef) if l.strip()]
    else:
        ephemeral = list(EPHEMERAL_MOUNTS)
    for m in ephemeral:
        if aupai == m or aupai.startswith(m + os.sep):
            return FAIL, f"root {aupai} is on {m}, a Kubernetes emptyDir -- a pod deletion erases it"
    return PASS, f"root {aupai} is not on a known-ephemeral mount"


def _broken_root_durable():
    """A root that IS under a (fake) ephemeral mount."""
    d = _tmp_repo()
    # A repo-real file so selftest does not skip this check as hand-written.
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w").close()
    # The root (d) is under its parent -- name the parent as the ephemeral mount.
    with open(os.path.join(d, ".ephemeral_mounts"), "w") as f:
        f.write(os.path.dirname(d) + "\n")
    return d


# Pipeline step -> the data paths it writes, relative to the repo root.
# The harness refuses to run a step whose path escapes AUPAI_ROOT.
_PIPELINE_DATA_PATHS = {
    "fetch": ("data/raw",),
    "clean": ("data/corpus",),
    "score": ("data/scores",),
    "dedup": ("data/corpus",),
}


def _check_data_under_root(step):
    """Refuse if the step's data paths escape AUPAI_ROOT. Symlinks are allowed --
    they are the migration mechanism (data/raw -> /data00/aupai_raw)."""
    aupai = aupai_root()
    for name in _PIPELINE_DATA_PATHS.get(step, ()):
        target = os.path.normpath(os.path.join(ROOT, name))
        if target != aupai and not target.startswith(aupai + os.sep):
            raise ValueError(
                f"{step} writes to {target}, outside AUPAI_ROOT {aupai}. "
                f"Set AUPAI_ROOT or move the data under it."
            )


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


def check_gemm_dims(root):
    """vocab 32773 cost 2.23x on the LM head because nothing checked shapes: it left the logits'
    leading dimension 2-byte aligned and cuBLAS fell back to an SM75 align-1 kernel on a Hopper
    card. Parsed from source rather than imported -- this must not need torch or a GPU.
    Full audit incl. every nn.Linear: scripts/shape_audit.py."""
    src_path = os.path.join(root, "train.py")
    if not os.path.exists(src_path):
        return SKIP, "train.py not present"
    tree = ast.parse(open(src_path, encoding="utf-8").read())
    cfg = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Cfg"), None)
    if cfg is None:
        return FAIL, "train.py has no Cfg"
    dims = {}
    for node in cfg.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("vocab", "d", "ffn_hidden"):
                    if isinstance(node.value.value, int):
                        dims[t.id] = node.value.value
    if not dims:
        return FAIL, "Cfg names none of vocab/d/ffn_hidden as an int literal"
    bad = [f"{k}={v} (%8={v % 8})" for k, v in dims.items() if v % 8]
    if bad:
        return FAIL, f"GEMM dims not 8-aligned: {', '.join(bad)} -- cuBLAS drops to an align-1 kernel"
    bad16 = [f"{k}={v} (%16={v % 16})" for k, v in dims.items() if v % 16]
    if bad16:
        return FAIL, f"GEMM dims not 16-aligned: {', '.join(bad16)} -- _fp8_ok rejects them, the run silently stays bf16"
    return PASS, f"{', '.join(f'{k}={v}' for k, v in sorted(dims.items()))} all 8/16-aligned"


def check_restartability(root):
    """A two-hour job that writes once at the end loses everything when interrupted
    (datagen/train_quality_head.py, killed at 50%, lost 100%). Ratcheted against
    scripts/restartability_baseline.json: only a NEW offender fails. Full report:
    scripts/restartability_audit.py."""
    audit = os.path.join(root, "scripts", "restartability_audit.py")
    base = os.path.join(root, "scripts", "restartability_baseline.json")
    if not os.path.exists(audit):
        return FAIL, "scripts/restartability_audit.py missing -- the check cannot run"
    if not os.path.exists(base):
        return FAIL, "scripts/restartability_baseline.json missing -- every script would read as new"
    out = subprocess.run([sys.executable, audit], cwd=root, capture_output=True, text=True)
    if out.returncode == 0:
        return PASS, out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "no new offenders"
    new = [ln for ln in out.stdout.splitlines() if ln.startswith("[NEW]")]
    return FAIL, "; ".join(new)[:300] or "restartability_audit failed"


def check_corpus_filters_fp(root):
    """A domain must record WHICH filters built it, not only what it contains.

    The gap: PROVENANCE records the Build command, and the same command run before and after a
    filters/ edit produces different corpora. corpus_fp_matches sees that the content changed;
    it cannot say why, and cannot answer 'did this batch go through pass3'. build_corpus.py now
    stamps filters_fp beside the content fingerprint.

    Domains built before the stamp existed carry no filters_fp. That is reported, not failed --
    it is unrecoverable, nothing wrote it down. A MISMATCH is a failure: it means the shards
    predate the filters currently in the tree."""
    sys.path.insert(0, os.path.join(root, "scripts"))
    import corpus_fingerprint as cf

    live = cf.fp_filters(root)
    if live is None:
        return SKIP, "no filters/ directory"
    corpus = os.path.join(root, "data", "corpus")
    if not os.path.isdir(corpus):
        return SKIP, "no data/corpus"
    stale, unrecorded, ok = [], [], 0
    for dom in sorted(os.listdir(corpus)):
        stats = os.path.join(corpus, dom, "build_corpus_stats.json")
        if not os.path.isfile(stats):
            continue
        with open(stats) as f:
            got = json.load(f).get("filters_fp")
        if got is None:
            unrecorded.append(dom)
        elif got != live:
            stale.append(f"{dom} built with filters {got}, tree is {live}")
        else:
            ok += 1
    if stale:
        return FAIL, "; ".join(stale)
    if unrecorded:
        # This is an honest unknown, not a green light: nothing recorded which filters built
        # those shards and nothing can recover it. Rebuild stamps them.
        note = (f"; UNKNOWN, not verified: {len(unrecorded)} domain(s) predate the stamp "
                f"({', '.join(unrecorded)}) -- rebuild to stamp them")
    else:
        note = ""
    return PASS, f"{ok} domain(s) match filters {live}{note}"


def _broken_corpus_filters_fp():
    d = _tmp_repo()
    os.makedirs(os.path.join(d, "filters"), exist_ok=True)
    with open(os.path.join(d, "filters", "pass1_garbage.py"), "w") as fh:
        fh.write("# a filter\n")
    dom = os.path.join(d, "data", "corpus", "web_hq")
    os.makedirs(dom, exist_ok=True)
    with open(os.path.join(dom, "build_corpus_stats.json"), "w") as fh:
        json.dump({"fingerprint": "deadbeef", "filters_fp": "0000000000000000"}, fh)
    return d


def _broken_restartability():
    """The real regression: a new script that accumulates in a loop and saves once at the end."""
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    for f in ("restartability_audit.py", "restartability_baseline.json"):
        shutil.copy(os.path.join(ROOT, "scripts", f), os.path.join(d, "scripts", f))
    os.makedirs(os.path.join(d, "datagen"), exist_ok=True)
    with open(os.path.join(d, "datagen", "new_long_job.py"), "w") as fh:
        fh.write("import numpy as np\nxs = []\nfor i in range(10):\n    xs.append(i)\n"
                 "np.save('out.npy', xs)\n")
    return d


def _broken_gemm_dims():
    # The REAL train.py with ffn_hidden 3072 -> 3400: 8-aligned (passes the cuBLAS
    # tier) but not 16-aligned, so _fp8_ok silently drops FP8. Mutated, not hand-written.
    import shutil

    d = _tmp_repo()
    p = os.path.join(d, "train.py")
    shutil.copy(os.path.join(ROOT, "train.py"), p)
    src = open(p, encoding="utf-8").read()
    src = src.replace("ffn_hidden = 3072", "ffn_hidden = 3400", 1)
    assert "ffn_hidden = 3400" in src, "real train.py no longer has 'ffn_hidden = 3072'; update _broken_gemm_dims"
    open(p, "w", encoding="utf-8").write(src)
    return d


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


DATA_PATH_RE = re.compile(r"data/[A-Za-z0-9_][A-Za-z0-9_./-]*")


def _cited_path_exists(root, tok):
    """A doc-cited data path that resolves. Gitignored artifacts (tokenizer.json, corpus
    bytes) are exempt -- absent from a clean checkout is their normal state; only a
    TRACKED path that is missing is rot. With no git (the pod), disk is the only truth."""
    if os.path.exists(os.path.join(root, tok)):
        return True
    is_repo = (
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True).returncode
        == 0
    )
    if not is_repo:
        return False
    r = subprocess.run(["git", "ls-files", "--error-unmatch", tok], cwd=root, capture_output=True, text=True)
    return r.returncode != 0  # tracked-but-missing -> False; untracked (gitignored) -> True


def _doc_data_paths(root):
    """Every data/ path cited in a doc, as (file, line, path). Templates (<domain>), globs
    (*), and brace expansions ({0.2b,...}) are not one path. A doc that recommends a path
    that does not exist is how 3b ran a wrong fingerprint off README's mix_v3.json."""
    out = []
    for pat in ("*.md", "docs/**/*.md", "data/*.md"):
        for md in glob.glob(os.path.join(root, pat), recursive=True):
            for i, line in enumerate(open(md, encoding="utf-8"), 1):
                for m in DATA_PATH_RE.finditer(line):
                    tok = m.group(0).rstrip("._-/")
                    if not tok or tok == "data":
                        continue
                    nxt = line[m.end()] if m.end() < len(line) else ""
                    if nxt in "{[*?<>":  # brace/glob expansion or template, not one path
                        continue
                    out.append((os.path.relpath(md, root), i, tok))
    return out


def check_doc_commands(root):
    """Every .sh/.py cited in an AGENTS.md command block exists, and every data/ path
    cited in any doc exists. A documented path that does not resolve is worse than none:
    README once recommended data/mix_v3.json, which has never existed, and a session ran
    a wrong fingerprint because of it. Only fenced blocks are scanned for scripts; prose
    citations of data files are scanned across all docs."""
    agents = os.path.join(root, "AGENTS.md")
    missing = set()
    if os.path.exists(agents):
        for block in CMD_BLOCK_RE.findall(open(agents, encoding="utf-8").read()):
            for tok in CMD_PATH_RE.findall(block):
                if not os.path.exists(os.path.join(root, tok)):
                    missing.add(tok)
    for f, _ln, tok in _doc_data_paths(root):
        if not _cited_path_exists(root, tok):
            missing.add(f"{f}:{_ln} {tok}")
    if missing:
        return FAIL, f"doc(s) cite path(s) not in the repo: {sorted(missing)[:5]}"
    if not os.path.exists(agents) and not _doc_data_paths(root):
        return SKIP, "no docs present"
    return PASS, "every doc-cited script and data path exists"


def _broken_doc_commands():
    """The REAL README with one data path swapped to a name that does not exist -- the
    FAIL tier for the data-path half (the script half appends a fake command block).
    The other data paths README cites are created, so the world fails ONLY on the swap."""
    import shutil

    d = _tmp_repo()
    shutil.copy(os.path.join(ROOT, "README.md"), os.path.join(d, "README.md"))
    p = os.path.join(d, "README.md")
    s = open(p, encoding="utf-8").read()
    assert "data/mix_scale_3.24b.json" in s, "real README no longer cites the default mix; update _broken_doc_commands"
    open(p, "w", encoding="utf-8").write(s.replace("data/mix_scale_3.24b.json", "data/mix_scale_nonexistent.json"))
    os.makedirs(os.path.join(d, "data", "corpus", "sample"), exist_ok=True)
    open(os.path.join(d, "data", "mix_sample.json"), "w").write("{}")
    open(os.path.join(d, "data", "tokenizer.json"), "w").write("{}")
    with open(os.path.join(d, "README.md"), "a", encoding="utf-8") as f:
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


def _read_ckpt_dict(path):
    """Read the full dict from a torch.save checkpoint without loading tensors.
    The .pt is a zip; the dict is plain data pickled in data.pkl. Tensor storages
    are referenced via persistent_load, which we stub, and torch rebuild
    functions resolve to dummies."""
    import pickle
    import zipfile

    class _Stub(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch"):
                return lambda *a, **kw: None
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                return lambda *a, **kw: None

        def persistent_load(self, pid):
            return None

    with zipfile.ZipFile(path) as z:
        pkl_name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        with z.open(pkl_name) as f:
            return _Stub(f).load()


def _read_ckpt_cfg(path):
    """Read the cfg dict from a torch.save checkpoint without importing torch."""
    return _read_ckpt_dict(path).get("cfg", {})


def check_env_fp_present(root):
    """Every checkpoint carries an environment fingerprint.

    A container restart can change the effective environment (dropping
    hand-installed packages) without anyone noticing. The fingerprint is
    compared on resume; a checkpoint without one predates the guard and
    cannot be safely resumed."""
    ckpts = sorted(glob.glob(os.path.join(root, "ckpt_*.pt")))
    if not ckpts:
        return SKIP, "no checkpoints"
    missing = []
    for p in ckpts:
        try:
            d = _read_ckpt_dict(p)
        except Exception:
            continue  # unreadable checkpoint is a different check's problem
        if "env_fp" not in d:
            missing.append(os.path.basename(p))
    if missing:
        return FAIL, f"{len(missing)} checkpoint(s) without env_fp: {', '.join(missing[:5])}"
    return PASS, f"all {len(ckpts)} checkpoints carry env_fp"


def _broken_env_fp_present():
    """A real torch checkpoint without env_fp."""
    import torch

    d = _tmp_repo()
    # A repo-real file so selftest does not skip this check as hand-written.
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w").close()
    # A REAL torch checkpoint (torch.save), just without env_fp -- the check
    # reads it the same way it reads a production one.
    torch.save(
        {"model": {}, "cfg": {}, "vocab_id": "test", "corpus_fp": {}},
        os.path.join(d, "ckpt_test.pt"),
    )
    return d


def check_opt_state_present(root):
    """A checkpoint that records a training step must carry optimizer state.

    A checkpoint with `step` but no `opt` cannot be safely resumed: Muon momentum
    and AdamW moments are zeroed, the loss dips and recovers, and it reads as
    noise rather than a bug. The ladder's short runs from scratch never resumed,
    so the gap stayed hidden. The 30B run will."""
    ckpts = sorted(glob.glob(os.path.join(root, "ckpt_*.pt")))
    if not ckpts:
        return SKIP, "no checkpoints"
    missing, resumable = [], 0
    for p in ckpts:
        try:
            d = _read_ckpt_dict(p)
        except Exception:
            continue  # unreadable checkpoint is a different check's problem
        if "step" not in d:
            continue  # final/eval checkpoint, not claiming to be resumable
        resumable += 1
        if "opt" not in d:
            missing.append(os.path.basename(p))
    if missing:
        return FAIL, f"{len(missing)} checkpoint(s) with step but no opt: {', '.join(missing[:5])}"
    return PASS, f"all {resumable} resumable checkpoints carry opt state"


def _broken_opt_state_present():
    """A real torch checkpoint with step but no opt state."""
    import torch

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    open(os.path.join(d, "scripts", "harness.py"), "w").close()
    torch.save(
        {"model": {}, "cfg": {}, "vocab_id": "test", "corpus_fp": {}, "env_fp": "x", "step": 100},
        os.path.join(d, "ckpt_test.pt"),
    )
    return d


def check_ladder_config(root):
    """Every ladder checkpoint's cfg matches the single frozen run config.
    Scope: checkpoints whose experiments row was launched via run_ddp.sh
    (harness run point or a bare launch that landed in the ledger). A/B runs
    use torchrun directly and are not bound by the frozen config.
    A missing field (None) is UNKNOWN, not divergence: the checkpoint predates
    the stamp. SKIP without checkpoints or the frozen config file."""
    fpath = os.path.join(root, "data", "mix_scale_run_config.json")
    if not os.path.exists(fpath):
        return SKIP, "data/mix_scale_run_config.json not present"
    frozen = json.load(open(fpath, encoding="utf-8"))
    ckpts = sorted(glob.glob(os.path.join(root, "ckpt_*.pt")))
    if not ckpts:
        return SKIP, "no checkpoints"
    # Ladder points are launched via run_ddp.sh; A/B runs use torchrun directly.
    ladder_names = set()
    exp_path = os.path.join(root, "runs", "experiments.jsonl")
    if os.path.exists(exp_path):
        for line in open(exp_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if "run_ddp.sh" in r.get("cmd", ""):
                    ladder_names.add(r.get("name", ""))
    bad, unknown, checked = [], [], 0
    for p in ckpts:
        name = os.path.basename(p)[5:-3]  # ckpt_<name>.pt
        if name not in ladder_names:
            continue
        try:
            cfg = _read_ckpt_cfg(p)
        except Exception as e:
            return FAIL, f"{os.path.basename(p)}: cannot read cfg: {e}"
        if not os.path.basename(cfg.get("mix", "")).startswith("mix_scale_"):
            continue
        checked += 1
        for k in (*_FROZEN_KEYS, *_CODE_FROZEN_KEYS):
            v = cfg.get(k)
            if v is None:
                unknown.append(f"{os.path.basename(p)}:{k}")
            elif v != frozen[k]:
                bad.append(f"{os.path.basename(p)}: {k}={v} != frozen {frozen[k]}")
    if bad:
        return FAIL, "; ".join(bad)
    if not checked:
        return SKIP, "no ladder checkpoints (experiments.jsonl is pod-authoritative; local copy may be stale)"
    msg = f"{checked} checkpoint(s) match the frozen config"
    if unknown:
        msg += f"; {len(unknown)} field(s) unverifiable (pre-stamp): {', '.join(sorted(set(unknown)))}"
    return PASS, msg


def _broken_ladder_config():
    """A checkpoint with warmup changed from the frozen 20 to 30 -- the silent
    recipe drift that produces a completed point under a different config."""
    import io
    import pickle
    import shutil
    import zipfile

    d = _tmp_repo()
    shutil.copy(
        os.path.join(ROOT, "data", "mix_scale_run_config.json"),
        os.path.join(d, "data", "mix_scale_run_config.json"),
    )
    cfg = {"mix": "data/mix_scale_0.2b.json", "warmup": 30, "batch": 16, "accum": 2,
           "vocab": 32784, "bucket_cap_mb": 50}
    buf = io.BytesIO()
    pickle.dump({"cfg": cfg, "vocab_id": "fake"}, buf)
    with zipfile.ZipFile(os.path.join(d, "ckpt_test.pt"), "w") as z:
        z.writestr("data.pkl", buf.getvalue())
    # An exp row launched via run_ddp.sh -- the scope filter.
    with open(os.path.join(d, "runs", "experiments.jsonl"), "w") as f:
        f.write(json.dumps({"name": "test", "started": time.strftime("%Y-%m-%d %H:%M"),
                            "status": "ok", "cmd": "bash run_ddp.sh --name test"}) + "\n")
    return d


def check_ladder_cfg_consistent(root):
    """All six ladder points record the same cfg (except mix). The frozen list
    prevents launch drift; this detects code-edit drift between points -- a
    chunk_size, layers, or optimizer-param edit that no CLI flag can make and
    no frozen key can see. Checkpoints record vars(Cfg): 46 fields, complete.
    The only legitimate per-point difference is mix (the D varies)."""
    names = [n for n, _ in LADDER]
    ckpts = [os.path.join(root, f"ckpt_{n}.pt") for n in names]
    ckpts = [p for p in ckpts if os.path.exists(p)]
    if len(ckpts) < 2:
        return SKIP, f"{len(ckpts)}/{len(names)} ladder checkpoints present; need 2+ to compare"
    cfgs = {}
    for p in ckpts:
        try:
            cfgs[p] = _read_ckpt_cfg(p)
        except Exception as e:
            return FAIL, f"{os.path.basename(p)}: cannot read cfg: {e}"
    base_p, base = ckpts[0], cfgs[ckpts[0]]
    diffs, unknown = [], []
    for p in ckpts[1:]:
        cfg = cfgs[p]
        for k in sorted(set(base) | set(cfg)):
            if k == "mix":
                continue  # the D varies; everything else must not
            if k not in base or k not in cfg:
                unknown.append(f"{os.path.basename(p)}:{k}")
            elif base[k] != cfg[k]:
                diffs.append(f"{os.path.basename(p)}:{k} {base[k]!r}->{cfg[k]!r}")
    if diffs:
        return FAIL, f"{len(diffs)} field(s) differ: {'; '.join(diffs[:5])}"
    note = f"; {len(unknown)} unverifiable (pre-stamp): {', '.join(sorted(set(unknown))[:5])}" if unknown else ""
    return PASS, f"{len(ckpts)} checkpoints, {len(base)} fields, all consistent{note}"


def _broken_ladder_cfg_consistent():
    """Two ladder checkpoints with chunk_size changed in one -- the code-edit
    drift the frozen list cannot see (no CLI flag touches chunk_size)."""
    import io
    import pickle
    import shutil
    import zipfile

    d = _tmp_repo()
    shutil.copy(
        os.path.join(ROOT, "data", "mix_scale_run_config.json"),
        os.path.join(d, "data", "mix_scale_run_config.json"),
    )
    for name, cs in [("p02_s0", 32), ("p03", 64)]:
        cfg = {"mix": f"data/mix_scale_{'0.2b' if name == 'p02_s0' else '0.3b'}.json",
               "chunk_size": cs, "batch": 16}
        buf = io.BytesIO()
        pickle.dump({"cfg": cfg, "vocab_id": "fake"}, buf)
        with zipfile.ZipFile(os.path.join(d, f"ckpt_{name}.pt"), "w") as z:
            z.writestr("data.pkl", buf.getvalue())
    return d


def _train_parser_flags(train_py):
    """CLI flag names from train.py's argparse section. Two shapes: direct
    add_argument("--flag", ...) and the loop over a dict whose keys are flag names."""
    src = open(train_py, encoding="utf-8").read()
    section = src[src.index("ArgumentParser"):src.index("parse_args")]
    flags = set()
    for m in re.finditer(r'add_argument\(\s*["\']--(\w+)["\']', section):
        flags.add(m.group(1))
    for m in re.finditer(r'^\s+"(\w+)":\s*"', section, re.M):
        flags.add(m.group(1))
    return flags


def check_frozen_keys_complete(root):
    """Every train.py parser flag that changes a Cfg field is either in _FROZEN_KEYS
    or in _UNFROZEN_ALLOWLIST. The frozen set rotted once: eight architecture/recipe
    flags were missing and nothing noticed. This check is the tripwire."""
    train_py = os.path.join(root, "train.py")
    if not os.path.exists(train_py):
        return SKIP, "train.py missing"
    flags = _train_parser_flags(train_py)
    known = set(_FROZEN_KEYS) | _UNFROZEN_ALLOWLIST
    missing = []
    for f in sorted(flags):
        cfg_key = _FLAG_TO_CFG.get(f, f)
        if cfg_key not in known:
            missing.append(f"--{f} (Cfg.{cfg_key})")
    if missing:
        return FAIL, f"{len(missing)} flag(s) in neither frozen set nor allow-list: {'; '.join(missing)}"
    return PASS, f"{len(flags)} parser flags, all in frozen set or allow-list"


def _broken_frozen_keys_complete():
    """The real train.py with a new architecture flag added to the parser --
    exactly how the eight missing fields escaped notice."""
    import shutil

    d = _tmp_repo()
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    # Add a new add_argument call inside the parser section, before parse_args.
    patched = src.replace(
        'args = parser.parse_args()',
        '    parser.add_argument("--new_arch_flag", action="store_true",\n'
        '                        help="a new architecture flag the frozen set does not know about")\n'
        '    args = parser.parse_args()',
    )
    with open(os.path.join(d, "train.py"), "w", encoding="utf-8") as fh:
        fh.write(patched)
    return d


def _token_cache_dir():
    """The directory holding token caches, from train.py's TOKEN_CACHE constant.
    HARNESS_TOKEN_CACHE_DIR overrides (selftest)."""
    forced = os.environ.get("HARNESS_TOKEN_CACHE_DIR")
    if forced:
        return forced
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    m = re.search(r'^TOKEN_CACHE\s*=\s*["\']([^"\']+)["\']', src, re.M)
    if not m:
        raise KeyError("train.py has no TOKEN_CACHE; the check that reads it cannot run")
    return os.path.dirname(m.group(1))


def _cache_rows(path, seq):
    """Number of rows in a token cache, from the zip storage size (no torch)."""
    import zipfile

    with zipfile.ZipFile(path) as z:
        info = next(i for i in z.infolist() if i.filename.endswith("data/0"))
    return (info.file_size // 4) // (seq + 1)


def check_mix_supply(root):
    """Per-domain demand vs epoch-capped cache supply at every budget point.
    FAILs when demand exceeds the FULL cache (data would repeat even after
    train.py's cap). The val-split reduction (demand > pool but <= cache) is
    a known, accepted condition -- the gate doc documents it as 1.53% at
    3.24b, handled by the fit-protocol reading D from the log. The val carve
    lands entirely in the anneal phase (roughly 2x the per-domain loss the
    whole-budget figure suggests), not spread across both phases. SKIP without
    caches (CPU CI, dev box)."""
    cache_dir = _token_cache_dir()
    if not os.path.isdir(cache_dir):
        return SKIP, f"token cache dir {cache_dir} not present"
    seq = cfg_default("seq")
    anneal_frac = cfg_default("anneal_frac")
    val_frac = cfg_default("val_frac")
    val_rows_max = cfg_default("val_rows_max")
    mixes = sorted(glob.glob(os.path.join(root, "data", "mix_scale_[0-9]*.json")))
    if not mixes:
        return SKIP, "no mix_scale_*.json budget points"
    bad = []
    val_loss_tokens = 0  # val-split loss at the largest budget point, in tokens
    largest = max(
        json.load(open(m, encoding="utf-8"))["total_tokens"] for m in mixes
    )
    for mp in mixes:
        mix = json.load(open(mp, encoding="utf-8"))
        rows = mix["total_tokens"] / seq
        is_largest = mix["total_tokens"] == largest
        for name, d in mix["domains"].items():
            cache = os.path.join(cache_dir, f"tokens_{name}.pt")
            if not os.path.exists(cache):
                bad.append(f"{os.path.basename(mp)}: {name} has no cache")
                continue
            try:
                cache_rows = _cache_rows(cache, seq)
            except Exception as e:
                bad.append(f"{os.path.basename(mp)}: {name} cache unreadable: {e}")
                continue
            used = 0
            for frac, key in ((1 - anneal_frac, "weight"), (anneal_frac, "anneal")):
                want = int(rows * frac * d.get(key, d["weight"]))
                cap = int(cache_rows * d.get("epochs", 1)) - used
                # 0.5% tolerance: weight->row rounding leaves sub-0.1% residue
                # at 3.24b (documented in the gate doc). Real oversupply FAILs.
                if want > cap * 1.005:
                    bad.append(f"{os.path.basename(mp)}: {name} {key} wants {want}, cache supplies {cap}")
                    break
                used += want
            else:
                # Both phases within cache: compute val-split loss for the report.
                if is_largest:
                    n_val = min(max(1, int(cache_rows * val_frac)), val_rows_max)
                    pool = cache_rows - n_val
                    demand = used
                    val_loss_tokens += max(0, demand - pool) * seq
    if bad:
        return FAIL, "; ".join(bad)
    pct = 100 * val_loss_tokens / largest if largest else 0
    return PASS, f"{len(mixes)} mixes, all within cache supply; val-split loss {pct:.2f}% at {largest / 1e9:.2f}B"


def _broken_mix_supply():
    """The real 0.2b mix with caches too small to supply it -- demand exceeds
    pool at every domain."""
    import shutil
    import zipfile

    d = _tmp_repo()
    shutil.copy(
        os.path.join(ROOT, "data", "mix_scale_0.2b.json"),
        os.path.join(d, "data", "mix_scale_0.2b.json"),
    )
    cache_dir = os.path.join(d, "fake_caches")
    os.makedirs(cache_dir)
    seq = cfg_default("seq")
    for dom in ("web_hq", "textbook", "wiki", "en", "math", "code", "chat"):
        with zipfile.ZipFile(os.path.join(cache_dir, f"tokens_{dom}.pt"), "w") as z:
            z.writestr("data/0", b"\x00" * (4 * (seq + 1) * 10))
    os.environ["HARNESS_TOKEN_CACHE_DIR"] = cache_dir
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


# Every third-party module this repo imports, and the pip name that supplies it. The
# container's image already carries most of them; a restart keeps /work but drops the
# writable layer, so only the hand-installed ones vanish -- and which ones those are is
# not knowable without a written list. That is the whole reason this exists.
_REQUIRED = {
    "torch": "torch", "numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib",
    "pyarrow": "pyarrow", "tokenizers": "tokenizers", "transformers": "transformers",
    "datasets": "datasets", "huggingface_hub": "huggingface_hub", "flask": "flask",
    "opencc": "opencc", "trackio": "trackio", "liger_kernel": "liger-kernel",
    "fla": "flash-linear-attention", "torchao": "torchao", "triton": "triton",
    "flash_attn": "flash-attn",
}
# Absent on a dev Mac by design; only a box that can train is expected to have them.
_LINUX_ONLY = {"liger_kernel", "fla", "torchao", "triton", "flash_attn"}


def check_env_importable(root):
    """Every third-party module the repo imports is importable.

    2026-08-30: a container restart dropped the writable layer and with it liger_kernel,
    fla, flask, opencc and trackio. The code was untouched, so the first symptom was a
    ModuleNotFoundError on a line that had worked an hour earlier -- which reads as a
    broken import, not as missing infrastructure, and sends the next person to debug the
    wrong thing. This check names the cause and prints the command that fixes it.
    """
    import importlib.util

    extra = os.environ.get("HARNESS_REQUIRE_EXTRA")  # selftest injects an unsatisfiable name
    req = dict(_REQUIRED, **({extra: extra} if extra else {}))
    missing = []
    for mod in sorted(req):
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    if not missing:
        return PASS, f"all {len(req)} imported packages present"
    # Strictness follows the ability to train, as in check_mix_shards: a dev box ships
    # none of the CUDA half and a permanent red there is no signal. The incident this
    # guards is a pod restart, and the pod has cards.
    if not _gpu_present():
        return SKIP, f"{len(missing)} absent on a box that cannot train: {', '.join(missing)}"
    # pip refuses to uninstall Debian's blinker (no RECORD file), which is what flask
    # pulls on; --ignore-installed is the difference between a working command and a
    # half-applied one, so it goes in only when flask is actually among the missing.
    pre = "--ignore-installed blinker " if "flask" in missing else ""
    pkgs = " ".join(req[m] for m in missing)
    return FAIL, (
        f"{len(missing)} package(s) missing: {', '.join(missing)} -- if this box worked "
        f"before, the container restarted and lost its writable layer (/work survives, "
        f"installed packages do not). Restore: python3 -m pip install {pre}{pkgs}"
    )


def _broken_env():
    """A world whose requirement list names a package that cannot exist."""
    os.environ["HARNESS_REQUIRE_EXTRA"] = "aupai_no_such_module"
    return _tmp_repo()


CHECKS = [
    (
        "env_importable",
        "every third-party module the repo imports is installed",
        "a container restart dropped the writable layer; SFT died on ModuleNotFoundError and read as a code bug",
        check_env_importable,
        _broken_env,
    ),
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
        "corpus_filters_fp",
        "every stamped corpus domain records the filters that built it, and they still match",
        "PROVENANCE recorded the build command but not the filter version; the same command before and after a filters/ edit yields different corpora",
        check_corpus_filters_fp,
        _broken_corpus_filters_fp,
    ),
    (
        "restartability",
        "no NEW script accumulates in a loop and writes only at the end",
        "a two-hour scoring job wrote once at the end; killed at 50% it lost 100% of the work",
        check_restartability,
        _broken_restartability,
    ),
    (
        "gemm_dims_aligned",
        "Cfg's GEMM dimensions are multiples of 16 (8 for cuBLAS fast kernels, 16 for _fp8_ok)",
        "vocab 32773 made cuBLAS pick an SM75 align-1 kernel on Hopper; the LM head ran at 41% of bf16 peak, unnoticed",
        check_gemm_dims,
        _broken_gemm_dims,
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
    (
        "ladder_config_frozen",
        "every ladder checkpoint's cfg matches data/mix_scale_run_config.json",
        "a silent recipe drift (wrong warmup, wrong bucket) produces a completed point that poisons the curve; the OOM was loud, the wrong-but-valid case is not",
        check_ladder_config,
        _broken_ladder_config,
    ),
    (
        "frozen_keys_complete",
        "every train.py parser flag is in _FROZEN_KEYS or _UNFROZEN_ALLOWLIST",
        "eight architecture/recipe flags escaped the frozen set and nothing noticed; the list rots the moment someone adds a flag",
        check_frozen_keys_complete,
        _broken_frozen_keys_complete,
    ),
    (
        "ladder_cfg_consistent",
        "all six ladder checkpoints record the same cfg (except mix)",
        "a code edit to chunk_size/layers/optimizer params between points is invisible to the frozen list (no CLI flag) and to pod_drift (manifest regenerated); this is the only check that sees it",
        check_ladder_cfg_consistent,
        _broken_ladder_cfg_consistent,
    ),
    (
        "mix_supply",
        "per-domain demand does not exceed epoch-capped pool at any budget point",
        "a mix that wants more rows than its pool allows trains on repeated data with nothing raising",
        check_mix_supply,
        _broken_mix_supply,
    ),
    (
        "root_durable",
        "AUPAI_ROOT is on a durable mount (/data00-/data03), not a Kubernetes emptyDir",
        "the 94 GB corpus, every checkpoint, and the repo lived in a 365 GB emptyDir for weeks; a pod deletion would erase all of it",
        check_root_durable,
        _broken_root_durable,
    ),
    (
        "env_fp_present",
        "every checkpoint carries an environment fingerprint",
        "a container restart changed the effective environment and three sessions chased wrong hypotheses for an hour because nothing recorded what the environment WAS",
        check_env_fp_present,
        _broken_env_fp_present,
    ),
    (
        "opt_state_present",
        "a checkpoint with a step number carries optimizer state",
        "resuming from a checkpoint with step but no opt zeroes Muon momentum and AdamW moments; the loss dips and recovers, looking like noise",
        check_opt_state_present,
        _broken_opt_state_present,
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
    ("corpus", ["corpus_filters_fp", "mix_not_unfiltered", "mix_shards_present"], "contamination scan recorded for every source"),
    ("pretrain", ["restartability", "gemm_dims_aligned", "guard_on_path", "no_stale_running", "score_matrix_present"], "checkpoint carries vocab_id; val loss recorded"),
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


# --------------------------------------------------------------------------- board


def _val_nll(name):
    """Last val NLL from runs/<name>.log, or None. The log line is
    'ep 1/1 train 3.281 val 3.322 615s'."""
    log = os.path.join(ROOT, "runs", f"{name}.log")
    if not os.path.exists(log):
        return None
    val = None
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"val (\d+\.\d+)", line)
        if m:
            val = float(m.group(1))
    return val


def _board_event(kind, msg):
    """Append an event to runs/events.jsonl. The harness knows when things
    happen; this is how it stops staying silent."""
    path = os.path.join(ROOT, "runs", "events.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M"), "kind": kind, "msg": msg}) + "\n")


def _board_data():
    """All board state, gathered from the same artifacts the checks read."""
    res = run_checks(ROOT, quiet=True)
    checks = [{"name": n, "state": s, "evidence": e} for n, s, e, _a, _i in res]
    n_skip = sum(1 for c in checks if c["state"] == SKIP)
    n_fail = sum(1 for c in checks if c["state"] == FAIL)
    # score matrix: ckpt -> metrics
    sm = {}
    sm_path = os.path.join(ROOT, "runs", "score_matrix.jsonl")
    if os.path.exists(sm_path):
        for line in open(sm_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                sm[r["ckpt"]] = r.get("metrics", {})
    # ladder points
    ladder = []
    for name, mix in LADDER:
        ckpt = f"ckpt_{name}.pt"
        m = sm.get(ckpt, {})
        dl = m.get("domain_loss", {})
        ladder.append({
            "name": name,
            "mix": os.path.basename(mix),
            "val_nll": _val_nll(name),
            "domain_loss": dl.get("unweighted_mean"),
            "minimal_pairs": m.get("minimal_pairs", {}).get("overall"),
            "lambada_zh": m.get("lambada_zh", {}).get("two_way_acc"),
            "math_v2_like": m.get("math_v2_like", {}).get("acc") or m.get("math_v2_like", {}).get("pass1"),
            "ceval": m.get("mc_ceval", {}).get("Average"),
            "scored": ckpt in sm,
        })
    # recent experiments (last 8)
    exps = []
    exp_path = os.path.join(ROOT, "runs", "experiments.jsonl")
    if os.path.exists(exp_path):
        rows = [json.loads(l) for l in open(exp_path, encoding="utf-8") if l.strip()]
        for r in rows[-8:]:
            exps.append({"name": r.get("name"), "status": r.get("status"),
                         "started": r.get("started"), "cmd": r.get("cmd", "")[:80]})
    # events (last 10)
    events = []
    ev_path = os.path.join(ROOT, "runs", "events.jsonl")
    if os.path.exists(ev_path):
        rows = [json.loads(l) for l in open(ev_path, encoding="utf-8") if l.strip()]
        events = rows[-10:]
    # staleness: newest artifact mtime
    newest = 0.0
    for p in [sm_path, exp_path, os.path.join(ROOT, "train.py")]:
        if os.path.exists(p):
            newest = max(newest, os.path.getmtime(p))
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "staleness": {"newest_artifact": time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)) if newest else None,
                      "skip_count": n_skip, "fail_count": n_fail},
        "checks": checks,
        "ladder": ladder,
        "experiments": exps,
        "events": events,
    }


def _render_board_html(d):
    """Self-contained HTML: inline CSS, system fonts, no external deps."""
    def pct(v):
        return f"{v*100:.1f}%" if isinstance(v, (int, float)) and v <= 1 else (f"{v}" if v is not None else "—")

    def num(v, fmt="{:.3f}"):
        return fmt.format(v) if isinstance(v, (int, float)) else "—"

    colors = {PASS: "#2d7d32", FAIL: "#c62828", SKIP: "#888", WARN: "#f9a825"}
    rows = ""
    for c in d["checks"]:
        color = colors.get(c["state"], "#888")
        rows += f'<tr><td>{c["name"]}</td><td style="color:{color};font-weight:600">{c["state"]}</td><td>{c["evidence"][:100]}</td></tr>\n'

    lrows = ""
    for p in d["ladder"]:
        status = "✓ scored" if p["scored"] else "…"
        lrows += (f'<tr><td>{p["name"]}</td><td>{p["mix"]}</td><td>{num(p["val_nll"])}</td>'
                  f'<td>{num(p["domain_loss"])}</td><td>{pct(p["minimal_pairs"])}</td>'
                  f'<td>{pct(p["lambada_zh"])}</td><td>{pct(p["math_v2_like"])}</td>'
                  f'<td>{num(p["ceval"], "{:.1f}")}</td><td>{status}</td></tr>\n')

    erows = ""
    for e in d["events"]:
        erows += f'<tr><td>{e["ts"]}</td><td>{e["kind"]}</td><td>{e["msg"]}</td></tr>\n'

    xprows = ""
    for x in d["experiments"]:
        xprows += f'<tr><td>{x["name"]}</td><td>{x["status"]}</td><td>{x["started"]}</td></tr>\n'

    st = d["staleness"]
    stale_warn = ""
    if st["fail_count"]:
        stale_warn = f'<p style="color:#c62828;font-weight:700">{st["fail_count"]} CHECK(S) RED</p>'
    if st["skip_count"]:
        stale_warn += f'<p style="color:#888">{st["skip_count"]} check(s) SKIPped — guard not running, not guard passed</p>'

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>aupai board</title><style>
body{{font:14px/1.5 system-ui,sans-serif;margin:2em;background:#fafafa;color:#222}}
h1{{font-size:1.3em}} h2{{font-size:1.1em;margin-top:1.5em;border-bottom:1px solid #ddd;padding-bottom:3px}}
table{{border-collapse:collapse;width:100%;margin:.5em 0}}
th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid #eee;font-size:13px}}
th{{color:#666;font-weight:600}} .meta{{color:#888;font-size:12px}}
</style></head><body>
<h1>aupai monitoring board</h1>
<p class="meta">rendered {d["timestamp"]} · newest artifact {st["newest_artifact"] or "—"}</p>
{stale_warn}
<h2>checks</h2><table><tr><th>check</th><th>state</th><th>evidence</th></tr>{rows}</table>
<h2>ladder points</h2><table><tr><th>point</th><th>mix</th><th>val NLL</th><th>domain loss</th>
<th>min pairs</th><th>lambada</th><th>math v2</th><th>ceval</th><th>status</th></tr>{lrows}</table>
<h2>recent experiments</h2><table><tr><th>name</th><th>status</th><th>started</th></tr>{xprows}</table>
<h2>events</h2><table><tr><th>time</th><th>kind</th><th>message</th></tr>{erows}</table>
</body></html>"""


def cmd_board(as_json=False, html_path=None):
    """harness board [--json | --html <path>]. Renders harness state as JSON or HTML.
    Default: writes runs/board.html. Every number is read at render time — nothing typed."""
    d = _board_data()
    if as_json:
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return 0
    path = html_path or os.path.join(ROOT, "runs", "board.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_render_board_html(d))
    print(f"board written to {path} ({len(d['checks'])} checks, {len(d['ladder'])} ladder points)")
    return 0


def _refresh_board():
    """Refresh runs/board.html after a state-changing step. Never blocks the step."""
    try:
        cmd_board(html_path=os.path.join(ROOT, "runs", "board.html"))
    except Exception as e:
        print(f"board refresh failed (non-blocking): {e}")


# ------------------------------------------------------------------------ selftest


def _demo():
    """Every check must FAIL on a world where its condition is violated."""
    import shutil

    assert score_from("math-hard 37/1032 = 3.6%") == 3.6, "took the numerator, not the percentage"
    assert score_from("math-hard deferred to the bench stage") is None, "invented a score"
    assert score_from("math-hard 1.7% (18/1032) vs k5 1.9%") == 1.7

    # run dispatch: a missing or unknown step is a usage error, not a silent exit 0
    assert run_dispatch([]) == 2 and run_dispatch(["bogus"]) == 2

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
    # env_importable joins it for the same reason: its artifact is process import state,
    # not a tree, so no world it builds can hold a repo file.
    synthetic_world = {"no_oversized_blob", "env_importable"}
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


STEPS = ("pretokenize", "point", "ladder", "fetch", "clean", "score", "dedup")

# The six 0830v1 budget points, in order. Each is a mix_scale_* mix at the
# frozen run config. Names double as checkpoint names: ckpt_<name>.pt.
LADDER = [
    # p02_s0, not p02: the 0.2b point is the seed-0 run (ckpt_p02_s0.pt), already
    # scored. Naming the entry p02_s0 makes the skip regex match it -- the curve's
    # 0.2b point and the sigma-hat measurement come from the same checkpoint.
    ("p02_s0", "data/mix_scale_0.2b.json"),
    ("p03", "data/mix_scale_0.3b.json"),
    ("p04", "data/mix_scale_0.4b.json"),
    ("p08", "data/mix_scale_0.8b.json"),
    ("p16", "data/mix_scale_1.6b.json"),
    ("p324", "data/mix_scale_3.24b.json"),
]


def _gate(force):
    """The red invariants, by name and evidence. A runnable step needs none: 'no GPU
    pretraining while harness is red' was a doc line nothing executed until this."""
    reds = [(n, ev) for n, s, ev, _a, _i in run_checks(ROOT, quiet=True) if s == FAIL]
    for n, ev in reds:
        print(f"  RED {n}: {ev}")
    if reds and not force:
        print("REFUSING to run while harness is red. Pass --force to override "
              "(the reds are recorded in the exp row).")
    return reds


def _exp(action, **kw):
    cmd = [sys.executable, os.path.join(HERE, "exp.py"), action]
    for k, v in kw.items():
        cmd += [f"--{k}", str(v)]
    subprocess.run(cmd, cwd=ROOT, check=False)


def _run_pretokenize(step_args, forced):
    cmd = [sys.executable, os.path.join(HERE, "pretokenize.py"), *step_args]
    _exp("start", name="pretokenize", cmd=" ".join(cmd),
         hypothesis=f"tokenize every mix domain into its cache before training{forced}")
    r = subprocess.run(cmd, cwd=ROOT)
    _exp("done", name="pretokenize", result=f"exit {r.returncode}",
         finding="caches warm" if r.returncode == 0 else "pretokenize failed",
         decision="training can launch on warm caches" if r.returncode == 0 else "fix the failure before launching",
         status="ok" if r.returncode == 0 else "fail")
    return r.returncode


def _step_name(step, step_args):
    """fetch --source web_hq -> fetch_web_hq; bare step name when no source/domain."""
    for i, a in enumerate(step_args):
        if a in ("--source", "--domain") and i + 1 < len(step_args):
            return f"{step}_{step_args[i + 1]}"
        if a.startswith(("--source=", "--domain=")):
            return f"{step}_{a.split('=', 1)[1]}"
    return step


def _run_pipeline_step(step, script, step_args, forced, env=None):
    """fetch/clean/score: gate (in dispatch) + exp start/done + run the script.
    The script owns the work, the output fingerprint, and shard-level resumability.
    Score pins CUDA_VISIBLE_DEVICES=0 -- a collision on GPU 0 is visible (benchmarks
    fail), a collision on 1-7 is silent (training corrupted)."""
    _check_data_under_root(step)
    cmd = [sys.executable, os.path.join(HERE, script), *step_args]
    name = _step_name(step, step_args)
    _exp("start", name=name, cmd=" ".join(cmd), hypothesis=f"{step} step{forced}")
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    ok = r.returncode == 0
    _exp("done", name=name, result=f"exit {r.returncode}",
         finding=f"{step} complete" if ok else f"{step} failed",
         decision="next step can run" if ok else "fix the failure before next step",
         status="ok" if ok else "fail")
    return r.returncode


_FROZEN_KEYS = (
    "batch", "accum", "warmup", "vocab", "bucket_cap_mb",  # recipe
    "attn_res_blocks", "attn_every", "attn_res", "attn_res_dyn_q",  # architecture
    "seq", "grad_ckpt", "fone", "doc_mask",  # architecture / training comparability
)

# Architecture constants with no CLI flag. They cannot drift via a launch, so
# _strip_frozen and frozen_args do not touch them. But they can drift via a code
# edit, and ladder_config_frozen compares them against the JSON as documented
# intent -- closing the gap where all six points agree with each other but not
# with what was intended (fb regenerated the manifest mid-ladder, blinding pod_drift).
_CODE_FROZEN_KEYS = ("chunk_size", "layers", "d", "heads", "ffn_hidden")

# CLI flags whose name differs from their Cfg field (--no_doc_mask sets Cfg.doc_mask).
_FLAG_TO_CFG = {"no_doc_mask": "doc_mask", "no_attn_res": "attn_res"}

# Parser flags intentionally outside the frozen set. Criterion: a flag that changes the
# architecture or the recipe is frozen; these are run-management, measurement, or
# deliberately variable. check_frozen_keys_complete forces a decision when a new flag lands.
_UNFROZEN_ALLOWLIST = {
    "seed",               # the quantity that is supposed to vary
    "name", "mix", "resume", "max_steps",  # run management
    "fp8",                # training precision, not architecture
    "track", "profile", "profile_warmup", "profile_steps",  # measurement
    "allow_corpus_drift", "allow_pod_drift", "allow_env_drift",  # safety overrides
    "lr_scale",           # optimizer multiplier, varies by experiment
    "no_static_graph", "no_bucket_view",  # DDP A/B, do not touch Cfg
    "val_every", "val_batches",  # validation cadence, not architecture
}


def _strip_frozen(passthrough, frozen):
    """Drop agreeing frozen flags from passthrough; refuse disagreeing ones.
    Returns (clean_passthrough, conflicts). An agreeing flag is accepted, not
    refused -- fb launched four runs with explicit --batch 16 --accum 2 that
    matched; refusing presence would block a correct launch.
    Bool flags: presence of --<bool> sets True; --no_<bool> sets False. An
    agreeing bool flag is kept (Cfg default may differ from frozen); a
    conflicting one is refused."""
    s = set(_FROZEN_KEYS)
    clean, conflicts = [], []
    i = 0
    while i < len(passthrough):
        a = passthrough[i]
        if not a.startswith("--"):
            clean.append(a)
            i += 1
            continue
        flag = a[2:].split("=", 1)[0]
        cfg_key = _FLAG_TO_CFG.get(flag, flag)
        if cfg_key not in s:
            clean.append(a)
            i += 1
            continue
        fv = frozen[cfg_key]
        if isinstance(fv, bool):
            sets_true = flag == cfg_key  # --attn_res, --fone, etc.
            sets_false = flag in _FLAG_TO_CFG  # --no_attn_res, --no_doc_mask
            if (sets_true and not fv) or (sets_false and fv):
                conflicts.append(f"{a} (frozen {cfg_key}={fv})")
            else:
                clean.append(a)
            i += 1
        else:
            if "=" in a:
                v = a.split("=", 1)[1]
                i += 1
            elif i + 1 < len(passthrough):
                v = passthrough[i + 1]
                i += 2
            else:
                conflicts.append(f"{a} (no value)")
                continue
            if int(v) != fv:
                conflicts.append(f"{a} {v} (frozen {fv})")
    return clean, conflicts


def _run_point(step_args, forced):
    """One 0830v1 budget point. run_ddp.sh already scores the checkpoint on success,
    so this only has to launch it and record the row. --name is required; --mix defaults
    to train.py's default; everything else passes through to train.py.

    Ladder mixes (mix_scale_*) carry a frozen run config in
    data/mix_scale_run_config.json: run point sets the env + flags from it and refuses
    a disagreeing CLI flag. The six points must differ only in D."""
    name, mix, passthrough = None, None, []
    i = 0
    while i < len(step_args):
        a = step_args[i]
        if a == "--name" and i + 1 < len(step_args):
            name, i = step_args[i + 1], i + 2
        elif a.startswith("--name="):
            name, i = a.split("=", 1)[1], i + 1
        elif a == "--mix" and i + 1 < len(step_args):
            mix, i = step_args[i + 1], i + 2
        elif a.startswith("--mix="):
            mix, i = a.split("=", 1)[1], i + 1
        else:
            passthrough.append(a)
            i += 1
    if not name:
        print("run point: --name <n> is required")
        return 2
    mix = mix or cfg_default("mix")
    env = None
    frozen_args = []
    if os.path.basename(mix).startswith("mix_scale_"):
        fpath = os.path.join(ROOT, "data", "mix_scale_run_config.json")
        if not os.path.exists(fpath):
            print(f"run point: {fpath} missing -- the ladder recipe is not optional")
            return 2
        frozen = json.load(open(fpath, encoding="utf-8"))
        passthrough, conflicts = _strip_frozen(passthrough, frozen)
        if conflicts:
            print(f"run point: refusing -- frozen config disagrees: {'; '.join(conflicts)}")
            print("  edit data/mix_scale_run_config.json to change the ladder recipe (reopens the ladder)")
            return 2
        env = dict(
            os.environ, CUDA_VISIBLE_DEVICES=frozen["cards"], NGPU=str(len(frozen["cards"].split(",")))
        )
        frozen_args = [v for k in _FROZEN_KEYS if not isinstance(frozen[k], bool)
                       for v in (f"--{k}", str(frozen[k]))]
        print(
            f"run point: frozen config -> cards={frozen['cards']} "
            + " ".join(f"{k}={frozen[k]}" for k in _FROZEN_KEYS)
        )
    cmd = ["bash", os.path.join(ROOT, "run_ddp.sh"), "--mix", mix, "--name", name, *frozen_args, *passthrough]
    _exp("start", name=name, cmd=" ".join(cmd),
         hypothesis=f"0830v1 budget point, mix {os.path.basename(mix)}{forced}")
    rec = os.path.join(ROOT, "runs", "score_matrix.jsonl")
    ckpt = f"ckpt_{name}.pt"

    def _ckpt_record():
        if not os.path.exists(rec):
            return None
        for line in open(rec, encoding="utf-8"):
            if f'"ckpt": "{ckpt}"' in line:
                return line
        return None

    before = _ckpt_record()
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    # A rerun of the same ckpt must not pass on the FIRST run's record: the line has to
    # have changed, not merely be present (score_matrix --json replaces same-ckpt lines,
    # and only on success -- a failed rescore leaves the stale line in place).
    after = _ckpt_record()
    scored = r.returncode == 0 and after is not None and after != before
    _exp("done", name=name,
         result=f"exit {r.returncode}; {ckpt} scored in score_matrix" if scored else f"exit {r.returncode}",
         finding="score_matrix record is the result; the fit interprets" if scored else "run failed before scoring",
         decision="proceed to next point" if r.returncode == 0 else "investigate before next point",
         status="ok" if r.returncode == 0 else "fail")
    if r.returncode == 0 and scored:
        val = _val_nll(name)
        _board_event("point_landed", f"{name} scored: val {val:.3f}" if val else f"{name} scored")
    elif r.returncode != 0:
        _board_event("point_failed", f"{name} exited {r.returncode}")
    _refresh_board()
    return r.returncode


def _run_ladder(step_args, forced):
    """All six budget points, sequential, resumable. A point with a
    score-matrix record is skipped; a failed point stops the ladder.
    Each point runs through _run_point, which enforces the frozen config.
    The gate re-fires before every point: a red at hour two banks the points
    already done and stops, rather than launching the next point blind."""
    rec = os.path.join(ROOT, "runs", "score_matrix.jsonl")
    done = set()
    if os.path.exists(rec):
        for line in open(rec, encoding="utf-8"):
            m = re.search(r'"ckpt": "ckpt_(.+?)\.pt"', line)
            if m:
                done.add(m.group(1))
    for name, mix in LADDER:
        if name in done:
            print(f"ladder: {name} already scored, skipping")
            continue
        if _gate(bool(forced)) and not forced:
            print(f"ladder: harness red, stopping with {len(done)} point(s) banked")
            _board_event("check_red", f"ladder stopped at {name}: harness red, {len(done)} point(s) banked")
            _refresh_board()
            return 1
        print(f"ladder: starting {name} ({mix})")
        rc = _run_point(["--name", name, "--mix", mix], forced)
        if rc != 0:
            print(f"ladder: {name} failed (exit {rc}), stopping")
            _board_event("ladder_stopped", f"{name} failed (exit {rc}), {len(done)} point(s) banked")
            _refresh_board()
            return rc
    print("ladder: all six points complete")
    _board_event("ladder_complete", "all six points scored")
    _refresh_board()
    return 0


def run_dispatch(rest):
    """`harness run <step>` -- the only verb that executes. Thin dispatch, no new logic:
    every step refuses while check is red (--force records the reds in the exp row),
    writes its own exp.py start/done, and scores what it produced."""
    if not rest or rest[0] not in STEPS:
        print(f"usage: harness.py run <{'|'.join(STEPS)}> [step flags] [--force]")
        return 2
    step, step_args = rest[0], list(rest[1:])
    force = "--force" in step_args
    if force:
        step_args.remove("--force")
    reds = _gate(force)
    if reds and not force:
        return 1
    forced = f" [FORCED, red: {', '.join(n for n, _ in reds)}]" if reds else ""
    if step == "pretokenize":
        return _run_pretokenize(step_args, forced)
    if step == "point":
        return _run_point(step_args, forced)
    if step == "ladder":
        return _run_ladder(step_args, forced)
    if step == "fetch":
        return _run_pipeline_step("fetch", "fetch_corpus.py", step_args, forced)
    if step == "clean":
        return _run_pipeline_step("clean", "clean_corpus.py", step_args, forced)
    if step == "score":
        return _run_pipeline_step("score", "score_corpus.py", step_args, forced,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
    if step == "dedup":
        return _run_pipeline_step("dedup", "dedup_corpus.py", step_args, forced)
    return 2


def main():
    # argparse with choices, not a hand-rolled scan: a bare-flag filter once resolved
    # cmd="7", matched no branch, printed nothing and exited 0 -- a silent no-op, the
    # failure mode this file exists to prevent.
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        return run_dispatch(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "cmd", nargs="?", default="all", choices=["all", "check", "ledger", "gaps", "measure", "stages", "board"]
    )
    ap.add_argument("--json", action="store_true", help="board: emit state as JSON instead of HTML")
    ap.add_argument("--html", default=None, help="board: output path (default runs/board.html)")
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
    if cmd == "board":
        return cmd_board(as_json=a.json, html_path=a.html)
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
