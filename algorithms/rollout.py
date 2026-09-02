#!/usr/bin/env python3
"""K=8 parallel rollouts: one sandbox and one workdir each, reward + isolation per rollout.

Stage 3 of docs/lessons/claude_code_rl.md, the piece between isolate.py (one process, one
level) and a GRPO step (K rollouts, one advantage). What it adds over calling reward_fn in a
loop:

  isolation of the rollouts FROM EACH OTHER, not only from the host. Each rollout gets its
  own directory -- a git worktree at the task's commit when a repo is given, a mkdtemp
  otherwise -- so rollout 3 cannot see the files rollout 5 wrote. Asserted with a pair of
  variants that write a marker and look for the other's.

  a record per rollout, carrying the level it ran at. A reward earned under rlimits_only and
  one earned under a namespace are different measurements; a group whose 8 rollouts ran at
  different levels is a group whose advantages are not comparable, and `levels_agree` in the
  group result says so instead of leaving it to be noticed.

  real concurrency. The work is subprocesses, so threads are enough; the suite measures wall
  clock against the serial sum, because "K=8 parallel" is a claim and not a design note.

WHAT AN ESCAPE RECORD CAN AND CANNOT SAY. `escaped` scans the rollout's output for the
REACHED marker its own test file asserts on. It catches a variant that tries the escape and
says so, which is what the artificial variants do; it says nothing about code that
exfiltrates quietly. The guarantee is the level, not the scan -- so `level` and `isolates`
travel in every record and the scan is corroboration.

    python3 algorithms/rollout.py --selftest    # 10 variants: network, ~/.ssh, hang, pass, ...
"""

# restartable: a group is K independent rollouts and nothing is carried between groups. Every
# rollout writes only inside its own mkdtemp or its own throwaway worktree, both removed in a
# finally; --pairs appends one JSON line per group, so an interrupt costs the group in flight
# and re-running re-does only that group. An interrupted --repo run can leave at most K
# worktrees registered, cleared with `git worktree prune`.

import argparse
import concurrent.futures as cf
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "algorithms"))

from code_reward import score  # noqa: E402
from isolate import ISOLATES, detect_level  # noqa: E402

ESCAPE_MARKER = "REACHED"


def _worktree(repo, commit, dest):
    subprocess.run(
        ["git", "-C", repo, "worktree", "add", "--detach", "-f", dest, commit],
        check=True,
        capture_output=True,
    )


def _worktree_rm(repo, dest):
    subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", dest], capture_output=True)


def rollout(code, tests, index=0, label="", repo=None, commit="HEAD", timeout=30, level=None):
    """One rollout. Returns a record: reward, level, isolates, escaped, rc, timed_out, secs."""
    parent = tempfile.mkdtemp(prefix=f"rollout{index}.")
    work = os.path.join(parent, "w")
    try:
        if repo:
            _worktree(repo, commit, work)
        else:
            os.makedirs(work)
        t0 = time.time()
        r = score(code, tests, timeout=timeout, level=level, workdir=work)
        secs = time.time() - t0
        return {
            "index": index,
            "label": label,
            "reward": r["reward"],
            "level": r["level"],
            "isolates": ISOLATES[r["level"]],
            "escaped": ESCAPE_MARKER in r["stdout"],
            "rc": r["rc"],
            "timed_out": r["timed_out"],
            "risk": r["risk"],
            "secs": round(secs, 2),
            "stdout": r["stdout"][-600:],
            "stderr": r["stderr"][-600:],
        }
    finally:
        if repo:
            _worktree_rm(repo, work)
        shutil.rmtree(parent, ignore_errors=True)


def run_group(variants, tests, k=8, repo=None, commit="HEAD", timeout=30, level=None):
    """Run up to k rollouts concurrently. variants: [(code, label)] or [(code, tests, label)].

    Returns {records, k, levels_agree, wall, serial}. `levels_agree` false means the rollouts
    did not all run at the same isolation level, so their rewards are not one group.
    """
    lvl = level or detect_level()
    items = []
    for i, v in enumerate(variants):
        code, t, label = v if len(v) == 3 else (v[0], tests, v[1])
        items.append((i, code, t, label))
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=k) as ex:
        futs = [ex.submit(rollout, c, t, i, lab, repo, commit, timeout, lvl) for i, c, t, lab in items]
        records = [f.result() for f in futs]
    records.sort(key=lambda r: r["index"])
    levels = {r["level"] for r in records}
    return {
        "records": records,
        "k": k,
        "levels_agree": len(levels) == 1,
        "levels": sorted(levels),
        "wall": round(time.time() - t0, 2),
        "serial": round(sum(r["secs"] for r in records), 2),
    }


_TESTS = (
    "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n"
)

# The escape attempts live in the TEST file, not the implementation, and they ASSERT rather
# than print. The first version printed the REACHED marker from the impl at import time and
# scanned stdout for it -- pytest captures that stdout, so the marker could never appear and
# `escaped: False` was true of every run including one that escaped. A check that cannot fire
# is not a check; the shape is "a world failing for the wrong reason proves nothing", one
# level down.
#
# As an assertion the verdict is the reward: isolation holds -> the test passes -> 1.0;
# the rollout reaches the network -> the test fails -> 0.0 AND pytest prints the assertion
# message, which is where the REACHED marker becomes visible to the stdout scan. So both
# signals work in the direction that matters, and neither is silently inert.
_NET_TEST = (
    "from solution import add\n"
    "import socket\n\n\n"
    "def test_add():\n    assert add(2, 3) == 5\n\n\n"
    "def test_network_is_unreachable():\n"
    "    try:\n"
    "        socket.socket().connect(('1.1.1.1', 80))\n"
    "        reached = True\n"
    "    except Exception:\n        reached = False\n"
    "    assert not reached, 'REACHED the network from inside a rollout'\n"
)

_SSH_TEST = (
    "from solution import add\n"
    "import os\n\n\n"
    "def test_add():\n    assert add(2, 3) == 5\n\n\n"
    "def test_home_secrets_unreadable():\n"
    "    hit = None\n"
    "    for c in ('/Users/bytedance/.ssh', '/root/.ssh',\n"
    "              os.path.expanduser('~/.ssh')):\n"
    "        try:\n"
    "            if os.listdir(c):\n                hit = c; break\n"
    "        except Exception:\n            pass\n"
    "    assert hit is None, f'REACHED {hit} from inside a rollout'\n"
)

# The eight variants fb named, plus the two halves of the cross-rollout assertion.
# (impl, tests, expected_reward, label)
VARIANTS = [
    ("def add(a, b):\n    return a + b\n", _TESTS, 1.0, "all tests pass"),
    ("def add(a, b):\n    return a - b\n", _TESTS, 0.0, "wrong answer"),
    (
        "def add(a, b):\n    return a + b\n",
        _NET_TEST,
        1.0,
        "tries the network (reward 1.0 means it was blocked)",
    ),
    ("def add(a, b):\n    return a + b\n", _SSH_TEST, 1.0, "tries ~/.ssh (reward 1.0 means it was blocked)"),
    ("def add(a, b):\n    while True:\n        pass\n", _TESTS, 0.0, "infinite loop"),
    (
        "import pathlib\n"
        "pathlib.Path('test_solution.py').write_text('def test_x():\\n    assert True\\n')\n"
        "def add(a, b):\n    return a - b\n",
        _TESTS,
        0.0,
        "rewrites the test file",
    ),
    ("def add(a, b:\n    return a\n", _TESTS, 0.0, "syntax error"),
    ("", _TESTS, 0.0, "empty implementation"),
    # The isolation of the rollouts from each other, as two halves of one assertion: each
    # writes a marker and asserts the other's is absent. In the test file for the same
    # reason as the probes above -- an assertion is observable, a print is captured.
    (
        "def add(a, b):\n    return a + b\n",
        "from solution import add\n"
        "import os\n\n\n"
        "def test_isolated_from_other_rollouts():\n"
        "    open('marker_a', 'w').write('a')\n"
        "    assert not os.path.exists('marker_b'), 'REACHED another rollout workdir'\n"
        "    assert add(2, 3) == 5\n",
        1.0,
        "cross-rollout marker A",
    ),
    (
        "def add(a, b):\n    return a + b\n",
        "from solution import add\n"
        "import os, time\n\n\n"
        "def test_isolated_from_other_rollouts():\n"
        "    open('marker_b', 'w').write('b')\n"
        "    time.sleep(2)\n"
        "    assert not os.path.exists('marker_a'), 'REACHED another rollout workdir'\n"
        "    assert add(2, 3) == 5\n",
        1.0,
        "cross-rollout marker B (sleeps 2s)",
    ),
]


def _selftest(repo=None):
    lvl = detect_level()
    print(
        f"level: {lvl}   k=8, {len(VARIANTS)} variants"
        + (f", worktrees of {repo}" if repo else ", mkdtemp workdirs")
    )
    g = run_group([(c, t, lab) for c, t, _, lab in VARIANTS], _TESTS, k=8, repo=repo, timeout=25, level=lvl)
    bugs = 0
    for r, (_, _, exp, lab) in zip(g["records"], VARIANTS, strict=True):
        ok = r["reward"] == exp
        bugs += 0 if ok else 1
        print(
            f"  {'ok  ' if ok else 'BUG '} reward {r['reward']:.1f} want {exp:.1f}  "
            f"rc={r['rc']:<4} {r['secs']:>5.1f}s  esc={r['escaped']!s:<5} {lab}"
        )

    assert g["levels_agree"], f"rollouts ran at different levels: {g['levels']}"
    for r in g["records"]:
        assert r["level"] == lvl, r
        assert r["isolates"] == ISOLATES[lvl], r

    # The escape probes. Each is an assertion inside the rollout's test file, so isolation
    # holding means the test passes: reward 1.0 and no marker. A breach fails the test, which
    # is reward 0.0 AND the marker in pytest's output -- both checks fire, in that order.
    net, ssh = g["records"][2], g["records"][3]
    if ISOLATES[lvl]["net"]:
        assert net["reward"] == 1.0 and not net["escaped"], (
            f"a rollout REACHED the network at level {lvl}: {net}"
        )
    if ISOLATES[lvl]["fs_read"]:
        assert ssh["reward"] == 1.0 and not ssh["escaped"], (
            f"a rollout read a home secret at level {lvl}: {ssh}"
        )

    loop = g["records"][4]
    assert loop["reward"] == 0.0 and loop["rc"] != 0, f"the infinite loop did not fail: {loop}"

    a, b = g["records"][8], g["records"][9]
    for r, other in ((a, "B"), (b, "A")):
        # Two ways this reward can be 0.0 and only one of them is an isolation breach. When
        # the uid drop landed, the marker write started failing with EACCES (the cwd was the
        # root-owned chroot root, not the workdir) and this assertion reported "a peer's
        # workdir was visible" -- a correct red for the wrong reason, which costs the time it
        # takes to disbelieve the message. The marker's own assertion carries REACHED, so
        # `escaped` separates the two; anything else is the rollout failing to run at all.
        if r["reward"] != 1.0 and not r["escaped"]:
            raise AssertionError(
                f"rollout {other}'s marker test did not run, so cross-rollout isolation is "
                f"UNTESTED (not breached -- no REACHED marker). Read the failure: {r}")
        assert r["reward"] == 1.0 and not r["escaped"], (
            f"rollout {other}'s workdir was visible to its peer: {r}"
        )

    # Concurrency, measured. B sleeps 2s and the loop burns its CPU limit, so a serial run
    # cannot come close to the wall clock a parallel one does.
    assert g["wall"] < g["serial"] * 0.6, (
        f"not parallel: wall {g['wall']}s vs serial sum {g['serial']}s -- k=8 rollouts ran one at a time"
    )
    print(
        f"\nwall {g['wall']}s vs serial {g['serial']}s "
        f"({g['serial'] / max(g['wall'], 0.01):.1f}x), levels {g['levels']}, "
        f"{bugs} bug(s)"
    )
    rewards = [r["reward"] for r in g["records"]]
    print(
        f"rewards: {rewards}  spread {min(rewards)}..{max(rewards)} "
        f"({'GRPO can separate this group' if len(set(rewards)) > 1 else 'DEGENERATE'})"
    )
    if bugs:
        print("FAIL: a variant's reward did not match. The reward is training signal.")
    return 1 if bugs else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--repo", default=None, help="git repo to make a per-rollout worktree from")
    ap.add_argument("--pairs", metavar="JSONL", help="run one group per pair {impl,tests}")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--json", metavar="OUT", help="write the records here")
    a = ap.parse_args()
    if a.selftest:
        return _selftest(repo=a.repo)
    if a.pairs:
        with open(a.pairs, encoding="utf-8") as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
        # Append-and-flush per group, not one write at the end: the restartable claim in the
        # header is that an interrupt costs the group in flight, and buffering every group
        # until the last one makes it cost all of them.
        with contextlib.ExitStack() as stack:
            out = stack.enter_context(open(a.json, "a", encoding="utf-8")) if a.json else None
            for i, row in enumerate(rows):
                g = run_group(
                    [(row["impl"], f"pair{i}") for _ in range(a.k)], row["tests"], k=a.k, repo=a.repo
                )
                print(
                    f"pair {i}: rewards {[r['reward'] for r in g['records']]} "
                    f"wall {g['wall']}s levels {g['levels']}",
                    flush=True,
                )
                if out:
                    out.write(json.dumps({"pair": i, **g}) + "\n")
                    out.flush()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
