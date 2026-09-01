#!/usr/bin/env python3
"""Automatically-triggered scoring must measure a free card, not inherit one
(de, 2026-09-01, fb ruling).

The incident: run_ddp.sh:15 scored the checkpoint it had just trained, inside the
training shell, where line 3 has exported CUDA_VISIBLE_DEVICES=the seven-card block.
So the scorer took card 0 whatever card 0 was doing. On 2026-09-01 card 0 held
another process at 14.37 GiB and the scorer died asking for 96.00 MiB, three times,
and runs/score_matrix.jsonl gained zero records that day.

Nothing read a card. The card number came from the environment, and the environment
was the block. That is what makes it silent: no reading was wrong, because no
reading happened.

The two halves this asserts:

1. run_ddp.sh's auto-score does not inherit CUDA_VISIBLE_DEVICES -- it sets it from
   a measured free card, and skips with a WARN when none frees.
2. harness free-card returns only a card measured idle across a window, and exits
   nonzero (queue) rather than naming a busy card when every lane card is held.

Both are checked against a fake nvidia-smi, so the second half runs on a dev box.

    python3 scripts/test_free_card.py
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SMI = r"""#!/bin/bash
# fake nvidia-smi. BUSY names the card indices that have a compute process.
if [[ "$*" == *"--query-gpu=index,uuid"* ]]; then
  for i in 0 1 2 3 4 5 6 7; do echo "$i, GPU-$i"; done
  exit 0
fi
if [[ "$*" == *"--query-compute-apps=gpu_uuid"* ]]; then
  for c in $BUSY; do echo "GPU-$c"; done
  exit 0
fi
if [[ "$*" == *"--query-compute-apps=pid"* ]]; then
  for a in "$@"; do
    if [[ "$a" == --id=* ]]; then card=${a#--id=}; fi
  done
  for c in $BUSY; do [ "$c" = "$card" ] && echo 99999; done
  exit 0
fi
exit 0
"""


def free_card(busy, wait=0, settle=1):
    """Run `harness free-card` with a fake nvidia-smi holding `busy`."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "nvidia-smi")
    with open(p, "w") as f:
        f.write(SMI)
    os.chmod(p, 0o755)
    env = dict(os.environ, PATH=d + os.pathsep + os.environ["PATH"], BUSY=" ".join(busy))
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "harness.py"), "free-card",
         "--wait", str(wait), "--settle", str(settle)],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=180)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def main():
    bad = []

    def want(cond, name):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            bad.append(name)

    sh = open(os.path.join(ROOT, "run_ddp.sh"), encoding="utf-8").read()
    auto = sh[sh.index("rc -eq 0"):] if "rc -eq 0" in sh else ""

    # 1. The half that actually OOMed. The bare `python eval/score_matrix.py` on this
    # line inherits the block; it must carry a card of its own.
    want(bool(re.search(r"CUDA_VISIBLE_DEVICES=\"?\$\{?CARD", auto)),
         "run_ddp.sh's auto-score sets CUDA_VISIBLE_DEVICES from a measured card")
    want("free-card" in auto,
         "the card comes from `harness free-card`, i.e. from a reading at that moment")
    want(bool(re.search(r"free-card[^\n]*--wait", auto)),
         "it waits for a card (queue) instead of failing immediately")
    want("unscored" in auto or "no free lane card" in auto,
         "when no card frees it says the checkpoint went unscored -- silence would "
         "read as scored")
    # This one was vacuous on the first draft: `^\s*python` did not match the real
    # defect line, which is indented two spaces inside the if. A case that stays
    # green on the world it is supposed to catch is an assertion, not a test.
    want(not re.search(r"^\s+python eval/score_matrix\.py", auto, re.M),
         "no bare score_matrix call left on the auto path -- a bare call inherits "
         "the block")

    # 2. free-card itself, against a fake smi.
    rc, out, err = free_card(busy=[])
    want(rc == 0 and out in [str(i) for i in range(8)],
         f"an idle lane yields a card (rc={rc}, out={out!r})")

    lane_free = out
    rc, out, err = free_card(busy=[lane_free])
    want(rc != 0, f"a held lane card is refused, not returned (rc={rc}, out={out!r})")
    want(out == "", f"nothing is printed on stdout when queueing (out={out!r})")
    want("Queue" in err or "queue" in err,
         f"the refusal says to queue rather than spill into the block (err={err!r})")

    # The whole point: it must never name the busy card. This is the assertion that
    # fails on the unfixed world, where the card number was a default.
    want(lane_free not in out.split(),
         "the busy card is never returned as free")

    if bad:
        print(f"\n{len(bad)} case(s) failed: {bad}")
        return 1
    print("\n9 cases pass: the auto-score reads a card before it takes one, and "
          "queues when none is free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
