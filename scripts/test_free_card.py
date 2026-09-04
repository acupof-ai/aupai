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
import json
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


def free_card(busy, wait=0, settle=1, grant=None):
    """Run `harness free-card` with a fake nvidia-smi holding `busy`, in a FIXTURE tree.

    THE GRANT IS A FIXTURE, not the live file, and that is the whole of CI red #4 (2026-09-04).
    This faked nvidia-smi and then ran against the real runs/card_assignment.json in cwd=ROOT, so
    its verdict moved with today's card allocation: 2eccf977 cleared the head-hybrid grant --
    launch_block_granted false, no block_cards, no lane_card, which is a correct and ordinary state
    -- and `an idle lane yields a card` went red on a CI runner while passing on every laptop whose
    checkout predated the clear. A test whose answer depends on which cards the controller granted
    an hour ago is measuring the allocation, not the code.

    The tree holds only the two files free-card reads, both root-relative: runs/card_assignment.json
    and data/mix_scale_run_config.json. Callers that care about the grant pass their own; the default
    is a block plus one lane card, which is the shape the lane rule assumes (AGENTS: a 7-card block
    and one lane card for everything else) and is what makes "idle lane yields a card" a question
    about free-card rather than about today.
    """
    d = tempfile.mkdtemp()
    p = os.path.join(d, "nvidia-smi")
    with open(p, "w") as f:
        f.write(SMI)
    os.chmod(p, 0o755)
    tree = os.path.join(d, "tree")
    os.makedirs(os.path.join(tree, "runs"))
    os.makedirs(os.path.join(tree, "data"))
    if grant is None:
        grant = {"launch_block_granted": True, "block_cards": "0-3", "lane_card": "4",
                 "granted_by": ["test_free_card fixture"]}
    with open(os.path.join(tree, "runs", "card_assignment.json"), "w") as f:
        json.dump(grant, f)
    with open(os.path.join(tree, "data", "mix_scale_run_config.json"), "w") as f:
        json.dump({"cards": "0,1,2,3", "world": 4}, f)
    env = dict(os.environ, PATH=d + os.pathsep + os.environ["PATH"], BUSY=" ".join(busy),
               AUPAI_ALLOC_ROOT=tree)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "harness.py"), "free-card",
         "--wait", str(wait), "--settle", str(settle)],
        capture_output=True, text=True, env=env, cwd=tree, timeout=180)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def main():
    bad = []
    n = 0

    def want(cond, name):
        nonlocal n
        n += 1
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

    # 2. free-card itself, against a fake smi AND a fixture grant.
    rc, out, err = free_card(busy=[])
    want(rc == 0 and out in [str(i) for i in range(8)],
         f"an idle lane yields a card (rc={rc}, out={out!r})")
    # THE FIXTURE IS IN EFFECT, asserted rather than assumed. Before d9ba571d this ran in cwd=ROOT
    # against the live runs/card_assignment.json, so it answered a question about today's grant:
    # 2eccf977 cleared the head-hybrid block (launch_block_granted false, no lane_card -- correct,
    # no job is granted) and the case above went red on CI while passing on any checkout that
    # predated the clear. If the fixture were ever bypassed again, `out` would be whatever card the
    # controller happens to have granted, so pinning it to the fixture's own lane is what makes the
    # case a question about free-card.
    want(out == "4", f"and it is the FIXTURE's lane card, not today's grant (out={out!r})")

    # NO LANE GRANTED -> refuse, naming the missing grant (6e's second case). Returning nothing when
    # no lane exists is the correct production behaviour and is exactly the state 2eccf977 wrote;
    # what must not happen is silence, because a caller reading an empty stdout as "no card yet"
    # queues forever against a grant that will never appear.
    rc_ng, out_ng, err_ng = free_card(
        busy=[], grant={"launch_block_granted": False, "block_cards": "", "lane_card": "",
                        "granted_by": ["test_free_card fixture: no grant"]})
    want(rc_ng != 0, f"no lane granted is refused, not answered (rc={rc_ng}, out={out_ng!r})")
    want(out_ng == "", f"and prints no card on stdout (out={out_ng!r})")
    want("grant" in err_ng.lower() or "lane" in err_ng.lower(),
         f"and the refusal names the missing grant rather than going silent (err={err_ng[:120]!r})")

    lane_free = "4"
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
    # COUNTED, not typed. It read "9 cases pass" while running 9, then 13 once this commit added
    # four -- the same defect card_claim's selftest carried ("10/10 pass" while running twelve): a
    # total that cannot notice a case going missing is not a total.
    print(f"\n{n} cases pass: the auto-score reads a card before it takes one, queues when none "
          f"is free, and reads its allocation from a fixture rather than today's grant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
