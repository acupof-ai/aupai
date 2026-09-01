#!/usr/bin/env python3
"""judge_pod_ps must FAIL on a foreground trainer and PASS on the four shapes that
were false positives (de, 2026-09-01).

Every row below is captured `ps -eo pid,sid,pgid,ppid,stat,args --no-headers` output
from the pod, verbatim -- including the FAIL case, which was produced deliberately:
`pod "cd /work/aupai && ./run_ddp.sh --name fixture_fg_probe --help; sleep 40"`
launches a real torchrun in a real crictl exec session, exits in seconds on --help,
and was confirmed off every card before capture. An earlier version of this file
hand-abridged the PASS fixture and the test failed on rows it had dropped, which is
the repo's own rule about broken worlds arriving from the other direction.

The check reads live pod state and its broken() raises SelftestSkip, so nothing
exercised it except whatever the pod happened to be doing. It produced four false
positives in one day; each refused a commit while the pod behaved as intended.

    python3 scripts/test_pod_ps_judge.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from harness import FAIL, PASS, judge_pod_ps  # noqa: E402


def rows(text):
    out = []
    for ln in text.strip().splitlines():
        p = ln.split(None, 5)
        if len(p) == 6 and p[0].isdigit():
            out.append(tuple(p))
    return out


# Two detached jobs and their whole session closure. Carries all four false-positive
# shapes at once: a launcher shell whose argv quotes the run (1312580), a zombie
# trainer (1367615), a zombie session LEADER (1370843), and trainers adopted by init.
LIVE = r"""
1312580 1312569 1312569       1 S    bash -lc cd /work/aupai && setsid nohup bash -c 'CUDA_VISIBLE_DEVICES=4,5,6 NGPU=3 PORT=29513 ./run_ddp.sh --mix data/mix_30b_stage2.json --name shape500_probe --dim 1024 --layers 32 --heads 8 --ffn_hidden 3072 --batch 16 --accum 2 --grad_ckpt --lr_scale 0.85 --max_steps 300 --save_every 100000 > runs/shape500_probe.log 2>&1' </dev/null >/dev/null 2>&1 &
1312581 1312581 1312581 1312580 Ss   bash -c CUDA_VISIBLE_DEVICES=4,5,6 NGPU=3 PORT=29513 ./run_ddp.sh --mix data/mix_30b_stage2.json --name shape500_probe --dim 1024 --layers 32 --heads 8 --ffn_hidden 3072 --batch 16 --accum 2 --grad_ckpt --lr_scale 0.85 --max_steps 300 --save_every 100000 > runs/shape500_probe.log 2>&1
1312582 1312581 1312581 1312581 S    /bin/bash ./run_ddp.sh --mix data/mix_30b_stage2.json --name shape500_probe --dim 1024 --layers 32 --heads 8 --ffn_hidden 3072 --batch 16 --accum 2 --grad_ckpt --lr_scale 0.85 --max_steps 300 --save_every 100000
1367615 1367423 1367423       1 Z    [run_ddp.sh] <defunct>
1370843 1370843 1370843       1 Zs   [bash] <defunct>
1382838 1370843 1370843       1 S    /bin/bash ./run_ddp.sh --mix data/mix_30b_stage2.json --name shape500_b32 --batch 32 --accum 1 --max_steps 120
1382840 1370843 1370843 1382838 Sl   /usr/bin/python3 /usr/local/bin/torchrun --nproc_per_node=3 --master_port=29515 train.py --fp8 --name shape500_b32
1382917 1382917 1382917 1382840 Ssl  /usr/bin/python3 -u train.py --fp8 --mix data/mix_30b_stage2.json --name shape500_b32
1382918 1382918 1382918 1382840 Rsl  /usr/bin/python3 -u train.py --fp8 --mix data/mix_30b_stage2.json --name shape500_b32
"""

# THE FAIL CASE, captured live. `pod "<cmd>"` becomes `crictl exec ... bash -lc <cmd>`,
# and that shell is its own session leader with the whole job inside its session. It
# dies with the tn tunnel after ~5 minutes and the trainer keeps the cards.
FOREGROUND = r"""
1389335 1389335 1389335       0 Ss   bash -lc cd /work/aupai && ./run_ddp.sh --name fixture_fg_probe --help >/dev/null 2>&1; sleep 40
1389346 1389335 1389335 1389335 S    /bin/bash ./run_ddp.sh --name fixture_fg_probe --help
1389348 1389335 1389335 1389346 Sl   /usr/bin/python3 /usr/local/bin/torchrun --nproc_per_node=8 --master_port=29500 train.py --fp8 --name fixture_fg_probe --help
1389417 1389417 1389417 1389348 Rsl  /usr/bin/python3 -u train.py --fp8 --name fixture_fg_probe --help
1389418 1389418 1389418 1389348 Rsl  /usr/bin/python3 -u train.py --fp8 --name fixture_fg_probe --help
"""

ZOMBIE_ONLY = "1367615 1367423 1367423       1 Z    [run_ddp.sh] <defunct>\n"

# Captured: the launcher shell alone, before its job appeared. Its argv quotes the
# whole run_ddp command, which is what the first false positive matched on.
LAUNCHER_ONLY = r"""
1312580 1312569 1312569       1 S    bash -lc cd /work/aupai && setsid nohup bash -c './run_ddp.sh --name shape500_probe > runs/x.log 2>&1' </dev/null >/dev/null 2>&1 &
"""

CASES = [
    ("a foreground launch is caught", FOREGROUND, FAIL),
    ("two detached jobs, zombie leader and zombie trainer", LIVE, PASS),
    ("the launcher shell alone is not a violation", LAUNCHER_ONLY, PASS),
    ("a zombie alone is no training process", ZOMBIE_ONLY, PASS),
    ("an idle pod", "", PASS),
]


def main():
    bad = []
    for name, fixture, want in CASES:
        state, ev = judge_pod_ps(rows(fixture))
        ok = state == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {state} -- {ev}")
        if not ok:
            bad.append(name)
    if bad:
        print(f"\n{len(bad)} case(s) wrong: {bad}")
        return 1
    print(f"\n{len(CASES)} cases pass: a foreground trainer is still caught, and none "
          "of the four false-positive shapes trips it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
