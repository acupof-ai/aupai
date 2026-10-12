"""E0-E6 evaluation ladder stage modules.

One module per stage, one author per module (docs/standards/eval_ladder_v1_1009.md
section 4). Each module exposes:

    main(argv) -> int          the stage, dispatched by eval/eval_ladder.py
    control_selftest() -> int  CPU-only positive/negative controls, no checkpoint

`eval/ladder_common.py` is owned by the main controller and is not edited here.
"""
