#!/usr/bin/env python3
"""Reviewer's mutant: TableMaster.resync on the rollback path, which nothing exercises.

Second-reader review of 1928e13e / 15d45870 / 440bd579 (tilerl for b0, 2026-09-05). The nine
listed cases cover construction, the disjointness of the two masters, pull_grads/push, the
optimizer group, the FP8 exclusion and readout 6's freeze detection. `resync` is called at
exactly one site -- train.py:2895, inside the 20-skip rollback -- and `grep -rn resync
scripts/ probes/` finds nothing, so the method has no live exercise at all.

WHY IT MATTERS RATHER THAN BEING A COVERAGE GAP. The rollback calls
raw_model.load_state_dict(good_state), which rewrites the bf16 table in place while the
master still holds the PRE-rollback fp32 copy. Without resync the next push() writes that
stale copy straight back over the restored table, so the rollback silently does not roll the
table back -- and it is silent: the run logs "rolled back to snapshot", the dense parameters
really are restored, and only the memory table is wrong. The docstring says this; nothing
tests it.

TWO WORLDS, because the assertion is about the DIFFERENCE. With resync the table after the
next push equals the snapshot; without it, the table equals the pre-rollback values. Asserting
only the first would pass on a build where load_state_dict happened to be a no-op.

    python3 scripts/test_table_master_resync.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def build():
    """A model whose only parameter of interest is a value table, at TableMaster's real
    shape contract: an nn.Embedding under a `...values.weight` fqn that _is_mem_fqn accepts.
    Built from the real ProductKeyMemory rather than a stub, so the fqn and dtype are the
    ones train.py produces."""
    import train
    from model import ProductKeyMemory

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory = ProductKeyMemory(64, 8, top_k=2)

    m = Holder().to(torch.bfloat16)
    tm = train.TableMaster(m)
    assert tm.pairs, "TableMaster matched no value table -- the fqn contract changed"
    return m, tm


def table_of(m):
    return m.memory.values.weight


def main():
    bad = 0
    torch.manual_seed(0)

    # SNAPSHOT, then diverge, then roll back -- train.py:2720/2886's sequence.
    m, tm = build()
    good_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    snap = table_of(m).detach().clone()

    with torch.no_grad():
        table_of(m).add_(1.0)          # the run moves on
        for _p, _mast in tm.pairs:
            _mast.copy_(table_of(m))   # and the master follows, as a step would leave it
    diverged = table_of(m).detach().clone()
    assert not torch.equal(snap, diverged), "the divergence did not take"

    # THE ROLLBACK, exactly train.py:2886-2895: load_state_dict then resync.
    m.load_state_dict(good_state)
    ok = torch.equal(table_of(m), snap)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} load_state_dict restores the table itself")

    for _m in [tm]:
        _m.resync()
    tm.push()
    ok = torch.equal(table_of(m), snap)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} with resync, the next push keeps the snapshot")

    # THE OTHER WORLD. Same sequence, resync SKIPPED -- the state the code would be in if
    # train.py:2894's loop were dropped. If this does not differ, the test above proves
    # nothing about resync.
    m2, tm2 = build()
    good2 = {k: v.cpu().clone() for k, v in m2.state_dict().items()}
    snap2 = table_of(m2).detach().clone()
    with torch.no_grad():
        table_of(m2).add_(1.0)
        for _p, _mast in tm2.pairs:
            _mast.copy_(table_of(m2))
    diverged2 = table_of(m2).detach().clone()
    m2.load_state_dict(good2)
    tm2.push()                                   # no resync
    ok = torch.equal(table_of(m2), diverged2) and not torch.equal(table_of(m2), snap2)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} WITHOUT resync the push reinstates the "
          f"pre-rollback table (max|table-snapshot| = "
          f"{(table_of(m2).float() - snap2.float()).abs().max():.3f})")

    # AND THE ROLLBACK IS SILENT ABOUT IT: the dense parameters really are restored, so
    # nothing in the run's own state says the table was not. This is the reason the gap is
    # worth a test rather than a comment.
    dense_ok = all(torch.equal(dict(m2.state_dict())[k], good2[k])
                   for k in good2 if not k.endswith("values.weight"))
    bad += 0 if dense_ok else 1
    print(f"  {'ok  ' if dense_ok else 'BUG '} every non-table parameter IS restored in that "
          f"world, so the failure shows nowhere else")

    # resync must not be a no-op in the direction that matters: it reads the TABLE into the
    # MASTER, never the reverse. A resync implemented backwards would pass the first
    # assertion above (push would write the master, which would still equal the table).
    m3, tm3 = build()
    with torch.no_grad():
        table_of(m3).fill_(2.0)
        for _p, _mast in tm3.pairs:
            _mast.fill_(9.0)
    tm3.resync()
    got = float(tm3.pairs[0][1].detach().flatten()[0])
    ok = abs(got - 2.0) < 1e-6
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} resync copies table -> master, not the reverse "
          f"(master reads {got:.1f}, table holds 2.0)")

    n = 5
    print(f"test_table_master_resync: {n - bad}/{n} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    # --selftest is what the pre-commit hook runs, and this file IS its own selftest: the
    # flag is accepted rather than ignored, because an unrecognised flag would make the hook
    # print a timing line for a run that did nothing.
    if len(sys.argv) > 1 and sys.argv[1] not in ("--selftest",):
        print(f"unknown flag {sys.argv[1]}", file=sys.stderr)
        sys.exit(2)
    sys.exit(main())
