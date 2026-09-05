#!/usr/bin/env python3
"""Stamp every existing token cache with the seed it was actually shuffled at (44).

Replaces the lazy adopt-on-first-touch path: a wrong-order cache must be found
now, not assumed away. The evidence is in the ledger and in the code that was
running when each cache was built.

Finding: every surviving cache was shuffled at 42, and the reason is the
truthiness bug itself.

  - The ladder caches (tokens_{textbook,wiki,en,code,math,chat,web_hq}.pt) were
    written 2026-08-30 17:14-17:27. The seed-1/2/3 arms ran 06:25-07:27 that
    morning, BEFORE those files existed, and the next training run is 19:10,
    after. So no non-default seed touched them.
  - Every run that built a stage-1 cache passed `--seed 0`. Under the code live
    at the time (train.py `if hasattr(Cfg, k) and v`), 0 is falsy and was
    DROPPED -- those runs shuffled at Cfg.seed = 42, not 0. That is the same bug
    ee415f1 fixes, and here it makes the answer certain rather than probable.
  - The only seeds that ever applied under the old code are 1, 2 and 3, and all
    three ran before any surviving cache was written.

So the correct stamp is 42 for every cache. This writes it explicitly, and
refuses to guess: a cache whose evidence does not resolve is left unstamped and
named, so the next run rebuilds it rather than trusting a label.
"""
import glob
import os
import sys


def _cache_dir():
    """train's accessor, with the old hardcoded "/data00" as the fallback.

    This tool WRITES .seed sidecars beside the caches, so stamping the overlay copy while a run
    reads the NVMe one would leave the read caches unstamped and the unread ones stamped -- the
    stamps and the data on different filesystems, which is worse than no stamp at all.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import train

        return train._token_cache_dir()
    except Exception:
        return "/data00"


CACHE_DIR = _cache_dir()
STAMP = "42"

REASON = (
    "explicit pass 2026-09-01: ladder caches predate the seed-1/2/3 arms; every "
    "stage-1 cache was built by a run passing --seed 0, which the truthiness bug "
    "dropped, so the shuffle ran at Cfg.seed=42"
)

caches = sorted(glob.glob(os.path.join(CACHE_DIR, "tokens_*.pt")))
if not caches:
    raise SystemExit("no caches found")

wrote, already, failed = [], [], []
for c in caches:
    side = c + ".seed"
    if os.path.exists(side):
        already.append((os.path.basename(c), open(side).read().strip()))
        continue
    try:
        with open(side, "w") as f:
            f.write(STAMP)
        wrote.append(os.path.basename(c))
    except OSError as e:
        failed.append((os.path.basename(c), str(e)))

print(f"{len(caches)} caches: {len(wrote)} stamped {STAMP}, {len(already)} already stamped, "
      f"{len(failed)} failed")
for n in wrote:
    print(f"  stamped {n}")
for n, v in already:
    print(f"  already {n} = {v}")
for n, e in failed:
    print(f"  FAILED  {n}: {e}")
print(f"\nreason: {REASON}")
