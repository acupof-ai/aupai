"""For each check that FAILs on an UNMUTATED world, does its own broken() FAIL for the
mutation or for the same absence?

unmutated_fail.py says which checks FAIL on an empty world. That alone is not a defect:
a broken() that writes the missing file back and then corrupts it is fine. The defect is
a broken() whose FAIL evidence is the SAME string as the unmutated world's -- there the
selftest green says nothing about the mutation.

Compares run(broken()) evidence against run(unmutated world of the same base) evidence.
Same first 60 chars => confounded.

  python3 runs/audit_0904/confounded_worlds.py
  python3 runs/audit_0904/confounded_worlds.py --selftest
"""

import os
import sys

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness as H  # noqa: E402
from scan_broken_worlds import scan  # noqa: E402


def measure(only=None):
    bare, shaped = H._tmp_repo(), H._tmp_repo_shaped()
    base = {name: tags for name, _, _, tags in scan()}
    out = []
    for name, _asserts, _incident, fn, broken in H.CHECKS:
        if only and name not in only:
            continue
        tags = base.get(name, set())
        which = "shaped" if "shaped" in tags else ("bare" if "bare" in tags else None)
        if which is None:
            out.append((name, None, "no-base", "", ""))
            continue
        try:
            st_u, ev_u = fn(bare if which == "bare" else shaped)
        except Exception as e:
            st_u, ev_u = "RAISE", f"{type(e).__name__}: {e}"
        if st_u != "FAIL":
            continue
        try:
            d = broken()
            st_b, ev_b = fn(d)
        except H.SelftestSkip as e:
            st_b, ev_b = "SELFTEST-SKIP", str(e)
        except Exception as e:
            st_b, ev_b = "RAISE", f"{type(e).__name__}: {e}"
        same = str(ev_u)[:60] == str(ev_b)[:60]
        out.append((name, which, "CONFOUNDED" if same else "distinct",
                    str(ev_u)[:150], f"[{st_b}] " + str(ev_b)[:150]))
    return out


def _selftest():
    # Known answer: mix_not_unfiltered FAILs the bare world for "cannot read the default
    # mix", and its broken() writes a real mix with a web domain, so the two evidences
    # must DIFFER. If this scan calls that confounded, its comparison is wrong.
    rows = measure(only={"mix_not_unfiltered"})
    assert rows, "mix_not_unfiltered did not reach the comparison"
    name, which, verdict, ev_u, ev_b = rows[0]
    assert verdict == "distinct", (
        f"mix_not_unfiltered called {verdict}; its broken() writes a mix naming `web`, so its "
        f"FAIL must not be the bare world's 'cannot read the default mix'\n  unmutated: {ev_u}\n  broken:    {ev_b}"
    )
    assert "cannot read" in ev_u, f"the unmutated evidence changed shape: {ev_u}"
    print("confounded_worlds selftest ok (known-answer distinct case held)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)
    rows = measure()
    bad = [r for r in rows if r[2] == "CONFOUNDED"]
    print(f"{len(rows)} checks FAIL an unmutated world of their own base; {len(bad)} CONFOUNDED\n")
    for name, which, v, ev_u, ev_b in sorted(rows, key=lambda r: (r[2] != "CONFOUNDED", r[0])):
        print(f"{v:11s} {name} ({which})")
        print(f"    unmutated: {ev_u}")
        print(f"    broken():  {ev_b}")
