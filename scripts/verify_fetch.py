#!/usr/bin/env python3
"""Post-fetch verification for a corpus fetch: bytes, last-line JSON, line counts.

WHY THIS EXISTS. `fetch_corpus.py` records `source_fp` (a content hash of the MANIFEST)
for every source, but `expected_bytes` only for some: `_manifest_rp1t_c4()` and
`_manifest_rp1t_arxiv()` build their tuples as `[(name, url, 0) ...]` -- the byte column
is hardcoded 0, so comparing it to any real size passes trivially. On those two sources
the only real check after a download was "the file exists", which cannot tell a complete
shard from a truncated one.

WHAT REPLACES IT (fb's ruling 2026-09-18, options 2+3+4):
  2. runtime content-length per URL, captured BEFORE the fetch, compared to bytes on disk
  3. no `.part` leftovers, file count
  4. each file reopened, its last bytes read, and its FINAL LINE parsed as JSON

(4) is the one that catches what (2) cannot: a transfer whose byte count matches but whose
tail is damaged (a partial write behind a correct content-length, or a mirror that served
a truncated body with a matching header). A truncated JSONL ends mid-object, so
`json.loads(last_line)` fails while the size check passes.

These are SUBSTITUTES, not the original criterion, and any report must say so: the real
gap is the missing `expected_bytes` column, which is filed separately so the next rebuild
of these sources has a real judge instead of this one.

    python3 scripts/verify_fetch.py --dir data/raw/rp1t_c4 [--lengths /tmp/len.json]
"""
# restartable: read-only over the fetched files; it opens each one, reads a bounded tail and
# counts lines. Nothing is written, so an interrupt costs only the re-run.
import argparse
import json
import os
import sys


def last_line_json_ok(path, tail_bytes=65536):
    """(ok, detail). Reopens the file and parses its final non-empty line as JSON.

    Reads a bounded tail rather than the whole file: a shard is ~845 MB and the question
    is only whether the stream ended cleanly, which the last line answers.
    """
    size = os.path.getsize(path)
    if size == 0:
        return False, "empty file"
    with open(path, "rb") as f:
        f.seek(max(0, size - tail_bytes))
        chunk = f.read()
    lines = [ln for ln in chunk.split(b"\n") if ln.strip()]
    if not lines:
        return False, "no non-empty line in the tail"
    try:
        obj = json.loads(lines[-1])
    except Exception as e:
        return False, f"last line is not JSON: {type(e).__name__}: {e}"
    keys = sorted(obj)[:6] if isinstance(obj, dict) else type(obj).__name__
    return True, f"last line parses, keys={keys}"


def count_lines(path, cap=5_000_000):
    """Line count, bounded. A count that silently stops would understate the corpus, so
    hitting the cap is REPORTED rather than returned as if it were the total."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
            if n >= cap:
                return n, True
    return n, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pattern", default=".jsonl")
    ap.add_argument(
        "--lengths", help="JSON file of {basename: content_length} captured before the fetch (option 2)"
    )
    a = ap.parse_args()

    expect = {}
    if a.lengths and os.path.exists(a.lengths):
        expect = json.load(open(a.lengths, encoding="utf-8"))

    files = sorted(f for f in os.listdir(a.dir) if f.endswith(a.pattern))
    parts = sorted(f for f in os.listdir(a.dir) if ".part" in f)
    total = 0
    bad = []
    rows = []
    for fn in files:
        p = os.path.join(a.dir, fn)
        sz = os.path.getsize(p)
        total += sz
        ok_last, why_last = last_line_json_ok(p)
        n, capped = count_lines(p)
        mism = ""
        if fn in expect and expect[fn] and expect[fn] != sz:
            mism = f"content-length {expect[fn]} != on-disk {sz}"
            bad.append((fn, mism))
        if not ok_last:
            bad.append((fn, why_last))
        rows.append((fn, sz, n, capped, ok_last, mism or why_last))

    print(f"{a.dir}: {len(files)} file(s), {total / 1e9:.2f} GB on disk")
    if parts:
        # COUNTED AS A FAILURE, not just printed. The first version reported the leftovers
        # and then, three lines later, printed "no .part leftovers" and returned 0 on the
        # same run -- measured on rp1t_c4 2026-09-18, where it listed 8 chunks and exited
        # clean. `.part` chunks are a download that did not finish; a reader who trusts the
        # summary line over the listing gets the opposite answer, which is worse than no
        # report at all.
        print(f"  .PART LEFTOVERS: {len(parts)} -> {parts[:4]}")
        bad.append((".part", f"{len(parts)} leftover chunk file(s), e.g. {parts[:3]}"))
    checked = sum(1 for fn, *_ in rows if fn in expect and expect[fn])
    print(
        f"  content-length compared on {checked}/{len(files)} file(s)"
        + ("" if checked == len(files) else "  <-- NOT ALL, say so in any report")
    )
    print(f"  last-line JSON ok on {sum(1 for r in rows if r[4])}/{len(files)}")
    for fn, sz, n, capped, ok, why in rows[:6]:
        print(f"    {fn}  {sz / 1e6:.1f} MB  {n} lines{'+ (CAPPED)' if capped else ''}  {why[:70]}")
    if len(rows) > 6:
        print(f"    ... and {len(rows) - 6} more")
    if bad:
        print(f"\n  FAILURES ({len(bad)}):")
        for fn, why in bad[:10]:
            print(f"    {fn}: {why}")
        return 1
    # Each clause restates a check that actually ran. "no .part leftovers" is printed only
    # when the `.part` scan found none -- the earlier version asserted it unconditionally.
    print(
        f"\nAll {len(files)} file(s): tail parses"
        + (", content-length matched" if checked == len(files) and checked else "")
        + ", no .part leftovers."
    )
    if not checked:
        print(
            "NOTE: expected_bytes does not exist for this source (manifest hardcodes 0); "
            "content-length was NOT compared -- bytes on disk are unverified against the "
            "upstream size. The tail-JSON check above is what stands in for it."
        )
    return 0


def _selftest():
    """Known-answer worlds for the two defects this file shipped with.

    A temp dir per world, built from real files: one clean (a valid jsonl, no .part), one
    with a leftover `.part`, one whose last line is torn. The assertions are on the EXIT
    CODE, because that is what a caller acts on -- the first version printed the leftovers
    and still returned 0.
    """
    import tempfile

    def run(d, extra=()):
        argv = sys.argv
        try:
            sys.argv = ["verify_fetch.py", "--dir", d, *extra]
            return main()
        finally:
            sys.argv = argv

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "a.jsonl"), "w") as fh:
            fh.write(json.dumps({"text": "hello", "meta": {}}) + "\n")
            fh.write(json.dumps({"text": "world"}) + "\n")
        with open(os.path.join(d, "b.jsonl"), "w") as fh:
            fh.write(json.dumps({"text": "ok"}) + "\n")
        assert run(d) == 0, "a clean dir must exit 0"

        # a .part leftover must FAIL: this is the defect (reported, then exit 0)
        with open(os.path.join(d, "c.jsonl.part.c0"), "wb") as fh:
            fh.write(b"x" * 16)
        rc = run(d)
        assert rc == 1, f"a .part leftover must exit nonzero, got {rc}"
        os.remove(os.path.join(d, "c.jsonl.part.c0"))

        # a torn tail must FAIL: the check that stands in for a missing content-length
        with open(os.path.join(d, "torn.jsonl"), "w") as fh:
            fh.write(json.dumps({"text": "complete"}) + "\n")
            fh.write('{"text": "truncated mid-obj')
        rc = run(d)
        assert rc == 1, f"a non-JSON final line must exit nonzero, got {rc}"
        os.remove(os.path.join(d, "torn.jsonl"))

        # content-length: a recorded length that disagrees must FAIL, and a matching one pass
        lens = os.path.join(d, "lengths.json")
        sizes = {f: os.path.getsize(os.path.join(d, f)) for f in ("a.jsonl", "b.jsonl")}
        with open(lens, "w") as fh:
            json.dump(sizes, fh)
        assert run(d, ["--lengths", lens]) == 0, "matching content-lengths must pass"
        sizes["a.jsonl"] += 1
        with open(lens, "w") as fh:
            json.dump(sizes, fh)
        rc = run(d, ["--lengths", lens])
        assert rc == 1, f"a content-length mismatch must exit nonzero, got {rc}"

    print(
        "verify_fetch selftest OK: clean passes; a .part leftover, a torn tail and a "
        "content-length mismatch each exit nonzero (the .part case was the bug: reported "
        "in the listing, then exit 0)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
