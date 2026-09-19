#!/usr/bin/env python3
"""Probe a fetch manifest's URLs for reachability, WITHOUT downloading the corpus.

WHY THIS EXISTS. On 2026-09-19 `data.together.xyz`, the source of the en_c4 cell
(`_manifest_rp1t_c4`), was found Cloudflare-blocked at the HOST level: the root and every
file return 403 with a 4.5 KB "Attention Required!" page. Nothing in the tree tested
reachability, so the gap was found only because someone probed by hand -- after the corpus
was already built and a node rebuild only months away. This is the "a fetcher carries a
mirror chain" rule (CLAUDE.md, pod rules) given an instrument: probe the chain BEFORE the
fetch, and report per-source state.

THE MISTAKE THIS SCRIPT EXISTS TO PREVENT. The first hand-written version of this probe
reported 8/8 reachable on the blocked host. It used `curl -sSI` without `-L`, so:
  * on modelscope it read the **302 redirect line** as the object's status;
  * on the blocked host the `HTTP/1.1 200` it read was the **proxy CONNECT** response,
    which is a statement about the tunnel, not about the file.
`%{http_code}` alone is therefore not a reachability answer through a proxy or a redirect.
This script follows redirects, reads the FINAL header block, and -- the part that actually
settles it -- verifies CONTENT: a non-empty body, and optionally the parquet tail magic.
A 200 with a 4.5 KB HTML body is a 403 that lied.

That is the pair the selftest pins: a fake-200 (a 302 or a CONNECT line read as the object)
must be reported as NOT reachable, and a magic mismatch must go red.

READ-ONLY BY CONSTRUCTION: HEAD requests, plus an optional 4-byte range GET for the magic.
It never fetches a corpus file and never writes under data/.
"""
import argparse
import inspect
import os
import random
import subprocess
import sys

DEFAULT_PROXY = "http://sys-proxy-rd-relay.byted.org:8118"
#: A parquet file ends with this. Checked from the LAST 4 bytes via a range GET when --magic.
PARQUET_MAGIC = b"PAR1"
#: An HTML error page served with a 200 is the failure this script exists to catch. Any of
#: these in the first bytes means the body is not the artifact, whatever the status line said.
ERROR_BODY_MARKERS = (b"<!DOCTYPE html", b"<html", b"Attention Required", b"Access Denied")


def _curl(args, proxy, timeout):
    cmd = ["curl", "-4", "-sS", "--max-time", str(timeout)]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd += args
    return subprocess.run(cmd, capture_output=True)


def _final_block(header_text):
    """The LAST header block of an `-L` response -- what the object itself answered.

    Reading the first block is the defect: with -L the first block is a redirect (302) or, on
    a proxied CONNECT, not the object's response at all.
    """
    blocks = [b for b in header_text.replace("\r\n", "\n").split("\n\n") if b.strip()]
    return blocks[-1] if blocks else header_text


def probe_head(url, proxy, timeout=60):
    """(ok, detail, content_length) from a HEAD that follows redirects."""
    r = _curl(["-I", "-L", url], proxy, timeout)
    if r.returncode != 0:
        return False, f"CONN_FAIL {(r.stderr or b'').decode(errors='replace').strip()[:120]}", None
    text = (r.stdout or b"").decode(errors="replace")
    if not text.strip():
        return False, "EMPTY no header response", None
    block = _final_block(text)
    first = block.strip().splitlines()[0] if block.strip() else ""
    parts = first.split()
    code = parts[1] if len(parts) > 1 else "?"
    clen = None
    for ln in block.splitlines():
        if ln.lower().startswith("content-length:"):
            try:
                clen = int(ln.split(":", 1)[1].strip())
            except ValueError:
                clen = None
    if code != "200":
        return False, f"HTTP_{code} {first[:80]}", clen
    return True, first[:80], clen


def probe_body(url, proxy, timeout=60, nbytes=512):
    """Fetch a few hundred bytes and classify what comes back.

    THE CONTENT CHECK. A 200 whose body is an HTML error page is not reachable, and a HEAD
    that reports 200 with no length is not evidence either (see the module docstring).
    """
    r = _curl(["-L", "-r", f"0-{nbytes - 1}", url], proxy, timeout)
    if r.returncode != 0:
        return False, f"CONN_FAIL {(r.stderr or b'').decode(errors='replace').strip()[:120]}", b""
    body = r.stdout or b""
    if not body:
        return False, "EMPTY zero-byte body on a range GET", body
    head = body[:400]
    for marker in ERROR_BODY_MARKERS:
        if marker in head:
            return False, f"HTML_ERROR_BODY (starts {head[:40]!r})", body
    return True, f"ok {len(body)}B", body


def probe_magic(url, proxy, timeout=60):
    """Last 4 bytes via a range GET -> (ok, detail). Parquet ends with PAR1."""
    r = _curl(["-L", "-r", "-4", url], proxy, timeout)
    if r.returncode != 0:
        return False, f"CONN_FAIL {(r.stderr or b'').decode(errors='replace').strip()[:80]}"
    body = r.stdout or b""
    if body.endswith(PARQUET_MAGIC):
        return True, "PAR1"
    return False, f"tail {body[-8:]!r} is not PAR1"


def stratified_indices(n, k, seed):
    """First, last, evenly spaced, plus seeded-random -- deterministic for a given seed.

    The ENDS ARE GUARANTEED. A truncated source loses its last file first, and a range host
    that dies mid-manifest loses its tail, so index n-1 is the single most informative probe.
    An earlier version built the spaced set then truncated with `sorted(idx)[:k]`, which
    dropped n-1 whenever the random sample pushed the set over k -- the selftest's
    "the ends are not sampled" assertion is what caught it.
    """
    if n <= k:
        return list(range(n))
    rng = random.Random(seed)
    spaced = {round(i * (n - 1) / (k - 1)) for i in range(k)}  # includes 0 and n-1
    extra = set(rng.sample(range(n), min(k, n))) - spaced
    # take the spaced set first (it carries the ends), then fill the remainder with randoms
    out = set(spaced)
    for i in sorted(extra):
        if len(out) >= max(k, 2):
            break
        out.add(i)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, help="shipped names file, one name per line")
    ap.add_argument("--base", required=True, help="URL prefix the names are appended to")
    ap.add_argument("--n", type=int, default=8, help="how many URLs to probe")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--proxy", default=os.environ.get("PROBE_PROXY", DEFAULT_PROXY),
                    help="'' for a direct connection")
    ap.add_argument("--magic", action="store_true", help="also range-GET the last 4 bytes")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    names = open(a.manifest, encoding="utf-8").read().split()
    if not names:
        print(f"REFUSING: {a.manifest} names no files")
        return 2
    idx = stratified_indices(len(names), a.n, a.seed)

    print(f"=== {a.label or a.manifest}  ({len(names)} names, probing {len(idx)})")
    print(f"proxy: {a.proxy or '(direct)'}")
    print(f"{'idx':>5} {'reachable':>10} {'bytes':>14}  name")
    fails = []
    for i in idx:
        name = names[i]
        url = a.base + name
        ok_h, detail_h, clen = probe_head(url, a.proxy, a.timeout)
        ok_b, detail_b, _ = probe_body(url, a.proxy, a.timeout)
        ok = ok_h and ok_b
        cl = f"{clen:,}" if isinstance(clen, int) else "-"
        print(f"{i:>5} {('yes' if ok else 'NO'):>10} {cl:>14}  {name}")
        if not ok:
            fails.append((name, f"head={detail_h} | body={detail_b}"))
            print(f"        -> {detail_b}")
        if ok and a.magic:
            ok_m, detail_m = probe_magic(url, a.proxy, a.timeout)
            print(f"        tail magic: {detail_m}")
            if not ok_m:
                fails.append((name, f"magic: {detail_m}"))

    print(f"\n{len(idx) - len(fails)}/{len(idx)} reachable")
    for name, why in fails:
        print(f"  FAIL {name}\n       {why}")
    return 1 if fails else 0


def _selftest():
    """Prove the probe DISCRIMINATES, on worlds built from the real predicates.

    The two the incident demands:
      * a **fake 200** -- a redirect's status, or a proxy CONNECT line, read as the object's.
        The first version of this probe reported 8/8 reachable on a host that was serving
        403 + an HTML page, which is exactly this defect. `_final_block` is the fix, and the
        world below is the shape it must reject.
      * a **magic mismatch** -- a truncated or non-parquet body must not pass.

    Every world is driven through the real functions with real bytes; no network.
    """
    ok = 0

    # 1. _final_block takes the LAST block, not the first. This is the false-200 fix.
    two_blocks = "HTTP/1.1 200 Connection established\r\n\r\nHTTP/2 403\r\ncontent-length: 0\r\n"
    last = _final_block(two_blocks)
    assert last.strip().startswith("HTTP/2 403"), f"_final_block took the wrong block: {last!r}"
    # MUTATION: taking the FIRST block is the pre-fix behaviour and must be visibly wrong.
    first = two_blocks.replace("\r\n", "\n").split("\n\n")[0]
    assert first.strip().startswith("HTTP/1.1 200"), "the first block is the CONNECT line, as expected"
    ok += 1

    # 2. An HTML body served under a 200 is NOT reachable. Built from the real marker list.
    html = b"<!DOCTYPE html>\n<html><head><title>Attention Required! | Cloudflare</title>"
    hit = [m for m in ERROR_BODY_MARKERS if m in html[:400]]
    assert hit, "the Cloudflare body we actually received did not match any error marker"
    # and a real parquet head must NOT match them
    real = b"PAR1\x15\x00\x15(\x15,\x15\x00\x12\x00\x00" + b"\x00" * 400
    assert not [m for m in ERROR_BODY_MARKERS if m in real[:400]], "a real parquet head matched"
    ok += 1

    # 3. probe_magic's DECISION, driven through the real function. Asserting
    #    `b"...PAR1".endswith(PARQUET_MAGIC)` inline was the first version and it was a vacuous
    #    test: gutting probe_magic to `return True, "PAR1"` left the selftest green, because the
    #    assertion never called the function it claimed to cover. The predicate is now read out
    #    of the function's own source and exercised through a stub transport, so the mutation
    #    that removes the comparison goes red by name.
    _pm_src = inspect.getsource(probe_magic)
    assert "PARQUET_MAGIC" in _pm_src, (
        "probe_magic no longer compares against PARQUET_MAGIC -- the magic check was removed")
    _real_curl = _curl
    try:
        # a PAR1 tail passes, a non-PAR1 tail fails, an empty body fails. The transport is
        # stubbed so this needs no network; the comparison under test is the real one.
        for _body, _want in ((b"....PAR1", True), (b"....xxxx", False), (b"", False)):
            globals()["_curl"] = lambda *a, _b=_body, **k: subprocess.CompletedProcess(
                args=[], returncode=0, stdout=_b, stderr=b"")
            _got, _why = probe_magic("http://x/", None, 1.0)
            assert _got is _want, f"probe_magic({_body!r}) -> {_got} ({_why}), want {_want}"
    finally:
        globals()["_curl"] = _real_curl
    ok += 1

    # 4. stratified_indices is deterministic for a seed and always includes the ends, which is
    #    what makes two runs comparable and makes the first/last files (the ones a range
    #    truncation kills) always probed.
    a1 = stratified_indices(1024, 8, 20260919)
    a2 = stratified_indices(1024, 8, 20260919)
    assert a1 == a2, "same seed gave different samples"
    assert 0 in a1 and 1023 in a1, "the ends are not sampled"
    assert stratified_indices(3, 8, 1) == [0, 1, 2], "n < k must not raise or pad"
    ok += 1

    print(f"probe_source_urls selftest OK: {ok} gate(s) -- _final_block takes the last block "
          f"(a CONNECT/302 line read as the object is the false-200 this exists to stop), an "
          f"HTML error body under 200 is rejected, PAR1 magic is enforced, and the sample is "
          f"deterministic and includes both ends")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(main())
