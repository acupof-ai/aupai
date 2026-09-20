#!/usr/bin/env python3
# restartable: one file per shard, a valid footer skips it, and a partial resumes with -C -.
# An interrupt costs at most the shard in flight (~1.08 GB, ~13 min at the measured rate).
"""Fetch openbmb/UltraData-Code python shards, resume-capable, with host failover.

    python datagen/fetch_ultradata.py --level L3 --first 1 --last 3

Shards land in data/raw/ultradata/. A shard with a valid parquet footer is
skipped; a partial file is resumed in place with curl -C -.

HOST FAILOVER. Until 2026-09-20 this script named hf-mirror in a module constant, so a day
like 2026-08-31 -- when hf-mirror answered rc=28 for the rest of the day while modelscope
served in 0.08 s -- stopped the fetch outright. The order is hf-mirror, modelscope,
huggingface.co; each candidate is probed for a FINAL 200 before use, and the host that
served a shard is recorded in fetch_stats.json.

huggingface.co is listed and probed but unreachable from the digest box (curl rc=000,
2026-09-20); it is kept in the list because the pod's reachability changes without notice
and a missing entry cannot fail over at all.

Permission to fetch is the caller's: this script downloads when run. Nothing here is a
guard against that.
"""

import argparse
import json
import os
import subprocess
import time

N_SHARDS = {"L2": 119, "L3": 147}

# Total wall-clock bound per shard attempt. A shard is ~1.08 GB and the measured sustained
# rate is 1.33 MB/s (8 MB continuous range, 2026-09-20), so the mean is ~13 min; 3600 s is
# ~4.6x that, which leaves room for a slow host without letting a stalled socket hold the
# serial loop forever. Re-measure before trusting it after a host or path change.
DEFAULT_MAX_TIME = 3600

#: The flags every transfer shares. ONE LINE, <= 88 chars, on purpose -- see _transfer_args.
_CURL_BASE = ("curl", "-4", "-fSL", "-C", "-", "--retry", "3", "--connect-timeout", "15")

#: The repo's standing egress relay, the same value scripts/probe_source_urls.py:35 carries.
#: Measured 2026-09-20 on digest: huggingface.co answers 000 without it and 200 through it,
#: while hf-mirror and modelscope answer 200 either way -- so the proxy adds exactly ONE arm,
#: and an unset variable that silently disabled it would remove that arm invisibly.
DEFAULT_PROXY = "http://sys-proxy-rd-relay.byted.org:8118"

#: The last probe result per host label, filled by resolve_hosts so a caller prints what was
#: OBSERVED without probing a second time. The pre-fix FAIL detail re-called final_status per
#: host (:148), which cost three extra real curls for every failed shard AND went around the
#: `probe=` seam, so the one line this module exists for could not be exercised offline.
_PROBED = {}


def _proxy_args():
    """["--proxy", url], or [] when AUPAI_FETCH_PROXY is set to the EMPTY string.

    Both spellings are needed and they are not the same setting: unset means "use the repo
    default", `""` means "go direct". curl errors on `--proxy ""`, so the direct case must
    DROP the flag. `_selftest` asserts both, because the two produce one reading otherwise --
    "the proxy is always on" and "hf-mirror served without it" are different measurements.
    """
    p = os.environ.get("AUPAI_FETCH_PROXY", DEFAULT_PROXY)
    return ["--proxy", p] if p else []


# (label, base template). {dataset} and {path} are filled per shard. modelscope's FilePath
# takes the FULL repo-relative path including the "data/" prefix -- a request without it
# 404s, which reads exactly like "this mirror does not have the dataset" (fb, 2026-09-20).
HOSTS = (
    ("hf-mirror.com", "https://hf-mirror.com/datasets/{dataset}/resolve/main/{path}"),
    (
        "www.modelscope.cn",
        "https://www.modelscope.cn/api/v1/datasets/{dataset}/repo?Revision=master&FilePath={path}",
    ),
    ("huggingface.co", "https://huggingface.co/datasets/{dataset}/resolve/main/{path}"),
)


def final_status(url, timeout=10):
    """(final status, curl rc) for one probe, the status None when curl could not run.

    THE rc IS RETURNED, NOT COLLAPSED INTO THE STATUS. Reporting only the status loses the
    discrimination this probe exists for: measured with a subprocess stub, 302/rc=28 (a
    followed redirect whose CDN then timed out) and 000/rc=7 (a connect refusal) both
    render as `None`, so a FAIL line built from statuses alone says the same thing for "the
    source is gone" and "this box cannot get out". Callers print both fields.

    -L matters: hf-mirror and modelscope both 302 to a CDN, so probing the first response
    line calls a serving mirror down. Measured 2026-09-04 in fetch_corpus._ot3_probe_ok:
    hf-mirror 302->AWS CDN and modelscope 302->cdn-lfs-cn, both final 200, and the pre-fix
    -sI form refused a fetchable source.

    -4 because the pod's IPv6 egress is broken: curl tries IPv6 first and the failure
    surfaces as Errno 99, which reads as "the host is unreachable" and is the local
    address family being unusable.

    A HEAD PROBE CANNOT RETURN 206. 206 is a Range-GET-only status, so the success test is
    `== 200` and must not be widened to {200, 206} here; the ranged-GET gate is a different
    instrument. There is no fail-open either way: a non-zero rc is never accepted, so
    302/rc=28 cannot be mistaken for a serving host.
    """
    argv = ["curl", "-4", "-sIL", "-o", "/dev/null", "-w", "%{http_code}", "-m", str(timeout)]
    argv += _proxy_args()
    try:
        p = subprocess.run(argv + [url], capture_output=True, text=True, timeout=timeout + 5)
    except (OSError, subprocess.SubprocessError):
        return None, None
    out = p.stdout.strip()
    code = int(out) if out.isdigit() else None
    return code, p.returncode


def resolve_hosts(dataset, path, probe=final_status):
    """[(url, label, code, rc)] for every host that answers a final 200, in HOSTS order.

    THE PROBE RUNS ONCE PER HOST AND ITS RESULT IS RETURNED. The detail line a caller
    prints on failure used to re-probe by calling final_status again (:148, pre-fix), which
    cost three extra real curls per failed shard AND went around the `probe=` seam, so the
    exact line this module exists for could not be unit-tested without a network.

    `probe` is injectable so the selftest can drive the ordering and the failure modes
    without a network. An empty list means no host answered -- the caller decides whether
    that is fatal (a shard) or skippable (a listing). Either way the caller gets every
    host's (code, rc) to print, including the hosts that did not answer 200.
    """
    out = []
    for label, tmpl in HOSTS:
        url = tmpl.format(dataset=dataset, path=path)
        code, rc = probe(url)
        entry = (url, label, code, rc)
        if code == 200:
            out.append(entry)
        _PROBED[label] = entry
    return out


def _probe_detail():
    """Every host's (status, rc) as one line, from the probes resolve_hosts already ran.

    A function rather than an inline join because this string IS the failure report -- the
    pre-fix version built it by probing a second time (:148) and it is the one path the
    `probe=` seam could not reach, so it shipped untested.
    """
    return ", ".join(f"{lbl}: {_PROBED[lbl][2]}/rc={_PROBED[lbl][3]}" for lbl, _ in HOSTS)


def footer_ok(path):
    if not os.path.exists(path) or os.path.getsize(path) < 8:
        return False
    with open(path, "rb") as fh:
        fh.seek(-4, 2)
        return fh.read(4) == b"PAR1"


def _transfer_args(url, out, max_time):
    """The curl argv for one shard attempt, with BOTH timeouts.

    `--connect-timeout` bounds only the handshake, so a connection that completes and then
    stalls sends no data and no error: measured 2026-09-20 on one 8 MB range, the same URL
    returned a 206 in 9.74 s once and nothing at all for >180 s the next time. The socket
    stays open, the shard never finishes, and -- because this loop is serial -- the whole
    fetch holds that slot indefinitely. `--max-time` is the total wall-clock bound and is
    what turns a silent hang into a non-zero rc the caller can retry.

    `--retry` re-runs the transfer, not the whole file: with `-C -` it resumes from whatever
    landed, so a retried shard keeps its bytes.

    The fixed prefix is a separate constant because check_curl_ipv4 reads this file LINE BY
    LINE: the line holding `curl` must also hold `-4`, and ruff explodes any multi-line
    collection one token per line, which separates them and turns the check red naming a
    curl call that does pass -4 (measured 2026-09-20). Kept to 76 chars so ruff leaves it
    on one line. Do not wrap it.

    THE TRANSFER CARRIES THE SAME EGRESS SETTING AS THE PROBE. A probe that resolves a host
    through the proxy and a transfer that then goes direct measures two different networks:
    the host list would be built for an egress the shard never uses.
    """
    return [*_CURL_BASE, *_proxy_args(), "--max-time", str(max_time), "-o", out, url]


def fetch(level, first, last, dest, max_time=DEFAULT_MAX_TIME):
    n = N_SHARDS[level]
    os.makedirs(dest, exist_ok=True)
    failed = 0
    served = {}
    for i in range(first, last + 1):
        name = f"UltraData-Code-{level}-py-part-{i:05d}-of-{n:05d}.parquet"
        out = os.path.join(dest, name)
        if footer_ok(out):
            print(f"SKIP {name} ({os.path.getsize(out)} bytes)", flush=True)
            continue
        rel = f"data/UltraData-Code-{level}/py/{name}"
        candidates = resolve_hosts("openbmb/UltraData-Code", rel)
        if not candidates:
            # SAY WHAT WAS OBSERVED, per host and per egress setting. "no host answered" alone is
            # the same line whether the source is gone or the box cannot reach the internet, and
            # those two send a reader to different places. Measured 2026-09-20 on digest:
            # huggingface.co is 000 without the proxy and 200 with it, while hf-mirror and
            # modelscope answer 200 either way -- so a missing proxy silently removes exactly one
            # arm, which is invisible until it is the arm that would have served.
            #
            # THE DETAIL IS BUILT FROM THE PROBE THAT ALREADY RAN (`_PROBED`), not by probing
            # again: the pre-fix line re-called final_status here, three extra real curls per
            # failed shard through the one code path the `probe=` seam cannot reach. Both fields
            # are printed -- status alone renders a followed-redirect timeout (302/rc=28) and a
            # connect refusal (000/rc=7) identically.
            detail = _probe_detail()
            print(
                f"FAIL {name}: no host answered a final 200 [{detail}]"
                f" proxy={os.environ.get('AUPAI_FETCH_PROXY', DEFAULT_PROXY) or '<direct>'}",
                flush=True,
            )
            failed += 1
            continue
        t0 = time.time()
        host = None
        for url, label, _code, _rc in candidates:
            r = subprocess.run(_transfer_args(url, out, max_time))
            if r.returncode == 0 and footer_ok(out):
                host = label
                break
            print(
                f"  {name}: {label} served rc={r.returncode} footer={footer_ok(out)} -- trying the next host",
                flush=True,
            )
        if host is None:
            print(f"FAIL {name}: every candidate host failed the transfer", flush=True)
            failed += 1
            continue
        served[name] = host
        print(f"OK {name} ({os.path.getsize(out)} bytes, {time.time() - t0:.0f}s, {host})", flush=True)
    if served:
        sp = os.path.join(dest, "fetch_stats.json")
        prior = {}
        if os.path.exists(sp):
            try:
                with open(sp, encoding="utf-8") as fh:
                    prior = json.load(fh).get("served_by_host", {})
            except (OSError, ValueError):
                prior = {}
        prior.update(served)
        with open(sp, "w", encoding="utf-8") as fh:
            json.dump({"served_by_host": prior}, fh, indent=1, sort_keys=True)
        print(f"stats {sp}", flush=True)
    return failed


def _selftest():
    """The ordering and the failure modes, driven through an injected probe. No network.

    A HOST IS USABLE ONLY IF ITS *FINAL* STATUS IS 200. Three cases decide this function and
    each is a real shape: a host that answers 200 first, a host whose first line is a 302
    that a status-line probe would call down, and a host that answers nothing at all
    (hf.co measures 000 from digest).
    """
    seen = []

    def probe(statuses):
        """A stand-in for final_status: {(host, (code, rc))} -> the tuple, (None, None) if absent.

        It returns the SAME SHAPE the real probe does, so every caller line under test here --
        including the FAIL detail -- runs exactly as it does against a network. A stub that
        returned a bare status would let the returned-rc path go untested.
        """

        def _p(url):
            seen.append(url)
            for host, st in statuses.items():
                if host in url:
                    return st
            return None, None

        return _p

    p = "data/UltraData-Code-L2/py/x.parquet"
    # the normal case: the first host is up, and it is used
    got = resolve_hosts(
        "openbmb/UltraData-Code", p, probe({"hf-mirror.com": (200, 0), "www.modelscope.cn": (200, 0)})
    )
    assert got and got[0][1] == "hf-mirror.com", got
    assert got[0][2:] == (200, 0), "the probe's (code, rc) must reach the caller"

    # THE FAILOVER: hf-mirror down (rc=28 on 2026-08-31), modelscope serves
    got = resolve_hosts(
        "openbmb/UltraData-Code", p, probe({"hf-mirror.com": (302, 28), "www.modelscope.cn": (200, 0)})
    )
    assert [e[1] for e in got] == ["www.modelscope.cn"], got

    # THE DETAIL LINE'S DISCRIMINATION, which is why rc is returned at all: a followed
    # redirect whose CDN then timed out and a connect refusal are the SAME status and
    # DIFFERENT rc. Both hosts are dropped (neither is a 200), and the two entries the
    # caller prints must stay distinguishable.
    resolve_hosts(
        "openbmb/UltraData-Code", p, probe({"hf-mirror.com": (302, 28), "huggingface.co": (None, 7)})
    )
    assert _PROBED["hf-mirror.com"][2:] == (302, 28), _PROBED
    assert _PROBED["huggingface.co"][2:] == (None, 7), _PROBED
    assert _PROBED["hf-mirror.com"][2:] != _PROBED["huggingface.co"][2:], (
        "collapsing rc into the status is the defect this pair exists to catch"
    )
    # every host is recorded, including the ones that answered nothing: the FAIL detail
    # reads this dict and a missing key would raise inside the failure path itself
    assert set(_PROBED) == {lbl for lbl, _ in HOSTS}, _PROBED
    # THE DETAIL MAKES NO NETWORK CALL. This is the whole rework: the pre-fix line called
    # final_status here, so every failed shard cost three real curls and the report could not
    # be rendered offline. subprocess is disabled outright and this runs FIRST, so the
    # mutation that reintroduces the re-probe fails naming this property rather than
    # whatever the live hosts happened to answer.
    real_run = subprocess.run
    subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"re-probed: {a}"))
    try:
        line = _probe_detail()
    finally:
        subprocess.run = real_run
    # the exact line the caller prints: rc survives into it, so the two
    # indistinguishable-status cases stay distinguishable in the output a reader sees
    assert "hf-mirror.com: 302/rc=28" in line and "huggingface.co: None/rc=7" in line, line
    assert len(line.split(", ")) == len(HOSTS), line
    # ...and resolve_hosts does not probe a second time anywhere: three hosts, three calls
    before = len(seen)
    resolve_hosts("openbmb/UltraData-Code", p, probe({"hf-mirror.com": (200, 0)}))
    assert len(seen) - before == len(HOSTS), f"{len(seen) - before} probes for {len(HOSTS)} hosts"

    # a 302 IS NOT DOWN: the CDN redirect is the working path, and a probe that reads the
    # first status line would drop this host
    assert resolve_hosts("openbmb/UltraData-Code", p, probe({"hf-mirror.com": (302, 0)})) == [], (
        "302 must not count as 200"
    )

    # nothing answers: an empty list, not an exception -- callers distinguish fatal from
    # skippable
    assert resolve_hosts("openbmb/UltraData-Code", p, probe({})) == []

    # THE EGRESS SETTING IS TWO DISTINCT STATES, and the direct one must DROP the flag
    # (curl errors on `--proxy ""`). Measured 2026-09-20: huggingface.co serves 200 through the
    # relay and nothing without it, so "unset" and "empty" decide which arms exist -- a single
    # reading here would merge "the proxy is always on" with "hf-mirror served without it".
    prior = os.environ.get("AUPAI_FETCH_PROXY")
    try:
        os.environ.pop("AUPAI_FETCH_PROXY", None)
        assert _proxy_args() == ["--proxy", DEFAULT_PROXY], _proxy_args()
        assert "--proxy" in _transfer_args("http://x/y", "/tmp/y.part", 60), "the transfer must use it too"
        os.environ["AUPAI_FETCH_PROXY"] = ""
        assert _proxy_args() == [], "an empty setting means DIRECT: curl rejects --proxy ''"
        assert "--proxy" not in _transfer_args("http://x/y", "/tmp/y.part", 60)
        os.environ["AUPAI_FETCH_PROXY"] = "http://relay.local:3128"
        assert _proxy_args() == ["--proxy", "http://relay.local:3128"], _proxy_args()
    finally:
        if prior is None:
            os.environ.pop("AUPAI_FETCH_PROXY", None)
        else:
            os.environ["AUPAI_FETCH_PROXY"] = prior

    # the modelscope entry carries the FULL path including data/: the URL built for it must
    # contain FilePath=data/... , because a request without the prefix 404s and reads as
    # "this mirror lacks the dataset"
    ms = [u for l, u in ((l, t.format(dataset="d", path=p)) for l, t in HOSTS) if l == "www.modelscope.cn"][0]
    assert f"FilePath={p}" in ms and p.startswith("data/"), ms

    # the order is the documented one, so a failover is predictable from the code
    assert [l for l, _ in HOSTS] == ["hf-mirror.com", "www.modelscope.cn", "huggingface.co"], HOSTS

    # footer_ok is the acceptance test the retry loop uses: a short or truncated file is not
    # a shard, and PAR1 must be at the END, not the start
    import tempfile

    d = tempfile.mkdtemp(prefix="ud_selftest_")
    good = os.path.join(d, "good.parquet")
    with open(good, "wb") as fh:
        fh.write(b"PAR1" + b"\0" * 100 + b"PAR1")
    assert footer_ok(good), "a real footer must pass"
    head_only = os.path.join(d, "head.parquet")
    with open(head_only, "wb") as fh:
        fh.write(b"PAR1" + b"\0" * 100 + b"XXXX")
    assert not footer_ok(head_only), "PAR1 at the START is not a footer"
    assert not footer_ok(os.path.join(d, "absent.parquet")), "a missing file is not a shard"
    tiny = os.path.join(d, "tiny.parquet")
    with open(tiny, "wb") as fh:
        fh.write(b"PAR1")
    assert not footer_ok(tiny), "a 4-byte file cannot hold both magics"

    # A STALLED SOCKET MUST BOUND. --connect-timeout bounds the handshake only; the measured
    # failure is a connection that completes and then sends nothing (>180 s, no error), which
    # a connect-only timeout cannot end. Both flags must be present or that shard holds the
    # serial loop forever.
    argv = _transfer_args("http://x/y.parquet", "/tmp/y.part", 3600)
    assert "--connect-timeout" in argv and "--max-time" in argv, argv
    assert argv[argv.index("--max-time") + 1] == "3600", "max-time must take the bound"
    assert "-C" in argv, "resume must survive the retry"
    assert "--retry" in argv, "a timeout must be retried, not fatal"

    # the stall is modelled end to end: a server that accepts and never writes, bounded by
    # --max-time, must end with a non-zero rc rather than hanging.
    #
    # ELAPSED TIME IS THE ASSERTION, not rc alone. The first version had the server sleep 30 s
    # and asserted only rc != 0 -- which a mutated argv (`--max-time` present but not honoured,
    # curl reading the LAST occurrence) PASSED, because the server closing at 30 s truncates the
    # body and gives a non-zero rc for the wrong reason. rc!=0 cannot tell "the bound fired" from
    # "the server hung up". So the server now stalls far past the bound and the test asserts the
    # call returned well before the server would have ended: discriminates the bound firing.
    import http.server
    import socketserver
    import threading

    STALL_S = 60

    class _Stall(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 -- http.server's spelling
            self.send_response(200)
            self.send_header("Content-Length", "1000000")
            self.end_headers()
            time.sleep(STALL_S)  # accept, then write nothing

        def log_message(self, *a):
            pass

    # A THREADING server, with daemon threads and no close-join: the handler sleeps STALL_S, and
    # a plain TCPServer's shutdown() will not return until that handler does -- which made this
    # selftest cost the full 60 s even though curl itself returned at the 2 s bound.
    class _Srv(socketserver.ThreadingTCPServer):
        daemon_threads = True
        block_on_close = False

    srv = _Srv(("127.0.0.1", 0), _Stall)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        stall_out = os.path.join(d, "stall.parquet")
        argv = _transfer_args(f"http://127.0.0.1:{srv.server_address[1]}/s.parquet", stall_out, 2)
        # --retry is asserted on the argv above; here it would only multiply the wall clock by
        # curl's exponential backoff (4 attempts, 1+2+4 s of delay, ~15 s measured) to prove a
        # property -- that --max-time bounds one attempt -- retries do not affect.
        argv[argv.index("--retry") + 1] = "0"
        t0 = time.time()
        r = subprocess.run(argv, capture_output=True, text=True)
        elapsed = time.time() - t0
        assert r.returncode != 0, (
            "a server that accepts and never writes must end with a non-zero rc under "
            "--max-time; rc=0 here means the bound did not apply"
        )
        assert elapsed < STALL_S / 2, (
            f"the call took {elapsed:.1f}s against a {STALL_S}s stall: it ended because the "
            f"SERVER closed, not because --max-time=2 fired -- the bound is not in the argv "
            f"the transfer actually runs"
        )
        assert not footer_ok(stall_out), "a stalled transfer must not leave a valid shard"
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=1)

    print(
        "fetch_ultradata selftest OK: host order [hf-mirror, modelscope, hf.co]; a final "
        "200 is required, a 302 is not a 200, no-host yields an empty list, the modelscope "
        "entry carries FilePath=data/...; footer_ok reads the END magic and rejects a "
        "head-only, a missing and a 4-byte file; a stalled socket is bounded by --max-time "
        "and leaves no shard"
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["L2", "L3"])
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--dest", default="data/raw/ultradata")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.level:
        ap.error("--level is required unless --selftest")
    return 1 if fetch(a.level, a.first, a.last, a.dest) else 0


if __name__ == "__main__":
    raise SystemExit(main())
