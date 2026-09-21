#!/usr/bin/env python3
"""Resolve the pod container's `/work` HOST path from a `crictl inspect` JSON on stdin.

`pod_push.sh` ships large files through the HOST (`tn exec`/`tn write`), so it needs the
host path that backs the container's `/work`. The old resolver globbed
`/var/lib/kubelet/pods/*/volumes/kubernetes.io~empty-dir/work` -- correct while /work was an
emptyDir, wrong after it became a hostPath/bind mount. Measured 2026-09-21: the live
sglang-test container mounts `/work` from host `/data00/aupai_work`, and the emptyDir glob
now matches directories that hold no `aupai/`, so the resolver came back empty and
`pod_push.sh` failed loud before sending anything (small files slipped through podput,
large files never shipped). The mount source is read from the container's own runtime
spec, which is true for emptyDir, hostPath and bind alike.

Usage:
  cid=$(crictl ps -q --name sglang-test --state Running | head -1)
  crictl inspect "$cid" | python3 scripts/pod_work_hostpath.py
  # prints the host source of /work, e.g. /data00/aupai_work ; empty stdout if absent

The parser reads stdin only, so pod_push runs it on the LAPTOP (where this file is always
the merged version) and pipes the host's crictl output in -- the host needs no copy of
this script. An empty answer is the caller's signal to fall back / fail, never a path.
"""

import json
import sys

DEST = "/work"


def work_host_source(inspect_text):
    """The host source mounted at the container's /work, or '' if not found.

    Both `info.runtimeSpec.mounts` and `status.runtimeSpec.mounts` are tried: crictl has
    moved the spec between the two across versions, and an exact destination match is
    required -- a `/work-other` mount must not satisfy a `/work` lookup.
    """
    d = json.loads(inspect_text)
    for root in ("info", "status"):
        spec = (d.get(root) or {}).get("runtimeSpec") or {}
        for m in spec.get("mounts") or []:
            if m.get("destination") == DEST and m.get("source"):
                return m["source"]
    return ""


def _selftest():
    # Positive: the current live shape -- a bind/hostPath source for /work, sitting next to
    # an unrelated mount and a same-prefix decoy that a startswith would wrongly accept.
    inspect = {
        "info": {"runtimeSpec": {"mounts": [
            {"destination": "/proc", "source": "proc"},
            {"destination": "/data00", "source": "/data00"},
            {"destination": "/work", "source": "/data00/aupai_work"},
            {"destination": "/workspace", "source": "/decoy/prefix"},
        ]}},
        "status": {},
    }
    got = work_host_source(json.dumps(inspect))
    assert got == "/data00/aupai_work", f"exact /work source, got {got!r}"

    # Status-only location (other crictl version) must resolve too.
    moved = {"info": {}, "status": {"runtimeSpec": {"mounts": [
        {"destination": "/work", "source": "/var/lib/kubelet/pods/x/volumes/empty-dir/work"}]}}}
    assert work_host_source(json.dumps(moved)) == \
        "/var/lib/kubelet/pods/x/volumes/empty-dir/work"

    # No /work mount -> EMPTY, never a guess, never the decoy.
    none = {"info": {"runtimeSpec": {"mounts": [
        {"destination": "/workspace", "source": "/decoy"}]}}, "status": {}}
    assert work_host_source(json.dumps(none)) == "", "missing /work must answer empty"
    assert work_host_source(json.dumps({"info": {}, "status": {}})) == ""
    assert work_host_source(json.dumps({"info": {"runtimeSpec": {}}, "status": {}})) == ""

    # An empty-string source is treated as absent (a half-described mount is not a path).
    blanksrc = {"info": {"runtimeSpec": {"mounts": [
        {"destination": "/work", "source": ""}]}}, "status": {}}
    assert work_host_source(json.dumps(blanksrc)) == ""
    print("pod_work_hostpath selftest OK: /work host source resolved (info/status), "
          "exact destination match, missing/blank -> empty")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        sys.stdout.write(work_host_source(sys.stdin.read()))
