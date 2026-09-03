#!/usr/bin/env python
"""Did the files pod_push just pushed actually land, byte for byte?

pod_push's only post-push evidence was `pod_drift.py --check`, which compares the pod
against data/pod_head_manifest.txt. That is silent about every path the manifest does not
list, and the manifest lists 398 of 815 tracked files: docs/lessons/, docs/audits/,
scripts/pod_sync_check.sh (explicitly out of SCOPE), and anything else outside SCOPE can
be pushed by name with nothing afterwards checking the bytes. This morning
pod_sync_check reported `4 UNREGISTERED .py not in manifest`, which is the same gap seen
from the other side.

So the push verifies what it pushed, per path, against the pod's own sha256sum, instead
of asking a manifest gate about a different set of files.

Usage:
  ~/bin/pod "cd /work/aupai && sha256sum <paths> 2>/dev/null" > pod.txt
  python3 scripts/pod_verify_landed.py pod.txt <path>...

Exits 0 when every path matches, 1 naming each that did not -- absent and differing
reported separately, because "the transport wrote nothing" and "the transport wrote the
wrong bytes" have different causes.

The selftest's negative worlds differ from the positive one by ONE byte and by nothing
but the sha, per shapes 146: a check verified only on far-apart worlds cannot tell a
correct comparison from a coarse one (a size compare, or a prefix compare, passes any
pair that differs enough).
"""

import hashlib
import os
import sys


def read_pod_shas(path):
    """{path: sha} from `sha256sum` output. Missing files are absent from it -- sha256sum
    writes their error to stderr, which the caller drops."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[1].strip()] = parts[0]
    return out


def sha_local(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify(pod_shas, paths, root="."):
    """[(path, 'absent'|'differs'|'no-local', local_sha_or_None, pod_sha_or_None)] for
    every path that did not land. Empty means all matched.

    A path missing LOCALLY is its own class, not a crash. Found by running this against
    the real pod rather than by reading the selftest: a typo'd argument raised
    FileNotFoundError with a traceback, and a refusal that arrives as a traceback is
    indistinguishable from the script being broken."""
    bad = []
    for p in paths:
        fp = os.path.join(root, p)
        if not os.path.isfile(fp):
            bad.append((p, "no-local", None, pod_shas.get(p)))
            continue
        want = sha_local(fp)
        got = pod_shas.get(p)
        if got is None:
            bad.append((p, "absent", want, None))
        elif got != want:
            bad.append((p, "differs", want, got))
    return bad


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "--selftest":
        selftest()
        return
    pod_shas = read_pod_shas(sys.argv[1])
    paths = sys.argv[2:]
    if not paths:
        print("verify_landed: no paths given -- nothing was checked", file=sys.stderr)
        sys.exit(2)
    bad = verify(pod_shas, paths)
    if not bad:
        print(f"verified on the pod: {len(paths)} file(s) sha256 match")
        return
    print(f"DID NOT LAND: {len(bad)} of {len(paths)} file(s)", file=sys.stderr)
    for p, why, want, got in bad:
        if why == "absent":
            print(f"  absent  {p} (local {want[:12]}, pod has no such file)", file=sys.stderr)
        elif why == "no-local":
            print(
                f"  no-local {p} (not a file here -- wrong path, or it was deleted after the push)",
                file=sys.stderr,
            )
        else:
            print(f"  differs {p} (local {want[:12]}, pod {got[:12]})", file=sys.stderr)
    sys.exit(1)


def selftest():
    import tempfile

    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    body = b"#!/bin/sh\necho landed\n"
    for name in ("scripts/a.sh", "scripts/b.sh", "scripts/c.sh"):
        with open(os.path.join(d, name), "wb") as f:
            f.write(body)
    good = hashlib.sha256(body).hexdigest()
    # ONE byte different, not a different file: the negative world sits immediately
    # beside the positive one.
    near_body = body.replace(b"landed", b"landeD")
    assert len(near_body) == len(body) and near_body != body
    near = hashlib.sha256(near_body).hexdigest()
    assert near != good
    assert len(near) == len(good), "a near miss must be the same shape as a hit"

    pod = os.path.join(d, "pod.txt")
    with open(pod, "w", encoding="utf-8") as f:
        f.write(f"{good}  scripts/a.sh\n{near}  scripts/b.sh\n")  # c.sh absent

    shas = read_pod_shas(pod)
    assert shas == {"scripts/a.sh": good, "scripts/b.sh": near}, shas

    bad = verify(shas, ["scripts/a.sh", "scripts/b.sh", "scripts/c.sh"], root=d)
    assert [(p, why) for p, why, _, _ in bad] == [
        ("scripts/b.sh", "differs"),
        ("scripts/c.sh", "absent"),
    ], bad

    # The positive world alone: every path matches, nothing reported.
    with open(pod, "w", encoding="utf-8") as f:
        f.write("".join(f"{good}  {p}\n" for p in ("scripts/a.sh", "scripts/b.sh", "scripts/c.sh")))
    assert verify(read_pod_shas(pod), ["scripts/a.sh", "scripts/c.sh"], root=d) == []

    # A comparison that only looked at length, or at a prefix, would pass the near miss.
    # Assert the near miss is caught with the same-length, same-alphabet sha it really has.
    assert verify({"scripts/a.sh": near}, ["scripts/a.sh"], root=d)[0][1] == "differs"

    # An EMPTY pod output means the whole batch is absent, never "all good": sha256sum
    # writes missing files to stderr, so a wholly-failed batch produces zero stdout lines
    # and a check that iterates the pod side instead of the requested side would report
    # nothing wrong.
    with open(pod, "w", encoding="utf-8") as f:
        f.write("")
    empty = verify(read_pod_shas(pod), ["scripts/a.sh", "scripts/b.sh"], root=d)
    assert [why for _, why, _, _ in empty] == ["absent", "absent"], empty

    # Trailing whitespace and sha256sum's binary marker ("*path") must not read as a
    # different path. The marker is part of the format, not part of the name.
    with open(pod, "w", encoding="utf-8") as f:
        f.write(f"{good}  scripts/a.sh  \n")
    assert read_pod_shas(pod) == {"scripts/a.sh": good}

    # A path that is not a file HERE is a reported class, not a traceback. Found by
    # running against the real pod: the raise looked like the script being broken.
    nl = verify({}, ["scripts/nope.py"], root=d)
    assert [(p, why) for p, why, _, _ in nl] == [("scripts/nope.py", "no-local")], nl
    # And it must not silently pass when the pod happens to hold it.
    nl2 = verify({"scripts/nope.py": good}, ["scripts/nope.py"], root=d)
    assert [why for _, why, _, _ in nl2] == ["no-local"], nl2

    print("pod_verify_landed selftest OK")


if __name__ == "__main__":
    main()
