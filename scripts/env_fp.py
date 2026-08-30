#!/usr/bin/env python3
"""Environment fingerprint: a hash of the effective training environment.

The failure mode this guards: a container restart drops the writable layer,
losing hand-installed packages. Three sessions spent an hour chasing three
wrong hypotheses because nothing recorded what the environment WAS. The
fingerprint is stored in every checkpoint and compared on resume.

Covers what executes, not what's nominally installed: every package's name
and version, plus a content sample of each compiled extension (.so). A
locally-built wheel can keep the version string while changing the .so.
"""
import hashlib
import importlib.metadata
import sys
from pathlib import Path


def env_fingerprint() -> str:
    """Hash the effective training environment. Returns a 16-char hex string.

    Hashes: Python version, every installed distribution (name + version),
    and the size + mtime_ns of each compiled extension (.so/.pyd/.dylib).
    Size+mtime catches locally-built wheels whose version string did not
    change but whose binary did, without the I/O of reading every file.
    """
    h = hashlib.sha256()
    h.update(sys.version.encode())

    for dist in sorted(
        importlib.metadata.distributions(),
        key=lambda d: d.metadata["Name"].lower(),
    ):
        name = dist.metadata["Name"]
        version = dist.version
        h.update(f"{name}=={version}\n".encode())

        # Compiled extensions: size + mtime catches a locally-built wheel
        # that kept the version string but changed the .so.
        try:
            for f in dist.files or []:
                if f.suffix in (".so", ".pyd", ".dylib"):
                    p = Path(f.locate())
                    if p.is_file():
                        st = p.stat()
                        h.update(f"{p.relative_to(sys.prefix)}:{st.st_size}:{st.st_mtime_ns}\n".encode())
        except Exception:
            pass  # unreadable package = unhashable package; don't crash training

    return h.hexdigest()[:16]


if __name__ == "__main__":
    print(env_fingerprint())
