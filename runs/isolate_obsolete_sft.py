#!/usr/bin/env python3
"""Reversibly isolate the obsolete D1e recovery chain during 100B pretraining."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/work/aupai")
TARGET_MARKERS = (
    "/work/aupai/runs/ladder/guard/d1e_chain.sh",
    "sft_math.py --resume /work/aupai/runs/recovery_v42",
)


def matching_pids() -> list[int]:
    own = os.getpid()
    found: list[tuple[int, int]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if any(marker in cmd for marker in TARGET_MARKERS):
            try:
                ppid = int((entry / "stat").read_text().split()[3])
            except OSError:
                ppid = 0
            found.append((pid, ppid))
    children = {ppid for _, ppid in found}
    return [pid for pid, _ in found if pid not in children] + [pid for pid, _ in found if pid in children]


def stop_targets() -> None:
    pids = matching_pids()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(3 if sig == signal.SIGTERM else 1)


def preserve_and_block(path: Path, note: str) -> None:
    disabled = path.with_name(path.name + ".disabled-for-100b")
    if path.is_dir():
        (path / "README").write_text(note + "\n", encoding="utf-8")
        return
    if path.exists():
        if disabled.exists() and path.read_bytes() == disabled.read_bytes():
            path.unlink()
        elif disabled.exists():
            stamped = path.with_name(path.name + f".disabled-for-100b.{int(time.time())}")
            shutil.move(path, stamped)
        else:
            shutil.move(path, disabled)
    path.mkdir(parents=False, exist_ok=True)
    (path / "README").write_text(note + "\n", encoding="utf-8")


os.chdir(ROOT)
stop_targets()
for name in ("recovery_d1e", "s3a_seg1"):
    subprocess.run(
        ["python3", "scripts/card_claim.py", "release", "--name", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

preserve_and_block(
    ROOT / "runs/ladder/guard/d1e_chain.sh",
    "Disabled during the 100B Engram run. The preserved script is ../d1e_chain.sh.disabled-for-100b.",
)
preserve_and_block(
    ROOT / "sft_math.py",
    "Disabled during the 100B Engram run. The preserved program is ../sft_math.py.disabled-for-100b.",
)

print("obsolete SFT isolated")
for path in (ROOT / "runs/ladder/guard/d1e_chain.sh", ROOT / "sft_math.py"):
    print(path, "directory" if path.is_dir() else "NOT_BLOCKED")
