#!/usr/bin/env python3
"""Move a host mount into a container's mount namespace: open_tree(CLONE) then move_mount.

WHY THIS AND NOT SOMETHING SIMPLER. The 22 token caches (231 GB) sit on /data00 INSIDE the
container, which is a directory on the overlay (vda2, rotational, 87% full) reading at 207 MB/s --
not the host's /data00, which is nvme0n1. Same name, different filesystem, and that is the whole
confusion: `stat -c %d` gives 173 (overlay root) inside and 259:0 outside.

Two cheaper routes were tried and refuted, in this order:

(a) A SECOND ext4 MOUNT FROM INSIDE, via mknod. All four NVMe are already mounted read-write on
    the host, so this reads a superblock the host is actively writing, and `norecovery` suppresses
    the journal replay that would make it consistent. That is how a 231 GB copy gets torn reads
    that sha256 then faithfully reports as different -- or an ext4 that decides to write. Not
    attempted on purpose.

(b) A HOST-SIDE BIND UNDER THE emptyDir PATH. Measured 2026-09-05: the bind succeeds on the host
    and does NOT appear in the container. `/proc/self/mountinfo` inside shows the /work mount
    (id 2783) with no shared: or master: tag -- it is private, so nothing propagates into it. The
    tell is that the DIRECTORY appears while its CONTENTS do not: the marker file was absent and
    `stat -c %d` still read 65026 (vda2). A mount point that exists and answers with the wrong
    filesystem is the same failure shape as the two filesystem views this repo already fights.

So the mount tree is cloned in the host namespace, where /data02 is visible, and attached inside
the container namespace, where it is not. open_tree(OPEN_TREE_CLONE) returns an fd naming a
detached copy of the tree; setns puts this process in the container's mount namespace; move_mount
attaches the fd there. The fd survives setns because it is an fd, which is the point of the API.

The mount lives as long as the container. A container restart loses it, which is why step 4 of this
task is a bootstrap_pod.sh stage rather than a one-off.
"""

import ctypes
import ctypes.util
import os
import sys

# x86_64 syscall numbers. Both landed in 5.2; the pod runs 5.4.250.
SYS_open_tree = 428
SYS_move_mount = 429

OPEN_TREE_CLONE = 1
AT_RECURSIVE = 0x8000
AT_EMPTY_PATH = 0x1000
AT_FDCWD = -100
MOVE_MOUNT_F_EMPTY_PATH = 0x00000004
CLONE_NEWNS = 0x00020000

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def _err(name):
    e = ctypes.get_errno()
    return OSError(e, f"{name}: {os.strerror(e)}")


def open_tree(path, recursive=True):
    flags = OPEN_TREE_CLONE
    if recursive:
        flags |= AT_RECURSIVE
    fd = _libc.syscall(SYS_open_tree, AT_FDCWD, path.encode(), flags)
    if fd < 0:
        raise _err(f"open_tree({path})")
    return fd


def move_mount(from_fd, to_path):
    r = _libc.syscall(SYS_move_mount, from_fd, b"", AT_FDCWD, to_path.encode(), MOVE_MOUNT_F_EMPTY_PATH)
    if r < 0:
        raise _err(f"move_mount(-> {to_path})")


def setns_mnt(pid):
    fd = os.open(f"/proc/{pid}/ns/mnt", os.O_RDONLY)
    try:
        if _libc.setns(fd, CLONE_NEWNS) != 0:
            raise _err(f"setns(/proc/{pid}/ns/mnt)")
    finally:
        os.close(fd)


def main():
    if len(sys.argv) != 4:
        print(
            f"usage: {sys.argv[0]} <host-source-dir> <container-pid> <target-path-in-container>",
            file=sys.stderr,
        )
        print("  the target must ALREADY EXIST inside the container", file=sys.stderr)
        return 2
    src, pid, target = sys.argv[1], sys.argv[2], sys.argv[3]

    # THE CLONE HAPPENS FIRST, in the host namespace, because after setns the source is gone.
    # Ordering is the only subtle thing in this file.
    if not os.path.isdir(src):
        print(f"REFUSING: {src} is not a directory in this (host) namespace", file=sys.stderr)
        return 1
    st = os.stat(src)
    fd = open_tree(src)
    print(f"open_tree({src}) -> fd {fd}, dev {st.st_dev} ({os.major(st.st_dev)}:{os.minor(st.st_dev)})")

    setns_mnt(pid)
    print(f"setns: now in the mount namespace of pid {pid}")

    # The target is checked AFTER setns, because that is the namespace it must exist in. A missing
    # target is the likeliest failure and it must not read as a mount problem.
    if not os.path.isdir(target):
        print(
            f"REFUSING: {target} does not exist in the container namespace -- create it there "
            f"first (mkdir -p) and re-run",
            file=sys.stderr,
        )
        return 1
    move_mount(fd, target)
    after = os.stat(target)
    print(
        f"move_mount -> {target}, dev now {after.st_dev} ({os.major(after.st_dev)}:{os.minor(after.st_dev)})"
    )
    if after.st_dev != st.st_dev:
        print(
            "WARNING: the target's device does not match the source's. The mount may not have "
            "taken effect where you think it did.",
            file=sys.stderr,
        )
        return 1
    print("OK: same device on both sides, so the mount is the source filesystem")
    return 0


if __name__ == "__main__":
    sys.exit(main())
