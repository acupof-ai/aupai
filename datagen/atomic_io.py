"""Durable, atomic publication of a fully-written temp file to its final path.

One shared implementation for the datagen writers. fsync on the file alone does
NOT persist the rename: the bytes are durable but the new directory entry is not,
so a crash between os.replace and the directory fsync can leave the final path
absent. durable_publish closes that gap, mirroring
v41f_l2/l2_census_scan._durably_rename (file fsync -> os.replace -> fsync parent).
"""

import os


def fsync_dir(dir_path):
    """fsync a directory so a file creation/rename inside it is durable (POSIX)."""
    fd = os.open(dir_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_publish(fh, tmp, dst):
    """Publish the open temp file `fh` (its path is `tmp`) to `dst`, durably and
    atomically. Call while `fh` is still open, after every byte is written:

        flush -> fsync(file bytes) -> os.replace(tmp, dst) -> fsync(parent dir)

    The file fsync precedes the rename so it never publishes a partial buffer; the
    parent-dir fsync follows it so the rename itself survives a crash.
    """
    fh.flush()
    os.fsync(fh.fileno())
    os.replace(tmp, dst)
    fsync_dir(os.path.dirname(os.path.abspath(dst)))
