"""Append one row per peer message received by the controller: who, words, when. No content."""
import sys, datetime, os

def add(sender, words):
    row = {"ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "from": sender, "words": int(words)}
    # Guarded append shared with every session ledger writer (de-98). The path is anchored at
    # the repo root, not the caller's cwd: a cwd-relative write is how a review row landed in
    # the integration tree. A guard that cannot import refuses, loud, never writes anyway.
    try:
        from harness_core import append_ledger
    except ImportError as e:
        print(f"msg_log: integration-tree guard unavailable ({e}); refusing to append", file=sys.stderr)
        raise SystemExit(1)
    append_ledger(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "msg_log.jsonl"), row,
                  "appending to msg_log.jsonl")

if __name__ == "__main__":
    add(sys.argv[1], sys.argv[2])
