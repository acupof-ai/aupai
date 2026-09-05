"""Append one row per peer message received by the controller: who, words, when. No content."""
import json, sys, datetime

def add(sender, words):
    row = {"ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "from": sender, "words": int(words)}
    with open("runs/msg_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    add(sys.argv[1], sys.argv[2])
