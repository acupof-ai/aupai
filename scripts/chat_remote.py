#!/usr/bin/env python3
"""Chat with model: encode on Mac, infer on pod, decode on Mac.

Usage: uv run python3 chat_remote.py [prompt]
"""
import json
import subprocess
import sys

from tokenizers import Tokenizer

TOK_PATH = "data/tokenizer.json"


def infer_remote(ids):
    """Run inference on pod, return output token IDs."""
    ids_json = json.dumps(ids)
    tmp_remote = "/tmp/infer_ids.json"

    # Push via base64 to avoid stdin issues
    import base64
    b64 = base64.b64encode(ids_json.encode()).decode()
    subprocess.run(
        ["bash", "-c", f"~/bin/pod 'echo {b64} | base64 -d > {tmp_remote}'"],
        capture_output=True, timeout=30,
    )
    # Run inference
    result = subprocess.run(
        ["bash", "-c", f"~/bin/pod 'cd /work/aupai && CUDA_VISIBLE_DEVICES=0 python3 infer.py {tmp_remote}'"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-300:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    tok = Tokenizer.from_file(TOK_PATH)
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        ids = tok.encode(f"问：{q}\n答：").ids
        out = infer_remote(ids)
        print(tok.decode(out[len(ids):], skip_special_tokens=True))
        return
    print("(empty line to quit)")
    while True:
        try:
            q = input("问 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ids = tok.encode(f"问：{q}\n答：").ids
        try:
            out = infer_remote(ids)
            print(tok.decode(out[len(ids):], skip_special_tokens=True))
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
