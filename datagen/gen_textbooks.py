#!/usr/bin/env python3
"""Full textbook generator: CS topics -> textbook chapters via a 27B teacher.

de-117 (round 3, phi-1 route: a small high-quality Python-centric set). Reads the
cs_v1 topic table, draws a lens per chapter for variation, writes JSONL shards under
data/corpus/textbooks_v41/. One process serves ONE endpoint shard; run one per vllm/
sglang card (8 shards for 8 TP1 serves). Shards are disjoint by a stable hash of
(topic, lens, n), so eight processes never write the same chapter and a rerun of any
shard fills only its own missing keys.

    python3 datagen/gen_textbooks.py --smoke 8              # 8 chapters on --port
    python3 datagen/gen_textbooks.py --shard 0 --shards 8   # one eighth of the plan
    python3 datagen/gen_textbooks.py --shard 0 --shards 8 --target-tokens 250e6

Resumable: a chapter's (topic, lens, n) key on disk is skipped by every shard, and
its token count is summed from the row's own usage field, so restarts keep coverage.
"""
import argparse
import hashlib
import json
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "data/topic_seeds/cs_v1/topic_seeds_cs.jsonl"
OUTDIR = ROOT / "data/corpus/textbooks_v41"
SEEDS_SHA256 = "76c3552f48998695fede0196b54a71fc6dc1463181681f4fd5986c04adeac26b"
SEEDS_RECOVERY = (
    "~/aupai-keep/topic_seeds_cs_v1/topic_seeds_cs.jsonl "
    "(two further byte-identical copies: ~/aupai-textgen/genA/ and ~/code/aupai-genA-work/)"
)


def verify_topic_seeds(path, expected=SEEDS_SHA256):
    """Raise SystemExit unless `path` holds the exact expected seeds bytes.

    Two failure paths, two messages, because the actions are opposite: a MISSING file
    means go copy the seeds in; a WRONG-HASH file means a stale or half-written copy is
    there and copying over it is not obviously right -- check the source first. The
    expected digest is one module constant shared by both messages and the check, so
    the number a reader is told to trust cannot drift from the number enforced.
    """
    if not path.exists():
        raise SystemExit(
            f"topic seeds missing: {path}\n"
            "The path is not tracked in git -- the pre-commit hook's ALLOWED list refuses "
            "new data/ paths (scripts/hooks/pre-commit), and new textbook generation was "
            "stopped 2026-09-14 (option B), so the seeds were kept out of the repo.\n"
            f"Recover them from {SEEDS_RECOVERY}\n"
            f"Expected sha256: {expected}"
        )
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise SystemExit(
            f"topic seeds at {path} have the wrong content.\n"
            f"  expected sha256: {expected}\n"
            f"  found sha256:    {got}\n"
            "This is a stale or half-written copy, not a missing one -- do NOT simply "
            "overwrite it. Check where it came from, then re-copy from a known-good "
            f"source: {SEEDS_RECOVERY}"
        )


def parse_ports(spec):
    ports = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ports.extend(range(int(a), int(b) + 1))
        else:
            ports.append(int(part))
    if not ports:
        raise SystemExit("no --ports given")
    return ports


MODEL = os.environ.get("TEACHER_MODEL", "")

LENSES = [
    ("with runnable Python examples and tests that assert the result", 4),
    ("through worked examples and common pitfalls, each with passing code", 4),
    ("with test-driven development: write the failing test, then the code", 3),
    ("building one small complete program step by step with tests", 3),
    ("with clear explanations and runnable code examples", 2),
    ("from first principles, building up to the full concept in code", 2),
    ("focused on debugging techniques and reading error messages", 1),
    ("with an emphasis on performance, profiling and internals", 1),
    ("focused on real-world use cases, edge cases and trade-offs", 1),
    ("with exercises and fully worked, tested solutions at the end", 2),
    ("for a student who already knows another programming language", 1),
    ("contrasting it with related and easily confused concepts", 1),
]
_LENS_NAMES = [l for l, _ in LENSES]
_LENS_WEIGHTS = [w for _, w in LENSES]

CHAPTERS_PER_TOPIC = 118
SHARD_EVERY = 10000

PROMPT = """You are writing a Python programming textbook for strong students. Write the chapter on \
"{topic}" {lens}. Every code block must be complete, correct Python 3, runnable on its own or in \
the chapter's order, and any test must actually pass. Prefer showing the output. Write in English.

## {topic}

"""


def chapter_key(topic, lens, n):
    return hashlib.sha256(f"{topic}\x1f{lens}\x1f{n}".encode()).hexdigest()


def shard_of(topic, lens, n, shards):
    return int(chapter_key(topic, lens, n), 16) % shards


def served_model_name(port, retries=40, wait=15):
    if MODEL:
        return MODEL
    last = ""
    for _ in range(retries):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
                data = json.loads(r.read())
            names = [m.get("id") for m in data.get("data", []) if m.get("id")]
            if names:
                return names[0]
        except Exception as e:
            last = str(e)[:100]
        time.sleep(wait)
    raise SystemExit(f"no model served on port {port} after waiting: {last}")


def gen_one(idx, topic, lens, n, port, model, retries=2):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(topic=topic, lens=lens)}],
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            content = resp["choices"][0]["message"]["content"]
            finish = resp["choices"][0].get("finish_reason", "")
            if "</think>" in content:
                content = content.split("</think>", 1)[1]
            content = content.strip()
            toks = int((resp.get("usage") or {}).get("completion_tokens", 0))
            if content and len(content) > 200:
                return idx, topic, lens, n, content, toks, finish, None
            err = f"too short: {len(content)} chars"
        except Exception as e:
            err = str(e)[:120]
    return idx, topic, lens, n, None, 0, "", err


def existing_keys():
    """(topic, lens, n) keys and summed tokens already on disk (all shards share OUTDIR)."""
    keys, total = set(), 0
    if not OUTDIR.is_dir():
        return keys, total
    for fp in sorted(OUTDIR.iterdir()):
        if not fp.name.endswith(".jsonl"):
            continue
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                keys.add((r["topic"], r["lens"], r["n"]))
                total += int(r.get("tokens", 0))
    return keys, total


def _selftest():
    """The seeds guard must fire on the RIGHT failure and stay silent on good bytes.

    Every case builds its own fixture, so the whole set runs anywhere -- CI, a fresh
    clone, a machine with no seeds. The first version gated its three strongest cases
    behind `if good.exists()`, reading the real seeds off one laptop: on CI only the
    "missing" case executed while the summary printed all four names, and the mutation
    stayed green there because the hash comparison was never reached. A case that is
    skipped must not be counted, and the count must not be printed as if it ran.
    """
    import tempfile

    ran, skipped = [], []

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Positive control: a fixture whose bytes we hash ourselves, so the guard has
        # something correct to accept without depending on any file outside this test.
        good_bytes = b'{"topic": "t0"}\n{"topic": "t1"}\n'
        good_sha = hashlib.sha256(good_bytes).hexdigest()
        good = td / "good.jsonl"
        with open(good, "wb") as fh:
            fh.write(good_bytes)
        verify_topic_seeds(good, good_sha)  # must not raise
        ran.append("fixture with the right bytes accepted")

        # Missing.
        missing = td / "absent.jsonl"
        try:
            verify_topic_seeds(missing, good_sha)
        except SystemExit as e:
            msg = str(e)
            assert "missing" in msg and good_sha in msg, msg
            assert "stale or half-written" not in msg, "missing must not use the wrong-hash wording"
            ran.append("missing file -> missing message")
        else:
            raise AssertionError("missing file did not raise")

        # Truncated: same prefix as the good bytes, shorter. This is the case that
        # distinguishes a hash check from the existence-only guard it replaced.
        trunc = td / "trunc.jsonl"
        with open(trunc, "wb") as fh:
            fh.write(good_bytes[: len(good_bytes) // 2])
        try:
            verify_topic_seeds(trunc, good_sha)
        except SystemExit as e:
            msg = str(e)
            assert "wrong content" in msg and good_sha in msg, msg
            assert msg.index("wrong content") < msg.index(good_sha), "wrong-hash message must lead"
            ran.append("truncated file -> wrong-content message")
        else:
            raise AssertionError("truncated file did not raise")

        # Wrong bytes, right length class.
        wrong = td / "wrong.jsonl"
        with open(wrong, "wb") as fh:
            fh.write(b'{"topic": "x"}\n')
        try:
            verify_topic_seeds(wrong, good_sha)
        except SystemExit as e:
            assert "wrong content" in str(e)
            ran.append("wrong bytes -> wrong-content message")
        else:
            raise AssertionError("wrong bytes did not raise")

    # The real seeds, when this machine has them. A SKIP, said out loud, and never
    # counted among the cases above.
    real = Path(SEEDS_RECOVERY.split()[0]).expanduser()
    if real.exists():
        verify_topic_seeds(real)
        ran.append("real seeds accepted (sha256 matches)")
    else:
        skipped.append(f"real seeds not present at {real} -- not checked here")

    print(f"gen_textbooks selftest OK: {len(ran)} case(s) ran")
    for r in ran:
        print(f"  ran: {r}")
    for s in skipped:
        print(f"  SKIP: {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run the seeds-guard self-check")
    ap.add_argument("--smoke", type=int, default=0, help="only generate N chapters (smoke test)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--target-tokens", type=float, default=1.0e9,
                    help="TOTAL target across all shards; this shard stops at target/shards")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--port", type=int, default=0, help="single endpoint; overridden by --ports")
    ap.add_argument("--ports", default="",
                    help="comma/range list, e.g. 30000-30007; one serve per shard in order")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.ports:
        ports = parse_ports(args.ports)
        args.shards = len(ports)
        args.shard = args.shard if 0 <= args.shard < args.shards else 0
        port = ports[args.shard]
    else:
        if not (0 <= args.shard < args.shards):
            raise SystemExit(f"--shard {args.shard} out of range for --shards {args.shards}")
        ports = None
        port = args.port or 30000
    if args.shards == 1 and args.ports:
        raise SystemExit("--ports lists several serves; run one process per shard, not one")

    model = served_model_name(port) if not args.smoke else (MODEL or None)
    if args.smoke and not model:
        try:
            model = served_model_name(port, retries=1, wait=1)
        except SystemExit:
            model = "qwen38-27b"

    verify_topic_seeds(SEEDS)
    with SEEDS.open(encoding="utf-8") as f:
        topics = [json.loads(l)["topic"] for l in f
                  if l.strip() and json.loads(l).get("topic")]
    done_keys, total_tok = existing_keys()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # restartable: each chapter is appended and flushed as it completes, and the plan is
    # filtered by `p not in done_keys` below, which reads those same files back
    # (existing_keys). An interrupt costs at most the in-flight batch; nothing is
    # accumulate-then-save-once. The audit flags this file because the seeds-guard
    # selftest writes two short temp fixtures, which its WRITERS scan cannot tell from
    # output writes -- the marker records the check a reader still has to make.

    def lens_for(topic, n):
        h = int(hashlib.sha256(f"{topic}\x1f{n}".encode()).hexdigest(), 16)
        x = (h % 10000) / 10000.0
        acc = 0
        for name, w in zip(_LENS_NAMES, _LENS_WEIGHTS, strict=True):
            acc += w / sum(_LENS_WEIGHTS)
            if x < acc:
                return name
        return _LENS_NAMES[-1]

    plan = [(t, lens_for(t, n), n) for n in range(CHAPTERS_PER_TOPIC) for t in topics]
    rng = random.Random(20260913)
    rng.shuffle(plan)
    plan = [p for p in plan if shard_of(*p, args.shards) == args.shard and p not in done_keys]
    if args.smoke:
        plan = plan[:args.smoke]
    shard_target = args.target_tokens / args.shards
    shard_have = 0
    print(f"shard {args.shard}/{args.shards} port {port} model {model}: {len(plan)} chapters; "
          f"{total_tok/1e9:.4f}B on disk; shard target {shard_target/1e6:.0f}M", flush=True)

    shard_idx = len([f for f in OUTDIR.iterdir()
                     if f.name.endswith(".jsonl") and f.name.startswith(f"textbooks_s{args.shard:02d}_")])
    fout = (OUTDIR / f"textbooks_s{args.shard:02d}_{shard_idx:04d}.jsonl").open(
        "a", encoding="utf-8")
    in_shard = 0
    ok = err = 0
    run_tok = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(gen_one, i, t, lens, n, port, model): i
                for i, (t, lens, n) in enumerate(plan)}
        for f in as_completed(futs):
            idx, topic, lens, n, text, toks, finish, e = f.result()
            if text is not None:
                fout.write(json.dumps({"topic": topic, "lens": lens, "n": n, "shard": args.shard,
                                       "text": text, "tokens": toks, "finish": finish},
                                      ensure_ascii=False) + "\n")
                fout.flush()
                total_tok += toks
                run_tok += toks
                shard_have += toks
                ok += 1
                in_shard += 1
                if in_shard >= SHARD_EVERY:
                    fout.close()
                    shard_idx += 1
                    in_shard = 0
                    fout = (OUTDIR / f"textbooks_s{args.shard:02d}_{shard_idx:04d}.jsonl").open(
                        "a", encoding="utf-8")
            else:
                err += 1
                if err <= 5:
                    print(f"  err [{idx}] {topic[:40]}: {e}", flush=True)
            done_n = ok + err
            if done_n % 100 == 0:
                rate = run_tok / max(time.time() - t0, 1)
                print(f"  s{args.shard} {done_n}/{len(plan)} ok={ok} err={err} "
                      f"shard={shard_have/1e6:.1f}M total={total_tok/1e9:.4f}B "
                      f"~{rate:.0f} tok/s this shard", flush=True)
            if shard_have >= shard_target and not args.smoke:
                print(f"shard {args.shard} target {shard_target/1e6:.0f}M reached "
                      f"at {shard_have/1e6:.1f}M", flush=True)
                break
    fout.close()
    print(f"done shard {args.shard}: ok={ok} err={err} shard={shard_have/1e6:.2f}M "
          f"total_on_disk={total_tok/1e9:.4f}B", flush=True)


if __name__ == "__main__":
    main()
