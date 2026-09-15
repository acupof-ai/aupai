# restartable: read-only scorer; an interrupt only reruns forward passes, writes nothing
# until the final print. No per-shard state to corrupt.
"""MMLU evaluation via multiple-choice log-likelihood scoring.

57 subjects, 4 options each. Prompt = question + "A. .. B. .. C. .. D. ..",
score the log-likelihood of each continuation letter, pick argmax.
"""
import sys
import os
from collections import defaultdict

import torch

sys.path.insert(0, "/work/aupai")
from scripts.loader import load_checkpoint, load_tokenizer

LETTERS = ["A", "B", "C", "D"]


MMLU_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval", "mmlu_test.jsonl")
# cais/mmlu "all" test split, 14,042 rows, 13-whitespace-gram screened against every r3 _dc
# domain 2026-09-14 (runs/contam_mmlu_r3.json). Auxiliary metric, deliberately NOT in the
# datagen holdout registry: that path forces the 5MiB holdout_hashes.txt over the tracked-blob
# cap (fb ruling 2026-09-14, option B), and the SFT pool is decontaminated independently.
MMLU_SHA1 = "d9c4079e4e04aec3ffcb0e636a77f43ab5f5f022"
# 13-gram exclusion manifest: 478 verbatim-question row ids over the 14,042-row base,
# leaving the 13,564 screened questions. Tracked (runs/ is pod-synced), unlike the
# per-domain audit runs/contam_mmlu_r3.json which is not in git.
EXCLUDE_MANIFEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs", "contam_mmlu_r3_final.json")
RAW_N = 14042
EXCLUDED_N = 478
CLEAN_N = RAW_N - EXCLUDED_N  # 13564


def _excluded_indices():
    import json
    with open(EXCLUDE_MANIFEST, encoding="utf-8") as fh:
        m = json.load(fh)["mmlu"]
    ids = {int(x.split(":", 1)[1]) for x in m["excluded_ids"]}
    if m.get("n") != RAW_N or m.get("raw_13gram_hit_problem_n") != EXCLUDED_N \
            or len(ids) != EXCLUDED_N:
        raise RuntimeError(
            f"{EXCLUDE_MANIFEST}: manifest counts disagree with mmlu.py "
            f"(n={m.get('n')}, hits={m.get('raw_13gram_hit_problem_n')}, "
            f"ids={len(ids)}); refusing to screen against a shifted manifest")
    return ids


def load_dataset(screened=True):
    """Parse the PHYSICAL lines (question/choices contain embedded newlines, so
    splitlines() fragments 1127 rows into unparseable pieces), sha1-gate the raw
    file, then default-exclude the 478 contam row indices -> 13,564 questions.
    """
    import hashlib
    import json

    with open(MMLU_PATH, "rb") as fh:
        raw = fh.read()
    got = hashlib.sha1(raw).hexdigest()
    if got != MMLU_SHA1:
        raise RuntimeError(
            f"{MMLU_PATH} sha1 {got} != screened {MMLU_SHA1}; refusing to score an "
            "unscreened MMLU copy. Rebuild from cais/mmlu 'all' test and rerun the 13-gram "
            "audit before changing MMLU_SHA1")
    # Iterate physical file lines, NOT splitlines(): 1127 question/choice strings
    # contain embedded newlines, and splitlines() fragments those rows (the original
    # JSONDecodeError). A jsonl physical line is exactly one record here.
    rows = []
    with open(MMLU_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != RAW_N:
        raise RuntimeError(f"{MMLU_PATH}: {len(rows)} physical rows != {RAW_N}; base shifted")
    if not screened:
        return rows
    excluded = _excluded_indices()
    clean = [r for j, r in enumerate(rows) if j not in excluded]
    if len(clean) != CLEAN_N:
        raise RuntimeError(f"screened count {len(clean)} != {CLEAN_N}")
    return clean


@torch.no_grad()
def evaluate(model, tok, device):
    ds = load_dataset()
    per_subj = defaultdict(lambda: [0, 0])  # subject -> [correct, total]
    correct = total = 0

    for q in ds:
        opts = " ".join(f"{L}. {c}" for L, c in zip(LETTERS, q["choices"]))
        prompt = f"{q['question']} {opts}"
        p_ids = tok.encode(prompt).ids

        scores = {}
        for L in LETTERS:
            a_ids = tok.encode(L).ids
            x = torch.tensor([p_ids + a_ids], device=device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            log_probs = torch.log_softmax(logits[0], dim=-1)
            scores[L] = sum(
                log_probs[len(p_ids) + i - 1, t].item() for i, t in enumerate(a_ids)
            )

        pred = LETTERS.index(max(scores, key=scores.get))
        gold = q["answer"]
        subj = q["subject"]
        per_subj[subj][1] += 1
        total += 1
        if pred == gold:
            per_subj[subj][0] += 1
            correct += 1

    acc = correct / total
    print(f"MMLU overall: {correct}/{total} = {acc:.2%}")
    print("Top 5 subjects:")
    for subj, (c, n) in sorted(
        per_subj.items(), key=lambda kv: kv[1][0] / kv[1][1], reverse=True
    )[:5]:
        print(f"  {subj}: {c}/{n} = {c / n:.2%}")
    return acc


def _selftest():
    """Known-answer worlds for the two loaders fixed here. No GPU, no real data:
    monkeypatch MMLU_PATH/EXCLUDE_MANIFEST/MMLU_SHA1 to temp files.

    1 embedded-newline question must parse as ONE physical record (splitlines
      used to split it into unparseable fragments).
    2 the 478-row exclusion is applied by physical row index -> RAW-EXCLUDED.
    3 a wrong sha1 refuses; a shifted manifest count refuses.
    """
    import json
    import tempfile

    def q(i, subj, answer, nl=False):
        question = ("line1\nline2 " if nl else "") + f"q{i}"
        return {"question": question,
                "choices": [f"c{i}a", "c{i}b", "c{i}c", "c{i}d"],
                "answer": answer, "subject": subj}

    with tempfile.TemporaryDirectory() as td:
        data = os.path.join(td, "mmlu.jsonl")
        # 5 physical rows; row index 2 carries an embedded newline; rows 1 and 3 excluded
        raw_rows = [q(0, "s1", 0), q(1, "s1", 1), q(2, "s2", 2, nl=True),
                    q(3, "s2", 3), q(4, "s2", 0)]
        with open(data, "w", encoding="utf-8") as fh:
            for r in raw_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        import hashlib
        with open(data, "rb") as _dfh:
            sha = hashlib.sha1(_dfh.read()).hexdigest()
        man = os.path.join(td, "mmlu_final.json")
        with open(man, "w", encoding="utf-8") as fh:
            json.dump({"mmlu": {"n": 5, "raw_13gram_hit_problem_n": 2,
                                "excluded_ids": ["mmlu:1", "mmlu:3"]}}, fh)

        global MMLU_PATH, MMLU_SHA1, EXCLUDE_MANIFEST, RAW_N, EXCLUDED_N, CLEAN_N
        op, osha, oman = MMLU_PATH, MMLU_SHA1, EXCLUDE_MANIFEST
        oraw, oexc, oclean = RAW_N, EXCLUDED_N, CLEAN_N
        MMLU_PATH, MMLU_SHA1, EXCLUDE_MANIFEST = data, sha, man
        RAW_N, EXCLUDED_N, CLEAN_N = 5, 2, 3
        try:
            # 1+2 physical parse survives the embedded newline and screens by index
            clean = load_dataset()
            assert len(clean) == 3, len(clean)
            tail = ["line2 q2" if "\n" in r["question"] else r["question"] for r in clean]
            assert tail == ["q0", "line2 q2", "q4"], tail
            assert "\n" in clean[1]["question"]  # embedded newline kept inside one record
            unscreened = load_dataset(screened=False)
            assert len(unscreened) == 5
            # 3a wrong sha refuses
            MMLU_SHA1 = "0" * 40
            try:
                load_dataset()
                raise AssertionError("bad sha1 was accepted")
            except RuntimeError as e:
                assert "sha1" in str(e), str(e)
            # 3b shifted manifest refuses
            MMLU_SHA1 = sha
            with open(man, "w", encoding="utf-8") as _mfh:
                json.dump({"mmlu": {"n": 5, "raw_13gram_hit_problem_n": 9,
                                    "excluded_ids": ["mmlu:0"]}}, _mfh)
            try:
                load_dataset()
                raise AssertionError("shifted manifest was accepted")
            except RuntimeError as e:
                assert "manifest counts disagree" in str(e), str(e)
        finally:
            MMLU_PATH, MMLU_SHA1, EXCLUDE_MANIFEST = op, osha, oman
            RAW_N, EXCLUDED_N, CLEAN_N = oraw, oexc, oclean
    print("mmlu selftest ok: embedded-newline parse, index screening, sha/manifest guards")


if __name__ == "__main__":
    import argparse as _ap
    _a = _ap.ArgumentParser()
    _a.add_argument("--selftest", action="store_true")
    _args = _a.parse_args()
    if _args.selftest:
        _selftest()
    else:
        model, cfg = load_checkpoint("ckpt_sft.pt", device="cuda")
        model = model.to(torch.bfloat16)
        tok = load_tokenizer("data/tokenizer.json", cfg)
        evaluate(model, tok, "cuda")
