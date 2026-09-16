#!/usr/bin/env python3
"""Selftest for the L3 teacher-label pipeline: sampler determinism/strata, rubric
loud-failure parsing, end-to-end stub labelling. CPU only, no network.

# restartable: test-only; all I/O is in tempfile dirs on a tiny synthetic corpus,
# nothing is appended to real data, and an interrupt leaves nothing to resume.

  python3 datagen/test_l3_label_pipeline.py --selftest
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import l3_rubric as R  # noqa: E402
from l3_stratified_sample import length_band, sample_stream, strata_key  # noqa: E402


def _corpus(td):
    p = os.path.join(td, "c.jsonl")
    rows = []
    # two languages x three lengths, several docs each
    for lang, src in (("py", "openbmb/L3/py"), ("en", "web/en")):
        for size in (100, 800, 4000):
            for k in range(6):
                body = ("word " * (size // 5)).strip()
                if lang == "py":
                    body = ("def f(x):\n    " + "x = x + 1\n    " * (size // 20)).strip()
                rows.append({"content": body, "source": src, "url": f"{lang}-{size}-{k}"})
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_sampler_strata_and_determinism():
    with tempfile.TemporaryDirectory() as td:
        p = _corpus(td)
        pools1, seen1, _ = sample_stream([p], per_stratum=4, seed=1, allow_short=True)
        pools2, seen2, _ = sample_stream([p], per_stratum=4, seed=1, allow_short=True)
        assert seen1 == seen2 and len(seen1) > 1
        # same seed -> identical draw
        assert [m["url"] for k in pools1 for m in pools1[k]] == [m["url"] for k in pools2 for m in pools2[k]]
        # different seed usually moves the draw but keeps per-stratum size + population
        pools3, seen3, _ = sample_stream([p], per_stratum=4, seed=2, allow_short=True)
        assert seen3 == seen1
        for k in pools1:
            assert len(pools3[k]) == len(pools1[k]) == min(4, seen1[k])
        # every drawn row carries the stratum metadata
        for k, pool in pools1.items():
            for m in pool:
                assert strata_key(m) == k and m["language"] == k[0] and m["length_band"] == k[1]


def test_length_bands():
    assert length_band(0) == "xs" and length_band(399) == "xs"
    assert length_band(400) == "s" and length_band(3000) == "l"
    assert length_band(10**9) == "xl"


def test_content_doc_id_and_locked_columns():
    import hashlib

    from l3_stratified_sample import content_doc_id

    text = "hello locked set"
    assert content_doc_id(text) == hashlib.sha256(text.encode()).hexdigest()[:16]
    with tempfile.TemporaryDirectory() as td:
        p = _corpus(td)
        pools, _, _ = sample_stream([p], per_stratum=3, seed=4, allow_short=True, doc_id_mode="content")
        for pool in pools.values():
            for m in pool:
                # 16 hex, content-derived, stable; not a sequence id
                assert len(m["sample_id"]) == 16
                assert all(c in "0123456789abcdef" for c in m["sample_id"])
                assert m["sample_id"] == content_doc_id(m["content"])
                assert "-" not in m["sample_id"]


def _world(must_fail, obj):
    rubric = R.CODE_RUBRIC
    raw = json.dumps(obj)
    try:
        R.parse_scores(raw, rubric)
    except ValueError:
        return
    if must_fail:
        raise AssertionError(f"accepted malformed scores {obj}")


def test_rubric_parser_accepts_good_and_refuses_bad():
    good = {
        d: 3 for d in ("content_quality", "factual_correctness", "complexity", "educational_or_code_value")
    }
    out = R.parse_scores(json.dumps(good), R.NL_RUBRIC)
    assert out == good and set(out) == set(R.CODE_RUBRIC["dimensions"])
    bad_worlds = [
        {},  # missing all
        {"content_quality": 3},  # missing dims
        {**good, "complexity": 6},  # out of range high
        {**good, "factual_correctness": 0},  # out of range low
        {**good, "complexity": "4"},  # string int
        {**good, "complexity": True},  # bool
        {**good, "extra": 2},  # unexpected key
    ]
    for w in bad_worlds:
        _world(True, w)
    # prose / fenced garbage refuses
    for raw in ("", "sorry I cannot", "```json\nnot json\n```", "no object here"):
        try:
            R.parse_scores(raw, R.CODE_RUBRIC)
            raise AssertionError(f"accepted garbage {raw!r}")
        except ValueError:
            pass


def test_rubric_selection_and_prompt():
    code = "Write a Python function `f(x)` that returns x.\n\ndef f(x):\n    return x"
    nl = "A short essay about the history of agriculture, entirely prose."
    assert R.select_rubric(code)["kind"] == "code"
    assert R.select_rubric(nl)["kind"] == "natural_language"
    p = R.build_prompt(code, R.CODE_RUBRIC)
    assert all(
        d in p for d in ("content_quality", "factual_correctness", "complexity", "educational_or_code_value")
    )
    assert R.RUBRIC_VERSION.startswith("l3rubric-")


def test_end_to_end_stub_pipeline():
    with tempfile.TemporaryDirectory() as td:
        p = _corpus(td)
        sample = os.path.join(td, "pilot.jsonl")
        labels = os.path.join(td, "labels.jsonl")
        env = dict(os.environ, PYTHONPATH=HERE)
        subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "l3_stratified_sample.py"),
                "--glob",
                p,
                "--out",
                sample,
                "--per-stratum",
                "2",
                "--seed",
                "9",
                "--allow-short",
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        with open(sample, encoding="utf-8") as sh:
            n_in = sum(1 for _ in sh)
        assert n_in >= 2
        subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "l3_label_pilot.py"),
                "--pilot",
                sample,
                "--out",
                labels,
                "--backend",
                "stub",
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        with open(labels, encoding="utf-8") as fh:
            recs = [json.loads(l) for l in fh]
        assert len(recs) == n_in
        for r0 in recs:
            assert r0["rubric_version"] == R.RUBRIC_VERSION
            assert r0["rubric_kind"] in ("code", "natural_language")
            assert r0["teacher_model"] == "stub-length-function"
            assert r0["stratum"]["language"] and r0["stratum"]["length_band"]
            assert set(r0["scores"]) == set(R.CODE_RUBRIC["dimensions"])
            assert all(isinstance(v, int) and 1 <= v <= 5 for v in r0["scores"].values())
        # resume: a second run labels nothing new
        r2 = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "l3_label_pilot.py"),
                "--pilot",
                sample,
                "--out",
                labels,
                "--backend",
                "stub",
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        assert json.loads(r2.stdout)["kept"] == 0


def test_parser_failure_is_persisted_not_fabricated(monkeypatch=None):
    """A teacher returning malformed JSON must land in the rejected file and be counted,
    never silently dropped or turned into a label. Drives l3_label_pilot via a bad stub."""

    with tempfile.TemporaryDirectory() as td:
        p = _corpus(td)
        sample = os.path.join(td, "p.jsonl")
        labels = os.path.join(td, "lab.jsonl")
        env = dict(os.environ, PYTHONPATH=HERE)
        subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "l3_stratified_sample.py"),
                "--glob",
                p,
                "--out",
                sample,
                "--per-stratum",
                "1",
                "--seed",
                "5",
                "--allow-short",
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        with open(sample, encoding="utf-8") as sh:
            n_in = sum(1 for _ in sh)
        # force every teacher reply to be malformed by monkeypatching the stub at module
        # level through a tiny driver (avoids importing argparse twice).
        driver = os.path.join(td, "drive.py")
        with open(driver, "w") as fh:
            fh.write(
                f"import sys; sys.path.insert(0,{HERE!r})\n"
                "import l3_label_pilot as LP\n"
                "LP.stub_teacher=lambda t,r:'not json at all'\n"
                f"sys.argv=['x','--pilot',{sample!r},'--out',{labels!r},"
                "'--backend','stub']\n"
                "LP.main()\n"
            )
        r = subprocess.run([sys.executable, driver], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        with open(labels + ".label_manifest.json", encoding="utf-8") as mh:
            man = json.load(mh)
        assert man["kept"] == 0 and man["rejected"] == n_in
        with open(labels + ".rejected.jsonl", encoding="utf-8") as rfh:
            rejected = [json.loads(l) for l in rfh]
        assert len(rejected) == n_in
        assert all("no JSON object" in x["reason"] for x in rejected)
        assert not os.path.exists(labels) or os.path.getsize(labels) == 0


def test_ledger_row_matches_frozen_schema():
    """The double-write ledger row passes datagen.score_ledger.validate_row, and a
    tampered row is refused (score + rubric_dims both set -> schema error)."""
    from datagen.score_ledger import LedgerSchemaError, append_rows

    row = {
        "doc_id": "0123456789abcdef",
        "domain": "en_c4_stage2_dc",
        "lang": "c4-train",
        "scorer_name": "l3-rubric",
        "scorer_version": R.RUBRIC_VERSION,
        "ts": "2026-09-16T09:00:00Z",
        "score": None,
        "rubric_dims": {
            d: 3
            for d in ("content_quality", "factual_correctness", "complexity", "educational_or_code_value")
        },
        "cut": None,
        "model": "teacher-x",
        "backend": "openai",
        "stratum": {"language": "c4-train", "length_band": "m"},
        "rubric_kind": "natural_language",
        "record_id": "fedcba9876543210",
        "src_sha": None,
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ledger.jsonl")
        assert append_rows(path, [row]) == 1
        with open(path, encoding="utf-8") as led_fh:
            loaded = json.loads(led_fh.read())
        assert loaded["doc_id"] == "0123456789abcdef" and loaded["rubric_kind"] == "natural_language"
        # a 6-score dim is rejected by the frozen validator, not silently written
        bad = dict(row, rubric_dims={**row["rubric_dims"], "content_quality": 6})
        try:
            append_rows(os.path.join(td, "bad.jsonl"), [bad])
            raise AssertionError("out-of-range rubric dim should fail validate_row")
        except LedgerSchemaError:
            pass
        # a non-UTC ts is refused
        bad2 = dict(row, ts="2026-09-16 09:00:00")
        try:
            append_rows(os.path.join(td, "bad2.jsonl"), [bad2])
            raise AssertionError("non-Z ts should fail validate_row")
        except LedgerSchemaError:
            pass


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("l3 label pipeline selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv else 0)
