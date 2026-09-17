# B1 runbook: delete the nine VOID keys from `facts/corpus_filters_baseline.json`

Status: **BLOCKED on provisioning + rebuild.** Nothing in this document runs until the corpus
is rebuilt on a new pod and the check below is green for all nine domains. Zero GPU is needed
to prepare; the deletion itself requires a corpus-holding machine.

Owner when unblocked: whoever owns the corpus rebuild (0e / ae on `docs/standards/data_pipeline_rebuild_0916.md`).
Written 2026-09-18 by genB, PR #482.

## Why the keys are there and why they must go

`facts/corpus_filters_baseline.json` is a **debt register that can only shrink**
(`scripts/harness.py`, `check_corpus_filters_fp`). Each entry exists so an already-ruled
mismatch does not re-FAIL forever.

Nine entries recorded a ruled mismatch as a **pair of byte-generation fingerprints**:
`33462c13868a2194 -> 88ee503b38941bf4` — the domains were built with the pod's weaker filters,
which lacked five CCI3-HQ rules `main` added 2026-08-30. Measured residue 0.25% of the training
corpus; fb ruled no rebuild on 2026-09-01.

PR #482 redefined `fp_filters` to hash the compiled `PATTERNS` instead of the filter files'
bytes, and gave the value a `p1-` generation marker. The ruled-pair mechanism matches on
**fingerprint values**, and neither value in those nine entries exists in any generation any
more. So the entries forgive nothing; they are dead weight that would quietly accumulate.

The physical fact they record is unchanged and is **not** being deleted with them: those nine
domains were still built with the pod's weaker filters, and the 0.25% residue measurement stands
(`facts/data_quality.json#dq.stage2_corpus_built_with_weaker_filters`). What dies is the
*forgiveness mechanism for one exact fingerprint pair*, which a rebuild replaces with a correct
stamp.

## The nine keys

All nine must be rebuilt before any of them is deleted. Deleting one early does not break
anything by itself — the entry is already inert — but it makes the register claim the migration
is further along than it is, and the verification step below is per-set, not per-key.

| key | domain dir | why it was ruled |
|---|---|---|
| `chat_qa` | `data/corpus/chat_qa` | weaker pod filters, 0.25% residue |
| `chatml` | `data/corpus/chatml` | same |
| `code_py_rp1t` | `data/corpus/code_py_rp1t` | same |
| `code_py_starcoder` | `data/corpus/code_py_starcoder` | same |
| `cot` | `data/corpus/cot` | same |
| `en_c4_stage2` | `data/corpus/en_c4_stage2` | same |
| `math_owm_stage2` | `data/corpus/math_owm_stage2` | same |
| `textbook_30b` | `data/corpus/textbook_30b` | same |
| `zh_web` | `data/corpus/zh_web` | same |

The other seven baseline entries are **not** part of B1 and must stay: `chat`, `code`, `en`,
`math`, `textbook`, `web_hq`, `wiki` carry "built before filters_fp existed" (no stamp at all,
a different debt than a dead pair).

## Preconditions — all four, in order

1. **Provisioning done.** A pod exists with the corpus volume mounted (`docs/standards/infra_persistent_rebuild_0916.md`, owner 66).
2. **The rebuild is done for all nine domains**, per `docs/standards/data_pipeline_rebuild_0916.md`. Each domain's `data/corpus/<d>/build_corpus_stats.json` must carry a `filters_fp` that starts with `p1-`.
3. **The check is green for all nine.** Run on the corpus machine:

   ```bash
   python3 -c "
   import sys; sys.path.insert(0,'scripts')
   import harness as H
   st, ev = H.check_corpus_filters_fp(H.ROOT)
   print(st); print(ev)"
   ```

   Expected before deletion: `PASS`, and the evidence names the nine under `UNMIGRATED debt`.
   Expected after deletion: `PASS` with **no** `UNMIGRATED` clause, and `9/9`-style
   `N/M domain(s) match filters p1-…`.
   A `FAIL` means a domain is still `p1`-generation but wrong, or carries junk: **stop, that is
   a rebuild defect, not a baseline task.**
4. **The branch is at current `main`**, so the check you ran is the check that ships.

## The deletion

Per key, one at a time, each with the reason recorded. Do not script a bulk `del` — a wrong key
here removes a forgiveness the register still owes, and the diff is the only review surface.

```bash
cd <worktree at current main>
git checkout -b b1-void-baseline-keys

python3 - <<'PY'
import json
p = "facts/corpus_filters_baseline.json"
d = json.load(open(p))
keys = ["chat_qa", "chatml", "code_py_rp1t", "code_py_starcoder", "cot",
        "en_c4_stage2", "math_owm_stage2", "textbook_30b", "zh_web"]
for k in keys:
    v = d.get(k, "")
    assert v.startswith("VOID"), f"{k} is not a VOID entry ({v[:60]!r}) -- refusing to delete"
    del d[k]
assert len(d) == 7, f"expected 7 entries left (the pre-stamp six + web-style five), got {len(d)}: {sorted(d)}"
json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
open(p, "a").write("\n")
print("deleted", len(keys), "VOID keys; remaining:", sorted(d))
PY
```

The two `assert`s are the safety: the first refuses to delete an entry that is not VOID, the
second refuses to land a register that lost the wrong number of entries.

Then update the pointer in `datagen/corpus_fingerprint.py`'s `fp_filters` docstring, which
currently says the nine entries "must be re-stamped by a rebuild" — after this PR they are gone,
and a docstring describing deleted keys is a claim the tree contradicts.

```bash
grep -n "re-stamped\|VOID" datagen/corpus_fingerprint.py   # the lines to update
```

## Verification — four checks, all required

```bash
# 1. the file parses and holds exactly the seven keys that stay
python3 -c "
import json; d=json.load(open('facts/corpus_filters_baseline.json'))
print(len(d), sorted(d))
assert len(d)==7, d
assert not any(str(v).startswith('VOID') for v in d.values()), 'a VOID entry survived'"

# 2. no VOID marker anywhere in the tree
grep -rn "VOID as of 2026-09-17" --include="*.json" --include="*.py" . | grep -v "^./.git" || echo "clean"

# 3. the check still passes on the corpus machine, with NO unmigrated clause
python3 -c "
import sys; sys.path.insert(0,'scripts')
import harness as H
st, ev = H.check_corpus_filters_fp(H.ROOT)
print(st); print(ev)
assert st == 'PASS', (st, ev)
assert 'UNMIGRATED' not in ev, f'a domain is still unmigrated: {ev}'"

# 4. the full gate, because this touches a check's own input
python3 scripts/harness.py check
```

## What must NOT happen

- **Do not delete a key whose domain is not rebuilt.** The entry is inert, so the deletion looks
  harmless and is not: the rebuild is what makes the check green for that domain, and deleting
  early removes the record that it was ever owed.
- **Do not widen the deletion to the other seven entries.** They are a different debt (no stamp,
  not a dead pair) and are owned by the same rebuild, later.
- **Do not restore the pair values.** They name generations that no longer exist; a re-added
  `33462c13 -> 88ee503b` entry would forgive nothing and read as though it did.
- **Do not edit `data/PROVENANCE.md` or `facts/*.json`'s historical values.** Those record what
  was stamped at the time and are correct as history.

## Scope

This is a **ledger-data** change plus a docstring: `facts/corpus_filters_baseline.json` and
`datagen/corpus_fingerprint.py`. No production code, no corpus bytes, no GPU. It needs a second
reader like any other PR, and it cannot be tested locally — the whole point of the check is that
it SKIPs on a machine without the corpus.
