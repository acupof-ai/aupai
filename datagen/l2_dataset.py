#!/usr/bin/env python3
"""Training data loader for the L2 quality encoder: join rubric labels to text chunks.

# restartable: pure read of a score ledger + a text pool into in-memory pairs; it writes
# nothing and has no shard/corpus scan, so an interrupt just re-runs the join.

The L2 encoder regresses the 66 L3 teacher rubric -- four INDEPENDENT 1..5 grades per
document. This module is the ONLY place labels meet text; the training script (0e) imports
it and supplies a tokenizer. It holds no model and no scoring math.

Input contract
--------------
1. Score ledger: the frozen datagen/score_ledger.py JSONL. Only scorer rows carrying
   rubric_dims are used (scorer_name default "l3-rubric", one scorer_version pinned). The
   four dimension NAMES and their order come from datagen/l3_rubric.py (code and
   natural_language kinds, 4 dims each), selected per row by its rubric_kind -- never
   redefined here. Every label row must carry all four dims of its kind, each int 1..5
   (validate_row enforces the range; this module enforces the SET and joins to text):
       content_quality, factual_correctness, complexity, educational_or_code_value
2. Text pool: JSONL, one text chunk per row, joined to a label by doc_id:
       {"doc_id": <content hash, equals the ledger doc_id>, "content": <text>,
        "chunk_idx": 0}
   The 66 sampler calls the id field "sample_id", which is accepted as an alias. chunk_idx
   is optional and defaults to 0; when one document is split into multiple chunks each row
   carries its own chunk_idx, and (doc_id, chunk_idx) must be unique. Labels are per
   document, so every chunk of a document inherits that document's 4-vector; chunk_idx is
   what proves a label is bound to the right chunk instead of by row position.

Guarantees
- join is by doc_id, never by file order: a missing text chunk is a loud LookupError and a
  duplicate (doc_id, chunk_idx) is a loud ValueError -- silent misalignment is the failure
  this loader exists to prevent;
- labels are the raw 1..5 floats in the rubric_kind's RUBRIC_DIMS order
  (regression targets, never one-hot);
- train/val split is deterministic from a hash of (seed, doc_id) at DOCUMENT granularity, so
  it is reproducible, independent of row order, and all chunks of a doc stay on one side; a
  fixed fraction (default 0.1) is val;
- collation truncates each encoded chunk to max_len and dynamically pads the batch; the
  tokenizer is an injected callable `encode(str) -> list[int]` (no hard tokenizer dep).

    pairs = load_pairs("labels.jsonl", "pool.jsonl")
    train, val = split_pairs(pairs, val_frac=0.1, seed=20260916)
    collate = make_collate(encode, max_len=512)
    batch = collate([train[0], train[1]])   # input_ids, attention_mask, labels

    python3 datagen/l2_dataset.py --selftest
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen.l3_rubric import CODE_RUBRIC, NL_RUBRIC  # noqa: E402
from datagen.score_ledger import load_rows  # noqa: E402

# Dimension names/order are NOT defined here: they are imported from the single source
# datagen/l3_rubric.py so the producer (66), this loader, and the trainer (0e) can never
# carry three silently-divergent copies. One ordered 4-tuple per rubric_kind; the regression
# head aligns to this order.
RUBRIC_DIMS = {
    CODE_RUBRIC["kind"]: tuple(CODE_RUBRIC["dimensions"]),
    NL_RUBRIC["kind"]: tuple(NL_RUBRIC["dimensions"]),
}
DEFAULT_SCORER = "l3-rubric"


class DatasetJoinError(ValueError):
    """A label cannot be correctly joined to exactly one text chunk."""


@dataclass
class Example:
    doc_id: str
    chunk_idx: int
    text: str
    labels: tuple  # floats in 1..5, ordered by RUBRIC_DIMS[rubric_kind]
    rubric_kind: str  # "code" | "natural_language"; identifies the label order
    domain: str  # corpus domain, carried into the batch for per-domain rank loss


def _iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield ln, json.loads(line)


def load_text_pool(path):
    """{(doc_id, chunk_idx): text}. Accepts sample_id as the doc_id alias. Loud on a
    duplicate chunk key or a missing/empty id or content."""
    pool = {}
    for ln, row in _iter_jsonl(path):
        doc_id = row.get("doc_id") or row.get("sample_id")
        if not doc_id:
            raise DatasetJoinError(f"{path}:{ln}: text row has no doc_id/sample_id")
        content = row.get("content")
        if not isinstance(content, str) or not content:
            raise DatasetJoinError(f"{path}:{ln}: text for {doc_id} has empty content")
        chunk_idx = row.get("chunk_idx", 0)
        if not isinstance(chunk_idx, int) or isinstance(chunk_idx, bool) or chunk_idx < 0:
            raise DatasetJoinError(f"{path}:{ln}: chunk_idx must be a non-negative int")
        key = (doc_id, chunk_idx)
        if key in pool:
            raise DatasetJoinError(f"{path}:{ln}: duplicate chunk {key}")
        pool[key] = content
    return pool


def load_pairs(
    ledger_path, text_pool_path, *, scorer_name=DEFAULT_SCORER, scorer_version=None, rubric_kind=None
):
    """Validate ledger rows (score_ledger.validate_row via load_rows), keep the pinned
    rubric scorer, and join each label to its text chunk. Returns list[Example].

    Two labels for the same (doc_id, chunk set) under the pinned scorer/version is a loud
    error: an append-only ledger can carry a re-label, and silently emitting duplicate
    training pairs would double-weight that document. Pin the version you mean."""
    rows = load_rows(ledger_path)
    pool = load_text_pool(text_pool_path)
    # one index pass: doc_id -> sorted chunk indices, so the join is O(labels + pool)
    chunks_of = {}
    for doc_id, ci in pool:
        chunks_of.setdefault(doc_id, []).append(ci)
    pairs = []
    labeled = set()
    for r in rows:
        if r["scorer_name"] != scorer_name or r["rubric_dims"] is None:
            continue
        if scorer_version is not None and r["scorer_version"] != scorer_version:
            continue
        if rubric_kind is not None and r["rubric_kind"] != rubric_kind:
            continue
        kind = r["rubric_kind"]
        dim_order = RUBRIC_DIMS.get(kind)
        if dim_order is None:
            raise DatasetJoinError(
                f"{r['doc_id']}: rubric_kind {kind!r} is not defined in l3_rubric; have {sorted(RUBRIC_DIMS)}"
            )
        dims = r["rubric_dims"]
        missing = [d for d in dim_order if d not in dims]
        if missing:
            raise DatasetJoinError(
                f"{r['doc_id']} ({kind}): rubric row missing dims {missing}; requires {list(dim_order)}"
            )
        doc_id = r["doc_id"]
        if doc_id in labeled:
            raise DatasetJoinError(
                f"{doc_id}: two {scorer_name}/{r['scorer_version']} rubric rows for one doc; "
                "pin a single version (a re-label must not silently double-weight a doc)"
            )
        chunk_idxs = chunks_of.get(doc_id)
        if not chunk_idxs:
            raise DatasetJoinError(f"{doc_id}: labeled but no text chunk in the pool")
        labeled.add(doc_id)
        labels = tuple(float(dims[d]) for d in dim_order)
        for ci in sorted(chunk_idxs):
            pairs.append(Example(doc_id, ci, pool[(doc_id, ci)], labels, kind, r["domain"]))
    return pairs


def _hash_bucket(seed, doc_id):
    h = hashlib.sha256(f"{seed}|{doc_id}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)  # deterministic [0,1)


def split_pairs(pairs, val_frac=0.1, seed=20260916):
    """Deterministic, order-independent train/val split at DOCUMENT granularity: the bucket
    is a hash of (seed, doc_id) only, so every chunk of one doc lands on the same side and
    near-identical chunks cannot leak across train/val. Returns (train, val)."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    train, val = [], []
    for ex in pairs:
        if _hash_bucket(seed, ex.doc_id) < val_frac:
            val.append(ex)
        else:
            train.append(ex)
    if not val:
        raise DatasetJoinError(
            f"val split empty at val_frac={val_frac}: need enough pairs for >=1 val example"
        )
    return train, val


def make_collate(encode, max_len=512, pad_id=0):
    """Build a PURE-PYTHON collate fn. `encode(text) -> list[int]`; truncate to max_len,
    dynamic pad. Returns Python lists (no torch import); use make_torch_loader for tensors.

    Every rubric row carries all four dims (load_pairs refuses a missing dim), so label_mask
    is all-ones. It is still emitted alongside labels so the trainer masks its MSE explicitly
    instead of treating a placeholder 0.0 as a true grade; if a future sparse label is ever
    admitted, represent it as None in Example.labels and this mask turns 0 for that dim."""
    if not callable(encode):
        raise TypeError("encode must be callable: text -> list[int token ids]")

    def collate(batch):
        seqs = [list(encode(ex.text))[:max_len] for ex in batch]
        if any(not s for s in seqs):
            raise DatasetJoinError("a chunk encoded to zero tokens; check the tokenizer")
        width = max(len(s) for s in seqs)
        input_ids, attn = [], []
        for s in seqs:
            pad = width - len(s)
            input_ids.append(s + [pad_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
        labels, label_mask = [], []
        for ex in batch:
            vals, msk = [], []
            for v in ex.labels:
                vals.append(0.0 if v is None else float(v))
                msk.append(0 if v is None else 1)
            labels.append(vals)
            label_mask.append(msk)
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "label_mask": label_mask,
            "domain": [ex.domain for ex in batch],
            "rubric_kind": [ex.rubric_kind for ex in batch],
            "doc_id": [ex.doc_id for ex in batch],
            "chunk_idx": [ex.chunk_idx for ex in batch],
        }

    return collate


def make_torch_loader(
    examples,
    encode,
    *,
    batch_size,
    max_len=512,
    pad_id=0,
    shuffle=False,
    seed=20260916,
    domain_vocab=None,
    kind_vocab=None,
):
    """Thin torch wrapper over the pure join. Returns a DataLoader whose batches are tensors:
    input_ids/attention_mask [B,L], labels/label_mask [B,4], plus domain_id [B] and
    rubric_kind_id [B] for per-domain / per-kind loss. `examples` is a train/val list from
    split_pairs. domain_vocab maps domain name -> stable int id; pass a shared one built with
    build_vocab() across train+val so train/val use identical ids. torch is imported lazily.

    An unseen domain/kind at batch time raises (KeyError) rather than getting a silent -1 id:
    build the vocab over every split before constructing the loaders."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    kind_vocab = kind_vocab or {k: i for i, k in enumerate(sorted(RUBRIC_DIMS))}
    if domain_vocab is None:
        domain_vocab = build_vocab(ex.domain for ex in examples)

    class _DS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def collate(batch):
        base = make_collate(encode, max_len, pad_id)(batch)
        return {
            "input_ids": torch.tensor(base["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(base["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(base["labels"], dtype=torch.float32),
            "label_mask": torch.tensor(base["label_mask"], dtype=torch.float32),
            "domain_id": torch.tensor([domain_vocab[d] for d in base["domain"]], dtype=torch.long),
            "rubric_kind_id": torch.tensor([kind_vocab[k] for k in base["rubric_kind"]], dtype=torch.long),
        }

    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        _DS(examples), batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=gen
    )


def build_vocab(names):
    """Stable str -> int id over an iterable of domain (or kind) names, sorted."""
    return {n: i for i, n in enumerate(sorted(set(names)))}


def _selftest():
    import tempfile

    # Contract: the loader's expected dimension set/order must equal what l3_rubric defines.
    # If the rubric dims are renamed in l3_rubric this FAILS, forcing a coordinated change
    # across producer (66), this loader, and trainer (0e) instead of a silent label shift.
    expected = ("content_quality", "factual_correctness", "complexity", "educational_or_code_value")
    for kind, rubric in (("code", CODE_RUBRIC), ("natural_language", NL_RUBRIC)):
        assert RUBRIC_DIMS[kind] == tuple(rubric["dimensions"]) == expected, kind

    tmp = tempfile.mkdtemp()
    led = os.path.join(tmp, "ledger.jsonl")
    pool = os.path.join(tmp, "pool.jsonl")
    # 20 code docs + 1 NL doc, one code doc gets two chunks; labels via frozen schema
    from datagen.score_ledger import ScoreRow

    with open(led, "w", encoding="utf-8") as f:
        for i in range(20):
            row = ScoreRow(
                doc_id=f"doc{i:02d}",
                domain="py",
                lang="en",
                scorer_name="l3-rubric",
                scorer_version="r1",
                ts="2026-09-16T00:00:00Z",
                rubric_dims={
                    "content_quality": (i % 5) + 1,
                    "factual_correctness": ((i + 1) % 5) + 1,
                    "complexity": ((i + 2) % 5) + 1,
                    "educational_or_code_value": ((i + 3) % 5) + 1,
                },
                rubric_kind="code",
                model="teacher",
                backend="stub",
                stratum={"language": "en", "length_band": "m"},
            )
            f.write(json.dumps(row.to_dict()) + "\n")
        # one natural_language row: same 4 dim names, kind selects its order from l3_rubric
        f.write(
            json.dumps(
                ScoreRow(
                    doc_id="docnl0",
                    domain="web",
                    lang="en",
                    scorer_name="l3-rubric",
                    scorer_version="r1",
                    ts="2026-09-16T00:00:00Z",
                    rubric_dims={d: g for d, g in zip(expected, (3, 4, 2, 5), strict=True)},
                    rubric_kind="natural_language",
                    model="teacher",
                    backend="stub",
                    stratum={"language": "en", "length_band": "l"},
                ).to_dict()
            )
            + "\n"
        )
        # an unrelated scalar scorer row must be ignored, not joined
        f.write(
            json.dumps(
                ScoreRow(
                    doc_id="doc00",
                    domain="py",
                    lang="en",
                    scorer_name="kenlm",
                    scorer_version="v1",
                    ts="2026-09-16T00:00:00Z",
                    score=42.0,
                    stratum=None,
                ).to_dict()
            )
            + "\n"
        )

    with open(pool, "w", encoding="utf-8") as f:
        for i in range(20):
            # doc00 also carries a second chunk to exercise chunk_idx alignment
            f.write(json.dumps({"sample_id": f"doc{i:02d}", "content": f"text {i}"}) + "\n")
        f.write(json.dumps({"doc_id": "doc00", "chunk_idx": 1, "content": "text 0 second half"}) + "\n")
        f.write(json.dumps({"doc_id": "docnl0", "content": "a coherent paragraph of prose."}) + "\n")

    pairs = load_pairs(led, pool)
    assert len(pairs) == 22, len(pairs)  # 20 code + 1 extra chunk + 1 NL; scalar ignored

    # the NL example keeps its kind and carries the NL-ordered label vector
    nl = next(p for p in pairs if p.doc_id == "docnl0")
    assert nl.rubric_kind == "natural_language" and nl.labels == (3.0, 4.0, 2.0, 5.0)

    # join correctness + label alignment: labels travel WITH the doc, not by row position
    by_key = {(p.doc_id, p.chunk_idx): p for p in pairs}
    assert by_key[("doc03", 0)].labels == (4.0, 5.0, 1.0, 2.0), by_key[("doc03", 0)].labels
    assert by_key[("doc00", 1)].labels == by_key[("doc00", 0)].labels  # chunk inherits label
    assert all(len(p.labels) == 4 and all(1.0 <= x <= 5.0 for x in p.labels) for p in pairs)

    # deterministic, reproducible, order-independent split
    t1, v1 = split_pairs(pairs, 0.1, seed=7)
    t2, v2 = split_pairs(list(reversed(pairs)), 0.1, seed=7)
    s1 = {(e.doc_id, e.chunk_idx) for e in t1}, {(e.doc_id, e.chunk_idx) for e in v1}
    s2 = {(e.doc_id, e.chunk_idx) for e in t2}, {(e.doc_id, e.chunk_idx) for e in v2}
    assert s1 == s2, "split must be independent of input order"
    assert abs(len(v1) / len(pairs) - 0.1) < 0.12 and 1 <= len(v1) < len(pairs)
    # all chunks of one doc land in the same split (no train/val leakage across chunks)
    for d in {p.doc_id for p in pairs}:
        sides = {side for side, exs in (("t", t1), ("v", v1)) for ex in exs if ex.doc_id == d}
        assert len(sides) == 1, f"{d} split across train and val"

    # missing text chunk -> loud
    bad_pool = os.path.join(tmp, "bad.jsonl")
    with open(bad_pool, "w", encoding="utf-8") as bf:
        bf.write(json.dumps({"doc_id": "doc99", "content": "x"}) + "\n")
    try:
        load_pairs(led, bad_pool)
    except DatasetJoinError:
        pass
    else:
        raise AssertionError("labeled doc with no pool text must raise DatasetJoinError")
    # duplicate chunk -> loud
    dup_pool = os.path.join(tmp, "dup.jsonl")
    with open(dup_pool, "w", encoding="utf-8") as df:
        df.write((json.dumps({"doc_id": "doc00", "content": "a"}) + "\n") * 2)
    try:
        load_text_pool(dup_pool)
    except DatasetJoinError:
        pass
    else:
        raise AssertionError("duplicate (doc_id,chunk_idx) must raise")

    # collate: truncation + dynamic padding + attention mask
    def enc(text):
        return [ord(c) % 100 + 1 for c in text]

    collate = make_collate(enc, max_len=4, pad_id=0)
    b = collate([by_key[("doc00", 0)], by_key[("doc19", 0)]])
    assert len(b["input_ids"]) == 2
    assert all(len(seq) <= 4 for seq in b["input_ids"])
    width = max(len([1 for m in row if m]) for row in b["attention_mask"])
    assert len(b["input_ids"][0]) == len(b["input_ids"][1]) == width
    assert b["labels"][0] == list(by_key[("doc00", 0)].labels)
    # all four dims are scored -> mask is all-ones, and domain/kind ride into the batch
    assert b["label_mask"] == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert b["domain"] == ["py", "py"] and b["rubric_kind"] == ["code", "code"]

    # torch wrapper: tensor labels/mask [B,4], stable domain_id/kind_id for rank loss
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        train, _val = split_pairs(pairs, 0.1, seed=7)
        dvoc = build_vocab(ex.domain for ex in pairs)  # shared across train+val
        loader = make_torch_loader(train, enc, batch_size=8, max_len=4, domain_vocab=dvoc)
        tb = next(iter(loader))
        assert tb["labels"].shape[1] == 4 and tb["label_mask"].shape == tb["labels"].shape
        assert tb["domain_id"].dtype == torch.long and tb["rubric_kind_id"].dtype == torch.long
        assert set(tb["rubric_kind_id"].tolist()) <= set(range(len(RUBRIC_DIMS)))
        assert tb["input_ids"].shape[0] == tb["labels"].shape[0] == 8
        # an unseen domain (vocab missing it) must raise, not silently get a -1 id
        nl_loader = make_torch_loader([nl], enc, batch_size=1, domain_vocab={"py": 0})
        try:
            next(iter(nl_loader))
        except KeyError:
            pass
        else:
            raise AssertionError("unseen domain must raise KeyError in the tensor collate")

    print(
        "l2_dataset selftest OK: doc_id join + chunk alignment, deterministic no-leak "
        "split, missing/duplicate refusal, truncate/pad collate, label mask, torch tensors"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
