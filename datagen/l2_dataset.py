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
2. Text pool: JSONL, one text chunk per row in the production shape (l2_label_pool_build):
       {"sample_id": content_doc_id(chunk), "parent_doc_id": content_doc_id(whole doc),
        "chunk_idx": <int>, "content": <chunk text>}
   The split/leak unit is parent_doc_id: all chunks of one source document share it. A row
   without parent_doc_id (single-chunk / legacy pools may use plain "doc_id") falls back to
   treating that chunk as its own document. (parent_doc_id, chunk_idx) must be unique, and a
   chunk sample_id may not repeat. A DOCUMENT-level label (ledger doc_id == parent_doc_id) is
   inherited by every chunk; a CHUNK-level label (doc_id == a sample_id) lands on one chunk.

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

    pairs = load_pairs("labels.jsonl", "pool.jsonl", scorer_version="r1")
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
from datagen.score_ledger import content_doc_id as _cdid  # noqa: E402
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
    doc_id: str  # PARENT document id (split/group key); all chunks of one doc share it
    chunk_idx: int
    text: str
    labels: tuple  # floats in 1..5, ordered by RUBRIC_DIMS[rubric_kind]
    rubric_kind: str  # "code" | "natural_language"; identifies the label order
    domain: str  # corpus domain, carried into the batch for per-domain rank loss
    chunk_id: str = ""  # the CHUNK content id (sample_id); the (doc,chunk) join handle
    truncated: bool = False  # teacher scored only a >6000-char prefix; excluded from train by default


def _iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield ln, json.loads(line)


def load_text_pool(path):
    """Index the chunk pool for a DOCUMENT-level join.

    Returns (chunks, chunk_to_parent):
      chunks           {(parent_doc_id, chunk_idx): (content, chunk_id)}
      chunk_to_parent   {chunk_id: (parent_doc_id, chunk_idx)}

    The production pool (l2_label_pool_build) writes one row PER CHUNK with
    ``sample_id = content_doc_id(chunk)`` (the chunk's own id), ``parent_doc_id`` (the whole
    source document) and ``chunk_idx``. The train/val split and leak prevention MUST key on
    the PARENT, never the per-chunk sample_id -- otherwise chunks of one source document hash
    to different sides and leak. The chunk_id survives only as the (doc,chunk) join handle.
    Rows without parent_doc_id (single-chunk / legacy pools) fall back to the chunk itself as
    the document. Loud on a duplicate (parent, chunk_idx), duplicate chunk_id, or bad id."""
    chunks = {}
    chunk_to_parent = {}
    present_parent = 0
    absent_parent = 0
    for ln, row in _iter_jsonl(path):
        chunk_id = row.get("sample_id") or row.get("doc_id")
        if not chunk_id:
            raise DatasetJoinError(f"{path}:{ln}: text row has no sample_id/doc_id")
        content = row.get("content")
        if not isinstance(content, str) or not content:
            raise DatasetJoinError(f"{path}:{ln}: text for {chunk_id} has empty content")
        chunk_idx = row.get("chunk_idx", 0)
        if not isinstance(chunk_idx, int) or isinstance(chunk_idx, bool) or chunk_idx < 0:
            raise DatasetJoinError(f"{path}:{ln}: chunk_idx must be a non-negative int")
        explicit_parent = row.get("parent_doc_id")
        if not explicit_parent:
            # Single-chunk / legacy rows fall back to chunk==doc, but a continuation chunk
            # (chunk_idx>0) or a self-declared multi-chunk row CANNOT be its own document:
            # accepting it would key the train/val split on the chunk and leak the other
            # halves of the same source document across sides.
            n_chunks = row.get("n_chunks")
            if chunk_idx > 0 or (isinstance(n_chunks, int) and not isinstance(n_chunks, bool) and n_chunks > 1):
                raise DatasetJoinError(
                    f"{path}:{ln}: chunk {chunk_id} (chunk_idx={chunk_idx}, n_chunks={n_chunks}) "
                    "has no parent_doc_id; multi-chunk rows must carry the parent id or the "
                    "document-level train/val split leaks across its chunks"
                )
            absent_parent += 1
            parent = chunk_id
        else:
            present_parent += 1
            parent = explicit_parent
        if not isinstance(parent, str) or not parent:
            raise DatasetJoinError(f"{path}:{ln}: parent_doc_id must be a non-empty string")
        key = (parent, chunk_idx)
        if key in chunks:
            raise DatasetJoinError(f"{path}:{ln}: duplicate chunk {key}")
        if chunk_id in chunk_to_parent:
            raise DatasetJoinError(
                f"{path}:{ln}: duplicate chunk sample_id {chunk_id} under "
                f"{chunk_to_parent[chunk_id]} and {key}"
            )
        chunks[key] = (content, chunk_id)
        chunk_to_parent[chunk_id] = key
    if present_parent and absent_parent:
        # A legacy/new `cat` is heterogeneous: some rows split by parent, others by chunk,
        # so the document-leak guarantee holds for only part of the pool. Refuse loudly.
        raise DatasetJoinError(
            f"{path}: {absent_parent} row(s) omit parent_doc_id while {present_parent} carry "
            "it; a mixed legacy/new pool cannot guarantee document-level splitting -- rebuild "
            "the pool so every multi-chunk row carries parent_doc_id"
        )
    return chunks, chunk_to_parent


def load_pairs(
    ledger_path,
    text_pool_path,
    *,
    scorer_name=DEFAULT_SCORER,
    scorer_version=None,
    rubric_kind=None,
    any_version=False,
    include_truncated=False,
    stats=None,
):
    """Validate ledger rows (score_ledger.validate_row via load_rows), keep the pinned
    rubric scorer, and join each label to its text chunk(s). Returns list[Example].

    The scorer_version MUST be pinned: pass ``scorer_version="r1"`` to train on exactly one
    teacher version. An append-only ledger can hold several versions (a re-label); silently
    mixing them averages incompatible targets and double-weights docs. Passing
    ``scorer_version=None`` raises -- set ``any_version=True`` ONLY to deliberately read
    every version (diagnostics); training never does. If a pinned version is absent entirely,
    that is a loud error rather than an empty pair list.

    Truncated-prefix labels are EXCLUDED by default: a ``truncated=true`` row means the
    teacher saw only the first 6000 chars of a long chunk, while the L2 encoder embeds the
    WHOLE chunk — regressing a whole-chunk embedding toward a prefix grade is target
    misalignment. Pass ``include_truncated=True`` to deliberately keep them. Every exclusion
    is counted, never silent: pass a mutable ``stats`` dict to receive
    ``excluded_truncated`` (and the count is also printed); the returned Examples carry the
    per-row ``truncated`` flag so a consumer can audit it.

    The split/leak unit is the PARENT document. A label's ``doc_id`` resolves to a parent in
    one of two ways, in order:
      1. it is a parent_doc_id in the pool -> the label covers the whole document and every
         chunk inherits it;
      2. it is a chunk sample_id -> the label targets that one chunk (resolved via
         chunk_to_parent), so it lands on exactly that (parent, chunk_idx).
    Two labels for one parent under the pinned scorer/version is a loud error."""
    if scorer_version is None and not any_version:
        raise DatasetJoinError(
            "load_pairs requires an explicit scorer_version (an append-only ledger may hold "
            "several teacher versions; mixing them silently is not allowed). Pass "
            'scorer_version="<pin>" or any_version=True for an intentional all-version read.'
        )
    rows = load_rows(ledger_path)
    chunks, chunk_to_parent = load_text_pool(text_pool_path)
    # one index pass: parent -> sorted chunk indices, O(labels + pool)
    chunks_of = {}
    for parent, ci in chunks:
        chunks_of.setdefault(parent, []).append(ci)
    # version presence is checked against the rows that survive scorer_name/kind BEFORE the
    # join, so a pinned-but-absent version raises instead of returning an empty pair list.
    def is_rubric_row(r):
        # A row for the pinned RUBRIC scorer must carry rubric_dims, never a scalar: the
        # l3-rubric scorer emits only multi-dim grades, so a scalar under its name is a
        # mislabeled/foreign row. .get keeps a hand-authored row missing the key from raising
        # a raw KeyError instead of the loader's DatasetJoinError.
        if r.get("scorer_name") != scorer_name:
            return False
        if r.get("rubric_dims") is not None:
            return rubric_kind is None or r.get("rubric_kind") == rubric_kind
        if r.get("score") is not None:
            raise DatasetJoinError(
                f"{r.get('doc_id')}: scorer {scorer_name!r} row carries a scalar score, "
                "not rubric_dims; the rubric scorer emits only 4-dim grades"
            )
        return False

    # version presence is checked against the rows that survive scorer_name/kind BEFORE the
    # join, so a pinned-but-absent version raises instead of returning an empty pair list.
    eligible = [r for r in rows if is_rubric_row(r)]
    if scorer_version is not None:
        present = {r["scorer_version"] for r in eligible}
        if scorer_version not in present:
            raise DatasetJoinError(
                f"pinned scorer_version {scorer_version!r} absent for scorer "
                f"{scorer_name!r}; ledger has {sorted(present)}"
            )
    pairs = []
    labeled_parents = set()
    excluded_truncated = 0
    for r in rows:
        if not is_rubric_row(r):
            continue
        if scorer_version is not None and r["scorer_version"] != scorer_version:
            continue
        if rubric_kind is not None and r["rubric_kind"] != rubric_kind:
            continue
        truncated = bool(r.get("truncated", False))
        if truncated and not include_truncated:
            excluded_truncated += 1
            continue
        kind = r["rubric_kind"]
        dim_order = RUBRIC_DIMS.get(kind)
        if dim_order is None:
            raise DatasetJoinError(
                f"{r['doc_id']}: rubric_kind {kind!r} is not defined in l3_rubric; have {sorted(RUBRIC_DIMS)}"
            )
        dims = r["rubric_dims"]
        missing = [d for d in dim_order if d not in dims]
        extra = [d for d in dims if d not in dim_order]
        if missing or extra:
            raise DatasetJoinError(
                f"{r['doc_id']} ({kind}): rubric_dims must be exactly {list(dim_order)}; "
                f"missing={missing} extra={extra}"
            )
        label_id = r["doc_id"]
        # A label id that is BOTH a parent_doc_id and a chunk sample_id of a DIFFERENT parent
        # is ambiguous: document-level vs chunk-level resolution would join it to different
        # text. Never guess (the historical shape silently labeled the wrong doc). A
        # single-chunk doc whose chunk text equals its whole doc resolves to the same parent,
        # which is harmless and allowed.
        if label_id in chunks_of and label_id in chunk_to_parent:
            chunk_parent, _ = chunk_to_parent[label_id]
            if chunk_parent != label_id:
                raise DatasetJoinError(
                    f"{label_id}: label id is both a parent_doc_id and a chunk sample_id of "
                    f"a different parent {chunk_parent!r}; ambiguous chunk-vs-document join"
                )
        if label_id in chunks_of:
            parent = label_id  # document-level label
            target_chunks = chunks_of[parent]
        elif label_id in chunk_to_parent:  # chunk-level label
            parent, ci = chunk_to_parent[label_id]
            target_chunks = [ci]
        else:
            raise DatasetJoinError(f"{label_id}: labeled but no matching parent doc or chunk in the pool")
        if parent in labeled_parents:
            raise DatasetJoinError(
                f"{parent}: two {scorer_name}/{r['scorer_version']} rubric labels for one "
                "parent doc; pin a single version (a re-label must not double-weight a doc)"
            )
        labeled_parents.add(parent)
        labels = tuple(float(dims[d]) for d in dim_order)
        for ci in sorted(target_chunks):
            content, chunk_id = chunks[(parent, ci)]
            pairs.append(Example(parent, ci, content, labels, kind, r["domain"], chunk_id, truncated))
    # loud accounting: a prefix-only label dropped from training must never vanish silently
    if excluded_truncated:
        print(
            f"l2_dataset: excluded {excluded_truncated} truncated-prefix label(s) from "
            f"training pairs (teacher saw only the first 6000 chars; pass include_truncated "
            "to keep)",
            file=sys.stderr,
            flush=True,
        )
    if stats is not None:
        stats["excluded_truncated"] = stats.get("excluded_truncated", 0) + excluded_truncated
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
            # production shape: every row is a CHUNK with its own sample_id and the PARENT
            # document id. doc00 carries a second chunk to exercise parent-level alignment.
            if i == 0:
                f.write(
                    json.dumps(
                        {
                            "sample_id": "doc00_chunk0",
                            "parent_doc_id": "doc00",
                            "chunk_idx": 0,
                            "content": "text 0",
                        }
                    )
                    + "\n"
                )
            else:
                f.write(
                    json.dumps(
                        {
                            "sample_id": f"{i:02d}_chunk0",
                            "parent_doc_id": f"doc{i:02d}",
                            "chunk_idx": 0,
                            "content": f"text {i}",
                        }
                    )
                    + "\n"
                )
        f.write(
            json.dumps(
                {
                    "sample_id": "doc00_chunk1",
                    "parent_doc_id": "doc00",
                    "chunk_idx": 1,
                    "content": "text 0 second half",
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "sample_id": "docnl0_chunk0",
                    "parent_doc_id": "docnl0",
                    "chunk_idx": 0,
                    "content": "a coherent paragraph of prose.",
                }
            )
            + "\n"
        )

    pairs = load_pairs(led, pool, scorer_version="r1")
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

    # LEAK FIX known-answer world. MANY multi-chunk parents (40 parents x 3 chunks), each
    # chunk with a DISTINCT sample_id but one parent_doc_id. We call the REAL split_pairs
    # (not the hash helper directly) at val_frac=0.5 across 50 seeds and assert, per seed,
    # that (a) every parent's three chunks land on ONE side, and (b) at least one parent goes
    # to val, so a keying change to the per-chunk id cannot pass by dumping everything into
    # train. Keying the bucket on chunk_id (the historical bug) splits a parent across
    # train/val on seed 0 of this exact world.
    leak_dir = os.path.join(tmp, "leak")
    os.makedirs(leak_dir, exist_ok=True)
    led_p = os.path.join(leak_dir, "ledger.jsonl")
    pool_p = os.path.join(leak_dir, "pool.jsonl")
    N_PARENTS = 40
    with open(led_p, "w") as lf, open(pool_p, "w") as pf:
        for pi in range(N_PARENTS):
            parent = f"parent{pi:03d}"
            lf.write(
                json.dumps(
                    ScoreRow(
                        doc_id=parent,
                        domain="py",
                        lang="en",
                        scorer_name="l3-rubric",
                        scorer_version="r1",
                        ts="2026-09-16T00:00:00Z",
                        rubric_dims={d: 4 for d in expected},
                        rubric_kind="code",
                        model="t",
                        backend="stub",
                        stratum=None,
                    ).to_dict()
                )
                + "\n"
            )
            for ci in range(3):
                ct = f"parent {pi} chunk {ci} " + f"word{pi}x{ci} " * 4
                pf.write(
                    json.dumps(
                        {"sample_id": _cdid(ct), "parent_doc_id": parent, "chunk_idx": ci, "content": ct}
                    )
                    + "\n"
                )
    leak_pairs = load_pairs(led_p, pool_p, scorer_version="r1")
    assert len(leak_pairs) == N_PARENTS * 3, len(leak_pairs)
    leak_parents = {p.doc_id for p in leak_pairs}
    assert len(leak_parents) == N_PARENTS
    # the three chunks of a parent have DISTINCT sample_ids (else the bug were unprovable)
    one = next(iter(leak_parents))
    assert len({p.chunk_id for p in leak_pairs if p.doc_id == one}) == 3
    split_seen_val = False
    for seed in range(50):
        tr, va = split_pairs(leak_pairs, 0.5, seed=seed)
        side_of = {}
        for exs, side in ((tr, "t"), (va, "v")):
            for ex in exs:
                side_of.setdefault(ex.doc_id, set()).add(side)
        split = [d for d, sides in side_of.items() if len(sides) > 1]
        assert not split, f"seed {seed}: parents split across train/val: {split[:3]}"
        if va:
            split_seen_val = True
    assert split_seen_val, "across 50 seeds no parent ever landed in val -- test is vacuous"

    # a CHUNK-level label (doc_id == a chunk sample_id, not the parent) lands on exactly one
    # (parent, chunk_idx); it must NOT fan out to the document's other chunks.
    cid0 = next(p.chunk_id for p in leak_pairs if p.chunk_idx == 0)
    cparent = next(p.doc_id for p in leak_pairs if p.chunk_id == cid0)
    led_c = os.path.join(leak_dir, "ledger_chunk.jsonl")
    with open(led_c, "w") as lf:
        lf.write(
            json.dumps(
                ScoreRow(
                    doc_id=cid0,
                    domain="py",
                    lang="en",
                    scorer_name="l3-rubric",
                    scorer_version="r1c",
                    ts="2026-09-16T00:00:00Z",
                    rubric_dims={d: 2 for d in expected},
                    rubric_kind="code",
                    model="t",
                    backend="stub",
                    stratum=None,
                ).to_dict()
            )
            + "\n"
        )
    chunk_pairs = load_pairs(led_c, pool_p, scorer_version="r1c")
    assert len(chunk_pairs) == 1
    assert chunk_pairs[0].doc_id == cparent and chunk_pairs[0].chunk_id == cid0

    # missing text chunk -> loud
    bad_pool = os.path.join(tmp, "bad.jsonl")
    with open(bad_pool, "w", encoding="utf-8") as bf:
        bf.write(json.dumps({"doc_id": "doc99", "content": "x"}) + "\n")
    try:
        load_pairs(led, bad_pool, scorer_version="r1")
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

    # scorer_version pinning (audit fix): training MUST name one version; a missing pin
    # raises, an absent pinned version raises (instead of an empty pair list), and only
    # any_version=True opts into an all-version read.
    try:
        load_pairs(led, pool)
    except DatasetJoinError:
        pass
    else:
        raise AssertionError("load_pairs without scorer_version/any_version must raise")
    try:
        load_pairs(led, pool, scorer_version="does-not-exist")
    except DatasetJoinError:
        pass
    else:
        raise AssertionError("a pinned-but-absent scorer_version must raise, not return []")
    pinned = load_pairs(led, pool, scorer_version="r1")
    assert pinned and all(isinstance(p, Example) for p in pinned)
    # any_version is the explicit escape hatch and returns the same single-version rows here
    assert len(load_pairs(led, pool, any_version=True)) == len(pinned)
    # a genuinely MIXED-version ledger under any_version is refused on the second label of a
    # parent (a re-label must never double-weight a doc); build one against the leak pool.
    mix_led = os.path.join(leak_dir, "mixed_versions.jsonl")
    parent0 = "parent000"
    with open(mix_led, "w") as mf:
        for ver in ("r1", "r2"):
            mf.write(
                json.dumps(
                    ScoreRow(
                        doc_id=parent0,
                        domain="py",
                        lang="en",
                        scorer_name="l3-rubric",
                        scorer_version=ver,
                        ts="2026-09-16T00:00:00Z",
                        rubric_dims={d: 3 for d in expected},
                        rubric_kind="code",
                        model="t",
                        backend="stub",
                        stratum=None,
                    ).to_dict()
                )
                + "\n"
            )
    try:
        load_pairs(mix_led, pool_p, any_version=True)
    except DatasetJoinError:
        pass
    else:
        raise AssertionError("two versions for one parent even under any_version must raise")
    # truncated-prefix labels (audit correctness gate): the teacher saw only the first 6000
    # chars while the L2 encoder embeds the whole chunk, so such a pair is excluded by default
    # and counted; include_truncated=True keeps it AND the Example carries the flag.
    trunc_pool = os.path.join(tmp, "trunc_pool.jsonl")
    trunc_led = os.path.join(tmp, "trunc_ledger.jsonl")
    with open(trunc_pool, "w") as f:
        f.write(json.dumps({"sample_id": "truncdoc0", "content": "a very long doc body"}) + "\n")
        f.write(json.dumps({"sample_id": "fulldoc0", "content": "a short doc body"}) + "\n")
    with open(trunc_led, "w") as f:
        f.write(
            json.dumps(
                ScoreRow(
                    doc_id="truncdoc0",
                    domain="py",
                    lang="en",
                    scorer_name="l3-rubric",
                    scorer_version="r1",
                    ts="2026-09-16T00:00:00Z",
                    rubric_dims={d: 5 for d in expected},
                    rubric_kind="code",
                    model="t",
                    backend="stub",
                    stratum=None,
                    truncated=True,
                ).to_dict()
            )
            + "\n"
        )
        f.write(
            json.dumps(
                ScoreRow(
                    doc_id="fulldoc0",
                    domain="py",
                    lang="en",
                    scorer_name="l3-rubric",
                    scorer_version="r1",
                    ts="2026-09-16T00:00:00Z",
                    rubric_dims={d: 4 for d in expected},
                    rubric_kind="code",
                    model="t",
                    backend="stub",
                    stratum=None,
                    truncated=False,
                ).to_dict()
            )
            + "\n"
        )
    tstats = {}
    default_pairs = load_pairs(trunc_led, trunc_pool, scorer_version="r1", stats=tstats)
    assert [p.doc_id for p in default_pairs] == ["fulldoc0"], [p.doc_id for p in default_pairs]
    assert tstats["excluded_truncated"] == 1, tstats
    kept = load_pairs(trunc_led, trunc_pool, scorer_version="r1", include_truncated=True)
    assert {p.doc_id for p in kept} == {"truncdoc0", "fulldoc0"}
    trow = next(p for p in kept if p.doc_id == "truncdoc0")
    assert trow.truncated is True
    assert all(p.truncated is False for p in kept if p.doc_id == "fulldoc0")
    # a row written BEFORE the field existed (absent) defaults to non-truncated, not excluded
    with open(trunc_led) as rf:
        raw = json.loads(rf.read().splitlines()[1])
    raw.pop("truncated", None)
    legacy_led = os.path.join(tmp, "trunc_legacy.jsonl")
    with open(legacy_led, "w") as f:
        f.write(json.dumps(raw) + "\n")
    legacy_pairs = load_pairs(legacy_led, trunc_pool, scorer_version="r1")
    assert [p.doc_id for p in legacy_pairs] == ["fulldoc0"]
    assert legacy_pairs[0].truncated is False

    # ---- rubric_dims must be EXACTLY the kind's four dims: a 5th/extra dim is refused, not
    # silently dropped (the mean-based quota selector would otherwise rank on it), and a
    # missing dim is still refused. All rows go through the real ledger file.
    def rubric_ledger(path, doc, dims, scorer="l3-rubric", score=None):
        row = {
            "doc_id": doc, "domain": "py", "lang": "en", "scorer_name": scorer,
            "scorer_version": "r1", "ts": "2026-09-16T00:00:00Z", "rubric_kind": "code",
            "model": "t", "backend": "stub", "stratum": None,
        }
        if dims is not None:
            row["rubric_dims"] = dims
        if score is not None:
            row["score"] = score
        with open(path, "w") as f:
            f.write(json.dumps(row) + "\n")

    shape_pool = os.path.join(tmp, "shape_pool.jsonl")
    with open(shape_pool, "w") as f:
        f.write(json.dumps({"sample_id": "sh0", "parent_doc_id": "shdoc",
                            "chunk_idx": 0, "content": "body text"}) + "\n")
    four = {d: 3 for d in expected}
    # extra 5th dim -> loud
    rubric_ledger(os.path.join(tmp, "extra.jsonl"), "shdoc", dict(four, rogue_fifth=5))
    try:
        load_pairs(os.path.join(tmp, "extra.jsonl"), shape_pool, scorer_version="r1")
    except DatasetJoinError:
        pass
    else:
        raise AssertionError("a 5th rubric dim must be refused, not silently dropped")
    # missing dim -> loud (unchanged contract)
    rubric_ledger(os.path.join(tmp, "missdim.jsonl"), "shdoc",
                  {d: 3 for d in expected if d != "complexity"})
    try:
        load_pairs(os.path.join(tmp, "missdim.jsonl"), shape_pool, scorer_version="r1")
    except DatasetJoinError:
        pass
    else:
        raise AssertionError("a missing rubric dim must be refused")
    # scalar score under the rubric scorer name -> DatasetJoinError naming it, never raw
    # KeyError. A second VALID rubric row keeps the pinned version present, so the refusal
    # is specifically about the scalar row (not the unrelated version-absent guard).
    scalar_pool = os.path.join(tmp, "scalar_pool.jsonl")
    with open(scalar_pool, "w") as f:
        f.write(json.dumps({"sample_id": "sh0", "parent_doc_id": "shdoc", "chunk_idx": 0, "content": "b1"}) + "\n")
        f.write(json.dumps({"sample_id": "ok0", "parent_doc_id": "okdoc", "chunk_idx": 0, "content": "b2"}) + "\n")
    with open(os.path.join(tmp, "scalar.jsonl"), "w") as f:
        f.write(json.dumps({
            "doc_id": "shdoc", "domain": "py", "lang": "en", "scorer_name": "l3-rubric",
            "scorer_version": "r1", "ts": "2026-09-16T00:00:00Z", "score": 3.0}) + "\n")
        f.write(json.dumps({
            "doc_id": "okdoc", "domain": "py", "lang": "en", "scorer_name": "l3-rubric",
            "scorer_version": "r1", "ts": "2026-09-16T00:00:00Z", "rubric_kind": "code",
            "rubric_dims": dict(four), "model": "t", "backend": "stub", "stratum": None}) + "\n")
    try:
        load_pairs(os.path.join(tmp, "scalar.jsonl"), scalar_pool, scorer_version="r1")
    except DatasetJoinError as e:
        assert "scalar score" in str(e), str(e)
    else:
        raise AssertionError("a scalar row under the rubric scorer must be refused loudly")

    # ---- missing parent_doc_id: a continuation chunk (chunk_idx>0) can not be its own
    # document, or the document-level split leaks its sibling chunks across train/val.
    cont_pool = os.path.join(tmp, "cont.jsonl")
    with open(cont_pool, "w") as f:
        f.write(json.dumps({"sample_id": "a0", "chunk_idx": 0, "content": "x"}) + "\n")
        f.write(json.dumps({"sample_id": "a1", "chunk_idx": 1, "content": "y"}) + "\n")
    try:
        load_text_pool(cont_pool)
    except DatasetJoinError:
        pass
    else:
        raise AssertionError("a chunk_idx>0 row without parent_doc_id must be refused")
    # homogeneous single-chunk legacy pool (every row chunk_idx 0, no parent) is still allowed
    single_pool = os.path.join(tmp, "single.jsonl")
    with open(single_pool, "w") as f:
        for i in range(3):
            f.write(json.dumps({"sample_id": f"only{i}", "content": f"b{i}"}) + "\n")
    _, c2p_single = load_text_pool(single_pool)
    assert all(parent == cid for cid, (parent, _) in c2p_single.items())
    # a mixed legacy/new pool (some rows carry parent, others do not) is refused: the
    # document-leak guarantee would hold for only part of the pool.
    mixed_pool = os.path.join(tmp, "mixed.jsonl")
    with open(mixed_pool, "w") as f:
        f.write(json.dumps({"sample_id": "n0", "parent_doc_id": "pnew", "chunk_idx": 0, "content": "a"}) + "\n")
        f.write(json.dumps({"sample_id": "o0", "content": "b"}) + "\n")
    try:
        load_text_pool(mixed_pool)
    except DatasetJoinError:
        pass
    else:
        raise AssertionError("a mixed with/without-parent pool must be refused")

    # ---- ambiguous label id: equal to a parent_doc_id AND a chunk sample_id of a DIFFERENT
    # parent must be refused; the benign same-content shape (chunk id == its own parent id)
    # still resolves.
    amb_pool = os.path.join(tmp, "amb.jsonl")
    with open(amb_pool, "w") as f:
        f.write(json.dumps({"sample_id": "X", "parent_doc_id": "Y", "chunk_idx": 0, "content": "A"}) + "\n")
        f.write(json.dumps({"sample_id": "Z", "parent_doc_id": "X", "chunk_idx": 0, "content": "B"}) + "\n")
    rubric_ledger(os.path.join(tmp, "amb_led.jsonl"), "X", dict(four))
    try:
        load_pairs(os.path.join(tmp, "amb_led.jsonl"), amb_pool, scorer_version="r1")
    except DatasetJoinError:
        pass
    else:
        raise AssertionError("a label id naming both a parent and another parent's chunk is ambiguous")
    benign_pool = os.path.join(tmp, "benign_pool.jsonl")
    with open(benign_pool, "w") as f:
        f.write(json.dumps({"sample_id": "Q", "parent_doc_id": "Q", "chunk_idx": 0, "content": "same"}) + "\n")
    rubric_ledger(os.path.join(tmp, "benign_led.jsonl"), "Q", dict(four))
    bp = load_pairs(os.path.join(tmp, "benign_led.jsonl"), benign_pool, scorer_version="r1")
    assert len(bp) == 1 and bp[0].doc_id == "Q"

    print(
        "l2_dataset selftest OK: doc_id join + chunk alignment, deterministic no-leak "
        "split, missing/duplicate refusal, scorer_version pin, truncate/pad collate, label "
        "mask, torch tensors; exact 4-dim set (5th/missing refused), scalar-under-rubric "
        "loud, continuation chunk without parent refused, ambiguous label id refused"
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
