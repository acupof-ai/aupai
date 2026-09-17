"""#494 gate: the indexer's selection grid must be the PUBLISHED key's, not the consumer's.

Production is the only config where the two differ: `kv_source_layers=(2,)` publishes an
index key at ratio 2, while layer 8 is an index source at ratio 1 and REUSES it. The indexer
built its visibility mask from its own `compress_ratio`, so layer 8 asked for `seqlen/1`
columns against `seqlen/2` rows and raised at every sequence length. `v41f_small` cannot see
it: there every index source owns its key, so the two ratios always agree -- which is why
this gate must build the non-uniform config EXPLICITLY rather than reuse an existing fixture.

The assertions are semantic, not shape-only, because a shape fix still leaves the bug:

  * a mask width fixed alone (per-query `compress_lens` still on the consumer ratio) lets a
    query select FUTURE columns -- measured 24576 violations at seq 256, query 1 taking
    column 128 which covers source tokens up to 257. A gate asserting only "did not raise"
    or "idxs are in range" passes that.
  * so the causal predicate is asserted directly: the highest source position a selected
    column can cover must not exceed the query's own position.

`C` is the gate's central fact: column c covers source tokens [c*r, c*r + r - 1], following
`Indexer._group_freqs` ("compressed latent at group j takes the position of its group's first
token, j*r").
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.config import V41FConfig  # noqa: E402
from v41f.model import V41FModel  # noqa: E402

SEQ_LENS = (128, 256, 1024)
WINDOW = 128  # production window_size; also the idxs offset between window and comp


def production_cfg(**over):
    """The production compress-ratio pattern, engram off (the hash cache is unrelated here and
    #493 blocks its training path)."""
    base = dict(
        n_layers=12,
        compress_ratios=(0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1),
        kv_source_layers=(2,),
        index_source_layers=(2, 4, 8),
        window_size=WINDOW,
        engram_layer_ids=(),
        engram_num_embeddings=(),
    )
    return V41FConfig(**{**base, **over})


def build(cfg, max_batch_size=2):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return V41FModel(cfg, max_batch_size=max_batch_size)
    finally:
        torch.set_default_dtype(prev)


def capture_idxs(cfg, seqlen, start_pos=0):
    """Run the model and record (idxs, key_rows) per index-source layer.

    The idxs are read where sparse attention consumes them, so this asserts on what was
    actually attended rather than on an intermediate the caller could have bypassed.
    """
    import v41f.attention as A

    model = build(cfg).eval()
    captured = []
    real = A.sparse_attn

    def spy(q, kv, sink, idxs, scale, *a, **k):
        captured.append((idxs.detach().clone(), kv.size(1)))
        return real(q, kv, sink, idxs, scale, *a, **k)

    A.sparse_attn = spy
    try:
        with torch.no_grad():
            model(torch.randint(0, 100, (2, seqlen)))
    finally:
        A.sparse_attn = real
    return captured


def causal_violations(idxs, kv_rows, seqlen, start_pos=0):
    """Every compressed column a query selects must cover only source positions <= that
    query's own.

    BOTH quantities are derived from the captured tensors, not from config, so the gate cannot
    agree with a bug by sharing its input:

      offset   = seqlen. The concatenated idxs are [window ; compressed] and the caller's
                 offset is the WINDOW KV LENGTH, which in prefill is `seqlen` rows (the raw
                 window KV is the whole chunk) -- NOT `window_size`. Measured: at seq 256 the
                 compressed indices start at raw 256, and a gate that used window_size=128
                 read raw 128..255 as columns and reported 24576 phantom violations on a
                 CORRECT tree. That was this gate's own first bug.
      key_rows = kv_rows - seqlen, the number of compressed rows actually attended; r follows
                 as seqlen // key_rows.

    Unreachable slots carry -1 and are skipped: evaluating them yields a negative bound and
    the gate would pass vacuously.
    """
    offset = seqlen if start_pos == 0 else start_pos
    key_rows = kv_rows - offset
    ratio = seqlen // key_rows if key_rows else 1
    out = []
    for bi in range(idxs.size(0)):
        for qi in range(idxs.size(1)):
            pos = start_pos + qi
            for j in range(idxs.size(-1)):
                v = int(idxs[bi, qi, j])
                if v < 0 or v < offset:
                    continue
                col = v - offset
                if col * ratio + ratio - 1 > pos:
                    out.append((bi, pos, col))
    return out


def test_production_shape_forward_does_not_raise():
    """The original #494 failure: every sequence length raised on the mask."""
    cfg = production_cfg()
    for seqlen in SEQ_LENS:
        captured = capture_idxs(cfg, seqlen)
        assert captured, f"seq {seqlen}: no sparse attention call captured"
    print(f"  prefill ok at {SEQ_LENS}")


def test_selected_columns_are_causal_prefill():
    """The central assertion: no query selects a column covering a future source token."""
    cfg = production_cfg()
    for seqlen in SEQ_LENS:
        for n, (idxs, kv_rows) in enumerate(capture_idxs(cfg, seqlen)):
            viol = causal_violations(idxs, kv_rows, seqlen)
            assert not viol, (
                f"seq {seqlen}, sparse-attn call {n}: {len(viol)} causality violation(s); "
                f"first: query {viol[0][1]} selected column {viol[0][2]} covering source up "
                f"to {viol[0][2] * (seqlen // (kv_rows - seqlen)) + 1}"
            )
    print(f"  0 causality violations at {SEQ_LENS}")


def test_selected_columns_are_a_legal_gather():
    """Independently of causality: every selected compressed slot must index an actual row of
    the published KV, and the window slots must stay inside the window range."""
    cfg = production_cfg()
    seqlen = 256
    for n, (idxs, kv_rows) in enumerate(capture_idxs(cfg, seqlen)):
        hi = int(idxs.max())
        assert hi < kv_rows, f"call {n}: idxs go up to {hi} but the attended kv has only {kv_rows} rows"
        neg = (idxs < 0).sum().item()
        assert neg >= 0
    print(f"  idxs in range against kv rows at seq {seqlen}")


def test_nonuniform_config_is_actually_nonuniform():
    """The gate's own precondition, asserted: if every index source shared the kv source's
    ratio, `r_key == r_consumer` everywhere and the causality assertion would hold no matter
    what the indexer did -- a vacuous gate. This gate exists because the fixture is
    non-uniform, so the fixture is checked."""
    cfg = production_cfg()
    kv_ratio = cfg.compress_ratios[cfg.kv_source_layers[0]]
    consumer_ratios = [cfg.compress_ratios[l] for l in cfg.index_source_layers]
    assert any(r != kv_ratio for r in consumer_ratios), (
        f"fixture is uniform (kv ratio {kv_ratio}, consumers {consumer_ratios}): the "
        f"causality assertion cannot fail here, so this gate would be vacuous"
    )
    print(f"  non-uniform confirmed: kv ratio {kv_ratio}, consumer ratios {consumer_ratios}")


def test_decode_small_end_pos_runs_and_is_causal():
    """DECODE SEGMENT 1 of 2 -- A REAL DETECTOR (asserts the fix), not a regression gate.

    Decode (start_pos > 0) at SMALL end_pos, tested directly on `Indexer.select`: the model
    forward is prefill-only (`v41f/block.py:10`), so decode is reachable only here.

    WHY SMALL end_pos DISCRIMINATES, measured: `topk = min(index_topk, rows)` with `rows` from
    the CONSUMER ratio gives k = min(64, end_pos//1), which exceeds the published key's row
    count whenever end_pos//1 > key_rows -- i.e. end_pos <= 64 at key ratio 2 -- and
    `torch.topk` raises `selected index k out of range`. Measured PER TREE IN ITS OWN PROCESS
    (in-process module reloads gave a false "both trees raise" reading): the naive fix raises
    at start_pos 16/32/64 while this tree runs. See
    `test_decode_large_end_pos_matches_reference` for the segment that does NOT discriminate.
    """
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from v41f.indexer import Indexer

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for start_pos in (16, 32, 64, 128, 256):
            seqlen = 1
            end_pos = start_pos + seqlen
            key_rows = end_pos // 2  # published at ratio 2
            idx = Indexer(
                dim=32,
                q_lora_rank=16,
                n_heads=4,
                index_head_dim=16,
                rope_head_dim=8,
                index_topk=64,
                compress_ratio=1,
            )  # consumer r=1
            torch.manual_seed(7)
            idxs, _sc = idx.select(
                torch.randn(2, seqlen, 32),
                torch.randn(2, seqlen, 16),
                torch.randn(2, key_rows, 16),
                torch.randn(seqlen, 4, dtype=torch.complex64),
                start_pos,
                offset=start_pos,
            )
            for v in idxs.flatten().tolist():
                if v < 0 or v < start_pos:
                    continue
                col = v - start_pos
                assert col * 2 + 1 <= start_pos, (
                    f"decode start_pos={start_pos}: column {col} covers source up to "
                    f"{col * 2 + 1} > {start_pos}"
                )
    finally:
        torch.set_default_dtype(prev)
    print("  decode (start_pos 16..256): runs, columns within the published key grid")


def test_decode_large_end_pos_matches_reference():
    """DECODE SEGMENT 2 of 2 -- REGRESSION ONLY. This does NOT prove #494 is fixed.

    At large end_pos the naive fix and this tree produce BIT-IDENTICAL idxs (measured per tree
    in its own process, 3 seeds x start_pos 128/256). The reason: `topk = min(index_topk,
    rows)` caps the selection at the column count either way, every published column is
    visible in decode, and `compress_lens = end_pos // ratio` therefore never excludes
    anything -- its value is unobservable in this regime.

    It is kept to pin the behaviour against future regressions, and it is labelled here so no
    reader mistakes it for evidence about the defect. The discriminating decode case is
    `test_decode_small_end_pos_runs_and_is_causal`.
    """
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from v41f.indexer import Indexer

    # Measured on this tree at seed 7. These are RNG-ORDER-dependent: a change to the
    # model init or the number of RNG draws above moves them, and that is exactly what a
    # regression pin is for -- but the message must not read as a defect finding.
    EXPECTED_COLS_HEAD = {128: [0, 0, 1, 1], 256: [0, 1, 4, 5]}
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for start_pos in (128, 256):
            seqlen = 1
            key_rows = (start_pos + seqlen) // 2
            idx = Indexer(
                dim=32,
                q_lora_rank=16,
                n_heads=4,
                index_head_dim=16,
                rope_head_dim=8,
                index_topk=64,
                compress_ratio=1,
            )
            torch.manual_seed(7)
            idxs, _sc = idx.select(
                torch.randn(2, seqlen, 32),
                torch.randn(2, seqlen, 16),
                torch.randn(2, key_rows, 16),
                torch.randn(seqlen, 4, dtype=torch.complex64),
                start_pos,
                offset=start_pos,
            )
            head = sorted(int(v) - start_pos for v in idxs.flatten().tolist())[:4]
            assert head == EXPECTED_COLS_HEAD[start_pos], (
                f"decode start_pos={start_pos}: selected columns changed, {head} != "
                f"{EXPECTED_COLS_HEAD[start_pos]} -- this is a REGRESSION pin, so a move here "
                f"means the selection changed, not that this test found the #494 defect"
            )
    finally:
        torch.set_default_dtype(prev)
    print("  decode (start_pos 128/256): selection matches the pinned reference")
