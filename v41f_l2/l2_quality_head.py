"""L2 quality regression head on top of a frozen (or lightly fine-tuned) document encoder.

Purpose
-------
The funnel's only layer fast enough to census-scan ~4-8e7 docs: a small sentence/document
encoder (Qwen3-Embedding-0.6B / bge-m3 class, chosen by scripts/probe_encoders*.py) feeds a
tiny 4-dim regression head predicting the teacher rubric. Trained from 66's score ledger
rows (datagen/score_ledger.py): one row per doc, ``rubric_dims`` a dict[str,int] with each
value in 1..5 (RUBRIC_MIN..RUBRIC_MAX in score_ledger.py). The head predicts a continuous
value per dimension; MSE against the ordinal label keeps the ORDERING (3>2), which is what
the downstream quota/threshold selector consumes.

Loss = per-dim MSE (ordinal-preserving) + lambda_rank * within-domain pairwise ranking.
The ranking term is load-bearing: without it the head just learns the domain prior
("Wikipedia-shaped text always scores high", "raw code always low") and becomes the
register/cross-domain failure the FineWeb-Edu head showed (see
docs/lessons/fineweb_edu_classifier_locked.md). Ranking WITHIN domain removes the between-
domain mean and forces the head to separate good from bad docs inside each corpus.

Deliberately NOT here (later stages, hooks left but unimplemented):
- isotonic / temperature calibration of the raw 1..5 output;
- encoder full fine-tuning (default frozen; train only the head for the first pass);
- the census-scan driver (that is the encoder probe throughput path, separate).

This module is a skeleton: shapes/losses are implemented and self-checked, the data
loader reads the ledger but the real DataLoader wiring/training loop is intentionally
minimal until labels land.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# The 4 rubric dimensions are owned by 66's teacher rubric, not defined here. Import the
# single source of truth so a rubric rename makes THIS module fail loudly instead of
# silently masking every label to nothing. l3_rubric is at repo root /datagen.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from datagen.l3_rubric import (  # noqa: E402
    _DIMS as RUBRIC_DIMS,
)
from datagen.l3_rubric import CODE_RUBRIC, NL_RUBRIC  # noqa: E402
from datagen.l3_rubric import (  # noqa: E402
    MAX_SCORE as RUBRIC_MAX,
)
from datagen.l3_rubric import (  # noqa: E402
    MIN_SCORE as RUBRIC_MIN,
)

# both rubrics share the SAME four dim keys; only the per-dim judging text differs (and is
# selected upstream by rubric_kind). The head therefore has one 4-dim output group; kind is
# carried per sample as metadata/conditioning, not a second disjoint output.
RUBRIC_KINDS = (CODE_RUBRIC["kind"], NL_RUBRIC["kind"])
DEFAULT_DIMS = tuple(RUBRIC_DIMS)


@dataclass
class L2Config:
    embed_dim: int = 1024  # both shortlisted encoders emit 1024
    hidden_dim: int = 256
    n_dims: int = len(DEFAULT_DIMS)
    dropout: float = 0.0
    freeze_encoder: bool = True
    lambda_rank: float = 0.5  # within-domain ranking weight vs per-dim MSE
    margin: float = 0.0  # use plain logistic pairwise (no fixed margin)


# Loud coupling: the regression head's output width MUST equal the teacher rubric. If 66
# adds/removes a rubric dimension this fires at import and the training side cannot start
# on a silently wrong shape (the failure class this whole module exists to prevent).
assert L2Config().n_dims == len(DEFAULT_DIMS) == 4
assert RUBRIC_MIN == 1 and RUBRIC_MAX == 5
assert CODE_RUBRIC["kind"] != NL_RUBRIC["kind"]
assert tuple(CODE_RUBRIC["dimensions"]) == tuple(NL_RUBRIC["dimensions"]) == DEFAULT_DIMS


class QualityHead(nn.Module):
    """pooled [B, embed_dim] -> n_dims continuous scores (unbounded; MSE keeps them in range)."""

    def __init__(self, cfg: L2Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.n_dims),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings.float())


HEAD_CKPT_VERSION = 1


def save_head(path, head: QualityHead, cfg: L2Config):
    """Save ONLY the head (the encoder is a frozen public base, never checkpointed). The
    scanner's HeadPredictor loads this exact shape: {"version","cfg","head"}."""
    from dataclasses import asdict

    torch.save({"version": HEAD_CKPT_VERSION, "cfg": asdict(cfg), "head": head.state_dict()}, path)


def load_head_state(path):
    """Return (head_state_dict, cfg_kwargs) for save_head's format. Refuses a foreign blob."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "head" not in state:
        raise ValueError(f"{path} is not an L2 head checkpoint (need keys version/cfg/head)")
    return state["head"], state.get("cfg", {})


def per_dim_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error per dimension over rows where that dim is labelled.

    pred,target: [B, n_dims]; mask: [B, n_dims] 1 where a 1..5 label exists. Averaging per
    dim then across dims keeps a sparsely-labelled dimension from being swamped.
    """
    se = (pred - target).square() * mask
    denom = mask.sum(0).clamp_min(1.0)
    return (se.sum(0) / denom).mean()


def within_domain_rank_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, domain_id: torch.Tensor
) -> torch.Tensor:
    """Logistic pairwise ranking WITHIN each domain, averaged over active dimensions.

    For every (i,j) in the same domain where dim d is labelled for both and target_i >
    target_j, encourage pred_i > pred_j via softplus(-(pred_i-pred_j)). Comparing only
    within domain subtracts the domain mean and kills the "domain = score" prior.

    Normalization is composition-invariant: per dim, take the MEAN over every comparable
    pair (each pair weighted equally, so repeating data to grow B or regrouping rows into
    domains does not change an equal-quality batch), then average across only the dims that
    actually had a comparable pair. This mirrors per_dim_mse's across-dim mean; a prior
    version accumulated pair_count/B^2 fractions as the denominator, which let the effective
    rank weight wobble with B and the batch's domain/label composition.

    O(B^2) in the batch; training uses large batches drawn per-domain (see L2Dataset), and
    the pairs are computed as a vectorised mask, no python pair loop.
    """
    D = pred.shape[1]
    total = pred.new_zeros(())
    n_active_dims = 0
    mbool = mask.bool()
    same_domain = domain_id[:, None] == domain_id[None, :]  # [B,B]
    for d in range(D):
        m = mbool[:, d]
        labelled = m[:, None] & m[None, :]  # both rows labelled
        td = target[:, d]
        higher = td[:, None] > td[None, :]  # i better than j
        pair_mask = same_domain & labelled & higher
        if not pair_mask.any():
            continue
        diff = pred[:, d, None] - pred[None, :, d]  # pred_i - pred_j
        total = total + F.softplus(-diff)[pair_mask].mean()
        n_active_dims += 1
    if n_active_dims == 0:
        return pred.new_zeros(())
    return total / n_active_dims


def quality_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    domain_id: torch.Tensor,
    cfg: L2Config,
) -> tuple[torch.Tensor, dict]:
    mse = per_dim_mse(pred, target, mask)
    rank = within_domain_rank_loss(pred, target, mask, domain_id)
    total = mse + cfg.lambda_rank * rank
    return total, {"mse": mse.detach(), "rank": rank.detach(), "total": total.detach()}


class L2QualityModel(nn.Module):
    """Frozen document encoder (1024-d pooled, L2-normalized) + the regression head.

    pooling follows the chosen encoder's shipped convention: 'lasttoken' for Qwen3-Emb,
    'cls' for bge. The encoder is frozen by default (first pass trains the head only);
    unfreezing is a later decision once head-only results are known.
    """

    def __init__(self, encoder: nn.Module, cfg: L2Config, pooling: str = "cls"):
        super().__init__()
        self.encoder = encoder
        self.head = QualityHead(cfg)
        self.pooling = pooling
        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def pool(self, last_hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            v = last_hidden[:, 0]
        elif bool(attn[:, -1].all()):
            v = last_hidden[:, -1]
        else:
            pos = attn.sum(1) - 1
            v = last_hidden[torch.arange(last_hidden.size(0)), pos]
        return F.normalize(v.float(), p=2, dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.head_training_only()):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emb = self.pool(h, attention_mask)
        return self.head(emb)

    def head_training_only(self) -> bool:
        return not next(self.encoder.parameters()).requires_grad


def train_step(model, batch, cfg: L2Config) -> tuple[torch.Tensor, dict]:
    """One forward+loss over an encoded/text batch. batch has input_ids, attention_mask,
    target [B,n_dims] 1..5 (0 where masked), mask, domain_id."""
    pred = model(batch["input_ids"], batch["attention_mask"])
    return quality_loss(pred, batch["target"], batch["mask"], batch["domain_id"], cfg)


class L2Dataset:
    """Maps ledger rows to training tensors. Minimal on purpose; replace with a streaming
    shard reader once 66's labels land.

    Each ledger row: {doc_id, domain, rubric_kind, rubric_dims: {dim:int 1..5}, ...}.
    Document TEXT is not in the ledger; production joins doc_id -> corpus content. This
    skeleton accepts an already-encoded tensor (encoder embedding run offline/online), so
    it can train against cached embeddings without re-running the encoder.

    rubric_dims keys MUST be the teacher's RUBRIC_DIMS (any other name raises); rubric_kind
    must be one of the two l3_rubric kinds when present. Both kinds share the four dims, so
    they collate onto the same 4-column target; kind is returned for optional conditioning.
    """

    def __init__(
        self,
        ledger_path: str | Path,
        dim_names: tuple[str, ...] = DEFAULT_DIMS,
        domain_to_idx: dict[str, int] | None = None,
    ):
        self.dim_names = tuple(dim_names)
        if set(self.dim_names) != set(DEFAULT_DIMS):
            raise ValueError(f"head dims {self.dim_names} != teacher rubric dims {DEFAULT_DIMS}")
        with open(ledger_path) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        self.rows = [r for r in rows if r.get("rubric_dims")]
        domains = sorted({r["domain"] for r in self.rows})
        kinds = sorted({r.get("rubric_kind") for r in self.rows if r.get("rubric_kind")})
        self.domain_to_idx = domain_to_idx or {d: i for i, d in enumerate(domains)}
        self.kind_to_idx = {k: i for i, k in enumerate(kinds)}
        self._validate()

    def _validate(self):
        for r in self.rows:
            dims = r["rubric_dims"]
            unknown = set(dims) - set(DEFAULT_DIMS)
            if unknown:
                raise ValueError(
                    f"{r['doc_id']} unknown rubric dim(s) {sorted(unknown)}; "
                    f"only {list(DEFAULT_DIMS)} are defined in l3_rubric"
                )
            kind = r.get("rubric_kind")
            if kind is not None and kind not in RUBRIC_KINDS:
                raise ValueError(f"{r['doc_id']} rubric_kind {kind!r} not in {RUBRIC_KINDS}")
            for k, v in dims.items():
                if not (isinstance(v, int) and not isinstance(v, bool) and RUBRIC_MIN <= v <= RUBRIC_MAX):
                    raise ValueError(f"{r['doc_id']} dim {k}={v} outside {RUBRIC_MIN}..{RUBRIC_MAX}")

    def __len__(self):
        return len(self.rows)

    def collate_labels(self, idx: list[int]) -> dict:
        B, D = len(idx), len(self.dim_names)
        target = torch.zeros(B, D)
        mask = torch.zeros(B, D)
        dom = torch.zeros(B, dtype=torch.long)
        kind = torch.full((B,), -1, dtype=torch.long)  # -1 = no kind recorded
        for bi, i in enumerate(idx):
            r = self.rows[i]
            dom[bi] = self.domain_to_idx[r["domain"]]
            kn = r.get("rubric_kind")
            if kn is not None:
                kind[bi] = self.kind_to_idx[kn]
            for d, name in enumerate(self.dim_names):
                if name in r["rubric_dims"]:
                    target[bi, d] = float(r["rubric_dims"][name])
                    mask[bi, d] = 1.0
        return {
            "target": target,
            "mask": mask,
            "domain_id": dom,
            "rubric_kind_id": kind,
            "rubric_kind": [self.rows[i].get("rubric_kind") for i in idx],
            "doc_id": [self.rows[i]["doc_id"] for i in idx],
        }
