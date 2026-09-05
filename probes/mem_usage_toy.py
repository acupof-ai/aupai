#!/usr/bin/env python3
"""Key-usage collapse in ProductKeyMemory, and whether a query normalisation fixes it.

WHY THIS EXISTS. M1/M2/M3 were stopped 2026-09-05 under readout 4: pool_touched_frac fell
monotonically to 0.0945 (M1, step 1000) and 0.0700 (M3, step 600) while key_gini rose to 0.919
and 0.938 and topk_entropy fell to 0.93 of a 3.47 maximum. The four readout-4 diagnostics are
functions of the SELECTION distribution, and readout 6 was healthy throughout
(rows_changed_since_prev 1.09-1.25 of touched rows), so the value updates land and the defect is
in which rows get selected. This measures the same three quantities on a CPU toy at 1/256 of M1's
value count, so a candidate fix can be judged before an arm is relaunched.

THE MEASUREMENT IS ON THE REAL MODULE, monkey-patched, not on a reimplementation: the collapse is
a property of the query -> half-key top-k path in model.ProductKeyMemory, and a toy copy of that
path would let the two drift exactly where the answer lives.

WHAT A FIX MUST DO. The bar is not "gini goes down": it is that touched_fraction reaches the
uniform-draw prediction for the same number of draws, which this computes rather than assumes.
With B*T tokens x top_k draws per step over V values and windows of `window` steps, a uniform
selector touches 1 - (1 - 1/V)^D of the pool, D = window*B*T*top_k. Reported beside every arm so
a number that merely improved is not mistaken for one that arrived.

    python3 probes/mem_usage_toy.py                 # baseline vs the two candidates
    python3 probes/mem_usage_toy.py --steps 600     # longer, to see whether a fix holds
    python3 probes/mem_usage_toy.py --selftest      # the metric detects a collapse it is given
"""
import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ProductKeyMemory  # noqa: E402

# 128x128 = 16,384 values at d256, top_k 32. top_k MATCHES THE ARMS' 32 rather than mem_toy's 8:
# the collapse is a statement about how concentrated the top-32 of a Cartesian combine is, and
# reading it at top_k 8 would measure a different selector. side 128 >= 32 satisfies the module's
# own constraint.
D, SIDE, TOP_K = 256, 128, 32
B, T = 8, 128


def uniform_touch_prediction(n_values, draws):
    """Fraction of the pool a UNIFORM selector touches in `draws` draws with replacement.

    The reference the fix is judged against. Written as expm1/log1p rather than (1-1/V)**D so it
    does not lose the small-probability regime to floating point at V = 16,384 and D = 3.2e6.
    """
    return -math.expm1(draws * math.log1p(-1.0 / n_values))


class QueryBatchNorm(nn.Module):
    """Lample et al. 2019 section 3.3: BatchNorm the query network's output.

    The paper applies it to the query network and reports key usage as the reason. Over the flat
    (B*T, 2*key_dim) query it is a per-feature normalisation across tokens, so it carries no
    position information and cannot leak across the causal boundary -- the statistic is over the
    batch of tokens, which is why it is legal here at all. Momentum default; running stats are
    used in eval so a single-token forward is not normalised by its own mean.
    """

    def __init__(self, width):
        super().__init__()
        self.bn = nn.BatchNorm1d(width)

    def forward(self, q):
        return self.bn(q)


class QueryL2Temp(nn.Module):
    """Per-half L2-normalise the query and the keys, scale by a learned temperature.

    The alternative when BatchNorm is objectionable: it needs no batch statistic, so it is
    identical under DDP, under a batch of one, and in eval. It bounds every score to
    [-temp, +temp] instead of letting one key's norm dominate, which is the same failure BN
    removes by a different route. Applied to the QUERY here; the key side is done in the patched
    forward because the keys are a Parameter on the module, not something a query hook can see.
    """

    def __init__(self, key_dim, init_temp=math.sqrt(1.0 / 32)):
        super().__init__()
        # log-temperature so the optimizer cannot drive it negative.
        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    def forward(self, q):
        return F.normalize(q, dim=-1) * self.log_temp.exp()


def _patched_forward(self, x):
    """model.ProductKeyMemory.forward with two hooks: `q_norm` on the query, `l2_keys` on keys.

    A COPY of the real forward with two lines added, and the copy is the cost of this probe: if
    the module's forward changes, this drifts. Kept because the alternative -- editing model.py
    before the fix is verified -- is worse, and the assert below pins the shapes the copy assumes.
    """
    B_, T_, d = x.shape
    h = self.n_mem(x)
    q = self.query(h).view(B_ * T_, 2, self.key_dim)
    qn = getattr(self, "q_norm", None)
    if qn is not None:
        q = qn(q.reshape(B_ * T_, 2 * self.key_dim)).view(B_ * T_, 2, self.key_dim)
    k0, k1 = self.keys[0], self.keys[1]
    if getattr(self, "l2_keys", False):
        k0, k1 = F.normalize(k0, dim=-1), F.normalize(k1, dim=-1)
    s0 = torch.einsum("nk,ck->nc", q[:, 0], k0)
    s1 = torch.einsum("nk,ck->nc", q[:, 1], k1)
    v0, i0 = s0.topk(self.top_k, dim=-1)
    v1, i1 = s1.topk(self.top_k, dim=-1)
    cand = (v0[:, :, None] + v1[:, None, :]).view(B_ * T_, self.top_k * self.top_k)
    idx = (i0[:, :, None] * self.side + i1[:, None, :]).view(B_ * T_, self.top_k * self.top_k)
    w, sel = cand.topk(self.top_k, dim=-1)
    flat = idx.gather(1, sel)
    w = torch.softmax(w.float(), dim=-1).to(x.dtype)
    vals = self.values(flat)
    read = torch.einsum("nkd,nk->nd", vals, w).view(B_, T_, d)
    with torch.no_grad():
        self.touched[flat.reshape(-1)] = True
        p = w.float()
        ent = -(p * (p + 1e-12).log()).sum(-1).mean()
        self.last_entropy += ent
        self.windows += 1
        _s0 = sel // self.top_k
        _s1 = sel % self.top_k
        self.key_hits[0] += torch.bincount(i0.gather(1, _s0).reshape(-1), minlength=self.side)
        self.key_hits[1] += torch.bincount(i1.gather(1, _s1).reshape(-1), minlength=self.side)
    return self.out(F.silu(self.gate(h)) * read)


def build(arm, seed=0):
    """The real module plus whichever query normalisation `arm` names."""
    torch.manual_seed(seed)
    m = ProductKeyMemory(SIDE * SIDE, D, top_k=TOP_K, sparse=False)
    m.forward = _patched_forward.__get__(m, ProductKeyMemory)
    if arm == "bn":
        m.q_norm = QueryBatchNorm(2 * m.key_dim)
    elif arm == "l2":
        m.q_norm = QueryL2Temp(m.key_dim)
        m.l2_keys = True
    elif arm != "base":
        raise ValueError(f"unknown arm {arm!r}")
    return m


def make_task(kind, seed):
    """The training signal, and CHOOSING IT IS THE WHOLE VALIDITY OF THIS PROBE.

    `smooth` -- regress onto tanh(x @ W) for a fixed random W on Gaussian inputs. Measured
    2026-09-05 and REJECTED as an adjudicator: one shared low-rank read fits it, so a collapsed
    selector is OPTIMAL and the loss column says so (BatchNorm at selector lr 0.02 reached the
    lowest touched of the nine grid cells, 0.0465, AND the lowest loss, 0.0730). On a task like
    that, "which arm spreads usage" and "which arm fits" point opposite ways, and any fix that
    spreads usage is penalised by the objective it is being judged under. Kept, because it is the
    world that reproduces the arms' collapse signature and the world that proves the trap exists.

    `recall` -- n_facts distinct discrete inputs, each with its own random target vector. No
    smooth map fits it: the only way to drive the loss down is to store per-fact information, and
    the table is where there is room for it. Here spreading usage and fitting the task are the
    SAME direction, so a fix that raises touched should also lower loss, and one that raises
    touched while raising loss is trading the task away rather than fixing the selector. This is
    the world a relaunch decision should be read from.

    Returns (sample, describe) where sample(step) -> (x, target).
    """
    g = torch.Generator().manual_seed(1234 + seed)
    if kind == "smooth":
        w = torch.randn(D, D, generator=g) * D ** -0.5

        def sample(_step):
            x = torch.randn(B, T, D, generator=g)
            return x, torch.tanh(x @ w)

        return sample, "regress tanh(x @ W), Gaussian x -- one smooth map fits it"

    if kind == "recall":
        # n_facts > the table's row count would make the task unlearnable for reasons that have
        # nothing to do with the selector; n_facts well under it leaves room for a collapsed
        # selector to still fit, which is the thing being measured. Half the rows.
        n_facts = (SIDE * SIDE) // 2
        cue = torch.randn(n_facts, D, generator=g)
        cue = F.normalize(cue, dim=-1)          # equal norms: no fact is easier to select
        val = torch.randn(n_facts, D, generator=g) * D ** -0.5

        def sample(_step):
            j = torch.randint(0, n_facts, (B, T), generator=g)
            return cue[j], val[j]

        return sample, f"{n_facts} discrete facts, each with its own random target vector"

    if kind == "zipf":
        # UNIFORM FACTS DO NOT COLLAPSE, measured 2026-09-05: every cell of the arm x selector-lr
        # grid on `recall` read touched 1.0000 and gini 0.02-0.08, because a uniform draw applies
        # symmetric gradient pressure to every key and there is nothing to concentrate on. A task
        # in which the failure never occurs cannot rank fixes for it either -- the mirror image of
        # `smooth`, where the failure is optimal.
        #
        # ZIPF IS THE PROPERTY REAL TEXT HAS that both of those lack: token and n-gram frequency
        # is heavily skewed, so a selector can drive loss down fast by serving the head and
        # abandoning the tail -- which is what a falling touched_fraction beside a falling loss
        # looks like, and what M1 did (loss 2.629 -> 1.948 while touched fell 0.306 -> 0.0945).
        # Here the tail is still worth points, so collapse is tempting but not optimal, and the
        # two directions the other tasks confounded are separated.
        n_facts = (SIDE * SIDE) // 2
        cue = F.normalize(torch.randn(n_facts, D, generator=g), dim=-1)
        val = torch.randn(n_facts, D, generator=g) * D ** -0.5
        # s=1.0 Zipf over ranks, sampled by inverse-CDF on the normalised cumulative weights.
        rank = torch.arange(1, n_facts + 1, dtype=torch.float64)
        p = (1.0 / rank)
        cdf = (p / p.sum()).cumsum(0)

        def sample(_step):
            u = torch.rand(B * T, generator=g, dtype=torch.float64)
            j = torch.searchsorted(cdf, u).clamp_(max=n_facts - 1).view(B, T)
            return cue[j], val[j]

        return sample, (f"{n_facts} facts drawn Zipf(s=1) -- head is cheap, tail still scores; "
                        f"the skew real text has and the other two tasks lack")

    raise ValueError(f"unknown task {kind!r}")


def run(arm, steps, window, lr, seed=0, quiet=False, task="zipf", sel_lr=None):
    """Train the toy and report the three diagnostics per window.

    THE TASK IS THE SAME FOR EVERY ARM AND SEEDED, so a difference between arms is the
    normalisation and not the data. See make_task for why the default is `recall` and not the
    smooth regression: on the smooth task a collapsed selector wins on loss, so that task cannot
    adjudicate a fix meant to spread usage.

    `sel_lr` splits the selector (query + keys) into its own lr group. None means one group at
    `lr`, which is what train.build_optimizers does today -- all six memory tensors in ONE
    Adagrad group at cfg.mem_lr.
    """
    m = build(arm, seed=seed)
    sample, _ = make_task(task, seed)
    if sel_lr is None:
        opt = torch.optim.Adagrad(m.parameters(), lr=lr)
    else:
        sel = [m.query.weight, m.keys]
        ids = {id(p) for p in sel}
        rest = [p for p in m.parameters() if id(p) not in ids]
        opt = torch.optim.Adagrad([{"params": rest, "lr": lr},
                                   {"params": sel, "lr": sel_lr}])
    out = []
    for step in range(1, steps + 1):
        x, tgt = sample(step)
        y = m(x)
        loss = F.mse_loss(y, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % window == 0:
            d = m.diagnostics(reset=True)
            pred = uniform_touch_prediction(d["n_values"], window * B * T * TOP_K)
            row = {"step": step, "loss": float(loss.detach()), "touched": d["touched_fraction"],
                   "ent": d["topk_entropy"], "ent_max": d["topk_entropy_max"],
                   "gini": d["key_gini"], "uniform_pred": pred}
            out.append(row)
            if not quiet:
                print(f"  {arm:<4} step {step:>4}  loss {row['loss']:.4f}  "
                      f"touched {row['touched']:.4f} (uniform {pred:.4f})  "
                      f"ent {row['ent']:.3f}/{row['ent_max']:.3f}  gini {row['gini']:.4f}")
    return out


def selftest():
    """The metric must separate a collapsed selector from a spread one, on inputs it is GIVEN.

    Not a test of the fix -- a test that the instrument can see the thing at all. A probe whose
    metric is insensitive would report every arm as healthy and the comparison would be
    decoration. Both worlds are constructed by writing the counters directly, so the assertion
    does not depend on training reproducing a collapse.
    """
    m = build("base")
    # SPREAD world: every key used equally, top-k weights uniform.
    m.key_hits.fill_(100)
    m.touched.fill_(True)
    m.last_entropy.fill_(math.log(TOP_K))
    m.windows.fill_(1)
    spread = m.diagnostics(reset=True)
    assert spread["key_gini"] < 0.01, f"uniform counts must give gini ~0, got {spread['key_gini']}"
    assert spread["touched_fraction"] == 1.0
    assert abs(spread["topk_entropy"] - math.log(TOP_K)) < 1e-5

    # COLLAPSED world: one key per half takes every hit, one row touched, one-hot weights.
    m.key_hits.zero_()
    m.key_hits[0, 0] = 10_000
    m.key_hits[1, 0] = 10_000
    m.touched.zero_()
    m.touched[0] = True
    m.last_entropy.zero_()
    m.windows.fill_(1)
    coll = m.diagnostics(reset=True)
    assert coll["key_gini"] > 0.99, f"one-key world must give gini ~1, got {coll['key_gini']}"
    assert coll["touched_fraction"] < 1e-4
    assert coll["topk_entropy"] == 0.0
    print(f"selftest: spread gini {spread['key_gini']:.4f} touched {spread['touched_fraction']:.4f}"
          f" ent {spread['topk_entropy']:.3f}  |  collapsed gini {coll['key_gini']:.4f} "
          f"touched {coll['touched_fraction']:.2e} ent {coll['topk_entropy']:.3f}")

    # The uniform prediction must be a probability and must rise with draws, or the bar the arms
    # are judged against is not a bar. Checked at the arms' own scale, not the toy's.
    p1 = uniform_touch_prediction(1_048_576, 100 * 262144 * 32)
    p2 = uniform_touch_prediction(1_048_576, 10 * 262144 * 32)
    assert 0.0 < p2 <= p1 <= 1.0, (p2, p1)
    print(f"uniform prediction at M1's scale: 10-step window {p2:.6f}, 100-step {p1:.6f} "
          f"-- so M1's measured 0.0945 is {0.0945 / p1:.4f} of it")

    # TWO PROPERTIES, and a task needs BOTH to rank a usage fix. Measured 2026-09-05, both
    # directions found the hard way:
    #   (a) collapse must COST loss. `smooth` fails this -- a one-row selector fits it slightly
    #       BETTER (0.1002 against 0.1033), so the lowest-touched cell of the first grid was also
    #       its lowest-loss cell and every arm was ranked under an objective preferring the
    #       failure.
    #   (b) collapse must be REACHABLE. `recall` with uniform facts fails this -- all nine cells of
    #       the second grid read touched 1.0000 and gini 0.02-0.08, because a uniform draw presses
    #       every key symmetrically and there is nothing to concentrate on. `zipf`, built to add
    #       the skew real text has, fails it too: touched plateaus at 0.949 over 800 steps while
    #       the loss keeps falling.
    #
    # SO NO TASK HERE SATISFIES BOTH, and this function REPORTS that instead of asserting it away.
    # The measured table below is the finding: on CPU at 1/64 of M1's scale, the only task that
    # reproduces the collapse is the one where the collapse is correct. That is why the decision
    # moved to six single-card real-data cells (4c's ruling 2026-09-05) rather than being taken
    # from this file.
    #
    # WHAT IS ASSERTED: each task's CHARACTER, so a change in behaviour is loud. If `zipf` ever
    # starts collapsing, or `smooth` ever starts penalising it, the numbers this file was read
    # against have changed and the assertion says so. `can_rank` is the gate any future caller
    # must check before believing a ranking, and it is computed, not declared.
    can_rank = {}
    for task, want_hurt, want_collapse in (("zipf", True, False),
                                           ("recall", True, False),
                                           ("smooth", False, True)):
        torch.manual_seed(0)
        honest = build("base")
        torch.manual_seed(0)
        crippled = build("base")
        # One row for every token: the table can hold exactly one vector of useful information.
        crippled.forward = _one_row_forward.__get__(crippled, ProductKeyMemory)
        losses, touched = {}, {}
        for nm, mod in (("honest", honest), ("crippled", crippled)):
            sample, _ = make_task(task, 0)   # same data for both
            opt = torch.optim.Adagrad(mod.parameters(), lr=0.02)
            # 240 STEPS, AND THE USAGE IS READ FROM THE SECOND WINDOW. A 120-step read is too
            # early to see reachability: on `smooth` the honest selector reads 0.966 of the pool
            # over steps 1-120 and 0.091 over 121-240, so the first window reports "the failure
            # does not occur" for the one task where it certainly does. The counter is reset at
            # 120 so the window that decides is 121-240, after the selector has had time to
            # concentrate.
            for step in range(1, 241):
                x, tgt = sample(step)
                loss = F.mse_loss(mod(x), tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step == 120:
                    mod.diagnostics(reset=True)
            losses[nm] = float(loss.detach())
            touched[nm] = mod.diagnostics(reset=True)["touched_fraction"]
        hurt = losses["crippled"] > losses["honest"] * 1.02
        # The HONEST module's own usage after 120 steps at the arms' lr: the reachability half. A
        # task on which the honest selector still reads most of the pool cannot exhibit the
        # failure, so it cannot rank a fix for it.
        collapses = touched["honest"] < 0.90
        can_rank[task] = hurt and collapses
        print(f"task {task:<7}: honest {losses['honest']:.4f} vs one-row {losses['crippled']:.4f}"
              f"  -> collapse {'COSTS loss' if hurt else 'costs nothing'};"
              f" honest touched {touched['honest']:.4f}"
              f" -> failure {'REACHABLE' if collapses else 'does not occur'}"
              f"  => {'CAN rank' if can_rank[task] else 'cannot rank'}")
        assert hurt == want_hurt, (
            f"the {task} task changed character: collapse now "
            f"{'costs' if hurt else 'does not cost'} loss (honest {losses['honest']:.4f}, one-row "
            f"{losses['crippled']:.4f}), where it did the opposite when this file's numbers were "
            f"measured. Any ranking read off it before is void.")
        assert collapses == want_collapse, (
            f"the {task} task changed character: the honest selector now reads "
            f"{touched['honest']:.4f} of the pool, so the failure "
            f"{'is' if collapses else 'is not'} reachable, the opposite of when this file's "
            f"numbers were measured. Any ranking read off it before is void.")
    assert not any(can_rank.values()), (
        f"a task now satisfies both properties: {[t for t, v in can_rank.items() if v]}. That is "
        f"good news and it invalidates this docstring -- re-run the arm x selector-lr grid on it "
        f"and report the ranking, which no task here could give on 2026-09-05.")
    print(f"selftest OK -- no task can rank ({', '.join(can_rank)} all fail one property); "
          f"the decision belongs to the six real-data cells")


def _one_row_forward(self, x):
    """forward with the selector destroyed: every token reads row 0, top_k times.

    The known-answer negative world for "does this task need a distributed memory". Not a
    collapsed selector that TRAINED to concentrate -- one that cannot do anything else, so the
    comparison isolates the value of spreading from whatever else training changes.
    """
    B_, T_, d = x.shape
    h = self.n_mem(x)
    flat = torch.zeros(B_ * T_, self.top_k, dtype=torch.long, device=x.device)
    w = torch.full((B_ * T_, self.top_k), 1.0 / self.top_k, dtype=x.dtype, device=x.device)
    vals = self.values(flat)
    read = torch.einsum("nkd,nk->nd", vals, w).view(B_, T_, d)
    return self.out(F.silu(self.gate(h)) * read)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.02, help="the arms' mem_lr")
    ap.add_argument("--sel-lr", type=float, default=None,
                    help="separate lr for query+keys (default: one group, as train.py does today)")
    ap.add_argument("--task", default="zipf", choices=("zipf", "recall", "smooth"),
                    help="zipf is the only one that both reaches and penalises the collapse -- see make_task")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--arms", default="base,bn,l2")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    _, task_desc = make_task(a.task, 0)
    print(f"torch {torch.__version__}  ProductKeyMemory({SIDE*SIDE}, {D}, top_k={TOP_K})  "
          f"B*T={B*T} lr={a.lr} sel_lr={a.sel_lr} window={a.window} steps={a.steps}")
    print(f"task {a.task}: {task_desc}")
    print(f"uniform-draw prediction per window: "
          f"{uniform_touch_prediction(SIDE*SIDE, a.window*B*T*TOP_K):.4f}\n")
    final = {}
    for arm in a.arms.split(","):
        finals = []
        for s in range(a.seeds):
            rows = run(arm, a.steps, a.window, a.lr, seed=s, task=a.task, sel_lr=a.sel_lr)
            finals.append(rows[-1])
        final[arm] = finals
        print()
    print("final window, mean over seeds:")
    for arm, rows in final.items():
        n = len(rows)
        t = sum(r["touched"] for r in rows) / n
        e = sum(r["ent"] for r in rows) / n
        g = sum(r["gini"] for r in rows) / n
        l = sum(r["loss"] for r in rows) / n
        print(f"  {arm:<4} touched {t:.4f} of uniform {rows[0]['uniform_pred']:.4f} "
              f"({t/rows[0]['uniform_pred']:.4f})  ent {e:.3f}/{rows[0]['ent_max']:.3f}  "
              f"gini {g:.4f}  loss {l:.4f}   (n={n} seeds)")


if __name__ == "__main__":
    main()
