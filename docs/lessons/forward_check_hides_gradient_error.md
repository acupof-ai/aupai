---
question: Why did a forward comparison stay green while the gradient it was standing in for was 7.6% wrong?
status: measured
source: algorithms/attnres_fused.py self-check; model.py:244 (Source.of), :269-273; b0-review-7899ea1
---

# A forward check cannot stand in for a gradient check

Building the fused AttnRes node, the first comparison against the real
`model.AttnRes` read:

```
forward  rel 1.71e-07     ← agrees
dV       rel 1.08e-01     ← 10% wrong
```

The algebra was not wrong. Isolating it:

```
scale detached : dV rel 0.00e+00
scale live     : dV rel 7.57e-02
```

`Source.scale` is `rms_scale(v)` (`model.py:244`) — **a function of the same
`v`**. So `v` reaches the output by two routes: through the mixing, and through
its own scale into the logits. The node owns one of them and returns `dscale`
so autograd can chain the other. My harness passed `scale` detached, which
silently deleted the second route.

## What makes it dangerous

**The forward is identical in both worlds.** Detaching a tensor changes no
value — it changes only what the graph records. So a check that compares
outputs agrees to 1.5e-07 whether or not the gradient path exists, and a suite
built on forward parity would ship a 7.6% gradient error with every light
green. Nothing in the passing run reports the route it did not exercise.

The failure is also silent downstream: a 7.6% gradient error does not crash,
does not NaN, and does not obviously diverge — it trains, slightly wrong.

## Two independent paths to the same rule

b0 reached "the gate must be on `dV`, against autograd's **total**, subtracting
nothing" by reading `known_answer()` and finding it asserted only the forward
(`b0-review-7899ea1`). I reached it by walking into an actual wrong
implementation. **A rule derived from code structure and a rule derived from a
live failure are separate evidence**; either alone is a hypothesis about where
the gate belongs, and together they are a measurement.

## The contract had two gradients; there are four

The design named `dV` and `dlogit`. Deriving the node turned up two more:

```
dgq     = Σ_i Σ_bt (dlogit_i · scale_i) · v_i      verified 4.3e-15
dscale_i = dlogit_i · <v_i, gq>                     verified 1e-12
```

Both ride the same `[B,T,D]` traversal as `dV`, so they cost no extra pass —
but `dscale` is the one that closes the second route above. **A contract that
enumerates gradients can be incomplete in a way that reads as complete**, and
the omission surfaced only when the implementation was checked against the
module rather than against the contract.

## Rule

**Never let a forward comparison stand in for a gradient check.** Detach
changes the graph without changing a single value, so output parity is blind to
it by construction. When a node is wired into a module, check its gradients
against *that module* — not against the expression the design wrote down, which
is where the missing route was missing in the first place.

Corollary for calling conventions: if a node takes a tensor that is derived
from another of its inputs, **the live/detached distinction is part of the
contract**, and belongs in the docstring next to the shapes.
