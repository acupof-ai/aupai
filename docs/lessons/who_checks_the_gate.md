---
question: "who checks that the launch gate has no missing gate"
status: recorded
source: "e1, 2026-09-01; written alongside scripts/launch_gate.py, in answer to fb's question"
---

# Who checks the gate

fb asked the right question before I wrote a line: **`launch-gate` constructs a
set of conditions and then makes a universal claim over that set, so its truth
depends entirely on the completeness of a construction that nothing checks.**
That is the `selftests_are_gated` shape, and it is worth saying plainly that the
gate cannot fully escape it.

What follows is what I could and could not do about it.

## What is genuinely defended

**1. A gate that cannot fail is a selftest failure, not a silent pass.**
`selftest()` iterates `GATES` — the same list the runner iterates — and asserts
every entry has a broken world. Add a gate without one and the selftest raises
by name:

```
assert not ungated, "gate(s) with no broken world: ..."
```

So the failure mode "someone added a tenth gate and nobody made it fail" is
covered mechanically. This is the one real structural defence.

**2. Every broken world is a damaged copy of a real artifact.** Not a
hand-written fixture. Hand-written worlds share the check's assumptions, and
2026-09-01 produced two cases where a fixture and the code it tested believed
the same fiction. Each world here copies `data/`, `runs/`, `scripts/` and breaks
exactly one thing.

**3. A gate that raises is NO-GO, never a pass.** `run()` catches `Exception`
and converts it to NO-GO with the exception text. A crashing gate used to be the
most likely way for this file to report GO on nothing.

**4. Silence is not evidence.** The `harness check` gate asserts that check
lines were *produced* before concluding none failed. **My first version did not**
— it returned GO whenever no `[FAIL]` appeared, and the selftest caught it
reporting *"0 FAIL"* on a world where `harness.py` had been deleted. No output
contains no failures. Same shape as a monitor closing a row `ok` on log silence.

## What is not defended, stated plainly

**Nothing checks that the nine conditions are the right nine.** If the real
launch has a tenth prerequisite nobody thought of, `launch-gate` will print GO
and be wrong, and no amount of internal rigour detects it. The construction is
still a human artifact.

I considered three ways to close this and rejected all three:

| approach | why rejected |
|---|---|
| derive gates from a written checklist | moves the completeness problem into a document nobody executes — the current situation with extra steps |
| require every past launch incident to map to a gate | good in principle; the incident record is prose, so the mapping is a judgement, and a judgement encoded as a test is a test that passes by construction |
| have a second session enumerate conditions independently and diff | the strongest option, and it is a review, not a check — it cannot run at commit time |

**So the honest answer to "who checks the gate" is: a person, once, per launch,
and the gate's job is to make that reading short rather than to replace it.**

## What makes the human read tractable

Three properties, deliberate:

- **Nine conditions, each one paragraph.** Small enough to read in full before a
  launch. A gate list long enough to skim has failed at its actual job.
- **`UNKNOWN` is a distinct state from `NO-GO`.** Two gates return it today —
  card ownership and shape-test results — because those facts genuinely do not
  exist in any artifact yet. **Collapsing "I checked and it is wrong" into "I
  cannot check" is the error that makes a green board meaningless**, and the
  distinction is what tells a reader which gates still need a human.
- **Every NO-GO names the artifact it read.** Not "corpora failed" but
  "`code_rp1t`: no `data/corpus/code_rp1t`". A reader can verify the gate's claim
  faster than they can re-derive the condition.

## The residual risk, named

**`launch-gate` printing GO will feel like authority it does not have.** It
computes nine things from artifacts; it does not know whether nine is enough.
The sentence I would want attached to any GO it prints, and which is now in the
tool's own docstring:

> Nine conditions computed from artifacts. This is not a proof that the run is
> safe to start; it is a proof that these nine specific failure modes are not
> present.

That is weaker than "all gates green" and it is the true statement. The previous
version of the claim was stronger and lived in one person's memory.
