#!/usr/bin/env python3
# restartable: pure in-process wrapper + selftest. No files written, no card needed for --selftest.
"""N7 Stage A: run middle layers twice at inference and score it. No training.

    python3 eval/loop_wrapper.py --selftest                       # no card, no model
    python3 eval/loop_wrapper.py --ckpt ckpt_data_leg_206m_8b.pt --loop 4 7 --bench

THE RULING THIS IMPLEMENTS (6e, option 3; b0 read the same shape as its A1, "source count stays
25"). Under AttnRes there is no `x = b(x)`: model.py:_body carries a source LEDGER and each sublayer
reads `ar(done + partial)`, a softmax-weighted sum over every prior source. So "layers 4-7 run
twice" has three readings, and the ruled one is that the second visit's output must NOT lengthen
the ledger -- it adds into the source the first visit already put there. Source count unchanged, so
final_ar still sees 25 (24 sublayers + the embedding).

TWO PREMISES OF THE RULING'S WORDING ARE FALSE ON THIS CHECKPOINT, measured here, and the wrapper
implements the ruled SEMANTICS rather than its wording. Reported to 6e before any number was taken.

  1. "adds into the same `partial` accumulator" names a path this checkpoint never runs.
     ar_block_ends is {round((j+1)*n_sub/n_blocks)} at model.py:396, and attn_res_blocks defaults
     to 0 = Full (train.py:219, and every launch in EXPERIMENTS.md passes 0 explicitly). At
     layers 12 that is n_sub 24, n_blocks 24, ends {1..24} -- EVERY sublayer is a block end, so
     `partial` is flushed to `done` after every sublayer and is ALWAYS [] at the `ar()` read.
     A wrapper accumulating into `partial` would accumulate into a list that is empty by
     construction. The accumulator that exists here is the `done` ENTRY the first visit appended,
     so that is what the second visit adds into. Same arithmetic, live variable. Both cases are
     coded because ar_block_ends is a config, not a constant.
  2. "every trained ar()/final_ar weight still matches its input length" is not why option 3 is
     right, because AttnRes HAS NO PER-SOURCE WEIGHT. Its only parameters are g:(d,) and q:(d,)
     (model.py:257-259); the softmax runs OVER the source axis with nothing indexed by source, and
     it was called here with 3 and with 5 sources and returned the same shape both times. So
     option 1 (append) would not have needed new weights either -- it is a different experiment
     because it changes the depth-attention DISTRIBUTION every later layer reads, not because a
     weight vector runs out. Option 3 is still the ruled reading; only the reason changes.

THE BLOCK IS WHAT REPEATS, NOT THE SUBLAYER. "Layers 4-7 twice" means mixer,ffn,mixer,ffn per
looped block. Advancing one sublayer at a time and doubling it in place gives mixer,mixer,ffn,ffn,
which is a different network -- so the second visit replays the block's whole sublayer list, in
order, each sublayer merging into the slot its own first visit wrote.

RULE (a) IS VACUOUS ON THIS PATH -- a measurement about the code, not a choice, and 6e/b0/3b
concur, so the prereg states it as the definition and never as "(a) beat (b)". The spec pins KDA
state ("first visit's update discarded, second reads state_in and writes back") and MLA KV
("overwritten by the second visit"). Neither is reachable: model.py:125 calls chunk_kda with NO
initial_state and DISCARDS the returned state (`out, _`), a grep for state_in|initial_state|
output_final_state in model.py finds nothing, and the scoring path is one teacher-forced forward
with no KV slots. Each visit's scan therefore starts from zero state over its own input, which is
what (a) prescribes -- reached by the ABSENCE of state threading, not by code honouring the rule.
Anyone extending this to cached decoding must implement (a) explicitly; it is not inherited here.
"""

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def looped_body(model, x, cu=None, loop=(4, 7)):
    """model._body with blocks loop[0]..loop[1] visited twice, option 3.

    A REPLACEMENT for _body rather than a hook, because the ledger lives in local variables: there
    is no seam inside _body to patch. The unlooped path here must stay BITWISE equal to
    model._body -- main() asserts that on the real model before timing anything, because a drift
    between this copy and model.py turns the A/B into two implementations instead of one
    intervention.

    grad_ckpt IS HONOURED, and it has to be for Stage B rather than Stage A. Stage A scores under
    no_grad, where `self.grad_ckpt and self.training` is False and the branch is dead. Stage B
    TRAINS with the loop, and sft_math.py defaults --grad_ckpt True with the reason recorded at
    sft_math.py:110 -- FP8 e4m3 backward goes NaN without it. A wrapper that dropped the
    checkpoint would train an arm with no activation checkpointing under FP8 and report the NaN,
    or the memory failure, as something the LOOP did.
    """
    from model import Source  # noqa: PLC0415

    ckpt = model.grad_ckpt and model.training
    lo, hi = loop
    if not model.attn_res:
        raise SystemExit(
            "REFUSING: this wrapper implements the AttnRes source-ledger ruling (option 3) and "
            "this model has attn_res=False. On a plain residual stack 'run twice' is unambiguous "
            "and needs no ledger decision -- write that path separately rather than letting this "
            "one silently measure something else.")

    def run(norm, f, h):
        # AttnRes stays OUTSIDE the checkpoint, as in model.py:_body: only [B,T] logits go on the
        # tape, never an [B,T,D] stack of the values.
        fn = lambda t, norm=norm, f=f: f(norm(t))  # noqa: E731
        if ckpt:
            return torch.utils.checkpoint.checkpoint(fn, h, use_reentrant=False)
        return fn(h)

    done, partial, n = [Source.of(x)], [], 0
    for i, b in enumerate(model.blocks):
        subs = list(b.sublayers(cu))
        slots = []          # where each first-visit sublayer's output landed, in order
        for ar, norm, f in subs:
            h = ar(done + partial)
            out = run(norm, f, h)
            partial = [Source.of(partial[0].v + out if partial else out)]
            n += 1
            if n in model.ar_block_ends:
                done, partial = done + partial, []
                slots.append(len(done) - 1)     # an index into `done`
            else:
                slots.append(None)              # still open in `partial`
        if lo <= i <= hi:
            # THE SECOND VISIT. The whole block replays in sublayer order, each sublayer reading
            # the ledger as it stands and ADDING into the slot its own first visit wrote -- never
            # appending. `n` is NOT advanced: it indexes ar_block_ends, so advancing it would
            # repartition done/partial for every later sublayer, which is option 1 wearing
            # option 3's source count (count right, partition changed).
            for (ar, norm, f), slot in zip(subs, slots):
                h2 = ar(done + partial)
                out2 = run(norm, f, h2)
                if slot is None:
                    partial = [Source.of(partial[0].v + out2)]
                else:
                    done = done[:slot] + [Source.of(done[slot].v + out2)] + done[slot + 1:]
    return model.final_ar(done + partial)


def _selftest():
    """The ledger properties, on a stub with hand-derived arithmetic. No model, no card.

    The stub mirrors the REAL config in the two ways that matter: ar_block_ends = every sublayer
    (what attn_res_blocks=0 produces at any depth, and what this checkpoint runs), and TWO
    sublayers per block like the real Block. A one-sublayer stub cannot see the mixer,mixer,ffn,ffn
    ordering bug, and a sparse-ends stub would exercise a branch the run never takes.

    Each sublayer's output DEPENDS on its input (f(t) = t + k). A constant-output stub would make
    the second visit unable to propagate, so option 2 and option 3 would print the same number and
    the fixture would have no power to separate them.
    """
    class Src:
        def __init__(self, v):
            self.v = v

        @staticmethod
        def of(v):
            return Src(v)

    counts, order = [], []

    class AR:
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, sources):
            counts.append(len(sources))
            order.append(self.tag)
            return sum(s.v for s in sources)

    class Blk:
        def __init__(self, i, ks):
            self.i, self.ks = i, ks

        def sublayers(self, cu=None):
            return tuple((AR(f"b{self.i}s{j}"), (lambda t: t), (lambda t, k=k: t + k))
                         for j, k in enumerate(self.ks))

    class M:
        attn_res = True
        grad_ckpt = False       # Stage A scores under no_grad; Stage B flips both of these
        training = False

        def __init__(self, blocks):
            self.blocks = [Blk(i, ks) for i, ks in enumerate(blocks)]
            n_sub = sum(len(ks) for ks in blocks)
            self.ar_block_ends = set(range(1, n_sub + 1))   # attn_res_blocks=0 -> Full

        def final_ar(self, sources):
            counts.append(len(sources))
            order.append("final")
            return sum(s.v for s in sources)

    import model as real_model
    saved = real_model.Source
    real_model.Source = Src
    try:
        # ---- one sublayer per block: the ledger arithmetic, hand-derived ----
        # blocks emit +1, +10, +100; ends {1,2,3}; f(t) = t + k; x = 0.
        # UNLOOPED: h=0 -> out 1, done=[0,1]; h=1 -> out 11, done=[0,1,11];
        #           h=12 -> out 112, done=[0,1,11,112]; final_ar = 124.
        # LOOPED(1,1): first visit as above to done=[0,1,11]; second visit h2=12 -> out2 22,
        #           merged into done[2]: 11+22 = 33, done=[0,1,33]; then h=34 -> out 134;
        #           final_ar = 0+1+33+134 = 168 over FOUR sources.
        # Option 2 (overwrite) would give done=[0,1,22] -> 146. Option 1 (append) would give
        # done=[0,1,11,22,134] -> the same 168 over FIVE sources. So the value separates 3 from 2
        # and the source count separates 3 from 1; both assertions are needed.
        m = M([(1,), (10,), (100,)])
        counts.clear()
        base = looped_body(m, 0, loop=(9, 9))       # no block in range: the unlooped ledger
        base_counts = list(counts)
        assert base == 124, base
        assert base_counts == [1, 2, 3, 4], base_counts

        counts.clear()
        looped = looped_body(m, 0, loop=(1, 1))
        loop_counts = list(counts)
        assert looped == 168, (
            f"looped value {looped}, hand-derived option 3 is 168; option 2 (overwrite) gives 146")
        # THE LEDGER LENGTH IS UNTOUCHED: final_ar still sees 4, and the looped run makes exactly
        # one extra ar() call whose count is the first visit's +1 (the first visit flushed its own
        # output into `done` before the second visit reads -- that is "as the first visit left it").
        assert loop_counts[-1] == base_counts[-1] == 4, (base_counts, loop_counts)
        L = 1
        assert loop_counts[:L + 1] == base_counts[:L + 1], (base_counts, loop_counts)
        assert loop_counts[L + 2:] == base_counts[L + 1:], (
            f"looping changed a LATER call's source count: {loop_counts[L + 2:]} against "
            f"{base_counts[L + 1:]}. That is option 1 (appended sources), not option 3.")
        assert loop_counts[L + 1] == loop_counts[L] + 1, loop_counts

        # LOOPING NOTHING IS A NO-OP, which is what main() gates the real A/B on.
        assert looped_body(m, 0, loop=(-2, -1)) == base

        # ---- two sublayers per block: THE BLOCK REPEATS, NOT THE SUBLAYER ----
        m2 = M([(1, 2), (10, 20), (100, 200)])
        order.clear()
        looped_body(m2, 0, loop=(1, 1))
        assert order == ["b0s0", "b0s1", "b1s0", "b1s1", "b1s0", "b1s1",
                         "b2s0", "b2s1", "final"], order
        # The bug this pins: doubling each sublayer in place gives b1s0,b1s0,b1s1,b1s1 -- a
        # different network from running the block twice, and its loss/BPB would be a number for
        # an architecture nobody ruled on.
        assert order[2:6] != ["b1s0", "b1s0", "b1s1", "b1s1"]

        counts.clear()
        b2 = looped_body(m2, 0, loop=(9, 9))
        n_base = len(counts)
        base_final = counts[-1]
        counts.clear()
        l2 = looped_body(m2, 0, loop=(1, 2))
        assert len(counts) == n_base + 4, (n_base, len(counts))   # 2 blocks x 2 sublayers
        assert counts[-1] == base_final, (base_final, counts[-1])
        assert l2 != b2

        # ---- grad_ckpt: STAGE B TRAINS WITH THE LOOP, so the checkpoint branch must be live ----
        # sft_math.py defaults --grad_ckpt True and records why at :110 (FP8 e4m3 backward goes NaN
        # without it). A wrapper that dropped the checkpoint would train an arm with no activation
        # checkpointing and report the NaN, or the OOM, as something the LOOP did. Checked by
        # COUNTING checkpoint calls, not by reading the flag: the flag says what was intended and
        # the count says what ran.
        import torch.utils.checkpoint as tuc
        real_ckpt = tuc.checkpoint
        calls = []

        def counting_ckpt(fn, *args, **kw):
            calls.append(1)
            return fn(*args)

        tuc.checkpoint = counting_ckpt
        try:
            m4 = M([(1,), (10,), (100,)])
            m4.grad_ckpt, m4.training = True, True
            calls.clear()
            v_train = looped_body(m4, 0, loop=(1, 1))
            n_ckpt = len(calls)
            # 3 blocks + 1 replayed block = 4 sublayer executions, each one checkpointed.
            assert n_ckpt == 4, (
                f"{n_ckpt} checkpoint calls, expected 4 (3 sublayers + the second visit). The "
                f"second visit must be checkpointed too, or Stage B's looped arm keeps the whole "
                f"replayed block's activations while the unlooped arm does not.")
            # THE VALUE IS UNCHANGED by checkpointing -- it is a memory/compute trade, not a
            # different function. Same 168 as the eval path.
            assert v_train == 168, (v_train, "checkpointing changed the result")
            # grad_ckpt True but training False (eval on a train-configured model) checkpoints
            # NOTHING, matching model.py's `self.grad_ckpt and self.training`.
            m4.training = False
            calls.clear()
            assert looped_body(m4, 0, loop=(1, 1)) == 168
            assert not calls, f"{len(calls)} checkpoint calls under .eval(); model.py gates on training"
        finally:
            tuc.checkpoint = real_ckpt

        # attn_res=False REFUSES instead of quietly running an undefined loop.
        plain = M([(1,)])
        plain.attn_res = False
        try:
            looped_body(plain, 0, loop=(0, 0))
            raise AssertionError("attn_res=False was accepted")
        except SystemExit as e:
            assert "attn_res=False" in str(e), e

        # ---- patch_body: the seam the scorers are driven through ----
        # It must be an EXACT redirect (same value as calling looped_body directly), it must UNDO
        # to the original, and a second patch must refuse rather than nest -- a nested patch would
        # run three visits and report them as two.
        m3 = M([(1,), (10,), (100,)])
        m3._body = lambda x, cu=None: looped_body(m3, x, cu, (9, 9))   # stand-in for model._body
        unlooped_via_body = m3._body(0)
        assert unlooped_via_body == 124, unlooped_via_body
        undo = patch_body(m3, loop=(1, 1))
        assert m3._body(0) == 168, m3._body(0)
        try:
            patch_body(m3, loop=(1, 1))
            raise AssertionError("a second patch was accepted")
        except SystemExit as e:
            assert "already patched" in str(e), e
        undo()
        assert m3._body(0) == unlooped_via_body, (
            "undo() did not restore the original _body, so a control arm scored after the looped "
            "arm in the same process would silently carry the loop")
    finally:
        real_model.Source = saved

    print("loop_wrapper self-test OK: the looped value matches the hand-derived option-3 figure "
          "168 (option 2 gives 146) while final_ar still sees the same 4 sources (option 1 would "
          "give 5); every call after the looped block keeps its source count; the looped BLOCK "
          "replays as s0,s1,s0,s1 and not s0,s0,s1,s1; an out-of-range loop reproduces the "
          "unlooped value; grad_ckpt checkpoints all 4 sublayer executions including the second "
          "visit when training and none under eval, without changing the value; attn_res=False "
          "refuses; and patch_body redirects exactly, refuses to nest, and undoes back to the "
          "original _body")


def patch_body(model, loop=(4, 7)):
    """Redirect this model's _body through looped_body, and return an undo callable.

    THE SEAM EXISTS BECAUSE THE SCORERS CALL THE WHOLE MODEL. humaneval_bpb's OursModel.logprobs
    runs `self.model(x)` (eval/humaneval_bpb.py:88-93), and domain_loss goes through forward too,
    so there is nowhere to hand a looped hidden state in. Patching _body on the INSTANCE means the
    looped arm is scored by the very same scorer, the same tokenizer and the same byte divisor as
    the unlooped arm -- the alternative, a second scorer that loops, would report two
    implementations under one metric name.

    PER-INSTANCE, not on HybridLM: a class patch would silently apply to any other model an
    already-imported scorer loads in the same process, and the control arm is exactly such a model.
    """
    if getattr(model, "_loop_patched", None) is not None:
        raise SystemExit("REFUSING: this model's _body is already patched for loop "
                         f"{model._loop_patched}. Nesting the patch would run the loop twice over "
                         f"and report the result as one extra visit.")
    original = model._body
    model._body = lambda x, cu=None: looped_body(model, x, cu, loop)
    model._loop_patched = tuple(loop)

    def undo():
        model._body = original
        model._loop_patched = None

    return undo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--loop", nargs=2, type=int, default=[4, 7], metavar=("LO", "HI"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bench", action="store_true", help="ms/token, looped vs unlooped")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if not a.ckpt:
        ap.error("--ckpt required (or --selftest)")

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from loader import load_checkpoint  # noqa: PLC0415
    model, cfg = load_checkpoint(a.ckpt, device=a.device, dtype=torch.bfloat16)
    model.eval()

    # THE UNLOOPED ARM MUST BE model._body ITSELF, not this file's copy with an empty range, so the
    # A/B is one intervention. This asserts the copy reproduces it bitwise before any number.
    idx = torch.randint(0, cfg.vocab_real, (1, a.seq), device=a.device)
    with torch.no_grad():
        emb = model.tok(idx)
        ref = model._body(emb)
        mine = looped_body(model, emb, loop=(-2, -1))   # no block in range
        d = (ref - mine).abs().max().item()
    print(f"unlooped parity: max|diff| {d:.3e} (must be 0 -- this copy vs model._body)")
    if d != 0.0:
        sys.exit("REFUSING: this file's unlooped path does not reproduce model._body bitwise, so a "
                 "looped-vs-unlooped delta would mix the intervention with a code difference.")

    if a.bench:
        for label, fn in (("unlooped", lambda e: model._body(e)),
                          ("looped", lambda e: looped_body(model, e, loop=tuple(a.loop)))):
            with torch.no_grad():
                for _ in range(3):
                    fn(emb)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(a.iters):
                    fn(emb)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / a.iters / a.seq * 1e3
            print(f"{label:9} {ms:.4f} ms/token  (seq {a.seq}, {a.iters} iters, batch 1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
