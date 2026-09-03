#!/usr/bin/env python3
"""Is the document-isolation leak in chunk_kda itself, or in model.py's plumbing around it?

WHY THIS SPLIT IS THE WHOLE QUESTION. scripts/n7c_pack_isolation.py established that block 0's KDA
output for a document inside a packed row differs from the same document alone by 48.88 against that
layer's own 0.93 tolerance, largest at each document's start and decaying into it, with document 0
exactly clean and no dependence on position in the row. That says a state crosses the boundary. It
does NOT say which code lets it, and the two candidates need opposite fixes:

  the kernel      chunk_kda receives cu_seqlens (model.py:131) and does not honour it
  the plumbing    chunk_kda is correct and model.py hands it inputs already contaminated

AND I HAVE TO RETRACT THE REASON I RULED OUT THE PLUMBING. I called the short_conv refuted because
it can only touch positions 0-2 of a document while the measured contamination ran to position 9 and
beyond. That inference is wrong. The short_conv produces k/v/g; if positions 0-2 of those are
contaminated, chunk_kda writes the bad values INTO the recurrent state at those positions and then
carries the state forward with the forget gate. A 3-position input error therefore produces exactly
the decaying-over-many-positions output I used as evidence against it. The two candidates are not
distinguishable from the output profile at all -- only by feeding the kernel clean inputs, which is
what this script does.

THE TEST. Build q, k, v, g, beta directly as random tensors in the shapes model.py:115-124 produces,
so nothing upstream of the kernel is involved and the short_conv never runs. Call chunk_kda through
the EXACT call at model.py:125-142 -- every kwarg, same dtypes, same chunk_size -- once on two
documents packed with cu_seqlens, once on each document alone. If the packed result matches the solo
results, the kernel isolates and the leak is in model.py's plumbing. If it does not, the kernel is
ignoring cu_seqlens and the installed version is the thing to name.

EXTRA PROBES 6e asked for, each testing a specific way cu_seqlens could be silently ignored:
  initial_state=None passed explicitly   -- a default that is not None would carry state in
  reversed document order                -- if doc B's output changes when it follows a different
                                            doc, the dependence is real and not a coincidence
  a single document padded vs alone      -- separates "cu is ignored" from "any longer row differs"

USAGE
    CUDA_VISIBLE_DEVICES=4 python3 scripts/n8_kda_kernel_repro.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LENS = [170, 232]   # two document lengths, both even so chunk_kda's alignment is not in play
H, HD = 8, 64       # heads and head dim; overridden below from the real config if it loads
CHUNK = 64


def main():
    import torch  # noqa: PLC0415

    import model as M  # noqa: PLC0415

    if M.chunk_kda is None:
        raise SystemExit("REFUSING: fla.ops.kda.chunk_kda did not import; nothing to test.")

    # NAME THE VERSION FIRST. If the kernel leaks, the version is the finding, and reading it after
    # the fact from a different shell is how a fact ends up citing a package that was not the one
    # measured.
    import importlib.metadata as md  # noqa: PLC0415
    vers = {}
    for pkg in ("fla", "flash-linear-attention", "flash_kda", "flash-kda", "triton", "torch"):
        try:
            vers[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    import fla  # noqa: PLC0415
    print("versions: " + ", ".join(f"{k}={v}" for k, v in vers.items()))
    print(f"fla at {os.path.dirname(fla.__file__)}")
    print(f"chunk_kda from {M.chunk_kda.__module__}")

    dev = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(0)  # a fixed seed: the same tensors every run, so a rerun is comparable

    # THE SHAPES model.py:115-124 BUILDS, and the flattened varlen layout it uses at :122-123.
    # A_log and dt_bias come from a real DeltaRecurrence so the parameter shapes and init are the
    # ones the kernel actually receives, not shapes I guessed.
    class Cfg:
        d, heads, chunk_size = H * HD, H, CHUNK
    mixer = M.DeltaRecurrence(Cfg()).to(dev).to(dt)

    n_max = max(LENS)

    def call(segs, explicit_initial_state=False):
        """chunk_kda over the documents named in `segs`, in that order, via model.py's exact call.

        segs is a list of DOCUMENT IDS, and each document's length comes from LENS[id] -- not from a
        parallel list. The first version took lengths and an order separately and indexed one by the
        other, which happened to be right for the forward case and wrong for the reversed one: a
        comparison of two different token sets would have printed as a leak.
        """
        cu = torch.tensor([0, *torch.tensor([LENS[i] for i in segs]).cumsum(0).tolist()],
                          dtype=torch.int32, device=dev)
        total = int(cu[-1])
        q, k, v, g, beta = (torch.cat([bank[name][i][:, :LENS[i]] for i in segs], dim=1)
                            for name in ("q", "k", "v", "g", "beta"))
        assert q.shape[1] == total, (q.shape, total)
        kw = dict(
            g=g, beta=beta, cu_seqlens=cu, A_log=mixer.A_log, dt_bias=mixer.dt_bias,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0,
            state_v_first=True, disable_recompute=True, chunk_size=CHUNK,
        )
        if explicit_initial_state:
            kw["initial_state"] = None
        with torch.no_grad():
            out, _ = M.chunk_kda(q, k, v, **kw)
        return out[0].float()

    # ONE BANK OF RANDOM TENSORS, sliced per call, so "the same document" is bit-identical whether it
    # is scored alone or packed. Fresh randoms per call would make every comparison a comparison of
    # different inputs, which would show a difference with nothing wrong.
    bank = {name: [torch.randn(1, n_max, H, HD, device=dev, dtype=dt) for _ in range(2)]
            for name in ("q", "k", "v", "g")}
    bank["beta"] = [torch.randn(1, n_max, H, device=dev, dtype=dt) for _ in range(2)]

    solo = [call([i]) for i in range(2)]
    packed = call([0, 1])

    print(f"\n== two documents of {LENS[0]} and {LENS[1]} tokens, H={H} HD={HD} chunk_size={CHUNK}, "
          f"dtype {dt}")
    print(f"  {'doc':>3s} {'tokens':>7s} {'maxdiff':>9s} {'pos0-2':>8s} {'pos3-9':>8s} "
          f"{'pos10+':>8s}   reading")
    tol = 0.05
    leaks = False
    at = 0
    for i, n in enumerate(LENS):
        d = (packed[at:at + n] - solo[i]).abs().amax(dim=-1)
        seg = (d.max().item(), d[:3].max().item(), d[3:10].max().item(), d[10:].max().item())
        leaks = leaks or (i > 0 and seg[0] > tol)
        print(f"  {i:3d} {n:7d} {seg[0]:9.4f} {seg[1]:8.4f} {seg[2]:8.4f} {seg[3]:8.4f}   "
              f"{'DIFFERS' if seg[0] > tol else 'agrees'}")
        at += n
    print("  document 0 must agree exactly (nothing precedes it); document 1 is the test.")

    # REVERSED ORDER: if document 1's output depends on what precedes it, putting it first must
    # change it. This is the control that turns a diff into a demonstrated dependence.
    rev = call([1, 0])
    d_rev = (rev[:LENS[1]] - solo[1]).abs().max().item()
    print(f"\n  document 1 placed FIRST instead of second: maxdiff vs alone {d_rev:.4f} "
          f"({'agrees, so its earlier diff came from what preceded it' if d_rev <= tol else 'still differs -- the dependence is not on the predecessor'})")

    # EXPLICIT initial_state=None: rules out a non-None default carrying state in.
    p2 = call([0, 1], explicit_initial_state=True)
    d_is = (p2 - packed).abs().max().item()
    print(f"  initial_state=None passed explicitly: maxdiff vs the default call {d_is:.4f} "
          f"({'identical, so the default is already None' if d_is == 0 else 'DIFFERENT -- the default is not None'})")

    print("\n== VERDICT")
    if leaks:
        print(f"  THE KERNEL LEAKS. chunk_kda received cu_seqlens {list(range(0, 1))}-style "
              f"boundaries and document 1's output still depends on document 0, with no short_conv "
              f"and no model.py plumbing anywhere in this script -- the inputs are random tensors. "
              f"The finding is the installed version above.")
    else:
        print("  THE KERNEL ISOLATES. With random inputs and cu_seqlens, document 1's output is "
              "identical packed or alone, so chunk_kda honours the boundaries and the leak measured "
              "by n7c_pack_isolation.py is in model.py's plumbing between the reshape and this call. "
              "The candidates, in order: the short_conv at model.py:109-113, which left-pads the "
              "WHOLE ROW and never sees cu, so positions 0-2 of k/v/g in every document after the "
              "first are convolved with the previous document's last tokens -- and the kernel then "
              "writes those bad values into the recurrent state and decays them forward, which "
              "reproduces the decaying profile that I WRONGLY read as evidence against the "
              "short_conv; and the gate path at model.py:119-121, which computes g and beta from x "
              "with no boundary handling either.")

        # THE ATTENTION TWIN, b0's request via 6e. b0 reproduced the invariant on a second checkpoint
        # and data source and saw block 7 (MLA) read "uniform" rather than decaying. That alone proves
        # nothing -- block 7's inputs are already contaminated everywhere by block 0, so a uniform
        # profile there is what a clean layer fed dirty inputs looks like. The question it raises is
        # whether the varlen ATTENTION also fails to honour cu, and that needs the same random-input
        # control the KDA kernel got: nothing upstream, so nothing to inherit.
        print("\n== attention twin: flash_attn_varlen_func with cu, packed vs alone, random inputs")
        if not M.HAS_FA:
            print("  SKIPPED: HAS_FA is False.")
        else:
            aq, ak, av = (torch.randn(sum(LENS), H, HD, device=dev, dtype=dt) for _ in range(3))
            cu_a = torch.tensor([0, LENS[0], sum(LENS)], dtype=torch.int32, device=dev)

            def attn(q_, k_, v_, cu_, max_len):
                """model.py:191-192's call, EVERY kwarg included.

                The first version of this control omitted max_seqlen_q/max_seqlen_k and reported the
                attention leaking by 4.33. That was my bug, not a finding: model.py passes both, and
                on the .cute path the fourth POSITIONAL is qv rather than cu_seqlens_q
                (model.py:43-44), so a call assembled from memory rather than copied from the site is
                how a control ends up measuring something else. max_seqlen must also be the packed
                row's own max, not a constant shared between the packed and solo calls, or the two
                differ by their tiling instead of by their masking.
                """
                out = M.flash_attn_varlen_func(q_, k_, v_, cu_seqlens_q=cu_, cu_seqlens_k=cu_,
                                               max_seqlen_q=max_len, max_seqlen_k=max_len,
                                               causal=True)
                return (out[0] if isinstance(out, tuple) else out).float()

            with torch.no_grad():
                # max_seqlen is the LONGEST DOCUMENT in each call, which is what model.py:192 means
                # by T for a packed row: the per-segment maximum, not the row's total length.
                packed_a = attn(aq, ak, av, cu_a, max(LENS))
                solo_a = [attn(aq[:LENS[0]], ak[:LENS[0]], av[:LENS[0]],
                               torch.tensor([0, LENS[0]], dtype=torch.int32, device=dev), LENS[0]),
                          attn(aq[LENS[0]:], ak[LENS[0]:], av[LENS[0]:],
                               torch.tensor([0, LENS[1]], dtype=torch.int32, device=dev), LENS[1])]
            # THE OFFSET MUST ADVANCE. It did not in the first version, so document 1 was compared
            # against packed_a[0:232] -- document 0's rows -- and the 4.328125 that printed was two
            # different token sets, not a leak. An SDPA reference settled it: packed and solo are both
            # 0.005710 from it and 0.000000 from each other. Third time this family of bug has cost a
            # false finding here, so the loop now derives the span from cu itself.
            attn_leaks = False
            for i, n in enumerate(LENS):
                lo, hi = int(cu_a[i]), int(cu_a[i + 1])
                assert hi - lo == n, (i, lo, hi, n)
                d = (packed_a[lo:hi] - solo_a[i]).abs().max().item()
                attn_leaks = attn_leaks or (i > 0 and d > 0.05)
                print(f"  doc {i} ({n} tokens): max|diff| packed vs alone {d:.6f}   "
                      f"{'DIFFERS' if d > 0.05 else 'agrees'}")
            print("  " + ("THE ATTENTION ALSO LEAKS -- a second independent site, and the fix at "
                          "model.py:109-113 would not cover it. Before reporting that: check the "
                          "span arithmetic and the kwargs against model.py:191-192, because a "
                          "non-advancing offset already produced a false 4.328125 here."
                          if attn_leaks else
                          "THE ATTENTION ISOLATES. So block 7's uniform profile in b0's run is a "
                          "clean layer fed inputs block 0 already contaminated, not a second site: "
                          "flash_attn_varlen_func honours cu with nothing upstream to inherit."))

        # NAME THE LINE. The kernel is clean, so the contamination enters through one of the tensors
        # model.py builds before the call. Run the REAL DeltaRecurrence on two packed documents and
        # on each alone, and diff each intermediate at document 1's positions: whichever tensor first
        # differs is the site. This is a measurement of model.py, not a change to it.
        print("\n== which tensor model.py hands the kernel is already contaminated")
        x_bank = [torch.randn(1, n, mixer.qkv.in_features, device=dev, dtype=dt) for n in LENS]
        def intermediates(x):
            """Recompute model.py:102-124 verbatim on x and return each named tensor."""
            B, T, D = x.shape
            w, K = mixer.short_conv.weight, mixer.short_conv.kernel_size[0]
            h = torch.nn.functional.pad(x.transpose(1, 2), (K - 1, 0))
            y = h[:, :, :T] * w[:, 0, 0].unsqueeze(-1)
            for j in range(1, K):
                y = y + h[:, :, j : j + T] * w[:, 0, j].unsqueeze(-1)
            conv = torch.nn.functional.silu(
                (y + mixer.short_conv.bias.unsqueeze(-1)).transpose(1, 2))
            q_, k_, v_ = mixer.qkv(conv).chunk(3, dim=-1)
            gb = mixer.gb(x)
            return {"short_conv out": conv, "q": q_, "k": k_, "v": v_,
                    "g (from gb, gate path)": gb[..., :D],
                    "beta (from gb)": gb[..., D : D + mixer.h]}

        with torch.no_grad():
            packed_i = intermediates(torch.cat(x_bank, dim=1))
            solo_i = [intermediates(xb) for xb in x_bank]
        start = LENS[0]
        print(f"  {'tensor':24s} {'maxdiff':>9s} {'pos0-2':>8s} {'pos3-9':>8s} {'pos10+':>8s}   "
              f"reading")
        for name in packed_i:
            a = packed_i[name][0, start:start + LENS[1]].float()
            b = solo_i[1][name][0].float()
            d = (a - b).abs().amax(dim=-1)
            print(f"  {name:24s} {d.max().item():9.4f} {d[:3].max().item():8.4f} "
                  f"{d[3:10].max().item():8.4f} {d[10:].max().item():8.4f}   "
                  f"{'CONTAMINATED' if d.max().item() > 1e-3 else 'clean'}")
        print("  A tensor contaminated ONLY at positions 0-2 is the k=4 short_conv reading across\n"
              "  the boundary (model.py:109-113). The kernel then writes those values into the\n"
              "  recurrent state and decays them forward, which is how a 3-position input error\n"
              "  becomes the 48.88 output difference over many positions that n7c_pack_isolation.py\n"
              "  measured. gb is a per-position Linear on x, so g and beta must be clean -- if they\n"
              "  are not, the mechanism is something else and this table says so.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
