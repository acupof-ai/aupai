#!/usr/bin/env python3
"""Known-answer tests for the CODE RL trainer (algorithms/rl_code_trainer.py).

    python3 algorithms/test_rl_code.py

Three things, each one a property a glance at a green log would not establish:

  1. advantage hand-computed: mean-subtracted, NO std division, constant group 0.
  2. reward known answer through the REAL sandbox (seatbelt locally, sandbox_exec
     on CI): a correct continuation scores 1, a wrong one 0, in both the call-style
     (pytest) and stdin contracts; ALLOW_UNISOLATED=1 must refuse on the rollout path.
  3. one CPU small-shape round through the same functions main() chains:
     generate -> reconstruct -> reward -> gspo loss -> backward -> optimizer step,
     with a bf16 1-D parameter receiving a sub-ULP update that must survive writeback.

No data, no GPU.
"""

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

# The rollout path must refuse this; make sure it is not inherited from the shell.
os.environ.pop("ALLOW_UNISOLATED", None)

from rl_code_trainer import (  # noqa: E402
    IMPL_HEADER,
    _adam_step_fp32,
    filter_rows_by_solution_len,
    gspo_code_loss,
    group_advantage,
    load_code_pool,
    program_source,
    score_row,
    solution_body,
)
from code_reward import score as call_score  # noqa: E402
from code_reward import score_stdin  # noqa: E402
from rlvr_generate import generate  # noqa: E402
from rlvr_trainer import seq_logprob  # noqa: E402

FAILS = []
SKIPPED = []


def check(name, cond, detail=""):
    print(("ok  " if cond else "FAIL") + " " + name + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(f"{name}: {detail}")


def isolation_available():
    """True only when generated code can run under a real sandbox on THIS host.

    GitHub runners are non-root Linux with no bwrap/nsjail/firejail, so
    detect_level() is rlimits_only there and code_reward refuses (correctly). The
    reward-execution assertions then SKIP and are COUNTED rather than pretending to
    run -- the same NEEDS_DATA discipline as test_stdin_reward_known. They run for
    real on a developer Mac (seatbelt) and on root/pod Linux (sandbox_exec).
    """
    from isolate import detect_level
    return detect_level() != "rlimits_only"


# ---------------------------------------------------------------- 1. advantage

def test_advantage():
    # Hand-computed, binary rewards [1,1,0,0]: mean 0.5, so adv = [.5,.5,-.5,-.5],
    # with NO std rescaling. The old /std formula divides this group by 0.5 -> +/-1.
    adv = group_advantage([1.0, 1.0, 0.0, 0.0], normalize_std=False)
    want = torch.tensor([0.5, 0.5, -0.5, -0.5])
    check("advantage mixed group is mean-subtracted, not std-divided",
          torch.allclose(adv, want), f"{adv.tolist()}")

    # A constant group is exactly zero (the degenerate-group contract).
    for r in ([1.0] * 8, [0.0] * 8):
        a = group_advantage(r, normalize_std=False)
        check(f"constant group {r[0]} -> all-zero advantage", torch.count_nonzero(a) == 0,
              f"{a.tolist()}")

    # The std-dividing path still exists for the math trainer. torch .std() is
    # Bessel-corrected: for [1,1,0,0] it is 0.577, so the values are +/-0.866.
    a_std = group_advantage([1.0, 1.0, 0.0, 0.0], normalize_std=True)
    check("math path keeps std normalisation (sample-std 0.577 -> +/-0.866)",
          torch.allclose(a_std, torch.tensor([0.8660254, 0.8660254, -0.8660254, -0.8660254]),
                         atol=1e-6), f"{a_std.tolist()}")


# ---------------------------------------------------------------- 2. reward

CALL_PROMPT = 'def add(a, b):\n    """\n    Return the sum of a and b.\n    """\n'
CALL_TESTS = (
    "from solution import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "    assert add(-1, 1) == 0\n"
)
CALL_ROW = {"kind": "call", "prompt": CALL_PROMPT, "tests": CALL_TESTS, "entry": "add"}

STDIN_PROMPT = (
    '"""\nRead one integer n and print twice its value.\n\nInput: a single integer.\n'
    'Output: 2*n.\n\nRead integers/strings from standard input in the input format '
    'described in the docstring and print the required answer to standard output.\n"""\n'
)
STDIN_CASES = [{"input": "3\n", "output": "6\n"}, {"input": "-4\n", "output": "-8\n"}]
STDIN_ROW = {"kind": "stdin", "prompt": STDIN_PROMPT, "cases": STDIN_CASES}


def test_reward():
    good = program_source(CALL_PROMPT, "    return a + b\n")
    bad = program_source(CALL_PROMPT, "    return a - b\n")
    check("call prompt+correct continuation parses", bool(good), repr(good))
    good_s = program_source(STDIN_PROMPT, "\nn = int(input())\nprint(n * 2)\n")
    bad_s = program_source(STDIN_PROMPT, "\nn = int(input())\nprint(n)\n")
    check("stdin prompt+continuation parses", bool(good_s), repr(good_s))

    if not isolation_available():
        msg = ("no process isolation on this host (code reward correctly refuses); the "
               "call/stdin 1-vs-0 and ALLOW_UNISOLATED assertions SKIP here and run on a "
               "seatbelt Mac or root/pod Linux")
        SKIPPED.append(msg)
        print("SKIP " + msg)
        return

    # call-style: correct continuation 1, wrong 0, via the real isolated pytest run.
    # Use score()/score_stdin() (the evidence-returning layer), not the scalar
    # reward_fn wrappers, so a CI sandbox failure reports WHY (rc/reason/stderr/level)
    # instead of a bare 0 -- a wrong answer and a sandbox that cannot start must be
    # distinguishable from the test output.
    rc_good = call_score(good, CALL_TESTS, timeout=30)
    check("call correct solution rewards 1", rc_good["reward"] == 1.0,
          f"{rc_good['reason']} rc={rc_good['rc']} level={rc_good['level']} "
          f"err={rc_good.get('stderr','')[-300:]}")
    rc_bad = call_score(bad, CALL_TESTS, timeout=30)
    check("call wrong solution rewards 0", rc_bad["reward"] == 0.0,
          f"{rc_bad['reason']} rc={rc_bad['rc']}")
    # stdin: whole script piped the case input.
    rs_good = score_stdin(good_s, STDIN_CASES, timeout=10)
    check("stdin correct script rewards 1", rs_good["reward"] == 1.0,
          f"{rs_good['reason']} rc={rs_good['rc']} level={rs_good['level']} "
          f"out={rs_good.get('stdout','')[-120:]} err={rs_good.get('stderr','')[-300:]}")
    rs_bad = score_stdin(bad_s, STDIN_CASES, timeout=10)
    check("stdin wrong script rewards 0", rs_bad["reward"] == 0.0,
          f"{rs_bad['reason']} rc={rs_bad['rc']}")

    # Non-code continuation (prose) reconstructs to empty -> 0, never an error.
    check("prose continuation scores 0", score_row(CALL_ROW, "the answer is five") == 0.0)

    # THE ROLLOUT PATH MUST REFUSE ALLOW_UNISOLATED=1 (1e order 2026-09-25).
    os.environ["ALLOW_UNISOLATED"] = "1"
    try:
        refused = False
        try:
            score_row(CALL_ROW, good)
        except RuntimeError as e:
            refused = "ALLOW_UNISOLATED" in str(e)
        check("reward refuses ALLOW_UNISOLATED=1 on the rollout path", refused)
    finally:
        os.environ.pop("ALLOW_UNISOLATED", None)


# ---------------------------------------------------------------- 3. full round

class CharVocab:
    """Minimal char-level tokenizer for the smoke: char <-> id, eos=1."""

    def __init__(self, alphabet):
        self.itos = ["<pad>", "<eos>"] + sorted(set(alphabet))
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.eos = 1

    def encode(self, text):
        class E:
            ids = [self.stoi[c] for c in text if c in self.stoi]
        return E()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.itos[i] for i in ids if i < len(self.itos)
                       and not (skip_special_tokens and i == self.eos))


class ScriptedModel(nn.Module):
    """Emits a fixed continuation: at generation position t the target token's
    logit dominates, so sampling is effectively greedy at any temperature. The
    trainable table shifts every logit and supplies a gradient path; bias1d is a
    1-D parameter exercising the AdamW/SR branch."""

    def __init__(self, vocab, prompt_len, target_ids):
        super().__init__()
        self.vocab = vocab
        self.prompt_len = prompt_len
        self.register_buffer("target", torch.tensor(target_ids))
        self.table = nn.Parameter(torch.zeros(len(vocab.itos)))
        self.bias1d = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T) - self.prompt_len
        want = self.target[pos.clamp(min=0, max=len(self.target) - 1)].view(1, T)
        logits = torch.full((B, T, len(self.vocab.itos)), -1e3)
        logits.scatter_(-1, want.expand(B, T).unsqueeze(-1), 1e3)
        return logits + self.table.view(1, 1, -1), None


def test_full_round():
    # Two completions for the stdin doubling problem, char-vocab shared.
    good_code = "\nn = int(input())\nprint(n * 2)\n"
    bad_code = "\nn = int(input())\nprint(n)\n"
    text = STDIN_PROMPT + good_code + bad_code
    vocab = CharVocab(text + chr(0))
    eos = vocab.eos

    def ids(s):
        return vocab.encode(STDIN_PROMPT + s).ids

    good_ids, bad_ids = ids(good_code), ids(bad_code)

    torch.manual_seed(0)
    prompt_ids = vocab.encode(STDIN_PROMPT).ids
    # Model scripted to emit the GOOD sequence; the BAD one is handed to the loss
    # directly as a second rollout (rewards differ within the group, so grad is live).
    model = ScriptedModel(vocab, len(prompt_ids), good_ids + [eos])
    ref = ScriptedModel(vocab, len(prompt_ids), good_ids + [eos])
    for p in ref.parameters():
        p.requires_grad = False

    # GENERATE really runs (sampling is near-greedy from the 1e3 logit gap).
    gen = generate(model, prompt_ids, 1, max_new=len(good_ids) + 2, temperature=0.8,
                   top_p=0.95, device="cpu")
    decoded = vocab.decode(gen[0])
    check("generate emitted the scripted good continuation",
          good_code.strip() in decoded, repr(decoded[:60]))

    gens = [good_ids, bad_ids]
    if isolation_available():
        rewards = [score_row(STDIN_ROW, program_source(STDIN_PROMPT, vocab.decode(g)))
                   for g in gens]
        check("round rewards are [1, 0] (mixed group)", rewards == [1.0, 0.0], str(rewards))
    else:
        # No sandbox on this host: the executor path SKIPS, but the model/loss graph still
        # has to run, so use the binary rewards the sandbox would have produced (the two
        # continuations are a known passing/failing pair). Counted as a skip, not a pass.
        SKIPPED.append("full-round reward execution (no isolation host); fixed [1,0] used")
        rewards = [1.0, 0.0]
        print("SKIP full-round sandbox reward execution; driving the graph with fixed [1,0]")

    with torch.no_grad():
        old_lp, _, _ = seq_logprob(model, prompt_ids, gens, 2,
                                   max(len(g) for g in gens), False, "cpu", False)
    # Perturb the policy away from the rollout policy so the ratio is a real number.
    with torch.no_grad():
        model.table.add_(0.05)
    loss = gspo_code_loss(model, ref, prompt_ids, gens, rewards, 2,
                          max(len(g) for g in gens), False, "cpu", False,
                          clip_eps=0.2, kl_beta=0.02, old_lp=old_lp)
    check("loss is finite and scalar", torch.isfinite(loss).item() and loss.dim() == 0,
          str(loss.item()))
    loss.backward()
    g_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                     for p in model.parameters())
    check("backward produced gradients", g_has_grad)

    # THE BF16 SMALL-UPDATE CASE on the 1-D branch: the fp32 AdamW update is computed
    # exactly (lr=1e-6, and near zero bf16 itself resolves ~1e-9 steps), so this verifies
    # the arithmetic; the survival of a SUB-ULP-at-magnitude update is the SR unbiasedness
    # test below (round-nearest would truncate 1+2^-10 to 1 every time).
    p = nn.Parameter(torch.tensor([0.0], dtype=torch.bfloat16))
    p.grad = torch.tensor([1.0], dtype=torch.bfloat16)
    st = {}
    group = {"lr": 1e-6, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0}
    w0 = p.detach().clone()
    _adam_step_fp32(p, st, group, lambda x: x.to(torch.bfloat16))
    moved = (p.float() - w0.float()).abs().item()
    # First Adam step applies exactly lr=1e-6 in fp32; the bf16 cast of 1e-6 is
    # 9.984e-7. The point is it moved at all (a plain bf16 in-place optimizer absorbs
    # sub-ULP updates), not that a bf16 tensor holds 1e-6 exactly.
    check("bf16 1-D param moved by the ~1e-6 AdamW update", 0 < moved < 2e-6, f"moved {moved}")


def test_sr_unbiased():
    """The trainer's writeback uses the shared sr_cast.StochasticRounder and it is
    unbiased at a known sub-ULP offset. The SR arithmetic itself is gated in
    scripts/test_sr_cast.py; this pins the TRAINER's wiring (the same rounder Muon
    and the AdamW fp32 path share) and its expectation.

    At magnitude 1 bf16 ULP is 2^-7; 1 + 0.25*ULP rounds UP with probability 0.25,
    so the sample mean is unbiased and round-to-nearest (always 1) fails the check.
    """
    from sr_cast import StochasticRounder

    # Wiring + nontriviality, not a re-derivation of the SR arithmetic (that expectation
    # is gated by scripts/test_sr_cast.py:test_expectation_unbiased, 400 reps). Pin that
    # the trainer's shared StochasticRounder is the official cast and actually ROUNDS a
    # sub-grid value: choose an interior point (0.35 of the bf16 grid step), confirm the
    # measured up-rate matches the official frac, and that round-to-nearest would freeze
    # it (so the stochastic move is what is under test).
    from sr_cast import _bf16_neighbors, stochastic_round_bf16

    lo0, up0, _ = _bf16_neighbors(torch.tensor([0.61]))
    step = (up0 - lo0)[0]
    target = (lo0[0] + 0.35 * step).to(torch.float32)
    x = target.expand(60_000).contiguous()
    _, _, frac = _bf16_neighbors(x[:1])
    f = float(frac[0])
    assert 0.05 < f < 0.95, f"fixture point not interior: {f}"
    # the same seeded generator through the trainer rounder and the official function
    r1 = StochasticRounder(seed=77).round(x.clone())
    g2 = torch.Generator(device="cpu").manual_seed(77)
    r2 = stochastic_round_bf16(x.clone(), g2)
    out = r1.float()
    frac_up = (out == up0[0]).float().mean().item()
    check("trainer SR up-rate matches the official grid fraction",
          abs(frac_up - f) < 0.01, f"got {frac_up:.3f} vs frac {f:.3f}")
    check("trainer rounder is the official stochastic cast (same seed -> same bytes)",
          torch.equal(r1, r2))
    check("trainer SR moves a value round-to-nearest would freeze",
          frac_up > 0.05 and set(torch.unique(out).tolist()) <= {lo0[0].item(), up0[0].item()})


def test_pool_loader_shape(tmp_dir=None):
    # load_code_pool skips stats/quarantine and reads both kinds; validated on a fixture
    # so it does not need the multi-GB real pool.
    import tempfile

    d = tempfile.mkdtemp(prefix="rlcode_pool.")
    with open(os.path.join(d, "rl_code_apps.jsonl"), "w") as f:
        f.write('{"kind":"stdin","prompt":"p","cases":[]}\n')
    with open(os.path.join(d, "rl_code_x_nondet.jsonl"), "w") as f:
        f.write('{"kind":"stdin","prompt":"QUARANTINED"}\n')
    with open(os.path.join(d, "rl_code_taco_stats.json"), "w") as f:
        f.write("{}\n")
    rows = load_code_pool(d)
    check("pool loader keeps pool rows, skips nondet quarantine and stats",
          len(rows) == 1 and rows[0]["prompt"] == "p", str(len(rows)))


def test_length_filter():
    # The filter must measure the solution BODY (the fixed prepended header stripped),
    # use each mode's own cap, and keep/drop without mutating the rows.
    long = "x" * 300
    rows = [
        {"kind": "stdin", "impl": IMPL_HEADER + "n=1\n", "prompt": "s"},   # body tiny -> keep
        {"kind": "stdin", "impl": IMPL_HEADER + long, "prompt": "s"},      # body 300 > 562? no
        {"kind": "call", "impl": IMPL_HEADER + "def f():\n    return 1", "prompt": "c"},
    ]
    # force a clear drop: call cap is smaller than a long body
    rows.append({"kind": "call", "impl": IMPL_HEADER + long, "prompt": "c2"})
    kept, dropped = filter_rows_by_solution_len(rows, len, call_cap=25, stdin_cap=562)
    check("header stripped before measuring", solution_body(rows[0]) == "n=1\n")
    check("length filter keeps short call/stdin, drops the long call row",
          len(kept) == 3 and len(dropped) == 1 and dropped[0]["prompt"] == "c2",
          f"{len(kept)}/{len(dropped)}")
    # A row WITHOUT the header is measured whole (no silent strip).
    k2, _ = filter_rows_by_solution_len([{"impl": "print(1)"}], len, 20, 562)
    check("headerless row measured whole", len(k2) == 1)


def test_shared_net_gate():
    """Known answer for the bwrap shared-net safety gate.

    When bwrap cannot build an isolated net namespace, run() must RAISE by default
    (untrusted RL code never silently shares the pod's host network); with
    AUPAI_SANDBOX_ALLOW_SHARED_NET=1 it runs with --share-net and reports
    isolates.net=False. Host-independent: force the bwrap level and a False netns
    probe, and stub the actual Popen so no real bwrap is needed.
    """
    import isolate
    from isolate import SharedNetNotPermitted

    class FakeP:
        returncode = 0

        def __init__(self, *a, **k):
            self.argv = a[0] if a else k.get("args")

        def communicate(self, input=None, timeout=None):
            return b"", b""

    orig_detect, orig_netns, orig_popen, orig_which = (
        isolate.detect_level, isolate._bwrap_netns_ok, isolate.subprocess.Popen,
        isolate.shutil.which)
    orig_env = dict(os.environ)
    captured = {}

    def fake_popen(argv, **k):
        captured["argv"] = argv
        return FakeP()

    try:
        isolate.detect_level = lambda: "bwrap"
        isolate._bwrap_netns_ok = lambda: False
        isolate.subprocess.Popen = fake_popen
        isolate.shutil.which = lambda name: "/usr/bin/bwrap" if name == "bwrap" else orig_which(name)

        os.environ.pop("AUPAI_SANDBOX_ALLOW_SHARED_NET", None)
        raised = False
        try:
            isolate.run("print(1)", level="bwrap")
        except SharedNetNotPermitted:
            raised = True
        check("bwrap without netns RAISES when shared-net override is unset", raised)

        os.environ["AUPAI_SANDBOX_ALLOW_SHARED_NET"] = "1"
        r = isolate.run("print(1)", level="bwrap")
        check("override lets bwrap run with --share-net", r["rc"] == 0 and
              "--share-net" in captured["argv"], str(captured.get("argv")))
        check("shared-net result honestly reports isolates.net=False",
              r["isolates"]["net"] is False, str(r["isolates"]))
    finally:
        isolate.detect_level = orig_detect
        isolate._bwrap_netns_ok = orig_netns
        isolate.subprocess.Popen = orig_popen
        isolate.shutil.which = orig_which
        os.environ.clear()
        os.environ.update(orig_env)


def test_rounder_seed_rank_independent():
    """Every DDP rank's SR rounder must use the SAME seed.

    DDP all-reduces gradients but never synchronizes weights, so a per-rank SR seed
    rounds the same update differently on each replica and they random-walk apart with
    nothing reconverging them; rank 0 would save a model no other rank trained. A
    rounder built "as rank 3" must be byte-identical in its draws to the rank-0 one,
    and a rounder that seeds by rank (the bug) must fail this.
    """
    from rl_code_trainer import RL_SR_SEED, build_rounder
    from sr_cast import StochasticRounder

    x = torch.randn(4096, dtype=torch.float32)
    r0 = build_rounder()                       # what main() builds on rank 0
    r3 = build_rounder()                       # the same call on rank 3
    out0, out3 = r0.round(x.clone()), r3.round(x.clone())
    check("code-RL rounder seed is identical across ranks", torch.equal(out0, out3))
    check("build_rounder uses the fixed shared seed",
          r0.seed == RL_SR_SEED and r3.seed == RL_SR_SEED,
          f"{r0.seed} {r3.seed}")

    # The per-rank-seed regression must be caught: StochasticRounder(seed, rank=3)
    # seeds differently and therefore draws differently.
    bad = StochasticRounder(seed=RL_SR_SEED, rank=3)
    check("a rank-offset seed genuinely diverges (the guard has discriminating power)",
          not torch.equal(bad.round(x.clone()), out0))


def main():
    test_advantage()
    test_reward()
    test_full_round()
    test_sr_unbiased()
    test_rounder_seed_rank_independent()
    test_pool_loader_shape()
    test_length_filter()
    test_shared_net_gate()
    if FAILS:
        print(f"\n{len(FAILS)} FAIL:")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print(f"\nrl_code trainer tests OK: advantage hand-computed, generate->reward->loss->step "
          f"round on CPU, SR unbiasedness{', call+stdin reward 1/0 through the real sandbox' if not SKIPPED else ''}")
    for s in SKIPPED:
        print(f"  SKIPPED (counted): {s}")


if __name__ == "__main__":
    main()
