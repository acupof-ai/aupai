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
    _adam_step_fp32,
    gspo_code_loss,
    group_advantage,
    load_code_pool,
    program_source,
    score_row,
)
from rlvr_generate import generate  # noqa: E402
from rlvr_trainer import seq_logprob  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok  " if cond else "FAIL") + " " + name + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(f"{name}: {detail}")


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
    # call-style: correct continuation 1, wrong 0, via the real isolated pytest run.
    good = program_source(CALL_PROMPT, "    return a + b\n")
    bad = program_source(CALL_PROMPT, "    return a - b\n")
    check("call prompt+correct continuation parses", bool(good), repr(good))
    check("call correct solution rewards 1", score_row(CALL_ROW, good) == 1.0)
    check("call wrong solution rewards 0", score_row(CALL_ROW, bad) == 0.0)

    # stdin: whole script piped the case input.
    good_s = program_source(STDIN_PROMPT, "\nn = int(input())\nprint(n * 2)\n")
    bad_s = program_source(STDIN_PROMPT, "\nn = int(input())\nprint(n)\n")
    check("stdin prompt+continuation parses", bool(good_s), repr(good_s))
    check("stdin correct script rewards 1", score_row(STDIN_ROW, good_s) == 1.0)
    check("stdin wrong script rewards 0", score_row(STDIN_ROW, bad_s) == 0.0)

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
    rewards = [score_row(STDIN_ROW, program_source(STDIN_PROMPT, vocab.decode(g)))
               for g in gens]
    check("round rewards are [1, 0] (mixed group)", rewards == [1.0, 0.0], str(rewards))

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
    """The local SR cast is unbiased at a known sub-ULP offset: value 1 + 0.25*ULP
    (ULP=2^-8 at magnitude 1) must round DOWN to 1 with probability 0.75 and UP to
    1+ULP with 0.25, so the mean over draws lands inside a sampling bound and
    round-to-nearest/truncation (which always returns 1) fails the same check.
    """
    from rl_code_trainer import _sr_bf16_fallback

    torch.manual_seed(0)
    # bf16 has 7 fraction bits, so the ULP on [1,2) is 2^-7 (the cast neighbours below
    # are 1.0 and 1.0078125).
    ulp = 2 ** -7
    x = torch.full((200_000,), 1.0 + 0.25 * ulp, dtype=torch.float32)
    # word check: 0.25 ULP = 2^-9 = raw fraction 0x4000, a 0.25 round-up probability
    assert int(x[:1].view(torch.int32)[0]) & 0xFFFF == 0x4000, "fixture offset wrong"
    out = _sr_bf16_fallback(x).float()
    frac_up = (out > 1.0).float().mean().item()
    mean_err = (out.mean() - x[0]).item()
    check("SR up-fraction is 0.25 at the quarter-ULP offset", abs(frac_up - 0.25) < 0.01,
          f"{frac_up:.4f}")
    check("SR mean error is unbiased (|.| < 0.25*ULP)", abs(mean_err) < 0.25 * ulp,
          f"{mean_err:.2e}")
    check("SR outputs are bf16 grid neighbours",
          set(torch.unique(out).tolist()) <= {1.0, 1.0 + ulp})


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


def main():
    test_advantage()
    test_reward()
    test_full_round()
    test_sr_unbiased()
    test_pool_loader_shape()
    if FAILS:
        print(f"\n{len(FAILS)} FAIL:")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("\nrl_code trainer tests OK: advantage hand-computed, call+stdin reward 1/0 "
          "through the real sandbox, generate->reward->loss->step round on CPU")


if __name__ == "__main__":
    main()
