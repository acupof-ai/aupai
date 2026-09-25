#!/usr/bin/env python3
"""Real-model CPU smoke for the code RL trainer: MoE + CSA2 + CED forward/backward.

algorithms/test_rl_code.py proves the arithmetic with a char-level stub model; this
proves the REAL model path that the stub cannot reach. A small d256/L4/h4 CED
HybridLM with MoE, CSA2 and the 2/2 encoder-decoder split is built exactly the way
scripts/test_sft_ced_cpu.py builds one, and one GSPO code-RL step is driven through
the trainer's own functions:

    generate (real model, eval/no_grad)
      -> seq_logprob old policy
      -> gspo_code_loss (mixed 1/0 rewards, mean-only advantage, KL anchor)
      -> backward (CSA2 + MoE grouped + CED cross-boundary grads)
      -> Muon step on 2-D params + fp32 AdamW writeback on 1-D params

The reward is not executed here: the call/stdin sandbox contract is covered with the
real executor in test_rl_code.py; this gate is about the model graph, so rewards are
the fixed mixed group that makes the gradient live. No data, no GPU, single thread.

    python algorithms/test_rl_code_ced_smoke.py
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

VOCAB, EOS = 512, 1
N, MAXNEW, PLEN = 4, 8, 12


def build():
    import train

    C = train.Cfg
    C.d = 256
    C.heads = 4
    C.layers = 4
    C.ffn_hidden = 256
    C.vocab = VOCAB
    C.vocab_real = VOCAB
    C.num_id = VOCAB - 1
    C.seq = PLEN + MAXNEW
    C.csa = True
    C.csa2 = True
    C.ced = True
    C.ced_enc_layers = 2
    C.ced_kc_norm = True
    C.rope_dims = 64
    C.n_swa_only_layers = 0
    C.attn_res = False
    C.attn_every = 1
    C.doc_mask = True
    C.compile = False
    C.grad_ckpt = False
    C.fone = False
    C.mix = None
    C.moe_experts = 8
    C.moe_top_k = 3
    C.moe_shared = 1
    C.moe_expert_ffn = 64  # (3+1)*64 == ffn_hidden 256: active-width parity
    C.moe_layers = "0-3"
    C.fp8 = False
    torch.manual_seed(42)
    return train.HybridLM(C), C


def main():
    torch.set_num_threads(1)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import copy

    from rl_code_trainer import _adam_step_fp32, build_optimizer, gspo_code_loss
    from rlvr_generate import generate
    from rlvr_trainer import seq_logprob
    from train import Muon

    model, cfg = build()
    n_params = sum(p.numel() for p in model.parameters())

    # The frozen KL reference: the only duplicated weights in the real trainer.
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    torch.manual_seed(0)
    prompt = torch.randint(10, VOCAB - 1, (PLEN,)).tolist()

    # GENERATE on the training model itself (eval + no_grad), as the trainer does.
    model.eval()
    with torch.no_grad():
        gens = generate(model, prompt, N, MAXNEW, temperature=0.8, top_p=0.95, device="cpu")
    assert all(len(g) >= 1 for g in gens), "empty generation from the real model"
    with torch.no_grad():
        old_lp, gen_t, mask = seq_logprob(model, prompt, gens, N, MAXNEW, False, "cpu", False)
    model.train()

    # Mixed group [1,1,0,0]: nonzero mean-subtracted advantage -> live gradient.
    rewards = [1.0, 1.0, 0.0, 0.0]
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    loss = gspo_code_loss(model, ref, prompt, gens, rewards, N, MAXNEW, False, "cpu", False,
                          clip_eps=0.2, kl_beta=0.02, old_lp=old_lp)
    assert torch.isfinite(loss), f"non-finite loss {loss}"
    loss.backward()
    grads_seen = sum(1 for p in model.parameters() if p.grad is not None
                     and p.grad.abs().sum() > 0)
    assert grads_seen > 0, "backward produced no gradients"

    # The same optimizer split the trainer uses: Muon on 2-D, fp32 AdamW on the rest.
    # If this branch's train.Muon predates de's SR flags, build_optimizer falls back.
    opts = build_optimizer(list(model.parameters()), lr=1e-4)
    assert any(isinstance(o, Muon) for o in opts), "no Muon group for a CED+MoE model"

    def sr(x):
        return x.to(torch.bfloat16)

    for opt in opts:
        if isinstance(opt, torch.optim.AdamW):
            for g in opt.param_groups:
                for p in g["params"]:
                    if p.grad is not None:
                        _adam_step_fp32(p, opt.state[p], g, sr)
        else:
            opt.step()

    # The step actually moved parameters (a zero-move round is the failure the SR design
    # exists to prevent; here at lr=1e-4 even the bf16 cast moves visibly).
    moved = [(n, (p.detach().float() - w0[n].float()).abs().max().item())
             for n, p in model.named_parameters() if p.ndim >= 1]
    n_moved = sum(1 for _, d in moved if d > 0)
    assert n_moved > 0, "optimizer step moved no parameter"

    # A second forward/backward after the step must be finite: the update did not break
    # the graph or a weight (the cheap stand-in for the training loop repeating).
    model.eval()
    with torch.no_grad():
        old2, _, _ = seq_logprob(model, prompt, gens, N, MAXNEW, False, "cpu", False)
    model.train()
    loss2 = gspo_code_loss(model, ref, prompt, gens, rewards, N, MAXNEW, False, "cpu", False,
                           clip_eps=0.2, kl_beta=0.02, old_lp=old2)
    assert torch.isfinite(loss2), f"post-step loss non-finite {loss2}"

    print(f"rl_code CED smoke OK: real CED 2/2 + CSA2 + MoE(8) model "
          f"({n_params/1e6:.2f}M params), {grads_seen} tensors with gradient, "
          f"Muon+AdamW step moved {n_moved} params; loss {loss.item():.4f} -> {loss2.item():.4f}")


if __name__ == "__main__":
    main()
