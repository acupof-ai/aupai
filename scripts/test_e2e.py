#!/usr/bin/env python3
"""Run the whole pipeline end to end on the shipped sample corpus.

Every stage has a unit test; the CHAIN never had one, which is how a pack whose
fingerprint could never equal any checkpoint's vocab_id shipped. This runs

    mix -> tokenize -> pretrain N steps -> checkpoint
        -> load it back -> SFT pack -> SFT steps -> generate

on the sample corpus and asserts the JOINS, not the stages.

E2E_GPU is REQUIRED and there is no CPU half. KDA is fla/Triton only, so without a
card this file could only re-check the mix and the vocabulary -- which `harness.py
check` already does -- and exit 0, which reads as "the chain works". It was wired
into CI in exactly that shape. A test whose green means nothing is worse than no
test, so the skip path is deleted rather than documented.

E2E_GPU is exported as CUDA_VISIBLE_DEVICES before torch is imported, so every
stage says `cuda:0` -- naming `cuda:1` directly raises an illegal memory access
under fla/Triton, whose kernels launch on the current device. It will not pick a
card on its own: the pod's cards are shared.

    E2E_GPU=7 python scripts/test_e2e.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

GPU = os.environ.get("E2E_GPU", "").strip()
assert GPU, "set E2E_GPU=<idx>. The pod's cards are shared, so this test will not pick one."
os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # before any torch import, incl. transitive ones
STEPS = int(os.environ.get("E2E_STEPS", "6"))
MIX = "data/mix_sample.json"
DOMAIN = "sample"


def run(cmd, env=None, cwd=ROOT, timeout=3600):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:], file=sys.stderr)
        raise AssertionError(f"FAILED: {' '.join(cmd)}")
    return p.stdout


def stage(n, msg):
    print(f"\n[{n}] {msg}", flush=True)


def rm(*paths):
    for p in paths:
        for q in (p, p + ".ep1"):  # train.py also drops an end-of-epoch copy
            if os.path.exists(q):
                os.remove(q)


def main():
    from tokenizers import Tokenizer

    from loader import vocab_fingerprint

    tmp = tempfile.mkdtemp(prefix="e2e_")
    try:
        stage(1, "the sample mix parses and names a domain that exists")
        mix_path = os.path.join(ROOT, MIX)
        mix = json.load(open(mix_path, encoding="utf-8"))
        doms = list(mix["domains"])
        assert doms == [DOMAIN], doms
        shards = os.listdir(os.path.join(ROOT, "data", "corpus", DOMAIN))
        n_shard = len([s for s in shards if s.endswith(".jsonl")])
        assert n_shard, f"data/corpus/{DOMAIN} has no shards"
        # train.py has no knob for the token-cache location (_domain_cache_path writes
        # /data00/tokens_<domain>[_fone].pt), so the domain NAME is the only thing keeping
        # this six-step run out of a multi-day pretrain's cache file.
        v3 = os.path.join(ROOT, "data", "mix_v3.json")
        if os.path.exists(v3):
            real = list(json.load(open(v3, encoding="utf-8"))["domains"])
            assert DOMAIN not in real, (
                f"domain {DOMAIN!r} is also in mix_v3.json; both would share the token cache "
                f"/data00/tokens_{DOMAIN}.pt and this test would clobber a pretrain's."
            )
        print(f"    {n_shard} shards, domains={doms}")

        stage(2, "the vocabulary the pipeline will use, and its fingerprint")
        tok_path = os.path.join(ROOT, "data", "tokenizer.json")
        assert os.path.exists(tok_path), "data/tokenizer.json missing"
        fp = vocab_fingerprint(Tokenizer.from_file(tok_path))
        print(f"    fingerprint {fp}")

        stage(3, f"pretrain {STEPS} steps on the sample mix -> a checkpoint")
        name = "e2e_tmp"
        ckpt = os.path.join(ROOT, f"ckpt_{name}.pt")
        rm(ckpt)
        # --max_steps, not --steps: train.py has no `steps` flag, and no --layers/--d/--heads
        # either, so this runs the real 12x1024 architecture and --seq/--batch make it small.
        # --val_every cannot be passed: train.py applies an override only `if v`, so 0 reads
        # as unset. The end-of-epoch validation runs and is part of the run.
        out = run(
            [
                sys.executable,
                "train.py",
                "--name",
                name,
                "--mix",
                MIX,
                "--max_steps",
                str(STEPS),
                "--batch",
                "1",
                "--seq",
                "512",
                "--warmup",
                "2",
            ]
        )
        assert os.path.exists(ckpt), f"train.py produced no {ckpt}\n{out[-1500:]}"
        print(f"    {os.path.getsize(ckpt) / 1e6:.1f} MB")

        stage(4, "the pretrain actually took optimizer steps")
        import torch

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        # Stages 4-9 all passed on a 206M random init with the training loop stubbed to zero
        # iterations -- the chain test could not see that no training happened, which is the
        # whole class of silent failure it exists for. train.py seeds with Cfg.seed before
        # building HybridLM, so a fresh init here is bitwise identical to the run's step 0;
        # measured cos is -0.028 after six real steps and exactly 1.0 after none.
        from train import Cfg, HybridLM

        for k, v in ck["cfg"].items():
            setattr(Cfg, k, v)
        torch.manual_seed(Cfg.seed)
        init = HybridLM(Cfg).state_dict()["tok.weight"].float()
        trained = ck["model"]["tok.weight"].float()
        cos0 = torch.nn.functional.cosine_similarity(init.flatten(), trained.flatten(), dim=0).item()
        assert cos0 < 0.9, f"cos(fresh init, checkpoint) = {cos0:.4f}: the pretrain ran no steps"
        print(
            f"    cos(fresh init, checkpoint) = {cos0:.4f}, ‖Δ‖/‖init‖ = {(trained - init).norm() / init.norm():.2f}"
        )

        stage(5, "THE JOIN: the checkpoint records the vocabulary it actually saw")
        assert ck.get("vocab_id") == fp, (
            f"checkpoint vocab_id {ck.get('vocab_id')} != tokenizer {fp}. This is the join that "
            "has never been tested end to end, and a mismatch here scores as noise rather than raising."
        )
        print(f"    vocab_id {ck['vocab_id']} == tokenizer fingerprint")

        stage(6, "loader reads it back and builds the model from ck['cfg']")
        from loader import load_checkpoint

        model, cfg = load_checkpoint(ckpt, device="cpu")
        # Provenance, not particular numbers: every field must come from the checkpoint's
        # own dict, not from today's Cfg defaults.
        for k, v in ck["cfg"].items():
            if k == "grad_ckpt":
                continue  # loader forces it off for inference
            assert getattr(cfg, k) == v, f"cfg.{k} = {getattr(cfg, k)!r}, checkpoint says {v!r}"
        n_par = sum(p.numel() for p in model.parameters())
        print(f"    {n_par / 1e6:.1f}M params, d={cfg.d} layers={cfg.layers} seq={cfg.seq} from the ckpt")
        del model

        stage(7, "pack an SFT set; its fingerprint must equal the checkpoint's")
        sft_jsonl = os.path.join(tmp, "sft.jsonl")
        with open(sft_jsonl, "w", encoding="utf-8") as f:
            for i in range(64):
                f.write(
                    json.dumps({"instruction": f"{i} + {i} = ", "output": str(2 * i)}, ensure_ascii=False)
                    + "\n"
                )
        pack = os.path.join(tmp, "sft.pt")
        run([sys.executable, "prepare_sft_math.py", "--sources", sft_jsonl, "--out", pack])
        pk = torch.load(pack, map_location="cpu", weights_only=False)
        pack_fp = pk.get("vocab") if isinstance(pk, dict) else None
        assert pack_fp == fp, (
            f"pack fingerprint {pack_fp} != {fp}. prepare_sft.py once hashed str(id) before the "
            "token, so a pack's fingerprint could NEVER equal a checkpoint's vocab_id and the "
            "check that exists to catch a wrong vocabulary could never fire."
        )
        print(f"    pack vocab {pack_fp} == checkpoint vocab_id, {len(pk['input_ids'])} rows")

        stage(8, "SFT resumes the pretrained weights (not a fresh init)")
        sft_ckpt = os.path.join(ROOT, "ckpt_e2e_tmp_sft.pt")
        rm(sft_ckpt)
        run(
            [
                sys.executable,
                "sft_math.py",
                "--resume",
                ckpt,
                "--sft_path",
                pack,
                "--out",
                sft_ckpt,
                "--epochs",
                "1",
                "--batch",
                "1",
            ]
        )
        assert os.path.exists(sft_ckpt), "sft_math.py produced no checkpoint"
        sk = torch.load(sft_ckpt, map_location="cpu", weights_only=False)
        before = ck["model"]["tok.weight"].float()
        after = sk["model"]["tok.weight"].float()
        assert before.shape == after.shape
        # A fresh init is uncorrelated with the pretrained embedding. This is the join that
        # "loss 4.77 instead of 1.28, with nothing raising" was made of.
        cos = torch.nn.functional.cosine_similarity(before.flatten(), after.flatten(), dim=0).item()
        assert cos > 0.9, (
            f"SFT embeddings barely resemble the pretrained ones (cos={cos:.3f}); it reinitialised"
        )
        print(f"    embedding cos(pretrained, sft) = {cos:.4f}")

        stage(9, "the finished model generates")
        # bf16, as eval/math_hard.py does: flash_attn_func in the SWA blocks raises
        # "FlashAttention only support fp16 and bf16 data type" on the fp32 weights a
        # checkpoint loads as.
        model, cfg = load_checkpoint(sft_ckpt, device="cuda:0")
        model = model.to(torch.bfloat16)
        tok = Tokenizer.from_file(tok_path)
        ids = tok.encode("1 + 1 = ", add_special_tokens=False).ids
        with torch.no_grad():
            for _ in range(4):
                lg = model(torch.tensor([ids], device="cuda:0"))
                lg = lg[0] if isinstance(lg, tuple) else lg
                # argmax over all-NaN logits returns 0 silently, and FP8 e4m3 backward NaN
                # without grad_ckpt is a documented failure here (sft_math.py:56). Without
                # this the stage proves only that forward() did not raise.
                assert torch.isfinite(lg).all(), "logits contain NaN/Inf"
                ids.append(int(lg[0, -1].argmax()))
        print(f"    generated: {tok.decode(ids)!r}")

        stage(10, "THE JOIN: the SFT checkpoint still says which vocabulary it speaks")
        # Without vocab_id, load_tokenizer only warns and loads whatever data/tokenizer.json
        # happens to be -- a file rebuilt in place. It has to survive SFT, or the artifact
        # people actually evaluate is the one that cannot be checked.
        assert sk.get("vocab_id") == fp, (
            f"the SFT checkpoint's vocab_id is {sk.get('vocab_id')!r}, not {fp}. train.py writes "
            "vocab_id into every checkpoint it saves and sft_math.py's torch.save does not, so the "
            "identity is present on the base and gone from the model that gets evaluated."
        )
        print(f"    vocab_id {sk['vocab_id']} survived SFT")

        print(f"\ne2e OK: mix -> tokenize -> pretrain -> ckpt -> pack -> sft -> generate, {STEPS} steps")
        return 0
    finally:
        # On the success path only, this leaked ckpt_e2e_tmp.pt (824MB) plus its .ep1
        # (1.78GB, it carries optimizer state) into the shared repo root on every failure.
        rm(os.path.join(ROOT, "ckpt_e2e_tmp.pt"), os.path.join(ROOT, "ckpt_e2e_tmp_sft.pt"))
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
