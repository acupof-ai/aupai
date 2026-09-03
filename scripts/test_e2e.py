#!/usr/bin/env python3
# restartable: a 6-step smoke test whose writes all land in a mktemp dir plus two
# ckpt_e2e_tmp* files it removes itself; an interrupt costs one rerun of ~6 steps. It
# REFUSES to touch a shared token cache (stage 2), so an interrupt cannot leave a
# pretrain's cache half-written -- which is the state this check exists to prevent.
"""Run the whole pipeline end to end on the shipped sample corpus, asserting the JOINS, not the stages.

    mix -> tokenize -> pretrain N steps -> checkpoint
        -> load it back -> SFT pack -> SFT steps -> generate

E2E_GPU is REQUIRED and there is no CPU half: without a card this could only re-check the mix and
the vocabulary, which `harness.py check` already does -- a test whose green means nothing is worse
than no test.

E2E_GPU is exported as CUDA_VISIBLE_DEVICES before torch is imported, so every stage says `cuda:0`
-- naming `cuda:1` directly raises an illegal memory access under fla/Triton, whose kernels launch
on the current device. It will not pick a card on its own: the pod's cards are shared.

    E2E_GPU=7 python scripts/test_e2e.py
"""

import glob
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
# The shape to build, as train.py flags. Default: whatever Cfg says, which is what this
# test has always run. E2E_LAYERS=32 runs the joins at the 493.6M shape -- the launch
# gate asks whether the pipeline holds at the shape being launched, and a run at
# Cfg.layers answers that only while Cfg.layers happens to be that shape. The flag is
# --dim, not --d: run_ddp.sh's args pass through torchrun's parser first, where prefix
# matching makes --d ambiguous (754b624).
E2E_LAYERS = os.environ.get("E2E_LAYERS", "").strip()
E2E_SHAPE = ["--layers", E2E_LAYERS] if E2E_LAYERS else []
# The mix, same discipline as E2E_LAYERS. Default: the sample mix, which is what this test
# has always run -- and running it is what let the 20B launch die at step 0 on a
# KeyError('content') that e2e could not reach: data/corpus/sample holds zero holdout
# slices, so the shape the test pinned was right and the DATA was a different question.
# A test that holds fixed exactly the variable that breaks answers a question the gate is
# not asking. Set E2E_MIX=data/mix_500m.json to run the joins on the mix being launched.
E2E_MIX = os.environ.get("E2E_MIX", "").strip()
MIX = E2E_MIX or "data/mix_sample.json"
# The recipe knobs train.py requires (ead2d2b) that this test was previously taking from Cfg
# without naming them. NOT new choices and NOT recipe_provenance.json's values: read from Cfg
# at 32a7a4a, the last commit before ead2d2b, and confirmed identical at HEAD, so the test's
# behaviour is unchanged. --layers 12 in particular keeps the "real 12x1024 architecture" the
# E2E_LAYERS comment above promises; recipe_provenance's 32 belongs to the 500M run and would
# silently turn an 8-step smoke test into a 500M one. E2E_SHAPE is appended AFTER these, so
# E2E_LAYERS still overrides the depth (argparse takes the last occurrence).
E2E_RECIPE = [
    "--dim", "1024", "--layers", "12", "--heads", "8", "--ffn_hidden", "3072",
    "--accum", "1", "--lr_scale", "1.0", "--warmdown", "0.65", "--anneal_frac", "0.1",
    "--no-grad_ckpt", "--save_every", "1000",
]
if not os.path.exists(os.path.join(ROOT, MIX)):
    raise SystemExit(f"E2E_MIX={MIX!r} does not exist. A missing mix is a refusal, not a "
                     f"fall back to the sample mix: falling back is how a sample-mix pass "
                     f"came to be read as clearing the launch mix.")
_mix_obj = json.load(open(os.path.join(ROOT, MIX), encoding="utf-8"))
DOMAINS = list(_mix_obj["domains"])
# The cache-collision guard below keys on the domain name, so it needs every domain the
# mix names, not one. The sample mix has exactly one and the hardcoded name hid that.
DOMAIN = DOMAINS[0]


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


def _read(path):
    """Stripped contents, or None when absent -- the stamp files _would_rebuild compares."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _would_rebuild(dom, vocab_id):
    """(bool, why) -- would _domain_seqs REWRITE this domain's cache, or reuse it?

    THE REAL PREDICATE, replacing "does the name collide with the default mix" (6e's
    ruling, 2026-09-03). The name test is a proxy, and it is wrong in both directions:
    it refused a mix whose nine domains would all REUSE (measured: every .vocab stamp
    equal to the live vocab id, every .srcfp equal to the live source fingerprint, so
    the run writes nothing), and it would pass a differently-named domain whose cache
    is stale and therefore WOULD be rewritten -- which is the actual hazard.

    The conjuncts are train.py's own (train.py:1411-1419), read from that module rather
    than restated here: cache exists, shards exist, same vocab, same source fingerprint,
    same sample seed, and the cache newer than every shard. A copy of a six-part
    condition is a second thing to keep correct, and the mtime clause is the one a
    hand-written copy drops -- I nearly did.
    """
    import train

    cache = train._domain_cache_path(dom)
    ddir = os.path.join(ROOT, "data", "corpus", dom)
    shards = sorted(glob.glob(os.path.join(ddir, "*.jsonl")))
    if not os.path.exists(cache):
        return True, f"{dom}: no cache at {cache} -- this run would CREATE it"
    if not shards:
        return True, f"{dom}: no shards, so _domain_seqs cannot judge freshness"
    stamp, srcfp, seedfp = cache + ".vocab", cache + ".srcfp", cache + ".seed"
    got = _read(stamp)
    if got != vocab_id:
        return True, (f"{dom}: cache vocab {(got or 'unstamped')[:16]!r} != live "
                      f"{vocab_id[:16]!r} -- would retokenize")
    live_fp = train._corpus_fp(ddir)
    if _read(srcfp) != live_fp:
        return True, f"{dom}: cache .srcfp != the source's fingerprint -- would retokenize"
    want_seed = str(train._sample_seed())
    was = _read(seedfp)
    if was != want_seed:
        return True, (f"{dom}: cache shuffled at sample_seed {was or 'unstamped'}, now "
                      f"{want_seed} -- would retokenize")
    newest = max(os.path.getmtime(p) for p in shards)
    if os.path.getmtime(cache) < newest:
        return True, f"{dom}: a shard is newer than the cache -- would retokenize"
    return False, f"{dom}: reuse"


def main():
    from tokenizers import Tokenizer

    from loader import vocab_fingerprint

    tmp = tempfile.mkdtemp(prefix="e2e_")
    try:
        stage(1, f"the mix parses and every domain it names exists ({MIX})")
        mix_path = os.path.join(ROOT, MIX)
        mix = json.load(open(mix_path, encoding="utf-8"))
        doms = list(mix["domains"])
        assert doms == DOMAINS, (doms, DOMAINS)
        n_shard = 0
        for dom in doms:
            ddir = os.path.join(ROOT, "data", "corpus", dom)
            assert os.path.isdir(ddir), f"data/corpus/{dom} is absent, so {MIX} cannot run here"
            k = len([s for s in os.listdir(ddir) if s.endswith(".jsonl")])
            assert k, f"data/corpus/{dom} has no shards"
            n_shard += k
        # THE CACHE GUARD MOVED TO STAGE 2, where the vocabulary fingerprint exists.
        # The real predicate needs it, and the name test that stood here did not -- which
        # is why it could run this early. Proxy replaced by the thing itself (6e's ruling).
        print(f"    {n_shard} shards, domains={doms}")

        stage(2, "the vocabulary the pipeline will use, and its fingerprint")
        tok_path = os.path.join(ROOT, "data", "tokenizer.json")
        assert os.path.exists(tok_path), "data/tokenizer.json missing"
        fp = vocab_fingerprint(Tokenizer.from_file(tok_path))
        print(f"    fingerprint {fp}")

        # WOULD THIS RUN REWRITE A CACHE? The guard that stood in stage 1 asked whether a
        # domain NAME collides with the default mix's. That proxy is wrong in both
        # directions: it refused mix_200m_8b, whose nine domains would every one REUSE
        # (measured 2026-09-03), and a differently-named domain with a stale cache -- the
        # actual hazard -- passed it. What must never happen is a six-step test WRITING a
        # multi-day pretrain's cache (de's 0-byte vocab stamp on the shared cot cache cost
        # tilerl's gate run 5 minutes of 8 idle cards, 2026-09-02); rewriting is the
        # hazard, and now it is the question. REFUSED, not skipped.
        rebuilds = []
        for dom in doms:
            hot, why = _would_rebuild(dom, fp)
            print(f"    cache: {why}")
            if hot:
                rebuilds.append(why)
        if rebuilds:
            raise SystemExit(
                f"REFUSING: {len(rebuilds)} of {len(doms)} domain(s) in {MIX} would have their "
                f"token cache REWRITTEN by this run, and a six-step test must never be what "
                "writes a pretrain's cache:\n  " + "\n  ".join(rebuilds) +
                "\nRun it on a card whose /data00 is not shared with a live pretrain, or point "
                "E2E_MIX at a mix whose caches are already fresh. This is the launch-mix half "
                "of the test and it is REFUSED, not skipped."
            )

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
                *E2E_RECIPE,
                *E2E_SHAPE,
            ]
        )
        assert os.path.exists(ckpt), f"train.py produced no {ckpt}\n{out[-1500:]}"
        print(f"    {os.path.getsize(ckpt) / 1e6:.1f} MB")

        stage(4, "the pretrain actually took optimizer steps")
        import torch

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        # train.py seeds with Cfg.seed before building HybridLM, so a fresh init here is
        # bitwise identical to the run's step 0; cos < 0.9 proves the pretrain actually stepped.
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
        # A requested shape that did not take is the failure this test cannot afford to
        # miss: the run would be green, at the wrong shape, and read as a pass of the
        # shape it was launched for. train.py's shape flags are generated from a help
        # dict rather than written as add_argument calls, so a rename reaches the parser
        # and not the caller (754b624 renamed --d to --dim for exactly that reason).
        if E2E_LAYERS:
            assert cfg.layers == int(E2E_LAYERS), (
                f"asked for --layers {E2E_LAYERS}, the checkpoint says {cfg.layers}: the "
                f"flag did not take, and everything below tests the wrong shape")
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
        run([sys.executable, "datagen/prepare_sft_math.py", "--sources", sft_jsonl, "--out", pack])
        pk = torch.load(pack, map_location="cpu", weights_only=False)
        pack_fp = pk.get("vocab_id") if isinstance(pk, dict) else None
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
        # A fresh init is uncorrelated with the pretrained embedding; cos > 0.9 proves SFT
        # resumed them rather than reinitialising.
        cos = torch.nn.functional.cosine_similarity(before.flatten(), after.flatten(), dim=0).item()
        assert cos > 0.9, (
            f"SFT embeddings barely resemble the pretrained ones (cos={cos:.3f}); it reinitialised"
        )
        print(f"    embedding cos(pretrained, sft) = {cos:.4f}")

        stage(9, "the finished model generates")
        # bf16, as eval/math_hard.py does: flash_attn_func in the SWA blocks raises
        # "FlashAttention only support fp16 and bf16 data type" on the fp32 weights a
        # checkpoint loads as.
        from train import generate_batch

        model, cfg = load_checkpoint(sft_ckpt, device="cuda:0", dtype=torch.bfloat16)
        tok = Tokenizer.from_file(tok_path)
        ids = tok.encode("1 + 1 = ", add_special_tokens=False).ids
        (out,) = generate_batch(model, [ids], 4, "cuda:0")
        # argmax over all-NaN logits returns 0 silently, and FP8 e4m3 backward NaN without
        # grad_ckpt is a documented failure here, so "it did not raise" is not enough.
        with torch.no_grad():
            lg = model(torch.tensor([ids + out], device="cuda:0"))
            assert torch.isfinite(lg[0] if isinstance(lg, tuple) else lg).all(), "logits are NaN/Inf"
        print(f"    generated: {tok.decode(ids + out)!r}")

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
        # The gate's record, and the shape comes from the checkpoint this run produced
        # -- ck["cfg"], not the flag that was requested. The flag says what was asked
        # for; the checkpoint says what was built, and only the second is evidence. This
        # test requires a GPU, so a run that reaches here ran real kernels.
        from launch_tests import record_launch_test

        cf = ck["cfg"]
        record_launch_test(__file__, "pass",
                           {k: cf[k] for k in ("d", "layers", "heads", "ffn_hidden")},
                           real_kernel=True, mix=MIX)
        return 0
    finally:
        # STAGE 11 ONLY IF THE RUN GOT FAR ENOUGH TO NEED IT. This `finally` runs on the
        # refusal paths too, and `name` is bound at stage 4 -- so a stage-1/2 refusal fell
        # through to here and raised `UnboundLocalError: cannot access local variable
        # 'name'` WHILE HANDLING THE SystemExit. The refusal text printed, then the stack
        # top said UnboundLocalError: a reader seeing the last line judges "the tool is
        # broken" when the truth is "this mix may not run here". Refusal and crash shared
        # one exit (gate_failure_shapes §140), and the guard's REFUSAL PATH had never been
        # run -- which is more basic than whether the guard blocks the right thing.
        #
        # `locals()` rather than a flag: the condition is exactly "did the body reach the
        # point that created what stage 11 consumes", and the binding IS that fact.
        _have = {"name", "ckpt"} <= set(locals())
        if not _have:
            print("\n[11] skipped: the run refused before a checkpoint existed, so there "
                  "is nothing to attempt a resume from (and nothing was written)")
        else:
            stage(11, "resume from the step-less final save refuses (fb 2026-09-02)")
        # The end-of-run save writes neither step nor opt, so a resume from it would
        # silently restart at step 0 with a cold optimizer. The guard is field-based,
        # not filename-based: milestone hardlinks without stepN in the name pass.
        p = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--name",
                name + "_resume",
                "--mix",
                MIX,
                "--resume",
                ckpt,
                "--max_steps",
                "1",
                "--batch",
                "1",
                "--seq",
                "512",
                "--warmup",
                "2",
                *E2E_RECIPE,
                *E2E_SHAPE,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert p.returncode != 0 and "refusing to resume" in (p.stdout + p.stderr), (
            f"resume from step-less {ckpt} did not refuse (rc={p.returncode})\n"
            f"{p.stdout[-1200:]}\n{p.stderr[-1200:]}"
        )
        print("    refused as designed")

        # Always clean up the temp checkpoints; the .ep1 carries optimizer state and is the
        # bigger file.
        rm(os.path.join(ROOT, "ckpt_e2e_tmp.pt"), os.path.join(ROOT, "ckpt_e2e_tmp_sft.pt"))
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
