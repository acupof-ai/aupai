#!/usr/bin/env python3
"""Export a training checkpoint as a Hugging Face model directory.

    python3 scripts/export_hf.py --ckpt ckpt_v42_gate_1001r.pt.step16000 --out export/aupai-v42
    python3 scripts/export_hf.py --selftest          # no checkpoint needed

What lands: safetensors shards + index, config.json (AupaiV42Config), the two remote-code
modules from hf/, a copy of v41f/, the tokenizer the checkpoint was trained with,
generation_config.json, and a model card carrying the measured numbers it was given.

TWO THINGS THIS REFUSES RATHER THAN GUESSES.
  1. A checkpoint whose `vocab_id` disagrees with the tokenizer being copied. Scoring a
     checkpoint against another vocabulary is the loudest skipped-check bug this repo has had
     (a k5 SFT trained at loss 4.77 instead of 1.28), and an export that pairs the wrong
     tokenizer.json reproduces it for every downstream user.
  2. An export whose logits differ from the source model's. `--verify` rebuilds V42LM from the
     checkpoint, loads the exported directory through transformers, and compares both on a
     fixed input; the default is to verify, because an export nobody checked is a claim about
     weights nobody read.
"""

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SHARD_BYTES = 5 * 1000**3  # HF's conventional 5GB shard
# The file the export carries its persistent=False buffers in. Named here because the wrapper
# reads it by the same constant.
NONPERSISTENT_BUFFERS = "nonpersistent_buffers.safetensors"


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _dtype_size(t):
    return t.numel() * t.element_size()


def plan_shards(state, limit=SHARD_BYTES):
    """Greedy pack in state_dict order -> [(filename, [keys])], plus the weight_map.

    Order is the state_dict's, not sorted: a loader streams shards in index order, and keeping
    module order means one shard holds contiguous layers. A single tensor larger than the limit
    gets its own shard rather than raising -- the limit is a convention, not a format rule.
    """
    keys = list(state)
    shards, cur, cur_bytes = [], [], 0
    for k in keys:
        b = _dtype_size(state[k])
        if cur and cur_bytes + b > limit:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += b
    if cur:
        shards.append(cur)
    n = len(shards)
    if n == 1:
        names = ["model.safetensors"]
    else:
        names = [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]
    weight_map = {k: names[i] for i, ks in enumerate(shards) for k in ks}
    return list(zip(names, shards, strict=True)), weight_map


def _read_ckpt(path):
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    for field in ("model", "cfg"):
        if field not in ck:
            raise SystemExit(f"{path} has no ck['{field}']: not a train.py checkpoint")
    cfg = ck["cfg"]
    v42 = cfg.get("v42_cfg")
    if not v42:
        raise SystemExit(
            f"{path} carries no cfg['v42_cfg'], so it is not an --arch v42 checkpoint. The HF "
            f"wrapper builds V42LM only; a HybridLM checkpoint needs its own wrapper."
        )
    return ck, dict(v42)


def _tokenizer_vocab_id(path):
    """The fingerprint train.py stamps, by CALLING the repo's implementation rather than
    restating it.

    A hand-written version of this was wrong in two ways at once and refused a correct pair
    (measured 2026-10-03 on the real gate tokenizer: it computed 9897b9f2 where the checkpoint
    and AGENTS.md both say f1f860970d15d623). Both mistakes came from guessing the algorithm:
    it hashed `f"{id}\\t{token}\\n"` where the real one hashes the token BYTES ALONE in id
    order, and it read the raw JSON's `model.vocab` where the real one reads
    `Tokenizer.get_vocab()`, which also carries the added tokens -- `<eos>` and `[NUM]` among
    them. `scripts/loader.vocab_fingerprint` and `train.vocab_fingerprint` are line-for-line
    the same function and agree on a fixture (selftest below); loader's is used here because
    it has no torch dependency and imports in 0.1s against train.py's 21s.
    """
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from loader import vocab_fingerprint
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    return vocab_fingerprint(tok), tok.get_vocab_size()


def export(ckpt, out, tokenizer=None, card_numbers=None, verify=True):
    # restartable: every artifact is written whole with _write_json or save_file, and the
    # directory is rebuilt from the checkpoint in minutes. An interrupt leaves a partial
    # directory that the next run overwrites; nothing upstream is consumed or mutated.
    from safetensors.torch import save_file

    from v41f.config import V41FConfig
    from v41f.lm import V42LM

    ck, v42 = _read_ckpt(ckpt)
    state = {f"model.{k}": v for k, v in ck["model"].items()}
    os.makedirs(out, exist_ok=True)

    tok_path = tokenizer or os.path.join(ROOT, "data", "tokenizer.json")
    tok_id = None
    if os.path.exists(tok_path):
        tok_id, _n_slots = _tokenizer_vocab_id(tok_path)
        ck_id = ck.get("vocab_id")
        # Compare on an 8-char prefix: train.py has stamped vocab_id at more than one width,
        # so an equality test on the full strings would refuse a correct pair.
        if ck_id and str(ck_id)[:8] != tok_id[:8]:
            raise SystemExit(
                f"REFUSING: checkpoint vocab_id {ck_id} != tokenizer {tok_path} vocab_id "
                f"{tok_id}. Pass --tokenizer with the vocabulary this checkpoint was trained "
                f"on; an export that ships the wrong tokenizer.json mis-scores every "
                f"downstream read."
            )
        shutil.copy(tok_path, os.path.join(out, "tokenizer.json"))
        _write_json(
            os.path.join(out, "tokenizer_config.json"),
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": v42.get("max_seq_len", 4096),
                "eos_token": "<eos>",
                "bos_token": None,
                "unk_token": None,
                "clean_up_tokenization_spaces": False,
            },
        )
    else:
        print(
            f"WARNING: {tok_path} absent, exporting without a tokenizer (it is gitignored; "
            f"copy it from the pod)",
            file=sys.stderr,
        )

    # THE DTYPE POLICY IS READ OFF THE TENSORS, NOT ASSUMED. train.py holds v41f.lm.KEEP_FP32 in
    # fp32 while the rest is bf16, and transformers loads everything into fp32 unless told
    # otherwise -- which doubles the memory and moves the logits by bf16 rounding (measured
    # 1.7e-2 on the tiny fixture). Recording the majority dtype plus the exceptions lets the
    # wrapper restore exactly what was saved, for a mixed checkpoint and an all-fp32 one alike.
    by_dtype = {}
    for k, v in state.items():
        by_dtype.setdefault(str(v.dtype).replace("torch.", ""), []).append(k[len("model.") :])
    default_dtype = max(by_dtype, key=lambda d: len(by_dtype[d]))
    exceptions = {k: d for d, ks in by_dtype.items() if d != default_dtype for k in ks}

    shards, weight_map = plan_shards(state)
    total = sum(_dtype_size(t) for t in state.values())
    for name, keys in shards:
        save_file(
            {k: state[k].contiguous() for k in keys}, os.path.join(out, name), metadata={"format": "pt"}
        )
    if len(shards) > 1:
        _write_json(
            os.path.join(out, "model.safetensors.index.json"),
            {"metadata": {"total_size": total}, "weight_map": weight_map},
        )

    cfg_json = {
        "architectures": ["AupaiV42ForCausalLM"],
        "model_type": "aupai_v42",
        "auto_map": {
            "AutoConfig": "configuration_aupai.AupaiV42Config",
            "AutoModelForCausalLM": "modeling_aupai.AupaiV42ForCausalLM",
        },
        "v42_cfg": v42,
        "vocab_id": ck.get("vocab_id"),
        "train_step": ck.get("step"),
        "vocab_size": v42.get("vocab_size"),
        "hidden_size": v42.get("dim"),
        "num_hidden_layers": v42.get("n_layers"),
        "num_attention_heads": v42.get("n_heads"),
        "max_position_embeddings": v42.get("max_seq_len", 4096),
        "tie_word_embeddings": False,
        "use_cache": False,
        "torch_dtype": default_dtype,
        "param_dtype_default": default_dtype,
        "param_dtype_exceptions": exceptions,
    }
    _write_json(os.path.join(out, "config.json"), cfg_json)
    _write_json(
        os.path.join(out, "generation_config.json"),
        {"eos_token_id": 1, "max_new_tokens": 280, "do_sample": False, "use_cache": False},
    )

    # NON-PERSISTENT BUFFERS SHIP WITH THE EXPORT. 100 of them (25.2 MB, 25.17 of it the 24
    # per-layer freqs_cis RoPE tables) are registered persistent=False, so they are absent from
    # ck["model"] by design and a fresh build computes them in __init__. transformers never runs
    # that __init__ against a real device: it materialises the module from the file alone, and
    # every buffer the file does not carry comes back as uninitialised memory. The real
    # step16000 export loaded with 0 missing keys, 0 unexpected keys and 0 NaN PARAMETERS, and
    # produced NaN logits from 8-9 NaN buffers (compressor.kv_state, score_state, freqs_cis...);
    # low_cpu_mem_usage=False does not help, because transformers 5.6 no longer honours it.
    # So the values travel in the export and the wrapper copies them in after loading: no new
    # dependency, and no reliance on which init path transformers takes.
    buf_src = V42LM(V41FConfig(**v42))
    bufs = {
        name: b.clone() for name, b in buf_src.named_buffers() if name not in ck["model"] and b is not None
    }
    del buf_src
    if bufs:
        save_file(
            {k: v.contiguous() for k, v in bufs.items()},
            os.path.join(out, NONPERSISTENT_BUFFERS),
            metadata={"format": "pt"},
        )

    for mod in ("configuration_aupai.py", "modeling_aupai.py"):
        shutil.copy(os.path.join(ROOT, "hf", mod), os.path.join(out, mod))
    v41f_dst = os.path.join(out, "v41f")
    if os.path.exists(v41f_dst):
        shutil.rmtree(v41f_dst)
    shutil.copytree(
        os.path.join(ROOT, "v41f"), v41f_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )

    n_params = sum(t.numel() for t in state.values())
    card = _model_card(ckpt, v42, n_params, total, ck.get("step"), tok_id, card_numbers)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    print(
        f"exported {out}: {len(shards)} shard(s), {total / 1e9:.2f} GB, {n_params:,} tensors' "
        f"elements, step {ck.get('step')}"
    )
    if verify:
        verify_export(ckpt, out)
    return out


def _model_card(ckpt, v42, n_params, total, step, tok_id, numbers):
    measured = numbers or "No evaluation number was passed to the exporter (`--card-number`)."
    return f"""---
library_name: transformers
tags: [code, math, moe, causal-lm]
---

# aupai v42

A from-scratch coding/math MoE transformer. {n_params:,} parameter elements
({total / 1e9:.2f} GB bf16), {v42["n_layers"]} layers, d={v42["dim"]},
{v42["n_heads"]} heads, {v42["n_routed_experts"]} routed experts top-{v42["n_activated_experts"]}
plus {v42["n_shared_experts"]} shared, vocabulary {v42["vocab_size"]}.

Exported from `{os.path.basename(ckpt)}`{f" at training step {step}" if step else ""}.
Tokenizer vocab_id `{tok_id}`: score this model only with the `tokenizer.json` in this
directory, because ids do not survive a vocabulary rebuild.

## Measured

{measured}

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("<this directory>", trust_remote_code=True)
t = AutoTokenizer.from_pretrained("<this directory>")
```

A LOCAL PATH, not a repo id, unless the whole repo has been snapshotted: the architecture
lives in `v41f/`, and `trust_remote_code` fetches only the .py modules it can resolve.

## Two ceilings

- **No KV cache.** `V42LM` recomputes the whole prefix each step, so `generate()` is correct
  and O(n^2) in the prefix.
- **No padding path.** The stack is causal over whole rows; a non-all-ones `attention_mask`
  raises rather than being silently ignored. Batch equal-length rows, or call
  `v41f.lm.V42LM` with `cu_seqlens`.
"""


def verify_export(ckpt, out):
    """Logits from the exported directory must equal logits from V42LM built on the checkpoint.

    Both sides run on CPU over the same fixed token ids, and both are given the checkpoint's OWN
    dtypes: `load_state_dict(..., assign=True)` replaces the freshly built fp32 parameters with
    the saved tensors rather than copying into them, so the source carries the same bf16/fp32
    split the export does. Without assign=True the source would be all-fp32 while the export is
    mixed, and the comparison would measure bf16 rounding instead of the wrapper. The source is
    then run through the wrapper's own autocast helper, for the same reason.

    The assertion is exact equality of the argmax and a tight bound on the values: with the
    weights and the dtype context identical, the wrapper adds no arithmetic, so any difference
    is a renamed key, a cast dtype or a dropped config field -- each of which a looser tolerance
    would hide.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from v41f.config import V41FConfig
    from v41f.lm import V42LM

    ck, v42 = _read_ckpt(ckpt)
    torch.manual_seed(0)
    ids = torch.randint(0, v42["vocab_size"], (1, 16))

    src = V42LM(V41FConfig(**v42))
    src.load_state_dict(ck["model"], assign=True)
    src.eval()
    mixed = any(p.dtype == torch.bfloat16 for p in src.parameters())
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=mixed):
        want, _ = src(ids)

    got_model = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True)
    got_model.eval()
    with torch.no_grad():
        got = got_model(input_ids=ids).logits

    if want.shape != got.shape:
        raise SystemExit(f"VERIFY FAILED: shape {tuple(got.shape)} != source {tuple(want.shape)}")
    same_argmax = bool((want.argmax(-1) == got.argmax(-1)).all())
    max_abs = float((want.float() - got.float()).abs().max())
    if not same_argmax or max_abs > 1e-4:
        raise SystemExit(
            f"VERIFY FAILED: argmax_equal={same_argmax} max|delta|={max_abs:.3e}. The wrapper adds "
            f"no arithmetic, so this is a renamed key, a cast dtype or a dropped config field."
        )
    print(f"verify OK: argmax identical, max|delta| {max_abs:.3e} over {ids.numel()} positions")


def selftest():
    """Known answers for the two things the exporter decides: how tensors pack into shards, and
    whether a vocab_id is computed over the id->token map rather than the file bytes."""
    import tempfile

    import torch

    bad = 0

    # (1) SHARD PACKING. 3 tensors of 2 GB under a 5 GB limit -> 2 shards, first holds two.
    st = {k: torch.empty(2 * 1000**3, dtype=torch.uint8) for k in ("a", "b", "c")}
    shards, wmap = plan_shards(st)
    ok = (
        len(shards) == 2
        and shards[0][1] == ["a", "b"]
        and shards[1][1] == ["c"]
        and wmap["a"] == "model-00001-of-00002.safetensors"
        and wmap["c"] == "model-00002-of-00002.safetensors"
    )
    bad += 0 if ok else 1
    print(
        f"  {'ok  ' if ok else 'BUG '} 3x2GB under 5GB packs 2+1 and names shards 1-of-2, 2-of-2"
        + ("" if ok else f" -- got {[(n, k) for n, k in shards]}")
    )

    # The negative: one tensor over the limit gets its own shard instead of raising, and a
    # single-shard export is named model.safetensors with no index.
    shards1, wmap1 = plan_shards({"big": torch.empty(6 * 1000**3, dtype=torch.uint8)})
    ok = len(shards1) == 1 and shards1[0][0] == "model.safetensors"
    bad += 0 if ok else 1
    print(
        f"  {'ok  ' if ok else 'BUG '} a lone over-limit tensor is one shard named "
        f"model.safetensors" + ("" if ok else f" -- got {shards1[0][0]}")
    )

    # (2) THE FINGERPRINT IS THE REPO'S, NOT A RESTATEMENT. The known answer is arithmetic a
    #     reader can check by hand: three tokens a,b,c at ids 0,1,2 hash the bytes "abc" in id
    #     order, so the value is sha256("abc")[:16] = ba7816bf8f01cfea -- which is also what
    #     train.vocab_fingerprint returns on the same object (verified 2026-10-03, both 16 chars
    #     identical). A version that hashed ids alongside tokens, or read the raw JSON vocab
    #     instead of get_vocab(), lands elsewhere and is what refused a correct pair on the pod.
    #     The second file swaps two ids: the token ORDER changes, so the value must change.
    import hashlib

    from tokenizers import Tokenizer, models

    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.json")
        c = os.path.join(d, "c.json")
        Tokenizer(models.WordLevel(vocab={"a": 0, "b": 1, "c": 2}, unk_token="a")).save(a)
        Tokenizer(models.WordLevel(vocab={"a": 2, "b": 1, "c": 0}, unk_token="a")).save(c)
        ida, n_a = _tokenizer_vocab_id(a)
        idc, _ = _tokenizer_vocab_id(c)
    want = hashlib.sha256(b"abc").hexdigest()[:16]
    ok = ida == want and n_a == 3 and idc != ida
    bad += 0 if ok else 1
    print(
        f"  {'ok  ' if ok else 'BUG '} vocab_id is sha256 over the token bytes in id order "
        f"({want}) and changes when two ids swap"
        + ("" if ok else f" -- got {ida} (want {want}), n={n_a}, swapped={idc}")
    )

    # (3) THE REFUSALS EXIST. A non-v42 checkpoint must be refused by name rather than exported
    # as an empty directory.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.pt")
        torch.save({"model": {}, "cfg": {"arch": "hybrid"}}, p)
        try:
            _read_ckpt(p)
            ok = False
            why = "a HybridLM checkpoint was accepted"
        except SystemExit as e:
            ok = "v42_cfg" in str(e)
            why = str(e)[:60]
    bad += 0 if ok else 1
    print(
        f"  {'ok  ' if ok else 'BUG '} a checkpoint without cfg['v42_cfg'] is refused by name"
        + ("" if ok else f" -- {why}")
    )

    # (4) THE WHOLE PATH, at a shape that fits a laptop. Build a tiny V42LM, save it in the
    #     checkpoint schema, export, load the directory back through transformers, compare
    #     logits. This is the only case that exercises the joins the real export depends on --
    #     key prefixing, the config round-trip through V41FConfig, auto_map resolution and the
    #     v41f copy on the import path -- and a mutation to any of them turns it red. The real
    #     3.2B export cannot be a test: it needs 26 GB of RAM for the two copies.
    with tempfile.TemporaryDirectory() as d:
        from v41f.config import v42_s24
        from v41f.lm import V42LM

        c = v42_s24(
            n_layers=2,
            dim=64,
            n_heads=2,
            o_groups=2,
            head_dim=32,
            rope_head_dim=16,
            vocab_size=128,
            n_routed_experts=4,
            n_activated_experts=2,
            moe_inter_dim=32,
            index_topk=8,
            index_n_heads=2,
            index_head_dim=16,
            compress_ratios=(2, 2),
            kv_source_layers=(0,),
            index_source_layers=(0,),
        )
        torch.manual_seed(0)
        m = V42LM(c)
        ckpt = os.path.join(d, "tiny.pt")
        import dataclasses

        torch.save(
            {
                "model": {k: v.cpu() for k, v in m.state_dict().items()},
                "cfg": {"arch": "v42", "v42_cfg": dataclasses.asdict(c)},
                "vocab_id": None,
                "step": 7,
            },
            ckpt,
        )
        out = os.path.join(d, "hf")
        try:
            export(ckpt, out, tokenizer=os.devnull + "/absent", verify=True)
            ok, why = True, ""
        except SystemExit as e:
            ok, why = False, str(e)[:160]
        bad += 0 if ok else 1
        print(
            f"  {'ok  ' if ok else 'BUG '} a tiny checkpoint exports and reloads through "
            f"transformers with identical logits" + ("" if ok else f" -- {why}")
        )
        if ok:
            # The index is absent at one shard, and the model card names the step it was cut at.
            has_index = os.path.exists(os.path.join(out, "model.safetensors.index.json"))
            with open(os.path.join(out, "README.md"), encoding="utf-8") as f:
                card = f.read()
            sub = (not has_index) and "step 7" in card and "No evaluation number" in card
            bad += 0 if sub else 1
            print(
                f"  {'ok  ' if sub else 'BUG '} one shard writes no index, and the card states "
                f"the step and that no number was passed" + ("" if sub else f" -- index={has_index}")
            )

        # (5) THE SAME PATH ON A MIXED-DTYPE CHECKPOINT, which is what a real one is. Case (4)
        #     saves a freshly built model, so every tensor is fp32 and the dtype split never
        #     appears -- it passed on the day the real step16000 export wrote all ten files and
        #     then died in its own verify with "expected m1 and m2 to have the same dtype". The
        #     fixture lacked the precondition, so it could not refute anything. Here the saved
        #     tensors carry the real split: v41f.lm._keep_fp32's set stays fp32, everything else
        #     goes bf16, exactly as train.py writes it.
        from v41f.lm import _keep_fp32

        mixed_sd = {
            k: (v.cpu() if _keep_fp32(k) else v.cpu().to(torch.bfloat16)) for k, v in m.state_dict().items()
        }
        n_fp32 = sum(1 for k in mixed_sd if _keep_fp32(k))
        ckpt_m = os.path.join(d, "tiny_mixed.pt")
        torch.save(
            {
                "model": mixed_sd,
                "cfg": {"arch": "v42", "v42_cfg": dataclasses.asdict(c)},
                "vocab_id": None,
                "step": 8,
            },
            ckpt_m,
        )
        try:
            export(ckpt_m, os.path.join(d, "hf_mixed"), tokenizer=os.devnull + "/absent", verify=True)
            ok, why = True, ""
        except (SystemExit, RuntimeError) as e:
            ok, why = False, str(e)[:160]
        ok = ok and 0 < n_fp32 < len(mixed_sd)
        bad += 0 if ok else 1
        print(
            f"  {'ok  ' if ok else 'BUG '} a MIXED bf16/fp32 checkpoint ({n_fp32} of "
            f"{len(mixed_sd)} tensors held fp32 by _keep_fp32) exports and reloads with "
            f"identical logits" + ("" if ok else f" -- {why}")
        )

        # (6) THE BUFFER FILE COVERS EVERY persistent=False BUFFER, AND ITS ABSENCE RAISES.
        #     This case exists because cases (4) and (5) CANNOT catch the defect it guards: at
        #     the tiny shape the uninitialised buffer memory happened to be benign and both
        #     passed before the fix, while the real 3.2B export returned NaN logits. Luck is not
        #     a criterion, so the assertion is set-equality on the names plus the refusal --
        #     neither of which depends on what the allocator handed back.
        import shutil as _sh

        from safetensors.torch import load_file

        from v41f.config import V41FConfig as _VC
        from v41f.lm import V42LM as _VL

        ref = _VL(_VC(**dataclasses.asdict(c)))
        want_names = {n for n, _ in ref.named_buffers() if n not in mixed_sd}
        del ref
        bpath = os.path.join(d, "hf_mixed", NONPERSISTENT_BUFFERS)
        have_names = set(load_file(bpath)) if os.path.exists(bpath) else set()
        ok = bool(want_names) and have_names == want_names
        bad += 0 if ok else 1
        print(
            f"  {'ok  ' if ok else 'BUG '} the export carries every one of the "
            f"{len(want_names)} persistent=False buffer(s)"
            + (
                ""
                if ok
                else f" -- missing {sorted(want_names - have_names)[:4]}, "
                f"extra {sorted(have_names - want_names)[:4]}"
            )
        )

        # The negative: with the file gone, loading must RAISE rather than answer NaN.
        gone = os.path.join(d, "hf_gone")
        _sh.copytree(os.path.join(d, "hf_mixed"), gone)
        os.remove(os.path.join(gone, NONPERSISTENT_BUFFERS))
        from transformers import AutoModelForCausalLM as _AM

        try:
            _AM.from_pretrained(gone, trust_remote_code=True)
            ok, why = False, "it loaded without the buffer file"
        except Exception as e:  # noqa: BLE001 - any raise is the contract; NaN silence is not
            ok, why = "nonpersistent_buffers" in str(e), str(e)[:80]
        bad += 0 if ok else 1
        print(
            f"  {'ok  ' if ok else 'BUG '} loading an export whose buffer file was deleted "
            f"RAISES by name" + ("" if ok else f" -- {why}")
        )

    print(f"export_hf: {9 - bad}/9 pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="training checkpoint (ck['model'] + ck['cfg']['v42_cfg'])")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--tokenizer", help="tokenizer.json to ship (default data/tokenizer.json)")
    ap.add_argument("--card-number", help="one line of measured results for the model card")
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the logit comparison against the source model (not recommended)",
    )
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.ckpt or not a.out:
        ap.error("--ckpt and --out are required unless --selftest")
    export(a.ckpt, a.out, tokenizer=a.tokenizer, card_numbers=a.card_number, verify=not a.no_verify)


if __name__ == "__main__":
    main()
