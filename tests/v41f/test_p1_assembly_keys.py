"""Assembly key-set / dtype structural gate for the v41f A/B wiring (assembly plan #468).

This file does NOT wire anything and never edits v41f/model.py. It freezes the EXPECTED
state_dict key set and dtype contract so that, when de lands step A (Engram) and step B
(DSpark draft), a wrong key name, a duplicated tied key, or a silently-changed dtype turns
this red immediately. It is config-derived rather than a hand-typed per-layer list because
v41f_small is intentionally ASYMMETRIC: compress_ratios (0,2,2,1,1) plus the kv/index
source split means layer 1 carries compressor+index_key+indexer (52 keys), layer 3 carries
compressor+index_key+indexer with no fp32 gate (51), and layers 0/2/4 are window-ish (45).

Three sections:
  - BASELINE runs today: v41f_small with engram and MTP OFF must stay exactly 241 sd keys /
    236 parameters, with the dtype contract below. G-A2 says this must still hold after the
    engram wiring (the feature off == no change).
  - STEP_A / STEP_B are skipped out loud until the corresponding module is actually
    registered on the model (a missing attribute), never silently xfailed. The wiring PR
    flips them from SKIP to a real assertion by turning the feature on; a wrong key/dtype
    then fails by name.

Run:  python3 tests/v41f/test_p1_assembly_keys.py --selftest
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
from v41f.config import v41f_small  # noqa: E402
from v41f.model import V41FModel  # noqa: E402

# --- Baseline contract, MEASURED on main (all off). Do not hand-edit; if a real change to the
# off model moves these, the assembly plan must say so.
BASELINE_SD_KEYS = 241
BASELINE_PARAMS = 236
# persistent buffers that are in state_dict but are not nn.Parameters (one per layer).
PERSISTENT_BUFFER_SUFFIX = "ffn.gate.bias"
BASELINE_PERSISTENT_BUFFERS = 5
# non-persistent runtime state that must NEVER enter the checkpoint blob.
NONPERSISTENT_NAMES = ("freqs_cis", "kv_state", "score_state")
# top-level keys and their dtype.
TOP_LEVEL = {
    "embed.weight": torch.bfloat16,
    "norm.weight": torch.bfloat16,
    "head.weight": torch.float32,   # V41FHead is a single fp32 Parameter, no sub-module
}
# fp32 suffixes inside every backbone layer (HC six tables + attention sink).
LAYER_FP32_SUFFIXES = (
    "attn.attn_sink",
    "hc.hc_attn_fn", "hc.hc_ffn_fn", "hc.hc_attn_base", "hc.hc_ffn_base",
    "hc.hc_attn_scale", "hc.hc_ffn_scale",
)
GATE_BIAS_SUFFIX = "ffn.gate.bias"   # persistent fp32 SELECTION-ONLY buffer

# --- Field-intersection contract (durable gate from #468 comment 5715773669).
# test_p1_model derives ref ModelArgs by `asdict(cfg) filtered to fields(ModelArgs)`: any
# MODEL-SHAPE field that exists only on the ref side is silently dropped and the ref builds
# it at its own default. This gate enumerates the difference instead of trusting a per-field
# spot check, so a future ref-only shape field fails by name.
#
# REF_ONLY_LEGITIMATE is the exact set of ref ModelArgs fields absent from V41FConfig that
# are NOT model shapes v41f must own: dtype/expert/temperature/image are forced by bf16_args,
# vision_* is out of scope, and max_batch_size/max_seq_len are RUNTIME/harness parameters
# (bf16_args supplies them; max_seq_len must never become a config field — that would make a
# runtime length look like a controlled model shape).
REF_ONLY_RUNTIME_OR_OUT_OF_SCOPE = frozenset({
    "dtype", "expert_dtype", "temperature", "image_token_id", "max_batch_size",
    "max_seq_len",
    "vision_dim", "vision_downsample_ratio", "vision_inter_dim", "vision_max_n_token",
    "vision_max_wh_ratio", "vision_min_pixels", "vision_n_heads", "vision_n_layers",
    "vision_patch_size", "vision_rope_theta",
})
# Ref-only MODEL-SHAPE fields still missing from V41FConfig. Step A landed
# engram_num_embeddings, so this is EMPTY on main. A FUTURE ref-only shape field belongs here
# exactly until it is added to V41FConfig; the intersection gate fails on any unnamed member.
REF_ONLY_SHAPE_PENDING = frozenset()

# V41F-ONLY TRAINING KNOBS: V41FConfig fields that deliberately have NO counterpart in the
# vendored ref ModelArgs. The reference is an INFERENCE port; it never trains the CSA2
# second-level indexer, so the straight-through training switch (design #456, step C #485)
# is a v41f-defined field the ref ModelArgs cannot consume. It is classified here on its own
# rather than folded into REF_ONLY_SHAPE_PENDING: that set names a ref-side shape the port
# still owes upstream; a knob is the opposite, an OUR-side control the ref is right not to
# have.
#
# EXPLICIT ENUMERATION, never a prefix/regex/wildcard: every member must point at a doc that
# marks it a v41f-only training flag. The guard is symmetric: a config field the ref cannot
# consume that is NOT listed here fails, and a field listed here that V41FConfig no longer
# carries (or that the ref has since grown) fails. Adding a knob is therefore a deliberate,
# documented act, never a silent wildcard match.
CFG_ONLY_TRAINING_KNOBS = frozenset({
    "indexer_train_mode",   # docs/standards/v41f_indexer_trainability_design.md (#456/#485)
})

# Prime bucket sum independently recomputed = 786,862; x head_dim 128 = 100.7M rows.
STEP_A_LAYER = 1
STEP_A_NEW_KEYS = {
    # suffix:                       (shape,            dtype)
    "embed.weight":                 ((786862, 128), torch.bfloat16),  # engram n-gram table
    "wkv.weight":                   ((3072, 1536),   torch.bfloat16),  # Linear [out,in]; 12*128 in
    "q_weight":                     ((2, 1024),      torch.bfloat16),  # explicit bf16 (was ambient)
    "k_weight":                     ((2, 1024),      torch.bfloat16),
}
# NgramHashState runtime state: ALL persistent=False, must be absent from state_dict.
ENGRAM_NONPERSISTENT = ("primes", "offsets", "multipliers", "token_map", "cache")

# --- Step B (DSpark draft) tied-embed/head contract.
STEP_B_MTP_PREFIX = "mtp.0."
# A DSparkBlock owns these backbone-shaped keys (45 on v41f_small: attn/ffn/hc/norms) plus,
# when rank0 Markov is on, main_proj.weight and main_norm.weight.
# The TIED tensors must register ONCE: these duplicate keys must NOT exist.
STEP_B_FORBIDDEN_TIED_KEYS = ("mtp.0.embed.weight", "mtp.0.head.weight")
STEP_B_SINGLE_KEYS = ("embed.weight", "head.weight")


class _Skip(RuntimeError):
    """A future assembly section whose module is not registered yet — skip out loud."""


def _build_baseline():
    torch.set_default_dtype(torch.bfloat16)
    model = V41FModel(v41f_small())
    torch.set_default_dtype(torch.float32)
    return model


def _layer_subpaths(sd, layer):
    p = f"layers.{layer}."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def gate_baseline():
    """The all-off floor. Must remain byte-for-byte true after A/B land with features off."""
    m = _build_baseline()
    sd = m.state_dict()
    fail = []
    keys = list(sd.keys())
    if len(keys) != BASELINE_SD_KEYS:
        fail.append(f"state_dict keys {len(keys)} != {BASELINE_SD_KEYS}")
    nparam = sum(1 for _ in m.parameters())
    if nparam != BASELINE_PARAMS:
        fail.append(f"named_parameters {nparam} != {BASELINE_PARAMS}")

    # top-level keys present with exact dtype
    for name, dt in TOP_LEVEL.items():
        if name not in sd:
            fail.append(f"missing top-level key {name}")
        elif sd[name].dtype != dt:
            fail.append(f"{name} dtype {sd[name].dtype} != {dt}")

    # persistent gate.bias present fp32 once per layer, counted exactly
    biases = [k for k in keys if k.endswith(GATE_BIAS_SUFFIX)]
    if len(biases) != BASELINE_PERSISTENT_BUFFERS:
        fail.append(f"persistent {GATE_BIAS_SUFFIX} count {len(biases)} != "
                    f"{BASELINE_PERSISTENT_BUFFERS}")
    for k in biases:
        if sd[k].dtype != torch.float32:
            fail.append(f"{k} must be fp32, got {sd[k].dtype}")

    # every non-bias layer tensor carrying a known-fp32 suffix is fp32; HC/sink never bf16
    for layer in range(5):
        sub = _layer_subpaths(sd, layer)
        for suf in LAYER_FP32_SUFFIXES:
            if suf in sub and sub[suf].dtype != torch.float32:
                fail.append(f"layers.{layer}.{suf} must be fp32, got {sub[suf].dtype}")

    # nothing non-persistent leaks into the blob
    leaked = [k for k in keys if any(tok in k for tok in NONPERSISTENT_NAMES)]
    if leaked:
        fail.append(f"non-persistent runtime state entered state_dict: {leaked}")

    # parameter/buffer bookkeeping: sd keys minus persistent buffers == parameter count
    pnames = {n for n, _ in m.named_parameters()}
    only_buf = [k for k in keys if k not in pnames]
    if sorted(only_buf) != sorted(biases):
        fail.append(f"state_dict/parameter gap must be exactly the gate biases, got {only_buf}")

    if fail:
        raise AssertionError("; ".join(fail))
    return (f"baseline {BASELINE_SD_KEYS} sd keys / {BASELINE_PARAMS} params; "
            f"head+hc+sink fp32, {BASELINE_PERSISTENT_BUFFERS} fp32 gate biases, "
            "no non-persistent leak")


def _build_engram_on_small():
    """An engram-ON v41f_small on the disk-free synthetic tokenizer (#481 harness shapes).

    Returns None when the wiring is absent (step A not landed); the gate then skips. Once the
    feature is on this constructs for real so the key/dtype contract is exercised on every
    commit rather than waiting for a production tokenizer.
    """
    import sys as _sys
    here = Path(__file__).resolve().parent
    if str(here) not in _sys.path:
        _sys.path.insert(0, str(here))
    try:
        from ref_oracle import synthetic_tokenizer  # noqa: WPS433
    except Exception:
        return None
    tok = synthetic_tokenizer()
    cfg = v41f_small(
        vocab_size=len(tok), tokenizer=tok,
        engram_layer_ids=(1,), engram_max_ngram_size=4, engram_n_heads=2,
        engram_head_dim=8, engram_vocab_size=20, engram_pad_id=2,
    )
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = V41FModel(cfg, max_batch_size=2, max_seq_len=64, tokenizer=tok).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    return model, cfg


def gate_step_a_engram():
    """When Engram is wired and enabled, exactly four engrams.1.* keys appear, all bf16, with
    the table/wkv/q/k shapes, and NgramHashState's five buffers stay out of the blob.

    Runs on the SMALL synthetic config (shapes below); the production 786862x128 shapes in
    STEP_A_NEW_KEYS document the gate-config contract the param counter/checkpoint use.
    """
    built = _build_engram_on_small()
    if built is None:
        raise _Skip("engram wiring/tokenizer harness unavailable — pre-step-A world")
    m, cfg = built
    if not any(getattr(e, "embed", None) is not None for e in getattr(m, "engrams", [])):
        raise _Skip("engram not wired (self.engrams all None) — step A section pending")
    sd = m.state_dict()
    layer = cfg.engram_layer_ids[0]
    prefix = f"engrams.{layer}."
    present = {k[len(prefix):]: tuple(v.shape) for k, v in sd.items() if k.startswith(prefix)}
    fail = []
    # structural contract: exactly the four expected suffixes, all bf16 (small-config shapes)
    SMALL_A_SHAPES = {
        "embed.weight": (cfg.engram_num_embeddings[0], cfg.engram_head_dim),
        "q_weight": (cfg.hc_mult, cfg.dim),
        "k_weight": (cfg.hc_mult, cfg.dim),
        # wkv is [dim*(hc_mult+1), (max_ngram-1)*n_heads*head_dim]
        "wkv.weight": (cfg.dim * (cfg.hc_mult + 1),
                       (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads * cfg.engram_head_dim),
    }
    for suf, shape in SMALL_A_SHAPES.items():
        if suf not in present:
            fail.append(f"missing {prefix}{suf}")
        elif present[suf] != shape:
            fail.append(f"{prefix}{suf} shape {present[suf]} != {shape}")
        elif sd[prefix + suf].dtype != torch.bfloat16:
            fail.append(f"{prefix}{suf} must be bf16, got {sd[prefix+suf].dtype}")
    extra = set(present) - set(SMALL_A_SHAPES)
    if extra:
        fail.append(f"unexpected engram keys {sorted(extra)}")
    if set(present) != set(SMALL_A_SHAPES):
        fail.append(f"engram key count {len(present)} != 4: {sorted(present)}")
    leaked = [k for k in sd if "engram_hash" in k and
              any(tok in k for tok in ENGRAM_NONPERSISTENT)]
    if leaked:
        fail.append(f"NgramHashState persistent=False buffers leaked into state_dict: {leaked}")
    if fail:
        raise AssertionError("; ".join(fail))
    return (f"step A: 4 engrams.{layer}.* bf16 keys, table "
            f"{cfg.engram_num_embeddings[0]}x{cfg.engram_head_dim}, 5 hash buffers absent")


def _build_mtp_on_small():
    """A DSpark-draft-ON v41f_small (assembly plan #468 step B, live after #484).

    n_mtp_layers=1 wires self.mtp; the tied embed/head are call-site references, not
    DSparkBlock registrations. Disk-free (no tokenizer), same small shape as baseline.
    """
    cfg = v41f_small(n_mtp_layers=1, dspark_block_size=5, dspark_target_layer_ids=(2,))
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = V41FModel(cfg, max_batch_size=2)
    finally:
        torch.set_default_dtype(torch.float32)
    return model, cfg


def gate_step_b_mtp():
    """DSpark draft wired ON: mtp.0.* keeps its own backbone keys but the tied embed/head
    register ONCE at the top — no mtp.0.embed.weight / mtp.0.head.weight.

    Armed: the gate builds the ON config itself, so a feature that fails to wire under
    n_mtp_layers=1 is a regression AssertionError, NOT a _Skip (the old builder was all-off,
    so this gate skipped on every run and printed [ok ] anyway).
    """
    m, _cfg = _build_mtp_on_small()
    if not getattr(m, "mtp", None):
        raise AssertionError(
            "step B armed but self.mtp is empty under n_mtp_layers=1: the DSpark draft did "
            "not wire (an armed gate must not SKIP)")
    sd = m.state_dict()
    fail = []
    for dup in STEP_B_FORBIDDEN_TIED_KEYS:
        if dup in sd:
            fail.append(f"tied tensor double-registered: {dup} (use a non-registered shared "
                        "reference so embed/head have one name)")
    for single in STEP_B_SINGLE_KEYS:
        if single not in sd:
            fail.append(f"tied tensor missing its single top-level name: {single}")
    own = [k for k in sd if k.startswith(STEP_B_MTP_PREFIX)]
    if not own:
        fail.append("mtp.0 present but exposes no backbone keys")
    if fail:
        raise AssertionError("; ".join(fail))
    return f"step B: {len(own)} mtp.0.* own keys, tied embed/head single-registered"


def gate_field_intersection():
    """No MODEL-SHAPE field silently fails to cross the ModelArgs<->V41FConfig intersection.

    Asserts the ref-only set partitions exactly into (a) runtime/out-of-scope names and
    (b) the explicitly-pending shape field(s), and that every V41FConfig field the ref
    ModelArgs cannot consume is either such a pending shape or a NAMED v41f-only training
    knob. Adding engram_num_embeddings to V41FConfig removes it from (b); any OTHER
    ref-only/cfg-only field appearing here fails by name.
    """
    import dataclasses

    from ref_oracle import load_reference  # noqa: E402 (test-only sibling on tests/v41f path)
    model, _ = load_reference()
    ref_fields = {f.name for f in dataclasses.fields(model.ModelArgs)}
    cfg_fields = set(dataclasses.asdict(v41f_small()))
    ref_only = ref_fields - cfg_fields
    cfg_only = cfg_fields - ref_fields

    legit = REF_ONLY_RUNTIME_OR_OUT_OF_SCOPE
    pending = REF_ONLY_SHAPE_PENDING
    knobs = CFG_ONLY_TRAINING_KNOBS
    fail = []
    unexpected = ref_only - legit - pending
    if unexpected:
        fail.append(f"ref-only model-shape fields silently dropped by _model_args "
                    f"intersection: {sorted(unexpected)} (add to V41FConfig or explicitly "
                    f"classify runtime/out-of-scope)")
    stale = legit & cfg_fields  # a runtime/scope name that became a real config field
    if stale:
        fail.append(f"field now present in V41FConfig but still whitelisted as non-shape: "
                    f"{sorted(stale)}")
    # SYMMETRIC to `stale`: a field listed as a still-missing PENDING shape must actually be
    # absent from V41FConfig. Once it is added (step A added engram_num_embeddings), leaving
    # it in REF_ONLY_SHAPE_PENDING makes the gate keep reporting "1 pending" forever — the
    # permanent-amber shape. This asserts the pending bookkeeping is cleared when the field
    # lands, mirroring the `stale` check for the runtime/scope whitelist.
    resolved_but_listed = pending & cfg_fields
    if resolved_but_listed:
        fail.append(f"field already present in V41FConfig but still listed in "
                    f"REF_ONLY_SHAPE_PENDING (clear it): {sorted(resolved_but_listed)}")
    # --- v41f-only training knobs: a category, not a wildcard.
    # (1) every cfg-only field must be classified; one the ref cannot consume that is not a
    #     named knob fails. This is the "someone added a config field the ref silently drops"
    #     tripwire, and it is what makes listing a knob deliberate rather than permissive.
    unlisted_knob = cfg_only - knobs
    if unlisted_knob:
        fail.append(f"V41FConfig fields the ref ModelArgs cannot consume and that are not "
                    f"classified v41f-only training knobs: {sorted(unlisted_knob)} (cross "
                    f"them normally, or add the name to CFG_ONLY_TRAINING_KNOBS with a doc "
                    f"that marks it a v41f-only training flag)")
    # (2) symmetric: every listed knob must still be a real V41FConfig field. A rename/removal
    #     that leaves the name here makes the whitelist vouch for a field that does not exist.
    missing_knobs = knobs - cfg_fields
    if missing_knobs:
        fail.append(f"CFG_ONLY_TRAINING_KNOBS names absent V41FConfig fields "
                    f"(remove the stale knob entry): {sorted(missing_knobs)}")
    # (3) the real guardrail of the category: a knob must have NO ref counterpart. If the ref
    #     ModelArgs ever grows the same-named field, it is no longer v41f-only and must cross
    #     the intersection like every other shape field, not ride the training-knob exemption.
    knob_now_in_ref = knobs & ref_fields
    if knob_now_in_ref:
        fail.append(f"training knob now exists in the ref ModelArgs; cross it normally and "
                    f"drop it from CFG_ONLY_TRAINING_KNOBS: {sorted(knob_now_in_ref)}")
    # (4) the two categories are disjoint: a knob is an our-side control, a pending entry is
    #     a ref-side shape the port owes. A name in both is a classification error.
    knob_in_pending = knobs & pending
    if knob_in_pending:
        fail.append(f"field classified as both a training knob and a pending ref shape; "
                    f"pick one category: {sorted(knob_in_pending)}")
    if fail:
        raise AssertionError("; ".join(fail))
    return (f"intersection: ref-only partitions into {len(legit)} runtime/scope + "
            f"{len(pending)} pending shape ({sorted(pending) or 'none'}); "
            f"{len(knobs & cfg_only)} v41f-only training knob(s) {sorted(knobs)}")


def _selftest():
    # Baseline always runs now.
    msg = gate_baseline()
    print(f"[ok ] {msg}")
    print(f"[ok ] {gate_field_intersection()}")
    # Assembly gates, each tagged armed. An ARMED gate builds its own ON config, so a _Skip
    # from it means a feature silently stopped wiring -- that FAILS, it must never print
    # [ok ] with exit 0 (the green-but-blind bug: step B's builder was all-off, so it
    # skipped on every run and the catch below reported it as ok). A not-yet-implemented
    # section is registered armed=False: it prints a loud [SKIP-unarmed] marker and is not
    # counted as ok, but does not fail. Distinguish "implemented but did not arm" from
    # "not implemented" by the flag, not by catching _Skip into ok.
    # Forward scaffolding: as of this commit BOTH members are armed=True; the armed=False
    # branch has NO user yet. It exists only so a genuinely future, not-yet-wired assembly
    # section can be registered loud-skip without weakening the A/B contract -- do not read
    # the branch as evidence any live section is unimplemented.
    armed_gates = (("step A engram", gate_step_a_engram, True),
                   ("step B mtp", gate_step_b_mtp, True))
    rc = 0
    for name, fn, armed in armed_gates:
        try:
            print(f"[ok ] {fn()}")
        except _Skip as s:
            if armed:
                print(f"[FAIL] {name}: armed gate SKIPPED: {s}")
                rc = 1
            else:
                print(f"[SKIP-unarmed] {name}: {s}")
        except AssertionError as e:
            # An armed gate's contract failing is a FAILURE, not a crash: print the named
            # section and keep going so every broken section reports in one run, then exit
            # nonzero. Letting it propagate printed only a traceback -- rc was already 1 but
            # there was no [FAIL] <section> line, so a reader/grep could not see which gate
            # failed (loudness gap, 2026-09-18).
            print(f"[FAIL] {name}: {e}")
            rc = 1
    if rc:
        print("p1 assembly key/dtype gate FAIL: an armed section did not wire or its contract failed")
        return rc
    print("p1 assembly key/dtype gate OK: baseline frozen; A/B armed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
