#!/usr/bin/env bash
# Make a pod CONTAINER ready to run the repo: assert the image-baked kernel,
# install the pinned Python surface, verify the CUDA build. That is the whole job.
#
# What this deliberately is NOT:
#   It does not fetch data, build corpora, train a tokenizer, or dry-run a mix.
#   Those are the V4.1 gate-domain rebuild and live in their own recipes (they
#   change with the mix; copying them here would make a second truth source that
#   drifts):
#     docs/standards/infra_persistent_rebuild_0916.md   (node / disk / mount)
#     docs/standards/data_pipeline_rebuild_0916.md      (fetch -> build -> decontam)
#
#   bash scripts/bootstrap_pod.sh            # image gate, deps, cuda, caches
#   bash scripts/bootstrap_pod.sh image      # just the image-baked kernel gate
#   bash scripts/bootstrap_pod.sh deps       # just the pinned pip surface
#   bash scripts/bootstrap_pod.sh cuda       # assert torch is the +cu CUDA build
#   bash scripts/bootstrap_pod.sh caches     # HOST only: re-attach NVMe token-cache mount
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE=${1:-all}
LOG=/tmp/bootstrap.log
mkdir -p data
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { say "BOOTSTRAP FAILED at $1"; exit 1; }
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# --- image: the kernel pip can never install, checked FIRST ------------------
# HARD PRECONDITION -- the base image must already carry flash-attn 4
# (`flash_attn` / `flash_attn.cute`). train.py sets HAS_FA from that import and
# REFUSES to launch on GPU without it; it is NOT on any pip index (verified
# 2026-09-20: not on PyPI, the volc mirror, or any reachable byted mirror; no
# wheel on disk), so it can only exist by being baked into the image. A dependency
# that holds only because "the image happens to contain it" is asserted, not
# documented -- a comment was the only guard the three times this silently
# regressed (cutlass / liger_kernel / flash_attn). Fail before the long pip.
if want image; then
  say "stage image"
  python3 - <<'PY' || die "image (flash_attn is image-baked; provision an image that contains it)"
import sys

# Mirror model.py's HAS_FA resolution exactly: flash_attn first, then the flash-attn-4
# flash_attn.cute namespace (on this image .cute is the one present). A find_spec on the
# dotted child raises when the parent is absent; an attempted import is the real test.
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: F401
except ImportError:
    try:
        from flash_attn.cute import flash_attn_func, flash_attn_varlen_func  # noqa: F401
    except ImportError:
        print(
            "  GATE FAIL flash_attn is missing. train.py needs flash-attn 4 (import "
            "flash_attn / flash_attn.cute) to set HAS_FA and refuses to launch on GPU "
            "without it. It is NOT pip-installable (no public/index wheel) and must be "
            "BAKED INTO THE BASE IMAGE. This image lacks it -- rebuild/provision from an "
            "image containing flash-attn 4.0.0b15; do not try to pip install it.",
            file=sys.stderr,
        )
        sys.exit(1)
print("image gate ok: flash_attn is present (image-baked).")
PY
  say "stage image: done"
fi

# --- deps: pinned training surface ------------------------------------------
# pod-constraints.txt is the pod's `pip freeze` minus the unresolvable lines (OS
# apt packages python-apt/devscripts; the flashinfer cu129 sidecar; and the sglang
# serving stack from deleted /tmp wheels and an editable git -- none of which
# train/eval imports). It is a CONSTRAINTS file: pip validates a pin only when it
# selects that package, so unselected apt pins never break the install. The direct
# list names what training/eval imports; torch/torchao resolve to +cu129 from the
# PyTorch index (plain PyPI serves no cu129 build -- verified torch==2.11.0+cu129).
# Refreeze from a healthy pod and replace the file; do not hand-edit versions.
if want deps; then
  say "stage deps"
  python3 -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    -c scripts/pod-constraints.txt \
    -r scripts/pod-training-direct.txt || die "deps"
  say "stage deps: done"
fi

# --- cuda: assert the installed torch is the CUDA build ----------------------
if want cuda; then
  say "stage cuda"
  python3 - <<'PY' || die "cuda (torch resolved to a non-CUDA build)"
import sys

try:
    import torch
except Exception as e:  # noqa: BLE001 - name the missing import rather than mask it
    print(f"  GATE FAIL torch does not import: {e!r} -- run the deps stage first.",
          file=sys.stderr)
    sys.exit(1)
if "+cu" not in torch.__version__:
    print(
        f"  GATE FAIL torch is {torch.__version__}, not a +cu CUDA build. Reinstall "
        "with --extra-index-url https://download.pytorch.org/whl/cu129 (the deps "
        "stage sets it); a plain-PyPI torch cannot drive the GPU kernels.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"verify ok: torch {torch.__version__} is a CUDA build.")
PY
  say "stage cuda: done"
fi

# --- caches: re-attach the NVMe token-cache mount ---------------------------
# HOST ONLY and last (needs crictl, which does not exist in the container). This
# is current container-ready infra, not the deleted data pipeline: train.py's
# AUPAI_TOKEN_CACHE_DIR refusal tells the operator to run exactly this, and the
# mount lives only as long as the container, so a restart re-runs it. Idempotent --
# the script verifies first and exits 0 when the mount is already live.
if want caches; then
  say "stage caches"
  if command -v crictl >/dev/null 2>&1; then
    python3 scripts/attach_nvme_caches.py ${ATTACH_ARGS:-} || die "attach_nvme_caches"
    say "stage caches: done (mounted and verified by reading)"
  else
    # Not a failure and not a silent skip: crictl absent means this is the container
    # view, so nobody should read a green log as "mount checked".
    say "stage caches: SKIPPED -- crictl is absent, so this is the container view, not the host."
    say "  Run it on the HOST: tn exec 'cd /work/aupai && bash scripts/bootstrap_pod.sh caches'"
    say "  The mount was NOT verified by this run."
  fi
fi

say "BOOTSTRAP: $STAGE complete -- container ready. Data/corpus and the launch are"
say "separate: see docs/standards/data_pipeline_rebuild_0916.md and infra_persistent_rebuild_0916.md."
