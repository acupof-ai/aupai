# Sets torch._inductor.config.pad_dynamic_shapes from PAD_DYNAMIC_SHAPES before train.py
# compiles. A shim on PYTHONPATH so train.py stays frozen during stage 1 (t57, e1's find).
import os
if os.environ.get("PAD_DYNAMIC_SHAPES") == "1":
    try:
        import torch._inductor.config as _ic
        _ic.pad_dynamic_shapes = True
        print("[padshim] pad_dynamic_shapes = True", flush=True)
    except Exception as _e:
        print(f"[padshim] FAILED: {_e}", flush=True)
