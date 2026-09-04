"""Does DDP accept a COO grad, and does the optimizer still see row-wise updates?

Asserted rather than reasoned: I just got a prediction backwards on this program by
arguing from arithmetic instead of measuring, so this runs both settings.
"""
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def probe(sparse):
    m = nn.Embedding(1024, 64, sparse=sparse).cuda()
    idx = torch.randint(0, 1024, (128,), device="cuda")
    out = m(idx).sum()
    out.backward()
    g = m.weight.grad
    kind = "COO" if g.is_sparse else "dense"
    touched = g._nnz() if g.is_sparse else int((g.abs().sum(1) > 0).sum())
    # SparseAdam refuses a dense grad; Adagrad accepts both. Which one works is the
    # question b0 actually has to answer.
    ok = {}
    for name, ctor in (("SparseAdam", torch.optim.SparseAdam), ("Adagrad", torch.optim.Adagrad)):
        mm = nn.Embedding(1024, 64, sparse=sparse).cuda()
        mm(idx).sum().backward()
        try:
            ctor([mm.weight], lr=0.1).step()
            ok[name] = "ok"
        except Exception as e:
            ok[name] = type(e).__name__ + ": " + str(e)[:60]
    return kind, touched, ok

if __name__ == "__main__":
    for s in (True, False):
        kind, touched, ok = probe(s)
        print(f"sparse={s}: grad={kind} rows_touched={touched} {ok}")
    # DDP: does it accept a module holding a sparse-grad parameter at all?
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        for s in (True, False):
            m = nn.Embedding(1024, 64, sparse=s).cuda()
            try:
                d = DDP(m, device_ids=[int(os.environ["LOCAL_RANK"])])
                idx = torch.randint(0, 1024, (128,), device="cuda")
                d(idx).sum().backward()
                g = m.weight.grad
                print(f"DDP sparse={s}: OK, grad={'COO' if g.is_sparse else 'dense'}")
            except Exception as e:
                print(f"DDP sparse={s}: {type(e).__name__}: {str(e)[:110]}")
        dist.destroy_process_group()
