# ruff: noqa
import torch
from flash_attn.cute import flash_attn_varlen_func as f

torch.manual_seed(0)
dev="cuda"
# B*T queries, each a 1-token segment in Q; K/V ragged with per-segment lengths
B,T,H,D=1,4,2,16
# two "queries": seg0 sees 3 keys, seg1 sees 2 keys
nq=B*T
q=torch.randn(nq,H,D,device=dev,dtype=torch.bfloat16)
# ragged KV: lengths per query segment
lens=torch.tensor([3,2,1,3],device=dev)
tot=int(lens.sum())
k=torch.randn(tot,H,D,device=dev,dtype=torch.bfloat16)
v=torch.randn(tot,H,D,device=dev,dtype=torch.bfloat16)
cuq=torch.arange(0,nq+1,dtype=torch.int32,device=dev)
cuk=torch.cat([torch.zeros(1,dtype=torch.int32,device=dev),lens.cumsum(0).to(torch.int32)])
out,lse=f(q,k,v,cu_seqlens_q=cuq,cu_seqlens_k=cuk,max_seqlen_q=1,max_seqlen_k=int(lens.max()),
          causal=False,return_lse=True,softmax_scale=D**-0.5)
# reference
acc=[]
s=0
for i in range(nq):
    sc=(q[i:i+1].transpose(0,1)@k[s:s+lens[i]].transpose(0,1).transpose(-1,-2))*(D**-0.5)
    o=torch.softmax(sc,-1)@v[s:s+lens[i]].transpose(0,1)
    acc.append(o.transpose(0,1))
    s+=int(lens[i])
ref=torch.cat(acc,0)
print("ragged varlen maxdiff", float((out.float()-ref.float()).abs().max()))
print("OK")