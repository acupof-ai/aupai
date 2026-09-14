# ruff: noqa
# Prove: indexer STE gradient = gradient of (entry-branch weighted values) wrt isc through
# the soft_sel bridge, computed with ONLY soft_sel (small B,H,T,NB), no T*T attention.
import torch
torch.manual_seed(1)
B,H,T,NB,D=1,1,6,5,3
isc=torch.randn(B,H,T,NB,dtype=torch.float64,requires_grad=True)
vis=torch.rand(B,H,T,NB)>0.2; vis[...,0]|=True
sel=torch.zeros_like(isc,dtype=torch.bool)
sel.scatter_(-1,isc.detach().masked_fill(~vis,float('-inf')).topk(3,-1).indices,True); sel&=vis
# materialized: ste gates attention entry weights. upstream "value contribution" G[b,h,t,n]
# = w_entry[t,n] @ vc (here fold a per-entry scalar output sensitivity g_n).
g=torch.randn(B,H,T,NB,dtype=torch.float64)  # dL/d(effective entry weight)
alive=vis.any(-1,keepdim=True)
soft_sel=torch.softmax(torch.where(alive,isc.masked_fill(~vis,float('-inf')),torch.zeros_like(isc)),-1)*alive
ste=sel.to(torch.float64)+soft_sel-soft_sel.detach()
L=(g*ste).sum(); L.backward()
ref=isc.grad.clone()
# analytic: only the soft_sel term differentiates (hard part const); dL/dsoft = g,
# softmax Jacobian: soft*(g - sum_n soft*g)
isc.grad=None
sm=soft_sel
num=sm*(g-(sm*g).sum(-1,keepdim=True))
print("indexer grad analytic maxdiff", float((num-ref).abs().max()))