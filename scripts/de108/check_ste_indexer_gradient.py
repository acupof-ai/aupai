# ruff: noqa
import torch
# materialized entry path: sel hard-selects; ste = hard + soft_sel - soft_sel.detach()
# y = (w_entry * ste) @ vc ; w_entry = masked attention softmax over selected+window.
# Does indexer score isc (which makes sel/soft_sel) get a gradient?
T,NB,H,D=4,5,1,2
isc=torch.randn(H,T,NB,dtype=torch.float64,requires_grad=True)  # indexer scores
vc=torch.randn(H,NB,D,dtype=torch.float64,requires_grad=True)
sc=torch.randn(H,T,NB,dtype=torch.float64)  # attention scores q@kc
sel=torch.zeros(H,T,NB,dtype=torch.bool); sel.scatter_(-1,isc.topk(2,dim=-1).indices,True)
vis=torch.ones_like(sel)
alive=vis.any(-1,keepdim=True)
soft_sel=torch.softmax(torch.where(alive,isc,torch.zeros_like(isc)),-1)*alive
ste=sel.to(soft_sel.dtype)+soft_sel-soft_sel.detach()
# attention over selected entries only (window omitted for isolation)
s=sc.masked_fill(~sel,float('-inf'))
w=torch.softmax(s,-1)  # H,T,NB selected-only
y=(w*ste)@vc
y.sum().backward()
print("indexer isc grad nonzero:", bool(isc.grad is not None and isc.grad.abs().sum()>0))
print("vc grad nonzero:", bool(vc.grad.abs().sum()>0))
# HARD gather equivalent WITHOUT soft bridge:
isc2=isc.detach().clone().requires_grad_(True)
sel2=torch.zeros_like(sel2); sel2.scatter_(-1,isc2.topk(2,-1).indices,True)
# gather selected vc/sc only
idx=isc2.topk(2,-1).indices  # H,T,2
ke2=torch.gather(vc.unsqueeze(1).expand(H,T,NB,D),2,idx.unsqueeze(-1).expand(H,T,2,D))
sc2=torch.gather(sc,2,idx)
w2=torch.softmax(sc2,-1); y2=(w2@ke2)
y2.sum().backward()
print("hard-gather isc grad nonzero:", bool(isc2.grad is not None and isc2.grad.abs().sum()>0))