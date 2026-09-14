# ruff: noqa
import torch
B,T,NB,H,D,kk=1,3,4,1,2,3
kc=torch.randn(B,H,NB,D,dtype=torch.float64,requires_grad=True)
sel=torch.tensor([[1,1,0,0],[1,1,1,0],[1,0,1,1]],dtype=torch.bool)
n_sel=sel.sum(-1)
order=sel.float().argsort(dim=-1,descending=True,stable=True)
idx=order[:,:kk]
kg=torch.gather(kc.unsqueeze(2).expand(B,H,T,NB,D),3,
                idx[None,None,:,:,None].expand(B,H,T,kk,D))
keep=(torch.arange(kk)[None,:]<n_sel[:,None])[None,:,None,:,None].expand_as(kg.permute(0,2,1,3,4))
flat=kg.permute(0,2,1,3,4)[keep]
g=torch.randn_like(flat); flat.backward(g)
print("grad finite:", bool(torch.isfinite(kc.grad).all()), "nonzero slots:", int((kc.grad.abs()>0).all(-1).sum()), "of", kc.numel()//D)
print("grad[0,0]", kc.grad[0,0].sum(-1).tolist())