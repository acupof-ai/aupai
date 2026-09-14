# ruff: noqa
# validate the ragged-entry flash concept on CPU-equivalent math first
import torch
B,H,T,NB,kk,D=1,1,6,5,3,2
vis=torch.tensor([[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=torch.bool)[:,None,:]  # T,1,NB
sel=torch.zeros(T,1,NB,dtype=torch.bool)
idx=torch.tensor([[0,1,0],[0,1,2],[1,2,3],[0,1,2],[1,3,4],[0,2,3]])  # T,kk (last dup for fill)
sel.scatter_(-1,idx[:,:,None] if False else idx.unsqueeze(1).expand(T,1,kk),True)
sel&=vis
valid=sel.sum(-1)  # T,1
print("valid per q:", valid.flatten().tolist())
# ragged flatten: for each q take the first valid-count selected entries in entry order
# gather: sort sel so valid entries come first
order=sel.float().argsort(dim=-1,descending=True,stable=True)  # T,1,NB
sorted_sel=sel.gather(-1,order)
print("order",order.flatten(1).tolist())
# positions and valid mask are consistent
print("first valid-count entries selected:", all(bool(sorted_sel[t,0,:valid[t,0]].all()) for t in range(T)))