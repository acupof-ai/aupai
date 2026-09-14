# ruff: noqa
import torch
torch.manual_seed(0)
B,H,T,Ke,Kw,Dv=2,2,5,4,6,3
se=torch.randn(B,H,T,Ke,dtype=torch.float64,requires_grad=True)
sw=torch.randn(B,H,T,Kw,dtype=torch.float64,requires_grad=True)
Ve=torch.randn(B,H,Ke,Dv,dtype=torch.float64,requires_grad=True)
Vw=torch.randn(B,H,Kw,Dv,dtype=torch.float64,requires_grad=True)
me=torch.rand(B,H,T,Ke)>0.3; mw=torch.rand(B,H,T,Kw)>0.4; neg=float("-inf")
def ref():
    c=torch.cat([se.masked_fill(~me,neg),sw.masked_fill(~mw,neg)],-1)
    al=c.any(-1,keepdim=True)
    p=torch.softmax(torch.where(al,c,torch.zeros_like(c)),-1)*al
    return p@torch.cat([Ve,Vw],2)
class Split(torch.autograd.Function):
    @staticmethod
    def forward(ctx,se,sw,Ve,Vw):
        pe=torch.softmax(se.masked_fill(~me,neg),-1)
        pw=torch.softmax(sw.masked_fill(~mw,neg),-1)
        ae_=me.any(-1,keepdim=True);aw_=mw.any(-1,keepdim=True)
        le=se.masked_fill(~me,neg).logsumexp(-1,keepdim=True)
        lw=sw.masked_fill(~mw,neg).logsumexp(-1,keepdim=True)
        m=torch.maximum(torch.where(ae_,le,torch.full_like(le,neg)),torch.where(aw_,lw,torch.full_like(lw,neg)))
        ae=torch.where(ae_,(le-m).exp(),torch.zeros_like(le));aw=torch.where(aw_,(lw-m).exp(),torch.zeros_like(lw))
        den=ae+aw;ce=ae/den;cw=aw/den
        oe=pe@Ve;ow=pw@Vw
        ctx.save_for_backward(pe,pw,ce,cw,Ve,Vw)
        return ce*oe+cw*ow
    @staticmethod
    def backward(ctx,dy):
        pe,pw,ce,cw,Ve,Vw=ctx.saved_tensors
        # feed each branch a scaled dout = c_j*dy and the BRANCH dLSE so its softmax bwd
        # yields the within-branch score gradient. flash computes dp=p(dV@dout-dLSE).
        # branch j dLSE = sum_d (c_j dy).o_j  -- the lse of the weighted branch output.
        doe=ce*(dy@Ve.transpose(-1,-2)); dow=cw*(dy@Vw.transpose(-1,-2))
        dLe=(ce*dy@Ve.transpose(-1,-2)).sum(-1,keepdim=True)  # placeholder
        # correct: branch score grad must equal concat: dp_global = c_j p (dy.V_j - dL_g)
        dLg=(dy*( (pe@Ve) and None) ) if False else None
        # build directly: dse = c_j pe (dy.Ve - dL_global)
        oe=pe@Ve;ow=pw@Vw; y=(ce*oe+cw*ow)
        dLg=(dy*y).sum(-1,keepdim=True)
        dse=torch.where(me,ce*pe*(dy@Ve.transpose(-1,-2)-dLg),torch.zeros_like(pe))
        dsw=torch.where(mw,cw*pw*(dy@Vw.transpose(-1,-2)-dLg),torch.zeros_like(pw))
        dVe=torch.einsum('bhtd,bhte->bhed',dy*ce,pe)
        dVw=torch.einsum('bhtd,bhtw->bhwd',dy*cw,pw)
        return dse,dsw,dVe,dVw
ys=Split.apply(se,sw,Ve,Vw); print("fwd",float((ref()-ys).abs().max()))
ref().sum().backward()
ge_=(se.grad.clone(),sw.grad.clone(),Ve.grad.clone(),Vw.grad.clone())
se.grad=sw.grad=Ve.grad=Vw.grad=None
ys.sum().backward()
for n,a,b in zip("se sw Ve Vw".split(),(se.grad,sw.grad,Ve.grad,Vw.grad),ge_):
    print(n,float((a-b).abs().max()))