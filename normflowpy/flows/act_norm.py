import torch
from normflowpy.flows.affine import AffineConstantFlow
from torch import nn
from normflowpy.base_flow import UnconditionalBaseFlowLayer


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, eps=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        # first batch is used for init
        if self.data_dep_init_done == 0:
            assert self.s is not None and self.t is not None  # for now
            self.s.data = (-torch.log(torch.sqrt(x.var(dim=0, keepdim=True) + self.eps))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done.data = torch.ones(1, device=x.device)
        return super().forward(x)


class InputNorm(UnconditionalBaseFlowLayer):
    def __init__(self, mu, std):
        super().__init__()
        self.t = nn.Parameter(mu, requires_grad=False)
        self.s = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        z = (x - self.t) / self.s
        log_det = -torch.sum(torch.log(self.s), dim=1)
        return z, log_det

    def backward(self, z):
        x = self.s * z + self.t
        log_det = torch.sum(torch.log(self.s), dim=1)
        return x, log_det
