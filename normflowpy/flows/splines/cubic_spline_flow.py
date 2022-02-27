import torch

import torch.nn.functional as F
from normflowpy.base_nets import generate_mlp_class
from normflowpy.base_flow import UnconditionalBaseFlowLayer
from normflowpy.flows.splines.cubic_spline_function import unconstrained_cubic_spline

M = 2


class CSF_CL(UnconditionalBaseFlowLayer):
    """ Cubic Spline flow, coupling layer, [Durkan et al. 2019] """

    def __init__(self, dim, K=5, B=3, hidden_dim=8, base_network=generate_mlp_class()):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B

        self.f1 = base_network([dim], (M * K + 2) * dim // 2, hidden_dim)
        self.f2 = base_network([dim], (M * K + 2) * dim // 2, hidden_dim)

    def split_param(self, in_param):
        out = in_param.reshape(-1, self.dim // 2, M * self.K + 2)
        DL = out[:, :, -1].unsqueeze(dim=-1)
        DH = out[:, :, -2].unsqueeze(dim=-1)
        W, H = torch.split(out[:, :, :-2], self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        return W, H, DL, DH

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        out = self.f1(lower)
        W, H, D_L, D_H = self.split_param(out)
        upper, ld = unconstrained_cubic_spline(upper, W, H, D_L,D_H, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)

        out = self.f2(upper)
        W, H, D_L, D_H = self.split_param(out)
        lower, ld = unconstrained_cubic_spline(lower, W, H, D_L, D_H, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)

        return torch.cat([lower, upper], dim=1), log_det

    def backward(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device)
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        out = self.f2(upper)
        W, H, DL, DH = self.split_param(out)
        lower, ld = unconstrained_cubic_spline(lower, W, H, DL, DH, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        out = self.f1(lower)
        W, H, DL, DH = self.split_param(out)
        upper, ld = unconstrained_cubic_spline(upper, W, H, DL, DH, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        return torch.cat([lower, upper], dim=1), log_det
