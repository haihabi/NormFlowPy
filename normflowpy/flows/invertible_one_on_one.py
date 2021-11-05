import torch
from torch import nn
from torch.nn import functional as F
from normflowpy.base_flow import UnconditionalBaseFlowLayer


class BaseInvertible(UnconditionalBaseFlowLayer):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim, transpose=False):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = nn.Parameter(P, requires_grad=False)  # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S
        self.transpose = transpose

    def _assemble_W(self, device):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        if self.transpose:
            W = torch.transpose(W, dim0=0, dim1=1)
        return W


class InvertibleFullyConnected(BaseInvertible):
    def forward(self, x):
        W = self._assemble_W(x.device)
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z):
        W = self._assemble_W(z.device)
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


class InvertibleConv2d1x1(BaseInvertible):
    def __init__(self, dim):
        super().__init__(dim, True)

    def forward(self, x):
        W = self._assemble_W(x.device)
        W = W.reshape([W.shape[0], W.shape[1], 1, 1])
        z = F.conv2d(x, W, None, 1, 0, 1, 1)
        log_det = torch.sum(torch.log(torch.abs(self.S)))*x.shape[2]*x.shape[3]
        return z, log_det

    def backward(self, z):
        W = self._assemble_W(z.device)
        W_inv = torch.inverse(W)
        W_inv = W_inv.reshape([W_inv.shape[0], W_inv.shape[1], 1, 1])
        x = F.conv2d(z, W_inv, None, 1, 0, 1, 1)
        log_det = -torch.sum(torch.log(torch.abs(self.S)))*x.shape[2]*x.shape[3]
        return x, log_det
