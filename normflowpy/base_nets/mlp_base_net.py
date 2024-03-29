"""
Various helper network modules
"""

import torch
from torch import nn

from normflowpy.base_nets.made import MADE


class ScaledTanh(nn.Module):
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * init_scale,requires_grad=True)

    def forward(self, x):
        return self.p * torch.tanh(x)


class ScaledSigmoid(nn.Module):
    def __init__(self, init_scale=2.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * init_scale,requires_grad=False)
        self.eps = eps

    def forward(self, x):
        return self.p * torch.sigmoid(x) + self.eps


class LeafParam(nn.Module):
    """
    just ignores the input and outputs a parameter tensor, lol
    todo maybe this exists in PyTorch somewhere?
    """

    def __init__(self, n):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, n))

    def forward(self, x):
        return self.p.expand(x.size(0), self.p.size(1))


class PositionalEncoder(nn.Module):
    """
    Each dimension of the input gets expanded out with sins/coses
    to "carve" out the space. Useful in low-dimensional cases with
    tightly "curled up" data.
    """

    def __init__(self, freqs=(.5, 1, 2, 4, 8)):
        super().__init__()
        self.freqs = freqs

    def forward(self, x):
        sines = [torch.sin(x * f) for f in self.freqs]
        coses = [torch.cos(x * f) for f in self.freqs]
        out = torch.cat(sines + coses, dim=1)
        return out


def generate_mlp_class(n_layer=3, non_linear_function=nn.LeakyReLU, bias=True, output_nl=None):
    class MLPC(nn.Module):
        """ a simple n-layer MLP """

        # TODO: move n_hidden to generator
        def __init__(self, x_shape, nout, n_hidden):
            nin = x_shape[-1] // 2
            super().__init__()
            if n_layer == 1:  # The case of singe layer
                layer_list = [nn.Linear(nin, nout, bias=True)]
            else:
                layer_list = [nn.Linear(nin, n_hidden), non_linear_function()]
                for i in range(max(n_layer - 2, 0)):
                    layer_list.append(nn.Linear(n_hidden, n_hidden))
                    torch.nn.init.xavier_normal_(layer_list[-1].weight)
                    layer_list.append(non_linear_function())
                layer_list.append(nn.Linear(n_hidden, nout, bias=bias))
                torch.nn.init.xavier_normal_(layer_list[-1].weight)
            if output_nl is not None:
                layer_list.append(output_nl)
            self.net = nn.Sequential(*layer_list)

        def forward(self, x):
            return self.net(x)

    return MLPC


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class PosEncMLP(nn.Module):
    """
    Position Encoded MLP, where the first layer performs position encoding.
    Each dimension of the input gets transformed to len(freqs)*2 dimensions
    using a fixed transformation of sin/cos of given frequencies.
    """

    def __init__(self, nin, nout, nh, freqs=(.5, 1, 2, 4, 8)):
        super().__init__()
        self.net = nn.Sequential(
            PositionalEncoder(freqs),
            MLP(nin * len(freqs) * 2, nout, nh),
        )

    def forward(self, x):
        return self.net(x)


class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout, num_masks=1, natural_ordering=True)

    def forward(self, x):
        return self.net(x)
