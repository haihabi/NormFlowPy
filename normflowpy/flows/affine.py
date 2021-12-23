import torch
from torch import nn
from normflowpy import base_nets
from normflowpy.base_flow import ConditionalBaseFlowLayer, UnconditionalBaseFlowLayer
from enum import Enum


class AffineScale(Enum):
    IDENTITY = 0
    SCALE_TANH = 1


class AffineConstantFlow(UnconditionalBaseFlowLayer):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class ConditionalAffineHalfFlow(ConditionalBaseFlowLayer):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, parity, net_class=base_nets.MLP, nh=24, scale=True, shift=True, condition_vector_size=1):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.condition_vector_size = condition_vector_size
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.condition_vector_size + (self.dim // 2), self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.condition_vector_size + (self.dim // 2), self.dim // 2, nh)

    def forward(self, x, cond=None):
        x0, x1 = x[:, ::2], x[:, 1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(torch.cat([x0, cond], dim=-1))
        t = self.t_cond(torch.cat([x0, cond], dim=-1))

        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z, cond=None):
        z0, z1 = z[:, ::2], z[:, 1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(torch.cat([z0, cond], dim=-1))
        t = self.t_cond(torch.cat([z0, cond], dim=-1))

        x0 = z0  # this was the same
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class AffineCouplingFlowVector(UnconditionalBaseFlowLayer):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, parity, net_class=base_nets.MLP, nh=24, scale=True, shift=True,
                 scale_mode: AffineScale = AffineScale.IDENTITY):
        super().__init__()
        self.dim = dim
        self.dim_half = self.dim // 2
        self.parity = parity
        self.scale = nn.Parameter(torch.ones([]), requires_grad=True)
        self.scale_mode = scale_mode
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)

    def forward(self, x):
        x0, x1 = x[:, :self.dim_half], x[:, self.dim_half:]
        if self.parity:
            x0, x1 = x1, x0

        s = self.s_cond(x0)
        t = self.t_cond(x0)
        if self.scale_mode == AffineScale.SCALE_TANH:
            s = self.scale * torch.tanh(s)

        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        z0, z1 = z[:, :self.dim_half], z[:, self.dim_half:]
        if self.parity:
            z0, z1 = z1, z0

        s = self.s_cond(z0)
        t = self.t_cond(z0)
        if self.scale_mode == AffineScale.SCALE_TANH:
            s = self.scale * torch.tanh(s)
        x0 = z0  # this was the same
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class AffineInjector(ConditionalBaseFlowLayer):
    """
    """

    def __init__(self, dim, net_class=base_nets.generate_mlp_class(24), scale=True, shift=True,
                 condition_vector_size=1):
        super().__init__()
        self.dim = dim
        # self.parity = parity
        self.condition_vector_size = condition_vector_size
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim)
        if scale:
            self.s_cond = net_class(self.condition_vector_size, self.dim)
        if shift:
            self.t_cond = net_class(self.condition_vector_size, self.dim)

    def forward(self, x, cond):

        s = self.s_cond(cond)
        t = self.t_cond(cond)
        z = torch.exp(s) * x + t  # transform this half as a function of the other

        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z, cond):
        # z
        s = self.s_cond(cond)
        t = self.t_cond(cond)
        x = (z - t) * torch.exp(-s)  # reverse the transform on this half

        log_det = torch.sum(-s, dim=1)
        return x, log_det


class AffineCouplingFlow2d(UnconditionalBaseFlowLayer):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, x_shape, parity, net_class=base_nets.RealNVPConvBaseNet, nh=4,
                 scale_mode: AffineScale = AffineScale.SCALE_TANH):
        super().__init__()
        self.dim = x_shape[0]
        self.parity = parity
        # self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        # self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.scale = nn.Parameter(torch.ones([]), requires_grad=True)
        self.scale_mode = scale_mode
        # if scale:
        self.input_size = self.dim // 2
        output_size = 2 * self.input_size
        self.add_module("s_cond", net_class(self.input_size, output_size, x_shape, nh))
        # self.s_cond = net_class(self.input_size, output_size, x_shape, nh)
        # if shift:
        #     self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)

    def forward(self, x):
        x0, x1 = x[:, :self.input_size, :, :], x[:, self.input_size:, :, :]
        if self.parity:
            x0, x1 = x1, x0

        s = self.s_cond(x0)
        t, log_scale = torch.split(s, split_size_or_sections=self.input_size, dim=1)
        if self.scale_mode == AffineScale.SCALE_TANH:
            s = self.scale * torch.tanh(log_scale)
        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=(1, 2, 3))
        return z, log_det

    def backward(self, z):
        z0, z1 = z[:, :self.input_size, :, :], z[:, self.input_size:, :, :]
        if self.parity:
            z0, z1 = z1, z0

        s = self.s_cond(z0)
        t, log_scale = torch.split(s, split_size_or_sections=self.input_size, dim=1)
        if self.scale_mode == AffineScale.SCALE_TANH:
            s = self.scale * torch.tanh(log_scale)

        x0 = z0  # this was the same
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=(1, 2, 3))
        return x, log_det
