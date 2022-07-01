from typing import List

import torch
import math
from torch import nn
from normflowpy import base_nets
from normflowpy.base_flow import ConditionalBaseFlowLayer, UnconditionalBaseFlowLayer
from normflowpy.flows import helpers
import numpy as np


class BaseAffineCoupling(object):
    def __init__(self, parity, input_size, neighbor_splitting=False):
        self.neighbor_splitting = neighbor_splitting
        self.parity = parity
        self.input_size = input_size

    def split_input(self, zx):
        if self.neighbor_splitting:
            zx0, zx1 = zx[:, 0::2], zx[:, 1::2]
        else:
            zx0, zx1 = zx[:, :self.input_size], zx[:, self.input_size:]
        if self.parity:
            zx0, zx1 = zx1, zx0
        return zx0, zx1

    def joint_output(self, zx0, zx1):
        if self.parity:
            zx0, zx1 = zx1, zx0
        if self.neighbor_splitting:
            n = zx0.shape[1] + zx1.shape[1]
            connect = torch.stack(
                [zx0[:, math.floor(i / 2)] if i % 2 == 0 else zx1[:, math.floor(i / 2)] for i in range(n)], dim=1)
            return connect
        else:
            return torch.cat([zx0, zx1], dim=1)


class AffineConstantFlow(UnconditionalBaseFlowLayer):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, x_shape, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, *x_shape, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, *x_shape, requires_grad=True)) if shift else None

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s.reshape([1, -1]), dim=1)
        return z, log_det

    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s.reshape([1, -1]), dim=1)
        return x, log_det


class UserDefinedAffineFlow(UnconditionalBaseFlowLayer):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        self.s = nn.Parameter(torch.ones(1, requires_grad=False) * scale)
        self.t = nn.Parameter(torch.ones(1, requires_grad=False) * shift)

    def forward(self, x):
        z = (x - self.t) / self.s
        m = np.prod(x.shape[1:])
        log_det = -m * torch.log(self.s)
        return z, log_det

    def backward(self, z):
        x = self.s * z + self.t
        m = np.prod(x.shape[1:])
        log_det = m * torch.log(self.s)
        return x, log_det


class ConditionalAffineCoupling(ConditionalBaseFlowLayer, BaseAffineCoupling):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, x_shape, parity, condition_vector_size, net_class=base_nets.MLP, nh=24, scale=True, shift=True):
        super().__init__()
        BaseAffineCoupling.__init__(self, parity, x_shape[0] // 2)
        self.condition_vector_size = condition_vector_size
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.input_size)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.input_size)
        if scale:
            self.s_cond = net_class(self.condition_vector_size + self.input_size, self.input_size, nh)
        if shift:
            self.t_cond = net_class(self.condition_vector_size + self.input_size, self.input_size, nh)

    def forward(self, x, cond=None, **kwargs):
        x0, x1 = self.split_input(x)

        s = self.s_cond(torch.cat([x0, cond], dim=-1))
        t = self.t_cond(torch.cat([x0, cond], dim=-1))

        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other

        z = self.joint_output(z0, z1)

        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z, cond=None, **kwargs):

        z0, z1 = self.split_input(z)
        s = self.s_cond(torch.cat([z0, cond], dim=-1))
        t = self.t_cond(torch.cat([z0, cond], dim=-1))

        x0 = z0  # this was the same
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half

        x = self.joint_output(x0, x1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class AffineInjector(ConditionalBaseFlowLayer):
    def __init__(self, x_shape: list, cond_name_list: List[str], condition_vector_size: int, n_hidden: int,
                 net_class: nn.Module = base_nets.generate_mlp_class(), scale: bool = True,
                 shift: bool = True):
        """

        :param x_shape:
        :param cond_name:
        :param condition_vector_size:
        :param n_hidden:
        :param net_class:
        :param scale:
        :param shift:
        """
        super().__init__()
        self.dim = x_shape[0]
        self.cond_name_list = cond_name_list
        self.condition_vector_size = condition_vector_size
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim)
        if scale:
            self.s_cond = net_class([2 * self.condition_vector_size], self.dim, n_hidden)
        if shift:
            self.t_cond = net_class([2 * self.condition_vector_size], self.dim, n_hidden)

    def build_condition_vector(self, x, **kwargs):
        cond_list = []
        for cond_name in self.cond_name_list:
            cond = kwargs[cond_name]
            if cond.shape[0] == 1:
                cond = cond.repeat([x.shape[0], *[1 for _ in range(len(cond.shape[1:]))]])
            cond_list.append(cond)
        return torch.cat(cond_list, dim=-1)

    def forward(self, x, **kwargs):
        cond = self.build_condition_vector(x, **kwargs)
        s = self.s_cond(cond)
        t = self.t_cond(cond)
        z = torch.exp(s) * x + t  # transform this half as a function of the other

        log_det = torch.sum(s.reshape([s.shape[0], -1]), dim=1)
        return z, log_det

    def backward(self, z, **kwargs):
        cond = self.build_condition_vector(z, **kwargs)
        s = self.s_cond(cond)
        t = self.t_cond(cond)
        x = (z - t) * torch.exp(-s)  # reverse the transform on this half

        log_det = torch.sum(-s.reshape([x.shape[0], -1]), dim=1)
        return x, log_det


class AffineCoupling(UnconditionalBaseFlowLayer, BaseAffineCoupling):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, x_shape, parity, net_class=base_nets.RealNVPConvBaseNet, nh=4, scale=True, shift=True,
                 neighbor_splitting=False):
        super().__init__()
        BaseAffineCoupling.__init__(self, parity, x_shape[0] // 2, neighbor_splitting=neighbor_splitting)
        output_size = (int(scale) + int(shift)) * self.input_size
        self.add_module("s_cond", net_class(x_shape, output_size, nh))
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        x0, x1 = self.split_input(x)
        s = self.s_cond(x0)
        if self.scale and self.shift:
            t, s = torch.split(s, split_size_or_sections=self.input_size, dim=1)  # Split output to scale and shift
        elif self.scale:
            t = 0
        elif self.shift:
            t = s
            s = torch.zeros([x0.shape[0], 1], device=x.device)  # TODO: change to correct shape
        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        z = self.joint_output(z0, z1)
        log_det = torch.sum(s.reshape([x0.shape[0], -1]), dim=(1))
        return z, log_det

    def backward(self, z):
        z0, z1 = self.split_input(z)
        s = self.s_cond(z0)
        if self.scale and self.shift:
            t, s = torch.split(s, split_size_or_sections=self.input_size, dim=1)
        elif self.scale:
            t = 0
        elif self.shift:
            t = s
            s = torch.zeros([z0.shape[0], 1], device=z.device)

        x0 = z0  # this was the same

        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        x = self.joint_output(x0, x1)
        log_det = torch.sum(-s.reshape([x0.shape[0], -1]), dim=(1))
        return x, log_det


# import torch.nn as nn
#
# from models.flowplusplus import log_dist as logistic
# from models.flowplusplus.nn import NN


class MixLogCoupling(UnconditionalBaseFlowLayer, BaseAffineCoupling):
    NUM_PARAM_AFFINE = 2
    NUM_PARAM_MIX = 3

    def __init__(self, x_shape, net_class: nn.Module = base_nets.generate_mlp_class(), n_hidden: int = 24,
                 parity=False, k_clusters: int = 3,
                 neighbor_splitting=False):
        """

        :param x_shape:
        :param net_class:
        :param n_hidden:
        :param parity:
        :param neighbor_splitting:
        """
        super().__init__()
        BaseAffineCoupling.__init__(self, parity, x_shape[0] // 2, neighbor_splitting=neighbor_splitting)

        self.k_clusters = k_clusters
        output_size = self.NUM_PARAM_AFFINE * self.input_size + k_clusters * self.NUM_PARAM_MIX
        self.nn = net_class(x_shape, output_size, n_hidden)

    def forward(self, x):
        x0, x1 = self.split_input(x)
        param_x0 = self.nn(x0)
        affine_param = param_x0[:, :self.NUM_PARAM_AFFINE * self.input_size]
        cluster_param = param_x0[:, self.NUM_PARAM_AFFINE * self.input_size:]
        a, b = torch.split(affine_param, self.input_size, dim=-1)
        pi, mu, s = torch.split(cluster_param, self.k_clusters, dim=-1)
        # a, b, pi, mu, s = torch.split(param_x0, self.input_size, dim=-1)

        # if reverse:
        # else:
        out = helpers.logistic.mixture_log_cdf(x1, pi, mu, s).exp()
        out = out.clamp(1e-5, 1. - 1e-5)  # Add this for numeric stability
        out, scale_ldj = helpers.logistic.inverse(out)
        out = (out + b) * a.exp()

        ######################
        # Build log det
        ######################
        logistic_ldj = helpers.logistic.mixture_log_pdf(x1, pi, mu, s)
        log_det = (logistic_ldj + scale_ldj + a).flatten(1).sum(-1)

        z0 = x0  # untouched half
        z1 = out  # transform this half as a function of the other
        z = self.joint_output(z0, z1)

        return z, log_det

    def backward(self, z):
        z0, z1 = self.split_input(z)
        param_z0 = self.nn(z0)
        affine_param = param_z0[:, :self.NUM_PARAM_AFFINE * self.input_size]
        cluster_param = param_z0[:, self.NUM_PARAM_AFFINE * self.input_size:]
        a, b = torch.split(affine_param, self.input_size, dim=-1)
        pi, mu, s = torch.split(cluster_param, self.k_clusters, dim=-1)

        out = z1 * a.mul(-1).exp() - b
        out, scale_ldj = helpers.logistic.inverse(out, reverse=True)
        out = out.clamp(1e-5, 1. - 1e-5)
        out = helpers.logistic.mixture_inv_cdf(out, pi, mu, s)

        ######################
        # Build log det
        ######################
        logistic_ldj = helpers.logistic.mixture_log_pdf(out, pi, mu, s)
        log_det = - (a + scale_ldj + logistic_ldj).flatten(1).sum(-1)

        x0 = z0  # untouched half
        x1 = out  # transform this half as a function of the other
        x = self.joint_output(x0, x1)

        return x, log_det
