from torch import nn
import torch
import numpy as np
from normflowpy.base_flow import UnconditionalBaseFlowLayer
from normflowpy.flows.helpers import safe_log


class Sigmoid(UnconditionalBaseFlowLayer):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        z = torch.logit(x, eps=self.eps)

        log_det_jacob = -torch.sum(safe_log(x) + safe_log(1 - x), dim=-1)
        return z, log_det_jacob

    def backward(self, z):
        x_sig = torch.sigmoid(z)
        x = (1 - 2 * self.eps) * x_sig + self.eps
        dxdz = (1 - 2 * self.eps) * x_sig * (1 - x_sig)
        log_det_jacob = torch.sum(safe_log(dxdz), dim=-1)
        return x, log_det_jacob
