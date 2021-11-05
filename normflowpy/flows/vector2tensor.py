import torch
from normflowpy.base_flow import UnconditionalBaseFlowLayer


class Tensor2Vector(UnconditionalBaseFlowLayer):
    def __init__(self, in_shape):
        super().__init__()
        self.shape = in_shape

    def forward(self, x):
        z = x.flatten()

        log_det = 0
        return z, log_det

    def backward(self, z):
        x = z.reshape([-1, *self.shape])

        log_det = 0
        return x, log_det
