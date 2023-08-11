import torch

from normflowpy.base_flow import UnconditionalBaseFlowLayer


class ToComplex(UnconditionalBaseFlowLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.stack([torch.real(x), torch.imag(x)], dim=-1), 0

    def backward(self, z):
        real, imag = torch.split(z, split_size_or_sections=1, dim=-1)
        x = (real + 1j * imag)
        return x.squeeze(dim=-1), 0
