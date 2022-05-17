import torch
from torch import nn


class BaseFlowLayer(nn.Module):
    """

    """

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        raise NotImplemented

    def backward(self, *args):
        raise NotImplemented


class ConditionalBaseFlowLayer(BaseFlowLayer):
    """

    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        raise NotImplemented

    def backward(self, z, **kwargs):
        raise NotImplemented


class UnconditionalBaseFlowLayer(BaseFlowLayer):
    """

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplemented

    def backward(self, z):
        raise NotImplemented


class BaseFlowModel(nn.Module):

    def forward(self, x, **kwargs):
        raise NotImplemented

    def backward(self, z, **kwargs):
        raise NotImplemented

    def nll(self, x, **kwargs):
        raise NotImplemented

    def nll_mean(self, x, **kwargs):
        return torch.mean(self.nll(x, **kwargs)) / x.shape[1:].numel()  # NLL per dim

    def sample(self, num_samples, temperature: float = 1, **kwargs):
        raise NotImplemented

    def sample_nll(self, num_samples, temperature=1.0, **kwargs):
        raise NotImplemented
