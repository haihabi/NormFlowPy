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
