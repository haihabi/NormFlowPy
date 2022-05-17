import torch
from abc import ABC
from torch import nn
from torch.distributions import Distribution


class TrainablePrior(nn.Module, Distribution, ABC):
    def __init__(self, prior: Distribution, parameters_list: list):
        super().__init__()
        self.prior = prior
        for i, p in enumerate(parameters_list):
            setattr(self, f"p{i}", p)

    def log_prob(self, value):
        return self.prior.log_prob(value)

    def sample(self, sample_shape=torch.Size()):
        return self.prior.sample(sample_shape)

    @property
    def mean(self):
        return self.prior.mean
