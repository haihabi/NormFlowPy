import numpy as np
import torch
from torch import nn


class NumPyArray2Tensor(nn.Module):
    def forward(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x)
