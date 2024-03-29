from torch import nn
import torch


class Dequatization(nn.Module):
    def __init__(self, bit_width=1, threshold_max=1.0, threshold_min=0.0) -> None:
        super().__init__()
        self.delta = (threshold_max - threshold_min) / (2 ** bit_width - 1)
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantization_noise = (torch.rand(x.shape, device=x.device) - 0.5) * self.delta
        return x + quantization_noise
