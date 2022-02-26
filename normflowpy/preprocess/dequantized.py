from torch import nn
import torch


class Dequatization(nn.Module):
    def __init__(self, bit_width=1, threshold_max=1.0, threshold_min=0.0) -> None:
        super().__init__()
        self.scale_div_2 = (threshold_max - threshold_min) / ((2 ** bit_width - 1) )
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantization_noise = (torch.rand(x.shape, device=x.device)-0.5) * self.scale_div_2
        return torch.clamp(x + quantization_noise, self.threshold_min, self.threshold_max)
