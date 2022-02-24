from torch import nn
import torch


class Dequatization(nn.Module):
    def __init__(self, scale=1) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantization_noise = torch.rand(x.shape, device=x.device) * self.scale
        return x + quantization_noise
