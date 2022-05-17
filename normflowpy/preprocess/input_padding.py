from torch import nn
import torch


class InputPadding(nn.Module):
    def __init__(self, padding_size=-1):
        super().__init__()
        self.padding_size = padding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding_shape = x.shape
        if self.padding_size > 0:
            padding_shape[1] = self.padding_size
        pad_noise = torch.rand(padding_shape, device=x.device)
        return torch.cat([x, pad_noise], dim=0)
