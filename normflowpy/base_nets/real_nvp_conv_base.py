import torch
from torch import nn
import numpy as np


class EdgePadding(nn.Module):
    def __init__(self, x_shape, padding=1):
        super().__init__()
        a = padding
        b = padding
        hw_shape = x_shape[1:3]
        hw_shape[0] += 2 * padding
        hw_shape[1] += 2 * padding
        pad = np.zeros([1, 1] + hw_shape, dtype='float32')
        pad[:, 0, :a, :] = 1.
        pad[:, 0, -a:, :] = 1.
        pad[:, 0, :, :b] = 1.
        pad[:, 0, :, -b:] = 1.
        self.pad_constant = nn.Parameter(torch.tensor(pad), requires_grad=False)
        self.simple_pad = nn.ZeroPad2d(padding)

    def forward(self, x):
        x_tilde = self.simple_pad(x)
        return torch.cat([x_tilde, self.pad_constant], dim=1)


class LogExpScale(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.logs = nn.Parameter(torch.randn(n_channels))

    def forward(self, x):
        return x * torch.exp(self.logs.reshape([1, -1, 1, 1]) * 3)


class RealNVPConvBaseNet(nn.Module):
    def __init__(self, n_channels, n_outputs, x_shape, width=512, activation_function=nn.ReLU):
        super().__init__()
        self.n_channels = n_channels
        self.num_output = n_outputs
        module_list = [nn.Conv2d(self.n_channels, width, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                       nn.BatchNorm2d(width, affine=False, eps=1e-4),
                       activation_function(),
                       nn.Conv2d(width, width, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
                       nn.BatchNorm2d(width, affine=False, eps=1e-4),
                       activation_function(),
                       EdgePadding(x_shape),
                       nn.Conv2d(width + 1, self.num_output, kernel_size=(3, 3), padding=(0, 0), stride=(1, 1)),
                       LogExpScale(self.num_output)
                       ]
        self.seq = nn.Sequential(*module_list)

    def forward(self, x):
        for i, m in enumerate(self.seq):
            x = m(x)
        return x
