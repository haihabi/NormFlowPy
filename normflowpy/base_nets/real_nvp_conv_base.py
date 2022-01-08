import torch
from torch import nn
import numpy as np


def generate_pad_constant(x, padding):
    x_shape = list(x.shape)
    a = padding
    b = padding
    hw_shape = x_shape[2:4]
    hw_shape[0] += 2 * padding
    hw_shape[1] += 2 * padding
    pad = np.zeros([1, 1] + hw_shape, dtype='float32')
    pad[:, 0, :a, :] = 1.
    pad[:, 0, -a:, :] = 1.
    pad[:, 0, :, :b] = 1.
    pad[:, 0, :, -b:] = 1.
    return torch.tensor(pad)


class EdgePadding(nn.Module):
    def __init__(self, x_shape, padding=1):
        super().__init__()
        # a = padding
        # b = padding
        # hw_shape = x_shape[1:3]
        # hw_shape[0] += 2 * padding
        # hw_shape[1] += 2 * padding
        # pad = np.zeros([1, 1] + hw_shape, dtype='float32')
        # pad[:, 0, :a, :] = 1.
        # pad[:, 0, -a:, :] = 1.
        # pad[:, 0, :, :b] = 1.
        # pad[:, 0, :, -b:] = 1.
        # self.pad_constant = torch.tensor(pad)
        self.padding = padding
        self.simple_pad = nn.ZeroPad2d(padding)

    def forward(self, x):
        x_tilde = self.simple_pad(x)
        return torch.cat([x_tilde, generate_pad_constant(x, self.padding).repeat([x.shape[0], 1, 1, 1]).to(x.device)],
                         dim=1)


class LogExpScale(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.logs = nn.Parameter(torch.randn(1, n_channels))

    def forward(self, x):
        return x * torch.exp(self.logs.reshape([1, -1, 1, 1]) * 3)


class RealNVPConvBaseNet(nn.Module):
    def __init__(self, x_shape, n_outputs, width=512, activation_function=nn.ReLU, edge_bias=True):
        super().__init__()
        self.n_channels = x_shape[0] // 2
        self.num_output = n_outputs
        self.scale = nn.Parameter(torch.ones([]), requires_grad=True)
        p = 0 if edge_bias else 1
        module_list = [nn.Conv2d(self.n_channels, width, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                       nn.BatchNorm2d(width, affine=False, eps=1e-4),
                       activation_function(),
                       nn.Conv2d(width, width, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
                       nn.BatchNorm2d(width, affine=False, eps=1e-4),
                       activation_function(),
                       EdgePadding(x_shape) if edge_bias else nn.Identity(),
                       nn.Conv2d(width + edge_bias, self.num_output, kernel_size=(3, 3), padding=(p, p), stride=(1, 1)),
                       LogExpScale(self.num_output)
                       ]
        self.seq = nn.Sequential(*module_list)

    def forward(self, x):
        for i, m in enumerate(self.seq):
            x = m(x)
        t, log_scale = torch.split(x, split_size_or_sections=self.n_channels, dim=1)
        s = self.scale * torch.tanh(log_scale)
        return torch.cat([t, s], dim=1)
