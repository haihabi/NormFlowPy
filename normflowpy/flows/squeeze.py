from normflowpy.base_flow import UnconditionalBaseFlowLayer


def space2depth(x, bs):
    N, C, H, W = x.size()
    x = x.view(N, C, H // bs, bs, W // bs, bs)  # (N, C, H//bs, bs, W//bs, bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
    x = x.view(N, C * (bs ** 2), H // bs, W // bs)  # (N, C*bs^2, H//bs, W//bs)
    return x


def depth2space(x, bs):
    N, C, H, W = x.size()
    x = x.view(N, bs, bs, C // (bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
    x = x.view(N, C // (bs ** 2), H * bs, W * bs)  # (N, C//bs^2, H * bs, W * bs)
    return x


class Squeeze(UnconditionalBaseFlowLayer):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        return space2depth(x, self.ratio), 0

    def backward(self, z):
        return depth2space(z, self.ratio), 0
