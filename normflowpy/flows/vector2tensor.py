from normflowpy.base_flow import UnconditionalBaseFlowLayer


class Tensor2Vector(UnconditionalBaseFlowLayer):
    def __init__(self, in_shape):
        super().__init__()
        self.shape = in_shape

    def forward(self, x):
        z = x.reshape([x.shape[0], -1])
        return z, 0

    def backward(self, z):
        x = z.reshape([-1, *self.shape])
        return x, 0


