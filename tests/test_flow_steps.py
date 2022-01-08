import unittest
import normflowpy as nfp
import torch
import numpy as np


class TestFlowSteps(unittest.TestCase):

    def check_zero(self, x):
        if isinstance(x, torch.Tensor):
            x = torch.abs(x).sum()  # Sum batch axis
            x = x.item()
        self.assertTrue(np.isclose(x, 0))
        print("a") # TODO: check load and save

    def base_flow_step(self, flow_step, input_shape):
        z = torch.randn(input_shape)
        x, logdet = flow_step.backward(z)
        z_hat, log_det_hat = flow_step.forward(x)
        log_det_sum = logdet + log_det_hat
        error = torch.pow(z_hat - z, 2.0).max().item()
        self.check_zero(error)
        self.check_zero(log_det_sum)

    def test_tensor2vector(self):
        flow_step = nfp.flows.Tensor2Vector([2, 5])
        input_shape = [5, 10]
        self.base_flow_step(flow_step, input_shape)

    def test_affine_coupling_2d(self):
        flow_step = nfp.flows.AffineCoupling([10, 4, 4], 0)
        input_shape = [4, 10, 4, 4]
        self.base_flow_step(flow_step, input_shape)

    def test_conv2d_1x1(self):
        flow_step = nfp.flows.InvertibleConv2d1x1(10)
        input_shape = [4, 10, 4, 4]
        self.base_flow_step(flow_step, input_shape)


if __name__ == '__main__':
    unittest.main()
