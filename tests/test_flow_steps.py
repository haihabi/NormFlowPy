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

    def check_save_load(self, t_model, r_model, input_shape):
        r_model.load_state_dict(t_model.state_dict())
        z = torch.randn(input_shape)
        z_hat1, log_det_hat = t_model.forward(z)
        z_hat2, log_det_hat = r_model.forward(z)
        error = torch.pow(z_hat1 - z_hat2, 2.0).max()
        self.check_zero(error)

    def base_flow_step(self, flow_step, flow_step_sec, input_shape):
        z = torch.randn(input_shape)
        x, logdet = flow_step.backward(z)
        z_hat, log_det_hat = flow_step.forward(x)
        log_det_sum = logdet + log_det_hat
        error = torch.pow(z_hat - z, 2.0).max().item()
        self.check_zero(error)
        self.check_zero(log_det_sum)
        self.check_save_load(flow_step, flow_step_sec, input_shape)

    def test_tensor2vector(self):
        flow_step = nfp.flows.Tensor2Vector([2, 5])
        flow_step_sec = nfp.flows.Tensor2Vector([2, 5])
        input_shape = [5, 10]
        self.base_flow_step(flow_step, flow_step_sec, input_shape)

    def test_affine_coupling_2d(self):
        flow_step = nfp.flows.AffineCoupling([10, 4, 4], 0)
        flow_step_sec = nfp.flows.AffineCoupling([10, 4, 4], 0)
        input_shape = [4, 10, 4, 4]
        self.base_flow_step(flow_step, flow_step_sec, input_shape)

    def test_cubic_flow(self):
        flow_step = nfp.flows.CSF_CL(10)
        flow_step_sec = nfp.flows.CSF_CL(10)
        input_shape = [4, 10]
        self.base_flow_step(flow_step, flow_step_sec, input_shape)

    def test_conv2d_1x1(self):
        flow_step = nfp.flows.InvertibleConv2d1x1(10)
        flow_step_sec = nfp.flows.InvertibleConv2d1x1(10)
        input_shape = [4, 10, 4, 4]
        self.base_flow_step(flow_step, flow_step_sec, input_shape)

    def test_act_norm(self):
        flow_step = nfp.flows.ActNorm(10)
        flow_step_sec = nfp.flows.ActNorm(10)
        input_shape = [32, 10]
        z = torch.randn(input_shape)
        flow_step.forward(z)
        self.base_flow_step(flow_step, flow_step_sec, input_shape)


if __name__ == '__main__':
    unittest.main()
