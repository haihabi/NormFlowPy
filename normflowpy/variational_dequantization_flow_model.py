import torch

from normflowpy.nf_model import BaseFlowModel, NormalizingFlowModel
from normflowpy import flows


class VariationalDequantizationNormalizingFlowModel(BaseFlowModel):
    def __init__(self, bit_width, threshold, base_flow: NormalizingFlowModel,
                 dequantization_flow: NormalizingFlowModel):
        super().__init__()
        self.base_flow: NormalizingFlowModel = base_flow
        self.dequantization_flow: NormalizingFlowModel = dequantization_flow
        self.bit_width = bit_width
        self.threshold = threshold
        self.delta = 2 * threshold / (2 ** bit_width - 1)
        self.dequantization_flow.insert_flow_step(flows.Sigmoid())
        self.dequantization_flow.insert_flow_step(flows.UserDefinedAffineFlow(self.delta, -self.delta / 2))

    def _quantized(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.delta * torch.floor(x / self.delta) + self.delta / 2, -self.threshold, self.threshold)

    def sample(self, num_samples, temperature: float = 1, **kwargs):
        return self._quantized(self.base_flow.sample(num_samples, temperature, **kwargs))

    def nll(self, x, **kwargs):
        u = self.dequantization_flow.prior.sample((x.shape[0],)).to(x.device)
        epsilon, log_det_deq = self.dequantization_flow.backward(u, **kwargs)
        nll = self.base_flow.nll(x + epsilon[-1], **kwargs)
        nll_loss = nll + self.dequantization_flow.prior.log_prob(u) - log_det_deq  # - nll_eps  # Negative LL
        return nll_loss
