import torch
import math
from torch import nn
from normflowpy.base_flow import UnconditionalBaseFlowLayer, ConditionalBaseFlowLayer


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x, cond=None):
        m = x.shape[0]
        log_det = torch.zeros(m, device=x.device)
        zs = [x]
        for i, flow in enumerate(self.flows):
            try:
                if isinstance(flow, UnconditionalBaseFlowLayer):
                    x, ld = flow.forward(x)
                elif isinstance(flow, ConditionalBaseFlowLayer):
                    x, ld = flow.forward(x, cond=cond)
                else:
                    raise Exception("Unknown flow type")
                if torch.any(torch.isnan(x)):
                    raise Exception("Output results is Not a Number")
                log_det += ld
                zs.append(x)

            except Exception as e:
                raise Exception(f"Error {e} in flow type:{type(flow)} at index {i}")
        return zs, log_det

    def backward(self, z, cond=None):
        m = z.shape[0]
        log_det = torch.zeros(m, device=z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            if isinstance(flow, UnconditionalBaseFlowLayer):
                z, ld = flow.backward(z)
            elif isinstance(flow, ConditionalBaseFlowLayer):
                z, ld = flow.backward(z, cond=cond)
            else:
                raise Exception(f"Unknown flow type:{type(flow)}")
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows, condition_network=None):
        super().__init__()
        self.prior = prior
        self.condition_network = condition_network
        self.flow = NormalizingFlow(flows)

    def forward(self, x, cond=None):
        if self.condition_network is not None:
            cond = self.condition_network(cond)
        zs, log_det = self.flow.forward(x, cond=cond)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z, cond=None):
        if self.condition_network is not None:
            cond = self.condition_network(cond)
        xs, log_det = self.flow.backward(z, cond)
        return xs, log_det

    def nll(self, x, cond=None):
        zs, prior_logprob, log_det = self(x, cond=cond)
        logprob = prior_logprob + log_det  # Log-likelihood (LL)
        return -logprob  # Negative LL

    def nll_mean(self, x, cond=None):
        return torch.mean(self.nll(x, cond)) / x.shape[1:].numel()  # NLL per dim

    def sample(self, num_samples, cond=None, temperature: float = 1):
        param_list = list(self.flow.parameters())
        device = list(self.flow.parameters())[0].device if len(param_list) > 0 else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        z = self.prior.sample((num_samples,)).to(device)
        z = math.sqrt(temperature) * z
        xs, _ = self.backward(z, cond)
        return xs[-1]

    def sample_nll(self, num_samples, cond=None, temperature=1.0):
        y = self.sample(num_samples, cond=cond, temperature=temperature)
        y = y.detach()
        return self.nll(y, cond)
