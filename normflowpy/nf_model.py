import torch
import math
import traceback

from torch import nn
from normflowpy.base_flow import UnconditionalBaseFlowLayer, ConditionalBaseFlowLayer, BaseFlowModel, BaseFlowLayer
from normflowpy.priors.trainable_prior import TrainablePrior


def check_is_number(in_xz):
    if torch.any(torch.logical_or(torch.isnan(in_xz), torch.isinf(in_xz))):  # TODO: move to function
        raise Exception("Output results is Not a Number")


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def insert_flow_step(self, flow_step: BaseFlowLayer):
        self.flows.insert(0, flow_step)

    def _step_flow(self, in_flow, in_flow_function, in_zx, **kwargs):
        try:
            if isinstance(in_flow, UnconditionalBaseFlowLayer):
                out_zx, ld = in_flow_function(in_zx)
            elif isinstance(in_flow, ConditionalBaseFlowLayer):
                out_zx, ld = in_flow_function(in_zx, **kwargs)
            else:
                raise Exception("Unknown flow type")
            check_is_number(out_zx)
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"Error {e} in flow type:{type(in_flow)}")
        return out_zx, ld

    def forward(self, x, **kwargs):
        m = x.shape[0]
        log_det = torch.zeros(m, device=x.device)
        zs = [x]

        for i, flow in enumerate(self.flows):
            # TODO:Make it a function
            x, ld = self._step_flow(flow, flow.forward, x, **kwargs)
            log_det += ld
            zs.append(x)
            # _x = x
            # try:
            #     if isinstance(flow, UnconditionalBaseFlowLayer):
            #         x, ld = flow.forward(_x)
            #     elif isinstance(flow, ConditionalBaseFlowLayer):
            #         x, ld = flow.forward(_x, **kwargs)
            #     else:
            #         raise Exception("Unknown flow type")
            #     check_is_number(x)
            #     log_det += ld
            #     zs.append(x)
            # except Exception as e:
            #     print(traceback.format_exc())
            #     raise Exception(f"Error {e} in flow type:{type(flow)} at index {i}")
        return zs, log_det

    def backward(self, z, **kwargs):
        m = z.shape[0]
        log_det = torch.zeros(m, device=z.device)
        xs = [z]
        for i, flow in enumerate(self.flows[::-1]):
            z, ld = self._step_flow(flow, flow.backward, z, **kwargs)
            # if isinstance(flow, UnconditionalBaseFlowLayer):
            #     z, ld = flow.backward(z)
            # elif isinstance(flow, ConditionalBaseFlowLayer):
            #     z, ld = flow.backward(z, **kwargs)
            # else:
            #     raise Exception(f"Unknown flow type:{type(flow)}")
            # check_is_number(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(BaseFlowModel):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows, condition_network=None):
        super().__init__()
        if isinstance(prior, TrainablePrior):  # If prior is trainable add it as module.
            self.add_module("prior", prior)
        else:
            self.prior = prior
        self.condition_network = condition_network
        self.flow = NormalizingFlow(flows)

    def insert_flow_step(self, flow_step: BaseFlowModel):
        self.flow.insert_flow_step(flow_step)

    def forward(self, x, **kwargs):
        if self.condition_network is not None:
            kwargs = self.condition_network(**kwargs)
        zs, log_det = self.flow.forward(x, **kwargs)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z, **kwargs):
        if self.condition_network is not None:
            kwargs = self.condition_network(**kwargs)
        xs, log_det = self.flow.backward(z, **kwargs)
        return xs, log_det

    def nll(self, x, **kwargs):
        zs, prior_logprob, log_det = self(x, **kwargs)
        logprob = prior_logprob + log_det  # Log-likelihood (LL)
        return -logprob  # Negative LL

    def sample(self, num_samples, temperature: float = 1, **kwargs):
        param_list = list(self.flow.parameters())
        device = list(self.flow.parameters())[0].device if len(param_list) > 0 else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        z = self.prior.sample((num_samples,)).to(device)
        z = math.sqrt(temperature) * z
        xs, _ = self.backward(z, **kwargs)
        return xs[-1]

    def sample_nll(self, num_samples, temperature=1.0, **kwargs):
        y = self.sample(num_samples, temperature=temperature, **kwargs)
        y = y.detach()
        return self.nll(y, **kwargs)

    def nll_mean(self, x, **kwargs):
        return torch.mean(self.nll(x, **kwargs)) / x.shape[1:].numel()  # NLL per dim
