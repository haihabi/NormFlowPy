import torch


def safe_log(x: torch.Tensor, eps=1e-22) -> torch.Tensor:
    return torch.log(x.clamp(min=eps))
