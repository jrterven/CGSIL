import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str | None = None) -> torch.Tensor:
        reduction = reduction or self.reduction
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - target_probs).pow(self.gamma)
        losses = -focal_factor * target_log_probs

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            losses = losses * alpha.gather(0, targets)

        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        return losses.mean()
