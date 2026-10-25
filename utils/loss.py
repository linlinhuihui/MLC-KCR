import torch
from torch import nn


class FinetuneBinaryLoss(nn.Module):
    def __init__(self, smooth: float = 1e-8, gamma: float = 0.7) -> None:
        super().__init__()
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        pos = targets.sum()
        neg = targets.numel() - pos
        total = pos + neg + 2 * self.smooth

        alpha_p = (neg + self.smooth) / total
        alpha_n = (pos + self.smooth) / total

        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)

        alpha = alpha_p * targets + alpha_n * (1 - targets)
        focal_weight = (1.0 - pt).pow(self.gamma)

        loss = -alpha * focal_weight * torch.log(pt.clamp_min(self.smooth))
        return loss.mean()
