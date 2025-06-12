import torch
from torch import nn

class _BaseLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Contrastive losses are optional and not implemented in this package."
        )

class SupConLoss(_BaseLoss):
    pass

class SupConLossV2(_BaseLoss):
    pass

class SupConLossWithPrototype(_BaseLoss):
    pass

class SupConLossWithStorage(_BaseLoss):
    pass

class ContrastiveHead(nn.Module):
    """Simple two-layer MLP used when contrastive losses are enabled."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.fc(x)
