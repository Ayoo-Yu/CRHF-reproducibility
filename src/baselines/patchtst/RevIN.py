"""Reversible per-instance normalization with the experiment's affine convention."""
import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, values, mode):
        if mode == "norm":
            axes = tuple(range(1, values.ndim - 1))
            self.center = (values[:, -1:, :] if self.subtract_last else
                           values.mean(dim=axes, keepdim=True).detach())
            var = values.var(dim=axes, keepdim=True, unbiased=False)
            self.spread = (var + self.eps).sqrt().detach()
            norm = (values - self.center) / self.spread
            return norm * self.affine_weight + self.affine_bias if self.affine else norm
        if mode == "denorm":
            if self.affine:
                values = (values - self.affine_bias) / (self.affine_weight + self.eps ** 2)
            return values * self.spread + self.center
        raise ValueError(f"Unknown normalization mode {mode!r}")
