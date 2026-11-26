"""P5 mixing by image-conditioned ridge regression with leave-one-out prediction."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("AIFI_LeaveOneOutRidge",)


class AIFI_LeaveOneOutRidge(nn.Module):
    """Read scene context and prediction errors from the projected P5 feature map.

    ViT^3 (CVPR 2026) motivates fitting an inner model to visual tokens.
    Gated KalmaNet (CVPR 2026) uses ridge regression to account for covariance.
    Here a noncausal fit yields analytical leave-one-out predictions during both
    training and inference.
    """

    def __init__(self, c1, cm=1024, num_heads=8, ridge=0.03):
        super().__init__()
        if num_heads < 1 or c1 % num_heads:
            raise ValueError("c1 must be divisible by a positive num_heads")
        self.num_heads = num_heads
        self.ridge_min = 0.005
        self.ridge_max = 0.25
        if not self.ridge_min < ridge < self.ridge_max:
            raise ValueError("ridge must be between 0.005 and 0.25")
        ridge_logit = math.log((ridge - self.ridge_min) / (self.ridge_max - ridge))
        self.ridge_logit = nn.Parameter(torch.full((num_heads, 1, 1), ridge_logit))

        self.key = nn.Linear(c1, c1)
        self.value = nn.Linear(c1, c1)
        self.readout = nn.Linear(2 * c1, c1, bias=False)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.25)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

    @staticmethod
    def _leave_one_out(keys, values, ridge):
        """Return the excluded prediction and its residual for each token.

        Centering fits an unpenalized intercept. Each deletion keeps the full
        fit's penalty n*ridge in the unnormalized least-squares objective.
        """
        n = keys.size(-2)
        keys = keys - keys.mean(dim=-2, keepdim=True)
        values_centered = values - values.mean(dim=-2, keepdim=True)
        keys_t = keys.transpose(-2, -1)
        identity = torch.eye(keys.size(-1), device=keys.device, dtype=keys.dtype)
        gram = keys_t @ (keys / n) + ridge * identity
        inverse = torch.linalg.solve_ex(gram, identity.expand_as(gram)).result
        precision_keys = keys @ inverse

        fitted = precision_keys @ (keys_t @ (values_centered / n))
        leverage = (1 + (precision_keys * keys).sum(dim=-1, keepdim=True)) / n
        # PRESS identity, including the intercept's contribution to leverage.
        residual = (values_centered - fitted) / (1 - leverage).clamp_min(1e-4)
        return values - residual, residual

    def _mix(self, tokens, height, width):
        batch, n, channels = tokens.shape
        keys = self.key(tokens).reshape(batch, n, self.num_heads, -1).transpose(1, 2)
        values = self.value(tokens).reshape(batch, n, self.num_heads, -1).transpose(1, 2)

        # Protect the covariance, solve and leverage correction during AMP.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            dtype = torch.promote_types(keys.dtype, torch.float32)
            keys = F.normalize(F.silu(keys.to(dtype)), dim=-1, eps=1e-6)
            y = (torch.arange(height, device=tokens.device, dtype=dtype) + 0.5) * (2 / height) - 1
            x = (torch.arange(width, device=tokens.device, dtype=dtype) + 0.5) * (2 / width) - 1
            y, x = torch.meshgrid(y, x, indexing="ij")
            position = torch.stack((x, y, x * y, (x.square() + y.square()) / 2), dim=-1)
            position = position.reshape(1, 1, n, 4).expand(batch, self.num_heads, -1, -1)
            keys = torch.cat((keys, position * 0.25), dim=-1)

            ridge = self.ridge_min + (self.ridge_max - self.ridge_min) * self.ridge_logit.to(dtype).sigmoid()
            context, residual = self._leave_one_out(keys, values.to(dtype), ridge)
            power = residual.square().mean(dim=-1, keepdim=True)
            evidence = power / (power + power.mean(dim=-2, keepdim=True) + 1e-6)
            features = torch.cat((context, evidence * residual), dim=-1)

        features = features.transpose(1, 2).reshape(batch, n, 2 * channels)
        return self.readout(features.to(tokens.dtype))

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        if height * width > 1:
            tokens = tokens + self._mix(tokens, height, width)
        tokens = self.norm1(tokens)
        tokens = self.norm2(tokens + self.fc2(F.gelu(self.fc1(tokens))))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()
