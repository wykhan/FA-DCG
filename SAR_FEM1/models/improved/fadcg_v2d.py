import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeckleRobustFrequencyGate(nn.Module):
    """SAR frequency gate that separates boundary cues from speckle-like noise."""

    def __init__(self, channels, kernel_size=3, speckle_bias_init=-4.0):
        super().__init__()
        self.channels = channels
        self.avg3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avg7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.boundary_head = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.speckle_head = nn.Conv2d(
            channels * 3,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.speckle_bias_init = float(speckle_bias_init)
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.zeros_(self.boundary_head.bias)
        nn.init.zeros_(self.speckle_head.weight)
        nn.init.constant_(self.speckle_head.bias, self.speckle_bias_init)

    def describe(self, x):
        local_low = self.avg3(x)
        local_context = self.avg7(x)
        local_high = (x - local_low).abs()
        local_var = (self.avg3(x * x) - local_low * local_low).clamp_min(0.0)
        var_norm = local_var / (local_var.mean(dim=(2, 3), keepdim=True) + 1e-6)
        multi_scale_consistency = (local_low - local_context).abs()
        boundary_descriptor = torch.cat([local_high, multi_scale_consistency], dim=1)
        speckle_descriptor = torch.cat([local_high, var_norm, multi_scale_consistency], dim=1)
        return boundary_descriptor, speckle_descriptor, local_high, local_var, multi_scale_consistency

    def forward(self, x, beta, collect_diagnostics=False):
        boundary_descriptor, speckle_descriptor, local_high, local_var, multi_scale_consistency = self.describe(x)
        boundary_gate = torch.sigmoid(self.boundary_head(boundary_descriptor)) * 2.0
        speckle_gate = torch.sigmoid(self.speckle_head(speckle_descriptor))
        suppression = 1.0 - beta * speckle_gate
        gate = boundary_gate * suppression
        if collect_diagnostics:
            self.last_diagnostics = {
                "boundary_gate_mean": boundary_gate.detach().mean().cpu(),
                "boundary_gate_std": boundary_gate.detach().std(unbiased=False).cpu(),
                "speckle_gate_mean": speckle_gate.detach().mean().cpu(),
                "speckle_gate_std": speckle_gate.detach().std(unbiased=False).cpu(),
                "suppression_ratio": suppression.detach().mean().cpu(),
                "combined_gate_mean": gate.detach().mean().cpu(),
                "combined_gate_std": gate.detach().std(unbiased=False).cpu(),
                "local_high_descriptor_mean": local_high.detach().mean().cpu(),
                "local_variance_proxy_mean": local_var.detach().mean().cpu(),
                "multi_scale_consistency_mean": multi_scale_consistency.detach().mean().cpu(),
            }
        return gate


class FADCGV2d(nn.Module):
    """FA-DCG V2d: speckle-robust frequency gate on the V2a residual path."""

    def __init__(self, channels, kernel_size=3, alpha_init=0.5, beta_init=0.1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        hidden = max(channels // 4, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.frequency_gate = SpeckleRobustFrequencyGate(channels)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        beta_init = min(max(float(beta_init), 1e-4), 1.0 - 1e-4)
        self.beta_logit = nn.Parameter(torch.logit(torch.tensor(beta_init)))
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size, kernel_size))
        self.last_diagnostics = {}

    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def forward(self, x, collect_diagnostics=False):
        padding = (self.kernel_size - 1) // 2
        z = F.conv2d(x, self.weight, padding=padding, groups=self.channels)
        channel_gate = self.channel_gate(z)
        beta = self.beta()
        frequency_gate = self.frequency_gate(x, beta, collect_diagnostics=collect_diagnostics)
        out = x + self.alpha * channel_gate * frequency_gate * z
        if collect_diagnostics:
            self.last_diagnostics = {
                **self.frequency_gate.last_diagnostics,
                "alpha_value": self.alpha.detach().cpu(),
                "beta_value": beta.detach().cpu(),
                "channel_gate_mean": channel_gate.detach().mean().cpu(),
                "channel_gate_std": channel_gate.detach().std(unbiased=False).cpu(),
            }
        return out

    def collect_input_diagnostics(self, x):
        with torch.no_grad():
            padding = (self.kernel_size - 1) // 2
            z = F.conv2d(x, self.weight, padding=padding, groups=self.channels)
            channel_gate = self.channel_gate(z)
            beta = self.beta()
            self.frequency_gate(x, beta, collect_diagnostics=True)
            self.last_diagnostics = {
                **self.frequency_gate.last_diagnostics,
                "alpha_value": self.alpha.detach().cpu(),
                "beta_value": beta.detach().cpu(),
                "channel_gate_mean": channel_gate.detach().mean().cpu(),
                "channel_gate_std": channel_gate.detach().std(unbiased=False).cpu(),
            }
        return self.last_diagnostics

    def get_diagnostics(self, x=None):
        if x is not None:
            self.collect_input_diagnostics(x)
        return self.last_diagnostics
