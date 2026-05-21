import torch
import torch.nn as nn
import torch.nn.functional as F


class SARFrequencyGate(nn.Module):
    """Lightweight SAR local-frequency gate for FA-DCG V2a.

    The gate uses local residual and multi-scale consistency cues. It is
    initialized to 1.0 because the final sigmoid output is multiplied by 2.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.avg3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avg7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.head = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.last_diagnostics = {}
        self.reset_parameters()

    def describe(self, x):
        local_low = self.avg3(x)
        local_context = self.avg7(x)
        local_high = (x - local_low).abs()
        multi_scale_consistency = (local_low - local_context).abs()
        descriptor = torch.cat([local_high, multi_scale_consistency], dim=1)
        return descriptor, local_low, local_high, multi_scale_consistency

    def reset_parameters(self):
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, collect_diagnostics=False):
        descriptor, local_low, local_high, multi_scale_consistency = self.describe(x)
        gate = torch.sigmoid(self.head(descriptor)) * 2.0
        if collect_diagnostics:
            self.last_diagnostics = {
                "frequency_gate_mean": gate.detach().mean().cpu(),
                "frequency_gate_std": gate.detach().std(unbiased=False).cpu(),
                "local_high_descriptor_mean": local_high.detach().mean().cpu(),
                "local_low_descriptor_mean": local_low.detach().mean().cpu(),
                "multi_scale_consistency_mean": multi_scale_consistency.detach().mean().cpu(),
            }
        return gate


class FADCGV2a(nn.Module):
    """FA-DCG V2a: V1.1 residual enhancement modulated by SAR frequency gate."""

    def __init__(self, channels, kernel_size=3, alpha_init=0.5):
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
        self.frequency_gate = SARFrequencyGate(channels)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size, kernel_size))
        self.last_diagnostics = {}

    def forward(self, x, collect_diagnostics=False):
        padding = (self.kernel_size - 1) // 2
        z = F.conv2d(x, self.weight, padding=padding, groups=self.channels)
        channel_gate = self.channel_gate(z)
        frequency_gate = self.frequency_gate(x, collect_diagnostics=collect_diagnostics)
        out = x + self.alpha * channel_gate * frequency_gate * z
        if collect_diagnostics:
            self.last_diagnostics = {
                **self.frequency_gate.last_diagnostics,
                "alpha_value": self.alpha.detach().cpu(),
                "channel_gate_mean": channel_gate.detach().mean().cpu(),
                "channel_gate_std": channel_gate.detach().std(unbiased=False).cpu(),
            }
        return out

    def collect_input_diagnostics(self, x):
        with torch.no_grad():
            padding = (self.kernel_size - 1) // 2
            z = F.conv2d(x, self.weight, padding=padding, groups=self.channels)
            channel_gate = self.channel_gate(z)
            self.frequency_gate(x, collect_diagnostics=True)
            self.last_diagnostics = {
                **self.frequency_gate.last_diagnostics,
                "alpha_value": self.alpha.detach().cpu(),
                "channel_gate_mean": channel_gate.detach().mean().cpu(),
                "channel_gate_std": channel_gate.detach().std(unbiased=False).cpu(),
            }
        return self.last_diagnostics

    def get_diagnostics(self, x=None):
        if x is not None:
            self.collect_input_diagnostics(x)
        return self.last_diagnostics
