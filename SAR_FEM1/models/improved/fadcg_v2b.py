import torch
import torch.nn as nn
import torch.nn.functional as F

from models.improved.fadcg_v2a import SARFrequencyGate


class FrequencyDilationBias(nn.Module):
    """Frequency-conditioned dilation preference for FA-DCG V2b.

    The logits are initialized toward dilation=1, with small but non-zero
    capacity for dilation=2/3 so the model can learn broader context.
    """

    def __init__(self, channels, num_branches=3, kernel_size=3, init_logits=(2.0, 0.0, -2.0)):
        super().__init__()
        self.channels = channels
        self.num_branches = num_branches
        self.head = nn.Conv2d(
            channels * 2,
            channels * num_branches,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.register_buffer("init_logits", torch.tensor(init_logits, dtype=torch.float32).view(1, num_branches, 1, 1, 1))
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, descriptor, collect_diagnostics=False):
        logits = self.head(descriptor)
        b, _, h, w = logits.shape
        logits = logits.view(b, self.channels, self.num_branches, h, w).permute(0, 2, 1, 3, 4)
        logits = logits + self.init_logits.to(dtype=logits.dtype, device=logits.device)
        weights = torch.softmax(logits, dim=1)
        if collect_diagnostics:
            branch_means = weights.detach().mean(dim=(0, 2, 3, 4)).cpu()
            self.last_diagnostics = {
                "branch_weight_mean_d1": branch_means[0],
                "branch_weight_mean_d2": branch_means[1],
                "branch_weight_mean_d3": branch_means[2],
                "small_dilation_weight": branch_means[0],
                "large_dilation_weight": branch_means[1:].sum(),
            }
        return weights


class FADCGV2b(nn.Module):
    """FA-DCG V2b: V2a frequency gate plus conservative dilation preference."""

    def __init__(self, channels, kernel_size=3, alpha_init=0.5, dilations=(1, 2, 3)):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        hidden = max(channels // 4, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.frequency_gate = SARFrequencyGate(channels)
        self.dilation_bias = FrequencyDilationBias(channels, num_branches=len(self.dilations))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size, kernel_size))
        self.last_diagnostics = {}

    def _dilated_response(self, x, branch_weights):
        responses = []
        for dilation in self.dilations:
            padding = dilation * (self.kernel_size - 1) // 2
            responses.append(F.conv2d(x, self.weight, padding=padding, dilation=dilation, groups=self.channels))
        stacked = torch.stack(responses, dim=1)
        return (branch_weights * stacked).sum(dim=1)

    def forward(self, x, collect_diagnostics=False):
        descriptor, local_low, local_high, multi_scale_consistency = self.frequency_gate.describe(x)
        branch_weights = self.dilation_bias(descriptor, collect_diagnostics=collect_diagnostics)
        z = self._dilated_response(x, branch_weights)
        channel_gate = self.channel_gate(z)
        frequency_gate = self.frequency_gate(x, collect_diagnostics=collect_diagnostics)
        out = x + self.alpha * channel_gate * frequency_gate * z
        if collect_diagnostics:
            self.last_diagnostics = {
                **self.frequency_gate.last_diagnostics,
                **self.dilation_bias.last_diagnostics,
                "alpha_value": self.alpha.detach().cpu(),
                "channel_gate_mean": channel_gate.detach().mean().cpu(),
                "channel_gate_std": channel_gate.detach().std(unbiased=False).cpu(),
            }
        return out

    def collect_input_diagnostics(self, x):
        with torch.no_grad():
            descriptor, _, _, _ = self.frequency_gate.describe(x)
            branch_weights = self.dilation_bias(descriptor, collect_diagnostics=True)
            z = self._dilated_response(x, branch_weights)
            channel_gate = self.channel_gate(z)
            self.frequency_gate(x, collect_diagnostics=True)
            self.last_diagnostics = {
                **self.frequency_gate.last_diagnostics,
                **self.dilation_bias.last_diagnostics,
                "alpha_value": self.alpha.detach().cpu(),
                "channel_gate_mean": channel_gate.detach().mean().cpu(),
                "channel_gate_std": channel_gate.detach().std(unbiased=False).cpu(),
            }
        return self.last_diagnostics

    def get_diagnostics(self, x=None):
        if x is not None:
            self.collect_input_diagnostics(x)
        return self.last_diagnostics
