import torch
import torch.nn as nn
import torch.nn.functional as F


FCRG_VARIANTS = ("v3_fcrg_a", "v3_fcrg_b")


class FADCGV3FCRG(nn.Module):
    """FA-DCG V3 frequency-conditioned response gate."""

    def __init__(self, channels, kernel_size=3, variant="v3_fcrg_b", alpha_init=0.5):
        super().__init__()
        if variant not in FCRG_VARIANTS:
            raise ValueError(f"Unknown V3 FCRG variant: {variant}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.variant = variant
        response_hidden = max(channels // 4, 1)
        frequency_hidden = min(max(channels // 16, 1), 64)

        self.avg3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size, kernel_size))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        if variant == "v3_fcrg_a":
            self.response_gate = nn.Sequential(
                nn.Conv2d(channels * 3, frequency_hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(frequency_hidden, channels, 1),
                nn.Sigmoid(),
            )
            self.base_response_gate = None
            self.frequency_reliability = None
            self.lambda_param = None
        else:
            self.base_response_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, response_hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(response_hidden, channels, 1),
                nn.Sigmoid(),
            )
            self.frequency_reliability = nn.Sequential(
                nn.Conv2d(channels * 2, frequency_hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(frequency_hidden, channels, 1),
                nn.Tanh(),
            )
            self.lambda_param = nn.Parameter(torch.tensor(0.0))
            self.response_gate = None

        self.last_diagnostics = {}

    def describe(self, x):
        local_low = self.avg3(x)
        local_high = (x - local_low).abs()
        local_var = (self.avg3(x * x) - local_low * local_low).clamp_min(0.0)
        var_norm = local_var / (local_var.mean(dim=(2, 3), keepdim=True) + 1e-6)
        local_high_gap = F.adaptive_avg_pool2d(local_high, 1)
        local_var_gap = F.adaptive_avg_pool2d(var_norm, 1)
        return local_high, local_var, var_norm, local_high_gap, local_var_gap

    def _residual_response(self, x):
        padding = (self.kernel_size - 1) // 2
        return F.conv2d(x, self.weight, padding=padding, groups=self.channels)

    def _diagnostics_common(self, local_high_gap, local_var_gap):
        return {
            "alpha_value": self.alpha.detach().cpu(),
            "local_high_gap_mean": local_high_gap.detach().mean().cpu(),
            "local_var_gap_mean": local_var_gap.detach().mean().cpu(),
        }

    def forward(self, x, collect_diagnostics=False):
        z = self._residual_response(x)
        _, _, _, local_high_gap, local_var_gap = self.describe(x)

        if self.variant == "v3_fcrg_a":
            z_gap = F.adaptive_avg_pool2d(z, 1)
            descriptor = torch.cat([z_gap, local_high_gap, local_var_gap], dim=1)
            response_gate = self.response_gate(descriptor)
            out = x + self.alpha * response_gate * z
            if collect_diagnostics:
                self.last_diagnostics = {
                    **self._diagnostics_common(local_high_gap, local_var_gap),
                    "response_gate_mean": response_gate.detach().mean().cpu(),
                    "response_gate_std": response_gate.detach().std(unbiased=False).cpu(),
                }
            return out

        base_gate = self.base_response_gate(z)
        frequency_descriptor = torch.cat([local_high_gap, local_var_gap], dim=1)
        modulation = self.frequency_reliability(frequency_descriptor)
        final_gate = base_gate * (1.0 + self.lambda_param * modulation)
        out = x + self.alpha * final_gate * z
        if collect_diagnostics:
            self.last_diagnostics = {
                **self._diagnostics_common(local_high_gap, local_var_gap),
                "lambda_value": self.lambda_param.detach().cpu(),
                "base_response_gate_mean": base_gate.detach().mean().cpu(),
                "base_response_gate_std": base_gate.detach().std(unbiased=False).cpu(),
                "frequency_modulation_mean": modulation.detach().mean().cpu(),
                "frequency_modulation_std": modulation.detach().std(unbiased=False).cpu(),
                "final_response_gate_mean": final_gate.detach().mean().cpu(),
                "final_response_gate_std": final_gate.detach().std(unbiased=False).cpu(),
            }
        return out

    def collect_input_diagnostics(self, x):
        with torch.no_grad():
            self.forward(x, collect_diagnostics=True)
        return self.last_diagnostics

    def get_diagnostics(self, x=None):
        if x is not None:
            self.collect_input_diagnostics(x)
        return self.last_diagnostics
