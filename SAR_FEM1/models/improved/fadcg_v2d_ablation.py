import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANT_CONFIGS = {
    "full": {
        "channel": True,
        "boundary": True,
        "speckle": True,
        "msc": True,
        "local_var": True,
    },
    "no_channel": {
        "channel": False,
        "boundary": True,
        "speckle": True,
        "msc": True,
        "local_var": True,
    },
    "no_boundary": {
        "channel": True,
        "boundary": False,
        "speckle": True,
        "msc": True,
        "local_var": True,
    },
    "no_speckle": {
        "channel": True,
        "boundary": True,
        "speckle": False,
        "msc": True,
        "local_var": False,
    },
    "no_boundary_no_speckle": {
        "channel": True,
        "boundary": False,
        "speckle": False,
        "msc": False,
        "local_var": False,
    },
    "no_all_gates": {
        "channel": False,
        "boundary": False,
        "speckle": False,
        "msc": False,
        "local_var": False,
    },
    "no_msc": {
        "channel": True,
        "boundary": True,
        "speckle": True,
        "msc": False,
        "local_var": True,
    },
    "no_local_var": {
        "channel": True,
        "boundary": True,
        "speckle": True,
        "msc": True,
        "local_var": False,
    },
    "no_boundary_no_local_var": {
        "channel": True,
        "boundary": False,
        "speckle": True,
        "msc": False,
        "local_var": False,
    },
    "no_boundary_keep_msc_no_local_var": {
        "channel": True,
        "boundary": False,
        "speckle": True,
        "msc": True,
        "local_var": False,
    },
    "boundary_no_msc_no_local_var": {
        "channel": True,
        "boundary": True,
        "speckle": True,
        "msc": False,
        "local_var": False,
    },
}


class AblationFrequencyGate(nn.Module):
    """Configurable V2d frequency gate for minimal-sufficient ablations."""

    def __init__(self, channels, variant="full", kernel_size=3, speckle_bias_init=-4.0):
        super().__init__()
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unknown V2d ablation variant: {variant}")
        self.channels = channels
        self.variant = variant
        self.config = VARIANT_CONFIGS[variant]
        self.avg3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avg7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.last_diagnostics = {}

        if self.config["boundary"]:
            boundary_inputs = 1 + int(self.config["msc"])
            self.boundary_head = nn.Conv2d(
                channels * boundary_inputs,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=True,
            )
        else:
            self.boundary_head = None

        if self.config["speckle"]:
            speckle_inputs = 1 + int(self.config["local_var"]) + int(self.config["msc"])
            self.speckle_head = nn.Conv2d(
                channels * speckle_inputs,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=True,
            )
        else:
            self.speckle_head = None

        self.speckle_bias_init = float(speckle_bias_init)
        self.reset_parameters()

    def reset_parameters(self):
        if self.boundary_head is not None:
            nn.init.zeros_(self.boundary_head.weight)
            nn.init.zeros_(self.boundary_head.bias)
        if self.speckle_head is not None:
            nn.init.zeros_(self.speckle_head.weight)
            nn.init.constant_(self.speckle_head.bias, self.speckle_bias_init)

    def describe(self, x):
        local_low = self.avg3(x)
        local_context = self.avg7(x)
        local_high = (x - local_low).abs()
        local_var = (self.avg3(x * x) - local_low * local_low).clamp_min(0.0)
        var_norm = local_var / (local_var.mean(dim=(2, 3), keepdim=True) + 1e-6)
        multi_scale_consistency = (local_low - local_context).abs()
        return local_high, var_norm, local_var, multi_scale_consistency

    def _boundary_descriptor(self, local_high, multi_scale_consistency):
        parts = [local_high]
        if self.config["msc"]:
            parts.append(multi_scale_consistency)
        return torch.cat(parts, dim=1)

    def _speckle_descriptor(self, local_high, var_norm, multi_scale_consistency):
        parts = [local_high]
        if self.config["local_var"]:
            parts.append(var_norm)
        if self.config["msc"]:
            parts.append(multi_scale_consistency)
        return torch.cat(parts, dim=1)

    def forward(self, x, beta=None, collect_diagnostics=False):
        local_high, var_norm, local_var, multi_scale_consistency = self.describe(x)
        if self.boundary_head is None:
            boundary_gate = torch.ones_like(x)
        else:
            boundary_gate = torch.sigmoid(self.boundary_head(self._boundary_descriptor(local_high, multi_scale_consistency))) * 2.0

        if self.speckle_head is None:
            speckle_gate = torch.zeros_like(x)
            suppression = torch.ones_like(x)
        else:
            if beta is None:
                raise ValueError("beta is required when speckle gate is enabled")
            speckle_gate = torch.sigmoid(self.speckle_head(self._speckle_descriptor(local_high, var_norm, multi_scale_consistency)))
            suppression = 1.0 - beta * speckle_gate

        gate = boundary_gate * suppression
        if collect_diagnostics:
            self.last_diagnostics = {
                "boundary_gate_mean": boundary_gate.detach().mean().cpu() if self.boundary_head is not None else "NA",
                "boundary_gate_std": boundary_gate.detach().std(unbiased=False).cpu() if self.boundary_head is not None else "NA",
                "speckle_gate_mean": speckle_gate.detach().mean().cpu() if self.speckle_head is not None else "NA",
                "speckle_gate_std": speckle_gate.detach().std(unbiased=False).cpu() if self.speckle_head is not None else "NA",
                "suppression_ratio": suppression.detach().mean().cpu() if self.speckle_head is not None else "NA",
                "combined_gate_mean": gate.detach().mean().cpu() if (self.boundary_head is not None or self.speckle_head is not None) else "NA",
                "combined_gate_std": gate.detach().std(unbiased=False).cpu() if (self.boundary_head is not None or self.speckle_head is not None) else "NA",
                "local_high_descriptor_mean": local_high.detach().mean().cpu() if (self.boundary_head is not None or self.speckle_head is not None) else "NA",
                "local_variance_proxy_mean": local_var.detach().mean().cpu() if self.config["local_var"] else "NA",
                "multi_scale_consistency_mean": multi_scale_consistency.detach().mean().cpu() if self.config["msc"] else "NA",
            }
        return gate


class FADCGV2dAblation(nn.Module):
    """FA-DCG V2d ablation block."""

    def __init__(self, channels, kernel_size=3, variant="full", alpha_init=0.5, beta_init=0.1, freq_init="random"):
        super().__init__()
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unknown V2d ablation variant: {variant}")
        if freq_init not in {"random", "laplacian"}:
            raise ValueError(f"Unknown frequency initialization: {freq_init}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.variant = variant
        self.freq_init = freq_init
        self.config = VARIANT_CONFIGS[variant]
        hidden = max(channels // 4, 1)
        if self.config["channel"]:
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.channel_gate = None
        self.frequency_gate = AblationFrequencyGate(channels, variant=variant)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        if self.config["speckle"]:
            beta_init = min(max(float(beta_init), 1e-4), 1.0 - 1e-4)
            self.beta_logit = nn.Parameter(torch.logit(torch.tensor(beta_init)))
        else:
            self.register_parameter("beta_logit", None)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        self.reset_frequency_weight()
        self.last_diagnostics = {}

    def reset_frequency_weight(self):
        if self.freq_init == "random":
            nn.init.normal_(self.weight)
            return
        if self.kernel_size != 3:
            raise ValueError("laplacian frequency initialization requires kernel_size=3")
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        # Normalize by L1 norm so the high-pass prior is explicit but not overly large.
        kernel = kernel / kernel.abs().sum()
        with torch.no_grad():
            self.weight.copy_(kernel.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1))

    def beta(self):
        if self.beta_logit is None:
            return None
        return torch.sigmoid(self.beta_logit)

    def forward(self, x, collect_diagnostics=False):
        padding = (self.kernel_size - 1) // 2
        z = F.conv2d(x, self.weight, padding=padding, groups=self.channels)
        if self.channel_gate is None:
            channel_gate = torch.ones_like(z)
        else:
            channel_gate = self.channel_gate(z)
        beta = self.beta()
        frequency_gate = self.frequency_gate(x, beta, collect_diagnostics=collect_diagnostics)
        out = x + self.alpha * channel_gate * frequency_gate * z
        if collect_diagnostics:
            self.last_diagnostics = {
                **self.frequency_gate.last_diagnostics,
                "alpha_value": self.alpha.detach().cpu(),
                "beta_value": beta.detach().cpu() if beta is not None else "NA",
                "channel_gate_mean": channel_gate.detach().mean().cpu() if self.channel_gate is not None else "NA",
                "channel_gate_std": channel_gate.detach().std(unbiased=False).cpu() if self.channel_gate is not None else "NA",
                "freq_init": self.freq_init,
                "depthwise_weight_mean": self.weight.detach().mean().cpu(),
                "depthwise_weight_std": self.weight.detach().std(unbiased=False).cpu(),
            }
        return out

    def collect_input_diagnostics(self, x):
        with torch.no_grad():
            padding = (self.kernel_size - 1) // 2
            z = F.conv2d(x, self.weight, padding=padding, groups=self.channels)
            if self.channel_gate is None:
                channel_gate = torch.ones_like(z)
            else:
                channel_gate = self.channel_gate(z)
            beta = self.beta()
            self.frequency_gate(x, beta, collect_diagnostics=True)
            self.last_diagnostics = {
                **self.frequency_gate.last_diagnostics,
                "alpha_value": self.alpha.detach().cpu(),
                "beta_value": beta.detach().cpu() if beta is not None else "NA",
                "channel_gate_mean": channel_gate.detach().mean().cpu() if self.channel_gate is not None else "NA",
                "channel_gate_std": channel_gate.detach().std(unbiased=False).cpu() if self.channel_gate is not None else "NA",
                "freq_init": self.freq_init,
                "depthwise_weight_mean": self.weight.detach().mean().cpu(),
                "depthwise_weight_std": self.weight.detach().std(unbiased=False).cpu(),
            }
        return self.last_diagnostics

    def get_diagnostics(self, x=None):
        if x is not None:
            self.collect_input_diagnostics(x)
        return self.last_diagnostics
