import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline.unet import UNet


class SpeckleReliability(nn.Module):
    """Conservative SRFG-style reliability map for SAR speckle-like responses."""

    def __init__(self, channels, use_local_var=False, beta_init=0.1):
        super().__init__()
        self.channels = channels
        self.use_local_var = use_local_var
        self.avg3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        inputs = 2 if use_local_var else 1
        self.score = nn.Conv2d(channels * inputs, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        beta_init = min(max(float(beta_init), 1e-4), 1.0 - 1e-4)
        self.beta_logit = nn.Parameter(torch.logit(torch.tensor(beta_init)))
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def descriptors(self, x):
        local_low = self.avg3(x)
        local_high = (x - local_low).abs()
        local_var = (self.avg3(x * x) - local_low * local_low).clamp_min(0.0)
        local_var_norm = local_var / (local_var.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return local_high, local_var, local_var_norm

    def forward(self, x, collect_diagnostics=False):
        local_high, local_var, local_var_norm = self.descriptors(x)
        parts = [local_high]
        if self.use_local_var:
            parts.append(local_var_norm)
        score = self.score(torch.cat(parts, dim=1))
        reliability = 1.0 - self.beta() * torch.sigmoid(score)
        if collect_diagnostics:
            self.last_diagnostics = {
                "sr_beta": self.beta().detach().cpu(),
                "sr_reliability_mean": reliability.detach().mean().cpu(),
                "sr_reliability_std": reliability.detach().std(unbiased=False).cpu(),
                "sr_local_high_mean": local_high.detach().mean().cpu(),
                "sr_local_var_mean": local_var.detach().mean().cpu() if self.use_local_var else torch.tensor(float("nan")),
            }
        return reliability


class SARSpatialMask(nn.Module):
    """CBAM-style spatial mask with optional SAR high/variance descriptors."""

    def __init__(self, channels, use_sar=True, use_local_var=False, gamma_init=0.0):
        super().__init__()
        self.channels = channels
        self.use_sar = use_sar
        self.use_local_var = use_local_var
        self.avg3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        num_inputs = 2
        if use_sar:
            num_inputs += 1 + int(use_local_var)
        self.conv = nn.Conv2d(num_inputs, 1, kernel_size=7, padding=3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def descriptors(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        parts = [avg_out, max_out]
        local_high_mean = torch.tensor(float("nan"), device=x.device)
        local_var_mean = torch.tensor(float("nan"), device=x.device)
        if self.use_sar:
            local_low = self.avg3(x)
            local_high = (x - local_low).abs()
            local_var = (self.avg3(x * x) - local_low * local_low).clamp_min(0.0)
            local_var_norm = local_var / (local_var.mean(dim=(2, 3), keepdim=True) + 1e-6)
            parts.append(local_high.mean(dim=1, keepdim=True))
            local_high_mean = local_high.detach().mean()
            if self.use_local_var:
                parts.append(local_var_norm.mean(dim=1, keepdim=True))
                local_var_mean = local_var.detach().mean()
        return torch.cat(parts, dim=1), local_high_mean, local_var_mean

    def forward(self, x, collect_diagnostics=False):
        desc, local_high_mean, local_var_mean = self.descriptors(x)
        mask = torch.sigmoid(self.conv(desc))
        scale = 1.0 + self.gamma * (mask - 0.5) * 2.0
        out = x * scale
        if collect_diagnostics:
            self.last_diagnostics = {
                "spatial_gamma": self.gamma.detach().cpu(),
                "spatial_mask_mean": mask.detach().mean().cpu(),
                "spatial_mask_std": mask.detach().std(unbiased=False).cpu(),
                "spatial_mask_min": mask.detach().min().cpu(),
                "spatial_mask_max": mask.detach().max().cpu(),
                "spatial_local_high_mean": local_high_mean.detach().cpu(),
                "spatial_local_var_mean": local_var_mean.detach().cpu(),
            }
        return out


class FADCSRFGHybrid(nn.Module):
    """FADC-aligned main operator with optional SRFG reliability and SAR spatial masks."""

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation_rates=(1, 2, 3, 4),
        k_list=(3, 5, 7),
        reduction=16,
        alpha_init=0.1,
        sr_mode=None,
        sr_use_local_var=False,
        spatial_mode=None,
        spatial_use_local_var=False,
        spatial_gamma_init=0.0,
    ):
        super().__init__()
        if sr_mode not in {None, "high", "branch"}:
            raise ValueError(f"Unsupported sr_mode: {sr_mode}")
        if spatial_mode not in {None, "sar", "cbam"}:
            raise ValueError(f"Unsupported spatial_mode: {spatial_mode}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_rates = tuple(dilation_rates)
        self.k_list = tuple(k_list)
        self.sr_mode = sr_mode
        self.sr_use_local_var = sr_use_local_var
        self.spatial_mode = spatial_mode
        self.spatial_use_local_var = spatial_use_local_var
        self.pools = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReplicationPad2d(k // 2),
                    nn.AvgPool2d(kernel_size=k, stride=1, padding=0),
                )
                for k in self.k_list
            ]
        )
        self.freq_heads = nn.ModuleList(
            [nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True) for _ in range(len(self.k_list) + 1)]
        )
        self.branch_weight_head = nn.Conv2d(channels, len(self.dilation_rates), kernel_size=3, padding=1)
        self.low_high_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 16), 1, bias=False),
            nn.ReLU(inplace=True),
        )
        hidden_kernel = max(channels // reduction, 16)
        self.low_gate = nn.Conv2d(hidden_kernel, channels, 1)
        self.high_gate = nn.Conv2d(hidden_kernel, channels, 1)
        self.branch_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size)) for _ in self.dilation_rates]
        )
        hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.reliability = SpeckleReliability(channels, use_local_var=sr_use_local_var) if sr_mode else None
        self.spatial = (
            SARSpatialMask(
                channels,
                use_sar=(spatial_mode == "sar"),
                use_local_var=spatial_use_local_var,
                gamma_init=spatial_gamma_init,
            )
            if spatial_mode
            else None
        )
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self):
        for head in self.freq_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.branch_weight_head.weight)
        nn.init.zeros_(self.branch_weight_head.bias)
        with torch.no_grad():
            self.branch_weight_head.bias[0] = 1.0
        nn.init.zeros_(self.low_gate.weight)
        nn.init.ones_(self.low_gate.bias)
        nn.init.zeros_(self.high_gate.weight)
        nn.init.ones_(self.high_gate.bias)
        for weight in self.branch_weights:
            nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")

    def _apply_weight(self, x, weight):
        return x * (weight.sigmoid() * 2.0)

    def _frequency_select(self, x, reliability=None):
        current = x
        parts = []
        high_weights = []
        for idx, pool in enumerate(self.pools):
            low = pool(current)
            band = current - low
            if reliability is not None and self.sr_mode == "high":
                band = band * reliability
            weight = self.freq_heads[idx](x)
            high_weights.append(weight.sigmoid() * 2.0)
            parts.append(self._apply_weight(band, weight))
            current = low
        low_weight = self.freq_heads[len(self.pools)](x)
        parts.append(self._apply_weight(current, low_weight))
        return torch.stack(parts, dim=0).sum(dim=0), high_weights, low_weight.sigmoid() * 2.0

    def _adaptive_depthwise_conv(self, x, base_weight, low_gate, high_gate, dilation):
        b, c, h, w = x.shape
        mean = base_weight.mean(dim=(-1, -2), keepdim=True)
        high = base_weight - mean
        adapted = mean.unsqueeze(0) * low_gate.view(b, c, 1, 1, 1)
        adapted = adapted + high.unsqueeze(0) * high_gate.view(b, c, 1, 1, 1)
        adapted = adapted.reshape(b * c, 1, self.kernel_size, self.kernel_size)
        x_grouped = x.reshape(1, b * c, h, w)
        padding = ((self.kernel_size - 1) * dilation) // 2
        out = F.conv2d(x_grouped, adapted, padding=padding, dilation=dilation, groups=b * c)
        return out.reshape(b, c, h, w)

    def forward(self, x, collect_diagnostics=False):
        reliability = None
        if self.reliability is not None:
            reliability = self.reliability(x, collect_diagnostics=collect_diagnostics)
        x_freq, high_weights, low_weight = self._frequency_select(x, reliability=reliability)
        logits = self.branch_weight_head(x_freq)
        branch_mix = torch.softmax(logits, dim=1)
        kernel_feat = self.low_high_gate(x_freq)
        low_gate = self.low_gate(kernel_feat).sigmoid() * 2.0
        high_gate = self.high_gate(kernel_feat).sigmoid() * 2.0

        branch_outs = []
        for weight, dilation in zip(self.branch_weights, self.dilation_rates):
            branch = self._adaptive_depthwise_conv(x_freq, weight, low_gate, high_gate, dilation)
            if reliability is not None and self.sr_mode == "branch":
                branch = branch * reliability
            branch_outs.append(branch)

        z_stack = torch.stack(branch_outs, dim=1)
        z = (z_stack * branch_mix.unsqueeze(2)).sum(dim=1)
        gate = self.gate(z)
        out = x + self.alpha * gate * z
        if self.spatial is not None:
            out = self.spatial(out, collect_diagnostics=collect_diagnostics)

        if collect_diagnostics:
            self.last_diagnostics = self._diagnostics(branch_mix, high_weights, low_weight, low_gate, high_gate, gate)
            if self.reliability is not None:
                self.last_diagnostics.update(self.reliability.last_diagnostics)
            if self.spatial is not None:
                self.last_diagnostics.update(self.spatial.last_diagnostics)
        return out

    def _diagnostics(self, branch_mix, high_weights, low_weight, low_gate, high_gate, gate):
        branch_mean = branch_mix.detach().mean(dim=(0, 2, 3)).cpu()
        high = torch.stack(high_weights, dim=0)
        diagnostics = {
            "high_frequency_weight_mean": high.detach().mean().cpu(),
            "high_frequency_weight_std": high.detach().std(unbiased=False).cpu(),
            "low_frequency_weight_mean": low_weight.detach().mean().cpu(),
            "low_frequency_weight_std": low_weight.detach().std(unbiased=False).cpu(),
            "adaptive_kernel_low_gate_mean": low_gate.detach().mean().cpu(),
            "adaptive_kernel_high_gate_mean": high_gate.detach().mean().cpu(),
            "alpha_value": self.alpha.detach().cpu(),
            "gate_mean": gate.detach().mean().cpu(),
            "gate_std": gate.detach().std(unbiased=False).cpu(),
        }
        for idx, value in enumerate(branch_mean, start=1):
            diagnostics[f"branch_weight_mean_d{idx}"] = value
        diagnostics["small_dilation_weight"] = branch_mean[:2].sum()
        diagnostics["large_dilation_weight"] = branch_mean[2:].sum()
        return diagnostics

    def collect_input_diagnostics(self, x):
        with torch.no_grad():
            self.forward(x, collect_diagnostics=True)
        return self.last_diagnostics


class UNetWithFADCSRFGHybrid(UNet):
    """U-Net wrapper with one FADC/SRFG hybrid block at the bottleneck."""

    def __init__(self, in_channels=1, num_classes=1, features=[64, 128, 256, 512], **hybrid_kwargs):
        super().__init__(in_channels, num_classes, features)
        self.hybrid = FADCSRFGHybrid(features[-1] * 2, **hybrid_kwargs)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.hybrid(x)

        skip_connections = skip_connections[::-1]
        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            x = torch.cat([x, skip_connections[idx]], dim=1)
            x = decoder(x)
        return self.final_conv(x)
