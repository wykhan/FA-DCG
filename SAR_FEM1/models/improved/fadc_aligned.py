import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySelectionAligned(nn.Module):
    """Local PyTorch frequency selection aligned with the FADC FreqSelect idea."""

    def __init__(
        self,
        channels,
        k_list=(3, 5, 7),
        lp_type="avgpool",
        spatial_group=1,
        lowfreq_att=True,
        act="sigmoid",
    ):
        super().__init__()
        if lp_type not in {"avgpool", "laplacian"}:
            raise ValueError(f"Unsupported lp_type: {lp_type}")
        self.channels = channels
        self.k_list = tuple(k_list)
        self.lp_type = lp_type
        self.lowfreq_att = lowfreq_att
        self.act = act
        self.spatial_group = min(max(int(spatial_group), 1), channels)
        if channels % self.spatial_group != 0:
            self.spatial_group = 1

        num_weights = len(self.k_list) + (1 if lowfreq_att else 0)
        self.weight_heads = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    self.spatial_group,
                    kernel_size=3,
                    padding=1,
                    groups=self.spatial_group,
                    bias=True,
                )
                for _ in range(num_weights)
            ]
        )
        self.pools = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReplicationPad2d(k // 2),
                    nn.AvgPool2d(kernel_size=k, stride=1, padding=0),
                )
                for k in self.k_list
            ]
        )
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self):
        for head in self.weight_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _activate(self, weight):
        if self.act == "sigmoid":
            return weight.sigmoid() * 2.0
        if self.act == "softmax":
            return weight.softmax(dim=1) * weight.shape[1]
        raise ValueError(f"Unsupported act: {self.act}")

    def _apply_group_weight(self, x, weight):
        b, _, h, w = x.shape
        grouped = x.reshape(b, self.spatial_group, -1, h, w)
        weighted = grouped * weight.reshape(b, self.spatial_group, 1, h, w)
        return weighted.reshape(b, -1, h, w)

    def _laplacian_bands(self, x):
        bands = []
        current = x
        out_h, out_w = x.shape[-2:]
        for _ in self.k_list:
            low_small = F.interpolate(
                current,
                size=((current.shape[-2] + 1) // 2, (current.shape[-1] + 1) // 2),
                mode="bilinear",
                align_corners=False,
            )
            low = F.interpolate(low_small, size=(out_h, out_w), mode="bilinear", align_corners=False)
            current_aligned = F.interpolate(current, size=(out_h, out_w), mode="bilinear", align_corners=False)
            bands.append(current_aligned - low)
            current = low_small
        low_final = F.interpolate(current, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return bands, low_final

    def forward(self, x, collect_diagnostics=False):
        bands = []
        weights = []

        if self.lp_type == "avgpool":
            current = x
            for pool in self.pools:
                low = pool(current)
                bands.append(current - low)
                current = low
            low_final = current
        else:
            bands, low_final = self._laplacian_bands(x)

        out_parts = []
        for idx, band in enumerate(bands):
            weight = self._activate(self.weight_heads[idx](x))
            weights.append(weight)
            out_parts.append(self._apply_group_weight(band, weight))

        if self.lowfreq_att:
            low_weight = self._activate(self.weight_heads[len(bands)](x))
            weights.append(low_weight)
            out_parts.append(self._apply_group_weight(low_final, low_weight))
        else:
            out_parts.append(low_final)

        out = torch.stack(out_parts, dim=0).sum(dim=0)
        if collect_diagnostics:
            self.last_diagnostics = self.diagnostics_from(weights)
        return out

    def diagnostics_from(self, weights):
        high = torch.stack(weights[: len(self.k_list)], dim=0)
        diagnostics = {
            "high_frequency_weight_mean": high.detach().mean().cpu(),
            "high_frequency_weight_std": high.detach().std(unbiased=False).cpu(),
        }
        if self.lowfreq_att:
            diagnostics["low_frequency_weight_mean"] = weights[-1].detach().mean().cpu()
            diagnostics["low_frequency_weight_std"] = weights[-1].detach().std(unbiased=False).cpu()
        return diagnostics


class OmniAttentionLite(nn.Module):
    """Small attention block used for AdaKern-aligned low/high kernel gates."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.low_gate = nn.Conv2d(hidden, channels, 1)
        self.high_gate = nn.Conv2d(hidden, channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.low_gate.weight)
        nn.init.ones_(self.low_gate.bias)
        nn.init.zeros_(self.high_gate.weight)
        nn.init.ones_(self.high_gate.bias)

    def forward(self, x):
        feat = self.net(x)
        return self.low_gate(feat).sigmoid() * 2.0, self.high_gate(feat).sigmoid() * 2.0


class FADCAligned(nn.Module):
    """Local PyTorch FADC-aligned block.

    FreqSelect is implemented with learnable high/low band reweighting.
    AdaDR is approximated with spatially adaptive dilation-branch mixing.
    AdaKern is represented by low/high kernel decomposition gates.
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation_rates=(1, 2, 3, 4),
        k_list=(3, 5, 7),
        lp_type="avgpool",
        spatial_group=1,
        reduction=16,
        alpha_init=0.1,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_rates = tuple(dilation_rates)
        self.freq_select = FrequencySelectionAligned(
            channels,
            k_list=k_list,
            lp_type=lp_type,
            spatial_group=spatial_group,
            lowfreq_att=True,
        )
        self.branch_weight_head = nn.Conv2d(channels, len(self.dilation_rates), kernel_size=3, padding=1)
        self.kernel_attention = OmniAttentionLite(channels, reduction=reduction)
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
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.branch_weight_head.weight)
        nn.init.zeros_(self.branch_weight_head.bias)
        with torch.no_grad():
            self.branch_weight_head.bias[0] = 1.0
        for weight in self.branch_weights:
            nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")

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
        x_freq = self.freq_select(x, collect_diagnostics=collect_diagnostics)
        logits = self.branch_weight_head(x_freq)
        branch_mix = torch.softmax(logits, dim=1)
        low_gate, high_gate = self.kernel_attention(x_freq)

        branch_outs = []
        for weight, dilation in zip(self.branch_weights, self.dilation_rates):
            branch_outs.append(self._adaptive_depthwise_conv(x_freq, weight, low_gate, high_gate, dilation))

        z_stack = torch.stack(branch_outs, dim=1)
        z = (z_stack * branch_mix.unsqueeze(2)).sum(dim=1)
        gate = self.gate(z)
        y = x + self.alpha * gate * z

        if collect_diagnostics:
            self.last_diagnostics = self.diagnostics_from(branch_mix, low_gate, high_gate, gate)
        return y

    def diagnostics_from(self, branch_mix, low_gate, high_gate, gate):
        branch_mean = branch_mix.detach().mean(dim=(0, 2, 3)).cpu()
        diagnostics = dict(self.freq_select.last_diagnostics)
        for idx, value in enumerate(branch_mean, start=1):
            diagnostics[f"branch_weight_mean_d{idx}"] = value
        diagnostics["small_dilation_weight"] = branch_mean[:2].sum()
        diagnostics["large_dilation_weight"] = branch_mean[2:].sum()
        diagnostics["adaptive_kernel_low_gate_mean"] = low_gate.detach().mean().cpu()
        diagnostics["adaptive_kernel_high_gate_mean"] = high_gate.detach().mean().cpu()
        diagnostics["mask_mean_if_deform_conv_used"] = torch.tensor(float("nan"))
        diagnostics["alpha_value"] = self.alpha.detach().cpu()
        diagnostics["gate_mean"] = gate.detach().mean().cpu()
        diagnostics["gate_std"] = gate.detach().std(unbiased=False).cpu()
        return diagnostics

    def get_diagnostics(self, x=None):
        if x is not None:
            _ = self.forward(x, collect_diagnostics=True)
        return self.last_diagnostics
