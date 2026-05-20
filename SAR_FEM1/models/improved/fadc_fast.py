import torch
import torch.nn as nn
import torch.nn.functional as F


class FastFADCG(nn.Module):
    """Vectorized FA-DCG-v1 module with differentiable dilation preference."""

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation_rates=(1, 2, 3, 4),
        reduction=16,
        alpha_init=0.5,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_rates = tuple(dilation_rates)
        self.register_buffer("dilation_values", torch.tensor(self.dilation_rates, dtype=torch.float32))

        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=((kernel_size - 1) * dilation) // 2,
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                )
                for dilation in self.dilation_rates
            ]
        )

        hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.full((channels,), -4.0))
        self.theta = nn.Parameter(torch.zeros(channels, len(self.dilation_rates)))
        self.beta_raw = nn.Parameter(torch.full((channels,), 0.5413248546))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self):
        for branch in self.branches:
            nn.init.kaiming_normal_(branch.weight, mode="fan_out", nonlinearity="relu")
        with torch.no_grad():
            self.theta.zero_()
            self.theta[:, 0] = 0.25

    def branch_weights(self):
        d_min = self.dilation_values.min()
        d_max = self.dilation_values.max()
        d_cont = d_min + (d_max - d_min) * torch.sigmoid(self.gamma)
        beta = F.softplus(self.beta_raw)
        logits = self.theta - beta[:, None] * torch.abs(d_cont[:, None] - self.dilation_values[None, :])
        weights = torch.softmax(logits, dim=1)
        return weights, d_cont, beta

    def forward(self, x, collect_diagnostics=False):
        branch_outputs = [branch(x) for branch in self.branches]
        z_stack = torch.stack(branch_outputs, dim=2)
        weights, d_cont, beta = self.branch_weights()
        weight_view = weights.view(1, self.channels, len(self.dilation_rates), 1, 1)
        z = torch.sum(z_stack * weight_view, dim=2)
        gate = self.gate(z)
        y = x + self.alpha * gate * z

        if collect_diagnostics:
            self.last_diagnostics = self.diagnostics_from(weights, d_cont, beta, gate)
        return y

    def diagnostics_from(self, weights, d_cont, beta, gate):
        return {
            "branch_weights_mean": weights.detach().mean(dim=0).cpu(),
            "small_dilation_weight": weights[:, :2].detach().sum(dim=1).mean().cpu(),
            "large_dilation_weight": weights[:, 2:].detach().sum(dim=1).mean().cpu(),
            "gate_mean": gate.detach().mean().cpu(),
            "gate_std": gate.detach().std(unbiased=False).cpu(),
            "alpha_value": self.alpha.detach().cpu(),
            "d_cont_mean": d_cont.detach().mean().cpu(),
            "d_cont_min": d_cont.detach().min().cpu(),
            "d_cont_max": d_cont.detach().max().cpu(),
            "beta_mean": beta.detach().mean().cpu(),
        }

    def get_diagnostics(self, x=None):
        if x is not None:
            _ = self.forward(x, collect_diagnostics=True)
        return self.last_diagnostics
