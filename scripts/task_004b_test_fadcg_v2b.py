#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.improved.fadcg_v2b import FADCGV2b, FrequencyDilationBias  # noqa: E402
from models.improved.fcn_fadcg_v2b import FCNWithFADCGV2b  # noqa: E402
from models.improved.unet_fadcg_v2b import UNetWithFADCGV2b  # noqa: E402


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def check_dilation_bias(channels):
    module = FrequencyDilationBias(channels)
    x = torch.randn(2, channels * 2, 16, 16, requires_grad=True)
    before_modules = len(list(module.modules()))
    weights = module(x, collect_diagnostics=True)
    after_modules = len(list(module.modules()))
    assert_true(weights.shape == (2, 3, channels, 16, 16), "FrequencyDilationBias shape mismatch")
    assert_true(torch.isfinite(weights).all(), "FrequencyDilationBias produced NaN/Inf")
    assert_true(before_modules == after_modules, "FrequencyDilationBias created modules in forward")
    assert_true(torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-6), "branch weights do not sum to 1")
    assert_true(float(weights[:, 0].mean()) > 0.80, "dilation=1 is not dominant at initialization")
    assert_true(float(weights[:, 2].mean()) < 0.05, "large dilation is not conservative at initialization")
    weights.mean().backward()
    assert_true(module.head.weight.grad is not None, "FrequencyDilationBias head missing gradient")


def check_v2b(channels, kernel_size):
    module = FADCGV2b(channels, kernel_size=kernel_size)
    x = torch.randn(2, channels, 16, 16, requires_grad=True)
    before_modules = len(list(module.modules()))
    out = module(x, collect_diagnostics=True)
    after_modules = len(list(module.modules()))
    assert_true(out.shape == x.shape, f"FADCGV2b({channels}) shape mismatch")
    assert_true(torch.isfinite(out).all(), f"FADCGV2b({channels}) produced NaN/Inf")
    assert_true(before_modules == after_modules, f"FADCGV2b({channels}) created modules in forward")
    for key in [
        "frequency_gate_mean",
        "frequency_gate_std",
        "local_high_descriptor_mean",
        "local_low_descriptor_mean",
        "alpha_value",
        "channel_gate_mean",
        "channel_gate_std",
        "branch_weight_mean_d1",
        "branch_weight_mean_d2",
        "branch_weight_mean_d3",
        "small_dilation_weight",
        "large_dilation_weight",
    ]:
        assert_true(key in module.last_diagnostics, f"missing diagnostic: {key}")
    out.mean().backward()
    assert_true(module.weight.grad is not None, "depthwise weight missing gradient")
    assert_true(module.alpha.grad is not None, "alpha missing gradient")
    gate_grads = [p.grad for p in module.channel_gate.parameters()]
    assert_true(all(g is not None for g in gate_grads), "channel gate parameter missing gradient")
    assert_true(module.frequency_gate.head.weight.grad is not None, "frequency gate missing gradient")
    assert_true(module.dilation_bias.head.weight.grad is not None, "dilation bias missing gradient")


def check_wrapper(factory, name):
    model = factory(in_channels=1, num_classes=1)
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    before_modules = len(list(model.modules()))
    out = model(x)
    after_modules = len(list(model.modules()))
    assert_true(out.shape == x.shape, f"{name} output shape mismatch")
    assert_true(torch.isfinite(out).all(), f"{name} produced NaN/Inf")
    assert_true(before_modules == after_modules, f"{name} created modules in forward")
    out.mean().backward()
    modules = [m for m in model.modules() if isinstance(m, FADCGV2b)]
    expected = 2 if name.startswith("FCN") else 1
    assert_true(len(modules) == expected, f"{name} should contain {expected} FADCGV2b modules")
    assert_true(all(m.alpha.grad is not None for m in modules), f"{name} alpha missing gradient")


def main():
    source = inspect.getsource(FADCGV2b) + inspect.getsource(FrequencyDilationBias)
    assert_true(".item()" not in source, "FADCGV2b source uses .item()")
    check_dilation_bias(128)
    check_v2b(512, 7)
    check_v2b(1024, 3)
    check_wrapper(FCNWithFADCGV2b, "FCNWithFADCGV2b")
    check_wrapper(UNetWithFADCGV2b, "UNetWithFADCGV2b")
    print("task_004b FA-DCG V2b tests passed")


if __name__ == "__main__":
    main()
