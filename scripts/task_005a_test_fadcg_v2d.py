#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.improved.fadcg_v2d import FADCGV2d, SpeckleRobustFrequencyGate  # noqa: E402
from models.improved.fcn_fadcg_v2d import FCNWithFADCGV2d  # noqa: E402
from models.improved.unet_fadcg_v2d import UNetWithFADCGV2d  # noqa: E402


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def check_speckle_gate(channels):
    module = SpeckleRobustFrequencyGate(channels)
    beta = torch.tensor(0.1)
    x = torch.randn(2, channels, 16, 16, requires_grad=True)
    before_modules = len(list(module.modules()))
    gate = module(x, beta, collect_diagnostics=True)
    after_modules = len(list(module.modules()))
    assert_true(gate.shape == x.shape, "SpeckleRobustFrequencyGate shape mismatch")
    assert_true(torch.isfinite(gate).all(), "SpeckleRobustFrequencyGate produced NaN/Inf")
    assert_true(before_modules == after_modules, "SpeckleRobustFrequencyGate created modules in forward")
    assert_true(abs(float(module.last_diagnostics["boundary_gate_mean"]) - 1.0) < 1e-5, "boundary gate is not identity-initialized")
    assert_true(float(module.last_diagnostics["speckle_gate_mean"]) < 0.03, "speckle gate is not conservatively initialized")
    assert_true(float(module.last_diagnostics["suppression_ratio"]) > 0.997, "suppression is too strong at initialization")
    gate.mean().backward()
    assert_true(module.boundary_head.weight.grad is not None, "boundary head missing gradient")
    assert_true(module.speckle_head.weight.grad is not None, "speckle head missing gradient")


def check_v2d(channels, kernel_size):
    module = FADCGV2d(channels, kernel_size=kernel_size)
    x = torch.randn(2, channels, 16, 16, requires_grad=True)
    before_modules = len(list(module.modules()))
    out = module(x, collect_diagnostics=True)
    after_modules = len(list(module.modules()))
    assert_true(out.shape == x.shape, f"FADCGV2d({channels}) shape mismatch")
    assert_true(torch.isfinite(out).all(), f"FADCGV2d({channels}) produced NaN/Inf")
    assert_true(before_modules == after_modules, f"FADCGV2d({channels}) created modules in forward")
    for key in [
        "boundary_gate_mean",
        "boundary_gate_std",
        "speckle_gate_mean",
        "speckle_gate_std",
        "suppression_ratio",
        "local_high_descriptor_mean",
        "local_variance_proxy_mean",
        "multi_scale_consistency_mean",
        "alpha_value",
        "beta_value",
        "channel_gate_mean",
        "channel_gate_std",
    ]:
        assert_true(key in module.last_diagnostics, f"missing diagnostic: {key}")
    assert_true(abs(float(module.last_diagnostics["beta_value"]) - 0.1) < 1e-5, "beta init changed")
    out.mean().backward()
    assert_true(module.weight.grad is not None, "depthwise weight missing gradient")
    assert_true(module.alpha.grad is not None, "alpha missing gradient")
    assert_true(module.beta_logit.grad is not None, "beta missing gradient")
    assert_true(module.frequency_gate.boundary_head.weight.grad is not None, "boundary gate missing gradient")
    assert_true(module.frequency_gate.speckle_head.weight.grad is not None, "speckle gate missing gradient")
    gate_grads = [p.grad for p in module.channel_gate.parameters()]
    assert_true(all(g is not None for g in gate_grads), "channel gate parameter missing gradient")


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
    modules = [m for m in model.modules() if isinstance(m, FADCGV2d)]
    expected = 2 if name.startswith("FCN") else 1
    assert_true(len(modules) == expected, f"{name} should contain {expected} FADCGV2d modules")
    assert_true(all(m.alpha.grad is not None for m in modules), f"{name} alpha missing gradient")
    assert_true(all(m.beta_logit.grad is not None for m in modules), f"{name} beta missing gradient")


def main():
    source = inspect.getsource(FADCGV2d) + inspect.getsource(SpeckleRobustFrequencyGate)
    assert_true(".item()" not in source, "FADCGV2d source uses .item()")
    check_speckle_gate(128)
    check_v2d(512, 7)
    check_v2d(1024, 3)
    check_wrapper(FCNWithFADCGV2d, "FCNWithFADCGV2d")
    check_wrapper(UNetWithFADCGV2d, "UNetWithFADCGV2d")
    print("task_005a FA-DCG V2d tests passed")


if __name__ == "__main__":
    main()
