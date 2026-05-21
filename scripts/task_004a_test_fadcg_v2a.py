#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.improved.fadcg_v2a import FADCGV2a, SARFrequencyGate  # noqa: E402
from models.improved.fcn_fadcg_v2a import FCNWithFADCGV2a  # noqa: E402
from models.improved.unet_fadcg_v2a import UNetWithFADCGV2a  # noqa: E402


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def check_frequency_gate(channels):
    module = SARFrequencyGate(channels)
    x = torch.randn(2, channels, 16, 16, requires_grad=True)
    before_modules = len(list(module.modules()))
    gate = module(x, collect_diagnostics=True)
    after_modules = len(list(module.modules()))
    assert_true(gate.shape == x.shape, "SARFrequencyGate shape mismatch")
    assert_true(torch.isfinite(gate).all(), "SARFrequencyGate produced NaN/Inf")
    assert_true(before_modules == after_modules, "SARFrequencyGate created modules in forward")
    assert_true(abs(float(gate.mean()) - 1.0) < 1e-5, "SARFrequencyGate is not identity-initialized")
    gate.mean().backward()
    assert_true(module.head.weight.grad is not None, "SARFrequencyGate head missing gradient")


def check_v2a(channels, kernel_size):
    module = FADCGV2a(channels, kernel_size=kernel_size)
    x = torch.randn(2, channels, 16, 16, requires_grad=True)
    before_modules = len(list(module.modules()))
    out = module(x, collect_diagnostics=True)
    after_modules = len(list(module.modules()))
    assert_true(out.shape == x.shape, f"FADCGV2a({channels}) shape mismatch")
    assert_true(torch.isfinite(out).all(), f"FADCGV2a({channels}) produced NaN/Inf")
    assert_true(before_modules == after_modules, f"FADCGV2a({channels}) created modules in forward")
    for key in [
        "frequency_gate_mean",
        "frequency_gate_std",
        "local_high_descriptor_mean",
        "local_low_descriptor_mean",
        "alpha_value",
        "channel_gate_mean",
        "channel_gate_std",
    ]:
        assert_true(key in module.last_diagnostics, f"missing diagnostic: {key}")
    out.mean().backward()
    assert_true(module.weight.grad is not None, "depthwise weight missing gradient")
    assert_true(module.alpha.grad is not None, "alpha missing gradient")
    gate_grads = [p.grad for p in module.channel_gate.parameters()]
    assert_true(all(g is not None for g in gate_grads), "channel gate parameter missing gradient")
    assert_true(module.frequency_gate.head.weight.grad is not None, "frequency gate missing gradient")


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
    modules = [m for m in model.modules() if isinstance(m, FADCGV2a)]
    expected = 2 if name.startswith("FCN") else 1
    assert_true(len(modules) == expected, f"{name} should contain {expected} FADCGV2a modules")
    assert_true(all(m.alpha.grad is not None for m in modules), f"{name} alpha missing gradient")


def main():
    source = inspect.getsource(FADCGV2a) + inspect.getsource(SARFrequencyGate)
    assert_true(".item()" not in source, "FADCGV2a source uses .item()")
    check_frequency_gate(128)
    check_v2a(512, 7)
    check_v2a(1024, 3)
    check_wrapper(FCNWithFADCGV2a, "FCNWithFADCGV2a")
    check_wrapper(UNetWithFADCGV2a, "UNetWithFADCGV2a")
    print("task_004a FA-DCG V2a tests passed")


if __name__ == "__main__":
    main()
