#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.improved.fadcg_v2d_ablation import FADCGV2dAblation, VARIANT_CONFIGS  # noqa: E402
from models.improved.fcn_fadcg_v2d_ablation import FCNWithFADCGV2dAblation  # noqa: E402
from models.improved.unet_fadcg_v2d_ablation import UNetWithFADCGV2dAblation  # noqa: E402


DIAGNOSTIC_KEYS = [
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
]


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def is_na(value):
    return isinstance(value, str) and value == "NA"


def check_block(variant, channels=64, kernel_size=3):
    config = VARIANT_CONFIGS[variant]
    module = FADCGV2dAblation(channels, kernel_size=kernel_size, variant=variant)
    x = torch.randn(2, channels, 16, 16, requires_grad=True)
    before_modules = len(list(module.modules()))
    out = module(x, collect_diagnostics=True)
    after_modules = len(list(module.modules()))
    assert_true(out.shape == x.shape, f"{variant}: output shape mismatch")
    assert_true(torch.isfinite(out).all(), f"{variant}: produced NaN/Inf")
    assert_true(before_modules == after_modules, f"{variant}: created modules in forward")
    for key in DIAGNOSTIC_KEYS:
        assert_true(key in module.last_diagnostics, f"{variant}: missing diagnostic {key}")

    if config["boundary"]:
        assert_true(abs(float(module.last_diagnostics["boundary_gate_mean"]) - 1.0) < 1e-5, f"{variant}: boundary init changed")
    else:
        assert_true(is_na(module.last_diagnostics["boundary_gate_mean"]), f"{variant}: boundary diagnostic should be NA")

    if config["speckle"]:
        assert_true(float(module.last_diagnostics["speckle_gate_mean"]) < 0.03, f"{variant}: speckle init too strong")
        assert_true(float(module.last_diagnostics["suppression_ratio"]) > 0.997, f"{variant}: suppression init too strong")
        assert_true(abs(float(module.last_diagnostics["beta_value"]) - 0.1) < 1e-5, f"{variant}: beta init changed")
    else:
        assert_true(is_na(module.last_diagnostics["speckle_gate_mean"]), f"{variant}: speckle diagnostic should be NA")
        assert_true(is_na(module.last_diagnostics["beta_value"]), f"{variant}: beta diagnostic should be NA")

    if config["channel"]:
        assert_true(not is_na(module.last_diagnostics["channel_gate_mean"]), f"{variant}: channel diagnostic should exist")
    else:
        assert_true(is_na(module.last_diagnostics["channel_gate_mean"]), f"{variant}: channel diagnostic should be NA")

    out.mean().backward()
    assert_true(module.weight.grad is not None, f"{variant}: depthwise weight missing gradient")
    assert_true(module.alpha.grad is not None, f"{variant}: alpha missing gradient")
    if config["speckle"]:
        assert_true(module.beta_logit.grad is not None, f"{variant}: beta missing gradient")
    else:
        assert_true(module.beta_logit is None, f"{variant}: beta parameter should be removed")
    if config["channel"]:
        assert_true(all(p.grad is not None for p in module.channel_gate.parameters()), f"{variant}: channel gate missing gradient")
    if config["boundary"]:
        assert_true(module.frequency_gate.boundary_head.weight.grad is not None, f"{variant}: boundary gate missing gradient")
    if config["speckle"]:
        assert_true(module.frequency_gate.speckle_head.weight.grad is not None, f"{variant}: speckle gate missing gradient")


def check_wrapper(factory, name, variant, expected_blocks):
    model = factory(in_channels=1, num_classes=1, variant=variant)
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    before_modules = len(list(model.modules()))
    out = model(x)
    after_modules = len(list(model.modules()))
    assert_true(out.shape == x.shape, f"{name} {variant}: output shape mismatch")
    assert_true(torch.isfinite(out).all(), f"{name} {variant}: produced NaN/Inf")
    assert_true(before_modules == after_modules, f"{name} {variant}: created modules in forward")
    out.mean().backward()
    modules = [m for m in model.modules() if isinstance(m, FADCGV2dAblation)]
    assert_true(len(modules) == expected_blocks, f"{name} {variant}: expected {expected_blocks} ablation blocks")
    assert_true(all(m.alpha.grad is not None for m in modules), f"{name} {variant}: alpha missing gradient")


def main():
    source = inspect.getsource(FADCGV2dAblation)
    assert_true(".item()" not in source, "FADCGV2dAblation source uses .item()")
    for variant in VARIANT_CONFIGS:
        check_block(variant)
    for variant in ["full", "no_channel", "no_speckle", "no_all_gates", "no_msc", "no_local_var"]:
        check_wrapper(FCNWithFADCGV2dAblation, "FCNWithFADCGV2dAblation", variant, 2)
        check_wrapper(UNetWithFADCGV2dAblation, "UNetWithFADCGV2dAblation", variant, 1)
    print("task_006 FA-DCG V2d ablation tests passed")


if __name__ == "__main__":
    main()
