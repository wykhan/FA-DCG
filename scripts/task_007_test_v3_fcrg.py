#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.improved.fadcg_v3_fcrg import FADCGV3FCRG, FCRG_VARIANTS  # noqa: E402
from models.improved.fcn_fadcg_v3_fcrg import FCNWithFADCGV3FCRG  # noqa: E402
from models.improved.unet_fadcg_v3_fcrg import UNetWithFADCGV3FCRG  # noqa: E402


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def check_block(variant, channels=64, kernel_size=3):
    module = FADCGV3FCRG(channels, kernel_size=kernel_size, variant=variant)
    x = torch.randn(2, channels, 16, 16, requires_grad=True)
    before_modules = len(list(module.modules()))
    out = module(x, collect_diagnostics=True)
    after_modules = len(list(module.modules()))
    assert_true(out.shape == x.shape, f"{variant}: output shape mismatch")
    assert_true(torch.isfinite(out).all(), f"{variant}: produced NaN/Inf")
    assert_true(before_modules == after_modules, f"{variant}: created modules in forward")
    assert_true("alpha_value" in module.last_diagnostics, f"{variant}: missing alpha diagnostic")
    assert_true("local_high_gap_mean" in module.last_diagnostics, f"{variant}: missing local high diagnostic")
    assert_true("local_var_gap_mean" in module.last_diagnostics, f"{variant}: missing local var diagnostic")
    if variant == "v3_fcrg_a":
        assert_true("response_gate_mean" in module.last_diagnostics, f"{variant}: missing response diagnostic")
    else:
        assert_true(abs(float(module.last_diagnostics["lambda_value"])) < 1e-8, f"{variant}: lambda should initialize to 0")
        assert_true("final_response_gate_mean" in module.last_diagnostics, f"{variant}: missing final response diagnostic")
    out.mean().backward()
    assert_true(module.weight.grad is not None, f"{variant}: residual weight missing gradient")
    assert_true(module.alpha.grad is not None, f"{variant}: alpha missing gradient")
    if variant == "v3_fcrg_b":
        assert_true(module.lambda_param.grad is not None, f"{variant}: lambda missing gradient")


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
    modules = [m for m in model.modules() if isinstance(m, FADCGV3FCRG)]
    assert_true(len(modules) == expected_blocks, f"{name} {variant}: expected {expected_blocks} FCRG blocks")
    assert_true(all(m.alpha.grad is not None for m in modules), f"{name} {variant}: alpha missing gradient")


def main():
    for variant in FCRG_VARIANTS:
        check_block(variant)
        check_wrapper(FCNWithFADCGV3FCRG, "FCNWithFADCGV3FCRG", variant, 2)
        check_wrapper(UNetWithFADCGV3FCRG, "UNetWithFADCGV3FCRG", variant, 1)
    print("task_007 FA-DCG V3 FCRG tests passed")


if __name__ == "__main__":
    main()

