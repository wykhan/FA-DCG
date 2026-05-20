#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.improved.fadc_optimized import VectorizedLightFADC  # noqa: E402
from models.improved.fcn_fadc_optimized import FCNWithOptimizedFADCG  # noqa: E402
from models.improved.unet_fadc_optimized import UNetWithOptimizedFADCG  # noqa: E402


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def check_block(channels, kernel_size):
    x = torch.randn(2, channels, 16, 16, requires_grad=True)
    model = VectorizedLightFADC(channels, kernel_size=kernel_size)
    before_modules = len(list(model.modules()))
    out = model(x)
    after_modules = len(list(model.modules()))
    assert_true(out.shape == x.shape, f"VectorizedLightFADC({channels}) shape mismatch")
    assert_true(torch.isfinite(out).all(), f"VectorizedLightFADC({channels}) produced NaN/Inf")
    assert_true(before_modules == after_modules, "VectorizedLightFADC created modules in forward")
    out.mean().backward()
    assert_true(model.weight.grad is not None, "depthwise weight did not receive gradients")
    assert_true(model.alpha.grad is not None, "alpha did not receive gradients")


def check_wrapper(factory, name):
    model = factory(in_channels=1, num_classes=1)
    before_modules = len(list(model.modules()))
    x = torch.randn(1, 1, 256, 256)
    out = model(x)
    after_modules = len(list(model.modules()))
    assert_true(out.shape == x.shape, f"{name} output shape mismatch")
    assert_true(before_modules == after_modules, f"{name} created modules in forward")
    blocks = [m for m in model.modules() if isinstance(m, VectorizedLightFADC)]
    expected = 2 if name.startswith("FCN") else 1
    assert_true(len(blocks) == expected, f"{name} should contain {expected} optimized FA-DCG blocks")


def main():
    check_block(512, 7)
    check_block(1024, 3)
    check_wrapper(FCNWithOptimizedFADCG, "FCNWithOptimizedFADCG")
    check_wrapper(UNetWithOptimizedFADCG, "UNetWithOptimizedFADCG")
    source = inspect.getsource(VectorizedLightFADC)
    assert_true("for c in range" not in source, "VectorizedLightFADC still has per-channel loop")
    assert_true(".item()" not in source, "VectorizedLightFADC source uses .item()")
    print("task_002c Optimized FA-DCG tests passed")


if __name__ == "__main__":
    main()
