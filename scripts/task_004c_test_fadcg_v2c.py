#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.improved.fadcg_v2a import FADCGV2a  # noqa: E402
from models.improved.fadcg_v2b import FADCGV2b  # noqa: E402
from models.improved.fcn_fadcg_v2c import FCNWithFADCGV2c  # noqa: E402
from models.improved.unet_fadcg_v2c import UNetWithFADCGV2c  # noqa: E402


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


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
    return model


def main():
    fcn = check_wrapper(FCNWithFADCGV2c, "FCNWithFADCGV2c")
    fcn_v2a_modules = [m for m in fcn.modules() if isinstance(m, FADCGV2a)]
    fcn_v2b_modules = [m for m in fcn.modules() if isinstance(m, FADCGV2b)]
    assert_true(len(fcn_v2a_modules) == 2, "FCN V2c should contain two V2a modules")
    assert_true(len(fcn_v2b_modules) == 0, "FCN V2c should not contain V2b modules")
    assert_true(all(m.alpha.grad is not None for m in fcn_v2a_modules), "FCN V2c alpha missing gradient")

    unet = check_wrapper(UNetWithFADCGV2c, "UNetWithFADCGV2c")
    unet_v2a_modules = [m for m in unet.modules() if isinstance(m, FADCGV2a)]
    unet_v2b_modules = [m for m in unet.modules() if isinstance(m, FADCGV2b)]
    assert_true(len(unet_v2a_modules) == 1, "U-Net V2c should contain one skip V2a module")
    assert_true(len(unet_v2b_modules) == 1, "U-Net V2c should contain one bottleneck V2b module")
    assert_true(unet.skip_frequency_calibration.alpha.detach().item() == 0.25, "skip calibration alpha init changed")
    assert_true(all(m.alpha.grad is not None for m in unet_v2a_modules + unet_v2b_modules), "U-Net V2c alpha missing gradient")

    print("task_004c FA-DCG V2c tests passed")


if __name__ == "__main__":
    main()
