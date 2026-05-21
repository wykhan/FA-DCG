#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.improved.fadc_aligned import FADCAligned, FrequencySelectionAligned  # noqa: E402
from models.improved.fcn_fadc_aligned import FCNWithFADCAligned  # noqa: E402
from models.improved.unet_fadc_aligned import UNetWithFADCAligned  # noqa: E402


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def check_no_forbidden_dependency():
    for path in [
        SAR_ROOT / "models" / "improved" / "fadc_aligned.py",
        SAR_ROOT / "models" / "improved" / "fcn_fadc_aligned.py",
        SAR_ROOT / "models" / "improved" / "unet_fadc_aligned.py",
    ]:
        text = path.read_text(encoding="utf-8")
        assert_true("mmcv" not in text.lower(), f"{path} imports or mentions mmcv")
        assert_true("mmseg" not in text.lower(), f"{path} imports or mentions mmseg")


def check_frequency_selection(channels):
    module = FrequencySelectionAligned(channels)
    x = torch.randn(2, channels, 32, 32, requires_grad=True)
    before_modules = len(list(module.modules()))
    out = module(x, collect_diagnostics=True)
    after_modules = len(list(module.modules()))
    assert_true(out.shape == x.shape, "FrequencySelectionAligned shape mismatch")
    assert_true(torch.isfinite(out).all(), "FrequencySelectionAligned produced NaN/Inf")
    assert_true(before_modules == after_modules, "FrequencySelectionAligned created modules in forward")
    out.mean().backward()
    grads = [p.grad for p in module.parameters()]
    assert_true(all(g is not None for g in grads), "FrequencySelectionAligned parameter missing gradient")
    diagnostics = module.last_diagnostics
    assert_true(torch.isfinite(torch.as_tensor(diagnostics["high_frequency_weight_mean"])), "high frequency diagnostic invalid")
    assert_true(torch.isfinite(torch.as_tensor(diagnostics["low_frequency_weight_mean"])), "low frequency diagnostic invalid")


def check_fadc(channels):
    module = FADCAligned(channels)
    x = torch.randn(2, channels, 32, 32, requires_grad=True)
    before_modules = len(list(module.modules()))
    out = module(x, collect_diagnostics=True)
    after_modules = len(list(module.modules()))
    assert_true(out.shape == x.shape, f"FADCAligned({channels}) shape mismatch")
    assert_true(torch.isfinite(out).all(), f"FADCAligned({channels}) produced NaN/Inf")
    assert_true(before_modules == after_modules, f"FADCAligned({channels}) created modules in forward")
    branch_keys = [f"branch_weight_mean_d{i}" for i in range(1, 5)]
    branch_sum = sum(float(module.last_diagnostics[key]) for key in branch_keys)
    assert_true(abs(branch_sum - 1.0) < 1e-5, f"FADCAligned({channels}) branch weights do not sum to 1")
    out.mean().backward()
    assert_true(module.alpha.grad is not None, "alpha missing gradient")
    grad_names = {
        "frequency selection": module.freq_select.parameters(),
        "branch head": module.branch_weight_head.parameters(),
        "adaptive kernel": module.kernel_attention.parameters(),
        "gate": module.gate.parameters(),
    }
    for name, params in grad_names.items():
        grads = [p.grad for p in params]
        assert_true(all(g is not None for g in grads), f"{name} parameter missing gradient")
    for idx, weight in enumerate(module.branch_weights):
        assert_true(weight.grad is not None, f"branch weight {idx} missing gradient")


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
    modules = [m for m in model.modules() if isinstance(m, FADCAligned)]
    assert_true(len(modules) == 1, f"{name} should contain exactly one FADCAligned")
    assert_true(modules[0].alpha.grad is not None, f"{name} FADC alpha missing gradient")


def main():
    check_no_forbidden_dependency()
    source = inspect.getsource(FADCAligned) + inspect.getsource(FrequencySelectionAligned)
    assert_true(".item()" not in source, "FADCAligned source uses .item()")
    check_frequency_selection(128)
    check_fadc(128)
    check_fadc(256)
    check_wrapper(FCNWithFADCAligned, "FCNWithFADCAligned")
    check_wrapper(UNetWithFADCAligned, "UNetWithFADCAligned")
    print("task_003 FADC-aligned tests passed")


if __name__ == "__main__":
    main()
