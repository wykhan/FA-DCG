#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.improved.fadc_fast import FastFADCG
from models.improved.fcn_fadc_fast import FCNWithFastFADCG
from models.improved.unet_fadc_fast import UNetWithFastFADCG


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def check_fast_fadcg(channels, size=16):
    model = FastFADCG(channels, kernel_size=3)
    x = torch.randn(2, channels, size, size, requires_grad=True)
    before_modules = set(dict(model.named_modules()).keys())
    out = model(x, collect_diagnostics=True)
    after_modules = set(dict(model.named_modules()).keys())
    assert_true(out.shape == x.shape, f"FastFADCG({channels}) shape mismatch")
    assert_true(torch.isfinite(out).all(), f"FastFADCG({channels}) produced NaN/Inf")
    assert_true(before_modules == after_modules, f"FastFADCG({channels}) created modules in forward")

    weights, d_cont, beta = model.branch_weights()
    assert_true(torch.allclose(weights.sum(dim=1), torch.ones(channels), atol=1e-6), "branch weights do not sum to 1")
    gate = model.gate(torch.randn(2, channels, size, size))
    assert_true(float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0, "gate outside [0, 1]")
    assert_true(model.alpha.requires_grad, "alpha is not trainable")
    assert_true(torch.all(d_cont >= 1.0) and torch.all(d_cont <= 4.0), "d_cont outside [1, 4]")
    assert_true(torch.all(beta >= 0.0), "beta is not non-negative")

    loss = out.mean()
    loss.backward()
    grad_names = ["gamma", "theta", "beta_raw", "alpha"]
    for name in grad_names:
        param = getattr(model, name)
        assert_true(param.grad is not None and torch.isfinite(param.grad).all(), f"{name} missing finite gradient")
    for idx, branch in enumerate(model.branches):
        assert_true(branch.weight.grad is not None, f"branch {idx} weight missing gradient")
    gate_grads = [p.grad for p in model.gate.parameters()]
    assert_true(all(g is not None for g in gate_grads), "gate parameter missing gradient")
    return {
        "module": f"FastFADCG({channels})",
        "output_shape": tuple(out.shape),
        "weights_sum_min": float(weights.sum(dim=1).min()),
        "weights_sum_max": float(weights.sum(dim=1).max()),
        "gate_min": float(gate.min()),
        "gate_max": float(gate.max()),
        "d_cont_mean": float(d_cont.mean()),
    }


def check_wrapper(model_cls, name):
    model = model_cls(in_channels=1, num_classes=1)
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    before_modules = set(dict(model.named_modules()).keys())
    out = model(x)
    after_modules = set(dict(model.named_modules()).keys())
    assert_true(out.shape == (1, 1, 256, 256), f"{name} output shape mismatch")
    assert_true(torch.isfinite(out).all(), f"{name} produced NaN/Inf")
    assert_true(before_modules == after_modules, f"{name} created modules in forward")
    loss = out.mean()
    loss.backward()
    fast_modules = [m for m in model.modules() if isinstance(m, FastFADCG)]
    assert_true(len(fast_modules) == 1, f"{name} should contain exactly one FastFADCG")
    fast = fast_modules[0]
    for param_name in ["gamma", "theta", "beta_raw", "alpha"]:
        param = getattr(fast, param_name)
        assert_true(param.grad is not None, f"{name}.{param_name} missing gradient")
    return {"module": name, "output_shape": tuple(out.shape), "fast_modules": len(fast_modules)}


def main():
    source = inspect.getsource(FastFADCG)
    assert_true(".item()" not in source, "FastFADCG source uses .item()")
    results = [
        check_fast_fadcg(512),
        check_fast_fadcg(1024),
        check_wrapper(UNetWithFastFADCG, "UNetWithFastFADCG"),
        check_wrapper(FCNWithFastFADCG, "FCNWithFastFADCG"),
    ]
    for row in results:
        print(row)
    print("task_002b FastFADCG tests passed")


if __name__ == "__main__":
    main()
