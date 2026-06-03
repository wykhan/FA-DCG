#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(SAR_ROOT))

from task_008_main_module_table import (  # noqa: E402
    FloodDatasetLocal,
    benchmark_latency,
    compute_scores,
    count_params,
    evaluate,
    profile_flops,
    seed_worker,
    set_seed,
    split_dirs,
)
from models.baseline.fcn import FCN  # noqa: E402
from models.improved.fadcg_v2d_ablation import FADCGV2dAblation, VARIANT_CONFIGS  # noqa: E402
from models.improved.fcn_fadcg_v2d_ablation import FCNWithFADCGV2dAblation  # noqa: E402
from models.improved.fcn_residual_only import FCNWithResidualOnly, ResidualOnlyBlock  # noqa: E402


PRIMARY_METHODS = [
    "FCN + Residual-only",
    "FCN + V2d-noBoundary-noLocalVar",
]

MODEL_SPECS = [
    {
        "backbone": "FCN",
        "method": "FCN + Residual-only",
        "variant": "residual_only",
        "factory": lambda **kwargs: FCNWithResidualOnly(**kwargs),
        "notes": "depthwise residual control: z=depthwise_conv(x), out=x+alpha*z; no channel, boundary, speckle, MSC, local variance, or frequency-descriptor gates",
    },
    {
        "backbone": "FCN",
        "method": "FCN + V2d-noBoundary-noLocalVar",
        "variant": "no_boundary_no_local_var",
        "factory": lambda **kwargs: FCNWithFADCGV2dAblation(variant="no_boundary_no_local_var", **kwargs),
        "notes": "V2d ablation config: channel=True, boundary=False, speckle=True, msc=False, local_var=False, freq_init=random",
    },
]


@dataclass
class RunResult:
    backbone: str
    method: str
    variant: str
    seed: int
    val_miou: float
    val_dice: float
    test_miou: float
    test_dice: float
    best_epoch: int
    checkpoint_path: str
    status: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--data-root", default="/home/superws/dataset/HISEA1_flooding_dataset")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-complexity", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-iters", type=int, default=300)
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def slug(text: str) -> str:
    return (
        text.lower()
        .replace(" + ", "_")
        .replace("-", "")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def raw_fieldnames() -> List[str]:
    return [
        "backbone",
        "method",
        "variant",
        "seed",
        "val_miou",
        "val_dice",
        "test_miou",
        "test_dice",
        "best_epoch",
        "checkpoint_path",
        "status",
        "notes",
    ]


def read_raw(path: Path) -> Dict[Tuple[str, int], Dict[str, object]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["method"], int(row["seed"])): row for row in rows if row.get("status") == "completed"}


def build_loaders(args: argparse.Namespace, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = Path(args.data_root)
    train_img, train_mask = split_dirs(data_root, "train")
    val_img, val_mask = split_dirs(data_root, "val")
    test_img, test_mask = split_dirs(data_root, "test")
    train_ds = FloodDatasetLocal(train_img, train_mask, args.img_size, True)
    val_ds = FloodDatasetLocal(val_img, val_mask, args.img_size, False)
    test_ds = FloodDatasetLocal(test_img, test_mask, args.img_size, False)
    if args.max_train_samples:
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(args.max_train_samples, len(train_ds)))))
    if args.max_val_samples:
        val_ds = torch.utils.data.Subset(val_ds, list(range(min(args.max_val_samples, len(val_ds)))))
    if args.max_test_samples:
        test_ds = torch.utils.data.Subset(test_ds, list(range(min(args.max_test_samples, len(test_ds)))))
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
            worker_init_fn=seed_worker,
        ),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available()),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available()),
    )


def make_model(spec: Dict[str, object], device: torch.device) -> nn.Module:
    model = spec["factory"](in_channels=1, num_classes=1).to(device)  # type: ignore[misc]
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, 256, 256, device=device))
    return model


def train_one(args: argparse.Namespace, spec: Dict[str, object], seed: int, out_dir: Path) -> RunResult:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_loaders(args, seed)
    model = make_model(spec, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.BCEWithLogitsLoss()

    best = {"val_miou": -1.0, "val_dice": 0.0, "epoch": -1, "checkpoint": ""}
    log_path = out_dir / "logs" / f"{slug(str(spec['method']))}_seed{seed}.log"
    ckpt_dir = out_dir / "checkpoints"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"method={spec['method']}, variant={spec['variant']}, seed={seed}\n")
        log.write(f"protocol=256x256, batch={args.batch_size}, epochs={args.epochs}, Adam, lr={args.lr}, cosine_min_lr={args.min_lr}\n")
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for images, masks in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            val_miou, val_dice = evaluate(model, val_loader, device)
            log.write(f"epoch={epoch}, train_loss={np.mean(losses):.6f}, val_miou={val_miou:.6f}, val_dice={val_dice:.6f}\n")
            log.flush()
            if val_miou > best["val_miou"]:
                ckpt_path = ckpt_dir / f"{slug(str(spec['method']))}_seed{seed}_best.pt"
                torch.save(model.state_dict(), ckpt_path)
                best = {"val_miou": val_miou, "val_dice": val_dice, "epoch": epoch, "checkpoint": str(ckpt_path)}

        model.load_state_dict(torch.load(str(best["checkpoint"]), map_location=device))
        test_miou, test_dice = evaluate(model, test_loader, device)
        log.write(
            f"selected_best_epoch={best['epoch']}, val_miou={best['val_miou']:.6f}, "
            f"val_dice={best['val_dice']:.6f}, test_miou={test_miou:.6f}, test_dice={test_dice:.6f}\n"
        )

    write_model_diagnostics(out_dir, spec, model, test_loader, device, seed)
    return RunResult(
        backbone="FCN",
        method=str(spec["method"]),
        variant=str(spec["variant"]),
        seed=seed,
        val_miou=float(best["val_miou"]),
        val_dice=float(best["val_dice"]),
        test_miou=float(test_miou),
        test_dice=float(test_dice),
        best_epoch=int(best["epoch"]),
        checkpoint_path=str(Path(best["checkpoint"]).resolve()),
        status="completed",
        notes=str(spec["notes"]),
    )


def build_complexity_model(method: str) -> nn.Module:
    if method == "FCN":
        return FCN(in_channels=1, num_classes=1)
    if method == "FCN + Residual-only":
        return FCNWithResidualOnly(in_channels=1, num_classes=1)
    if method == "FCN + V2d-noBoundary-noLocalVar":
        return FCNWithFADCGV2dAblation(variant="no_boundary_no_local_var", in_channels=1, num_classes=1)
    raise ValueError(f"Unknown complexity method: {method}")


def complexity_rows(args: argparse.Namespace, out_dir: Path) -> List[Dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    rows = []
    for method in ["FCN", *PRIMARY_METHODS]:
        set_seed(0)
        model = build_complexity_model(method).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(torch.zeros(1, 1, args.img_size, args.img_size, device=device))
        params_abs = count_params(model)
        macs_abs, flops_unit, flops_tool = profile_flops(model, device, args.img_size)
        latency_mean, latency_std, fps = benchmark_latency(
            model, device, args.img_size, args.benchmark_warmup, args.benchmark_iters
        )
        rows.append(
            {
                "backbone": "FCN",
                "method": method,
                "variant": "baseline" if method == "FCN" else next(spec["variant"] for spec in MODEL_SPECS if spec["method"] == method),
                "params_abs": params_abs,
                "params_delta_vs_fcn": 0,
                "macs_abs": macs_abs,
                "macs_delta_vs_fcn": 0,
                "flops_unit": flops_unit,
                "flops_tool": flops_tool,
                "latency_ms_mean": f"{latency_mean:.6f}",
                "latency_ms_std": f"{latency_std:.6f}",
                "fps": f"{fps:.6f}",
                "device": device.type,
                "gpu_name": gpu_name,
                "input_shape": f"[1,1,{args.img_size},{args.img_size}]",
                "notes": "THOP MACs; one multiply-add counted as one MAC",
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline_params = int(rows[0]["params_abs"])
    baseline_macs = int(rows[0]["macs_abs"])
    for row in rows:
        row["params_delta_vs_fcn"] = int(row["params_abs"]) - baseline_params
        row["macs_delta_vs_fcn"] = int(row["macs_abs"]) - baseline_macs
    write_csv(
        out_dir / "complexity_raw.csv",
        rows,
        [
            "backbone",
            "method",
            "variant",
            "params_abs",
            "params_delta_vs_fcn",
            "macs_abs",
            "macs_delta_vs_fcn",
            "flops_unit",
            "flops_tool",
            "latency_ms_mean",
            "latency_ms_std",
            "fps",
            "device",
            "gpu_name",
            "input_shape",
            "notes",
        ],
    )
    return rows


def summarize(raw_rows: List[Dict[str, object]], comp_rows: List[Dict[str, object]], out_dir: Path) -> List[Dict[str, object]]:
    comp_by_method = {str(row["method"]): row for row in comp_rows}
    summary = []
    for spec in MODEL_SPECS:
        items = [
            row
            for row in raw_rows
            if row["method"] == spec["method"] and row["status"] == "completed" and row["test_miou"] not in {"", "NA"}
        ]
        if not items:
            continue
        val_miou = np.array([float(row["val_miou"]) for row in items], dtype=np.float64)
        test_miou = np.array([float(row["test_miou"]) for row in items], dtype=np.float64)
        test_dice = np.array([float(row["test_dice"]) for row in items], dtype=np.float64)
        comp = comp_by_method.get(str(spec["method"]), {})
        summary.append(
            {
                "backbone": "FCN",
                "method": spec["method"],
                "variant": spec["variant"],
                "num_seeds": len(items),
                "val_miou_mean": f"{val_miou.mean():.6f}",
                "val_miou_std": f"{val_miou.std(ddof=1) if len(val_miou) > 1 else 0.0:.6f}",
                "test_miou_mean": f"{test_miou.mean():.6f}",
                "test_miou_std": f"{test_miou.std(ddof=1) if len(test_miou) > 1 else 0.0:.6f}",
                "test_dice_mean": f"{test_dice.mean():.6f}",
                "test_dice_std": f"{test_dice.std(ddof=1) if len(test_dice) > 1 else 0.0:.6f}",
                "params_abs": comp.get("params_abs", "NA"),
                "params_delta_vs_fcn": comp.get("params_delta_vs_fcn", "NA"),
                "macs_abs": comp.get("macs_abs", "NA"),
                "latency_ms_mean": comp.get("latency_ms_mean", "NA"),
                "fps": comp.get("fps", "NA"),
                "notes": "task_012 residual-control summary",
            }
        )
    write_csv(
        out_dir / "summary_mean_std.csv",
        summary,
        [
            "backbone",
            "method",
            "variant",
            "num_seeds",
            "val_miou_mean",
            "val_miou_std",
            "test_miou_mean",
            "test_miou_std",
            "test_dice_mean",
            "test_dice_std",
            "params_abs",
            "params_delta_vs_fcn",
            "macs_abs",
            "latency_ms_mean",
            "fps",
            "notes",
        ],
    )
    return summary


def paired_deltas(raw_rows: List[Dict[str, object]], out_dir: Path) -> List[Dict[str, object]]:
    by_key = {(str(row["method"]), int(row["seed"])): row for row in raw_rows if row["status"] == "completed"}
    seeds = sorted({seed for method, seed in by_key if method in PRIMARY_METHODS})
    rows = []
    for seed in seeds:
        residual = by_key.get(("FCN + Residual-only", seed))
        v2d = by_key.get(("FCN + V2d-noBoundary-noLocalVar", seed))
        if residual is None or v2d is None:
            continue
        rows.append(
            {
                "seed": seed,
                "residual_only_test_miou": f"{float(residual['test_miou']):.6f}",
                "v2d_test_miou": f"{float(v2d['test_miou']):.6f}",
                "delta_miou": f"{float(v2d['test_miou']) - float(residual['test_miou']):.6f}",
                "residual_only_test_dice": f"{float(residual['test_dice']):.6f}",
                "v2d_test_dice": f"{float(v2d['test_dice']):.6f}",
                "delta_dice": f"{float(v2d['test_dice']) - float(residual['test_dice']):.6f}",
            }
        )
    write_csv(
        out_dir / "paired_seed_delta.csv",
        rows,
        [
            "seed",
            "residual_only_test_miou",
            "v2d_test_miou",
            "delta_miou",
            "residual_only_test_dice",
            "v2d_test_dice",
            "delta_dice",
        ],
    )
    return rows


def tensor_to_value(value: object) -> object:
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        return float(value.item())  # type: ignore[union-attr]
    return value


def collect_diagnostics(model: nn.Module, images: torch.Tensor) -> List[Dict[str, object]]:
    modules = [module for module in model.modules() if isinstance(module, (ResidualOnlyBlock, FADCGV2dAblation))]
    handles = []

    def capture_input(current, inputs):
        current.collect_input_diagnostics(inputs[0])
        return None

    for module in modules:
        handles.append(module.register_forward_pre_hook(capture_input))
    with torch.no_grad():
        _ = model(images)
    for handle in handles:
        handle.remove()

    rows = []
    for idx, module in enumerate(modules):
        for key, value in getattr(module, "last_diagnostics", {}).items():
            rows.append(
                {
                    "module_index": idx,
                    "module_class": module.__class__.__name__,
                    "metric": key,
                    "value": tensor_to_value(value),
                }
            )
    return rows


def write_model_diagnostics(out_dir: Path, spec: Dict[str, object], model: nn.Module, loader: DataLoader, device: torch.device, seed: int) -> None:
    rows = []
    try:
        images, _ = next(iter(loader))
        images = images[:1].to(device)
        for row in collect_diagnostics(model, images):
            rows.append(
                {
                    "method": spec["method"],
                    "variant": spec["variant"],
                    "seed": seed,
                    "source": "first_test_batch",
                    **row,
                    "notes": "module input diagnostics collected with forward pre-hooks",
                }
            )
    except Exception as exc:
        rows.append(
            {
                "method": spec["method"],
                "variant": spec["variant"],
                "seed": seed,
                "source": "diagnostics",
                "module_index": "NA",
                "module_class": "NA",
                "metric": "error",
                "value": "NA",
                "notes": repr(exc),
            }
        )
    path = out_dir / "diagnostics_raw.csv"
    append = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "variant", "seed", "source", "module_index", "module_class", "metric", "value", "notes"],
        )
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def write_environment(out_dir: Path) -> None:
    with (out_dir / "environment.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"python --version\n{run_cmd(['python', '--version'])}\n\n")
        handle.write(f"nvidia-smi\n{run_cmd(['nvidia-smi'])}\n\n")
        handle.write(f"platform={platform.platform()}\n")
        handle.write(f"torch={torch.__version__}\n")
        handle.write(f"cuda={torch.version.cuda}\n")
        handle.write(f"cuda_available={torch.cuda.is_available()}\n")


def split_sizes(data_root: Path) -> Dict[str, int]:
    sizes = {}
    for split in ["train", "val", "test"]:
        image_dir, _ = split_dirs(data_root, split)
        sizes[split] = len([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    return sizes


def write_implementation_audit(args: argparse.Namespace, out_dir: Path, comp_rows: List[Dict[str, object]]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (out_dir / "implementation_audit.md").open("w", encoding="utf-8") as handle:
        handle.write("# Task 012 Implementation Audit\n\n")
        handle.write("## Required Variant Config\n\n")
        handle.write(f"- `no_boundary_no_local_var`: `{VARIANT_CONFIGS['no_boundary_no_local_var']}`\n")
        handle.write("- required config: `channel=True, boundary=False, speckle=True, msc=False, local_var=False, freq_init=random`\n\n")
        for spec in MODEL_SPECS:
            model = make_model(spec, device)
            before = len(list(model.modules()))
            x = torch.randn(2, 1, args.img_size, args.img_size, device=device)
            with torch.no_grad():
                y = model(x)
            after = len(list(model.modules()))
            handle.write(f"## {spec['method']}\n\n")
            handle.write(f"- variant: `{spec['variant']}`\n")
            handle.write(f"- class: `{model.__class__.__name__}`\n")
            handle.write(f"- params: `{count_params(model)}`\n")
            handle.write(f"- output shape: `{tuple(y.shape)}`\n")
            handle.write(f"- finite output: `{bool(torch.isfinite(y).all().item())}`\n")
            handle.write(f"- no lazy modules in forward: `{before == after}`\n")
            handle.write("- path: `conv1 -> conv2 -> conv3 -> conv4 -> conv5 -> block512 -> relu6/drop6/up_dim -> block4096 -> relu7/drop7/score -> interpolate`\n")
            modules = [module for module in model.modules() if isinstance(module, (ResidualOnlyBlock, FADCGV2dAblation))]
            handle.write(f"- enhancement modules: `{[m.__class__.__name__ for m in modules]}`\n")
            for idx, module in enumerate(modules):
                handle.write(f"  - module{idx} channels: `{getattr(module, 'channels', 'NA')}`; kernel_size: `{getattr(module, 'kernel_size', 'NA')}`\n")
                if isinstance(module, ResidualOnlyBlock):
                    handle.write("  - gates: `none`; descriptors: `none`; residual formula: `out = x + alpha * depthwise_conv(x)`\n")
                if isinstance(module, FADCGV2dAblation):
                    handle.write(f"  - config: `{module.config}`\n")
                    handle.write(f"  - freq_init: `{module.freq_init}`\n")
            handle.write("\n")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        handle.write("## Parameter Deltas\n\n")
        for row in comp_rows:
            handle.write(
                f"- {row['method']}: params={row['params_abs']}, delta_vs_fcn={row['params_delta_vs_fcn']}, "
                f"MACs={row['macs_abs']}, MACs_delta_vs_fcn={row['macs_delta_vs_fcn']}\n"
            )
        residual = next((row for row in comp_rows if row["method"] == "FCN + Residual-only"), None)
        v2d = next((row for row in comp_rows if row["method"] == "FCN + V2d-noBoundary-noLocalVar"), None)
        if residual and v2d:
            handle.write(
                f"- Params(V2d) - Params(Residual-only) = `{int(v2d['params_abs']) - int(residual['params_abs'])}`\n"
            )


def write_report(
    args: argparse.Namespace,
    out_dir: Path,
    raw_rows: List[Dict[str, object]],
    summary: List[Dict[str, object]],
    deltas: List[Dict[str, object]],
    comp_rows: List[Dict[str, object]],
) -> None:
    branch = run_cmd(["git", "branch", "--show-current"])
    commit = run_cmd(["git", "rev-parse", "HEAD"])
    status = run_cmd(["git", "status", "--short"])
    data_root = Path(args.data_root).resolve()
    sizes = split_sizes(data_root)
    completed = [row for row in raw_rows if row["status"] == "completed"]
    failures = [row for row in raw_rows if row["status"] != "completed"]
    delta_mious = np.array([float(row["delta_miou"]) for row in deltas], dtype=np.float64) if deltas else np.array([])
    delta_dices = np.array([float(row["delta_dice"]) for row in deltas], dtype=np.float64) if deltas else np.array([])

    with (out_dir / "task_012_report.md").open("w", encoding="utf-8") as handle:
        handle.write("# Task 012 FCN Residual-Control Report\n\n")
        handle.write(f"- git branch: `{branch}`\n")
        handle.write(f"- git commit hash: `{commit}`\n")
        handle.write(f"- dirty-worktree status:\n\n```text\n{status if status else 'clean'}\n```\n\n")
        handle.write(f"- dataset path: `{data_root}`\n")
        handle.write(f"- split sizes: train={sizes['train']}, val={sizes['val']}, test={sizes['test']}\n")
        handle.write(f"- output directory: `{out_dir}`\n")
        handle.write(f"- PyTorch: `{torch.__version__}`; CUDA: `{torch.version.cuda}`\n\n")
        handle.write("## Model Configs\n\n")
        handle.write("- `FCN + Residual-only`: two depthwise residual blocks at the same 512-channel and 4096-channel stages as FCN V2d; formula `out = x + alpha * z`, alpha initialized to 0.5; no gates or descriptors.\n")
        handle.write(f"- `FCN + V2d-noBoundary-noLocalVar`: `FCNWithFADCGV2dAblation(variant=\"no_boundary_no_local_var\")`; config `{VARIANT_CONFIGS['no_boundary_no_local_var']}`; `freq_init=random`.\n\n")
        handle.write("## Protocol\n\n")
        handle.write(
            f"- input size: {args.img_size} x {args.img_size}; batch size: {args.batch_size}; epochs: {args.epochs}\n"
            f"- optimizer: Adam, lr={args.lr}, cosine eta_min={args.min_lr}; loss=BCEWithLogitsLoss\n"
            f"- seeds: {', '.join(str(seed) for seed in args.seeds)}\n"
            "- train is used for training, val for checkpoint selection, and test for final evaluation only\n"
            "- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees\n"
            "- metrics: sigmoid threshold 0.5; mIoU averages foreground/background IoU; Dice is foreground Dice\n"
            "- complexity: THOP MACs when available; latency uses batch size 1, eval mode, no data loading\n\n"
        )
        handle.write("## Completed Runs\n\n")
        handle.write(f"- completed rows: {len(completed)}\n")
        handle.write(f"- complexity rows: {len(comp_rows)}\n\n")
        handle.write("## Failed Or Skipped Runs\n\n")
        if failures:
            for row in failures:
                handle.write(f"- {row['method']} seed {row['seed']}: {row['notes']}\n")
        else:
            handle.write("- None.\n")
        handle.write("\n## Raw Per-Seed Results\n\n")
        handle.write("| method | seed | val mIoU | test mIoU | test Dice | best epoch |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in sorted(completed, key=lambda x: (str(x["method"]), int(x["seed"]))):
            handle.write(
                f"| {row['method']} | {row['seed']} | {float(row['val_miou']):.6f} | "
                f"{float(row['test_miou']):.6f} | {float(row['test_dice']):.6f} | {row['best_epoch']} |\n"
            )
        handle.write("\n## Summary\n\n")
        handle.write("| method | seeds | test mIoU mean/std | test Dice mean/std | params | MACs | latency ms | FPS |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            handle.write(
                f"| {row['method']} | {row['num_seeds']} | {row['test_miou_mean']} +/- {row['test_miou_std']} | "
                f"{row['test_dice_mean']} +/- {row['test_dice_std']} | {row['params_abs']} | {row['macs_abs']} | "
                f"{row['latency_ms_mean']} | {row['fps']} |\n"
            )
        handle.write("\n## Paired Seed Delta\n\n")
        handle.write("| seed | residual mIoU | V2d mIoU | delta mIoU | residual Dice | V2d Dice | delta Dice |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in deltas:
            handle.write(
                f"| {row['seed']} | {row['residual_only_test_miou']} | {row['v2d_test_miou']} | {row['delta_miou']} | "
                f"{row['residual_only_test_dice']} | {row['v2d_test_dice']} | {row['delta_dice']} |\n"
            )
        if len(delta_mious):
            handle.write(
                f"\n- mean paired delta mIoU: `{delta_mious.mean():.6f}`; std: `{delta_mious.std(ddof=1) if len(delta_mious) > 1 else 0.0:.6f}`\n"
            )
            handle.write(
                f"- mean paired delta Dice: `{delta_dices.mean():.6f}`; std: `{delta_dices.std(ddof=1) if len(delta_dices) > 1 else 0.0:.6f}`\n"
            )
        handle.write("\n## Decision\n\n")
        if len(delta_mious):
            mean_delta = float(delta_mious.mean())
            positive = int((delta_mious > 0).sum())
            if mean_delta >= 0.005 and positive >= max(2, len(delta_mious) - 1):
                decision = "positive evidence for V2d beyond residual learning on FCN"
            elif mean_delta >= 0.002:
                decision = "weak positive evidence; extend to six seeds before making a paper claim"
            elif mean_delta < 0:
                decision = "residual-only is better; FCN should be treated as a negative residual-control result"
            else:
                decision = "below 0.002 mIoU or inconsistent; attribute FCN gain mainly to residual enhancement"
            handle.write(f"- Decision rule result: {decision}.\n")
            handle.write(f"- Positive paired seeds: {positive}/{len(delta_mious)}.\n")
        else:
            handle.write("- No complete paired seeds available.\n")
        handle.write("- No LaTeX paper edits were made.\n")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) if args.output_root else REPO_ROOT / "exp" / "task_012_fcn_residual_control" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    write_environment(out_dir)
    (out_dir / "run_commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"python scripts/task_012_fcn_residual_control.py --output-root {out_dir} "
        f"--data-root {args.data_root} --seeds {' '.join(map(str, args.seeds))} "
        f"--epochs {args.epochs} --batch-size {args.batch_size} --num-workers {args.num_workers} --resume\n",
        encoding="utf-8",
    )

    raw_path = out_dir / "raw_metrics_per_seed.csv"
    raw_by_key = read_raw(raw_path) if args.resume else {}

    if not args.skip_training:
        for spec in MODEL_SPECS:
            for seed in args.seeds:
                key = (str(spec["method"]), seed)
                if args.resume and key in raw_by_key:
                    continue
                try:
                    result = train_one(args, spec, seed, out_dir)
                    raw_by_key[key] = result.__dict__
                except Exception as exc:
                    raw_by_key[key] = {
                        "backbone": "FCN",
                        "method": spec["method"],
                        "variant": spec["variant"],
                        "seed": seed,
                        "val_miou": "NA",
                        "val_dice": "NA",
                        "test_miou": "NA",
                        "test_dice": "NA",
                        "best_epoch": "NA",
                        "checkpoint_path": "NA",
                        "status": "failed",
                        "notes": repr(exc),
                    }
                rows = sorted(raw_by_key.values(), key=lambda row: (str(row["method"]), int(row["seed"])))
                write_csv(raw_path, rows, raw_fieldnames())

    raw_rows = sorted(raw_by_key.values(), key=lambda row: (str(row["method"]), int(row["seed"])))
    write_csv(raw_path, raw_rows, raw_fieldnames())
    comp_rows = [] if args.skip_complexity else complexity_rows(args, out_dir)
    summary = summarize(raw_rows, comp_rows, out_dir)
    deltas = paired_deltas(raw_rows, out_dir)
    write_implementation_audit(args, out_dir, comp_rows)
    write_report(args, out_dir.resolve(), raw_rows, summary, deltas, comp_rows)
    print(f"experiment_dir={out_dir.resolve()}")
    print(f"summary_mean_std={out_dir.resolve() / 'summary_mean_std.csv'}")
    print(f"paired_seed_delta={out_dir.resolve() / 'paired_seed_delta.csv'}")


if __name__ == "__main__":
    main()
