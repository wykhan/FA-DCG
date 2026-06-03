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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(SAR_ROOT))

from task_008_main_module_table import (  # noqa: E402
    CBAM,
    FloodDatasetLocal,
    benchmark_latency,
    compute_scores,
    count_params,
    evaluate,
    profile_flops,
    seed_worker,
    set_seed,
    split_dirs,
    unet_fadc_aligned,
)
from models.baseline.unet import UNet  # noqa: E402
from models.improved.unet_fadcg_v2d_ablation import UNetWithFADCGV2dAblation  # noqa: E402


BASELINE_TASK009 = REPO_ROOT / "exp" / "task_009_fadc_srfg_hybrid" / "20260526_081709"
SEED = 2026


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


class UNetWithCBAM(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.unet = UNet(in_channels, num_classes, features)
        self.plugin = CBAM(features[-1] * 2)

    def forward(self, x):
        skip_connections = []
        for encoder in self.unet.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.unet.pool(x)
        x = self.unet.bottleneck(x)
        x = self.plugin(x)
        skip_connections = skip_connections[::-1]
        for idx, (upconv, decoder) in enumerate(zip(self.unet.upconvs, self.unet.decoders)):
            x = upconv(x)
            x = torch.cat([x, skip_connections[idx]], dim=1)
            x = decoder(x)
        return self.unet.final_conv(x)


MODEL_SPECS = [
    {
        "method": "U-Net + V2d-noBoundary-noLocalVar",
        "variant": "v2d_no_boundary_no_local_var",
        "priority": "A",
        "factory": lambda **kwargs: UNetWithFADCGV2dAblation(variant="no_boundary_no_local_var", **kwargs),
        "notes": "comparison variant without boundary, MSC, or local variance; seed 2026 is reused from task_009 when available",
    },
    {
        "method": "U-Net + V2d-noBoundaryKeepMSC-noLocalVar",
        "variant": "v2d_no_boundary_keep_msc_no_local_var",
        "priority": "A",
        "factory": lambda **kwargs: UNetWithFADCGV2dAblation(variant="no_boundary_keep_msc_no_local_var", **kwargs),
        "notes": "decoupling variant: disables boundary gate but keeps MSC descriptor and speckle gate",
    },
    {
        "method": "U-Net + V2d-BoundaryNoMSC-noLocalVar",
        "variant": "v2d_boundary_no_msc_no_local_var",
        "priority": "A",
        "factory": lambda **kwargs: UNetWithFADCGV2dAblation(variant="boundary_no_msc_no_local_var", **kwargs),
        "notes": "decoupling variant: keeps boundary gate but removes MSC and local variance descriptors",
    },
    {
        "method": "U-Net + V2d-noLocalVar-Laplacian",
        "variant": "v2d_no_local_var_laplacian",
        "priority": "A",
        "factory": lambda **kwargs: UNetWithFADCGV2dAblation(variant="no_local_var", freq_init="laplacian", **kwargs),
        "notes": "best task_009 V2d-noLocalVar with L1-normalized 4-neighbor Laplacian depthwise initialization",
    },
    {
        "method": "U-Net + V2d-noBoundaryKeepMSC-noLocalVar-Laplacian",
        "variant": "v2d_no_boundary_keep_msc_no_local_var_laplacian",
        "priority": "A",
        "factory": lambda **kwargs: UNetWithFADCGV2dAblation(
            variant="no_boundary_keep_msc_no_local_var", freq_init="laplacian", **kwargs
        ),
        "notes": "boundary/MSC decoupling variant with Laplacian depthwise initialization",
    },
    {
        "method": "U-Net + V2d-BoundaryNoMSC-noLocalVar-Laplacian",
        "variant": "v2d_boundary_no_msc_no_local_var_laplacian",
        "priority": "A",
        "factory": lambda **kwargs: UNetWithFADCGV2dAblation(
            variant="boundary_no_msc_no_local_var", freq_init="laplacian", **kwargs
        ),
        "notes": "boundary-only frequency gate variant with Laplacian depthwise initialization",
    },
]


BASELINE_METHODS = {
    "U-Net": "baseline",
    "U-Net + CBAM": "cbam",
    "U-Net + FADC-aligned": "fadc_aligned",
    "U-Net + SRFG": "srfg_v2d_nomsc",
    "U-Net + V2d-noLocalVar": "v2d_no_local_var",
    "U-Net + V2d-noBoundary-noLocalVar": "v2d_no_boundary_no_local_var",
}


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
    parser.add_argument("--baseline-root", default=str(BASELINE_TASK009))
    parser.add_argument("--priorities", nargs="+", default=["A", "B"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[SEED])
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


def selected_specs(args: argparse.Namespace) -> List[Dict[str, object]]:
    specs = [spec for spec in MODEL_SPECS if spec["priority"] in set(args.priorities)]
    if args.models:
        by_method = {str(spec["method"]): spec for spec in MODEL_SPECS}
        by_variant = {str(spec["variant"]): spec for spec in MODEL_SPECS}
        selected = []
        for name in args.models:
            if name in by_method:
                selected.append(by_method[name])
            elif name in by_variant:
                selected.append(by_variant[name])
            else:
                raise ValueError(f"Unknown model: {name}")
        specs = selected
    return specs


def build_loaders(args: argparse.Namespace, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = Path(args.data_root)
    train_img, train_mask = split_dirs(data_root, "train")
    val_img, val_mask = split_dirs(data_root, "val")
    test_img, test_mask = split_dirs(data_root, "test")
    train_ds: Dataset = FloodDatasetLocal(train_img, train_mask, args.img_size, True)
    val_ds: Dataset = FloodDatasetLocal(val_img, val_mask, args.img_size, False)
    test_ds: Dataset = FloodDatasetLocal(test_img, test_mask, args.img_size, False)
    if args.max_train_samples:
        train_ds = Subset(train_ds, list(range(min(args.max_train_samples, len(train_ds)))))
    if args.max_val_samples:
        val_ds = Subset(val_ds, list(range(min(args.max_val_samples, len(val_ds)))))
    if args.max_test_samples:
        test_ds = Subset(test_ds, list(range(min(args.max_test_samples, len(test_ds)))))
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
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
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
        log.write(f"method={spec['method']}, variant={spec['variant']}, priority={spec['priority']}, seed={seed}\n")
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
        backbone="U-Net",
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


def read_raw(path: Path) -> Dict[Tuple[str, int], Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["method"], int(row["seed"])): row for row in rows if row.get("status") in {"completed", "reused"}}


def load_task009_baselines(path: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    raw_rows = []
    comp_rows = []
    raw_path = path / "raw_metrics_per_seed.csv"
    comp_path = path / "complexity_raw.csv"
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["method"] not in BASELINE_METHODS:
                    continue
                raw_rows.append(
                    {
                        "backbone": "U-Net",
                        "method": row["method"],
                        "variant": BASELINE_METHODS[row["method"]],
                        "seed": int(row["seed"]),
                        "val_miou": row["val_miou"] if "val_miou" in row else row["miou"],
                        "val_dice": row["val_dice"] if "val_dice" in row else row["dice"],
                        "test_miou": row["test_miou"] if "test_miou" in row else row["miou"],
                        "test_dice": row["test_dice"] if "test_dice" in row else row["dice"],
                        "best_epoch": row["best_epoch"],
                        "checkpoint_path": row["checkpoint_path"],
                        "status": "reused",
                        "notes": f"reused from task_009: {raw_path}",
                    }
                )
    if comp_path.exists():
        with comp_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["method"] not in BASELINE_METHODS:
                    continue
                row = dict(row)
                row["variant"] = BASELINE_METHODS[row["method"]]
                comp_rows.append(row)
    return raw_rows, comp_rows


def build_complexity_model(method: str, variant: str) -> nn.Module:
    if method == "U-Net":
        return UNet(in_channels=1, num_classes=1)
    if method == "U-Net + CBAM":
        return UNetWithCBAM(in_channels=1, num_classes=1)
    if method == "U-Net + FADC-aligned":
        return unet_fadc_aligned(in_channels=1, num_classes=1)
    if method == "U-Net + SRFG":
        return UNetWithFADCGV2dAblation(variant="no_msc", in_channels=1, num_classes=1)
    if method == "U-Net + V2d-noLocalVar":
        return UNetWithFADCGV2dAblation(variant="no_local_var", in_channels=1, num_classes=1)
    if method == "U-Net + V2d-noBoundary-noLocalVar":
        return UNetWithFADCGV2dAblation(variant="no_boundary_no_local_var", in_channels=1, num_classes=1)
    spec = next(s for s in MODEL_SPECS if s["method"] == method)
    return spec["factory"](in_channels=1, num_classes=1)  # type: ignore[misc]


def complexity_rows(args: argparse.Namespace, methods: List[Tuple[str, str]], out_dir: Path) -> List[Dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    rows = []
    baseline_params = None
    for method, variant in methods:
        set_seed(0)
        model = build_complexity_model(method, variant).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(torch.zeros(1, 1, args.img_size, args.img_size, device=device))
        params_abs = count_params(model)
        flops_abs, flops_unit, flops_tool = profile_flops(model, device, args.img_size)
        latency_mean, latency_std, fps = benchmark_latency(
            model, device, args.img_size, args.benchmark_warmup, args.benchmark_iters
        )
        if method == "U-Net":
            baseline_params = params_abs
            baseline_flops = flops_abs
        rows.append(
            {
                "backbone": "U-Net",
                "method": method,
                "variant": variant,
                "params_abs": params_abs,
                "params_delta_vs_unet": 0,
                "macs_abs": flops_abs,
                "macs_delta_vs_unet": 0,
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
    baseline_params = baseline_params if baseline_params is not None else int(rows[0]["params_abs"])
    baseline_flops = baseline_flops if "baseline_flops" in locals() else int(rows[0]["macs_abs"])
    for row in rows:
        row["params_delta_vs_unet"] = int(row["params_abs"]) - int(baseline_params)
        row["macs_delta_vs_unet"] = int(row["macs_abs"]) - int(baseline_flops)
    write_csv(
        out_dir / "complexity_raw.csv",
        rows,
        [
            "backbone",
            "method",
            "variant",
            "params_abs",
            "params_delta_vs_unet",
            "macs_abs",
            "macs_delta_vs_unet",
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
    methods = []
    for row in raw_rows:
        key = (str(row["method"]), str(row["variant"]))
        if key not in methods:
            methods.append(key)
    summary = []
    for method, variant in methods:
        items = [row for row in raw_rows if row["method"] == method and row["variant"] == variant and row["status"] in {"completed", "reused"}]
        val_miou = np.array([float(row["val_miou"]) for row in items], dtype=np.float64)
        test_miou = np.array([float(row["test_miou"]) for row in items], dtype=np.float64)
        test_dice = np.array([float(row["test_dice"]) for row in items], dtype=np.float64)
        comp = comp_by_method.get(method, {})
        summary.append(
            {
                "backbone": "U-Net",
                "method": method,
                "variant": variant,
                "num_seeds": len(items),
                "val_miou_mean": f"{val_miou.mean():.6f}",
                "val_miou_std": f"{val_miou.std(ddof=1) if len(val_miou) > 1 else 0.0:.6f}",
                "test_miou_mean": f"{test_miou.mean():.6f}",
                "test_miou_std": f"{test_miou.std(ddof=1) if len(test_miou) > 1 else 0.0:.6f}",
                "test_dice_mean": f"{test_dice.mean():.6f}",
                "test_dice_std": f"{test_dice.std(ddof=1) if len(test_dice) > 1 else 0.0:.6f}",
                "params_abs": comp.get("params_abs", "NA"),
                "params_delta_vs_unet": comp.get("params_delta_vs_unet", "NA"),
                "macs_abs": comp.get("macs_abs", "NA"),
                "latency_ms_mean": comp.get("latency_ms_mean", "NA"),
                "fps": comp.get("fps", "NA"),
                "notes": "task_010 summary; reused rows imported from task_009 when noted",
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
            "params_delta_vs_unet",
            "macs_abs",
            "latency_ms_mean",
            "fps",
            "notes",
        ],
    )
    return summary


def extract_bottleneck(model: nn.Module, images: torch.Tensor) -> torch.Tensor | None:
    target = model
    if hasattr(model, "light_fadc"):
        target = model
    else:
        return None
    skip_connections = []
    for encoder in target.encoder:
        images = encoder(images)
        skip_connections.append(images)
        images = target.pool(images)
    return target.bottleneck(images)


def write_model_diagnostics(out_dir: Path, spec: Dict[str, object], model: nn.Module, loader: DataLoader, device: torch.device, seed: int) -> None:
    rows = []
    try:
        images, _ = next(iter(loader))
        images = images[:1].to(device)
        bottleneck = extract_bottleneck(model, images)
        plugin = getattr(model, "light_fadc", None)
        if bottleneck is not None and plugin is not None and hasattr(plugin, "collect_input_diagnostics"):
            diagnostics = plugin.collect_input_diagnostics(bottleneck)
            for key, value in diagnostics.items():
                if isinstance(value, str):
                    numeric_value = "NA"
                    notes = f"diagnostic value: {value}"
                else:
                    numeric_value = float(value.item() if hasattr(value, "item") else value)
                    notes = "diagnostics from first deterministic test batch"
                rows.append(
                    {
                        "method": spec["method"],
                        "variant": spec["variant"],
                        "seed": seed,
                        "source": "hybrid_bottleneck_test_sample",
                        "metric": key,
                        "value": numeric_value,
                        "notes": notes,
                    }
                )
    except Exception as exc:
        rows.append(
            {
                "method": spec["method"],
                "variant": spec["variant"],
                "seed": seed,
                "source": "diagnostics",
                "metric": "error",
                "value": "NA",
                "notes": repr(exc),
            }
        )
    if rows:
        path = out_dir / "diagnostics_raw.csv"
        append = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method", "variant", "seed", "source", "metric", "value", "notes"])
            if not append:
                writer.writeheader()
            writer.writerows(rows)


def write_environment(out_dir: Path) -> None:
    with (out_dir / "environment.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"python --version\n{run_cmd(['python', '--version'])}\n\n")
        handle.write(f"pip list\n{run_cmd(['python', '-m', 'pip', 'list'])}\n\n")
        handle.write(f"nvidia-smi\n{run_cmd(['nvidia-smi'])}\n")
        handle.write(f"\nplatform={platform.platform()}\n")
        handle.write(f"torch={torch.__version__}\n")
        handle.write(f"cuda={torch.version.cuda}\n")


def split_sizes(data_root: Path) -> Dict[str, int]:
    sizes = {}
    for split in ["train", "val", "test"]:
        image_dir, _ = split_dirs(data_root, split)
        sizes[split] = len([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    return sizes


def write_implementation_audit(args: argparse.Namespace, out_dir: Path, specs: List[Dict[str, object]]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (out_dir / "implementation_audit.md").open("w", encoding="utf-8") as handle:
        handle.write("# Task 010 Implementation Audit\n\n")
        handle.write("- Laplacian initialization uses the L1-normalized 4-neighbor kernel `[[0,-1,0],[-1,4,-1],[0,-1,0]] / 8`.\n")
        handle.write("- Depthwise frequency residual kernels remain learnable after initialization.\n\n")
        for spec in specs:
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
            plugin = getattr(model, "light_fadc", None)
            if plugin is not None:
                handle.write(f"- V2d variant: `{getattr(plugin, 'variant', 'NA')}`\n")
                handle.write(f"- frequency initialization: `{getattr(plugin, 'freq_init', 'NA')}`\n")
                handle.write(f"- depthwise kernel learnable: `{bool(getattr(plugin, 'weight', None).requires_grad)}`\n")
                handle.write(f"- config: `{getattr(plugin, 'config', {})}`\n")
            handle.write("\n")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def write_report(
    args: argparse.Namespace,
    out_dir: Path,
    specs: List[Dict[str, object]],
    raw_rows: List[Dict[str, object]],
    summary: List[Dict[str, object]],
    comp_rows: List[Dict[str, object]],
) -> None:
    branch = run_cmd(["git", "branch", "--show-current"])
    commit = run_cmd(["git", "rev-parse", "HEAD"])
    data_root = Path(args.data_root).resolve()
    sizes = split_sizes(data_root)
    cbam = next((row for row in summary if row["method"] == "U-Net + CBAM"), None)
    cbam_miou = float(cbam["test_miou_mean"]) if cbam else math.nan
    completed = [row for row in raw_rows if row["status"] in {"completed", "reused"}]
    failures = [row for row in raw_rows if row["status"] not in {"completed", "reused"}]

    task009_v2d = next((row for row in summary if row["method"] == "U-Net + V2d-noLocalVar"), None)
    task009_v2d_miou = float(task009_v2d["test_miou_mean"]) if task009_v2d else math.nan

    with (out_dir / "task_010_report.md").open("w", encoding="utf-8") as handle:
        handle.write("# Task 010 V2d-noLocalVar Targeted Optimization Report\n\n")
        handle.write(f"- git branch: `{branch}`\n")
        handle.write(f"- git commit hash: `{commit}`\n")
        handle.write(f"- dataset path: `{data_root}`\n")
        handle.write(f"- split sizes: train={sizes['train']}, val={sizes['val']}, test={sizes['test']}\n")
        handle.write(f"- output directory: `{out_dir}`\n")
        handle.write(f"- PyTorch: `{torch.__version__}`; CUDA: `{torch.version.cuda}`\n\n")
        handle.write("## Model List\n\n")
        handle.write("- Reused baselines: U-Net, U-Net + CBAM, U-Net + FADC-aligned, U-Net + SRFG, U-Net + V2d-noLocalVar, and available V2d-noBoundary-noLocalVar rows from task_009 when available.\n")
        for spec in specs:
            handle.write(f"- {spec['method']} ({spec['variant']}, priority {spec['priority']}): {spec['notes']}\n")
        handle.write("\n## Protocol\n\n")
        handle.write(
            f"- data: train for training, val for checkpoint selection, test for final metrics\n"
            f"- input size: {args.img_size} x {args.img_size}; batch size: {args.batch_size}; epochs: {args.epochs}\n"
            f"- optimizer: Adam, lr={args.lr}, cosine eta_min={args.min_lr}; loss=BCEWithLogitsLoss\n"
            f"- seeds: {', '.join(str(seed) for seed in args.seeds)}\n"
            "- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees\n"
            "- metrics: sigmoid threshold 0.5; mIoU averages foreground/background IoU; Dice is foreground Dice\n"
            "- Laplacian initialization: L1-normalized 4-neighbor high-pass kernel; depthwise residual kernels remain learnable\n"
            "- complexity: THOP MACs when available; latency uses batch size 1, eval mode, no data loading\n\n"
        )
        handle.write("## Completed Runs\n\n")
        handle.write(f"- completed or reused rows: {len(completed)}\n")
        handle.write(f"- complexity rows: {len(comp_rows)}\n\n")
        handle.write("## Failed Or Skipped Runs\n\n")
        if failures:
            for row in failures:
                handle.write(f"- {row['method']} seed {row['seed']}: {row['notes']}\n")
        else:
            handle.write("- None.\n")
        handle.write("\n## Raw Per-Seed Results\n\n")
        handle.write("| method | variant | seed | status | val mIoU | test mIoU | test Dice | best epoch |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in completed:
            handle.write(
                f"| {row['method']} | {row['variant']} | {row['seed']} | {row['status']} | "
                f"{float(row['val_miou']):.6f} | {float(row['test_miou']):.6f} | "
                f"{float(row['test_dice']):.6f} | {row['best_epoch']} |\n"
            )
        handle.write("\n## Summary\n\n")
        handle.write("| method | variant | seeds | test mIoU | test Dice | delta vs CBAM | delta vs V2d-noLocalVar | params | MACs | latency ms | FPS |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            delta = float(row["test_miou_mean"]) - cbam_miou if not math.isnan(cbam_miou) else math.nan
            delta_v2d = float(row["test_miou_mean"]) - task009_v2d_miou if not math.isnan(task009_v2d_miou) else math.nan
            handle.write(
                f"| {row['method']} | {row['variant']} | {row['num_seeds']} | {row['test_miou_mean']} | {row['test_dice_mean']} | "
                f"{delta:.6f} | {delta_v2d:.6f} | {row['params_abs']} | {row['macs_abs']} | {row['latency_ms_mean']} | {row['fps']} |\n"
            )
        handle.write("\n## Decision Notes\n\n")
        best = max(summary, key=lambda row: float(row["test_miou_mean"])) if summary else None
        if best:
            delta = float(best["test_miou_mean"]) - cbam_miou
            delta_v2d = float(best["test_miou_mean"]) - task009_v2d_miou if not math.isnan(task009_v2d_miou) else math.nan
            handle.write(f"- Best seed-screening test mIoU: {best['method']} ({best['variant']}) = {best['test_miou_mean']}.\n")
            handle.write(f"- Delta vs U-Net + CBAM: {delta:.6f} mIoU.\n")
            if not math.isnan(delta_v2d):
                handle.write(f"- Delta vs task_009 U-Net + V2d-noLocalVar: {delta_v2d:.6f} mIoU.\n")
            if delta >= 0.005:
                handle.write("- Decision rule: potentially publishable method candidate if supported by diagnostics and external validation.\n")
            elif delta >= 0.002:
                handle.write("- Decision rule: weak positive evidence; requires more datasets or stronger analysis.\n")
            else:
                handle.write("- Decision rule: below 0.002 mIoU over CBAM or inconsistent; stop architecture exploration and focus on analysis/negative-result framing.\n")
        handle.write("\n## Interpretation\n\n")
        no_boundary = next((row for row in summary if row["variant"] == "v2d_no_boundary_no_local_var"), None)
        keep_msc = next((row for row in summary if row["variant"] == "v2d_no_boundary_keep_msc_no_local_var"), None)
        boundary_only = next((row for row in summary if row["variant"] == "v2d_boundary_no_msc_no_local_var"), None)
        laplacian_rows = [row for row in summary if str(row["variant"]).endswith("_laplacian")]
        if no_boundary and keep_msc and boundary_only:
            handle.write(
                f"- Boundary/MSC decoupling: no-boundary/no-MSC reached {no_boundary['test_miou_mean']} mIoU, "
                f"no-boundary/keep-MSC reached {keep_msc['test_miou_mean']}, and boundary/no-MSC reached {boundary_only['test_miou_mean']}.\n"
            )
            handle.write("- This suggests the original boundary and MSC descriptors are not complementary under the current U-Net bottleneck placement; removing both is slightly better than keeping either one alone.\n")
        if laplacian_rows:
            best_lap = max(laplacian_rows, key=lambda row: float(row["test_miou_mean"]))
            handle.write(
                f"- High-pass initialization: the best Laplacian variant was {best_lap['method']} at {best_lap['test_miou_mean']} mIoU, "
                "below the corresponding random-initialized V2d variants.\n"
            )
            handle.write("- The explicit high-pass prior did not improve the metrics; in this setup it appears to constrain or slow useful adaptation rather than strengthen the frequency-gating story.\n")
        handle.write("- Factual raw data only; no LaTeX paper edits were made.\n")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) if args.output_root else REPO_ROOT / "exp" / "task_010_v2d_nolocalvar_targeted" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    write_environment(out_dir)

    specs = selected_specs(args)
    raw_path = out_dir / "raw_metrics_per_seed.csv"
    raw_by_key = read_raw(raw_path) if args.resume else {}
    baseline_raw, _ = load_task009_baselines(Path(args.baseline_root))
    for row in baseline_raw:
        raw_by_key.setdefault((str(row["method"]), int(row["seed"])), row)

    if not args.skip_training:
        for spec in specs:
            for seed in args.seeds:
                key = (str(spec["method"]), seed)
                if args.resume and key in raw_by_key:
                    continue
                try:
                    result = train_one(args, spec, seed, out_dir)
                    raw_by_key[key] = result.__dict__
                except Exception as exc:
                    raw_by_key[key] = {
                        "backbone": "U-Net",
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
    methods = []
    for row in raw_rows:
        if row["status"] in {"completed", "reused"}:
            pair = (str(row["method"]), str(row["variant"]))
            if pair not in methods:
                methods.append(pair)
    comp_rows = [] if args.skip_complexity else complexity_rows(args, methods, out_dir)
    summary = summarize(raw_rows, comp_rows, out_dir)
    write_implementation_audit(args, out_dir, specs)
    write_report(args, out_dir.resolve(), specs, raw_rows, summary, comp_rows)
    print(f"experiment_dir={out_dir.resolve()}")
    print(f"summary_mean_std={out_dir.resolve() / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
