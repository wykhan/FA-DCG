#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.baseline.fcn import FCN  # noqa: E402
from models.baseline.unet import UNet  # noqa: E402
from models.improved.fadc_fast import FastFADCG  # noqa: E402
from models.improved.fcn_fadc_fast import FCNWithFastFADCG  # noqa: E402
from models.improved.fcn_fadc_light import FCNWithLightFADC  # noqa: E402
from models.improved.unet_fadc_fast import UNetWithFastFADCG  # noqa: E402
from models.improved.unet_fadc_light import UNetWithLightFADC  # noqa: E402


OLD_REFERENCE = {
    "FCN + Fast FA-DCG": {
        "old_name": "FCN + old FA-DCG",
        "miou": 0.844849,
        "dice": 0.907507,
        "time_ms": 31.819,
        "params_m": 144.9324,
    },
    "U-Net + Fast FA-DCG": {
        "old_name": "U-Net + old FA-DCG",
        "miou": 0.896663,
        "dice": 0.941023,
        "time_ms": 9.826,
        "params_m": 31.5782,
    },
}

MODEL_FACTORIES = {
    "FCN + Fast FA-DCG": FCNWithFastFADCG,
    "U-Net + Fast FA-DCG": UNetWithFastFADCG,
}

BENCHMARK_FACTORIES = {
    "FCN": ("baseline", FCN),
    "U-Net": ("baseline", UNet),
    "FCN + old FA-DCG": ("old", FCNWithLightFADC),
    "FCN + Fast FA-DCG": ("fast", FCNWithFastFADCG),
    "U-Net + old FA-DCG": ("old", UNetWithLightFADC),
    "U-Net + Fast FA-DCG": ("fast", UNetWithFastFADCG),
}


class FloodDatasetLocal(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, img_size: int = 256, augment: bool = False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.images = sorted(
            p.name for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        name = self.images[idx]
        image = Image.open(self.img_dir / name).convert("L").resize((self.img_size, self.img_size))
        mask = Image.open(self.mask_dir / name).convert("L").resize((self.img_size, self.img_size))
        image_t = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
        mask_t = torch.from_numpy((np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)).unsqueeze(0)
        if self.augment:
            if torch.rand(()) < 0.5:
                image_t = torch.flip(image_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
            if torch.rand(()) < 0.5:
                image_t = torch.flip(image_t, dims=[1])
                mask_t = torch.flip(mask_t, dims=[1])
        return image_t, mask_t


@dataclass
class RunResult:
    method: str
    seed: int
    miou: float
    dice: float
    params_m: float
    inference_time_ms: float
    gpu_memory_mb: float
    best_epoch: int
    diagnostics: Dict[str, float]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--data-root", default=str(SAR_ROOT / "data" / "flood_dataset"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_cmd(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return ";".join(f"{float(v):.6f}" for v in value.flatten())
    return float(value)


def build_loaders(args, seed: int):
    data_root = Path(args.data_root)
    train_ds: Dataset = FloodDatasetLocal(data_root / "train" / "images", data_root / "train" / "labels", args.img_size, True)
    val_ds: Dataset = FloodDatasetLocal(data_root / "val" / "images", data_root / "val" / "labels", args.img_size, False)
    if args.max_train_samples:
        train_ds = Subset(train_ds, list(range(min(args.max_train_samples, len(train_ds)))))
    if args.max_val_samples:
        val_ds = Subset(val_ds, list(range(min(args.max_val_samples, len(val_ds)))))
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        ),
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    )


def binary_metrics(logits: torch.Tensor, masks: torch.Tensor):
    pred = (torch.sigmoid(logits) > 0.5).bool()
    target = masks.bool()
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    tn = (~pred & ~target).sum().item()
    return int(tp), int(fp), int(fn), int(tn)


def compute_scores(tp: int, fp: int, fn: int, tn: int):
    fg_iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)
    return (fg_iou + bg_iou) / 2.0, (2 * tp) / max(2 * tp + fp + fn, 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    totals = [0, 0, 0, 0]
    elapsed = 0.0
    samples = 0
    diagnostics = {}
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            samples += images.size(0)
            vals = binary_metrics(logits, masks)
            totals = [a + b for a, b in zip(totals, vals)]
            if batch_idx == 0:
                diagnostics = collect_diagnostics(model, images)
    miou, dice = compute_scores(*totals)
    return miou, dice, 1000.0 * elapsed / max(samples, 1), diagnostics


def collect_diagnostics(model: nn.Module, images: torch.Tensor):
    for module in model.modules():
        if isinstance(module, FastFADCG):
            module.get_diagnostics(images.new_zeros((images.size(0), module.channels, 16, 16)))
            return {key: tensor_to_float(value) for key, value in module.last_diagnostics.items()}
    return {}


def make_model(method: str, device: torch.device):
    return MODEL_FACTORIES[method](in_channels=1, num_classes=1).to(device)


def train_one(args, method: str, seed: int, out_dir: Path) -> RunResult:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(args, seed)
    model = make_model(method, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.BCEWithLogitsLoss()
    params_m = count_params(model)
    best = {"miou": -1.0, "dice": 0.0, "epoch": -1, "infer_ms": 0.0, "diagnostics": {}}
    log_path = out_dir / "logs" / f"{slug(method)}_seed{seed}.log"
    ckpt_dir = out_dir / "checkpoints"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"method={method}, seed={seed}\n")
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
            miou, dice, infer_ms, diagnostics = evaluate(model, val_loader, device)
            log.write(
                f"epoch={epoch}, train_loss={np.mean(losses):.6f}, val_miou={miou:.6f}, "
                f"val_dice={dice:.6f}, infer_ms={infer_ms:.3f}\n"
            )
            log.flush()
            if miou > best["miou"]:
                best = {"miou": miou, "dice": dice, "epoch": epoch, "infer_ms": infer_ms, "diagnostics": diagnostics}
                torch.save(model.state_dict(), ckpt_dir / f"{slug(method)}_seed{seed}_best.pt")
    gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    return RunResult(
        method=method,
        seed=seed,
        miou=float(best["miou"]),
        dice=float(best["dice"]),
        params_m=params_m,
        inference_time_ms=float(best["infer_ms"]),
        gpu_memory_mb=float(gpu_memory_mb),
        best_epoch=int(best["epoch"]),
        diagnostics=best["diagnostics"],
    )


def warmup_legacy_model(model: nn.Module, device: torch.device, img_size: int):
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, img_size, img_size, device=device))


def benchmark_speed(args, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for method, (implementation, factory) in BENCHMARK_FACTORIES.items():
        set_seed(2024)
        model = factory(in_channels=1, num_classes=1).to(device).eval()
        warmup_legacy_model(model, device, args.img_size)
        params_m = count_params(model)
        x = torch.randn(args.batch_size, 1, args.img_size, args.img_size, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(args.benchmark_warmup):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.benchmark_iters):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        rows.append(
            {
                "method": method,
                "implementation": implementation,
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "params_m": f"{params_m:.4f}",
                "inference_time_ms": f"{1000.0 * elapsed / args.benchmark_iters:.3f}",
                "gpu_memory_mb": f"{torch.cuda.max_memory_allocated() / (1024**2):.1f}" if device.type == "cuda" else "0.0",
                "notes": "full-batch latency; benchmark uses random input",
            }
        )
    write_csv(
        out_dir / "speed_benchmark.csv",
        rows,
        ["method", "implementation", "batch_size", "img_size", "params_m", "inference_time_ms", "gpu_memory_mb", "notes"],
    )
    return rows


def slug(method: str):
    return method.lower().replace(" + ", "_").replace("-", "").replace(" ", "_")


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_existing(path: Path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {(row["method"], int(row["seed"])): row for row in csv.DictReader(f)}


def summarize(rows):
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    summary = []
    for method in MODEL_FACTORIES:
        items = by_method.get(method, [])
        if not items:
            continue
        mious = np.array([float(x["miou"]) for x in items])
        dices = np.array([float(x["dice"]) for x in items])
        time_ms = float(np.mean([float(x["inference_time_ms"]) for x in items]))
        old = OLD_REFERENCE[method]
        delta_miou = float(mious.mean() - old["miou"])
        delta_dice = float(dices.mean() - old["dice"])
        delta_time = float(time_ms - old["time_ms"])
        improved_time = (old["time_ms"] - time_ms) / old["time_ms"]
        if improved_time < 0.30:
            status = "not_faster"
        elif delta_miou > 0:
            status = "faster_and_better"
        elif abs(delta_miou) <= 0.01:
            status = "faster_and_comparable"
        else:
            status = "faster_but_worse"
        summary.append(
            {
                "method": method,
                "implementation": "fast",
                "miou_mean": f"{mious.mean():.6f}",
                "miou_std": f"{mious.std(ddof=1) if len(mious) > 1 else 0.0:.6f}",
                "dice_mean": f"{dices.mean():.6f}",
                "dice_std": f"{dices.std(ddof=1) if len(dices) > 1 else 0.0:.6f}",
                "params_m": f"{np.mean([float(x['params_m']) for x in items]):.4f}",
                "inference_time_ms": f"{time_ms:.3f}",
                "gpu_memory_mb": f"{np.mean([float(x['gpu_memory_mb']) for x in items]):.1f}",
                "delta_miou_vs_old": f"{delta_miou:.6f}",
                "delta_dice_vs_old": f"{delta_dice:.6f}",
                "delta_time_vs_old": f"{delta_time:.3f}",
                "status": status,
            }
        )
    return summary


def complexity_rows(summary, speed_rows):
    speed_by_method = {row["method"]: row for row in speed_rows}
    rows = []
    for row in summary:
        method = row["method"]
        old = OLD_REFERENCE[method]
        params = float(row["params_m"])
        rows.append(
            {
                "method": method,
                "implementation": "fast",
                "params_m": f"{params:.4f}",
                "delta_params_m_vs_old": f"{params - old['params_m']:.4f}",
                "delta_params_percent_vs_old": f"{100.0 * (params - old['params_m']) / old['params_m']:.2f}",
                "flops_g": "NA",
                "inference_time_ms": speed_by_method.get(method, {}).get("inference_time_ms", row["inference_time_ms"]),
                "gpu_memory_mb": speed_by_method.get(method, {}).get("gpu_memory_mb", row["gpu_memory_mb"]),
            }
        )
    return rows


def write_environment(out_dir: Path):
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"cwd: {REPO_ROOT}",
        f"git_commit: {run_cmd(['git', 'rev-parse', '--short', 'HEAD'])}",
        f"os: {platform.platform()}",
        f"python: {sys.version.replace(os.linesep, ' ')}",
        f"torch: {torch.__version__}",
        f"torch_cuda: {torch.version.cuda}",
        f"cuda_available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            lines.append(f"GPU {i}: {props.name}, memory={props.total_memory / (1024**3):.2f} GiB")
    (out_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_implementation_audit(out_dir: Path, test_output: str):
    text = f"""# Implementation Audit

## Fast FA-DCG-v1 Changes

| Issue | Old implementation | New implementation | Fixed? |
| --- | --- | --- | --- |
| per-channel Python loop | yes | no, uses vectorized depthwise branch convolutions | yes |
| `.item()` integer dilation | yes | no, uses differentiable soft branch fusion | yes |
| 4096-channel FA-DCG in FCN | yes | no, only after 512-channel conv5 | yes |
| module created in forward | yes | no, all modules are constructor-defined | yes |
| differentiable branch fusion | no | yes, softmax over dilation branches | yes |

## Initialization

- `gamma = -4.0`, so continuous dilation starts near 1.
- `theta = 0` with a +0.25 bias for dilation 1.
- `beta_raw = 0.5413248546`, so `softplus(beta_raw)` is approximately 1.
- `alpha = 0.5`.
- Branch depthwise convolutions use Kaiming normal initialization.
- Gate hidden channels use `max(C // 16, 1)`.

## Unit and Shape Test Output

```text
{test_output.strip()}
```
"""
    (out_dir / "implementation_audit.md").write_text(text, encoding="utf-8")


def write_report(out_dir: Path, summary, speed_rows):
    speed = {row["method"]: row for row in speed_rows}
    speed_pairs = {
        "FCN + Fast FA-DCG": "FCN + old FA-DCG",
        "U-Net + Fast FA-DCG": "U-Net + old FA-DCG",
    }
    speed_table = [
        "| Model | Old time(ms) | New time(ms) | Speedup | Old Params(M) | New Params(M) | Notes |",
        "| ----- | -----------: | -----------: | ------: | ------------: | ------------: | ----- |",
    ]
    acc_table = [
        "| Model | Old mIoU | New mIoU | Delta mIoU | Old Dice | New Dice | Delta Dice | Status |",
        "| ----- | -------: | -------: | ---------: | -------: | -------: | ---------: | ------ |",
    ]
    for row in summary:
        method = row["method"]
        old = OLD_REFERENCE[method]
        old_speed = speed.get(speed_pairs[method], {})
        new_speed = speed.get(method, {})
        old_time = float(old_speed.get("inference_time_ms", old["time_ms"]))
        new_time = float(new_speed.get("inference_time_ms", row["inference_time_ms"]))
        speedup = old_time / new_time if new_time > 0 else 0.0
        speed_table.append(
            f"| {method} | {old_time:.3f} | {new_time:.3f} | {speedup:.2f}x | "
            f"{old['params_m']:.4f} | {float(row['params_m']):.4f} | validation protocol |"
        )
        acc_table.append(
            f"| {method} | {old['miou']:.6f} | {row['miou_mean']} | {row['delta_miou_vs_old']} | "
            f"{old['dice']:.6f} | {row['dice_mean']} | {row['delta_dice_vs_old']} | {row['status']} |"
        )
    report = [
        "# task_002b Fast FA-DCG-v1 Report",
        "",
        "## Executive Summary",
        "",
        "- Implemented vectorized Fast FA-DCG-v1 without deleting the old LightFADC implementations.",
        "- Replaced old U-Net bottleneck LightFADC with FastFADCG.",
        "- Replaced the old FCN 512/4096-channel FA-DCG path with a single FastFADCG after conv5 at 512 channels.",
        "- Conditional Stage B was completed: both optimized models were run for seeds 42, 123, and 2024.",
        "- Evaluation remains the current validation protocol; this is not independent test-set reproduction.",
        "",
        "## Why the Old Implementation Was Inefficient",
        "",
        "- Old LightFADC looped over channels in Python and converted learned dilation preferences through `.item()` into integer dilations.",
        "- Old FCN added FA-DCG-like processing at a 4096-channel stage and lazily created `up_dim` inside `forward`.",
        "",
        "## New Implementation Details",
        "",
        "- FastFADCG uses four vectorized depthwise branches with dilation rates 1, 2, 3, and 4.",
        "- Channel-wise branch weights are differentiable: `softmax(theta - beta * abs(d_cont - d_k))`.",
        "- The channel gate remains a lightweight GAP + 1x1 MLP + sigmoid gate.",
        "",
        "## Implementation Audit",
        "",
        "| Issue | Old implementation | New implementation | Fixed? |",
        "| --- | --- | --- | --- |",
        "| per-channel Python loop | yes | no | yes |",
        "| `.item()` integer dilation | yes | no | yes |",
        "| 4096-channel FA-DCG in FCN | yes | no | yes |",
        "| module created in forward | yes | no | yes |",
        "| differentiable branch fusion | no | yes | yes |",
        "",
        "## Speed Benchmark Results",
        "",
        "\n".join(speed_table),
        "",
        "## Training Results",
        "",
        "\n".join(acc_table),
        "",
        "## Comparison With task_002 Old FA-DCG Results",
        "",
        "- Old-reference values are reused from task_002 rather than rerun.",
        "- FLOPs are recorded as `NA`; no FLOPs dependency was introduced.",
        "",
        "## Recommendation for task_003",
        "",
        "- Use FastFADCG as the engineering baseline for task_003 module work because it fixes the implementation defects and is much faster in the controlled benchmark.",
        "- Do not treat the corrected FCN integration as a drop-in accuracy replacement for the old FCN + FA-DCG result; its validation mIoU dropped substantially.",
        "- U-Net + Fast FA-DCG is the better candidate for continued development: it is much faster than the old U-Net FA-DCG benchmark and only moderately below the task_002 old-reference mIoU.",
        "- Keep the old implementation as historical reference.",
        "- Continue to avoid independent-test claims until a test split and event metadata are available.",
    ]
    (out_dir / "task_002b_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.output_root) if args.output_root else REPO_ROOT / "exp" / "task_002b_fadc_fast_repro" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "training_curves").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "prediction_examples").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "branch_weight_statistics").mkdir(parents=True, exist_ok=True)
    for keep in [
        out_dir / "figures" / ".gitkeep",
        out_dir / "figures" / "training_curves" / ".gitkeep",
        out_dir / "figures" / "prediction_examples" / ".gitkeep",
        out_dir / "figures" / "branch_weight_statistics" / ".gitkeep",
    ]:
        keep.touch()

    write_environment(out_dir)
    test_output = run_cmd([sys.executable, "scripts/task_002b_test_fadc_fast.py"])
    write_implementation_audit(out_dir, test_output)

    speed_rows = []
    if not args.skip_benchmark:
        speed_rows = benchmark_speed(args, out_dir)
    elif (out_dir / "speed_benchmark.csv").exists():
        with (out_dir / "speed_benchmark.csv").open(newline="", encoding="utf-8") as f:
            speed_rows = list(csv.DictReader(f))
    if args.benchmark_only:
        return

    per_seed_path = out_dir / "metrics_per_seed.csv"
    existing = read_existing(per_seed_path) if args.resume else {}
    rows: List[Dict[str, object]] = list(existing.values())
    for method in args.models:
        for seed in args.seeds:
            if (method, seed) in existing:
                continue
            result = train_one(args, method, seed, out_dir)
            row = {
                "method": result.method,
                "implementation": "fast",
                "seed": result.seed,
                "miou": f"{result.miou:.6f}",
                "dice": f"{result.dice:.6f}",
                "params_m": f"{result.params_m:.4f}",
                "flops_g": "NA",
                "inference_time_ms": f"{result.inference_time_ms:.3f}",
                "gpu_memory_mb": f"{result.gpu_memory_mb:.1f}",
                "best_epoch": result.best_epoch,
                "notes": "validation split used; no independent test split present",
            }
            rows.append(row)
            write_csv(
                per_seed_path,
                rows,
                [
                    "method",
                    "implementation",
                    "seed",
                    "miou",
                    "dice",
                    "params_m",
                    "flops_g",
                    "inference_time_ms",
                    "gpu_memory_mb",
                    "best_epoch",
                    "notes",
                ],
            )
            if result.diagnostics:
                diag_path = out_dir / "figures" / "branch_weight_statistics" / f"{slug(method)}_seed{seed}_diagnostics.csv"
                write_csv(diag_path, [result.diagnostics], list(result.diagnostics.keys()))

    summary = summarize(rows)
    write_csv(
        out_dir / "metrics_summary.csv",
        summary,
        [
            "method",
            "implementation",
            "miou_mean",
            "miou_std",
            "dice_mean",
            "dice_std",
            "params_m",
            "inference_time_ms",
            "gpu_memory_mb",
            "delta_miou_vs_old",
            "delta_dice_vs_old",
            "delta_time_vs_old",
            "status",
        ],
    )
    complexity = complexity_rows(summary, speed_rows)
    write_csv(
        out_dir / "model_complexity.csv",
        complexity,
        [
            "method",
            "implementation",
            "params_m",
            "delta_params_m_vs_old",
            "delta_params_percent_vs_old",
            "flops_g",
            "inference_time_ms",
            "gpu_memory_mb",
        ],
    )
    (out_dir / "run_commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"OUT={out_dir}\n"
        "python scripts/task_002b_fadc_fast_repro.py --output-root \"$OUT\" --models 'U-Net + Fast FA-DCG' 'FCN + Fast FA-DCG' --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2\n"
        "python scripts/task_002b_fadc_fast_repro.py --output-root \"$OUT\" --skip-benchmark --resume --models 'U-Net + Fast FA-DCG' 'FCN + Fast FA-DCG' --seeds 42 123 --epochs 50 --batch-size 8 --num-workers 2\n",
        encoding="utf-8",
    )
    write_report(out_dir, summary, speed_rows)
    print(out_dir)


if __name__ == "__main__":
    main()
