#!/usr/bin/env python3
"""Task 002 baseline reproduction runner.

This script is intentionally outside SAR_FEM1/ so the preserved original
project code is not rewritten. It audits the local dataset/environment and
runs a reproducible train/validation loop for the four core models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
from models.improved.fcn_fadc_light import FCNWithLightFADC  # noqa: E402
from models.improved.unet_fadc_light import UNetWithLightFADC  # noqa: E402


MANUSCRIPT = {
    "FCN": {"params_m": 134.27, "miou": 0.6319, "dice": 0.7390},
    "U-Net": {"params_m": 31.04, "miou": 0.7048, "dice": 0.7894},
    "FCN + FA-DCG": {"params_m": 142.83, "miou": 0.8243, "dice": 0.8949},
    "U-Net + FA-DCG": {"params_m": 31.58, "miou": 0.8782, "dice": 0.9289},
}

MODEL_FACTORIES = {
    "FCN": FCN,
    "U-Net": UNet,
    "FCN + FA-DCG": FCNWithLightFADC,
    "U-Net + FA-DCG": UNetWithLightFADC,
}


class FloodDatasetLocal(Dataset):
    """Minimal local image/mask dataset with deterministic paired transforms."""

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.images[idx]
        image = Image.open(self.img_dir / name).convert("L").resize((self.img_size, self.img_size))
        mask = Image.open(self.mask_dir / name).convert("L").resize((self.img_size, self.img_size))

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        image_t = torch.from_numpy(image_arr).unsqueeze(0)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

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
    flops_g: str
    inference_time_ms: float
    gpu_memory_mb: float
    best_epoch: int
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--data-root", default=str(SAR_ROOT / "data" / "flood_dataset"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
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


def git_hash() -> str:
    return run_cmd(["git", "rev-parse", "--short", "HEAD"])


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def warmup_dynamic_modules(model: nn.Module, device: torch.device, img_size: int) -> None:
    """Register any modules created lazily by legacy model forwards."""
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, img_size, img_size, device=device))


def make_model(method: str, device: torch.device, img_size: int) -> nn.Module:
    model = MODEL_FACTORIES[method](in_channels=1, num_classes=1).to(device)
    warmup_dynamic_modules(model, device, img_size)
    return model


def binary_metrics(logits: torch.Tensor, masks: torch.Tensor) -> Tuple[int, int, int, int]:
    pred = (torch.sigmoid(logits) > 0.5).bool()
    target = masks.bool()
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    tn = (~pred & ~target).sum().item()
    return int(tp), int(fp), int(fn), int(tn)


def compute_scores(tp: int, fp: int, fn: int, tn: int) -> Tuple[float, float]:
    fg_iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)
    miou = (fg_iou + bg_iou) / 2.0
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    return miou, dice


def build_loaders(args: argparse.Namespace, seed: int, augment: bool = True) -> Tuple[DataLoader, DataLoader]:
    data_root = Path(args.data_root)
    train_ds: Dataset = FloodDatasetLocal(
        data_root / "train" / "images", data_root / "train" / "labels", args.img_size, augment=augment
    )
    val_ds: Dataset = FloodDatasetLocal(
        data_root / "val" / "images", data_root / "val" / "labels", args.img_size, augment=False
    )
    if args.max_train_samples:
        train_ds = Subset(train_ds, list(range(min(args.max_train_samples, len(train_ds)))))
    if args.max_val_samples:
        val_ds = Subset(val_ds, list(range(min(args.max_val_samples, len(val_ds)))))

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    totals = [0, 0, 0, 0]
    elapsed = 0.0
    batches = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            batches += images.size(0)
            vals = binary_metrics(logits, masks)
            totals = [a + b for a, b in zip(totals, vals)]
    miou, dice = compute_scores(*totals)
    inference_ms = 1000.0 * elapsed / max(batches, 1)
    return miou, dice, inference_ms


def train_one(args: argparse.Namespace, method: str, seed: int, out_dir: Path, smoke: bool = False) -> RunResult:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(args, seed, augment=True)
    model = make_model(method, device, args.img_size)
    params_m = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.BCEWithLogitsLoss()

    best = {"miou": -1.0, "dice": 0.0, "epoch": -1}
    log_path = out_dir / "logs" / f"{slug(method)}_seed{seed}{'_smoke' if smoke else ''}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"method={method}, seed={seed}, smoke={smoke}\n")
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
            miou, dice, infer_ms = evaluate(model, val_loader, device)
            train_loss = float(np.mean(losses)) if losses else float("nan")
            log.write(
                f"epoch={epoch}, train_loss={train_loss:.6f}, val_miou={miou:.6f}, "
                f"val_dice={dice:.6f}, infer_ms={infer_ms:.3f}\n"
            )
            log.flush()
            if miou > best["miou"]:
                best = {"miou": miou, "dice": dice, "epoch": epoch, "infer_ms": infer_ms}
                torch.save(model.state_dict(), ckpt_dir / f"{slug(method)}_seed{seed}_best.pt")

    gpu_memory_mb = 0.0
    if device.type == "cuda":
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    notes = "validation split used; no independent test split present"
    if smoke:
        notes = "smoke test on reduced setting; " + notes
    return RunResult(
        method=method,
        seed=seed,
        miou=float(best["miou"]),
        dice=float(best["dice"]),
        params_m=params_m,
        flops_g="NA",
        inference_time_ms=float(best.get("infer_ms", 0.0)),
        gpu_memory_mb=float(gpu_memory_mb),
        best_epoch=int(best["epoch"]),
        notes=notes,
    )


def slug(method: str) -> str:
    return method.lower().replace(" + ", "_").replace("-", "").replace(" ", "_")


def audit_dataset(data_root: Path, out_dir: Path, img_size: int) -> Dict[str, object]:
    rows = []
    total = 0
    for split in ["train", "val", "test"]:
        img_dir = data_root / split / "images"
        label_dir = data_root / split / "labels"
        exists = img_dir.exists() and label_dir.exists()
        files = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}) if exists else []
        total += len(files)
        label_values = set()
        water_pixels = 0
        all_pixels = 0
        size = "NA"
        channels = "NA"
        missing_labels = []
        for path in files:
            label = label_dir / path.name
            if not label.exists():
                missing_labels.append(path.name)
                continue
            with Image.open(path) as im:
                size = f"{im.size[0]}x{im.size[1]}"
                channels = len(im.getbands())
            with Image.open(label).convert("L") as mask:
                arr = np.asarray(mask)
                label_values.update(int(v) for v in np.unique(arr))
                water_pixels += int((arr > 127).sum())
                all_pixels += int(arr.size)
        rows.append(
            {
                "split": split,
                "exists": exists,
                "count": len(files),
                "image_size": size,
                "channels": channels,
                "label_values": sorted(label_values),
                "binary": set(label_values).issubset({0, 255}) if label_values else False,
                "water_ratio": water_pixels / all_pixels if all_pixels else None,
                "missing_labels": missing_labels[:10],
            }
        )

    audit = {
        "data_root": str(data_root),
        "total_image_files": total,
        "splits": rows,
        "flood_event_isolated": "unknown; no event metadata or split manifest found",
        "matches_manuscript_663": total == 663,
    }

    lines = [
        "# Dataset Audit",
        "",
        f"- Data root: `{data_root}`",
        f"- Total image files across train/val/test directories: {total}",
        f"- Matches manuscript 663 patches: {total == 663}",
        "- Flood-event-isolated split: unknown; no event identifiers, metadata, or split manifest were found.",
        "",
        "| Split | Exists | Images | Image size | Channels | Label values | Binary | Water pixel ratio | Missing labels |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        ratio = "NA" if row["water_ratio"] is None else f"{row['water_ratio']:.6f}"
        lines.append(
            f"| {row['split']} | {row['exists']} | {row['count']} | {row['image_size']} | "
            f"{row['channels']} | {row['label_values']} | {row['binary']} | {ratio} | {row['missing_labels']} |"
        )
    (out_dir / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def write_environment(out_dir: Path) -> None:
    device_lines = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_lines.append(f"GPU {i}: {props.name}, memory={props.total_memory / (1024**3):.2f} GiB")
    else:
        device_lines.append("CUDA GPU: unavailable")
    env = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"cwd: {REPO_ROOT}",
        f"git_commit: {git_hash()}",
        f"os: {platform.platform()}",
        f"python: {sys.version.replace(os.linesep, ' ')}",
        f"torch: {torch.__version__}",
        f"torch_cuda: {torch.version.cuda}",
        f"cuda_available: {torch.cuda.is_available()}",
        *device_lines,
        "",
        "pip_freeze_head:",
        run_cmd([sys.executable, "-m", "pip", "freeze"]),
    ]
    (out_dir / "environment.txt").write_text("\n".join(env) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_existing_metrics(path: Path) -> Dict[Tuple[str, int], Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {(row["method"], int(row["seed"])): row for row in csv.DictReader(f)}


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    for method in MODEL_FACTORIES:
        items = by_method.get(method, [])
        if not items:
            target = MANUSCRIPT[method]
            out.append(
                {
                    "method": method,
                    "miou_mean": "",
                    "miou_std": "",
                    "dice_mean": "",
                    "dice_std": "",
                    "params_m": target["params_m"],
                    "flops_g": "NA",
                    "inference_time_ms": "",
                    "gpu_memory_mb": "",
                    "delta_miou_vs_manuscript": "",
                    "delta_dice_vs_manuscript": "",
                    "status": "not_run",
                }
            )
            continue
        mious = np.array([float(x["miou"]) for x in items], dtype=float)
        dices = np.array([float(x["dice"]) for x in items], dtype=float)
        target = MANUSCRIPT[method]
        delta_miou = float(mious.mean() - target["miou"])
        delta_dice = float(dices.mean() - target["dice"])
        abs_delta = abs(delta_miou)
        status = "reproduced" if abs_delta <= 0.03 else "partial" if abs_delta <= 0.06 else "not_reproduced"
        out.append(
            {
                "method": method,
                "miou_mean": f"{mious.mean():.6f}",
                "miou_std": f"{mious.std(ddof=1) if len(mious) > 1 else 0.0:.6f}",
                "dice_mean": f"{dices.mean():.6f}",
                "dice_std": f"{dices.std(ddof=1) if len(dices) > 1 else 0.0:.6f}",
                "params_m": f"{np.mean([float(x['params_m']) for x in items]):.4f}",
                "flops_g": "NA",
                "inference_time_ms": f"{np.mean([float(x['inference_time_ms']) for x in items]):.3f}",
                "gpu_memory_mb": f"{np.mean([float(x['gpu_memory_mb']) for x in items]):.1f}",
                "delta_miou_vs_manuscript": f"{delta_miou:.6f}",
                "delta_dice_vs_manuscript": f"{delta_dice:.6f}",
                "status": status,
            }
        )
    return out


def write_report(out_dir: Path, dataset_audit: Dict[str, object], per_seed: List[Dict[str, object]], summary: List[Dict[str, object]], args: argparse.Namespace) -> None:
    comparison = [
        "| Method | Manuscript mIoU | Reproduced mIoU | Delta mIoU | Manuscript Dice | Reproduced Dice | Delta Dice | Status |",
        "| ------ | --------------: | --------------: | ---------: | --------------: | --------------: | ---------: | ------ |",
    ]
    for row in summary:
        method = str(row["method"])
        target = MANUSCRIPT[method]
        comparison.append(
            f"| {method} | {target['miou']:.4f} | {row['miou_mean'] or 'NA'} | "
            f"{row['delta_miou_vs_manuscript'] or 'NA'} | {target['dice']:.4f} | "
            f"{row['dice_mean'] or 'NA'} | {row['delta_dice_vs_manuscript'] or 'NA'} | {row['status']} |"
        )
    ran = len(per_seed)
    report = [
        "# task_002 Baseline Reproduction Report",
        "",
        "## Executive Summary",
        "",
        f"- Completed runs recorded: {ran}.",
        "- Evaluation uses the available validation split because no independent test directory or event-isolated split manifest exists in the local dataset.",
        "- Status labels therefore describe reproduction against manuscript numbers under the current repository/data constraints, not a strict independent-test reproduction.",
        "",
        "## Repository and Code Audit",
        "",
        "- Original source code: `SAR_FEM1/`.",
        "- Model definitions: `SAR_FEM1/models/baseline/` and `SAR_FEM1/models/improved/`.",
        "- FA-DCG implementations: `SAR_FEM1/models/improved/fcn_fadc_light.py` and `SAR_FEM1/models/improved/unet_fadc_light.py`.",
        "- Dataset loader code: `SAR_FEM1/data/dataset.py` and `SAR_FEM1/data/dataset_pt.py`.",
        "- Training script in tracked source is not present; `SAR_FEM1/train_cursive_T4.py` is a plotting script for checkpoint CSVs.",
        "- Task driver used for this report: `scripts/task_002_baseline_repro.py`.",
        "",
        "## Environment Audit",
        "",
        "See `environment.txt`.",
        "",
        "## Dataset Audit",
        "",
        f"- Data root: `{dataset_audit['data_root']}`.",
        f"- Total image files found: {dataset_audit['total_image_files']}.",
        f"- Matches manuscript 663 patches: {dataset_audit['matches_manuscript_663']}.",
        f"- Flood-event-isolated split: {dataset_audit['flood_event_isolated']}.",
        "",
        "See `dataset_audit.md` for split counts, label values, and water-pixel ratios.",
        "",
        "## Smoke Test Results",
        "",
        "- Smoke tests are logged in `logs/*_smoke.log` when run.",
        "- A smoke test is considered successful when epochs complete, losses remain finite, validation metrics are computed, and a checkpoint is written under the local ignored `checkpoints/` directory.",
        "",
        "## Main Reproduction Results",
        "",
        "\n".join(comparison),
        "",
        "## Deviations from Manuscript Settings",
        "",
        "- No independent `test/` split is present locally; validation metrics are reported.",
        "- No event metadata or split manifest is present, so flood-event isolation cannot be verified.",
        "- FLOPs are reported as `NA`; parameter counts, inference time, and GPU memory are recorded.",
        "- Checkpoints are generated locally for reproducibility but are ignored by Git and not committed.",
        "",
        "## Failure Cases or Unresolved Issues",
        "",
        "- Strict manuscript reproduction remains blocked until the exact independent test split and flood-event-isolated metadata are available.",
        "- `FCNWithLightFADC` creates `up_dim` lazily inside `forward`; the task runner performs a warmup forward before optimizer creation so the parameter is trainable without editing original source.",
        "",
        "## Recommendations for task_003",
        "",
        "- Add an explicit train/val/test split manifest with event IDs.",
        "- Move the training pipeline into a versioned script with CLI arguments for seed, split, and model.",
        "- Replace lazy module creation in model forward paths with constructor-defined modules.",
        "- Add deterministic metric tests for binary mIoU and Dice.",
    ]
    (out_dir / "task_002_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        args.output_root = str(REPO_ROOT / "exp" / "task_002_baseline_repro" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = Path(args.output_root)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "training_curves").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "prediction_examples").mkdir(parents=True, exist_ok=True)

    (out_dir / "run_commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"python scripts/task_002_baseline_repro.py --output-root {out_dir} "
        f"--epochs {args.epochs} --batch-size {args.batch_size} --seeds {' '.join(map(str, args.seeds))}\n",
        encoding="utf-8",
    )
    write_environment(out_dir)
    dataset_audit = audit_dataset(Path(args.data_root), out_dir, args.img_size)

    per_seed_path = out_dir / "metrics_per_seed.csv"
    existing = read_existing_metrics(per_seed_path) if args.resume else {}
    rows: List[Dict[str, object]] = list(existing.values())

    if not args.skip_smoke:
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.epochs = 1
        smoke_args.max_train_samples = min(args.max_train_samples or 16, 16)
        smoke_args.max_val_samples = min(args.max_val_samples or 16, 16)
        smoke_args.batch_size = min(args.batch_size, 2)
        for method in ["U-Net", "U-Net + FA-DCG"]:
            train_one(smoke_args, method, 42, out_dir, smoke=True)

    if not args.smoke_only:
        for method in args.models:
            for seed in args.seeds:
                if (method, seed) in existing:
                    continue
                result = train_one(args, method, seed, out_dir, smoke=False)
                row = {
                    "method": result.method,
                    "seed": result.seed,
                    "miou": f"{result.miou:.6f}",
                    "dice": f"{result.dice:.6f}",
                    "params_m": f"{result.params_m:.4f}",
                    "flops_g": result.flops_g,
                    "inference_time_ms": f"{result.inference_time_ms:.3f}",
                    "gpu_memory_mb": f"{result.gpu_memory_mb:.1f}",
                    "best_epoch": result.best_epoch,
                    "notes": result.notes,
                }
                rows.append(row)
                write_csv(
                    per_seed_path,
                    rows,
                    [
                        "method",
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

    summary = summarize(rows)
    write_csv(
        out_dir / "metrics_summary.csv",
        summary,
        [
            "method",
            "miou_mean",
            "miou_std",
            "dice_mean",
            "dice_std",
            "params_m",
            "flops_g",
            "inference_time_ms",
            "gpu_memory_mb",
            "delta_miou_vs_manuscript",
            "delta_dice_vs_manuscript",
            "status",
        ],
    )
    write_report(out_dir, dataset_audit, rows, summary, args)
    print(out_dir)


if __name__ == "__main__":
    main()
