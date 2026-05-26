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
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.baseline.fcn import FCN  # noqa: E402
from models.baseline.unet import UNet  # noqa: E402
from models.improved.fadc_aligned import FADCAligned  # noqa: E402
from models.improved.fcn_fadcg_v2d_ablation import FCNWithFADCGV2dAblation  # noqa: E402
from models.improved.unet_fadcg_v2d_ablation import UNetWithFADCGV2dAblation  # noqa: E402


SEEDS = [2026]
TRANSPOSE = getattr(Image, "Transpose", Image)
RESAMPLING = getattr(Image, "Resampling", Image)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.images[idx]
        image = Image.open(self.img_dir / name).convert("L").resize((self.img_size, self.img_size))
        mask = Image.open(self.mask_dir / name).convert("L").resize((self.img_size, self.img_size))

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(TRANSPOSE.FLIP_LEFT_RIGHT)
                mask = mask.transpose(TRANSPOSE.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(TRANSPOSE.FLIP_TOP_BOTTOM)
                mask = mask.transpose(TRANSPOSE.FLIP_TOP_BOTTOM)
            angle = random.uniform(-10.0, 10.0)
            image = image.rotate(angle, resample=RESAMPLING.BILINEAR, fillcolor=0)
            mask = mask.rotate(angle, resample=RESAMPLING.NEAREST, fillcolor=0)

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        return torch.from_numpy(image_arr).unsqueeze(0), torch.from_numpy(mask_arr).unsqueeze(0)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class FCNWithModule(FCN):
    def __init__(self, module_factory: Callable[[int], nn.Module], in_channels: int = 1, num_classes: int = 1):
        super().__init__(in_channels, num_classes)
        self.plugin = module_factory(512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.plugin(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        x = self.score(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)


class UNetWithModule(UNet):
    def __init__(
        self,
        module_factory: Callable[[int], nn.Module],
        in_channels: int = 1,
        num_classes: int = 1,
        features: List[int] = [64, 128, 256, 512],
    ):
        super().__init__(in_channels, num_classes, features)
        self.plugin = module_factory(features[-1] * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        x = self.plugin(x)
        skip_connections = skip_connections[::-1]
        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            x = torch.cat([x, skip_connections[idx]], dim=1)
            x = decoder(x)
        return self.final_conv(x)


def fcn_se(**kwargs) -> nn.Module:
    return FCNWithModule(lambda channels: SqueezeExcitation(channels), **kwargs)


def fcn_cbam(**kwargs) -> nn.Module:
    return FCNWithModule(lambda channels: CBAM(channels), **kwargs)


def fcn_fadc_aligned(**kwargs) -> nn.Module:
    return FCNWithModule(lambda channels: FADCAligned(channels, kernel_size=3, lp_type="avgpool", spatial_group=1), **kwargs)


def unet_se(**kwargs) -> nn.Module:
    return UNetWithModule(lambda channels: SqueezeExcitation(channels), **kwargs)


def unet_cbam(**kwargs) -> nn.Module:
    return UNetWithModule(lambda channels: CBAM(channels), **kwargs)


def unet_fadc_aligned(**kwargs) -> nn.Module:
    return UNetWithModule(lambda channels: FADCAligned(channels, kernel_size=3, lp_type="avgpool", spatial_group=1), **kwargs)


MODEL_SPECS = [
    {"backbone": "FCN", "method": "FCN", "module_type": "baseline", "factory": FCN},
    {"backbone": "FCN", "method": "FCN + SE", "module_type": "SE", "factory": fcn_se},
    {"backbone": "FCN", "method": "FCN + CBAM", "module_type": "CBAM", "factory": fcn_cbam},
    {"backbone": "FCN", "method": "FCN + FADC-aligned", "module_type": "FADC-aligned", "factory": fcn_fadc_aligned},
    {
        "backbone": "FCN",
        "method": "FCN + SRFG",
        "module_type": "SRFG",
        "factory": lambda **kwargs: FCNWithFADCGV2dAblation(variant="no_msc", **kwargs),
    },
    {"backbone": "U-Net", "method": "U-Net", "module_type": "baseline", "factory": UNet},
    {"backbone": "U-Net", "method": "U-Net + SE", "module_type": "SE", "factory": unet_se},
    {"backbone": "U-Net", "method": "U-Net + CBAM", "module_type": "CBAM", "factory": unet_cbam},
    {"backbone": "U-Net", "method": "U-Net + FADC-aligned", "module_type": "FADC-aligned", "factory": unet_fadc_aligned},
    {
        "backbone": "U-Net",
        "method": "U-Net + SRFG",
        "module_type": "SRFG",
        "factory": lambda **kwargs: UNetWithFADCGV2dAblation(variant="no_msc", **kwargs),
    },
]


@dataclass
class RunResult:
    backbone: str
    method: str
    module_type: str
    seed: int
    miou: float
    dice: float
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
    parser.add_argument("--models", nargs="+", default=[spec["method"] for spec in MODEL_SPECS])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
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
    by_method = {str(spec["method"]): spec for spec in MODEL_SPECS}
    missing = [name for name in args.models if name not in by_method]
    if missing:
        raise ValueError(f"Unknown models: {missing}")
    return [by_method[name] for name in args.models]


def split_dirs(data_root: Path, split: str) -> Tuple[Path, Path]:
    image_dir = data_root / split / "image"
    mask_dir = data_root / split / "label_1D"
    if not image_dir.exists():
        image_dir = data_root / split / "images"
    if not mask_dir.exists():
        mask_dir = data_root / split / "labels"
    return image_dir, mask_dir


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def make_model(spec: Dict[str, object], device: torch.device) -> nn.Module:
    factory = spec["factory"]
    model = factory(in_channels=1, num_classes=1).to(device)  # type: ignore[misc]
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, 256, 256, device=device))
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    totals = [0, 0, 0, 0]
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            vals = binary_metrics(logits, masks)
            totals = [a + b for a, b in zip(totals, vals)]
    return compute_scores(*totals)


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
        log.write(f"method={spec['method']}, backbone={spec['backbone']}, module_type={spec['module_type']}, seed={seed}\n")
        log.write(f"protocol=256x256, batch=8, epochs={args.epochs}, Adam, lr={args.lr}, cosine_min_lr={args.min_lr}\n")
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
            miou, dice = evaluate(model, val_loader, device)
            log.write(f"epoch={epoch}, train_loss={np.mean(losses):.6f}, val_miou={miou:.6f}, val_dice={dice:.6f}\n")
            log.flush()
            if miou > best["val_miou"]:
                ckpt_path = ckpt_dir / f"{slug(str(spec['method']))}_seed{seed}_best.pt"
                torch.save(model.state_dict(), ckpt_path)
                best = {"val_miou": miou, "val_dice": dice, "epoch": epoch, "checkpoint": str(ckpt_path)}

        if best["checkpoint"]:
            state = torch.load(str(best["checkpoint"]), map_location=device)
            model.load_state_dict(state)
        test_miou, test_dice = evaluate(model, test_loader, device)
        log.write(
            f"selected_best_epoch={best['epoch']}, val_miou={best['val_miou']:.6f}, "
            f"val_dice={best['val_dice']:.6f}, test_miou={test_miou:.6f}, test_dice={test_dice:.6f}\n"
        )

    return RunResult(
        backbone=str(spec["backbone"]),
        method=str(spec["method"]),
        module_type=str(spec["module_type"]),
        seed=seed,
        miou=float(test_miou),
        dice=float(test_dice),
        val_miou=float(best["val_miou"]),
        val_dice=float(best["val_dice"]),
        test_miou=float(test_miou),
        test_dice=float(test_dice),
        best_epoch=int(best["epoch"]),
        checkpoint_path=str(best["checkpoint"]),
        status="completed",
        notes="trained under task_008 protocol; checkpoint selected by validation mIoU; miou/dice columns are test metrics",
    )


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_raw_metrics(path: Path) -> Dict[Tuple[str, int], Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.setdefault("val_miou", row.get("miou", ""))
        row.setdefault("val_dice", row.get("dice", ""))
        row.setdefault("test_miou", row.get("miou", ""))
        row.setdefault("test_dice", row.get("dice", ""))
    return {(row["method"], int(row["seed"])): row for row in rows if row.get("status") == "completed"}


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def profile_flops(model: nn.Module, device: torch.device, img_size: int) -> Tuple[int, str, str]:
    try:
        from thop import profile

        model.eval()
        dummy = torch.randn(1, 1, img_size, img_size, device=device)
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return int(macs), "MACs", "thop"
    except Exception as exc:
        return -1, "MACs", f"thop_failed: {exc}"


def benchmark_latency(
    model: nn.Module, device: torch.device, img_size: int, warmup: int, iters: int
) -> Tuple[float, float, float]:
    model.eval()
    x = torch.randn(1, 1, img_size, img_size, device=device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    mean = float(np.mean(times))
    std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
    fps = 1000.0 / mean if mean > 0 else math.nan
    return mean, std, fps


def complexity_rows(args: argparse.Namespace, specs: List[Dict[str, object]], out_dir: Path) -> List[Dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    rows = []
    baseline_by_backbone = {}
    for spec in specs:
        set_seed(0)
        model = make_model(spec, device)
        params_abs = count_params(model)
        flops_abs, flops_unit, flops_tool = profile_flops(model, device, args.img_size)
        latency_mean, latency_std, fps = benchmark_latency(
            model, device, args.img_size, args.benchmark_warmup, args.benchmark_iters
        )
        row = {
            "backbone": spec["backbone"],
            "method": spec["method"],
            "module_type": spec["module_type"],
            "params_abs": params_abs,
            "params_delta_vs_backbone": 0,
            "flops_abs": flops_abs,
            "flops_delta_vs_backbone": 0,
            "flops_unit": flops_unit,
            "flops_tool": flops_tool,
            "latency_ms_mean": f"{latency_mean:.6f}",
            "latency_ms_std": f"{latency_std:.6f}",
            "fps": f"{fps:.6f}",
            "device": device.type,
            "gpu_name": gpu_name,
            "input_shape": f"[1,1,{args.img_size},{args.img_size}]",
            "notes": "FLOPs are THOP MACs; one multiply-add counted as one MAC",
        }
        rows.append(row)
        if spec["module_type"] == "baseline":
            baseline_by_backbone[str(spec["backbone"])] = row
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for row in rows:
        baseline = baseline_by_backbone.get(str(row["backbone"]))
        if baseline is None:
            row["params_delta_vs_backbone"] = "NA"
            row["flops_delta_vs_backbone"] = "NA"
            row["notes"] = f"{row['notes']}; baseline not included in selected subset"
            continue
        row["params_delta_vs_backbone"] = int(row["params_abs"]) - int(baseline["params_abs"])
        row["flops_delta_vs_backbone"] = int(row["flops_abs"]) - int(baseline["flops_abs"])

    write_csv(
        out_dir / "complexity_raw.csv",
        rows,
        [
            "backbone",
            "method",
            "module_type",
            "params_abs",
            "params_delta_vs_backbone",
            "flops_abs",
            "flops_delta_vs_backbone",
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
            if row["method"] == spec["method"] and row["status"] == "completed" and row["miou"] not in {"", "NA"}
        ]
        if not items:
            continue
        mious = np.array([float(row["miou"]) for row in items], dtype=np.float64)
        dices = np.array([float(row["dice"]) for row in items], dtype=np.float64)
        comp = comp_by_method.get(str(spec["method"]), {})
        summary.append(
            {
                "backbone": spec["backbone"],
                "method": spec["method"],
                "module_type": spec["module_type"],
                "num_seeds": len(items),
                "miou_mean": f"{mious.mean():.6f}",
                "miou_std": f"{mious.std(ddof=1) if len(mious) > 1 else 0.0:.6f}",
                "dice_mean": f"{dices.mean():.6f}",
                "dice_std": f"{dices.std(ddof=1) if len(dices) > 1 else 0.0:.6f}",
                "params_abs": comp.get("params_abs", "NA"),
                "params_delta_vs_backbone": comp.get("params_delta_vs_backbone", "NA"),
                "flops_abs": comp.get("flops_abs", "NA"),
                "flops_delta_vs_backbone": comp.get("flops_delta_vs_backbone", "NA"),
                "latency_ms_mean": comp.get("latency_ms_mean", "NA"),
                "fps": comp.get("fps", "NA"),
                "notes": "mean/std over completed task_008 seeds",
            }
        )
    write_csv(
        out_dir / "summary_mean_std.csv",
        summary,
        [
            "backbone",
            "method",
            "module_type",
            "num_seeds",
            "miou_mean",
            "miou_std",
            "dice_mean",
            "dice_std",
            "params_abs",
            "params_delta_vs_backbone",
            "flops_abs",
            "flops_delta_vs_backbone",
            "latency_ms_mean",
            "fps",
            "notes",
        ],
    )
    return summary


def write_environment(out_dir: Path) -> None:
    with (out_dir / "environment.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"python --version\n{run_cmd(['python', '--version'])}\n\n")
        handle.write(f"pip list\n{run_cmd(['python', '-m', 'pip', 'list'])}\n\n")
        handle.write(f"nvidia-smi\n{run_cmd(['nvidia-smi'])}\n")
        handle.write(f"\nplatform={platform.platform()}\n")
        handle.write(f"torch={torch.__version__}\n")
        handle.write(f"cuda={torch.version.cuda}\n")


def write_report(
    args: argparse.Namespace,
    out_dir: Path,
    summary: List[Dict[str, object]],
    raw_rows: List[Dict[str, object]],
    comp_rows: List[Dict[str, object]],
) -> None:
    completed = sum(1 for row in raw_rows if row["status"] == "completed")
    expected = len(selected_specs(args)) * len(args.seeds)
    failures = [row for row in raw_rows if row["status"] != "completed"]
    branch = run_cmd(["git", "branch", "--show-current"])
    commit = run_cmd(["git", "rev-parse", "HEAD"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    with (out_dir / "task_008_report.md").open("w", encoding="utf-8") as handle:
        handle.write("# Task 008 Raw Report\n\n")
        handle.write(f"- git branch: `{branch}`\n")
        handle.write(f"- git commit hash: `{commit}`\n")
        handle.write(f"- dataset path: `{Path(args.data_root).resolve()}`\n")
        handle.write(f"- output directory: `{out_dir}`\n")
        handle.write(f"- PyTorch: `{torch.__version__}`; CUDA: `{torch.version.cuda}`; device: `{device}`; GPU: `{gpu_name}`\n\n")
        handle.write("## Model List\n\n")
        for spec in selected_specs(args):
            handle.write(f"- {spec['method']} ({spec['module_type']})\n")
        handle.write("\n## Training Protocol\n\n")
        handle.write(
            f"- input size: {args.img_size} x {args.img_size}\n"
            "- input channels: 1\n"
            f"- batch size: {args.batch_size}\n"
            f"- epochs: {args.epochs}\n"
            f"- optimizer: Adam, initial lr={args.lr}, cosine annealing eta_min={args.min_lr}\n"
            "- loss: BCEWithLogitsLoss\n"
            "- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees\n"
            f"- seeds: {', '.join(str(x) for x in args.seeds)}\n\n"
        )
        handle.write("## Insertion Positions\n\n")
        handle.write("- FCN SE/CBAM/FADC-aligned: inserted after `conv5`, before `fc6`.\n")
        handle.write("- FCN SRFG: internal V2d-noMSC implementation mapped to `SRFG`; replaces FCN classifier gates as implemented in `FCNWithFADCGV2dAblation(variant=\"no_msc\")`.\n")
        handle.write("- U-Net SE/CBAM/FADC-aligned/SRFG: inserted after bottleneck, before decoder.\n")
        handle.write("- FADC-aligned is the local PyTorch implementation inspired by FADC, not the official implementation.\n\n")
        handle.write("## Metric Definitions\n\n")
        handle.write("- Existing repository binary metrics: sigmoid threshold 0.5; foreground IoU and background IoU averaged as mIoU; Dice computed on foreground mask.\n")
        handle.write("- Best epoch is selected by maximum mIoU on the validation split; final mIoU/Dice in CSV summaries are measured on the test split using that checkpoint.\n\n")
        handle.write("## FLOPs And FPS Protocol\n\n")
        handle.write("- FLOPs tool: THOP. Reported values are MACs; one multiply-add is counted as one MAC.\n")
        handle.write(f"- input tensor for complexity and latency: `[1, 1, {args.img_size}, {args.img_size}]`.\n")
        handle.write(f"- latency: eval mode, no_grad, warmup={args.benchmark_warmup}, timed_iters={args.benchmark_iters}, no data loading or preprocessing time.\n\n")
        handle.write("## Completed Runs\n\n")
        handle.write(f"- completed accuracy runs: {completed}/{expected}\n")
        handle.write(f"- summary rows: {len(summary)}/{len(selected_specs(args))}\n")
        handle.write(f"- complexity rows: {len(comp_rows)}/{len(selected_specs(args))}\n\n")
        handle.write("## Failed Or Skipped Runs\n\n")
        if failures:
            for row in failures:
                handle.write(f"- {row['method']} seed {row['seed']}: {row['notes']}\n")
        else:
            handle.write("- None.\n")
        handle.write("\n## Summary\n\n")
        handle.write("| backbone | method | seeds | test mIoU mean | test Dice mean | Params | MACs | latency ms | FPS |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            handle.write(
                f"| {row['backbone']} | {row['method']} | {row['num_seeds']} | {row['miou_mean']} | {row['dice_mean']} | "
                f"{row['params_abs']} | {row['flops_abs']} | {row['latency_ms_mean']} | {row['fps']} |\n"
            )
        handle.write("\n## Key Observations\n\n")
        handle.write("- Raw data only; no paper table formatting was changed.\n")
        handle.write("- SRFG in this report is V2d-noMSC.\n")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) if args.output_root else REPO_ROOT / "exp" / "task_008_main_module_table" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    write_environment(out_dir)

    specs = selected_specs(args)
    raw_path = out_dir / "raw_metrics_per_seed.csv"
    raw_rows_by_key = read_raw_metrics(raw_path) if args.resume else {}
    raw_rows: List[Dict[str, object]] = list(raw_rows_by_key.values())

    if not args.skip_training:
        for spec in specs:
            for seed in args.seeds:
                key = (str(spec["method"]), seed)
                if args.resume and key in raw_rows_by_key:
                    continue
                try:
                    result = train_one(args, spec, seed, out_dir)
                    raw_rows_by_key[key] = result.__dict__
                except Exception as exc:
                    raw_rows_by_key[key] = {
                        "backbone": spec["backbone"],
                        "method": spec["method"],
                        "module_type": spec["module_type"],
                        "seed": seed,
                        "miou": "NA",
                        "dice": "NA",
                        "val_miou": "NA",
                        "val_dice": "NA",
                        "test_miou": "NA",
                        "test_dice": "NA",
                        "best_epoch": "NA",
                        "checkpoint_path": "NA",
                        "status": "failed",
                        "notes": repr(exc),
                    }
                raw_rows = list(raw_rows_by_key.values())
                write_csv(
                    raw_path,
                    sorted(raw_rows, key=lambda x: (str(x["backbone"]), str(x["method"]), int(x["seed"]))),
                    [
                        "backbone",
                        "method",
                        "module_type",
                        "seed",
                        "miou",
                        "dice",
                        "val_miou",
                        "val_dice",
                        "test_miou",
                        "test_dice",
                        "best_epoch",
                        "checkpoint_path",
                        "status",
                        "notes",
                    ],
                )

    raw_rows = sorted(list(raw_rows_by_key.values()), key=lambda x: (str(x["backbone"]), str(x["method"]), int(x["seed"])))
    if not raw_path.exists():
        write_csv(
            raw_path,
            raw_rows,
            [
                "backbone",
                "method",
                "module_type",
                "seed",
                "miou",
                "dice",
                "val_miou",
                "val_dice",
                "test_miou",
                "test_dice",
                "best_epoch",
                "checkpoint_path",
                "status",
                "notes",
            ],
        )

    comp_rows = [] if args.skip_complexity else complexity_rows(args, specs, out_dir)
    summary = summarize(raw_rows, comp_rows, out_dir)
    write_report(args, out_dir.resolve(), summary, raw_rows, comp_rows)

    print(f"experiment_dir={out_dir.resolve()}")
    print(f"summary_mean_std={out_dir.resolve() / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
