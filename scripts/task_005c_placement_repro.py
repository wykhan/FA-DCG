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
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))

from models.improved.fadcg_v2d import FADCGV2d  # noqa: E402
from models.improved.placement_fadcg import FCNWithPlacementFADCG, UNetWithPlacementFADCG  # noqa: E402


ACCEPTED_REFERENCE = {
    ("fcn", "v1.1"): {"method": "FCN + FA-DCG V1.1 original", "miou": 0.842804, "dice": 0.905853},
    ("unet", "v1.1"): {"method": "U-Net + FA-DCG V1.1 original", "miou": 0.892461, "dice": 0.938098},
    ("fcn", "v2d"): {"method": "FCN + FA-DCG V2d original", "miou": 0.845975, "dice": 0.908181},
    ("unet", "v2d"): {"method": "U-Net + FA-DCG V2d original", "miou": 0.893472, "dice": 0.938617},
}

LOCAL_SEED_REFERENCE = {
    ("fcn", "v1.1"): {"method": "FCN + FA-DCG V1.1 original", "miou": 0.844849, "dice": 0.907507},
    ("unet", "v1.1"): {"method": "U-Net + FA-DCG V1.1 original", "miou": 0.896663, "dice": 0.941023},
    ("fcn", "v2d"): {"method": "FCN + FA-DCG V2d original", "miou": 0.845975, "dice": 0.908181},
    ("unet", "v2d"): {"method": "U-Net + FA-DCG V2d original", "miou": 0.893472, "dice": 0.938617},
}

MODEL_SPECS = {
    "FCN + V1.1 after conv3": ("fcn", "v1.1", "after_conv3"),
    "FCN + V1.1 after conv4": ("fcn", "v1.1", "after_conv4"),
    "FCN + V2d after conv3": ("fcn", "v2d", "after_conv3"),
    "FCN + V2d after conv4": ("fcn", "v2d", "after_conv4"),
    "U-Net + V1.1 after encoder2": ("unet", "v1.1", "after_encoder2"),
    "U-Net + V1.1 after encoder3": ("unet", "v1.1", "after_encoder3"),
    "U-Net + V2d after encoder2": ("unet", "v2d", "after_encoder2"),
    "U-Net + V2d after encoder3": ("unet", "v2d", "after_encoder3"),
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

    def __len__(self):
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
    architecture: str
    block: str
    placement: str
    seed: int
    miou: float
    dice: float
    params_m: float
    inference_time_ms: float
    gpu_memory_mb: float
    best_epoch: int
    diagnostics: Dict[str, object]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--data-root", default=str(SAR_ROOT / "data" / "flood_dataset"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--alpha-init", type=float, default=0.25)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
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


def slug(method: str):
    return method.lower().replace(" + ", "_").replace("-", "").replace(" ", "_").replace(".", "_")


def tensor_to_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return ";".join(f"{float(v):.6f}" for v in value.flatten())
    if isinstance(value, float):
        return value
    return str(value)


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


def make_model(method: str, args, device: torch.device):
    architecture, block, placement = MODEL_SPECS[method]
    if architecture == "fcn":
        model = FCNWithPlacementFADCG(
            in_channels=1,
            num_classes=1,
            block_type=block,
            placement=placement,
            alpha_init=args.alpha_init,
        )
    else:
        model = UNetWithPlacementFADCG(
            in_channels=1,
            num_classes=1,
            block_type=block,
            placement=placement,
            alpha_init=args.alpha_init,
        )
    return model.to(device)


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


def collect_diagnostics(model: nn.Module, images: torch.Tensor):
    rows = {}
    v2d_modules = [module for module in model.modules() if isinstance(module, FADCGV2d)]
    handles = []

    def capture_input(current, inputs):
        current.collect_input_diagnostics(inputs[0])
        return None

    for module in v2d_modules:
        handles.append(module.register_forward_pre_hook(capture_input))
    with torch.no_grad():
        _ = model(images)
    for handle in handles:
        handle.remove()
    for idx, module in enumerate(v2d_modules):
        prefix = f"module{idx}_"
        for key, value in module.last_diagnostics.items():
            rows[prefix + key] = tensor_to_value(value)
    if hasattr(model, "placement_fadcg") and hasattr(model.placement_fadcg, "alpha"):
        rows["placement_alpha_value"] = tensor_to_value(model.placement_fadcg.alpha.detach().cpu())
    rows["placement"] = getattr(model, "placement", "unknown")
    rows["block_type"] = getattr(model, "block_type", "unknown")
    return rows


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


def train_one(args, method: str, seed: int, out_dir: Path) -> RunResult:
    set_seed(seed)
    architecture, block, placement = MODEL_SPECS[method]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(args, seed)
    model = make_model(method, args, device)
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
        log.write(f"method={method}, seed={seed}, alpha_init={args.alpha_init}\n")
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
        architecture=architecture,
        block=block,
        placement=placement,
        seed=seed,
        miou=float(best["miou"]),
        dice=float(best["dice"]),
        params_m=params_m,
        inference_time_ms=float(best["infer_ms"]),
        gpu_memory_mb=float(gpu_memory_mb),
        best_epoch=int(best["epoch"]),
        diagnostics=best["diagnostics"],
    )


def benchmark_speed(args, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for method in args.models:
        set_seed(2024)
        model = make_model(method, args, device).eval()
        architecture, block, placement = MODEL_SPECS[method]
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
                "architecture": architecture,
                "block": block,
                "placement": placement,
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
        [
            "method",
            "architecture",
            "block",
            "placement",
            "batch_size",
            "img_size",
            "params_m",
            "inference_time_ms",
            "gpu_memory_mb",
            "notes",
        ],
    )
    return rows


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
    for method in MODEL_SPECS:
        items = by_method.get(method, [])
        if not items:
            continue
        architecture, block, placement = MODEL_SPECS[method]
        mious = np.array([float(x["miou"]) for x in items])
        dices = np.array([float(x["dice"]) for x in items])
        time_ms = float(np.mean([float(x["inference_time_ms"]) for x in items]))
        accepted = ACCEPTED_REFERENCE[(architecture, block)]
        local_ref = LOCAL_SEED_REFERENCE[(architecture, block)]
        delta_accepted = float(mious.mean() - accepted["miou"])
        delta_local = float(mious.mean() - local_ref["miou"])
        if delta_accepted > 0:
            status = "better_than_accepted_original"
        elif abs(delta_accepted) <= 0.005:
            status = "comparable_to_accepted_original"
        else:
            status = "under_accepted_original"
        summary.append(
            {
                "method": method,
                "architecture": architecture,
                "block": block,
                "placement": placement,
                "miou_mean": f"{mious.mean():.6f}",
                "miou_std": f"{mious.std(ddof=1) if len(mious) > 1 else 0.0:.6f}",
                "dice_mean": f"{dices.mean():.6f}",
                "dice_std": f"{dices.std(ddof=1) if len(dices) > 1 else 0.0:.6f}",
                "params_m": f"{np.mean([float(x['params_m']) for x in items]):.4f}",
                "inference_time_ms": f"{time_ms:.3f}",
                "gpu_memory_mb": f"{np.mean([float(x['gpu_memory_mb']) for x in items]):.1f}",
                "accepted_original_miou": f"{accepted['miou']:.6f}",
                "delta_miou_vs_accepted_original": f"{delta_accepted:.6f}",
                "local_original_miou": f"{local_ref['miou']:.6f}",
                "delta_miou_vs_local_original": f"{delta_local:.6f}",
                "status": status,
            }
        )
    return summary


def best_rows(summary):
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in summary:
        groups.setdefault((row["architecture"], row["block"]), []).append(row)
    rows = []
    for (architecture, block), items in sorted(groups.items()):
        best = max(items, key=lambda row: float(row["miou_mean"]))
        accepted = ACCEPTED_REFERENCE[(architecture, block)]
        local_ref = LOCAL_SEED_REFERENCE[(architecture, block)]
        rows.append(
            {
                "architecture": architecture,
                "block": block,
                "best_method": best["method"],
                "best_placement": best["placement"],
                "best_miou": best["miou_mean"],
                "best_dice": best["dice_mean"],
                "accepted_original_miou": f"{accepted['miou']:.6f}",
                "delta_miou_vs_accepted_original": f"{float(best['miou_mean']) - accepted['miou']:.6f}",
                "local_original_miou": f"{local_ref['miou']:.6f}",
                "delta_miou_vs_local_original": f"{float(best['miou_mean']) - local_ref['miou']:.6f}",
                "recommendation": "prefer_early_mid" if float(best["miou_mean"]) > accepted["miou"] else "keep_original_or_retest",
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


def write_audit(out_dir: Path, test_output: str, args):
    text = [
        "# Implementation Audit",
        "",
        "## task_005c Placement Design",
        "",
        "- No new FA-DCG mechanism is introduced.",
        "- V1.1 uses `VectorizedLightFADC`; V2d uses `FADCGV2d`.",
        "- One block is inserted per model to isolate placement.",
        "- FCN placements: after conv3 and after conv4.",
        "- U-Net placements: after encoder stage 2 and after encoder stage 3.",
        f"- Weak residual initialization: `alpha_init={args.alpha_init}`.",
        "",
        "## Unit Test Output",
        "",
        "```text",
        test_output.strip(),
        "```",
    ]
    (out_dir / "implementation_audit.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_report(out_dir: Path, summary, best, speed_rows, args):
    result_lines = [
        "| Method | mIoU | Dice | Accepted Original mIoU | Delta | Local Original mIoU | Delta | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        result_lines.append(
            f"| {row['method']} | {row['miou_mean']} | {row['dice_mean']} | {row['accepted_original_miou']} | "
            f"{row['delta_miou_vs_accepted_original']} | {row['local_original_miou']} | "
            f"{row['delta_miou_vs_local_original']} | {row['status']} |"
        )

    best_lines = [
        "| Architecture | Block | Best Placement | Best mIoU | Delta vs Accepted Original | Recommendation |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in best:
        best_lines.append(
            f"| {row['architecture']} | {row['block']} | {row['best_placement']} | {row['best_miou']} | "
            f"{row['delta_miou_vs_accepted_original']} | {row['recommendation']} |"
        )

    speed_lines = [
        "| Method | Params M | Full-batch latency ms | GPU MB |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in speed_rows:
        speed_lines.append(f"| {row['method']} | {row['params_m']} | {row['inference_time_ms']} | {row['gpu_memory_mb']} |")

    report = [
        "# task_005c Placement Ablation Report",
        "",
        "## Implementation Summary",
        "",
        "- This experiment tests the best insertion position for FA-DCG V1.1 and V2d.",
        "- It keeps the block definitions fixed and changes only where one block is inserted.",
        "- FCN tests after conv3 and after conv4; U-Net tests after encoder stage 2 and stage 3.",
        "- Existing deep/bottleneck results are used as original-placement references.",
        "",
        "## Paper Story",
        "",
        "FA-DCG is framed as a plug-and-play feature extraction module. Therefore, insertion position is a core design variable: early/mid placement may enhance water-boundary and narrow-channel cues before repeated downsampling, while too-early placement may amplify SAR speckle and low-level texture noise.",
        "",
        "## Seed 2024 Results",
        "",
        "\n".join(result_lines),
        "",
        "## Best Placement Summary",
        "",
        "\n".join(best_lines),
        "",
        "## Speed Benchmark",
        "",
        "\n".join(speed_lines),
        "",
        "## Run Settings",
        "",
        f"- Seeds: `{', '.join(map(str, args.seeds))}`.",
        f"- Epochs: `{args.epochs}`.",
        f"- Batch size: `{args.batch_size}`.",
        f"- Alpha init: `{args.alpha_init}`.",
        "- Evaluation uses the local validation split.",
    ]
    (out_dir / "task_005c_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.output_root is None:
        args.output_root = str(REPO_ROOT / "exp" / "task_005c_placement_repro" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = Path(args.output_root)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "fadcg_v2_diagnostics").mkdir(parents=True, exist_ok=True)
    write_environment(out_dir)
    test_output = run_cmd([sys.executable, "SAR_FEM1/models/improved/placement_fadcg.py"])
    write_audit(out_dir, test_output, args)
    quoted_models = " ".join(repr(m) for m in args.models)
    (out_dir / "run_commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"python scripts/task_005c_placement_repro.py --output-root {out_dir} "
        f"--models {quoted_models} --seeds {' '.join(map(str, args.seeds))} "
        f"--epochs {args.epochs} --batch-size {args.batch_size} --num-workers {args.num_workers} "
        f"--alpha-init {args.alpha_init}\n",
        encoding="utf-8",
    )

    speed_rows = []
    if not args.skip_benchmark:
        speed_rows = benchmark_speed(args, out_dir)

    per_seed_path = out_dir / "metrics_per_seed.csv"
    existing = read_existing(per_seed_path) if args.resume else {}
    rows: List[Dict[str, object]] = list(existing.values())
    if not args.benchmark_only:
        for method in args.models:
            if method not in MODEL_SPECS:
                raise ValueError(f"Unknown method: {method}")
            for seed in args.seeds:
                if (method, seed) in existing:
                    continue
                result = train_one(args, method, seed, out_dir)
                row = {
                    "method": result.method,
                    "architecture": result.architecture,
                    "block": result.block,
                    "placement": result.placement,
                    "seed": result.seed,
                    "miou": f"{result.miou:.6f}",
                    "dice": f"{result.dice:.6f}",
                    "params_m": f"{result.params_m:.4f}",
                    "flops_g": "NA",
                    "inference_time_ms": f"{result.inference_time_ms:.3f}",
                    "gpu_memory_mb": f"{result.gpu_memory_mb:.1f}",
                    "best_epoch": result.best_epoch,
                    "notes": "validation split used; one early/mid FA-DCG placement block",
                }
                rows.append(row)
                write_csv(
                    per_seed_path,
                    rows,
                    [
                        "method",
                        "architecture",
                        "block",
                        "placement",
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
                    diag_path = out_dir / "figures" / "fadcg_v2_diagnostics" / f"{slug(method)}_seed{seed}_diagnostics.csv"
                    write_csv(diag_path, [result.diagnostics], list(result.diagnostics.keys()))

    summary = summarize(rows)
    best = best_rows(summary)
    write_csv(
        out_dir / "metrics_summary.csv",
        summary,
        [
            "method",
            "architecture",
            "block",
            "placement",
            "miou_mean",
            "miou_std",
            "dice_mean",
            "dice_std",
            "params_m",
            "inference_time_ms",
            "gpu_memory_mb",
            "accepted_original_miou",
            "delta_miou_vs_accepted_original",
            "local_original_miou",
            "delta_miou_vs_local_original",
            "status",
        ],
    )
    write_csv(
        out_dir / "best_placement_summary.csv",
        best,
        [
            "architecture",
            "block",
            "best_method",
            "best_placement",
            "best_miou",
            "best_dice",
            "accepted_original_miou",
            "delta_miou_vs_accepted_original",
            "local_original_miou",
            "delta_miou_vs_local_original",
            "recommendation",
        ],
    )
    if not speed_rows and (out_dir / "speed_benchmark.csv").exists():
        with (out_dir / "speed_benchmark.csv").open(newline="", encoding="utf-8") as f:
            speed_rows = list(csv.DictReader(f))
    write_report(out_dir, summary, best, speed_rows, args)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
