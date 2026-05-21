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

from models.improved.fadcg_v2a import FADCGV2a  # noqa: E402
from models.improved.fcn_fadc_optimized import FCNWithOptimizedFADCG  # noqa: E402
from models.improved.fcn_fadcg_v2a import FCNWithFADCGV2a  # noqa: E402
from models.improved.unet_fadc_optimized import UNetWithOptimizedFADCG  # noqa: E402
from models.improved.unet_fadcg_v2a import UNetWithFADCGV2a  # noqa: E402


V11_REFERENCE = {
    "FCN + FA-DCG V2a": {
        "baseline_name": "FCN + FA-DCG V1.1",
        "miou": 0.842804,
        "dice": 0.905853,
        "time_ms": 8.317,
        "params_m": 144.9324,
    },
    "U-Net + FA-DCG V2a": {
        "baseline_name": "U-Net + FA-DCG V1.1",
        "miou": 0.892461,
        "dice": 0.938098,
        "time_ms": 20.350,
        "params_m": 31.5782,
    },
}

MODEL_FACTORIES = {
    "FCN + FA-DCG V2a": FCNWithFADCGV2a,
    "U-Net + FA-DCG V2a": UNetWithFADCGV2a,
}

BENCHMARK_FACTORIES = {
    "FCN + FA-DCG V1.1": ("v1.1", FCNWithOptimizedFADCG),
    "FCN + FA-DCG V2a": ("v2a", FCNWithFADCGV2a),
    "U-Net + FA-DCG V1.1": ("v1.1", UNetWithOptimizedFADCG),
    "U-Net + FA-DCG V2a": ("v2a", UNetWithFADCGV2a),
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
    modules = [module for module in model.modules() if isinstance(module, FADCGV2a)]
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
    rows = {}
    for idx, module in enumerate(modules):
        prefix = f"module{idx}_"
        for key, value in module.last_diagnostics.items():
            rows[prefix + key] = tensor_to_value(value)
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


def benchmark_speed(args, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for method, (implementation, factory) in BENCHMARK_FACTORIES.items():
        set_seed(2024)
        model = factory(in_channels=1, num_classes=1).to(device).eval()
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
    return method.lower().replace(" + ", "_").replace("-", "").replace(" ", "_").replace(".", "_")


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
        ref = V11_REFERENCE[method]
        delta_miou = float(mious.mean() - ref["miou"])
        delta_dice = float(dices.mean() - ref["dice"])
        delta_time = float(time_ms - ref["time_ms"])
        if delta_miou > 0:
            status = "v2a_better"
        elif abs(delta_miou) <= 0.005:
            status = "v2a_comparable"
        elif delta_time > ref["time_ms"] * 1.5:
            status = "v2a_too_slow"
        else:
            status = "v2a_underperforming"
        summary.append(
            {
                "method": method,
                "implementation": "v2a",
                "miou_mean": f"{mious.mean():.6f}",
                "miou_std": f"{mious.std(ddof=1) if len(mious) > 1 else 0.0:.6f}",
                "dice_mean": f"{dices.mean():.6f}",
                "dice_std": f"{dices.std(ddof=1) if len(dices) > 1 else 0.0:.6f}",
                "params_m": f"{np.mean([float(x['params_m']) for x in items]):.4f}",
                "inference_time_ms": f"{time_ms:.3f}",
                "gpu_memory_mb": f"{np.mean([float(x['gpu_memory_mb']) for x in items]):.1f}",
                "delta_miou_vs_v1_1": f"{delta_miou:.6f}",
                "delta_dice_vs_v1_1": f"{delta_dice:.6f}",
                "delta_time_vs_v1_1": f"{delta_time:.3f}",
                "status": status,
            }
        )
    return summary


def complexity_rows(summary, speed_rows):
    speed_by_method = {row["method"]: row for row in speed_rows}
    rows = []
    for row in summary:
        method = row["method"]
        ref = V11_REFERENCE[method]
        params = float(row["params_m"])
        rows.append(
            {
                "method": method,
                "implementation": "v2a",
                "params_m": f"{params:.4f}",
                "delta_params_m_vs_v1_1": f"{params - ref['params_m']:.4f}",
                "delta_params_percent_vs_v1_1": f"{100.0 * (params - ref['params_m']) / ref['params_m']:.2f}",
                "flops_g": "NA",
                "benchmark_inference_time_ms": speed_by_method.get(method, {}).get("inference_time_ms", row["inference_time_ms"]),
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


def write_audit(out_dir: Path, test_output: str):
    text = [
        "# Implementation Audit",
        "",
        "## FA-DCG V2a",
        "",
        "- Based on FA-DCG V1.1 residual depthwise enhancement.",
        "- Adds SARFrequencyGate with local residual and multi-scale consistency descriptors.",
        "- Frequency gate is initialized to 1.0 to preserve V1.1-like behavior at startup.",
        "- FCN integration keeps the V1.1-compatible deep path.",
        "- U-Net integration keeps the V1.1-compatible bottleneck placement.",
        "",
        "## Unit Test Output",
        "",
        "```text",
        test_output.strip(),
        "```",
    ]
    (out_dir / "implementation_audit.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_report(out_dir: Path, summary, speed_rows, args):
    speed_lines = [
        "| Method | Implementation | Full-batch latency ms | Params M |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in speed_rows:
        speed_lines.append(f"| {row['method']} | {row['implementation']} | {row['inference_time_ms']} | {row['params_m']} |")

    result_lines = [
        "| Method | V1.1 mIoU | V2a mIoU | Delta | V1.1 Dice | V2a Dice | Delta | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        ref = V11_REFERENCE[row["method"]]
        result_lines.append(
            f"| {row['method']} | {ref['miou']:.6f} | {row['miou_mean']} | {row['delta_miou_vs_v1_1']} | "
            f"{ref['dice']:.6f} | {row['dice_mean']} | {row['delta_dice_vs_v1_1']} | {row['status']} |"
        )

    report = [
        "# task_004a FA-DCG V2a Report",
        "",
        "## Implementation Summary",
        "",
        "- Candidate A: SAR Frequency-Gated V1.1.",
        "- V2a keeps V1.1's depthwise residual enhancement and channel gate.",
        "- V2a adds a lightweight SAR frequency gate based on local high-frequency residual and multi-scale consistency.",
        "- The frequency gate is initialized as identity, so the model starts close to V1.1.",
        "",
        "## Relation to V1.1 and FADC",
        "",
        "- Relation to V1.1: minimal conservative extension.",
        "- FADC idea borrowed: frequency-aware modulation.",
        "- SAR adaptation: local residual and multi-scale consistency descriptors target speckle/boundary cues.",
        "- Lightweight design: grouped descriptor head with about 18C extra weights per V2a block.",
        "",
        "## Speed Benchmark",
        "",
        "\n".join(speed_lines),
        "",
        "## Seed 2024 Results",
        "",
        "\n".join(result_lines),
        "",
        "## Recommendation for task_004b",
        "",
        "- If V2a improves or matches V1.1, use it as the stable base for frequency-biased dilation in 004b.",
        "- If V2a underperforms, inspect diagnostics before adding dilation dynamics.",
        "",
        "## Run Settings",
        "",
        f"- Seeds: `{', '.join(map(str, args.seeds))}`.",
        f"- Epochs: `{args.epochs}`.",
        f"- Batch size: `{args.batch_size}`.",
        "- Evaluation uses the local validation split.",
    ]
    (out_dir / "task_004a_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.output_root is None:
        args.output_root = str(REPO_ROOT / "exp" / "task_004a_fadcg_v2a_repro" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = Path(args.output_root)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "fadcg_v2_diagnostics").mkdir(parents=True, exist_ok=True)
    write_environment(out_dir)
    test_output = run_cmd([sys.executable, "scripts/task_004a_test_fadcg_v2a.py"])
    write_audit(out_dir, test_output)
    (out_dir / "run_commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"python scripts/task_004a_fadcg_v2a_repro.py --output-root {out_dir} "
        f"--models {' '.join(repr(m) for m in args.models)} --seeds {' '.join(map(str, args.seeds))} "
        f"--epochs {args.epochs} --batch-size {args.batch_size} --num-workers {args.num_workers}\n",
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
            for seed in args.seeds:
                if (method, seed) in existing:
                    continue
                result = train_one(args, method, seed, out_dir)
                row = {
                    "method": result.method,
                    "implementation": "v2a",
                    "seed": result.seed,
                    "miou": f"{result.miou:.6f}",
                    "dice": f"{result.dice:.6f}",
                    "params_m": f"{result.params_m:.4f}",
                    "flops_g": "NA",
                    "inference_time_ms": f"{result.inference_time_ms:.3f}",
                    "gpu_memory_mb": f"{result.gpu_memory_mb:.1f}",
                    "best_epoch": result.best_epoch,
                    "notes": "validation split used; SAR Frequency-Gated V1.1",
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
                    diag_path = out_dir / "figures" / "fadcg_v2_diagnostics" / f"{slug(method)}_seed{seed}_diagnostics.csv"
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
            "delta_miou_vs_v1_1",
            "delta_dice_vs_v1_1",
            "delta_time_vs_v1_1",
            "status",
        ],
    )
    if not speed_rows and (out_dir / "speed_benchmark.csv").exists():
        with (out_dir / "speed_benchmark.csv").open(newline="", encoding="utf-8") as f:
            speed_rows = list(csv.DictReader(f))
    write_csv(
        out_dir / "model_complexity.csv",
        complexity_rows(summary, speed_rows),
        [
            "method",
            "implementation",
            "params_m",
            "delta_params_m_vs_v1_1",
            "delta_params_percent_vs_v1_1",
            "flops_g",
            "benchmark_inference_time_ms",
            "gpu_memory_mb",
        ],
    )
    write_report(out_dir, summary, speed_rows, args)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
