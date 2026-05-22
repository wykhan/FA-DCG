#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import platform
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_ROOT = REPO_ROOT / "SAR_FEM1"
sys.path.insert(0, str(SAR_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from models.improved.fadcg_v2d_ablation import FADCGV2dAblation, VARIANT_CONFIGS  # noqa: E402
from models.improved.fcn_fadcg_v2d_ablation import FCNWithFADCGV2dAblation  # noqa: E402
from models.improved.unet_fadcg_v2d_ablation import UNetWithFADCGV2dAblation  # noqa: E402
from task_005a_fadcg_v2d_repro import (  # noqa: E402
    binary_metrics,
    build_loaders,
    compute_scores,
    count_params,
    set_seed,
    tensor_to_value,
)


ARCHITECTURES = {
    "fcn": ("FCN", FCNWithFADCGV2dAblation),
    "unet": ("U-Net", UNetWithFADCGV2dAblation),
}

DIAGNOSTIC_FIELDS = [
    "module0_boundary_gate_mean",
    "module0_boundary_gate_std",
    "module0_speckle_gate_mean",
    "module0_speckle_gate_std",
    "module0_suppression_ratio",
    "module0_combined_gate_mean",
    "module0_combined_gate_std",
    "module0_local_high_descriptor_mean",
    "module0_local_variance_proxy_mean",
    "module0_multi_scale_consistency_mean",
    "module0_alpha_value",
    "module0_beta_value",
    "module0_channel_gate_mean",
    "module0_channel_gate_std",
    "module1_boundary_gate_mean",
    "module1_boundary_gate_std",
    "module1_speckle_gate_mean",
    "module1_speckle_gate_std",
    "module1_suppression_ratio",
    "module1_combined_gate_mean",
    "module1_combined_gate_std",
    "module1_local_high_descriptor_mean",
    "module1_local_variance_proxy_mean",
    "module1_multi_scale_consistency_mean",
    "module1_alpha_value",
    "module1_beta_value",
    "module1_channel_gate_mean",
    "module1_channel_gate_std",
]


@dataclass
class RunResult:
    architecture: str
    variant: str
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
    parser.add_argument("--architectures", nargs="+", default=["fcn", "unet"], choices=list(ARCHITECTURES.keys()))
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_CONFIGS.keys()), choices=list(VARIANT_CONFIGS.keys()))
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
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def slug(text: str):
    return text.lower().replace(" + ", "_").replace("-", "").replace(" ", "_").replace(".", "_")


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
        rows = list(csv.DictReader(f))
    return {(row["architecture"], row["variant"], int(row["seed"])): row for row in rows}


def make_model(architecture: str, variant: str, device: torch.device):
    _, factory = ARCHITECTURES[architecture]
    return factory(in_channels=1, num_classes=1, variant=variant).to(device)


def method_name(architecture: str, variant: str):
    label, _ = ARCHITECTURES[architecture]
    return f"{label} + V2d {variant}"


def collect_diagnostics(model: nn.Module, images: torch.Tensor):
    modules = [module for module in model.modules() if isinstance(module, FADCGV2dAblation)]
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

    rows = {field: "NA" for field in DIAGNOSTIC_FIELDS}
    for idx, module in enumerate(modules):
        prefix = f"module{idx}_"
        for key, value in module.last_diagnostics.items():
            rows[prefix + key] = tensor_to_value(value)
    return rows


def evaluate(model: nn.Module, loader, device: torch.device):
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


def train_one(args, architecture: str, variant: str, seed: int, out_dir: Path) -> RunResult:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(args, seed)
    model = make_model(architecture, variant, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.BCEWithLogitsLoss()
    params_m = count_params(model)
    best = {"miou": -1.0, "dice": 0.0, "epoch": -1, "infer_ms": 0.0, "diagnostics": {}}
    log_path = out_dir / "logs" / f"{architecture}_{variant}_seed{seed}.log"
    ckpt_dir = out_dir / "checkpoints"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"architecture={architecture}, variant={variant}, seed={seed}\n")
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
                torch.save(model.state_dict(), ckpt_dir / f"{architecture}_{variant}_seed{seed}_best.pt")
    gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    return RunResult(
        architecture=architecture,
        variant=variant,
        method=method_name(architecture, variant),
        seed=seed,
        miou=float(best["miou"]),
        dice=float(best["dice"]),
        params_m=params_m,
        inference_time_ms=float(best["infer_ms"]),
        gpu_memory_mb=float(gpu_memory_mb),
        best_epoch=int(best["epoch"]),
        diagnostics=best["diagnostics"],
    )


def status_from_delta(delta_miou: float, delta_params: float):
    if delta_miou >= 0.001:
        return "better_than_full"
    if abs(delta_miou) < 0.001:
        return "equivalent_to_full"
    if -0.003 <= delta_miou < -0.001 and delta_params < 0:
        return "slightly_worse_but_simpler"
    if delta_miou < -0.003:
        return "important_drop"
    return "slightly_worse"


def summarize(rows: List[Dict[str, object]]):
    full_ref = {}
    for row in rows:
        if row["variant"] == "full":
            full_ref[str(row["architecture"])] = row

    by_key: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        by_key.setdefault((str(row["architecture"]), str(row["variant"])), []).append(row)

    summary = []
    for architecture in ARCHITECTURES:
        for variant in VARIANT_CONFIGS:
            items = by_key.get((architecture, variant), [])
            if not items:
                continue
            mious = np.array([float(x["miou"]) for x in items])
            dices = np.array([float(x["dice"]) for x in items])
            params = float(np.mean([float(x["params_m"]) for x in items]))
            time_ms = float(np.mean([float(x["inference_time_ms"]) for x in items]))
            memory = float(np.mean([float(x["gpu_memory_mb"]) for x in items]))
            ref = full_ref.get(architecture)
            if ref:
                delta_miou = float(mious.mean() - float(ref["miou"]))
                delta_dice = float(dices.mean() - float(ref["dice"]))
                delta_params = float(params - float(ref["params_m"]))
            else:
                delta_miou = 0.0
                delta_dice = 0.0
                delta_params = 0.0
            summary.append(
                {
                    "architecture": architecture,
                    "variant": variant,
                    "method": method_name(architecture, variant),
                    "miou_mean": f"{mious.mean():.6f}",
                    "miou_std": f"{mious.std(ddof=1) if len(mious) > 1 else 0.0:.6f}",
                    "dice_mean": f"{dices.mean():.6f}",
                    "dice_std": f"{dices.std(ddof=1) if len(dices) > 1 else 0.0:.6f}",
                    "params_m": f"{params:.4f}",
                    "inference_time_ms": f"{time_ms:.3f}",
                    "gpu_memory_mb": f"{memory:.1f}",
                    "delta_miou_vs_full_v2d": f"{delta_miou:.6f}",
                    "delta_dice_vs_full_v2d": f"{delta_dice:.6f}",
                    "delta_params_vs_full_v2d": f"{delta_params:.4f}",
                    "status": "full_reference" if variant == "full" else status_from_delta(delta_miou, delta_params),
                }
            )
    return summary


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


def write_report(out_dir: Path, summary: List[Dict[str, object]], args):
    lines = [
        "# task_006 V2d Minimal Sufficient Ablation Report",
        "",
        "## Goal",
        "",
        "Search for the minimal sufficient FA-DCG V2d by removing gates and descriptors from the full V2d block.",
        "",
        "## Results",
        "",
        "| Architecture | Variant | Params M | mIoU | Delta mIoU | Dice | Delta Dice | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['architecture']} | {row['variant']} | {row['params_m']} | {row['miou_mean']} | "
            f"{row['delta_miou_vs_full_v2d']} | {row['dice_mean']} | {row['delta_dice_vs_full_v2d']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- `equivalent_to_full` means abs(delta mIoU) < 0.001 from a single seed.",
            "- `slightly_worse_but_simpler` means the variant loses less than 0.003 mIoU and reduces parameters.",
            "- A simplified variant should only replace full V2d after checking both FCN and U-Net behavior.",
            "",
            "## Run Settings",
            "",
            f"- Architectures: `{', '.join(args.architectures)}`.",
            f"- Variants: `{', '.join(args.variants)}`.",
            f"- Seeds: `{', '.join(map(str, args.seeds))}`.",
            f"- Epochs: `{args.epochs}`.",
            f"- Batch size: `{args.batch_size}`.",
            "- Evaluation uses the local validation split.",
        ]
    )
    (out_dir / "task_006_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.output_root is None:
        args.output_root = str(REPO_ROOT / "exp" / "task_006_v2d_ablation_repro" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = Path(args.output_root)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "fadcg_v2d_ablation_diagnostics").mkdir(parents=True, exist_ok=True)
    write_environment(out_dir)
    test_output = run_cmd([sys.executable, "scripts/task_006_test_v2d_ablation.py"])
    (out_dir / "implementation_audit.md").write_text(
        "# Implementation Audit\n\n"
        "## Unit Test Output\n\n"
        "```text\n"
        f"{test_output.strip()}\n"
        "```\n",
        encoding="utf-8",
    )
    (out_dir / "run_commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"python scripts/task_006_v2d_ablation_repro.py --output-root {out_dir} "
        f"--architectures {' '.join(args.architectures)} --variants {' '.join(args.variants)} "
        f"--seeds {' '.join(map(str, args.seeds))} --epochs {args.epochs} "
        f"--batch-size {args.batch_size} --num-workers {args.num_workers}\n",
        encoding="utf-8",
    )

    per_seed_path = out_dir / "metrics_per_seed.csv"
    existing = read_existing(per_seed_path) if args.resume else {}
    rows: List[Dict[str, object]] = list(existing.values())
    for architecture in args.architectures:
        for variant in args.variants:
            for seed in args.seeds:
                if (architecture, variant, seed) in existing:
                    continue
                result = train_one(args, architecture, variant, seed, out_dir)
                row = {
                    "architecture": result.architecture,
                    "variant": result.variant,
                    "method": result.method,
                    "seed": result.seed,
                    "miou": f"{result.miou:.6f}",
                    "dice": f"{result.dice:.6f}",
                    "params_m": f"{result.params_m:.4f}",
                    "flops_g": "NA",
                    "inference_time_ms": f"{result.inference_time_ms:.3f}",
                    "gpu_memory_mb": f"{result.gpu_memory_mb:.1f}",
                    "best_epoch": result.best_epoch,
                    "notes": "validation split used; task_006 V2d ablation",
                }
                rows.append(row)
                write_csv(
                    per_seed_path,
                    rows,
                    [
                        "architecture",
                        "variant",
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
                diag_path = out_dir / "figures" / "fadcg_v2d_ablation_diagnostics" / f"{architecture}_{variant}_seed{seed}_diagnostics.csv"
                write_csv(diag_path, [result.diagnostics], DIAGNOSTIC_FIELDS)

    summary = summarize(rows)
    write_csv(
        out_dir / "metrics_summary.csv",
        summary,
        [
            "architecture",
            "variant",
            "method",
            "miou_mean",
            "miou_std",
            "dice_mean",
            "dice_std",
            "params_m",
            "inference_time_ms",
            "gpu_memory_mb",
            "delta_miou_vs_full_v2d",
            "delta_dice_vs_full_v2d",
            "delta_params_vs_full_v2d",
            "status",
        ],
    )
    write_report(out_dir, summary, args)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
