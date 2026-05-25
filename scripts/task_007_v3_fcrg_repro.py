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

from models.improved.fadc_optimized import VectorizedLightFADC  # noqa: E402
from models.improved.fadcg_v2d_ablation import FADCGV2dAblation  # noqa: E402
from models.improved.fadcg_v3_fcrg import FADCGV3FCRG, FCRG_VARIANTS  # noqa: E402
from models.improved.fcn_fadc_optimized import FCNWithOptimizedFADCG  # noqa: E402
from models.improved.fcn_fadcg_v2d_ablation import FCNWithFADCGV2dAblation  # noqa: E402
from models.improved.fcn_fadcg_v3_fcrg import FCNWithFADCGV3FCRG  # noqa: E402
from models.improved.unet_fadc_optimized import UNetWithOptimizedFADCG  # noqa: E402
from models.improved.unet_fadcg_v2d_ablation import UNetWithFADCGV2dAblation  # noqa: E402
from models.improved.unet_fadcg_v3_fcrg import UNetWithFADCGV3FCRG  # noqa: E402
from task_005a_fadcg_v2d_repro import (  # noqa: E402
    binary_metrics,
    build_loaders,
    compute_scores,
    count_params,
    set_seed,
    tensor_to_value,
)


ARCHITECTURES = {
    "fcn": "FCN",
    "unet": "U-Net",
}

VARIANTS = ("v1_1", "v2d_nomsc", *FCRG_VARIANTS)

REFERENCE_VALUES = {
    ("fcn", "v1_1"): {"miou": 0.842804, "dice": 0.905853},
    ("unet", "v1_1"): {"miou": 0.892461, "dice": 0.938098},
    ("fcn", "v2d_nomsc"): {"miou": 0.848363, "dice": 0.910157},
    ("unet", "v2d_nomsc"): {"miou": 0.895189, "dice": 0.939710},
}

DIAGNOSTIC_FIELDS = [
    "architecture",
    "variant",
    "seed",
    "module_index",
    "alpha_value",
    "response_gate_mean",
    "response_gate_std",
    "lambda_value",
    "base_response_gate_mean",
    "base_response_gate_std",
    "frequency_modulation_mean",
    "frequency_modulation_std",
    "final_response_gate_mean",
    "final_response_gate_std",
    "local_high_gap_mean",
    "local_var_gap_mean",
    "v1_response_gate_mean",
    "v1_response_gate_std",
    "v2d_boundary_gate_mean",
    "v2d_speckle_gate_mean",
    "v2d_combined_gate_mean",
    "v2d_channel_gate_mean",
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
    diagnostics: List[Dict[str, object]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--data-root", default=str(SAR_ROOT / "data" / "flood_dataset"))
    parser.add_argument("--architectures", nargs="+", default=["fcn", "unet"], choices=list(ARCHITECTURES.keys()))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
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


def method_name(architecture: str, variant: str):
    label = ARCHITECTURES[architecture]
    names = {
        "v1_1": "FA-DCG V1.1",
        "v2d_nomsc": "FA-DCG V2d-noMSC",
        "v3_fcrg_a": "FA-DCG V3-FCRG-A",
        "v3_fcrg_b": "FA-DCG V3-FCRG-B",
    }
    return f"{label} + {names[variant]}"


def make_model(architecture: str, variant: str, device: torch.device):
    if architecture == "fcn":
        if variant == "v1_1":
            model = FCNWithOptimizedFADCG(in_channels=1, num_classes=1)
        elif variant == "v2d_nomsc":
            model = FCNWithFADCGV2dAblation(in_channels=1, num_classes=1, variant="no_msc")
        else:
            model = FCNWithFADCGV3FCRG(in_channels=1, num_classes=1, variant=variant)
    else:
        if variant == "v1_1":
            model = UNetWithOptimizedFADCG(in_channels=1, num_classes=1)
        elif variant == "v2d_nomsc":
            model = UNetWithFADCGV2dAblation(in_channels=1, num_classes=1, variant="no_msc")
        else:
            model = UNetWithFADCGV3FCRG(in_channels=1, num_classes=1, variant=variant)
    return model.to(device)


def collect_diagnostics(model: nn.Module, images: torch.Tensor, architecture: str, variant: str, seed: int):
    rows = []
    modules = [
        module
        for module in model.modules()
        if isinstance(module, (FADCGV3FCRG, FADCGV2dAblation, VectorizedLightFADC))
    ]
    handles = []

    def capture_input(current, inputs):
        if isinstance(current, FADCGV3FCRG):
            current.collect_input_diagnostics(inputs[0])
        elif isinstance(current, FADCGV2dAblation):
            current.collect_input_diagnostics(inputs[0])
        elif isinstance(current, VectorizedLightFADC):
            with torch.no_grad():
                padding = (current.kernel_size - 1) // 2
                z = torch.nn.functional.conv2d(inputs[0], current.weight, padding=padding, groups=current.channels)
                gate = current.gate(z)
                current.last_diagnostics = {
                    "alpha_value": current.alpha.detach().cpu(),
                    "v1_response_gate_mean": gate.detach().mean().cpu(),
                    "v1_response_gate_std": gate.detach().std(unbiased=False).cpu(),
                }
        return None

    for module in modules:
        handles.append(module.register_forward_pre_hook(capture_input))
    with torch.no_grad():
        _ = model(images)
    for handle in handles:
        handle.remove()

    for idx, module in enumerate(modules):
        row = {field: "NA" for field in DIAGNOSTIC_FIELDS}
        row.update({"architecture": architecture, "variant": variant, "seed": seed, "module_index": idx})
        diagnostics = getattr(module, "last_diagnostics", {})
        for key, value in diagnostics.items():
            if key in row:
                row[key] = tensor_to_value(value)
            elif isinstance(module, FADCGV2dAblation):
                mapped = {
                    "boundary_gate_mean": "v2d_boundary_gate_mean",
                    "speckle_gate_mean": "v2d_speckle_gate_mean",
                    "combined_gate_mean": "v2d_combined_gate_mean",
                    "channel_gate_mean": "v2d_channel_gate_mean",
                }.get(key)
                if mapped:
                    row[mapped] = tensor_to_value(value)
        rows.append(row)
    return rows


def evaluate(model: nn.Module, loader, device: torch.device, architecture: str, variant: str, seed: int):
    model.eval()
    totals = [0, 0, 0, 0]
    elapsed = 0.0
    samples = 0
    diagnostics = []
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
                diagnostics = collect_diagnostics(model, images, architecture, variant, seed)
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
    best = {"miou": -1.0, "dice": 0.0, "epoch": -1, "infer_ms": 0.0, "diagnostics": []}
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
            miou, dice, infer_ms, diagnostics = evaluate(model, val_loader, device, architecture, variant, seed)
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


def status_from_delta(delta_miou: float):
    if delta_miou >= 0.0015:
        return "better_than_nomsc"
    if abs(delta_miou) <= 0.0015:
        return "comparable_to_nomsc"
    return "worse_than_nomsc"


def summarize(rows: List[Dict[str, object]]):
    by_arch_variant: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        by_arch_variant.setdefault((str(row["architecture"]), str(row["variant"])), []).append(row)

    summary = []
    for architecture in ARCHITECTURES:
        v11_items = by_arch_variant.get((architecture, "v1_1"), [])
        nomsc_items = by_arch_variant.get((architecture, "v2d_nomsc"), [])
        v11_miou = float(v11_items[0]["miou"]) if v11_items else REFERENCE_VALUES[(architecture, "v1_1")]["miou"]
        v11_dice = float(v11_items[0]["dice"]) if v11_items else REFERENCE_VALUES[(architecture, "v1_1")]["dice"]
        nomsc_miou = float(nomsc_items[0]["miou"]) if nomsc_items else REFERENCE_VALUES[(architecture, "v2d_nomsc")]["miou"]
        nomsc_dice = float(nomsc_items[0]["dice"]) if nomsc_items else REFERENCE_VALUES[(architecture, "v2d_nomsc")]["dice"]
        for variant in VARIANTS:
            items = by_arch_variant.get((architecture, variant), [])
            if not items:
                continue
            mious = np.array([float(x["miou"]) for x in items])
            dices = np.array([float(x["dice"]) for x in items])
            params = float(np.mean([float(x["params_m"]) for x in items]))
            time_ms = float(np.mean([float(x["inference_time_ms"]) for x in items]))
            memory = float(np.mean([float(x["gpu_memory_mb"]) for x in items]))
            delta_nomsc = float(mious.mean() - nomsc_miou)
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
                    "flops_g": "NA",
                    "inference_time_ms": f"{time_ms:.3f}",
                    "gpu_memory_mb": f"{memory:.1f}",
                    "delta_miou_vs_V1.1": f"{float(mious.mean() - v11_miou):.6f}",
                    "delta_miou_vs_V2d_noMSC": f"{delta_nomsc:.6f}",
                    "delta_Dice_vs_V1.1": f"{float(dices.mean() - v11_dice):.6f}",
                    "delta_Dice_vs_V2d_noMSC": f"{float(dices.mean() - nomsc_dice):.6f}",
                    "status": "v2d_nomsc_reference" if variant == "v2d_nomsc" else status_from_delta(delta_nomsc),
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
        "# task_007 V3 FCRG Seed-2024 Report",
        "",
        "## Executive Summary",
        "",
        "This run evaluates frequency-conditioned response gates under the validation protocol. "
        "It is not a final independent-test evaluation.",
        "",
        "## Rationale",
        "",
        "task_007 follows the GPT proposal: avoid making B/S more complex and instead transform "
        "the generic residual response gate into a SAR frequency-conditioned response gate.",
        "",
        "## Evaluated Structures",
        "",
        "- `v1_1`: residual enhancement plus original channel response gate.",
        "- `v2d_nomsc`: current V2d-noMSC reference.",
        "- `v3_fcrg_a`: direct frequency-conditioned response gate.",
        "- `v3_fcrg_b`: V1.1 response gate with residual SAR-frequency modulation.",
        "",
        "## Main Results",
        "",
        "| Architecture | Variant | Params M | mIoU | Delta vs V1.1 | Delta vs V2d-noMSC | Dice | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['architecture']} | {row['variant']} | {row['params_m']} | {row['miou_mean']} | "
            f"{row['delta_miou_vs_V1.1']} | {row['delta_miou_vs_V2d_noMSC']} | {row['dice_mean']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Run Settings",
            "",
            f"- Architectures: `{', '.join(args.architectures)}`.",
            f"- Variants: `{', '.join(args.variants)}`.",
            f"- Seeds: `{', '.join(map(str, args.seeds))}`.",
            f"- Epochs: `{args.epochs}`.",
            f"- Batch size: `{args.batch_size}`.",
            "- Evaluation uses the local validation split.",
            "",
            "## Recommendation Rule",
            "",
            "Use `v3_fcrg_b` as the final FA-DCG variant only if it matches or exceeds V2d-noMSC. "
            "Use `v3_fcrg_a` only if it clearly exceeds `v3_fcrg_b` and remains stable.",
        ]
    )
    (out_dir / "task_007_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.output_root is None:
        args.output_root = str(REPO_ROOT / "exp" / "task_007_v3_fcrg_repro" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = Path(args.output_root)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "frequency_gate_diagnostics").mkdir(parents=True, exist_ok=True)
    write_environment(out_dir)
    test_output = run_cmd([sys.executable, "scripts/task_007_test_v3_fcrg.py"])
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
        f"python scripts/task_007_v3_fcrg_repro.py --output-root {out_dir} "
        f"--architectures {' '.join(args.architectures)} --variants {' '.join(args.variants)} "
        f"--seeds {' '.join(map(str, args.seeds))} --epochs {args.epochs} "
        f"--batch-size {args.batch_size} --num-workers {args.num_workers}\n",
        encoding="utf-8",
    )

    per_seed_path = out_dir / "metrics_per_seed.csv"
    existing = read_existing(per_seed_path) if args.resume else {}
    rows: List[Dict[str, object]] = list(existing.values())
    diagnostics_rows = []
    diag_path = out_dir / "frequency_gate_diagnostics.csv"
    for architecture in args.architectures:
        for variant in args.variants:
            for seed in args.seeds:
                if (architecture, variant, seed) in existing:
                    continue
                result = train_one(args, architecture, variant, seed, out_dir)
                ref_v11 = REFERENCE_VALUES[(architecture, "v1_1")]
                ref_nomsc = REFERENCE_VALUES[(architecture, "v2d_nomsc")]
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
                    "delta_mIoU_vs_V1.1": f"{result.miou - ref_v11['miou']:.6f}",
                    "delta_mIoU_vs_V2d_noMSC": f"{result.miou - ref_nomsc['miou']:.6f}",
                    "delta_Dice_vs_V1.1": f"{result.dice - ref_v11['dice']:.6f}",
                    "delta_Dice_vs_V2d_noMSC": f"{result.dice - ref_nomsc['dice']:.6f}",
                    "status": status_from_delta(result.miou - ref_nomsc["miou"]),
                    "notes": "validation split used; task_007 V3 FCRG",
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
                        "delta_mIoU_vs_V1.1",
                        "delta_mIoU_vs_V2d_noMSC",
                        "delta_Dice_vs_V1.1",
                        "delta_Dice_vs_V2d_noMSC",
                        "status",
                        "notes",
                    ],
                )
                diagnostics_rows.extend(result.diagnostics)
                write_csv(diag_path, diagnostics_rows, DIAGNOSTIC_FIELDS)
                variant_diag_path = (
                    out_dir
                    / "figures"
                    / "frequency_gate_diagnostics"
                    / f"{architecture}_{variant}_seed{seed}_diagnostics.csv"
                )
                write_csv(variant_diag_path, result.diagnostics, DIAGNOSTIC_FIELDS)

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
            "flops_g",
            "inference_time_ms",
            "gpu_memory_mb",
            "delta_miou_vs_V1.1",
            "delta_miou_vs_V2d_noMSC",
            "delta_Dice_vs_V1.1",
            "delta_Dice_vs_V2d_noMSC",
            "status",
        ],
    )
    write_csv(
        out_dir / "model_complexity.csv",
        [
            {
                "architecture": row["architecture"],
                "variant": row["variant"],
                "method": row["method"],
                "params_m": row["params_m"],
                "flops_g": row["flops_g"],
                "inference_time_ms": row["inference_time_ms"],
                "gpu_memory_mb": row["gpu_memory_mb"],
            }
            for row in summary
        ],
        ["architecture", "variant", "method", "params_m", "flops_g", "inference_time_ms", "gpu_memory_mb"],
    )
    write_report(out_dir, summary, args)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()

