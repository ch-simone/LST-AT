from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from lstat.collate import pad_collate
from lstat.config import load_config, resolve_output_dir
from lstat.dataset import LstatDataset, Normalization, input_channel_count
from lstat.index import build_examples
from lstat.losses import masked_huber
from lstat.model import ResUNet
from lstat.train import _limit_examples, _resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_gpu.toml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--scatter-output", default="")
    parser.add_argument("--scatter-dir", default="")
    parser.add_argument("--max-scatter-points", type=int, default=100000)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = resolve_output_dir(config, args.config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = _resolve_device(args.device or config["training"].get("device", "auto"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", config)
    data_config = config["data"]
    split_years = {
        "train": config["split"]["train_years"],
        "val": config["split"]["val_years"],
        "test": config["split"]["test_years"],
    }[args.split]

    examples = build_examples(config["data_root"], years=split_years)
    limit_key = {
        "train": "max_train_examples",
        "val": "max_val_examples",
        "test": "max_test_examples",
    }[args.split]
    examples = _limit_examples(
        examples,
        config["training"].get(limit_key, 0),
        int(config.get("seed", 42)) + 10,
    )

    normalization = Normalization(
        lst_mean=float(data_config["lst_mean"]),
        lst_std=float(data_config["lst_std"]),
        tair_mean=float(data_config["tair_mean"]),
        tair_std=float(data_config["tair_std"]),
        apply_modis_correction=bool(data_config["apply_modis_correction"]),
    )
    dataset = LstatDataset(
        examples,
        normalization=normalization,
        include_mask_channel=bool(data_config["include_mask_channel"]),
        include_time_channels=bool(data_config["include_time_channels"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=partial(
            pad_collate,
            min_size=int(config["training"]["min_pad_size"]),
            multiple=int(config["training"]["pad_multiple"]),
        ),
    )

    in_channels = input_channel_count(
        include_mask_channel=bool(data_config["include_mask_channel"]),
        include_time_channels=bool(data_config["include_time_channels"]),
    )
    model = ResUNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=int(model_config["model"]["base_channels"]),
        depth=int(model_config["model"]["depth"]),
        dropout=float(model_config["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    metrics, scatter = evaluate_with_phase_metrics(
        model=model,
        loader=loader,
        device=device,
        loss_name=config["training"].get("loss", "huber"),
        target_std=normalization.tair_std,
        target_mean=normalization.tair_mean,
        max_scatter_points=args.max_scatter_points,
    )
    result = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_mae_c": checkpoint.get("val_mae_c"),
        "examples": len(examples),
        **metrics,
    }

    output_path = Path(args.output) if args.output else output_dir / f"{args.split}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    scatter_dir = Path(args.scatter_dir) if args.scatter_dir else output_dir
    scatter_path = (
        Path(args.scatter_output)
        if args.scatter_output
        else scatter_dir / f"{args.split}_pred_vs_actual.png"
    )
    save_scatter_plot(
        actual=scatter["actual"],
        predicted=scatter["predicted"],
        phase=scatter["phase"],
        path=scatter_path,
        title=f"{args.split} predicted vs actual AT",
    )
    day_scatter_path = scatter_dir / f"{args.split}_day_pred_vs_actual.png"
    night_scatter_path = scatter_dir / f"{args.split}_night_pred_vs_actual.png"
    save_phase_scatter_plot(
        scatter=scatter,
        phase_name="day",
        path=day_scatter_path,
        title=f"{args.split} day predicted vs actual AT",
    )
    save_phase_scatter_plot(
        scatter=scatter,
        phase_name="night",
        path=night_scatter_path,
        title=f"{args.split} night predicted vs actual AT",
    )
    result["scatter_plot"] = str(scatter_path)
    result["day_scatter_plot"] = str(day_scatter_path)
    result["night_scatter_plot"] = str(night_scatter_path)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print_metric_summary(result)
    print(json.dumps(result, indent=2))
    print(f"wrote: {output_path}")
    print(f"wrote: {scatter_path}")
    print(f"wrote: {day_scatter_path}")
    print(f"wrote: {night_scatter_path}")


@torch.no_grad()
def evaluate_with_phase_metrics(
    model,
    loader,
    device,
    loss_name: str,
    target_std: float,
    target_mean: float,
    max_scatter_points: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    totals = {
        "overall": _empty_totals(),
        "day": _empty_totals(),
        "night": _empty_totals(),
    }
    scatter_actual = []
    scatter_predicted = []
    scatter_phase = []
    max_scatter_points = max(int(max_scatter_points), 0)

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        pred = model(x)
        loss = _loss(pred, y, mask, loss_name)
        _accumulate_totals(
            totals["overall"],
            pred=pred,
            target=y,
            mask=mask,
            loss=loss,
            target_std=target_std,
        )

        for phase_name in ("day", "night"):
            indices = [i for i, phase in enumerate(batch["phase"]) if phase == phase_name]
            if not indices:
                continue
            idx = torch.tensor(indices, device=device)
            phase_pred = pred.index_select(0, idx)
            phase_y = y.index_select(0, idx)
            phase_mask = mask.index_select(0, idx)
            phase_loss = _loss(phase_pred, phase_y, phase_mask, loss_name)
            _accumulate_totals(
                totals[phase_name],
                pred=phase_pred,
                target=phase_y,
                mask=phase_mask,
                loss=phase_loss,
                target_std=target_std,
            )

        if _scatter_size(scatter_actual) < max_scatter_points:
            actual_c = (y * target_std + target_mean).detach().cpu().numpy()
            predicted_c = (pred * target_std + target_mean).detach().cpu().numpy()
            mask_np = (mask.detach().cpu().numpy() > 0)
            phase_labels = np.array(batch["phase"])
            _append_scatter_points(
                scatter_actual,
                scatter_predicted,
                scatter_phase,
                actual_c,
                predicted_c,
                mask_np,
                phase_labels,
                max_scatter_points,
            )

    metrics = {}
    for name, total in totals.items():
        prefix = "" if name == "overall" else f"{name}_"
        metrics.update(_finalize_totals(total, prefix=prefix))

    scatter = {
        "actual": _concat_or_empty(scatter_actual),
        "predicted": _concat_or_empty(scatter_predicted),
        "phase": _concat_or_empty(scatter_phase, dtype=str),
    }
    return metrics, scatter


def _empty_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "abs_c": 0.0,
        "sq_c": 0.0,
        "target_sum_c": 0.0,
        "target_sq_sum_c": 0.0,
        "pixels": 0.0,
    }


def _accumulate_totals(
    totals: dict[str, float],
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss: torch.Tensor,
    target_std: float,
) -> None:
    pixels = mask.sum().item()
    if pixels <= 0:
        return
    diff_c = (pred - target) * target_std
    target_c = target * target_std
    totals["loss"] += loss.item() * pixels
    totals["abs_c"] += (diff_c.abs() * mask).sum().item()
    totals["sq_c"] += ((diff_c**2) * mask).sum().item()
    totals["target_sum_c"] += (target_c * mask).sum().item()
    totals["target_sq_sum_c"] += ((target_c**2) * mask).sum().item()
    totals["pixels"] += pixels


def _finalize_totals(totals: dict[str, float], prefix: str) -> dict[str, float | int]:
    pixels = max(totals["pixels"], 1.0)
    target_mean_c = totals["target_sum_c"] / pixels
    target_ss_tot_c = totals["target_sq_sum_c"] - pixels * target_mean_c**2
    r2 = 1.0 - totals["sq_c"] / target_ss_tot_c if target_ss_tot_c > 0 else float("nan")
    return {
        f"{prefix}pixels": int(totals["pixels"]),
        f"{prefix}loss": totals["loss"] / pixels,
        f"{prefix}mae_c": totals["abs_c"] / pixels,
        f"{prefix}rmse_c": (totals["sq_c"] / pixels) ** 0.5,
        f"{prefix}r2": r2,
    }


def _loss(pred, target, mask, loss_name: str):
    if loss_name == "mae":
        diff = (pred - target).abs() * mask
        return diff.sum() / mask.sum().clamp_min(1.0)
    if loss_name == "huber":
        return masked_huber(pred, target, mask)
    raise ValueError(f"Unknown loss: {loss_name}")


def _scatter_size(chunks: list[np.ndarray]) -> int:
    return int(sum(chunk.size for chunk in chunks))


def _append_scatter_points(
    actual_chunks: list[np.ndarray],
    predicted_chunks: list[np.ndarray],
    phase_chunks: list[np.ndarray],
    actual_c: np.ndarray,
    predicted_c: np.ndarray,
    mask: np.ndarray,
    phase_labels: np.ndarray,
    max_points: int,
) -> None:
    remaining = max_points - _scatter_size(actual_chunks)
    if remaining <= 0:
        return
    actual_values = actual_c[mask]
    predicted_values = predicted_c[mask]
    phase_grid = np.repeat(phase_labels[:, None, None, None], mask.shape[1], axis=1)
    phase_grid = np.repeat(phase_grid, mask.shape[2], axis=2)
    phase_grid = np.repeat(phase_grid, mask.shape[3], axis=3)
    phase_values = phase_grid[mask]
    if actual_values.size > remaining:
        indices = np.linspace(0, actual_values.size - 1, remaining, dtype=np.int64)
        actual_values = actual_values[indices]
        predicted_values = predicted_values[indices]
        phase_values = phase_values[indices]
    actual_chunks.append(actual_values.astype("float32"))
    predicted_chunks.append(predicted_values.astype("float32"))
    phase_chunks.append(phase_values.astype(str))


def _concat_or_empty(chunks: list[np.ndarray], dtype="float32") -> np.ndarray:
    if not chunks:
        return np.array([], dtype=dtype)
    return np.concatenate(chunks)


def save_scatter_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    phase: np.ndarray,
    path: Path,
    title: str,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save scatter plots.") from exc
    if actual.size == 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 900
    left, top, right, bottom = 90, 70, 850, 820
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    lo = float(np.nanpercentile(np.concatenate([actual, predicted]), 0.5))
    hi = float(np.nanpercentile(np.concatenate([actual, predicted]), 99.5))
    if hi <= lo:
        hi = lo + 1.0

    draw.rectangle((left, top, right, bottom), outline=(0, 0, 0), width=2)
    for frac in np.linspace(0, 1, 6):
        x = left + frac * (right - left)
        y = bottom - frac * (bottom - top)
        value = lo + frac * (hi - lo)
        draw.line((x, bottom, x, bottom + 6), fill=(0, 0, 0))
        draw.line((left - 6, y, left, y), fill=(0, 0, 0))
        draw.text((x - 18, bottom + 10), f"{value:.0f}", fill=(0, 0, 0))
        draw.text((20, y - 7), f"{value:.0f}", fill=(0, 0, 0))

    def to_xy(a, p):
        x = left + (a - lo) / (hi - lo) * (right - left)
        y = bottom - (p - lo) / (hi - lo) * (bottom - top)
        return x, y

    x0, y0 = to_xy(lo, lo)
    x1, y1 = to_xy(hi, hi)
    draw.line((x0, y0, x1, y1), fill=(40, 40, 40), width=2)

    colors = {"day": (220, 70, 40), "night": (50, 90, 220)}
    order = np.arange(actual.size)
    if order.size > 50000:
        order = np.linspace(0, order.size - 1, 50000, dtype=np.int64)
    for idx in order:
        x, y = to_xy(float(actual[idx]), float(predicted[idx]))
        if left <= x <= right and top <= y <= bottom:
            color = colors.get(str(phase[idx]), (80, 80, 80))
            draw.point((x, y), fill=color)

    draw.text((left, 20), title, fill=(0, 0, 0))
    draw.text(((left + right) // 2 - 60, height - 45), "Actual AT (C)", fill=(0, 0, 0))
    draw.text((10, 35), "Predicted AT (C)", fill=(0, 0, 0))
    draw.rectangle((650, 35, 665, 50), fill=colors["day"])
    draw.text((670, 34), "day", fill=(0, 0, 0))
    draw.rectangle((720, 35, 735, 50), fill=colors["night"])
    draw.text((740, 34), "night", fill=(0, 0, 0))
    image.save(path)


def save_phase_scatter_plot(
    scatter: dict[str, np.ndarray],
    phase_name: str,
    path: Path,
    title: str,
) -> None:
    keep = scatter["phase"] == phase_name
    save_scatter_plot(
        actual=scatter["actual"][keep],
        predicted=scatter["predicted"][keep],
        phase=scatter["phase"][keep],
        path=path,
        title=title,
    )


def print_metric_summary(result: dict) -> None:
    print("")
    print("Metric summary")
    print("-------------")
    print(f"{'split':<10} {'MAE C':>10} {'RMSE C':>10} {'R2':>10}")
    print(
        f"{'overall':<10} "
        f"{result['mae_c']:>10.3f} "
        f"{result['rmse_c']:>10.3f} "
        f"{result['r2']:>10.4f}"
    )
    print(
        f"{'day':<10} "
        f"{result['day_mae_c']:>10.3f} "
        f"{result['day_rmse_c']:>10.3f} "
        f"{result['day_r2']:>10.4f}"
    )
    print(
        f"{'night':<10} "
        f"{result['night_mae_c']:>10.3f} "
        f"{result['night_rmse_c']:>10.3f} "
        f"{result['night_r2']:>10.4f}"
    )
    print("")


if __name__ == "__main__":
    main()
