"""Fit latency and energy parameters, then plot prediction against measurement.

Run after ``measure.py``::

    python calibrate.py

Train points are the base grid. Validation points use an image size or a batch
size from the random extra sample. OOM rows are kept for the memory comparison
and left out of the latency and energy fits.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import nnls

from equations import bytes_moved, energy, flops, latency, memory

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
    }
)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            memory_cell = raw["memory_bytes"]
            oom = memory_cell == "OOM"
            row = {
                "S": int(raw["S"]),
                "B": int(raw["B"]),
                "is_validation": int(raw["is_validation"]),
                "oom": oom,
            }
            if not oom:
                row["latency_s"] = float(raw["latency_s"])
                row["memory_bytes"] = float(memory_cell)
                row["energy_j"] = float(raw["energy_j"])
            rows.append(row)
    return rows


def fit_positive_linear(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Non-negative least squares on columns (1, FLOPs, bytes), relative weights 1/y."""
    weight = 1.0 / target
    design = features * weight[:, None]
    scale = np.maximum(np.linalg.norm(design, axis=0), 1e-30)
    coef_scaled, _residual = nnls(design / scale, target * weight)
    return coef_scaled / scale


def percentage_errors(measured: np.ndarray, predicted: np.ndarray) -> dict:
    if measured.size == 0:
        return {"n": 0, "mape": None, "median_ape": None}
    relative = np.abs(predicted - measured) / measured
    return {
        "n": int(measured.size),
        "mape": float(relative.mean()),
        "median_ape": float(np.median(relative)),
    }


def design_matrix(image_size: np.ndarray, batch: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(image_size.shape[0]),
            np.asarray(flops(image_size, batch), dtype=np.float64),
            np.asarray(bytes_moved(image_size, batch), dtype=np.float64),
        ]
    )


def split_arrays(rows: list[dict], field: str, validation: bool):
    chosen = [row for row in rows if (not row["oom"]) and bool(row["is_validation"]) == validation]
    image_size = np.array([row["S"] for row in chosen], dtype=np.float64)
    batch = np.array([row["B"] for row in chosen], dtype=np.float64)
    target = np.array([row[field] for row in chosen], dtype=np.float64)
    return image_size, batch, target


def scatter_identity(path: Path, measured, predicted, train_mask, xlabel, ylabel, title) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(
        measured[~train_mask],
        predicted[~train_mask],
        s=28,
        marker="^",
        c="#d97706",
        label="validation",
        zorder=3,
    )
    ax.scatter(
        measured[train_mask],
        predicted[train_mask],
        s=26,
        marker="o",
        c="#2563eb",
        label="train",
        zorder=3,
    )
    lo = float(min(measured.min(), predicted.min()) * 0.75)
    hi = float(max(measured.max(), predicted.max()) * 1.35)
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", label="predicted = measured", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def curves_vs_batch(path, rows, predict, ylabel, title, yscale="log") -> None:
    """Predicted curve and measured points versus batch, one colour per image size."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    sizes = np.array(sorted({row["S"] for row in rows}))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(sizes.min()), vmax=float(sizes.max()))
    batch_line = np.exp(np.linspace(np.log(1.0), np.log(256.0), 240))
    for image_size in sizes:
        predicted = np.asarray(predict(image_size, batch_line), dtype=np.float64)
        ax.plot(batch_line, predicted, color=cmap(norm(image_size)), lw=1.5, zorder=2)
    train_s, train_b, train_y, val_s, val_b, val_y = _point_groups(rows)
    ax.scatter(train_b, train_y, s=22, c=cmap(norm(train_s)), marker="o", zorder=3)
    if val_s.size:
        ax.scatter(val_b, val_y, s=28, c=cmap(norm(val_s)), marker="^", zorder=4)
    ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")
    ax.set_xlabel("Batch size B")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label("Image size S")
    ax.legend(
        handles=[
            Line2D([0], [0], color="0.25", lw=1.5, label="predicted"),
            Line2D([0], [0], marker="o", color="0.15", lw=0, label="measured, train"),
            Line2D([0], [0], marker="^", color="0.15", lw=0, label="measured, validation"),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def curves_vs_size(path, rows, predict, ylabel, title) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    batches = np.array(sorted({row["B"] for row in rows if row["B"] in (1, 4, 16, 64, 256)}))
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0.0, vmax=np.log2(256.0))
    size_line = np.linspace(32.0, 512.0, 240)
    for batch in batches:
        predicted = np.asarray(predict(size_line, batch), dtype=np.float64)
        ax.plot(
            size_line,
            predicted,
            color=cmap(norm(np.log2(batch))),
            lw=1.5,
            label=f"predicted, B={int(batch)}",
        )
        chosen = [row for row in rows if row["B"] == int(batch)]
        ax.scatter(
            [row["S"] for row in chosen],
            [row["value"] for row in chosen],
            s=26,
            color=cmap(norm(np.log2(batch))),
            marker="o",
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Image size S")
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\ncurves: predicted, markers: measured")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _point_groups(rows: list[dict]):
    train = [row for row in rows if not row["is_validation"]]
    val = [row for row in rows if row["is_validation"]]

    def pack(group):
        if not group:
            empty = np.array([], dtype=np.float64)
            return empty, empty, empty
        return (
            np.array([row["S"] for row in group], dtype=np.float64),
            np.array([row["B"] for row in group], dtype=np.float64),
            np.array([row["value"] for row in group], dtype=np.float64),
        )

    return (*pack(train), *pack(val))


def regime_figure(path, rows, theta) -> None:
    """Three image sizes: overhead, compute term, memory term, and measured latency."""
    t0, t_flop, t_byte = theta
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)
    batch_line = np.exp(np.linspace(np.log(1.0), np.log(256.0), 240))
    for ax, image_size in zip(axes, (32, 224, 512)):
        flop = np.asarray(flops(image_size, batch_line), dtype=np.float64)
        moved = np.asarray(bytes_moved(image_size, batch_line), dtype=np.float64)
        launch = np.full_like(batch_line, t0)
        compute = t_flop * flop
        memory_term = t_byte * moved
        total = launch + compute + memory_term
        ax.plot(batch_line, total, color="black", lw=2.0, label="predicted total")
        for series, color, label in (
            (launch, "#64748b", "launch term t0"),
            (compute, "#dc2626", "compute term"),
            (memory_term, "#059669", "memory term"),
        ):
            if np.any(series > 0):
                ax.plot(batch_line, np.maximum(series, 1e-12), color=color, ls="--", label=label)
        chosen = [row for row in rows if row["S"] == image_size]
        if chosen:
            ax.scatter(
                [row["B"] for row in chosen],
                [row["latency_s"] for row in chosen],
                s=28,
                c="black",
                marker="o",
                zorder=3,
                label="measured",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size B")
        ax.set_title(f"S = {image_size}")
    axes[0].set_ylabel("Latency (s)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Latency terms against measured forward time", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def memory_grid(path, rows, gpu_bytes: int) -> None:
    sizes = np.array(sorted({row["S"] for row in rows}))
    batches = np.array(sorted({row["B"] for row in rows}))
    ratio = np.full((sizes.size, batches.size), np.nan)
    oom = np.zeros_like(ratio, dtype=bool)
    for row in rows:
        i = int(np.where(sizes == row["S"])[0][0])
        j = int(np.where(batches == row["B"])[0][0])
        if row["oom"]:
            oom[i, j] = True
        else:
            predicted = float(memory(row["S"], row["B"]))
            ratio[i, j] = row["memory_bytes"] / predicted
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    image = ax.imshow(ratio, aspect="auto", cmap="magma", origin="lower")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Measured / predicted peak memory")
    for i in range(sizes.size):
        for j in range(batches.size):
            if oom[i, j]:
                ax.text(j, i, "OOM", ha="center", va="center", color="white", fontsize=7)
    ax.set_xticks(np.arange(batches.size), [str(int(b)) for b in batches], rotation=45, ha="right")
    ax.set_yticks(np.arange(sizes.size), [str(int(s)) for s in sizes])
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Image size S")
    cap_gib = gpu_bytes / 1024**3
    ax.set_title(f"Measured / predicted peak memory. GPU capacity {cap_gib:.2f} GiB, no OOM")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def power_figure(path, rows, theta_latency, theta_energy) -> None:
    image_size = np.array([row["S"] for row in rows], dtype=np.float64)
    batch = np.array([row["B"] for row in rows], dtype=np.float64)
    measured_power = np.array([row["energy_j"] / row["latency_s"] for row in rows])
    intensity = np.asarray(flops(image_size, batch), dtype=np.float64) / np.asarray(
        bytes_moved(image_size, batch), dtype=np.float64
    )
    train = np.array([not row["is_validation"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(intensity[train], measured_power[train], s=26, c="#2563eb", marker="o", label="measured, train", zorder=3)
    ax.scatter(
        intensity[~train], measured_power[~train], s=28, c="#d97706", marker="^", label="measured, validation", zorder=3
    )
    for batch, style in ((1, "--"), (256, "-")):
        size_line = np.linspace(32.0, 512.0, 160)
        pred_l = np.asarray(latency(size_line, batch, theta_latency), dtype=np.float64)
        pred_e = np.asarray(energy(size_line, batch, theta_energy), dtype=np.float64)
        pred_i = np.asarray(flops(size_line, batch), dtype=np.float64) / np.asarray(
            bytes_moved(size_line, batch), dtype=np.float64
        )
        ax.plot(pred_i, pred_e / pred_l, color="black", ls=style, lw=1.3, label=f"predicted, B={batch}", zorder=2)
    ax.set_ylim(0, 450)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Average GPU power (W)")
    ax.set_title("Whole-GPU power during one forward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _attach_value(rows: list[dict], field: str) -> list[dict]:
    attached = []
    for row in rows:
        if row["oom"]:
            continue
        item = dict(row)
        item["value"] = row[field]
        attached.append(item)
    return attached


def dominant_regimes(rows: list[dict], theta) -> dict:
    t0, t_flop, t_byte = theta
    image_size = np.array([row["S"] for row in rows], dtype=np.float64)
    batch = np.array([row["B"] for row in rows], dtype=np.float64)
    flop = np.asarray(flops(image_size, batch), dtype=np.float64)
    moved = np.asarray(bytes_moved(image_size, batch), dtype=np.float64)
    terms = np.column_stack([np.full(flop.shape, t0), t_flop * flop, t_byte * moved])
    names = ("launch", "compute", "memory")
    winner = terms.argmax(axis=1)
    predicted = np.asarray(latency(image_size, batch, theta), dtype=np.float64)
    measured = np.array([row["latency_s"] for row in rows], dtype=np.float64)
    report = {}
    for index, name in enumerate(names):
        mask = winner == index
        stats = percentage_errors(measured[mask], predicted[mask])
        report[name] = stats
    return report


def main() -> None:
    csv_path = RESULTS / "measurements.csv"
    hardware_path = RESULTS / "hardware.json"
    if not csv_path.is_file():
        raise SystemExit(f"missing {csv_path}; run measure.py first")
    rows = load_rows(csv_path)
    hardware = json.loads(hardware_path.read_text()) if hardware_path.is_file() else {}
    gpu_bytes = int(hardware.get("gpu_memory_bytes", 0))

    train_s, train_b, train_latency = split_arrays(rows, "latency_s", validation=False)
    val_s, val_b, val_latency = split_arrays(rows, "latency_s", validation=True)
    theta_latency = fit_positive_linear(design_matrix(train_s, train_b), train_latency)
    pred_train_l = np.asarray(latency(train_s, train_b, theta_latency), dtype=np.float64)
    pred_val_l = np.asarray(latency(val_s, val_b, theta_latency), dtype=np.float64)

    _train_s_e, _train_b_e, train_energy = split_arrays(rows, "energy_j", validation=False)
    _val_s_e, _val_b_e, val_energy = split_arrays(rows, "energy_j", validation=True)
    theta_energy = fit_positive_linear(design_matrix(train_s, train_b), train_energy)
    pred_train_e = np.asarray(energy(train_s, train_b, theta_energy), dtype=np.float64)
    pred_val_e = np.asarray(energy(val_s, val_b, theta_energy), dtype=np.float64)

    live = [row for row in rows if not row["oom"]]
    mem_s = np.array([row["S"] for row in live], dtype=np.float64)
    mem_b = np.array([row["B"] for row in live], dtype=np.float64)
    mem_meas = np.array([row["memory_bytes"] for row in live], dtype=np.float64)
    mem_pred = np.asarray(memory(mem_s, mem_b), dtype=np.float64)
    mem_train = np.array([not row["is_validation"] for row in live])
    mem_ratio = mem_meas / mem_pred

    oom_measured = sum(row["oom"] for row in rows)
    oom_predicted = 0
    if gpu_bytes:
        all_s = np.array([row["S"] for row in rows], dtype=np.float64)
        all_b = np.array([row["B"] for row in rows], dtype=np.float64)
        oom_predicted = int(np.sum(np.asarray(memory(all_s, all_b), dtype=np.float64) > gpu_bytes))

    latency_stats = {
        "t0_s": float(theta_latency[0]),
        "t_flop_s": float(theta_latency[1]),
        "t_byte_s": float(theta_latency[2]),
        "train": percentage_errors(train_latency, pred_train_l),
        "validation": percentage_errors(val_latency, pred_val_l),
        "regimes_on_measured_points": dominant_regimes(live, theta_latency),
    }
    energy_stats = {
        "e0_j": float(theta_energy[0]),
        "e_flop_j": float(theta_energy[1]),
        "e_byte_j": float(theta_energy[2]),
        "train": percentage_errors(train_energy, pred_train_e),
        "validation": percentage_errors(val_energy, pred_val_e),
    }
    memory_stats = {
        "train": percentage_errors(mem_meas[mem_train], mem_pred[mem_train]),
        "validation": percentage_errors(mem_meas[~mem_train], mem_pred[~mem_train]),
        "median_ratio_measured_over_predicted": float(np.median(mem_ratio)),
        "max_ratio_measured_over_predicted": float(np.max(mem_ratio)),
        "oom_measured": int(oom_measured),
        "oom_predicted_from_capacity": oom_predicted,
        "gpu_memory_bytes": gpu_bytes,
    }
    payload = {
        "fit": "non-negative least squares, weight 1/y, train = base grid only",
        "latency": latency_stats,
        "energy": energy_stats,
        "memory": memory_stats,
    }
    FIGURES.mkdir(parents=True, exist_ok=True)
    (RESULTS / "theta.json").write_text(json.dumps(payload, indent=2) + "\n")

    def lat_title(prefix):
        train_pct = 100 * latency_stats["train"]["median_ape"]
        val_pct = 100 * latency_stats["validation"]["median_ape"]
        return f"{prefix}: median |error| train {train_pct:.1f}%, validation {val_pct:.1f}%"

    def energy_title(prefix):
        train_pct = 100 * energy_stats["train"]["median_ape"]
        val_pct = 100 * energy_stats["validation"]["median_ape"]
        return f"{prefix}: median |error| train {train_pct:.1f}%, validation {val_pct:.1f}%"

    measured_l = np.concatenate([train_latency, val_latency])
    predicted_l = np.concatenate([pred_train_l, pred_val_l])
    train_mask_l = np.concatenate([np.ones(train_latency.size, dtype=bool), np.zeros(val_latency.size, dtype=bool)])
    scatter_identity(
        FIGURES / "latency_pred_vs_measured.png",
        measured_l,
        predicted_l,
        train_mask_l,
        "Measured latency (s)",
        "Predicted latency (s)",
        lat_title("Latency"),
    )
    measured_e = np.concatenate([train_energy, val_energy])
    predicted_e = np.concatenate([pred_train_e, pred_val_e])
    train_mask_e = np.concatenate([np.ones(train_energy.size, dtype=bool), np.zeros(val_energy.size, dtype=bool)])
    scatter_identity(
        FIGURES / "energy_pred_vs_measured.png",
        measured_e,
        predicted_e,
        train_mask_e,
        "Measured energy (J)",
        "Predicted energy (J)",
        energy_title("Energy"),
    )
    scatter_identity(
        FIGURES / "memory_pred_vs_measured.png",
        mem_meas,
        mem_pred,
        mem_train,
        "Measured peak memory (bytes)",
        "Predicted peak memory (bytes)",
        f"Memory: median measured/predicted = {np.median(mem_ratio):.2f}",
    )

    latency_rows = _attach_value(rows, "latency_s")
    energy_rows = _attach_value(rows, "energy_j")
    memory_rows = _attach_value(rows, "memory_bytes")
    curves_vs_batch(
        FIGURES / "latency_vs_batch.png",
        latency_rows,
        lambda image_size, batch: latency(image_size, batch, theta_latency),
        "Latency (s)",
        lat_title("Latency versus batch"),
    )
    curves_vs_batch(
        FIGURES / "energy_vs_batch.png",
        energy_rows,
        lambda image_size, batch: energy(image_size, batch, theta_energy),
        "Energy (J)",
        energy_title("Energy versus batch"),
    )
    curves_vs_batch(
        FIGURES / "memory_vs_batch.png",
        memory_rows,
        memory,
        "Peak memory (bytes)",
        "Peak allocated memory versus batch",
    )
    curves_vs_size(
        FIGURES / "latency_vs_image_size.png",
        latency_rows,
        lambda image_size, batch: latency(image_size, batch, theta_latency),
        "Latency (s)",
        lat_title("Latency versus image size"),
    )
    regime_figure(FIGURES / "latency_regimes.png", live, theta_latency)
    memory_grid(FIGURES / "memory_ratio_grid.png", rows, gpu_bytes)
    power_figure(FIGURES / "power_vs_intensity.png", live, theta_latency, theta_energy)

    print(json.dumps(payload, indent=2))
    print(f"wrote {RESULTS / 'theta.json'} and {FIGURES}/")


if __name__ == "__main__":
    main()
