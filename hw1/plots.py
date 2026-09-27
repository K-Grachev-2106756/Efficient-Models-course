"""Plots that put the fitted model next to the measurements.

Run after ``measure.py`` and ``calibrate.py``::

    python plots.py

Reads ``results/measurements.csv``, ``results/theta.json`` and, when present,
``results/oom.csv``. Writes ``results/figures/``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from equations import bytes_moved, energy, flops, latency, memory

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }
)


def load_measurements(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            oom = raw["memory_bytes"] == "OOM"
            row = {
                "S": int(raw["S"]),
                "B": int(raw["B"]),
                "is_validation": int(raw["is_validation"]),
                "oom": oom,
            }
            if not oom:
                row["latency_s"] = float(raw["latency_s"])
                row["memory_bytes"] = float(raw["memory_bytes"])
                row["energy_j"] = float(raw["energy_j"])
            rows.append(row)
    return rows


def load_oom(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            measured = raw["memory_bytes"]
            rows.append(
                {
                    "S": int(raw["S"]),
                    "B": int(raw["B"]),
                    "oom": measured == "OOM",
                    "memory_bytes": None if measured == "OOM" else float(measured),
                    "predicted_bytes": float(raw["predicted_bytes"]),
                    "limit_bytes": float(raw["limit_bytes"]),
                    "is_validation": int(raw["is_validation"]),
                }
            )
    return rows


def theta_vectors(payload: dict):
    latency_theta = (
        payload["latency"]["t0_s"],
        payload["latency"]["t_flop_s"],
        payload["latency"]["t_byte_s"],
    )
    energy_theta = (
        payload["energy"]["e0_j"],
        payload["energy"]["e_flop_j"],
        payload["energy"]["e_byte_j"],
    )
    return latency_theta, energy_theta


def panels(path: Path, rows: list[dict], field: str, predict, ylabel: str, title: str) -> None:
    """One panel per image size: measured points and the predicted curve."""
    sizes = sorted({row["S"] for row in rows})
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.6), sharex=True)
    batch_line = np.exp(np.linspace(np.log(1.0), np.log(256.0), 160))
    for ax, image_size in zip(axes.ravel(), sizes):
        chosen = [row for row in rows if row["S"] == image_size and not row["oom"]]
        predicted = np.asarray(predict(np.full_like(batch_line, image_size), batch_line), dtype=np.float64)
        ax.plot(batch_line, predicted, color="black", lw=1.3, zorder=2)
        train = [row for row in chosen if not row["is_validation"]]
        held = [row for row in chosen if row["is_validation"]]
        if train:
            ax.scatter(
                [row["B"] for row in train],
                [row[field] for row in train],
                s=22,
                c="#2563eb",
                marker="o",
                zorder=3,
            )
        if held:
            ax.scatter(
                [row["B"] for row in held],
                [row[field] for row in held],
                s=28,
                facecolors="none",
                edgecolors="#d97706",
                linewidths=1.3,
                marker="o",
                zorder=4,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"S = {image_size}")
    for ax in axes.ravel()[len(sizes) :]:
        ax.axis("off")
    for ax in axes[-1, :3]:
        ax.set_xlabel("Batch size B")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.3, label="predicted"),
            Line2D([0], [0], marker="o", color="#2563eb", lw=0, label="measured, train"),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="#d97706",
                lw=0,
                label="measured, validation",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.08),
    )
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _grid_values(rows: list[dict], value_of):
    sizes = np.array(sorted({row["S"] for row in rows}))
    batches = np.array(sorted({row["B"] for row in rows}))
    grid = np.full((sizes.size, batches.size), np.nan)
    for row in rows:
        i = int(np.where(sizes == row["S"])[0][0])
        j = int(np.where(batches == row["B"])[0][0])
        grid[i, j] = value_of(row)
    return sizes, batches, grid


def _draw_heatmap(ax, sizes, batches, grid, title: str, cbar_label: str, vmin: float, vcenter: float, vmax: float):
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    image = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(cbar_label)
    ax.set_xticks(np.arange(batches.size), [str(int(b)) for b in batches], rotation=45, ha="right")
    ax.set_yticks(np.arange(sizes.size), [str(int(s)) for s in sizes])
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Image size S")
    ax.set_title(title)
    ax.grid(False)


def ratio_maps(path: Path, rows: list[dict], latency_theta, energy_theta) -> None:
    live = [row for row in rows if not row["oom"]]

    def latency_ratio(row):
        predicted = float(latency(row["S"], row["B"], latency_theta))
        return row["latency_s"] / predicted

    def energy_ratio(row):
        predicted = float(energy(row["S"], row["B"], energy_theta))
        return row["energy_j"] / predicted

    def memory_ratio(row):
        predicted = float(memory(row["S"], row["B"]))
        return row["memory_bytes"] / predicted

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    specs = (
        (latency_ratio, "Latency  measured / predicted", 0.6, 2.4),
        (energy_ratio, "Energy  measured / predicted", 0.6, 2.4),
        (memory_ratio, "Memory  measured / predicted", 0.8, 3.4),
    )
    for ax, (getter, title, vmin, vmax) in zip(axes, specs):
        sizes, batches, grid = _grid_values(live, getter)
        _draw_heatmap(ax, sizes, batches, grid, title, "measured / predicted", vmin, 1.0, vmax)
    fig.suptitle("Red: measurement above the formula. Blue: measurement below it.", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def throughput(path: Path, rows: list[dict], latency_theta) -> None:
    live = [row for row in rows if not row["oom"]]
    image_size = np.array([row["S"] for row in live], dtype=np.float64)
    batch = np.array([row["B"] for row in live], dtype=np.float64)
    train = np.array([not row["is_validation"] for row in live])
    work = np.asarray(flops(image_size, batch), dtype=np.float64)
    measured = work / np.array([row["latency_s"] for row in live]) / 1e12
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.scatter(work[train], measured[train], s=26, c="#2563eb", marker="o", label="measured, train", zorder=3)
    ax.scatter(
        work[~train],
        measured[~train],
        s=32,
        facecolors="none",
        edgecolors="#d97706",
        linewidths=1.3,
        marker="o",
        label="measured, validation",
        zorder=4,
    )
    for image, style in ((32, "--"), (224, "-."), (512, "-")):
        batch_line = np.exp(np.linspace(np.log(1.0), np.log(256.0), 200))
        flop = np.asarray(flops(image, batch_line), dtype=np.float64)
        pred = np.asarray(latency(image, batch_line, latency_theta), dtype=np.float64)
        ax.plot(flop, flop / pred / 1e12, color="black", ls=style, lw=1.3, label=f"predicted, S={image}")
    t_byte = latency_theta[2]
    if t_byte > 0:
        from equations import BYTE_S2, FLOP_S2

        asymptote = (FLOP_S2 / BYTE_S2) / t_byte / 1e12
        ax.axhline(asymptote, color="#64748b", ls=":", lw=1.1, label=f"fit asymptote {asymptote:.1f} TFLOP/s")
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs per forward")
    ax.set_ylabel("Achieved throughput (TFLOP/s)")
    ax.set_title("Launch-bound at small work, then a throughput plateau")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def oom_map(path: Path, rows: list[dict]) -> None:
    limit = rows[0]["limit_bytes"]
    sizes, batches, measured_oom = _grid_values(rows, lambda row: 1.0 if row["oom"] else 0.0)
    _, _, predicted_oom = _grid_values(
        rows, lambda row: 1.0 if row["predicted_bytes"] > limit else 0.0
    )
    # 0 fit/fit, 1 formula late (measured OOM, predicted fit), 2 both OOM, 3 formula early
    code = np.zeros_like(measured_oom, dtype=int)
    code[(measured_oom == 1) & (predicted_oom == 0)] = 1
    code[(measured_oom == 1) & (predicted_oom == 1)] = 2
    code[(measured_oom == 0) & (predicted_oom == 1)] = 3
    colors = ["#dbeafe", "#f59e0b", "#b91c1c", "#7c3aed"]
    labels = [
        "both fit",
        "measured OOM, formula fits",
        "both OOM",
        "measured fits, formula OOM",
    ]
    from matplotlib.colors import ListedColormap

    present = [index for index in range(4) if np.any(code == index)]
    cmap = ListedColormap([colors[index] for index in present])
    remap = {old: new for new, old in enumerate(present)}
    shown = np.vectorize(lambda value: remap[int(value)])(code)
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.imshow(shown, aspect="auto", origin="lower", cmap=cmap, vmin=-0.5, vmax=len(present) - 0.5)
    ax.set_xticks(np.arange(batches.size), [str(int(b)) for b in batches], rotation=45, ha="right")
    ax.set_yticks(np.arange(sizes.size), [str(int(s)) for s in sizes])
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Image size S")
    ax.grid(False)
    limit_gib = limit / 1024**3
    counts = {index: int(np.sum(code == index)) for index in present}
    ax.set_title(f"OOM under a {limit_gib:.0f} GiB allocator cap")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[index], label=f"{labels[index]} ({counts[index]})")
        for index in present
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def error_table(rows: list[dict], latency_theta, energy_theta) -> None:
    """Print MAPE, median APE and max APE for the README."""

    def stats(measured, predicted, validation):
        mask = validation
        relative = np.abs(predicted - measured) / measured
        signed = (measured - predicted) / predicted
        return {
            "mape": float(relative[mask].mean()) if mask.any() else None,
            "median": float(np.median(relative[mask])) if mask.any() else None,
            "max": float(relative[mask].max()) if mask.any() else None,
            "signed_median": float(np.median(signed[mask])) if mask.any() else None,
        }

    live = [row for row in rows if not row["oom"]]
    image_size = np.array([row["S"] for row in live], dtype=np.float64)
    batch = np.array([row["B"] for row in live], dtype=np.float64)
    val = np.array([bool(row["is_validation"]) for row in live])
    lat = np.array([row["latency_s"] for row in live])
    eng = np.array([row["energy_j"] for row in live])
    mem = np.array([row["memory_bytes"] for row in live])
    pred_l = np.asarray(latency(image_size, batch, latency_theta), dtype=np.float64)
    pred_e = np.asarray(energy(image_size, batch, energy_theta), dtype=np.float64)
    pred_m = np.asarray(memory(image_size, batch), dtype=np.float64)
    report = {
        "latency": {"train": stats(lat, pred_l, ~val), "validation": stats(lat, pred_l, val)},
        "energy": {"train": stats(eng, pred_e, ~val), "validation": stats(eng, pred_e, val)},
        "memory": {"train": stats(mem, pred_m, ~val), "validation": stats(mem, pred_m, val)},
    }
    print(json.dumps(report, indent=2))


def main() -> None:
    measurements = RESULTS / "measurements.csv"
    theta_path = RESULTS / "theta.json"
    if not measurements.is_file() or not theta_path.is_file():
        raise SystemExit("run measure.py and calibrate.py before plots.py")
    rows = load_measurements(measurements)
    payload = json.loads(theta_path.read_text())
    latency_theta, energy_theta = theta_vectors(payload)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panels(
        FIGURES / "latency_panels.png",
        rows,
        "latency_s",
        lambda image_size, batch: latency(image_size, batch, latency_theta),
        "Latency (s)",
        "Latency versus batch, one panel per image size",
    )
    panels(
        FIGURES / "energy_panels.png",
        rows,
        "energy_j",
        lambda image_size, batch: energy(image_size, batch, energy_theta),
        "Energy (J)",
        "Energy versus batch, one panel per image size",
    )
    panels(
        FIGURES / "memory_panels.png",
        rows,
        "memory_bytes",
        memory,
        "Peak memory (bytes)",
        "Peak memory versus batch, one panel per image size",
    )
    ratio_maps(FIGURES / "ratio_maps.png", rows, latency_theta, energy_theta)
    throughput(FIGURES / "throughput.png", rows, latency_theta)
    oom_path = RESULTS / "oom.csv"
    if oom_path.is_file():
        oom_map(FIGURES / "oom.png", load_oom(oom_path))
    else:
        print("no results/oom.csv; skipped oom.png (run measure.py --oom-limit-gib 2)")
    error_table(rows, latency_theta, energy_theta)
    print(f"wrote {FIGURES}/")


if __name__ == "__main__":
    main()
