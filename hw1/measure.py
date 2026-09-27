"""Measure latency, peak memory, and whole-GPU energy on the homework grid.

Run from this directory after the venv in the README::

    python measure.py
    python measure.py --oom-limit-gib 2
    python measure.py --profile

The default run writes ``results/measurements.csv`` and ``results/hardware.json``.
``--oom-limit-gib`` repeats only the memory pass under a smaller allocator cap
and writes ``results/oom.csv``. On a 16 GiB card the full grid fits, so the
capped run is what produces ``OutOfMemoryError`` rows. ``--profile`` records
kernel counts for a few shapes in ``results/profile.json``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from equations import N_PARAMS
from models import SmallCNN

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

BASE_S = (32, 64, 128, 224, 256, 384, 512)
BASE_B = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SEED = 0


def build_grid(seed: int = SEED) -> list[dict]:
    """11 image sizes × 12 batch sizes. Extra S and B are the validation set."""
    rng = np.random.default_rng(seed)
    extra_s_pool = [s for s in range(32, 513, 16) if s not in BASE_S]
    extra_b_pool = [b for b in range(1, 257) if b not in BASE_B]
    extra_s = tuple(int(s) for s in np.sort(rng.choice(extra_s_pool, size=4, replace=False)))
    extra_b = tuple(int(b) for b in np.sort(rng.choice(extra_b_pool, size=3, replace=False)))
    sizes = tuple(sorted(BASE_S + extra_s))
    batches = tuple(sorted(BASE_B + extra_b))
    points = []
    for image_size in sizes:
        for batch in batches:
            held_out = image_size in extra_s or batch in extra_b
            points.append(
                {
                    "S": image_size,
                    "B": batch,
                    "is_validation": int(held_out),
                }
            )
    points.sort(key=lambda row: (row["S"] * row["S"] * row["B"], row["S"], row["B"]))
    return points


def configure_cuda() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = False


def _time_forward(model: torch.nn.Module, inputs: torch.Tensor) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        model(inputs)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _energy_joules(handle, repeats: int) -> int:
    """Return the NVML counter in millijoules. ``repeats`` is unused; kept for clarity at the call."""
    del repeats
    import pynvml

    return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))


def measure_point(model: torch.nn.Module, image_size: int, batch: int, energy_handle) -> dict:
    """One forward's peak memory, median wall-clock latency, and energy per forward."""
    inputs = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        inputs = torch.randn(batch, 3, image_size, image_size, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            model(inputs)
        torch.cuda.synchronize()
        memory_bytes = int(torch.cuda.max_memory_allocated())

        with torch.inference_mode():
            for _ in range(8):
                model(inputs)
        torch.cuda.synchronize()

        probe = _time_forward(model, inputs)
        n_latency = int(np.clip(math.ceil(0.20 / max(probe, 1e-6)), 10, 40))
        samples = [probe] + [_time_forward(model, inputs) for _ in range(n_latency - 1)]
        latency_s = float(np.median(samples))

        n_energy = int(np.clip(math.ceil(0.45 / max(latency_s, 1e-6)), 15, 8000))
        torch.cuda.synchronize()
        energy_before = _energy_joules(energy_handle, n_energy)
        with torch.inference_mode():
            for _ in range(n_energy):
                model(inputs)
        torch.cuda.synchronize()
        energy_after = _energy_joules(energy_handle, n_energy)
        if energy_after < energy_before:
            raise RuntimeError("NVML energy counter went backwards")
        energy_j = (energy_after - energy_before) / 1000.0 / n_energy
        return {
            "latency_s": latency_s,
            "memory_bytes": memory_bytes,
            "energy_j": energy_j,
            "oom": False,
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return {"latency_s": "", "memory_bytes": "OOM", "energy_j": "", "oom": True}
    finally:
        del inputs
        gc.collect()
        torch.cuda.empty_cache()


def hardware_record(points: list[dict]) -> dict:
    import pynvml

    props = torch.cuda.get_device_properties(0)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    extra_s = sorted({row["S"] for row in points if row["is_validation"] and row["S"] not in BASE_S})
    extra_b = sorted({row["B"] for row in points if row["is_validation"] and row["B"] not in BASE_B})
    name = props.name
    return {
        "gpu_name": name,
        "gpu_memory_bytes": int(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "driver": str(pynvml.nvmlSystemGetDriverVersion()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "seed": SEED,
        "base_S": list(BASE_S),
        "base_B": list(BASE_B),
        "extra_S": extra_s,
        "extra_B": extra_b,
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "dtype": "fp32",
        "mode": "eval + inference_mode",
    }


def warmup(model: torch.nn.Module) -> None:
    for image_size, batch in ((64, 4), (256, 8)):
        inputs = torch.randn(batch, 3, image_size, image_size, device="cuda")
        with torch.inference_mode():
            for _ in range(4):
                model(inputs)
        del inputs
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def measure_memory_only(model: torch.nn.Module, image_size: int, batch: int):
    """Peak memory of one forward, or OOM if the capped allocator refuses it."""
    inputs = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        inputs = torch.randn(batch, 3, image_size, image_size, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            model(inputs)
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return "OOM"
    finally:
        del inputs
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def measure_oom(limit_gib: float) -> None:
    """One forward per grid point under an allocator cap of ``limit_gib`` GiB."""
    from equations import memory

    if limit_gib <= 0:
        raise SystemExit("--oom-limit-gib must be positive")
    configure_cuda()
    total = int(torch.cuda.get_device_properties(0).total_memory)
    fraction = (limit_gib * 1024**3) / total
    if fraction >= 1:
        raise SystemExit(f"{limit_gib} GiB is not below the {total / 1024**3:.2f} GiB card")
    torch.cuda.set_per_process_memory_fraction(fraction)
    torch.manual_seed(SEED)
    model = SmallCNN().cuda().eval()
    points = build_grid(SEED)
    limit_bytes = int(limit_gib * 1024**3)
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "oom.csv"
    fields = ["S", "B", "memory_bytes", "predicted_bytes", "is_validation", "limit_bytes"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, point in enumerate(points, start=1):
            measured = measure_memory_only(model, point["S"], point["B"])
            predicted = int(round(float(memory(point["S"], point["B"]))))
            writer.writerow(
                {
                    "S": point["S"],
                    "B": point["B"],
                    "memory_bytes": measured,
                    "predicted_bytes": predicted,
                    "is_validation": point["is_validation"],
                    "limit_bytes": limit_bytes,
                }
            )
            handle.flush()
            if measured == "OOM":
                detail = "OOM"
            else:
                detail = f"mem={measured / 1024**2:.1f} MiB"
            pred_flag = "pred-OOM" if predicted > limit_bytes else "pred-fit"
            print(
                f"[{index:3d}/132] S={point['S']:4d} B={point['B']:4d}  {detail}  {pred_flag}",
                flush=True,
            )
    print(f"wrote {csv_path}", flush=True)


def profile_shapes() -> None:
    """Kernel count and GPU-busy time versus wall time for a few shapes."""
    from torch.profiler import ProfilerActivity, profile

    configure_cuda()
    torch.manual_seed(SEED)
    model = SmallCNN().cuda().eval()
    shapes = ((32, 1), (224, 16), (512, 64), (512, 128))
    report = []
    for image_size, batch in shapes:
        inputs = torch.randn(batch, 3, image_size, image_size, device="cuda")
        with torch.inference_mode():
            for _ in range(5):
                model(inputs)
        torch.cuda.synchronize()
        walls = []
        gpus = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(12):
            torch.cuda.synchronize()
            cpu0 = time.perf_counter()
            start.record()
            with torch.inference_mode():
                model(inputs)
            end.record()
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - cpu0)
            gpus.append(start.elapsed_time(end) / 1000.0)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.inference_mode():
                model(inputs)
            torch.cuda.synchronize()
        kernels = []
        for event in prof.key_averages():
            busy = int(getattr(event, "self_device_time_total", 0) or 0)
            if busy <= 0:
                continue
            kernels.append({"name": event.key, "device_us": busy})
        kernels.sort(key=lambda item: item["device_us"], reverse=True)
        report.append(
            {
                "S": image_size,
                "B": batch,
                "wall_median_s": float(np.median(walls)),
                "cuda_event_median_s": float(np.median(gpus)),
                "kernel_count": len(kernels),
                "top_kernels": kernels[:12],
            }
        )
        print(
            f"S={image_size} B={batch} wall={report[-1]['wall_median_s']*1e3:.3f} ms "
            f"cuda_event={report[-1]['cuda_event_median_s']*1e3:.3f} ms kernels={len(kernels)}",
            flush=True,
        )
        del inputs
        torch.cuda.empty_cache()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "profile.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Measurements need one NVIDIA GPU.")

    import pynvml

    configure_cuda()
    pynvml.nvmlInit()
    energy_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    torch.manual_seed(SEED)
    model = SmallCNN().cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != N_PARAMS:
        raise SystemExit(f"model has {n_params} parameters, equations expect {N_PARAMS}")

    points = build_grid(SEED)
    if len(points) != 132:
        raise SystemExit(f"expected 132 configurations, got {len(points)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    hardware_path = RESULTS / "hardware.json"
    csv_path = RESULTS / "measurements.csv"
    hardware_path.write_text(json.dumps(hardware_record(points), indent=2) + "\n")

    warmup(model)
    fields = ["S", "B", "latency_s", "memory_bytes", "energy_j", "is_validation"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, point in enumerate(points, start=1):
            measured = measure_point(model, point["S"], point["B"], energy_handle)
            row = {
                "S": point["S"],
                "B": point["B"],
                "latency_s": measured["latency_s"],
                "memory_bytes": measured["memory_bytes"],
                "energy_j": measured["energy_j"],
                "is_validation": point["is_validation"],
            }
            writer.writerow(row)
            handle.flush()
            if measured["oom"]:
                detail = "OOM"
            else:
                power_w = measured["energy_j"] / measured["latency_s"]
                detail = (
                    f"lat={measured['latency_s'] * 1e3:.3f} ms  "
                    f"mem={measured['memory_bytes'] / 1024**2:.1f} MiB  "
                    f"E={measured['energy_j']:.4g} J  "
                    f"P={power_w:.0f} W"
                )
            print(
                f"[{index:3d}/132] S={point['S']:4d} B={point['B']:4d} "
                f"val={point['is_validation']}  {detail}",
                flush=True,
            )

    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure the homework-1 CNN on one GPU.")
    parser.add_argument(
        "--oom-limit-gib",
        type=float,
        default=None,
        help="cap the allocator at this many GiB and record only peak memory or OOM",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="record kernel counts for a few shapes and exit",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Measurements need one NVIDIA GPU.")
    if args.profile:
        profile_shapes()
    elif args.oom_limit_gib is not None:
        measure_oom(args.oom_limit_gib)
    else:
        main()
