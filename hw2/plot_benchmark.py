"""Plot incremental loss-operator memory from results/loss_benchmark.csv."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

KIND_TITLE = {
    "loss-fw": "Только forward",
    "loss-bw": "Только backward",
    "loss-fw-bw": "Forward + backward",
}


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Пустой файл {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "loss_benchmark.csv",
    )
    parser.add_argument("--kind", default="loss-fw-bw", choices=tuple(KIND_TITLE))
    args = parser.parse_args()

    rows = [row for row in load_rows(args.csv) if row["kind"] == args.kind]
    if not rows:
        raise SystemExit(f"В {args.csv} нет kind={args.kind}")

    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    memory_figure, memory_axis = plt.subplots(figsize=(8, 4.5))
    time_figure, time_axis = plt.subplots(figsize=(8, 4.5))
    reference = rows[0]
    vocab_logits = [
        (int(row["n_tokens"]), float(row["logits_fp32_mib"]))
        for row in by_method[next(iter(by_method))]
    ]
    memory_axis.plot(
        [item[0] for item in vocab_logits],
        [item[1] for item in vocab_logits],
        linestyle="--",
        color="0.45",
        label="одна копия логитов, fp32",
    )

    oom_notes = []
    for method, method_rows in by_method.items():
        method_rows.sort(key=lambda row: int(row["n_tokens"]))
        fit = [row for row in method_rows if row["oom"] == "0"]
        failed = [row for row in method_rows if row["oom"] == "1"]
        if fit:
            xs = [int(row["n_tokens"]) for row in fit]
            memory_axis.plot(xs, [float(row["op_mem_mib"]) for row in fit], marker="o", label=method)
            time_axis.plot(xs, [float(row["runtime_ms"]) for row in fit], marker="o", label=method)
        if failed:
            oom_notes.append(f"{method}: N=" + ", ".join(row["n_tokens"] for row in failed))

    memory_axis.set_xscale("log", base=2)
    memory_axis.set_yscale("log")
    memory_axis.set_xlabel("Токены в одном вызове лосса")
    memory_axis.set_ylabel("Память оператора, МиБ")
    memory_axis.set_title(f"Лосс Gemma 2 2B, {KIND_TITLE[args.kind]}")
    memory_axis.grid(True, which="both", alpha=0.3)
    memory_axis.legend()
    if oom_notes:
        memory_axis.text(
            0.02,
            0.98,
            "OOM\n" + "\n".join(oom_notes),
            transform=memory_axis.transAxes,
            va="top",
            fontsize=8,
        )
    memory_figure.tight_layout()

    time_axis.set_xscale("log", base=2)
    time_axis.set_xlabel("Токены в одном вызове лосса")
    time_axis.set_ylabel("Время, мс")
    time_axis.set_title(f"Лосс Gemma 2 2B, {KIND_TITLE[args.kind]}")
    time_axis.grid(True, which="both", alpha=0.3)
    time_axis.legend()
    time_figure.tight_layout()

    out_dir = args.csv.parent
    memory_path = out_dir / f"loss_benchmark_memory_{args.kind}.png"
    time_path = out_dir / f"loss_benchmark_time_{args.kind}.png"
    memory_figure.savefig(memory_path, dpi=140)
    time_figure.savefig(time_path, dpi=140)
    print(f"source={reference['source']} kind={args.kind}")
    print(memory_path)
    print(time_path)


if __name__ == "__main__":
    main()
