#!/usr/bin/env bash
# Table 1 style sweep: memory of the loss operator, not of a training step.
# Extra arguments are forwarded, for example: --iterations 10
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/pc-01/prj/venv/bin/python
"$PY" benchmark_loss.py "$@"
"$PY" plot_benchmark.py --kind loss-fw-bw
