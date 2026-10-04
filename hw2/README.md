# ДЗ-2. Cut Cross-Entropy

Статья: Erik Wijmans, Brody Huval, Alexander Hertzberg, Vladlen Koltun, Philipp Krähenbühl. *Cut Your Losses in Large-Vocabulary Language Models*. ICLR 2025. [arXiv:2411.09009](https://arxiv.org/abs/2411.09009).

Код статьи — сабмодуль `ml-cross-entropy`, коммит `3de376c106a1916bc5e1b619f9c77c87a461ee1c`.

Воспроизводится Table 1: память и время оператора кросс-энтропии на форме Gemma 2 2B. Разбор чисел — в [report.pdf](report.pdf). Презентация — `Cut-Cross-Entropy.pptx`.

## Как повторить

```bash
git submodule update --init hw2/ml-cross-entropy
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r hw2/requirements.txt
./hw2/run_benchmark.sh
```

Скрипт пишет `results/loss_benchmark.csv` и два рисунка для forward+backward.
