"""Memory and time of the loss operator alone, Gemma 2 2B shapes.

This is the Table 1 measurement from Wijmans et al.: incremental GPU memory of
cross-entropy, not the peak of a full training step. Hidden size 2304, vocabulary
256000, softcap 30. The token count is swept up to 8192, the batch in the paper.

`cce` and `torch_compile` call `linear_cross_entropy` from the submodule.
`baseline` is `logits = E @ C.T` followed by `F.cross_entropy`, as in their
`benchmark/__main__.py`. Out of memory is a recorded result, not a crash.
"""

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100
SUBMODULE = Path(__file__).resolve().parent / "ml-cross-entropy"
HIDDEN = 2304
VOCAB = 256_000
SOFTCAP = 30.0
DEFAULT_TOKENS = (128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_METHODS = ("cce", "torch_compile", "baseline")
DEFAULT_KINDS = ("loss-fw", "loss-bw", "loss-fw-bw")
FIELDS = (
    "n_tokens",
    "method",
    "kind",
    "runtime_ms",
    "op_mem_mib",
    "oom",
    "resident_mib",
    "logits_fp32_mib",
    "source",
)


def require_submodule() -> None:
    marker = SUBMODULE / "cut_cross_entropy" / "__init__.py"
    if marker.is_file():
        if str(SUBMODULE) not in sys.path:
            sys.path.insert(0, str(SUBMODULE))
        return
    raise SystemExit(
        "Нет checkout сабмодуля hw2/ml-cross-entropy.\n"
        "Из корня репозитория:\n"
        "  git submodule update --init hw2/ml-cross-entropy"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(DEFAULT_TOKENS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--kinds", nargs="+", default=list(DEFAULT_KINDS))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--vocab", type=int, default=VOCAB)
    parser.add_argument("--softcap", type=float, default=SOFTCAP)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "loss_benchmark.csv",
    )
    return parser.parse_args()


def baseline(embedding, classifier, targets, softcap: float | None = None) -> torch.Tensor:
    logits = embedding @ classifier.T
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(logits.float(), targets, ignore_index=IGNORE_INDEX)


def call_loss(method: str, embedding, classifier, targets, softcap: float):
    if method == "baseline":
        return baseline(embedding, classifier, targets, softcap=softcap)
    from cut_cross_entropy import linear_cross_entropy

    return linear_cross_entropy(
        embedding,
        classifier,
        targets,
        softcap=softcap,
        impl=method,
        reduction="mean",
        ignore_index=IGNORE_INDEX,
    )


def mib(num_bytes: int) -> float:
    return num_bytes / 2**20


def resident_bytes(embedding, classifier) -> int:
    return embedding.numel() * embedding.element_size() + classifier.numel() * classifier.element_size()


def clear_grads(embedding, classifier) -> None:
    embedding.grad = None
    classifier.grad = None


def release_oom() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def measure(method: str, embedding, classifier, targets, softcap: float, kind: str, iterations: int):
    """Same peak-delta as benchmark/memory.py in the Apple repo.

    loss-fw: peak of the forward minus the peak before it.
    loss-bw: forward is not counted; peak is reset after the graph exists.
    loss-fw-bw: one peak covering forward and backward.
    """
    forward = kind in ("loss-fw", "loss-fw-bw")
    backward = kind in ("loss-bw", "loss-fw-bw")

    try:
        clear_grads(embedding, classifier)
        warmup = call_loss(method, embedding, classifier, targets, softcap)
        if backward:
            warmup.backward()
        del warmup
        clear_grads(embedding, classifier)
        release_oom()
    except torch.cuda.OutOfMemoryError:
        clear_grads(embedding, classifier)
        release_oom()
        return None

    def once():
        clear_grads(embedding, classifier)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_stats()
        loss = call_loss(method, embedding, classifier, targets, softcap)
        torch.cuda.synchronize()
        if forward:
            after = torch.cuda.memory_stats()
        if backward and not forward:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_stats()
        if backward:
            loss.backward()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
        delta = after["allocated_bytes.all.peak"] - before["allocated_bytes.all.peak"]
        return loss, delta

    try:
        _, op_bytes = once()
    except torch.cuda.OutOfMemoryError:
        clear_grads(embedding, classifier)
        release_oom()
        return None

    clear_grads(embedding, classifier)
    release_oom()

    start_events = []
    end_events = []
    try:
        for _ in range(iterations):
            clear_grads(embedding, classifier)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if forward:
                start.record()
            loss = call_loss(method, embedding, classifier, targets, softcap)
            if not forward:
                start.record()
            if backward:
                loss.backward()
            end.record()
            start_events.append(start)
            end_events.append(end)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        clear_grads(embedding, classifier)
        release_oom()
        return None

    elapsed = 0.0
    for start, end in zip(start_events, end_events, strict=True):
        elapsed += start.elapsed_time(end)
    clear_grads(embedding, classifier)
    release_oom()
    return elapsed / iterations, mib(op_bytes)


def random_batch(n_tokens: int, hidden: int, vocab: int, device: torch.device):
    scale = hidden**0.25
    embedding = torch.randn(n_tokens, hidden, device=device, dtype=torch.bfloat16) / scale
    classifier = torch.randn(vocab, hidden, device=device, dtype=torch.bfloat16) / scale
    targets = torch.randint(0, vocab, (n_tokens,), device=device)
    embedding.requires_grad_(True)
    classifier.requires_grad_(True)
    return embedding, classifier, targets


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Нужна CUDA GPU.")
    unknown = [kind for kind in args.kinds if kind not in DEFAULT_KINDS]
    if unknown:
        raise SystemExit(f"Неизвестный kind: {unknown}. Допустимо: {', '.join(DEFAULT_KINDS)}")
    if any(method != "baseline" for method in args.methods):
        require_submodule()
    else:
        sys.path.insert(0, str(SUBMODULE))

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(
        f"{props.name}  {mib(props.total_memory):.0f} МиБ  "
        f"D={args.hidden} V={args.vocab} softcap={args.softcap}",
        flush=True,
    )

    max_tokens = max(args.tokens)
    full_embedding, classifier, full_targets = random_batch(max_tokens, args.hidden, args.vocab, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for n_tokens in args.tokens:
            embedding = full_embedding[:n_tokens].detach().requires_grad_(True)
            targets = full_targets[:n_tokens]
            logits_mib = mib(n_tokens * args.vocab * 4)
            resident = mib(resident_bytes(embedding, classifier))
            print(
                f"N={n_tokens}  одна копия логитов fp32={logits_mib:.0f} МиБ  "
                f"E+C={resident:.0f} МиБ",
                flush=True,
            )
            for kind in args.kinds:
                for method in args.methods:
                    measured = measure(
                        method,
                        embedding,
                        classifier,
                        targets,
                        args.softcap,
                        kind,
                        args.iterations,
                    )
                    if measured is None:
                        runtime, op_mem, oom = "", "", 1
                        print(f"  {method:14} {kind:12} OOM", flush=True)
                    else:
                        runtime_value, op_mem_value = measured
                        runtime, op_mem, oom = f"{runtime_value:.2f}", f"{op_mem_value:.1f}", 0
                        print(
                            f"  {method:14} {kind:12} {op_mem_value:10.1f} МиБ  {runtime_value:8.1f} мс",
                            flush=True,
                        )
                    writer.writerow(
                        {
                            "n_tokens": n_tokens,
                            "method": method,
                            "kind": kind,
                            "runtime_ms": runtime,
                            "op_mem_mib": op_mem,
                            "oom": oom,
                            "resident_mib": f"{resident:.1f}",
                            "logits_fp32_mib": f"{logits_mib:.1f}",
                            "source": "randn",
                        }
                    )
                    handle.flush()
                    clear_grads(embedding, classifier)
                    clear_grads(full_embedding, classifier)
                    release_oom()
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
