"""Closed-form cost of the homework-1 CNN.

Network, eval, FP32. Image side S is a multiple of 16.

    Conv7x7 s2  3→32, ReLU, MaxPool 3×3 s2 p1
    Conv5×5     32→64, ReLU
    Conv3×3 s2  64→128, ReLU
    Conv1×1     128→256, ReLU
    Conv3×3 s2  256→256, ReLU
    Conv1×1     256→512, ReLU
    GlobalAvgPool, Linear 512→256, ReLU, Linear 256→100

Convolutions use padding = k // 2 and bias = False.
Linear layers use bias (PyTorch default). ReLU is inplace.

Assumptions
-----------
FLOPs
    1 multiply-accumulate = 2 FLOPs.
    A linear bias add is 1 FLOP. Global average pooling is one add per
    spatial element and one scale per channel.
    ReLU and MaxPool are comparisons, so they are not in the FLOP count.

Memory
    Peak of the bytes that are live together during one inference forward:
    parameters, the input (the caller keeps it), and the activations of the
    current op. Inplace ReLU allocates nothing. Autograd saves nothing.
    That peak is the MaxPool step. CUDA workspace, caching-allocator
    rounding, and context memory are not in the formula, so it is a lower
    bound on ``torch.cuda.max_memory_allocated()``.

Bytes moved
    Streaming traffic with no cache reuse and ReLU fused into the producer:
    every convolution, pool, pool-read, GAP, and linear reads its inputs
    and writes its output; every parameter is read once.

Latency
    Sequential kernels, three additive terms. θ = (t0, t_flop, t_byte):
    fixed per-forward overhead (launch-bound), seconds per FLOP
    (compute-bound), seconds per byte (memory-bound). The regime is
    whichever term is largest.

Energy
    Same shape, in joules. φ = (e0, e_flop, e_byte). Whole-GPU energy mixes
    static power paid for the duration of the forward with dynamic energy;
    both collapse into this linear form, so φ is fit on its own, not tied
    to the latency parameters.

Closed forms
------------
    FLOPs(S, B)   = B * (17714 * S**2 + 314212)
    Memory(S, B)  = 4161296 + 52 * B * S**2
    Bytes(S, B)   = 4161296 + 6544 * B + 196 * B * S**2
    Latency(S, B, θ) = t0 + t_flop * FLOPs + t_byte * Bytes
    Energy(S, B, φ)  = e0 + e_flop * FLOPs + e_byte * Bytes
"""

from __future__ import annotations

import numpy as np

# name, cin, cout, kernel, spatial divisor of the output (H = W = S / div)
_CONVS = (
    ("conv7x7", 3, 32, 7, 2),
    ("conv5x5", 32, 64, 5, 4),
    ("conv3x3_s2", 64, 128, 3, 8),
    ("conv1x1", 128, 256, 1, 8),
    ("conv3x3_s2_b", 256, 256, 3, 16),
    ("conv1x1_b", 256, 512, 1, 16),
)
_LINEAR = ((512, 256), (256, 100))
_BYTES = 4


def _param_elements() -> int:
    conv = sum(cin * cout * k * k for _, cin, cout, k, _ in _CONVS)
    linear = sum(cin * cout + cout for cin, cout in _LINEAR)
    return conv + linear


def _closed_form_coefficients() -> tuple[int, int, int, int, int]:
    """Return (flop_s2, flop_b, mem_s2, byte_s2, byte_b).

    flop  = B * (flop_s2 * S**2 + flop_b)
    memory = param_bytes + mem_s2 * B * S**2
    bytes  = param_bytes + byte_b * B + byte_s2 * B * S**2
    """
    flop_s2 = 0
    for _, cin, cout, k, div in _CONVS:
        numer = 2 * cin * cout * k * k
        denom = div * div
        if numer % denom != 0:
            raise RuntimeError(f"conv FLOP coefficient is not integer for k={k}")
        flop_s2 += numer // denom

    # GAP: B * 512 * (S/16)**2 adds, plus B * 512 scales.
    flop_s2 += 512 // 256
    flop_b = 512
    for cin, cout in _LINEAR:
        flop_b += 2 * cin * cout + cout

    # Element counts of one sample, as a coefficient of S**2.
    area = {"input": 3, "pool": 32 // 16}
    for name, _, cout, _, div in _CONVS:
        if cout % (div * div) != 0:
            raise RuntimeError(f"activation size is not an integer multiple of S**2 for {name}")
        area[name] = cout // (div * div)

    # Reads and writes of activations. Weights are added separately.
    traffic_s2 = 0
    traffic_s2 += area["input"] + area["conv7x7"]
    traffic_s2 += area["conv7x7"] + area["pool"]
    traffic_s2 += area["pool"] + area["conv5x5"]
    traffic_s2 += area["conv5x5"] + area["conv3x3_s2"]
    traffic_s2 += area["conv3x3_s2"] + area["conv1x1"]
    traffic_s2 += area["conv1x1"] + area["conv3x3_s2_b"]
    traffic_s2 += area["conv3x3_s2_b"] + area["conv1x1_b"]
    traffic_s2 += area["conv1x1_b"]  # GAP reads the last map

    # GAP writes 512, Linear reads and writes its vectors.
    traffic_b = 512 + 512 + 256 + 256 + 100

    # Live activation elements while the caller still holds the input.
    # Head vectors are O(B) and are smaller than the pool peak for every S >= 32.
    peaks = {
        "conv7x7": area["input"] + area["conv7x7"],
        "pool": area["input"] + area["conv7x7"] + area["pool"],
        "conv5x5": area["input"] + area["pool"] + area["conv5x5"],
        "conv3x3_s2": area["input"] + area["conv5x5"] + area["conv3x3_s2"],
        "conv1x1": area["input"] + area["conv3x3_s2"] + area["conv1x1"],
        "conv3x3_s2_b": area["input"] + area["conv1x1"] + area["conv3x3_s2_b"],
        "conv1x1_b": area["input"] + area["conv3x3_s2_b"] + area["conv1x1_b"],
    }
    peak_at = max(peaks, key=peaks.get)
    if peak_at != "pool" or peaks["pool"] != 13:
        raise RuntimeError(f"expected the memory peak at pool (13 S**2), got {peak_at}={peaks[peak_at]}")

    return flop_s2, flop_b, _BYTES * peaks["pool"], _BYTES * traffic_s2, _BYTES * traffic_b


def _check_spatial_schedule() -> None:
    """Output side equals S/4, S/8, S/16 on the measurement grid."""

    def side(h: int, k: int, stride: int, pad: int) -> int:
        return (h + 2 * pad - k) // stride + 1

    schedule = (
        (7, 2, 3, 2),
        (3, 2, 1, 4),
        (5, 1, 2, 4),
        (3, 2, 1, 8),
        (1, 1, 0, 8),
        (3, 2, 1, 16),
        (1, 1, 0, 16),
    )
    for image_size in (32, 48, 64, 80, 128, 224, 256, 384, 496, 512):
        h = image_size
        for k, stride, pad, div in schedule:
            h = side(h, k, stride, pad)
            if h != image_size // div:
                raise RuntimeError(f"S={image_size}: spatial {h} != S/{div}")


N_PARAMS = _param_elements()
PARAM_BYTES = _BYTES * N_PARAMS
FLOP_S2, FLOP_B, MEM_S2, BYTE_S2, BYTE_B = _closed_form_coefficients()
_check_spatial_schedule()

if (N_PARAMS, FLOP_S2, FLOP_B, MEM_S2, BYTE_S2, BYTE_B) != (
    1_040_324,
    17_714,
    314_212,
    52,
    196,
    6544,
):
    raise RuntimeError("closed form drifted from the handwritten coefficients")


def _pair(image_size, batch):
    image_size = np.asarray(image_size, dtype=np.float64)
    batch = np.asarray(batch, dtype=np.float64)
    return np.broadcast_arrays(image_size, batch)


def _result(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0:
        return float(value)
    return value


def _theta(theta) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if values.shape != (3,):
        raise ValueError(
            "theta must be three numbers (overhead, per_flop, per_byte), "
            f"got shape {values.shape}"
        )
    return values


def flops(image_size, batch):
    """FLOPs of one forward pass. 1 MAC = 2 FLOPs."""
    image_size, batch = _pair(image_size, batch)
    area = image_size * image_size
    return _result(batch * (FLOP_S2 * area + FLOP_B))


def memory(image_size, batch):
    """Peak live bytes of one forward: parameters plus the MaxPool working set."""
    image_size, batch = _pair(image_size, batch)
    return _result(PARAM_BYTES + MEM_S2 * batch * image_size * image_size)


def bytes_moved(image_size, batch):
    """Bytes read or written by one forward, with no cache reuse."""
    image_size, batch = _pair(image_size, batch)
    area = image_size * image_size
    return _result(PARAM_BYTES + BYTE_B * batch + BYTE_S2 * batch * area)


def latency(image_size, batch, theta):
    """Seconds. theta = (t0, seconds per FLOP, seconds per byte)."""
    t0, t_flop, t_byte = _theta(theta)
    return _result(t0 + t_flop * flops(image_size, batch) + t_byte * bytes_moved(image_size, batch))


def energy(image_size, batch, theta_energy):
    """Joules. theta_energy = (e0, joules per FLOP, joules per byte)."""
    e0, e_flop, e_byte = _theta(theta_energy)
    return _result(e0 + e_flop * flops(image_size, batch) + e_byte * bytes_moved(image_size, batch))
