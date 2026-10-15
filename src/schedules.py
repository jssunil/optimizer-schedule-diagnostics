"""Learning-rate schedules. `s` is the 0-based index of the step about to run (s = 0 is step 1).

Every schedule starts with a linear warmup: lr = peak * (s + 1) / warmup for s < warmup.
  constant: peak after warmup.
  cosine:   half a cosine from peak (at s = warmup) to min_ratio * peak (at s = total - 1).
  wsd:      peak until the decay starts at s = total - round(decay_frac * total),
            then a decay to min_ratio * peak at s = total - 1. decay_shape: linear | cosine | 1-sqrt.
"""
import math


def _decay(q, shape):
    """Fraction of (peak - min) still left after a fraction q of the decay phase (q in [0, 1])."""
    if shape == "linear":
        return 1 - q
    if shape == "cosine":
        return 0.5 * (1 + math.cos(math.pi * q))
    if shape == "1-sqrt":
        return 1 - math.sqrt(q)
    raise ValueError(f"unknown decay shape {shape!r}")


def lr_at(s, kind, peak, warmup, total, min_ratio=0.1, decay_frac=0.2, decay_shape="linear"):
    if warmup and s < warmup:
        return peak * (s + 1) / warmup
    lo = min_ratio * peak
    if kind == "constant":
        return peak
    if kind == "cosine":
        q = min(1.0, (s - warmup) / max(1, total - 1 - warmup))
        return lo + (peak - lo) * _decay(q, "cosine")
    if kind == "wsd":
        start = total - round(decay_frac * total)
        if s < start:
            return peak
        q = min(1.0, (s - start) / max(1, total - 1 - start))
        return lo + (peak - lo) * _decay(q, decay_shape)
    raise ValueError(f"unknown schedule {kind!r}")
