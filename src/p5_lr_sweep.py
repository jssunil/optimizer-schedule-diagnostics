"""Part 5: LR sweep at widths 256, 512 and 1024; prediction for width 4096.

Fixed: 4 layers, head size 64, context 512, batch 16, 300 steps (2.5 M tokens), warmup 30 + cosine to 0.1,
AdamW lambda = 0.1 (decay per step = lr * lambda, so it grows with lr; this is the usual practice).
LR grid: 9 values from 2.5e-4 at x sqrt(2) spacing, 2 seeds; an edge minimum is extended (train.lr_sweep).
LR* of a width: vertex of a parabola in log(lr) through the lowest point and its two neighbours.
Bootstrap (seeds resampled, 500 times) gives an interval for each LR* and for the width-4096 prediction.

P5_FAST=1: x2 spacing (5 values) and 1 seed (the cut in PLAN.md section 11).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import WIDTHS, FineWebData
from src.device import RESULTS, logged, pick_device, save_json, train_dtype
from src.train import QUICK, Run, lr_sweep

FAST = os.environ.get("P5_FAST") == "1"
TARGET_WIDTH, N_BOOT, B_SIZE, T_SIZE = 4096, 500, 16, 512
if QUICK:
    STEPS, WARMUP, LRS, SEEDS, N_EVAL, FACTOR = 20, 3, [1e-3, 2e-3, 4e-3], [0], 2, 2.0
elif FAST:
    STEPS, WARMUP, LRS, SEEDS, N_EVAL, FACTOR = 300, 30, [2.5e-4 * 2 ** k for k in range(5)], [0], 40, 2.0
else:
    STEPS, WARMUP, LRS, SEEDS, N_EVAL, FACTOR = 300, 30, [2.5e-4 * 2 ** (k / 2) for k in range(9)], [0, 1], 40, 2 ** 0.5


def vertex(lrs, losses):
    """LR* from a parabola in log(lr) through the lowest point and its neighbours. (lr*, at_edge)."""
    x, y = np.log(np.asarray(lrs)), np.asarray(losses)
    i = int(np.argmin(y))
    if i == 0 or i == len(y) - 1:
        return float(lrs[i]), True
    a, b, _ = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
    if a <= 0:
        return float(lrs[i]), False
    return float(np.exp(np.clip(-b / (2 * a), x[i - 1], x[i + 1]))), False


def fit_line(widths, lr_stars):
    """log lr* = c + slope * log width. Returns (slope, c)."""
    slope, c = np.polyfit(np.log(widths), np.log(lr_stars), 1)
    return float(slope), float(c)


@logged("p5_lr_sweep")
def run():
    device = pick_device()
    dtype = train_dtype(device)
    data = FineWebData()
    evb = data.eval_batches(N_EVAL, B_SIZE, T_SIZE)
    widths = sorted(WIDTHS)
    sweeps, per_width = {}, {}
    for d in widths:
        print(f"width {d}: LR sweep")
        make = lambda lr, seed, d=d: Run(WIDTHS[d], lr, schedule="cosine", total=STEPS, warmup=WARMUP, seed=seed,
                                         B=B_SIZE, T=T_SIZE)
        sweeps[d] = lr_sweep(make, LRS, SEEDS, data, device, dtype, evb, STEPS, factor=FACTOR, label=f"d={d}")
        lrs = list(sweeps[d])
        vals = np.array([[x["val"] for x in sweeps[d][q]] for q in lrs])  # lrs x seeds
        star, edge = vertex(lrs, vals.mean(1))
        per_width[d] = dict(lrs=lrs, val_mean=vals.mean(1).tolist(), val_by_seed=vals.tolist(),
                            lr_star=star, at_edge=edge)
        print(f"  width {d}: LR* = {star:.2e}{' (AT GRID EDGE: not a real minimum)' if edge else ''}")

    stars = [per_width[d]["lr_star"] for d in widths]
    slope, c = fit_line(widths, stars)
    pred = float(np.exp(c + slope * np.log(TARGET_WIDTH)))

    rng = np.random.default_rng(0)
    boot_stars, boot_pred, boot_slope = {d: [] for d in widths}, [], []
    for _ in range(N_BOOT):
        bs = []
        for d in widths:
            vals = np.array(per_width[d]["val_by_seed"])
            pick = rng.integers(0, vals.shape[1], size=vals.shape[1])
            s, _ = vertex(per_width[d]["lrs"], vals[:, pick].mean(1))
            boot_stars[d].append(s)
            bs.append(s)
        sl, cc = fit_line(widths, bs)
        boot_slope.append(sl)
        boot_pred.append(float(np.exp(cc + sl * np.log(TARGET_WIDTH))))
    for d in widths:
        per_width[d]["lr_star_interval_90"] = [float(np.percentile(boot_stars[d], 5)), float(np.percentile(boot_stars[d], 95))]
    pred_ci = [float(np.percentile(boot_pred, 5)), float(np.percentile(boot_pred, 95))]
    slope_ci = [float(np.percentile(boot_slope, 5)), float(np.percentile(boot_slope, 95))]
    any_edge = any(per_width[d]["at_edge"] for d in widths)

    print(f"\nLR* by width: " + ", ".join(f"{d}: {per_width[d]['lr_star']:.2e} "
                                         f"[{per_width[d]['lr_star_interval_90'][0]:.1e}, {per_width[d]['lr_star_interval_90'][1]:.1e}]"
                                         for d in widths))
    print(f"Fit: LR* ~ width^{slope:.2f} (90% bootstrap {slope_ci[0]:.2f}..{slope_ci[1]:.2f})")
    print(f"Prediction at width {TARGET_WIDTH}: {pred:.2e} (90% bootstrap {pred_ci[0]:.1e}..{pred_ci[1]:.1e})")
    confidence = ("LOW: a minimum is at the grid edge." if any_edge else
                  "LOW TO MODERATE: 4096 is 4x outside the measured widths; three points fix a line but do not test it; "
                  "a real width-4096 model trains on far more tokens (LR* falls with longer training); batch size and "
                  "depth are fixed here; the embedding is a large share of the weights at small width; the bootstrap "
                  "covers seed noise only, not these effects.")
    print(f"Confidence: {confidence}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for d in widths:
        p = per_width[d]
        line, = ax[0].plot(p["lrs"], p["val_mean"], "o-", label=f"width {d}")
        ax[0].axvline(p["lr_star"], color=line.get_color(), ls=":")
        ax[0].plot([p["lr_star"]], [min(p["val_mean"])], "*", ms=14, color=line.get_color())
    ax[0].set(xscale="log", xlabel="peak LR", ylabel=f"val loss at step {STEPS}", title="Loss against LR (stars: LR*)")
    ax[0].legend()
    xs = np.array(widths + [TARGET_WIDTH])
    ax[1].errorbar(widths, stars, yerr=[[s - per_width[d]["lr_star_interval_90"][0] for s, d in zip(stars, widths)],
                                        [per_width[d]["lr_star_interval_90"][1] - s for s, d in zip(stars, widths)]],
                   fmt="o", label="LR* (90% bootstrap)")
    ax[1].plot(xs, np.exp(c + slope * np.log(xs)), "--", label=f"fit: width^{slope:.2f}")
    ax[1].errorbar([TARGET_WIDTH], [pred], yerr=[[pred - pred_ci[0]], [pred_ci[1] - pred]], fmt="s", color="red",
                   label=f"prediction {pred:.1e}")
    ax[1].set(xscale="log", yscale="log", xlabel="width", ylabel="LR*", title="LR* against width")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "p5_lr_sweep.png", dpi=120)
    plt.close(fig)

    save_json("p5_lr_sweep.json", dict(quick=QUICK, fast=FAST, steps=STEPS, warmup=WARMUP, seeds=SEEDS, lr_grid=LRS,
                                        per_width={str(d): v for d, v in per_width.items()}, slope=slope,
                                        slope_interval_90=slope_ci, prediction_4096=pred,
                                        prediction_interval_90=pred_ci, confidence=confidence))


if __name__ == "__main__":
    run()
