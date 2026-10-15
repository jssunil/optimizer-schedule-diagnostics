"""Part 3: update-to-weight ratio for every layer during warmup.

rho = ||W_after - W_before|| / ||W_before|| for every 2-D weight at every step (weight decay and clipping included).
Schedule: linear warmup, then a CONSTANT lr (no decay, so only warmup changes the lr). Model S, 3 seeds.
Two warmup lengths: 100 steps, and 40 = 2 / (1 - beta2) (Ma and Yarats, 2019).

"Warmup stops changing rho" is found on the median over the block matrices (embeddings are reported apart):
  method 1: rolling slope of log(rho) (window 10) enters the noise band of the late slopes and stays there 20 steps;
  method 2: a two-line fit of log(rho); the break point is the step with the lowest squared error.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import CONFIG_S, FineWebData
from src.device import RESULTS, logged, pick_device, save_json, train_dtype
from src.train import QUICK, Run, train

PEAK_LR = 2e-3
STEPS, WARMUPS, SEEDS = (60, (20,), [0]) if QUICK else (300, (100, 40), [0, 1, 2])
WINDOW, HOLD = 10, 20


def is_embedding(name):
    return "wte" in name or "wpe" in name


def rolling_slope(y, w):
    """Least-squares slope of y over the last w points, for each index >= w - 1 (NaN before)."""
    out = np.full(len(y), np.nan)
    x = np.arange(w) - (w - 1) / 2
    for i in range(w - 1, len(y)):
        out[i] = np.dot(x, y[i - w + 1:i + 1] - y[i - w + 1:i + 1].mean()) / np.dot(x, x)
    return out


def change_point_slope(logm, warmup):
    """First step where the rolling slope is inside the late noise band for HOLD steps in a row."""
    sl = rolling_slope(logm, WINDOW)
    late = sl[min(len(sl) - 1, warmup + 50):]
    late = late[~np.isnan(late)]
    if len(late) < 5:
        return None, None
    lo, hi = late.mean() - 3 * late.std(), late.mean() + 3 * late.std()
    inside = (sl >= lo) & (sl <= hi)
    for s in range(WINDOW - 1, len(sl) - HOLD):
        if inside[s:s + HOLD].all():
            return s + 1, (float(lo), float(hi))  # 1-based step
    return None, (float(lo), float(hi))


def change_point_fit(logm):
    """Break point of a continuous two-line fit (1-based step)."""
    n, x = len(logm), np.arange(len(logm), dtype=float)
    best, best_b = np.inf, None
    for b in range(5, n - 5):
        X = np.column_stack([np.ones(n), x, np.maximum(0, x - b)])
        coef = np.linalg.lstsq(X, logm, rcond=None)[0]
        sse = float(np.sum((X @ coef - logm) ** 2))
        if sse < best:
            best, best_b = sse, b
    return best_b + 1


def one_warmup(warmup, data, device, dtype):
    runs = [train(Run(CONFIG_S, PEAK_LR, schedule="constant", total=STEPS, warmup=warmup, seed=s, log_ratio=True),
                  data, device, dtype) for s in SEEDS]
    names = runs[0]["ratio_names"]
    block = [i for i, n in enumerate(names) if not is_embedding(n)]
    emb = [i for i, n in enumerate(names) if is_embedding(n)]
    R = np.array([r["ratio"] for r in runs])  # seeds x steps x layers
    med = np.median(R[:, :, block], axis=2)  # seeds x steps
    logm = np.log(med).mean(0)  # mean over seeds of log(median rho)
    lrs = np.array(runs[0]["lr"])
    per_seed = []
    for k in range(len(SEEDS)):
        s1, _ = change_point_slope(np.log(med[k]), warmup)
        per_seed.append(dict(seed=SEEDS[k], slope_method=s1, fit_method=change_point_fit(np.log(med[k]))))
    step_slope, band = change_point_slope(logm, warmup)
    step_fit = change_point_fit(logm)
    after = slice(min(STEPS - 1, warmup + 20), STEPS)
    res = dict(warmup=warmup, peak_lr=PEAK_LR, layer_names=names, block_index=block, embedding_index=emb,
               median_block_rho=np.exp(logm).tolist(), median_block_rho_over_lr=(np.exp(logm) / lrs).tolist(),
               per_layer_rho_mean_over_seeds=R.mean(0).tolist(), lr=lrs.tolist(),
               change_step_slope=step_slope, slope_band=band, change_step_fit=step_fit, per_seed=per_seed,
               rho_after_warmup_median=float(np.median(np.exp(logm)[after])),
               embedding_rho_after_warmup={names[i]: float(np.median(R[:, after, i])) for i in emb},
               grad_clip_events=sum(sum(g > 1.0 for g in r["grad_norm"]) for r in runs))
    print(f"warmup {warmup}: rho stops changing at step {step_slope} (rolling slope) / {step_fit} (two-line fit); "
          f"per seed {[(p['slope_method'], p['fit_method']) for p in per_seed]}; "
          f"median block rho after warmup {res['rho_after_warmup_median']:.2e}; embeddings "
          + ", ".join(f"{k} {v:.1e}" for k, v in res["embedding_rho_after_warmup"].items())
          + f"; clip events {res['grad_clip_events']}")
    return res


def plot(results):
    fig, ax = plt.subplots(2, len(results), figsize=(6.5 * len(results), 8), squeeze=False)
    for j, r in enumerate(results):
        s = np.arange(1, len(r["lr"]) + 1)
        layers = np.array(r["per_layer_rho_mean_over_seeds"])
        a = ax[0, j]
        for i in r["block_index"]:
            a.plot(s, layers[:, i], color="tab:blue", lw=0.4, alpha=0.5)
        for i in r["embedding_index"]:
            a.plot(s, layers[:, i], lw=1.2, label=r["layer_names"][i])
        a.plot(s, r["median_block_rho"], color="k", lw=2, label="median of block matrices")
        a.axvline(r["warmup"], color="grey", ls="--", label=f"end of warmup ({r['warmup']})")
        for step, style, lab in [(r["change_step_slope"], ":", "change (slope)"), (r["change_step_fit"], "-.", "change (fit)")]:
            if step:
                a.axvline(step, color="red", ls=style, label=f"{lab}: {step}")
        a.axhline(1e-3, color="green", lw=0.8, label="1e-3 guide")
        a.set(yscale="log", title=f"Update-to-weight ratio, warmup {r['warmup']}", xlabel="step", ylabel="rho")
        a.legend(fontsize=7)
        b = ax[1, j]
        b.plot(s, r["median_block_rho_over_lr"], color="k")
        b.axvline(r["warmup"], color="grey", ls="--")
        b.set(yscale="log", title="median rho / lr (the schedule removed)", xlabel="step", ylabel="rho / lr")
    fig.tight_layout()
    fig.savefig(RESULTS / "p3_update_ratio.png", dpi=120)
    plt.close(fig)


@logged("p3_update_ratio")
def run():
    device = pick_device()
    dtype = train_dtype(device)
    data = FineWebData()
    results = [one_warmup(w, data, device, dtype) for w in WARMUPS]
    plot(results)
    save_json("p3_update_ratio.json", dict(quick=QUICK, steps=STEPS, seeds=SEEDS, results=results))


if __name__ == "__main__":
    run()
