"""Part 2: bias correction.

(a) Scalar weight, part 1 gradients repeated: the step with and without bias correction,
    for beta2 = 0.95 and 0.999. The measured ratio must equal r(t) = (1 - b1^t) / sqrt(1 - b2^t).
    "Stops mattering" for the step size: the first t after which |r(t) - 1| < 1% for good.
(b) Model S on FineWeb, RefAdamW with and without bias correction, constant LR, no warmup, 3 seeds.
    "Stops mattering" for the loss: the first step after which the two seed ranges overlap for good.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import CONFIG_S, FineWebData
from src.device import RESULTS, logged, pick_device, save_json, train_dtype
from src.p1_adam_by_hand import GRADS
from src.ref_adam import adam_by_hand, bias_ratio
from src.train import QUICK, Run, rolling_mean, train

BETA1, BETA2S, EPS = 0.9, (0.95, 0.999), 1e-8
PLOT_STEPS, LONG_STEPS, TOL = 20, 6000, 0.01
MODEL_STEPS, MODEL_LR, SEEDS, SMOOTH = (30, 1e-3, [0], 5) if QUICK else (300, 1e-3, [0, 1, 2], 10)


def settle_step(values, ok):
    """First 1-based index after which ok(value) holds for every later value; None if the last value fails."""
    last_bad = max((i for i, x in enumerate(values) if not ok(x)), default=-1)
    return last_bad + 2 if last_bad + 1 < len(values) else None


def scalar_part():
    res = {}
    grads = (GRADS * (LONG_STEPS // len(GRADS) + 1))[:LONG_STEPS]
    for b2 in BETA2S:
        on = adam_by_hand(1.0, grads, 1e-3, BETA1, b2, EPS, bias_correction=True)
        off = adam_by_hand(1.0, grads, 1e-3, BETA1, b2, EPS, bias_correction=False)
        measured = [o["step"] / c["step"] for o, c in zip(off, on)]
        analytic = [bias_ratio(t, BETA1, b2) for t in range(1, LONG_STEPS + 1)]
        res[b2] = dict(
            step_on=[r["step"] for r in on[:PLOT_STEPS]], step_off=[r["step"] for r in off[:PLOT_STEPS]],
            r_analytic=analytic[:100], r_measured=measured[:100],
            max_abs_measured_minus_analytic=max(abs(a - b) for a, b in zip(measured, analytic)),
            settle_step_analytic=settle_step(analytic, lambda x: abs(x - 1) < TOL),
            settle_step_measured=settle_step(measured, lambda x: abs(x - 1) < TOL),
            r_peak=max(analytic[:200]), r_peak_t=int(np.argmax(analytic[:200])) + 1)
        print(f"beta2 {b2}: r(1) {analytic[0]:.3f}, r(20) {analytic[19]:.3f}, peak {res[b2]['r_peak']:.2f} "
              f"at t={res[b2]['r_peak_t']}; |r-1| < 1% from t = {res[b2]['settle_step_analytic']} (analytic), "
              f"{res[b2]['settle_step_measured']} (measured); "
              f"max |measured - analytic| = {res[b2]['max_abs_measured_minus_analytic']:.1e}")
    return res


def model_part(data, device, dtype):
    res = {}
    for b2 in BETA2S:
        curves = {}
        for bc in (True, False):
            runs = [train(Run(CONFIG_S, MODEL_LR, schedule="constant", total=MODEL_STEPS, warmup=0, seed=s,
                              betas=(BETA1, b2), optimizer="ref", bias_correction=bc), data, device, dtype)
                    for s in SEEDS]
            curves[bc] = np.array([rolling_mean(r["loss"], SMOOTH) for r in runs])
        lo_on, hi_on = curves[True].min(0), curves[True].max(0)
        lo_off, hi_off = curves[False].min(0), curves[False].max(0)
        overlap = list(np.maximum(lo_on, lo_off) <= np.minimum(hi_on, hi_off))
        step = settle_step(overlap, bool)
        res[b2] = dict(mean_on=curves[True].mean(0).tolist(), mean_off=curves[False].mean(0).tolist(),
                       lo_on=lo_on.tolist(), hi_on=hi_on.tolist(), lo_off=lo_off.tolist(), hi_off=hi_off.tolist(),
                       loss_settle_step=step, final_on=float(curves[True][:, -1].mean()),
                       final_off=float(curves[False][:, -1].mean()))
        print(f"model, beta2 {b2}: seed ranges overlap for good from step {step}; "
              f"final smoothed loss {res[b2]['final_on']:.3f} (on) vs {res[b2]['final_off']:.3f} (off)")
    return res


def plot(scalar, model):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    t = np.arange(1, PLOT_STEPS + 1)
    for j, b2 in enumerate(BETA2S):
        a = ax[0, j]
        a.plot(t, np.abs(scalar[b2]["step_on"]), "o-", label="|step|, bias correction on")
        a.plot(t, np.abs(scalar[b2]["step_off"]), "s--", label="|step|, off")
        a.set(title=f"Scalar weight, beta2 = {b2}", xlabel="step t", ylabel="|step|")
        a2 = a.twinx()
        a2.plot(t, scalar[b2]["r_analytic"][:PLOT_STEPS], "k:", label="r(t) = off / on")
        a2.axhline(1, color="grey", lw=0.5)
        a2.set_ylabel("r(t)")
        a.legend(loc="upper left", fontsize=8)
        a2.legend(loc="lower right", fontsize=8)
        m, b = model[b2], ax[1, j]
        s = np.arange(1, len(m["mean_on"]) + 1)
        b.plot(s, m["mean_on"], label="on")
        b.fill_between(s, m["lo_on"], m["hi_on"], alpha=0.25)
        b.plot(s, m["mean_off"], label="off")
        b.fill_between(s, m["lo_off"], m["hi_off"], alpha=0.25)
        if m["loss_settle_step"]:
            b.axvline(m["loss_settle_step"], color="k", ls=":", label=f"overlap from step {m['loss_settle_step']}")
        b.set(title=f"Model S, beta2 = {b2}, lr {MODEL_LR}, no warmup", xlabel="step", ylabel="train loss (smoothed)")
        b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "p2_bias_correction.png", dpi=120)
    plt.close(fig)


@logged("p2_bias_correction")
def run():
    device = pick_device()
    dtype = train_dtype(device)
    scalar = scalar_part()
    model = model_part(FineWebData(), device, dtype)
    plot(scalar, model)
    save_json("p2_bias_correction.json", dict(beta1=BETA1, tolerance=TOL, quick=QUICK, model_lr=MODEL_LR, seeds=SEEDS,
                                               scalar={str(k): v for k, v in scalar.items()},
                                               model={str(k): v for k, v in model.items()}))


if __name__ == "__main__":
    run()
