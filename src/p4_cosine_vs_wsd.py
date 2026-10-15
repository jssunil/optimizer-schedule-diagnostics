"""Part 4: cosine compared with WSD, both planned for 300 steps and stopped at step 200.

Arms (model S, warmup 30, validation loss at step 200 on fixed batches, 3 seeds):
  A  cosine planned for 300, stopped at 200            (the task)
  B  WSD planned for 300 (decay 240-300), stopped at 200 (the task)
  C  WSD planned for 200 (decay 160-200), at B's best LR. Same seed, same data, same LR as B, so it is
     the same run as B up to step 160: a 40-step cooldown branch of B with the same 200-step compute.
  D  cosine planned for 200 (the correct horizon): the reference if the stop had been known.
Both sides are tuned: A, B and D each get the same LR grid (5 values) and the same seeds; a best value at an
edge of the grid is extended (see train.lr_sweep). Each arm is compared at its own best LR.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import CONFIG_S, FineWebData
from src.device import RESULTS, logged, pick_device, save_json, train_dtype
from src.train import QUICK, Run, best_lr, describe, lr_sweep, rolling_mean, train

if QUICK:
    PLAN, STOP, WARMUP, LRS, SEEDS, N_EVAL = 30, 20, 3, [1e-3, 2e-3], [0], 2
else:
    PLAN, STOP, WARMUP, LRS, SEEDS, N_EVAL = 300, 200, 30, [3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3], [0, 1, 2], 40
B_SIZE, T_SIZE = 16, 512

ARMS = {
    "A cosine-300 stopped at 200": dict(schedule="cosine", total=PLAN, stop=STOP),
    "B WSD-300 stopped at 200": dict(schedule="wsd", total=PLAN, stop=STOP),
    "D cosine-200 (known horizon)": dict(schedule="cosine", total=STOP),
}
ARM_C = "C WSD branch: cooldown 160-200"


def make(arm):
    return lambda lr, seed: Run(CONFIG_S, lr, warmup=WARMUP, seed=seed, B=B_SIZE, T=T_SIZE, **arm)


def summary(vals):
    return describe(vals)


def plot(sweeps, best, c_runs, out):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for name, res in sweeps.items():
        lrs = list(res)
        ax[0].plot(lrs, [np.mean([x["val"] for x in res[q]]) for q in lrs], "o-", label=name)
    ax[0].set(xscale="log", xlabel="peak LR", ylabel=f"val loss at step {STOP}", title="LR tuning, each arm")
    ax[0].legend(fontsize=8)
    curves = {name: sweeps[name][best[name]] for name in sweeps}
    curves[ARM_C] = c_runs
    for name, runs in curves.items():
        loss = np.mean([rolling_mean(r["loss"], 10) for r in runs], axis=0)
        ax[1].plot(np.arange(1, len(loss) + 1), loss, label=f"{name} (lr {runs[0]['lr_peak']:.1e})")
        ax[2].plot(np.arange(1, len(runs[0]["lr"]) + 1), runs[0]["lr"], label=name)
    ax[1].set(xlabel="step", ylabel="train loss (smoothed, mean of seeds)", title="At each arm's best LR")
    ax[1].legend(fontsize=8)
    ax[2].set(xlabel="step", ylabel="LR", title="Schedules (at each arm's best peak LR)")
    ax[2].axvline(STOP, color="grey", ls="--")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "p4_cosine_vs_wsd.png", dpi=120)
    plt.close(fig)


@logged("p4_cosine_vs_wsd")
def run():
    device = pick_device()
    dtype = train_dtype(device)
    data = FineWebData()
    evb = data.eval_batches(N_EVAL, B_SIZE, T_SIZE)
    sweeps, best = {}, {}
    for name, arm in ARMS.items():
        print(f"{name}: LR sweep")
        sweeps[name] = lr_sweep(make(arm), LRS, SEEDS, data, device, dtype, evb, STOP, label=name[:1])
        best[name] = best_lr(sweeps[name])
        for x in sweeps[name][best[name]]:
            x["lr_peak"] = best[name]

    lr_b = best["B WSD-300 stopped at 200"]
    print(f"{ARM_C}: at B's best LR {lr_b:.2e}")
    c_runs = []
    for s in SEEDS:
        r = train(make(dict(schedule="wsd", total=STOP))(lr_b, s), data, device, dtype, evb, eval_at={STOP})
        c_runs.append(dict(seed=s, val=r["val"][STOP], loss=r["loss"], lr=r["lr"], lr_peak=lr_b))

    table = {name: dict(best_lr=best[name], **summary([x["val"] for x in sweeps[name][best[name]]]),
                        best_at_grid_edge=best[name] in (min(sweeps[name]), max(sweeps[name])))
             for name in ARMS}
    table[ARM_C] = dict(best_lr=lr_b, **summary([x["val"] for x in c_runs]), best_at_grid_edge=False)
    print(f"\nValidation loss at step {STOP} (each arm at its own best peak LR; range = min..max over seeds):")
    for name, t in sorted(table.items(), key=lambda kv: kv[1]["mean"]):
        print(f"  {name:<32} lr {t['best_lr']:.2e}  mean {t['mean']:.4f}  range {t['min']:.4f}..{t['max']:.4f}")

    a, b = table["A cosine-300 stopped at 200"], table["B WSD-300 stopped at 200"]
    overlap = max(a["min"], b["min"]) <= min(a["max"], b["max"])
    lower = "A (cosine)" if a["mean"] < b["mean"] else "B (WSD)"
    decision = (f"At step {STOP}: A {a['mean']:.4f}, B {b['mean']:.4f}; lower: {lower}"
                + (" (the seed ranges overlap: no clear difference)." if overlap else ".")
                + f" If step {STOP} is final, keep the model with the lower loss. If training continues, keep the WSD "
                  f"checkpoint: it continues at its peak LR, and a {STOP - round(0.8 * STOP)}-step cooldown gives arm C "
                  f"({table[ARM_C]['mean']:.4f}) with the same compute.")
    print("\n" + decision)

    plot(sweeps, best, c_runs, table)
    slim = {name: {f"{lr:.3e}": [dict(seed=x["seed"], val=x["val"]) for x in res[lr]] for lr in res}
            for name, res in sweeps.items()}
    save_json("p4_cosine_vs_wsd.json", dict(quick=QUICK, plan=PLAN, stop=STOP, warmup=WARMUP, seeds=SEEDS,
                                             lr_grid=LRS, table=table, sweeps=slim, decision=decision))


if __name__ == "__main__":
    run()
