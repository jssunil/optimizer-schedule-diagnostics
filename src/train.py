"""One training function for parts 2-5. Same data order, same evaluation batches and same rules for every run.

AdamW with two parameter groups (nanoGPT rule): weight decay on tensors with 2 or more dimensions only
(the block matrices, the tied token embedding / output layer and the position embedding); no decay on
biases and LayerNorm weights. Gradient clipping at 1.0. bf16 autocast on the GPU.

QUICK=1 (environment) makes every part tiny, for a smoke test of the code path. Its numbers mean nothing.
"""
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from src.data import make_model
from src.device import Governor, autocast
from src.ref_adam import RefAdamW
from src.schedules import lr_at

QUICK = os.environ.get("QUICK") == "1"


@dataclass
class Run:
    model: dict
    peak_lr: float
    schedule: str = "cosine"  # constant | cosine | wsd
    total: int = 300  # planned length of the schedule
    stop: int | None = None  # stop early (part 4); None = total
    warmup: int = 30
    decay_frac: float = 0.2  # wsd only
    decay_shape: str = "linear"  # wsd only
    min_ratio: float = 0.1
    seed: int = 0  # model init and data order
    B: int = 16
    T: int = 512
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    clip: float = 1.0
    optimizer: str = "adamw"  # adamw (torch) | ref (RefAdamW, part 2)
    bias_correction: bool = True  # ref only
    log_ratio: bool = False  # part 3


def param_groups(model, weight_decay):
    decay = [p for _, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.dim() < 2]
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]


def make_optimizer(model, run, device):
    groups = param_groups(model, run.weight_decay)
    if run.optimizer == "ref":
        return RefAdamW(groups, lr=run.peak_lr, betas=run.betas, eps=run.eps, bias_correction=run.bias_correction)
    return torch.optim.AdamW(groups, lr=run.peak_lr, betas=run.betas, eps=run.eps, fused=device.type == "cuda")


@torch.no_grad()
def evaluate(model, batches, device, dtype):
    """Mean validation loss over a fixed list of batches."""
    model.eval()
    total = 0.0
    for x, y in batches:
        with autocast(device, dtype):
            _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    model.train()
    return total / len(batches)


def train(run, data, device, dtype, eval_batches=None, eval_at=()):
    """Train one run. Returns per-step logs and the validation loss at each step in eval_at (1-based)."""
    stop = run.stop or run.total
    model = make_model(run.model, device=device, seed=run.seed)
    opt = make_optimizer(model, run, device)
    gen = torch.Generator().manual_seed(10_000 + run.seed)  # data order: same seed, same batches
    watch = [(n, p) for n, p in model.named_parameters() if p.dim() >= 2] if run.log_ratio else []
    out = dict(run=asdict(run), loss=[], lr=[], grad_norm=[], val={},
               ratio_names=[n for n, _ in watch], ratio=[])
    gov, t0 = Governor(device), time.perf_counter()
    for s in range(stop):
        lr = lr_at(s, run.schedule, run.peak_lr, run.warmup, run.total, run.min_ratio, run.decay_frac, run.decay_shape)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = data.get_batch(run.B, run.T, gen=gen, device=device)
        with gov.step():  # the GPU is busy at most GPU_DUTY of the time
            with autocast(device, dtype):
                _, loss = model(x, y)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), run.clip)  # norm before clipping
            before = [p.detach().clone() for _, p in watch]
            opt.step()
            opt.zero_grad(set_to_none=True)
            if watch:  # update-to-weight ratio: ||W_after - W_before|| / ||W_before|| (includes decay and clipping)
                r = torch.stack([(p.detach() - b).norm() / b.norm() for (_, p), b in zip(watch, before)])
                out["ratio"].append(r.float().cpu().tolist())
        out["loss"].append(loss.item())
        out["lr"].append(lr)
        out["grad_norm"].append(gnorm.item())
        if eval_batches is not None and (s + 1) in eval_at:
            out["val"][s + 1] = evaluate(model, eval_batches, device, dtype)
    out["seconds_busy"], out["seconds_wall"] = gov.busy, time.perf_counter() - t0
    return out


def lr_sweep(make_run, lrs, seeds, data, device, dtype, eval_batches, eval_step, factor=2.0, label=""):
    """Train make_run(lr, seed) for every lr and seed; score = validation loss at eval_step.

    Fairness rule: if the best mean is at an edge of the grid, add one LR past that edge (lr * factor
    or lr / factor) and check again, at most twice. Returns {lr: [one result per seed]} sorted by lr.
    """
    results = {}

    def do(lr):
        results[lr] = []
        for s in seeds:
            r = train(make_run(lr, s), data, device, dtype, eval_batches, eval_at={eval_step})
            results[lr].append(dict(seed=s, val=r["val"][eval_step], loss=r["loss"], lr=r["lr"],
                                    grad_norm_max=max(r["grad_norm"]), seconds_wall=r["seconds_wall"]))
        vals = [x["val"] for x in results[lr]]
        print(f"  {label} lr {lr:.2e}: val loss {np.mean(vals):.4f} (seeds {', '.join(f'{v:.4f}' for v in vals)})")

    for lr in lrs:
        do(lr)
    for _ in range(2):
        grid = sorted(results)
        best = min(grid, key=lambda q: np.mean([x["val"] for x in results[q]]))
        if best == grid[0]:
            do(grid[0] / factor)
        elif best == grid[-1]:
            do(grid[-1] * factor)
        else:
            break
    return dict(sorted(results.items()))


def best_lr(results):
    return min(results, key=lambda q: np.mean([x["val"] for x in results[q]]))


def rolling_mean(xs, w):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < w:
        return xs
    c = np.cumsum(np.insert(xs, 0, 0.0))
    head = [xs[:i + 1].mean() for i in range(w - 1)]
    return np.concatenate([head, (c[w:] - c[:-w]) / w])


def describe(xs):
    xs = np.asarray(xs, dtype=float)
    return dict(mean=float(xs.mean()), min=float(xs.min()), max=float(xs.max()), n=int(len(xs)))
