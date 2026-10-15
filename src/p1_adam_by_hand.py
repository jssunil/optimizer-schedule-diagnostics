"""Part 1: Adam by hand.

One weight, five fixed gradients. The hand values (Python floats, float64) are the truth.
torch.optim.Adam must give the same m, v, m_hat, v_hat, step and weight:
  float64 paths: max abs difference < 1e-12
  float32 paths: max relative difference < 1e-6
The for-loop, foreach and fused paths are all checked (a path that this device does not support is reported).
Extra table: lambda = 0.1 with Adam (L2) and AdamW (decoupled), each against its own hand calculation.
"""
import torch

from src.device import logged, pick_device, save_json
from src.ref_adam import adam_by_hand

GRADS = [0.5, -1.0, 2.0, 0.01, -0.3]  # sign change, large after small, very small
THETA0, LR, BETAS, EPS = 1.0, 0.1, (0.9, 0.999), 1e-8
KEYS = ["m", "v", "m_hat", "v_hat", "step", "theta"]


def torch_rows(dtype, device, opt_cls=torch.optim.Adam, **kw):
    """Run a torch optimizer on one weight with the fixed gradients. m_hat and v_hat are derived from the state."""
    w = torch.tensor([THETA0], dtype=dtype, device=device, requires_grad=True)
    opt = opt_cls([w], lr=LR, betas=BETAS, eps=EPS, **kw)
    rows = []
    for g in GRADS:
        w.grad = torch.tensor([g], dtype=dtype, device=device)
        before = w.item()
        opt.step()
        st = opt.state[w]
        t = int(st["step"].item())
        m, v = st["exp_avg"].item(), st["exp_avg_sq"].item()
        rows.append(dict(t=t, m=m, v=v, m_hat=m / (1 - BETAS[0] ** t), v_hat=v / (1 - BETAS[1] ** t),
                         step=w.item() - before, theta=w.item()))
    return rows


def compare(hand, got):
    """Largest absolute and relative difference for each value, over the five steps."""
    res = {}
    for k in KEYS:
        ab = max(abs(h[k] - g[k]) for h, g in zip(hand, got))
        rel = max(abs(h[k] - g[k]) / max(abs(h[k]), 1e-30) for h, g in zip(hand, got))
        res[k] = dict(max_abs=ab, max_rel=rel)
    return res


@logged("p1_adam_by_hand")
def run():
    device = pick_device()
    hand = adam_by_hand(THETA0, GRADS, LR, *BETAS, EPS)
    print(f"Adam by hand: theta0 {THETA0}, lr {LR}, betas {BETAS}, eps {EPS}, gradients {GRADS}\n")
    print(f"{'t':>2} {'g':>6} {'m':>12} {'v':>12} {'m_hat':>12} {'v_hat':>12} {'step':>12} {'theta':>12}")
    for r in hand:
        print(f"{r['t']:>2} {r['g']:>6} {r['m']:>12.8f} {r['v']:>12.8f} {r['m_hat']:>12.8f} "
              f"{r['v_hat']:>12.8f} {r['step']:>12.8f} {r['theta']:>12.8f}")
    print()

    paths = [("for-loop", "cpu", dict(foreach=False)), ("foreach", str(device), dict(foreach=True)),
             ("fused", str(device), dict(fused=True))]
    checks = []
    for dtype, limit_key, limit in [(torch.float64, "max_abs", 1e-12), (torch.float32, "max_rel", 1e-6)]:
        for name, dev, kw in paths:
            try:
                diff = compare(hand, torch_rows(dtype, torch.device(dev), **kw))
            except (RuntimeError, ValueError, AssertionError, TypeError) as e:
                msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                checks.append(dict(path=name, device=dev, dtype=str(dtype), error=msg))
                print(f"{name:>8} {dev:>6} {str(dtype):>13}: not available here ({msg[:80]})")
                continue
            worst = max(d[limit_key] for d in diff.values())
            ok = worst < limit
            checks.append(dict(path=name, device=dev, dtype=str(dtype), diff=diff, worst=worst,
                               limit=f"{limit_key} < {limit}", passed=ok))
            print(f"{name:>8} {dev:>6} {str(dtype):>13}: worst {limit_key} {worst:.2e} "
                  f"(limit {limit:.0e}) -> {'PASS' if ok else 'FAIL'}")

    print("\nWeight decay 0.1: L2 (torch Adam) and decoupled (torch AdamW), float64, for-loop path")
    decay = {}
    for label, cls, decoupled in [("Adam weight_decay=0.1 (L2)", torch.optim.Adam, False),
                                  ("AdamW weight_decay=0.1 (decoupled)", torch.optim.AdamW, True)]:
        h = adam_by_hand(THETA0, GRADS, LR, *BETAS, EPS, weight_decay=0.1, decoupled=decoupled)
        g = torch_rows(torch.float64, torch.device("cpu"), cls, weight_decay=0.1, foreach=False)
        worst = max(d["max_abs"] for d in compare(h, g).values())
        decay[label] = dict(theta_after_5=h[-1]["theta"], worst_abs_diff_vs_torch=worst)
        print(f"  {label:<36} theta after 5 steps {h[-1]['theta']:.10f}  (torch differs by {worst:.1e})")
    print("  The two thetas differ: L2 goes through m and v and is divided by sqrt(v_hat); decoupled decay does not.")

    passed = all(c.get("passed", True) for c in checks)
    save_json("p1_adam_by_hand.json", dict(setup=dict(theta0=THETA0, lr=LR, betas=BETAS, eps=EPS, grads=GRADS),
                                           hand=hand, checks=checks, weight_decay=decay, all_passed=passed))
    print(f"\nAll available paths pass: {passed}")
    return passed


if __name__ == "__main__":
    run()
