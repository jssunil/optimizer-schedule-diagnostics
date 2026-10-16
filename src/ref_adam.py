"""Reference Adam, written from Algorithm 1 of Kingma and Ba (2014).

adam_by_hand: one scalar weight in Python floats (float64). It is the "truth" for part 1.
RefAdamW:     a torch optimizer with a bias-correction switch. Part 2 trains with it.
              With bias_correction=True it must equal torch.optim.AdamW (checked in the tests).
"""
import math

import torch


def bias_ratio(t, beta1, beta2):
    """Uncorrected step / corrected step at step t (eps ignored). It does not depend on the gradients."""
    return (1 - beta1 ** t) / math.sqrt(1 - beta2 ** t)


def adam_by_hand(theta0, grads, lr, beta1, beta2, eps, weight_decay=0.0, decoupled=False, bias_correction=True):
    """One scalar weight. Returns one row per step with every value Adam calculates.

    weight_decay with decoupled=False: L2, the term weight_decay * theta is added to the gradient (torch Adam).
    weight_decay with decoupled=True:  theta is multiplied by (1 - lr * weight_decay) first (torch AdamW).
    """
    theta, m, v, rows = float(theta0), 0.0, 0.0, []
    for t, g in enumerate(grads, start=1):
        g_used, before = float(g), theta
        if weight_decay and decoupled:
            theta *= 1 - lr * weight_decay
        elif weight_decay:
            g_used += weight_decay * theta
        m = beta1 * m + (1 - beta1) * g_used
        v = beta2 * v + (1 - beta2) * g_used * g_used
        m_hat = m / (1 - beta1 ** t) if bias_correction else m
        v_hat = v / (1 - beta2 ** t) if bias_correction else v
        theta += -lr * m_hat / (math.sqrt(v_hat) + eps)
        # step = total change of theta in this step (Adam step plus decoupled decay), as torch_rows measures it
        rows.append(dict(t=t, g=float(g), m=m, v=v, m_hat=m_hat, v_hat=v_hat, step=theta - before, theta=theta))
    return rows


class RefAdamW(torch.optim.Optimizer):
    """AdamW with a bias_correction switch. States are kept in fp32 (fp64 for fp64 weights)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, bias_correction=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      bias_correction=bias_correction))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                sdt = torch.promote_types(p.dtype, torch.float32)
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=sdt)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=sdt)
                state["step"] += 1
                t, m, v = state["step"], state["exp_avg"], state["exp_avg_sq"]
                g = p.grad.to(sdt)
                if wd:
                    p.mul_(1 - lr * wd)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                if group["bias_correction"]:
                    m_hat, v_hat = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
                else:
                    m_hat, v_hat = m, v
                p.add_((-lr * m_hat / (v_hat.sqrt() + eps)).to(p.dtype))
