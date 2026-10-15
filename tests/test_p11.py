"""Fast checks for the claims in PLAN.md and the README (CPU only, a few seconds, no FineWeb download needed)."""
import math

import numpy as np
import pytest
import torch

from src.data import HEADER_INTS, MAGIC, VERSION, VOCAB_SIZE, FineWebData, load_shard, make_model
from src.p1_adam_by_hand import BETAS, EPS, GRADS, LR, THETA0, compare, torch_rows
from src.ref_adam import RefAdamW, adam_by_hand, bias_ratio
from src.schedules import lr_at
from src.train import param_groups


def test_hand_adam_matches_torch_float64():
    hand = adam_by_hand(THETA0, GRADS, LR, *BETAS, EPS)
    diff = compare(hand, torch_rows(torch.float64, torch.device("cpu"), foreach=False))
    assert max(d["max_abs"] for d in diff.values()) < 1e-12


@pytest.mark.parametrize("cls,decoupled", [(torch.optim.Adam, False), (torch.optim.AdamW, True)])
def test_hand_weight_decay_matches_torch(cls, decoupled):
    hand = adam_by_hand(THETA0, GRADS, LR, *BETAS, EPS, weight_decay=0.1, decoupled=decoupled)
    diff = compare(hand, torch_rows(torch.float64, torch.device("cpu"), cls, weight_decay=0.1, foreach=False))
    assert max(d["max_abs"] for d in diff.values()) < 1e-12


def test_l2_and_decoupled_differ():
    l2 = adam_by_hand(THETA0, GRADS, LR, *BETAS, EPS, weight_decay=0.1, decoupled=False)[-1]["theta"]
    dec = adam_by_hand(THETA0, GRADS, LR, *BETAS, EPS, weight_decay=0.1, decoupled=True)[-1]["theta"]
    assert abs(l2 - dec) > 1e-4


def test_ref_adamw_matches_torch_adamw():
    torch.manual_seed(0)
    w1 = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    w2 = w1.detach().clone().requires_grad_(True)
    o1 = torch.optim.AdamW([w1], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1, foreach=False)
    o2 = RefAdamW([w2], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    for _ in range(10):
        g = torch.randn(4, 3, dtype=torch.float64)
        w1.grad, w2.grad = g.clone(), g.clone()
        o1.step()
        o2.step()
    assert torch.allclose(w1, w2, atol=1e-12, rtol=0)


@pytest.mark.parametrize("b2", [0.95, 0.999])
def test_bias_ratio_equals_measured(b2):
    on = adam_by_hand(1.0, GRADS * 4, 1e-3, 0.9, b2, 1e-12, bias_correction=True)
    off = adam_by_hand(1.0, GRADS * 4, 1e-3, 0.9, b2, 1e-12, bias_correction=False)
    for t, (a, b) in enumerate(zip(off, on), start=1):
        assert abs(a["step"] / b["step"] - bias_ratio(t, 0.9, b2)) < 1e-6


def test_bias_ratio_table_values():
    assert abs(bias_ratio(1, 0.9, 0.95) - 0.447) < 1e-3
    assert abs(bias_ratio(20, 0.9, 0.95) - 1.097) < 1e-3
    assert abs(bias_ratio(1, 0.9, 0.999) - 3.162) < 1e-3
    assert abs(bias_ratio(20, 0.9, 0.999) - 6.24) < 1e-2


def test_schedules():
    peak, w, total = 1e-3, 30, 300
    assert math.isclose(lr_at(w - 1, "cosine", peak, w, total), peak)
    assert math.isclose(lr_at(w, "cosine", peak, w, total), peak)
    assert math.isclose(lr_at(total - 1, "cosine", peak, w, total), 0.1 * peak)
    assert math.isclose(lr_at(199, "cosine", peak, w, total), 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * 169 / 269)))
    assert all(lr_at(s, "wsd", peak, w, total) == peak for s in range(w, 240))
    assert lr_at(240, "wsd", peak, w, total) == peak and lr_at(241, "wsd", peak, w, total) < peak
    assert math.isclose(lr_at(total - 1, "wsd", peak, w, total), 0.1 * peak)
    assert all(lr_at(s, "constant", peak, w, total) == peak for s in range(w, total))
    for kind in ("cosine", "wsd"):  # no jumps
        lrs = [lr_at(s, kind, peak, w, total) for s in range(total)]
        assert max(abs(a - b) for a, b in zip(lrs, lrs[1:])) < peak / w + 1e-12


def test_wsd_branch_equals_main_run_until_its_decay():
    """Arm C (WSD planned for 200) has the same LR as arm B (WSD planned for 300) up to step 160."""
    for s in range(160):
        assert lr_at(s, "wsd", 1e-3, 30, 200) == lr_at(s, "wsd", 1e-3, 30, 300)
    assert lr_at(199, "wsd", 1e-3, 30, 200) < lr_at(199, "wsd", 1e-3, 30, 300)


def test_weight_decay_groups():
    model = make_model(dict(n_layer=1, n_head=1, n_embd=64, block_size=16, dropout=0.0, bias=True), vocab_size=128)
    decay, no_decay = param_groups(model, 0.1)
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])
    assert no_decay["weight_decay"] == 0.0 and len(no_decay["params"]) > 0
    n_all = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for g in (decay, no_decay) for p in g["params"]) == n_all


def _fake_shard(path, n, seed):
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[:3] = [MAGIC, VERSION, n]
    toks = np.random.default_rng(seed).integers(0, 50257, size=n, dtype=np.uint16)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())
    return toks


def test_loader_reads_shard_and_fixed_eval_batches(tmp_path):
    toks = _fake_shard(tmp_path / "train.bin", 10_000, 0)
    _fake_shard(tmp_path / "val.bin", 5_000, 1)
    data = FineWebData(tmp_path / "train.bin", tmp_path / "val.bin")
    assert data.vocab_size == VOCAB_SIZE and len(data.train) == 10_000
    assert np.array_equal(np.asarray(load_shard(tmp_path / "train.bin")[:50]), toks[:50])
    x, y = data.get_batch(4, 32, gen=torch.Generator().manual_seed(3))
    assert x.shape == y.shape == (4, 32) and x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is x moved one place
    e1, e2 = data.eval_batches(3, 4, 32), data.eval_batches(3, 4, 32)
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(e1, e2))


def test_loader_rejects_bad_header(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(np.zeros(HEADER_INTS + 10, dtype=np.int32).tobytes())
    with pytest.raises(ValueError):
        load_shard(p)
