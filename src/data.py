"""FineWeb, GPT-2 BPE tokens (the modded-nanogpt shards, Hugging Face dataset kjj0/fineweb10B-gpt2).

Get the files first: python scripts/get_fineweb.py
Shard format: a header of 256 int32 (magic 20240520, version 1, token count), then the tokens as uint16.
"""
import pathlib

import numpy as np
import torch

from src.model import GPT, GPTConfig

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA / "fineweb_train_000001.bin"  # 100 M tokens
VAL_FILE = DATA / "fineweb_val_000000.bin"  # 100 M tokens
MAGIC, VERSION, HEADER_INTS = 20240520, 1, 256
VOCAB_SIZE = 50304  # GPT-2 has 50,257 tokens; nanoGPT pads to a multiple of 64 for speed

# Head size 64 at every width, so that only the width changes (PLAN.md section 4.1).
_BASE = dict(n_layer=4, block_size=512, dropout=0.0, bias=True)
WIDTHS = {d: dict(_BASE, n_embd=d, n_head=d // 64) for d in (256, 512, 1024)}
CONFIG_S = WIDTHS[256]  # parts 2, 3 and 4


def load_shard(path):
    """Memory-map one shard. Check the header before the tokens are used."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python scripts/get_fineweb.py")
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if header[0] != MAGIC or header[1] != VERSION:
        raise ValueError(f"{path.name}: not a FineWeb GPT-2 shard (magic {header[0]}, version {header[1]})")
    n = int(header[2])
    if path.stat().st_size != HEADER_INTS * 4 + 2 * n:
        raise ValueError(f"{path.name}: size does not match the {n} tokens in the header")
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4, shape=(n,))


class FineWebData:
    vocab_size = VOCAB_SIZE

    def __init__(self, train_file=TRAIN_FILE, val_file=VAL_FILE):
        self.train = load_shard(train_file)
        self.val = load_shard(val_file)

    def get_batch(self, B, T, split="train", gen=None, device="cpu"):
        """B random windows of T+1 tokens: x = the first T, y = the same window moved one place right.
        The windows come from `gen`, so the same seed gives the same data order."""
        src = self.train if split == "train" else self.val
        ix = torch.randint(len(src) - T - 1, (B,), generator=gen).tolist()
        w = torch.from_numpy(np.stack([src[i:i + T + 1] for i in ix]).astype(np.int64))
        return w[:, :-1].to(device), w[:, 1:].to(device)

    def eval_batches(self, n_batches, B, T, seed=0):
        """A fixed list of validation batches (on the CPU). Every run uses the same list."""
        gen = torch.Generator().manual_seed(seed)
        return [self.get_batch(B, T, "val", gen) for _ in range(n_batches)]


def make_model(cfg, vocab_size=VOCAB_SIZE, device="cpu", seed=1337):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=vocab_size, **cfg)).to(device)
