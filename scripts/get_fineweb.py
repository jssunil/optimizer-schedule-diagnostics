"""Download the FineWeb GPT-2 shards that this project uses (about 400 MB) into data/.

Source: Hugging Face dataset kjj0/fineweb10B-gpt2 (the modded-nanogpt speedrun data).
Usage:  python scripts/get_fineweb.py [number of training shards, default 1]
"""
import pathlib
import sys
import urllib.request

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
URL = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/{}"
SHARD_BYTES = 256 * 4 + 2 * 100_000_000  # header + 100 M uint16 tokens


def get(name):
    path = DATA / name
    if path.exists() and path.stat().st_size == SHARD_BYTES:
        print(f"{name}: already present")
        return
    tmp = path.with_suffix(".part")
    print(f"{name}: downloading 200 MB ...")
    urllib.request.urlretrieve(URL.format(name), tmp)
    if tmp.stat().st_size != SHARD_BYTES:
        raise SystemExit(f"{name}: got {tmp.stat().st_size} bytes, expected {SHARD_BYTES}")
    tmp.replace(path)


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    DATA.mkdir(exist_ok=True)
    get("fineweb_val_000000.bin")
    for i in range(1, n_train + 1):
        get(f"fineweb_train_{i:06d}.bin")


if __name__ == "__main__":
    main()
