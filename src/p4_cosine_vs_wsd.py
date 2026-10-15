"""Part 4: Cosine vs WSD.

Train the same model (same seed, data order and peak LR) for a 300-step schedule,
once with cosine decay and once with warmup-stable-decay. Stop both at step 200.
Tune the peak LR for each schedule separately before comparing.
Output: results/p4_cosine_vs_wsd.png and .json (loss at step 200 for each run)."""
from src.device import logged, save_json


@logged("p4_cosine_vs_wsd")
def run():
    raise NotImplementedError


if __name__ == "__main__":
    run()
