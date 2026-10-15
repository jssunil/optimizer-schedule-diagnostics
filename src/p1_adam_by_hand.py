"""Part 1: Adam by hand.

One weight, five fixed gradients. Compute m, v, m_hat, v_hat and the step by hand
for each of the five steps, then run torch.optim.Adam on the same weight and gradients.
Output: results/p1_adam_by_hand.json (per-step table, max abs difference per quantity)."""
from src.device import logged, save_json


@logged("p1_adam_by_hand")
def run():
    raise NotImplementedError


if __name__ == "__main__":
    run()
