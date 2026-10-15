"""Part 3: Update-to-weight ratio per layer.

Train the model with linear warmup. At every step, log ||lr * update|| / ||W|| for every layer.
Find the step at which warmup stops changing the ratio (ratio flattens).
Output: results/p3_update_ratio.png and .json."""
from src.device import logged, save_json


@logged("p3_update_ratio")
def run():
    raise NotImplementedError


if __name__ == "__main__":
    run()
