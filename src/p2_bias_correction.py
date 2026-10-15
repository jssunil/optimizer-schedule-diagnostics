"""Part 2: Bias correction ablation.

Run the first 20 Adam steps with and without bias correction (same weight, same gradients).
Plot step size against step for both. Report the step after which the difference
stops mattering, and state the threshold used (relative difference below 1%).
Output: results/p2_bias_correction.png and .json."""
from src.device import logged, save_json


@logged("p2_bias_correction")
def run():
    raise NotImplementedError


if __name__ == "__main__":
    run()
