"""Part 5: Learning-rate sweep across widths.

Sweep the LR at widths 256, 512 and 1024. Plot loss against LR (log x-axis) and mark
the three minima. Fit LR_opt against width and predict the value at width 4096,
with a confidence statement.
Output: results/p5_lr_sweep.png and .json."""
from src.device import logged, save_json


@logged("p5_lr_sweep")
def run():
    raise NotImplementedError


if __name__ == "__main__":
    run()
