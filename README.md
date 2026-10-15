# Optimizer and Schedule Diagnostics (Session 11)

ERA V5 Session 11 assignment: Adam by hand, bias correction, update-to-weight ratio, cosine vs WSD, and a learning-rate sweep across widths. The task text is in [assignment.md](assignment.md).

The model is nanoGPT from [training-loop-diagnostics](https://github.com/jssunil/training-loop-diagnostics) (Session 10). The data is FineWeb web text, pre-tokenized with GPT-2 BPE (`kjj0/fineweb10B-gpt2` on Hugging Face): one 100 M-token training shard and one validation shard.

## Documents

| File | Content |
|---|---|
| [PLAN.md](PLAN.md) | Implementation plan: design options for each part, with advantages, disadvantages and the reason for each choice; fairness protocol; compute budget; tests. |
| [SESSION_11_EXPLAINER.md](SESSION_11_EXPLAINER.md) | Session 11 in depth: Adam, AdamW, bias correction, warmup, cosine and WSD, μP, newer optimizers, fair comparison; 33 checked papers with the key points of each (for the assignment and the interview) and the repos that implement them. |
| [INTERVIEW_QA.md](INTERVIEW_QA.md) | Principal Architect interview questions and model answers based on this project, including PyTorch `Adam`/`AdamW` internals (section K). |

## How to run

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/get_fineweb.py      # about 400 MB into data/
python -m src.p1_adam_by_hand
python -m src.p2_bias_correction
python -m src.p3_update_ratio
python -m src.p4_cosine_vs_wsd
python -m src.p5_lr_sweep
pytest
```

Each part writes its numbers to `results/<part>.json`, its plot to `results/<part>.png` and its log to `results/logs/<part>.log`.

## Part 1: Adam by hand

## Part 2: Bias correction

## Part 3: Update-to-weight ratio during warmup

## Part 4: Cosine vs WSD, stopped at step 200

## Part 5: Learning-rate sweep at widths 256, 512 and 1024

## Fair comparison: how both sides were tuned

## Limits

## References

- Kingma and Ba, Adam: A Method for Stochastic Optimization, arXiv:1412.6980
- Loshchilov and Hutter, Decoupled Weight Decay Regularization (AdamW), arXiv:1711.05101
- Hu et al., MiniCPM (WSD schedule), arXiv:2404.06395
- Yang et al., Tensor Programs V (μP), arXiv:2203.03466
