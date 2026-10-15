# Session 11 Explainer: Optimizers and Learning-Rate Schedules

This document uses ASD-STE100 Simplified Technical English rules: short sentences, active voice, one meaning for each word. The terms are defined in the "Words" table of [PLAN.md](PLAN.md#2-words).

Sources: the course notes (`era-v5-capstone-tracker/sessions/session_11_optimizers_and_learning_rate_schedules.md`) and the published papers in section 13. Each arXiv number in this document was checked on arxiv.org on 15 Oct 2026. The PyTorch details were checked in the local `torch/optim/adam.py` (PyTorch 2.7).

---

## 1. How to study Session 11 with this assignment

The assignment is a map of the session. Each part makes you prove one idea with a measurement.

| Session idea | Assignment part | What you must be able to say after it |
|---|---|---|
| Adam and its two moments | Part 1 | "I can calculate each number Adam stores, and I know where PyTorch differs from the paper." |
| Bias correction | Part 2 | "I know how large the early-step error is, and that its sign depends on β2." |
| Warmup | Part 3 | "I know what warmup does to the size of each update, layer by layer." |
| Cosine and WSD | Part 4 | "I know why WSD lets a lab stop or continue a run without a new run." |
| LR and model size | Part 5 | "I know why a small-model LR is wrong for a large model, and what μP does about it." |
| Fair comparison | All parts | "I do not believe an optimizer claim until both sides are tuned." |

**Study procedure (about 6 hours with the assignment):**
1. Read the course notes once. Mark each formula.
2. Read Adam §2 and §3 (Kingma and Ba). Do part 1 by hand on paper first.
3. Read the r(t) section of this document. Do part 2.
4. Read section 5 (warmup) of this document. Do part 3.
5. Read the WSD section of MiniCPM and the cooldown results of Hägele et al. Do part 4.
6. Read Tensor Programs V §1–3 (μP). Do part 5.
7. Answer the questions in [INTERVIEW_QA.md](INTERVIEW_QA.md) without notes.
8. Read the corrections in section 12. They show where the course notes need care.

---

## 2. Why the optimizer matters

An optimizer changes the weights to reduce the loss. Plain stochastic gradient descent (SGD) uses one step size for every weight:

$$\theta_t = \theta_{t-1} - \eta\, g_t$$

In a transformer, different weights have very different gradient sizes. Embedding rows of rare tokens get small and rare gradients. Some attention weights get large gradients. One step size cannot be correct for all of them. SGD is then too slow for some weights and unstable for others.

Zhao et al. (2024) compared SGD, Adam, Adafactor, Lion and signed momentum on language models. With good tuning, all of them except SGD gave similar results. SGD was clearly worse. Thus, the important property is a step size that adapts to each weight.

---

## 3. Adam

### 3.1 The algorithm

Adam (Kingma and Ba, 2014) keeps two moving averages for each weight.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

- m is momentum. It averages the direction over about 1/(1 − β1) = 10 steps.
- v measures the typical gradient size. It averages over about 1/(1 − β2) steps.
- The step is m̂/√v̂. This ratio is about ±1 when the gradient has a steady sign, and smaller when the sign changes often.

**The step size does not depend on the gradient scale.** If all gradients are multiplied by 100, m̂/√v̂ does not change. Thus, η sets the step in weight units directly. This is why the update-to-weight ratio in part 3 follows η so closely.

### 3.2 Bias correction

m_0 = 0 and v_0 = 0. After step 1, m_1 = 0.1 · g_1, ten times too small. v_1 = (1 − β2) · g_1², too small by 1/(1 − β2). The division by (1 − β^t) corrects this.

Without correction, the step is wrong by the factor

$$r(t) = \frac{1-\beta_1^t}{\sqrt{1-\beta_2^t}}$$

This factor depends only on β1, β2 and t. It does not depend on the gradients.

| t | β2 = 0.95 | β2 = 0.999 |
|---|---|---|
| 1 | 0.45 (step too small) | 3.16 (step too large) |
| 10 | 1.03 | 6.53 |
| 20 | 1.10 | 6.24 |
| \|r − 1\| < 1% from | t ≈ 77 | t ≈ 3,900 |

With the paper default β2 = 0.999, no correction gives steps up to 6.6× too large in the first 20 steps. This can damage a new model. With β2 = 0.95, the error is smaller and changes sign near step 9.

Bias correction and warmup do related work. Both protect the first steps. Warmup also hides much of the bias-correction error, because η is small in the first steps.

### 3.3 ε and its position

Algorithm 1 adds ε to √v̂. PyTorch does the same: `denom = sqrt(v) / sqrt(1 − β2^t) + ε`. The Adam paper also gives a faster form with a different constant ε̂, added to √v (not √v̂). Some libraries use the faster form. Thus, the same ε value can give different steps in different libraries.

ε matters at scale. When the width grows, each gradient gets smaller. Then ε can become large compared with √v̂, and the step shrinks without warning. Wortsman et al. (2023) show this effect and reduce ε (for example to 1e-15). Everett et al. (2024) remove ε fully with "Adam-atan2".

### 3.4 β2: 0.999 or 0.95

The memory of v is about 1/(1 − β2) steps: 1,000 steps for 0.999, 20 steps for 0.95.

- A long memory gives a stable v, but v reacts late to a sudden gradient increase. Then m̂/√v̂ becomes very large for some steps. This can start a loss spike.
- A short memory reacts fast. The cost is a noisier step.

LLM pretraining uses β2 = 0.95 (GPT-3, LLaMA, nanoGPT `train.py`). The data mix changes and gradient spikes occur. A fast v is safer. nanoGPT uses β2 = 0.99 for the small Shakespeare config, because each step sees few tokens and the gradient is noisy.

### 3.5 Convergence note

Reddi et al. (2018/2019) found an error in the convergence proof of Adam. They gave a simple problem where Adam does not converge, and proposed AMSGrad (keep the maximum v). In LLM practice, AMSGrad gives no clear gain. PyTorch has it as `amsgrad=True`. Know the result, but do not expect it to change your training.

---

## 4. AdamW and weight decay

### 4.1 The problem with L2 in Adam

L2 regularization adds λθ to the gradient. In Adam, this term then goes into m and is divided by √v̂. A weight with large gradients (large v̂) gets a small decay. A weight with small gradients gets a large decay. The regularization is then not uniform.

### 4.2 Decoupled weight decay

AdamW (Loshchilov and Hutter, 2017) applies the decay directly to the weight:

$$\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

In PyTorch, `AdamW` does `param.mul_(1 − lr · weight_decay)` before the Adam step. Thus, the decay per step is η_t · λ. It follows the schedule. The PyTorch default λ is 0.01. LLM pretraining usually uses λ = 0.1.

### 4.3 Which weights to decay

Decay 2-D weight matrices only. Do not decay biases, LayerNorm or RMSNorm gains. These 1-D weights set the scale of a whole layer. Decay pulls them toward 0 and makes the signal smaller, with no regularization gain. nanoGPT `configure_optimizers` uses "dim ≥ 2" as the rule.

### 4.4 What weight decay really does in LLMs

In a long pretraining run, the model sees each token about once. Overfitting is not the main problem. Kosson et al. (2023) show a different role: with normalization layers, weight decay and the gradient updates reach a balance. The weight norm becomes stable, and each update rotates the weight by a stable angle. For AdamW they give this equilibrium angle as about √(2ηλ) per step. Thus, η and λ together set the effective learning speed of each layer. This is why part 3 sees ρ settle after warmup.

---

## 5. Warmup

### 5.1 What warmup is

Warmup increases η from near 0 to the peak over the first steps. The course notes give 1–2% of the run.

### 5.2 Why warmup works: four views

| View | Source | Main idea |
|---|---|---|
| Large batch SGD | Goyal et al. (2017) | Linear scaling of η with the batch needs a gradual warmup, or training fails early. |
| Variance of Adam's step | Liu et al. (2019), RAdam | Early v̂ uses few samples. Its variance is high. Warmup keeps the steps small until v̂ is reliable. |
| Simple rule | Ma and Yarats (2019) | Linear warmup over 2/(1 − β2) steps is enough. That is 40 steps for β2 = 0.95, and 2,000 steps for β2 = 0.999. |
| Curvature | Kalra and Barkeshli (2024) | Warmup moves the model to flatter regions. Then a larger peak η is stable. The main gain is a higher usable peak η. |

The views agree on the practical result. Warmup lets you use a larger peak η without an early failure.

### 5.3 Warmup and the update-to-weight ratio

During warmup, the update is about η_t times a value near 1 per element. Thus, ρ = ‖ΔW‖/‖W‖ grows with η_t. After warmup, ρ moves toward its equilibrium value. Part 3 measures this.

A guide value is ρ ≈ 1e-3 per step (Karpathy, cs231n). Layers far above it can be unstable. Layers far below it learn slowly. Muon and Moonlight (Liu et al., 2025) set the update size of each matrix on purpose, to match the per-element RMS of AdamW.

---

## 6. Learning-rate schedules

### 6.1 Cosine

Cosine decay was popular after SGDR (Loshchilov and Hutter, 2016). After warmup:

$$\eta_t = \eta_{min} + (\eta_{peak} - \eta_{min}) \cdot \tfrac{1}{2}\left(1 + \cos\left(\pi \frac{t - t_w}{T - t_w}\right)\right)$$

The weakness: T, the last step, is part of the formula. Chinchilla (Hoffmann et al., 2022) found that a cosine longer than the run gives a worse model. Thus, one cosine run gives a good model for one length only. Each length needs a new run.

### 6.2 WSD (Warmup-Stable-Decay)

MiniCPM (Hu et al., 2024) made WSD popular. The schedule has three phases: warmup, a long constant phase at η_peak, and a short decay at the end.

- **Continue:** a stable-phase checkpoint can continue at the same η. No restart is needed.
- **Branch:** any stable-phase checkpoint can be decayed in a short branch. You get a finished model at that length.
- **Data:** MiniCPM adds high-quality data in the decay phase. The decay phase has a strong effect on the final model.

Hägele et al. (2024) tested WSD at scale. A cooldown of 10–20% of the run matches cosine. A (1 − sqrt) decay shape was slightly better than linear. They used this to measure scaling laws with fewer runs: one long run with several cooldown branches replaces many cosine runs.

DeepSeek LLM (2024) uses a similar multi-step schedule: η drops in two steps near the end. They report a result close to cosine, with the gain of reuse.

### 6.3 Why the loss drops in the decay phase

Wen et al. (2024) describe a "river valley" loss surface. The valley is steep on the sides and nearly flat along the river. At a high η, the model moves fast along the river but bounces between the sides. The loss looks high. In the decay phase, the bounce stops and the model drops to the river floor. The loss falls fast. Thus, a high stable-phase loss does not mean bad progress. The progress is hidden until the decay.

This explains the step-200 result of part 4. WSD at step 200 is still bouncing. Its loss is higher, but its position along the river can be as good or better.

### 6.4 Decay to where?

The course notes decay to 0.1 · η_peak. Bergsma et al. (2025) show that a linear decay to zero is often better for LLMs, more so when the run uses many tokens per weight. Their explanation: the weights act like an average of recent updates, and a decay to zero averages over more updates.

### 6.5 No schedule at all

Schedule-free optimizers (Defazio et al., 2024) do not need T. They use an average of the iterates. They won a self-tuning track of the AlgoPerf benchmark. They are an alternative to WSD when the run length is unknown.

### 6.6 Weight averaging and EMA

An exponential moving average (EMA) of the weights keeps a second copy:

$$\theta_{EMA} \leftarrow \alpha\, \theta_{EMA} + (1-\alpha)\, \theta_t$$

An average of weights acts like a short decay: it removes the bounce. Sanyal et al. (2023, LAWA) show that averaging checkpoints during a high-η phase gives a better model early. The cost is one extra copy of the weights. On a small budget, keep the EMA on the CPU or average a few saved checkpoints.

---

## 7. Learning rate, model size and batch size

### 7.1 LR changes with width

In the standard parameterization (SP), the best η usually goes down as the width goes up. Kaplan et al. (2020) also note that larger models need a smaller η. Part 5 measures this on three widths.

### 7.2 μP

Yang et al. (2022), Tensor Programs V, change the initialization and the per-layer η so that the training behaviour does not change with width. Then the best η at width 256 is also good at width 4096. They call this zero-shot hyperparameter transfer ("μTransfer"). The `microsoft/mup` package and `EleutherAI/nanoGPT-mup` implement it.

For Adam, the main μP rules are: hidden-matrix η scales as 1/width, and the output layer gets an extra 1/width factor in its forward pass. Everett et al. (2024) show that SP with tuned per-layer η can also transfer. The key is per-layer η, not only the initialization.

MiniCPM uses μP to tune on small models and then trains the large model. They call the small tuning runs a "model wind tunnel".

### 7.3 Batch size

- Goyal et al. (2017): for SGD, scale η linearly with the batch size, and use warmup.
- Malladi et al. (2022): for Adam, a square-root rule (η ∝ √batch) fits better.
- McCandlish et al. (2018): above the "critical batch size", a larger batch does not reduce the number of steps. The critical size grows during training.

Gradient accumulation does not change these rules. The optimizer sees one step per full batch (see P10, part 3).

---

## 8. Newer optimizers

| Optimizer | Idea | Memory per weight (state) | Note |
|---|---|---|---|
| AdamW | Two moments, decoupled decay | m, v (2 values) | The baseline. |
| Adafactor (Shazeer and Stern, 2018) | Factor v into a row and a column vector | Much less than 2 values | Used in T5. Saves memory. |
| 8-bit Adam (Dettmers et al., 2021) | Store m and v in 8 bits, block-wise | 2 values in 8 bits | About 75% less optimizer memory. `bitsandbytes`. |
| Lion (Chen et al., 2023) | Sign of an interpolated momentum | m only | Needs a 3–10× smaller η and a larger λ. |
| SOAP (Vyas et al., 2024) | Adam in the eigenbasis of Shampoo's preconditioner | More than Adam | Fewer steps; more compute per step. |
| Muon (Jordan, 2024; Liu et al., 2025) | Orthogonalize the momentum of each 2-D matrix (Newton–Schulz) | m only | Strong in the modded-nanogpt speedruns. Moonlight reports about 2× compute efficiency over AdamW at scale. Use AdamW for embeddings, gains and the output layer. |

**Memory for a frugal budget.** Mixed-precision AdamW needs about 16 bytes per weight: bf16 weight (2), bf16 gradient (2), fp32 master weight (4), m (4), v (4). The optimizer part is 12 of the 16 bytes. 8-bit states, Adafactor, or ZeRO sharding (Session 12) reduce this.

---

## 9. Fair comparison of optimizers

The last line of the assignment is the most important for a principal role.

- Choi et al. (2019): optimizer rankings change with the tuning protocol. When all hyperparameters are tuned, a more general optimizer is never worse than its special case. Many "A beats B" results came from tuning only A.
- Schmidt et al. (2020): across many tasks, no optimizer was clearly best. Tuning mattered as much as the choice.
- Kaddour et al. (2023): many "efficient training" methods lost their gain when the baseline used a fully decayed LR schedule with the same compute.
- Dahl et al. (2023), AlgoPerf: a benchmark with fixed rules for tuning and time-to-result. It is the reference method for fair optimizer comparisons.

Rules for this project: same tuning budget, same seeds, same data, validation loss, report the noise band, no best value at the edge of a grid (PLAN.md section 10).

---

## 10. What this means for V5 on a frugal budget

| Decision | Recommendation | Reason |
|---|---|---|
| Optimizer | AdamW, β1 = 0.9, β2 = 0.95, λ = 0.1, 2-D weights only | Known to work. Every tool supports it. Lowest risk. |
| ε | 1e-8 or smaller; check √v̂ against ε at the real width | Large width gives small gradients (Wortsman et al.). |
| Schedule | WSD with a 10–20% cooldown | One run gives many checkpoints to continue or branch. A late budget change does not need a new run. |
| Cooldown data | Put the best data in the cooldown | MiniCPM shows a strong effect. |
| LR choice | μP or per-layer LR rules; tune on small proxies; confirm with one short run at full width | A sweep at full width costs too much. |
| Monitoring | Log grad norm, update-to-weight ratio per layer, and ε share | Early signs of a spike (Session 15). |
| Weight averaging | Average the last few checkpoints of the stable phase | Cheap. Gives a better model without a cooldown run. |
| New optimizers | Test Muon only on a small proxy with both sides tuned | A possible 2× gain, but a risk to the main run. |

---

## 11. GitHub repositories that implement Session 11

| Repo | What to read |
|---|---|
| `karpathy/nanoGPT` | `model.py` `configure_optimizers` (decay groups, fused AdamW); `train.py` `get_lr` (warmup + cosine). |
| `KellerJordan/modded-nanogpt` | Speedrun records. Muon for hidden matrices, Adam for the rest; WSD-like schedule. |
| `KellerJordan/Muon` | The Muon optimizer and its Newton–Schulz step. |
| `MoonshotAI/Moonlight` | Muon at scale, with weight decay and update-RMS matching. |
| `microsoft/mup` | μP for PyTorch: `MuReadout`, `MuAdamW`, coordinate check tests. |
| `EleutherAI/nanoGPT-mup` | μP in nanoGPT, small and readable. |
| `OpenBMB/MiniCPM` | WSD in a real model release. |
| `epfml/schedules-and-scaling` | Code for Hägele et al.: cosine compared with WSD and cooldown branches. |
| `facebookresearch/schedule_free` | Schedule-free AdamW. |
| `mlcommons/algorithmic-efficiency` | AlgoPerf rules and workloads. |
| `nikhilvyas/SOAP` | The SOAP optimizer. |
| `bitsandbytes-foundation/bitsandbytes` | 8-bit Adam. |
| PyTorch `torch/optim/adam.py`, `adamw.py` | The single-tensor, foreach and fused paths. |

Articles: Karpathy, "A Recipe for Training Neural Networks" (2019 blog); Stanford cs231n notes on the update-to-weight ratio; Keller Jordan, "Muon: An optimizer for hidden layers in neural networks" (2024 blog).

---

## 12. Corrections and cautions for the course notes

| Course note | Correction |
|---|---|
| MiniCPM paper title "…with WSD Schedule" | The title is "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies" (arXiv:2404.06395). |
| WSD gives "identical or superior perplexity compared to cosine" | Hägele et al. find that WSD *matches* cosine with a good cooldown. "Superior" is not a general result. |
| EMA "consistently improving out-of-distribution robustness" | Averaging often helps, but "consistently" is too strong for LLM pretraining. Measure it. |
| Cosine decays "to 0.1 × η_max" | This is common, but Bergsma et al. (2025) show that decay to 0 is often better. |
| β2 = 0.95 "adapts to non-stationary data" | Correct. Also note the cost: a noisier step. |

---

## 13. Reading list (checked)

**Adam and AdamW**
1. Kingma, Ba. *Adam: A Method for Stochastic Optimization.* arXiv:1412.6980 (2014). Read §2 and §3.
2. Loshchilov, Hutter. *Decoupled Weight Decay Regularization.* arXiv:1711.05101 (2017).
3. Reddi, Kale, Kumar. *On the Convergence of Adam and Beyond.* arXiv:1904.09237.

**Warmup**
4. Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv:1706.02677 (2017).
5. Liu et al. *On the Variance of the Adaptive Learning Rate and Beyond* (RAdam). arXiv:1908.03265 (2019).
6. Ma, Yarats. *On the adequacy of untuned warmup for adaptive optimization.* arXiv:1910.04209 (2019).
7. Kalra, Barkeshli. *Why Warmup the Learning Rate? Underlying Mechanisms and Improvements.* arXiv:2406.09405 (2024).
8. Kosson, Messmer, Jaggi. *Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks.* arXiv:2305.17212 (2023).

**Schedules and averaging**
9. Loshchilov, Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* arXiv:1608.03983 (2016).
10. Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* arXiv:2404.06395 (2024).
11. Hägele et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* arXiv:2405.18392 (2024).
12. Wen et al. *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective.* arXiv:2410.05192 (2024).
13. DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* arXiv:2401.02954 (2024).
14. Bergsma et al. *Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs.* arXiv:2502.15938 (2025).
15. Defazio et al. *The Road Less Scheduled.* arXiv:2405.15682 (2024).
16. Sanyal et al. *Early Weight Averaging meets High Learning Rates for LLM Pre-training.* arXiv:2306.03241 (2023).

**Scale, width and batch**
17. Kaplan et al. *Scaling Laws for Neural Language Models.* arXiv:2001.08361 (2020).
18. Hoffmann et al. *Training Compute-Optimal Large Language Models.* arXiv:2203.15556 (2022).
19. Yang et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* arXiv:2203.03466 (2022).
20. Everett et al. *Scaling Exponents Across Parameterizations and Optimizers.* arXiv:2407.05872 (2024).
21. Wortsman et al. *Small-scale proxies for large-scale Transformer training instabilities.* arXiv:2309.14322 (2023).
22. McCandlish et al. *An Empirical Model of Large-Batch Training.* arXiv:1812.06162 (2018).
23. Malladi et al. *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms.* arXiv:2205.10287 (2022).

**Fair comparison**
24. Choi et al. *On Empirical Comparisons of Optimizers for Deep Learning.* arXiv:1910.05446 (2019).
25. Schmidt, Schneider, Hennig. *Descending through a Crowded Valley – Benchmarking Deep Learning Optimizers.* arXiv:2007.01547 (2020).
26. Dahl et al. *Benchmarking Neural Network Training Algorithms* (AlgoPerf). arXiv:2306.07179 (2023).
27. Kaddour et al. *No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models.* arXiv:2307.06440 (2023).
28. Zhao et al. *Deconstructing What Makes a Good Optimizer for Language Models.* arXiv:2407.07972 (2024).

**Other optimizers**
29. Shazeer, Stern. *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.* arXiv:1804.04235 (2018).
30. Dettmers et al. *8-bit Optimizers via Block-wise Quantization.* arXiv:2110.02861 (2021).
31. Chen et al. *Symbolic Discovery of Optimization Algorithms* (Lion). arXiv:2302.06675 (2023).
32. Vyas et al. *SOAP: Improving and Stabilizing Shampoo using Adam.* arXiv:2409.11321 (2024).
33. Liu et al. *Muon is Scalable for LLM Training.* arXiv:2502.16982 (2025).

**Priority for 16 Oct (about 2 hours):** 1 (§2–3), 10 (WSD section), 11 (cooldown results), 19 (§1–3). Read the others after the form.
