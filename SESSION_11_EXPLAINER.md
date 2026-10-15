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

Zhao et al. (2024) compared SGD, Adafactor, Adam, Lion and Sophia on language models. With good tuning, all of them except SGD gave similar results. SGD was clearly worse. Thus, the important property is a step size that adapts to each weight.

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

### 4.5 PyTorch `Adam` and `AdamW`: the exact update that the code runs

Sources: the PyTorch 2.14 documentation pages for `torch.optim.Adam` and `torch.optim.AdamW`, and the installed source `torch/optim/adam.py` and `adamw.py` (PyTorch 2.7.0, used by this project). The algorithm is the same in both versions.

**Signatures and defaults**

| Argument | `Adam` | `AdamW` | Meaning |
|---|---|---|---|
| `lr` | 1e-3 | 1e-3 | γ. A float, or a tensor (a tensor only with `fused=True` or `capturable=True`). |
| `betas` | (0.9, 0.999) | (0.9, 0.999) | β1, β2. A float or a 1-element tensor. |
| `eps` | 1e-8 | 1e-8 | ε, added to √v̂. |
| `weight_decay` | **0** | **0.01** | λ. In `Adam` it is an L2 penalty. In `AdamW` it is decoupled decay. |
| `amsgrad` | False | False | Use the maximum of v (Reddi et al.). |
| `maximize` | False | False | Use −g, to maximize the objective. |
| `foreach` | None | None | Group the tensors in one kernel call per operation. |
| `fused` | None | None | Do the full update in one kernel. |
| `capturable` | False | False | Safe for CUDA graph capture. |
| `differentiable` | False | False | Let autograd go through the step. |
| `decoupled_weight_decay` | False | (always True) | `Adam(..., decoupled_weight_decay=True)` is AdamW. |

In PyTorch 2.7, `AdamW` is a subclass of `Adam`. It only sets `decoupled_weight_decay=True`. The same `step()` code runs for both.

**The algorithm (PyTorch documentation)**

For each step t = 1, 2, …:
1. g_t = ∇f(θ_{t−1}). If `maximize`, use −g_t.
2. Weight decay:
   - `Adam` with λ ≠ 0: g_t ← g_t + λθ_{t−1}. The decay goes into m and v.
   - `AdamW`: θ ← θ_{t−1} − γλθ_{t−1}. This occurs **before** the moment updates, and the decay does not go into m or v.
3. m_t = β1·m_{t−1} + (1 − β1)·g_t.
4. v_t = β2·v_{t−1} + (1 − β2)·g_t².
5. m̂_t = m_t / (1 − β1^t).
6. v̂_t = v_t / (1 − β2^t). With `amsgrad`, use the running maximum of v_t first.
7. θ_t = θ − γ·m̂_t / (√v̂_t + ε).

**What the code does (single-tensor path, `adam.py`)**

| Step | Code | Note |
|---|---|---|
| Step counter | `step_t += 1` | `state["step"]` is a scalar tensor. It is on the CPU, unless `fused` or `capturable` is set. |
| AdamW decay | `param.mul_(1 - lr * weight_decay)` | The decay uses the **current** lr. Thus, it follows the schedule. |
| Adam L2 | `grad = grad.add(param, alpha=weight_decay)` | The L2 term enters m and v. |
| m | `exp_avg.lerp_(grad, 1 - beta1)` | m + (1 − β1)(g − m). Same algebra as the formula. Different rounding. |
| v | `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)` | |
| Bias factors | `bias_correction1 = 1 - beta1 ** step`, same for β2 | Calculated as Python floats from the step value. |
| Update | `denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(eps)`, then `param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)` | Same as Algorithm 1: ε is added to √v̂. |

**State for each parameter:** `step`, `exp_avg` (m) and `exp_avg_sq` (v). With `amsgrad`, also `max_exp_avg_sq`. The code makes m and v with `torch.zeros_like(param)`. Thus, m and v have the **same data type as the parameter**. If the parameters are bf16, the moments are also bf16. This loses precision. Keep fp32 master weights, or use an optimizer that keeps fp32 states.

**Which code path runs**

| Setting | Path |
|---|---|
| `foreach=None, fused=None`, all tensors on CUDA | foreach (the default on GPU) |
| `fused=True` | fused (float64, float32, float16 and bfloat16 are supported) |
| `foreach=False` or `fused=False` | single-tensor for-loop |
| `differentiable=True` | single-tensor for-loop |

The documentation says that fused is "theoretically fastest", but it is not the default yet: "the fused implementation is relatively new, we want to give it sufficient bake-in time." foreach uses about one extra copy of the parameters in peak memory.

**Other documented rules**
- `Adam` does not accept sparse gradients. Use `SparseAdam`.
- Call `optimizer.load_state_dict()` after you make the LR scheduler. If you call it before, the scheduler overwrites the loaded learning rates.
- `zero_grad()` sets gradients to `None` by default (`set_to_none=True`). This saves memory. A `None` gradient skips the parameter. A zero gradient does not: Adam still updates m, v and the weight.

**Consequences for this project**
1. Part 1 can compare the hand values with all three paths. All three must agree with Algorithm 1.
2. In an LR sweep with `AdamW`, the decay per step is γλ. A sweep over γ with a fixed λ also changes the decay strength. Report λ, and keep it fixed across the sweep (part 4 and part 5). Some teams keep γλ fixed instead. State the choice.
3. A zero gradient still moves a weight in Adam, because m is not zero. The rows of rare tokens in the embedding keep moving after their gradient stops. This is one reason why part 3 reports the embeddings apart from the block matrices.

---

## 5. Warmup

### 5.1 What warmup is

Warmup increases η from near 0 to the peak over the first steps. The course notes give 1–2% of the run.

### 5.2 Why warmup works: four views

| View | Source | Main idea |
|---|---|---|
| Large batch SGD | Goyal et al. (2017) | Linear scaling of η with the batch needs a gradual warmup, or training fails early. |
| Variance of Adam's step | Liu et al. (2019), RAdam | Early v̂ uses few samples. Its variance is high. Warmup keeps the steps small until v̂ is reliable. |
| Update size, and a simple rule | Ma and Yarats (2019) | They refute the RAdam variance analysis. The real problem is the large size of the early update. Linear warmup over 2/(1 − β2) steps is enough: 40 steps for β2 = 0.95, 2,000 steps for β2 = 0.999. |
| Curvature | Kalra and Barkeshli (2024) | Warmup moves the model to flatter regions. Then a larger peak η is stable. The main gain is a higher usable peak η. |

The views do not agree on the cause. They agree on the practical result: warmup lets you use a larger peak η without an early failure.

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

**Priority for 16 Oct:** see section 15 (reading order).

---

## 14. Key points from each paper

For each paper: the **key points**, the **use** in the assignment (P1–P5 = part 1 to part 5) or in the session, and one **interview line** that you can say aloud. The key points come from the paper abstracts (checked on arxiv.org) and the main results. The numbers match the paper numbers in section 13.

### 14.1 Adam and AdamW

**1. Adam (Kingma and Ba, 2014)**
- Adam keeps moving averages of the gradient (m) and of the squared gradient (v). It adapts the step size of each weight.
- The step does not change when all gradients are multiplied by one constant ("invariant to diagonal rescaling").
- §3 gives bias correction, because m and v start at zero.
- §2 also gives a faster form of the step with a different ε̂.
- Defaults: η = 0.001, β1 = 0.9, β2 = 0.999, ε = 1e-8. The paper also gives AdaMax (infinity norm).
- **Use:** P1 (the formulas), P2 (bias correction), P3 (scale invariance explains why ρ follows η).
- **Interview line:** "Adam normalizes each weight's step by its own gradient size, so η sets the step in weight units."

**2. AdamW (Loshchilov and Hutter, 2017)**
- L2 regularization and weight decay are the same for SGD, but not for Adam.
- Many libraries called L2 "weight decay". In Adam, that L2 term is divided by √v̂, so the decay is not uniform.
- Decoupled weight decay makes the best λ almost independent of the best η.
- It made Adam competitive with SGD with momentum on image classification.
- **Use:** the default optimizer in P3–P5; the decay groups (2-D weights only).
- **Interview line:** "AdamW decays each weight at the same rate; Adam with L2 decays weights with small gradients more."

**3. On the Convergence of Adam and Beyond (Reddi et al.)**
- The exponential moving average in v has a short memory. This can make Adam fail to converge.
- The paper gives a simple convex problem where Adam does not converge, and shows the error in the old proof.
- The fix is a "long-term memory" of v (AMSGrad).
- **Use:** session theory; context for β2 choice.
- **Interview line:** "Adam has a known convergence counter-example; AMSGrad fixes the proof, but LLM practice did not change."

### 14.2 Warmup and the size of the update

**4. Accurate, Large Minibatch SGD (Goyal et al., 2017)**
- Linear scaling rule: when the batch is k× larger, use a k× larger η.
- A gradual warmup is necessary, or large-batch training fails early.
- Result: ResNet-50, batch 8,192, 256 GPUs, one hour, same accuracy.
- **Use:** the origin of warmup; session section on batch size.
- **Interview line:** "Warmup started as the fix that made the linear scaling rule work for large batches."

**5. RAdam (Liu et al., 2019)**
- Early in training, the adaptive LR of Adam has a very large variance, because v̂ uses few samples.
- Warmup acts as variance reduction.
- RAdam adds a term that rectifies this variance.
- **Use:** P3 (why warmup exists); compare with paper 6.
- **Interview line:** "RAdam says warmup reduces the variance of the early adaptive step."

**6. On the adequacy of untuned warmup (Ma and Yarats, 2019)**
- The paper refutes the RAdam analysis. The real cause is the large size of the early update.
- Simple untuned warmup gives about the same results as RAdam.
- Rule of thumb: linear warmup over 2/(1 − β2) steps.
- **Use:** P3 second arm (40-step warmup for β2 = 0.95).
- **Interview line:** "Linear warmup for 2/(1 − β2) steps is a good default; for β2 = 0.95 that is only 40 steps."

**7. Why Warmup the Learning Rate? (Kalra and Barkeshli, 2024)**
- The main gain of warmup: the network tolerates a larger target η.
- Warmup moves the network to better-conditioned (flatter) regions.
- Behaviour depends on the start: progressive sharpening or sharpness reduction. This depends on the initialization and the parameterization.
- A good η_init (from the "loss catapult") can save warmup steps. An initialization of Adam's v can give the warmup benefit.
- **Use:** P3 interpretation; P4 and P5 (a larger usable peak η).
- **Interview line:** "Warmup does not mainly help the start; it makes a higher peak LR stable."

**8. Rotational Equilibrium (Kosson et al., 2023)**
- Weight decay makes the size and the angle of each neuron's update reach a steady state.
- In this state, the average rotation is balanced across layers and neurons. Rotation is a proxy for the effective LR.
- This explains why AdamW works better than Adam with L2.
- Direct control of the rotation reduces the need for warmup.
- **Use:** P3 (why ρ settles after warmup).
- **Interview line:** "With weight decay, every layer reaches a steady rotation per step; η and λ together set it."

### 14.3 Schedules and weight averaging

**9. SGDR (Loshchilov and Hutter, 2016)**
- Warm restarts: the LR follows a cosine decay, then restarts.
- It improves the "anytime" performance of SGD.
- This paper made the cosine shape popular.
- **Use:** P4 (the cosine formula).
- **Interview line:** "Cosine came from SGDR; LLMs kept the cosine and dropped the restarts."

**10. MiniCPM (Hu et al., 2024)**
- Introduces the WSD schedule. It supports continuous training and domain adaptation.
- "Model wind tunnel" experiments: tune on small models, then scale.
- With WSD, they study the data–model scaling law without many retraining runs.
- They find a compute-optimal tokens-per-parameter ratio much higher than Chinchilla.
- **Use:** P4 (WSD); P5 (wind tunnel idea for the 4096 prediction).
- **Interview line:** "WSD lets one run produce models at many lengths, so scaling laws need fewer runs."

**11. Scaling Laws Beyond Fixed Training Durations (Hägele et al., 2024)**
- The cosine schedule makes scaling research complex: each length needs a new run.
- Constant LR with a cooldown scales as reliably as cosine.
- Stochastic weight averaging improves the model along the trajectory with no extra training cost.
- Reusable runs reduce the compute for scaling experiments.
- **Use:** P4 (arm C, the cooldown branch); code in `epfml/schedules-and-scaling`.
- **Interview line:** "Constant LR plus a short cooldown matches cosine and makes runs reusable."

**12. River Valley view of WSD (Wen et al., 2024)**
- WSD gives an unusual loss curve: high in the stable phase, a sharp drop in the decay.
- "River valley" landscape: at high η, the model oscillates between the walls but moves fast along the river.
- The decay removes the oscillation and shows the real progress. Both phases are necessary.
- WSD-S: resume from a decayed checkpoint and keep one main branch. It beat WSD and cyclic cosine at 0.1–1.2 B weights.
- **Use:** P4 (why WSD loses at step 200 but is not worse).
- **Interview line:** "The stable phase makes hidden progress along the river; the decay makes it visible."

**13. DeepSeek LLM (DeepSeek-AI, 2024)**
- 7 B and 67 B models, 2 T tokens.
- Uses a multi-step LR schedule instead of cosine, so that training stages can be reused.
- Fits scaling laws for the best batch size and LR as functions of the compute budget.
- **Use:** P4 (WSD-like schedules in production); P5 (LR depends on scale).
- **Interview line:** "DeepSeek uses a step schedule for reuse and fits the best LR and batch to the compute budget."

**14. Straight to Zero (Bergsma et al., 2025)**
- Under the best peak LR, linear decay to zero (D2Z) beats cosine to 10% at compute-optimal data sizes.
- The gain grows with the dataset size. It holds across model sizes, batch sizes, datasets and vocabularies.
- AdamW acts like an exponential moving average of weight updates. D2Z balances early movement and late averaging.
- Example: a 610 M model at 80 tokens per parameter with D2Z beat 200 tokens per parameter with 10× decay (about 60% compute saved).
- **Use:** P4 (η_min choice); a correction to the course notes.
- **Interview line:** "Decay to zero, not to 10%; at scale it can save more than half the compute."

**15. The Road Less Scheduled (Defazio et al., 2024)**
- Schedules that do not need the stop step T usually lose to schedules that need it.
- Schedule-Free removes the schedule. It uses a theory that unifies schedules and iterate averaging.
- No extra hyperparameters over momentum optimizers.
- Schedule-Free AdamW won the AlgoPerf 2024 self-tuning track.
- **Use:** session alternative to WSD when T is unknown.
- **Interview line:** "Schedule-free training replaces the schedule with averaging; it won AlgoPerf self-tuning."

**16. LAWA, Early Weight Averaging (Sanyal et al., 2023)**
- Averaging checkpoints along the run improves convergence and generalization early.
- The gain is larger at a high LR, and larger when the checkpoints are far apart.
- It beat EMA and SWA. Tested on nanoGPT-2 (125 M–770 M) and Pythia (1 B–12 B).
- **Use:** session section on EMA; a cheap way to judge WSD stable-phase progress.
- **Interview line:** "Averaging spaced checkpoints at high LR gives a better model for free."

### 14.4 Scale, width and batch size

**17. Scaling Laws for Neural Language Models (Kaplan et al., 2020)**
- The loss follows power laws in model size, data size and compute, over seven orders of magnitude.
- Width and depth have small effects within a wide range.
- Larger models are more sample-efficient.
- **Use:** P5 background (LR depends on model size).
- **Interview line:** "Loss is a power law in size, data and compute; the shape of the model matters much less."

**18. Chinchilla (Hoffmann et al., 2022)**
- For compute-optimal training, scale model size and tokens equally: double one, double the other.
- Large models of that time were undertrained. Chinchilla (70 B, 4× more data) beat Gopher (280 B).
- A cosine cycle longer than the run gives a worse model.
- **Use:** P4 (the cosine horizon must match the run); P5 (2.5 M tokens is far below compute-optimal).
- **Interview line:** "Cosine must match the run length; that is the problem WSD removes."

**19. Tensor Programs V, μP and μTransfer (Yang et al., 2022)**
- In μP, many best hyperparameters stay stable when the model size changes.
- μTransfer: tune on a small model, then use the values on the large model with no tuning.
- Results: from 13 M to beat BERT-large; from 40 M to beat the 6.7 B GPT-3 at 7% of its pretraining cost.
- Code: `microsoft/mup`.
- **Use:** P5 (the production fix for the 4096 prediction).
- **Interview line:** "With μP, I tune at small width and transfer the LR to the large model."

**20. Scaling Exponents Across Parameterizations and Optimizers (Everett et al., 2024)**
- Tens of thousands of models, up to 26.8 B weights, three optimizers, four parameterizations.
- All parameterizations, not only μP, can transfer hyperparameters. Their per-layer LR rule for SP beat μP.
- Adam's ε must scale correctly with width, or gradients underflow.
- Adam-atan2 removes ε.
- **Use:** P5 (per-layer LR; ε at large width); P1 (ε position).
- **Interview line:** "Per-layer LR is the key to transfer, and ε must be scaled or removed at large width."

**21. Small-scale proxies for instabilities (Wortsman et al., 2023)**
- Large-scale instabilities also appear in small models at high LR. Examples: attention logit growth, output logit divergence.
- The large-scale fixes (for example qk-layernorm and z-loss) also work at small scale.
- They measure LR sensitivity: how much the final loss changes over a range of LRs.
- Warmup, weight decay and μP reduce LR sensitivity.
- Some instabilities can be predicted from the growth of activation and gradient norms.
- **Use:** P5 (the LR sweep is a sensitivity curve); Session 15 link.
- **Interview line:** "I study instabilities on small proxies at high LR before I pay for a large run."

**22. An Empirical Model of Large-Batch Training (McCandlish et al., 2018)**
- The gradient noise scale predicts the largest useful batch size.
- The noise scale grows as the loss goes down during training.
- It explains the trade-off between compute efficiency and time efficiency.
- **Use:** session section on batch size; batch choice in P4 and P5.
- **Interview line:** "Above the critical batch, more batch buys wall time, not fewer tokens."

**23. SDEs and Scaling Rules for Adaptive Algorithms (Malladi et al., 2022)**
- Gives proven SDE approximations for RMSprop and Adam.
- Practical result: a square-root scaling rule for Adam when the batch size changes.
- **Use:** session section on batch size.
- **Interview line:** "For Adam, scale the LR with the square root of the batch, not linearly."

### 14.5 Fair comparison

**24. On Empirical Comparisons of Optimizers (Choi et al., 2019)**
- The tuning search space is the most important factor in optimizer rankings. Changing it can reverse published results.
- With enough tuning, a more general optimizer never loses to a special case of itself (Adam ≥ momentum ≥ SGD).
- In their tests, adaptive methods never lost to momentum or SGD.
- **Use:** the "tune both sides" rule of the assignment (P4, P5).
- **Interview line:** "Most optimizer rankings are rankings of tuning budgets."

**25. Descending through a Crowded Valley (Schmidt et al., 2020)**
- 15 optimizers, more than 50,000 runs.
- Performance varies a lot across tasks.
- Trying several optimizers with defaults works about as well as tuning one optimizer.
- No optimizer clearly beat Adam.
- **Use:** fairness section; P4 tuning budget.
- **Interview line:** "Across 50,000 runs, Adam stayed a strong default; no newer method beat it consistently."

**26. AlgoPerf (Dahl et al., 2023)**
- Three problems in optimizer comparisons: when training is complete, sensitivity to workload details, and fair tuning.
- AlgoPerf is a time-to-result benchmark on fixed hardware with several workloads and workload variants.
- **Use:** the method behind the fairness protocol (PLAN.md section 10).
- **Interview line:** "Compare time to a target result on fixed hardware, with a fixed tuning rule."

**27. No Train No Gain (Kaddour et al., 2023)**
- Tested layer stacking, layer dropping, selective backprop, RHO loss, Lion and Sophia on BERT and T5.
- With a fixed compute budget, the gains disappeared against a baseline with a fully decayed LR.
- Introduces "reference system time" to compare across machines.
- **Use:** P4 (always compare with a fully decayed baseline: arm D).
- **Interview line:** "Many speed-up claims vanish when the baseline schedule is fully decayed."

**28. Deconstructing What Makes a Good Optimizer (Zhao et al., 2024)**
- Compared SGD, Adafactor, Adam, Lion and Sophia on language models.
- All except SGD were comparable, at their best and across hyperparameters.
- Choose by practical needs: memory and ease of use.
- Signed momentum recovers Adam's performance. Adaptivity in the last layer and in LayerNorm is necessary for stability.
- **Use:** session section 2; P3 (layers behave differently).
- **Interview line:** "Optimizer choice is mostly a memory and engineering decision; adaptivity matters most in the last layer and the norms."

### 14.6 Other optimizers

**29. Adafactor (Shazeer and Stern, 2018)**
- Keeps only row and column sums of v for a matrix. Memory is sublinear.
- When β2 decays too slowly, adaptive methods give updates that are too large. Fixes: update clipping and an increasing β2 schedule.
- Scales the update by the size of the weight.
- **Use:** session β2 discussion (why β2 = 0.999 can be dangerous); memory options.
- **Interview line:** "Adafactor saves the v memory and showed that a slow β2 makes oversized updates."

**30. 8-bit Optimizers (Dettmers et al., 2021)**
- Stores optimizer state in 8 bits with block-wise dynamic quantization.
- A "stable embedding layer" reduces the gradient variance from non-uniform token frequencies.
- Same results as 32-bit states on 1.5 B language models and other tasks, with a two-line code change.
- **Use:** memory section; frugal-budget design.
- **Interview line:** "8-bit states cut optimizer memory by about 75% with no change in hyperparameters."

**31. Lion (Chen et al., 2023)**
- Found by program search. Lion keeps only momentum and uses the sign of the update.
- Every weight gets an update of the same size. Thus, Lion needs a smaller η than Adam.
- The gain grows with the batch size. On language modeling, Lion was similar to or better than Adam.
- **Use:** memory section; an example for fair-comparison questions.
- **Interview line:** "Lion halves optimizer memory; use a 3–10× smaller LR and a larger weight decay."

**32. SOAP (Vyas et al., 2024)**
- Shampoo (power 1/2) equals Adafactor in Shampoo's eigenbasis.
- SOAP runs Adam in that eigenbasis. It has one extra hyperparameter: the preconditioning frequency.
- At 360 M and 660 M, large batch: more than 40% fewer iterations and more than 35% less wall time than AdamW.
- **Use:** session section on newer optimizers.
- **Interview line:** "SOAP is Adam in a rotated basis; it cuts steps by about 40% at large batch."

**33. Muon is Scalable (Liu et al., 2025)**
- Muon orthogonalizes the momentum of each weight matrix.
- Two changes make it scale: add weight decay, and adjust the per-parameter update scale.
- Scaling-law runs: about 2× compute efficiency over AdamW at compute-optimal training.
- Moonlight: a 3 B/16 B MoE trained on 5.7 T tokens with Muon.
- **Use:** P3 (per-element update RMS as an extra metric); frugal-budget design.
- **Interview line:** "Muon needed weight decay and update-scale matching to work at scale; then it gave about 2×."

### 14.7 Ten points to remember for the interview

1. Adam's step does not depend on the gradient scale; η sets the step in weight units. (1)
2. Without bias correction, the step error is r(t) = (1 − β1^t)/√(1 − β2^t). It does not depend on the gradients. (1)
3. AdamW decouples λ from η; decay only 2-D weights. (2, 28)
4. Warmup makes a higher peak η stable. Use about 2/(1 − β2) steps as a start. (6, 7)
5. Weight decay sets a steady update rotation per layer. (8)
6. Cosine must match the run length; WSD does not need it. (11, 18)
7. The WSD stable phase hides progress; the decay shows it. (12)
8. Decay to zero often beats decay to 10%. (14)
9. Tune at small width with μP or per-layer LR rules; scale ε correctly. (19, 20)
10. Most optimizer claims fail when both sides are tuned and the baseline is fully decayed. (24, 27)

---

## 15. Reading order

**Method.** For each paper, read the abstract, the first two pages and the last two pages. This takes about 25 minutes for each paper. The numbers match the paper numbers in section 13.

**Caution.** The notes on what to look for come from the abstracts and the known main results. Check them while you read.

### 15.1 Must read (8 papers)

| No. | Paper | Why you must read it | What to look for in the first and last pages |
|---|---|---|---|
| 1 | Adam, arXiv:1412.6980 | It is the core of the session and of parts 1 and 2. | The first pages give Algorithm 1 and §2: the full update rule. Also read §3 (bias correction, one page). The conclusion is short; skim it. |
| 2 | AdamW, arXiv:1711.05101 | Decoupled weight decay is the V5 default. | The introduction shows that L2 and weight decay are the same for SGD but not for Adam. The conclusion says that the best λ no longer depends on η. |
| 6 | Ma and Yarats, untuned warmup, arXiv:1910.04209 | It gives the warmup rule for part 3. | The introduction refutes the RAdam explanation. The conclusion gives the rule: linear warmup over 2/(1 − β2) steps. |
| 10 | MiniCPM, arXiv:2404.06395 | It introduced WSD. | **Exception.** The introduction and conclusion are about small models. Read the abstract, then go directly to the WSD section and its loss curves. |
| 11 | Hägele et al., arXiv:2405.18392 | Constant η with a cooldown matches cosine (part 4, arm C). | The introduction tells why cosine makes scaling research expensive. The conclusion gives the cooldown length and the weight-averaging result. |
| 12 | Wen et al., river valley, arXiv:2410.05192 | It explains why WSD loses at step 200 but is not worse. | The first pages have the river-valley figure. This is the key picture. The conclusion gives WSD-S. |
| 19 | μP, Tensor Programs V, arXiv:2203.03466 | It answers the width-4096 question of part 5. | The first pages show the plot of η against width: SP moves, μP stays. That plot is a small copy of your part 5 result. The conclusion gives the cost savings. |
| 24 | Choi et al., arXiv:1910.05446 | It is the source of the "tune both sides" rule. | The introduction says that the search space decides the ranking. The conclusion gives practical tuning advice. |

### 15.2 Should read, after the form (6 papers)

| No. | Paper | Why |
|---|---|---|
| 7 | Kalra and Barkeshli, arXiv:2406.09405 | The real gain of warmup is a stable higher peak η. |
| 8 | Kosson et al., arXiv:2305.17212 | It explains why ρ settles after warmup (part 3). |
| 14 | Bergsma et al., arXiv:2502.15938 | Decay to zero beats decay to 10%. |
| 27 | Kaddour et al., arXiv:2307.06440 | Speed-up claims disappear against a fully decayed baseline. |
| 20 | Everett et al., arXiv:2407.05872 | Per-layer η, and the ε problem at large width. |
| 21 | Wortsman et al., arXiv:2309.14322 | LR sensitivity on small proxies. It links to Session 15. |

### 15.3 Articles (short; read all of each)

- Karpathy, "A Recipe for Training Neural Networks": practical training discipline.
- PyTorch documentation for `torch.optim.Adam` and `torch.optim.AdamW`: the exact update that the code runs. Part 1 needs it.
- Keller Jordan, "Muon": optional, for interview depth only.

### 15.4 Not now

Papers 3, 4, 5, 9, 13, 15–18, 22, 23, 25, 26, 28–33. The key points in section 14 are enough for the interview.

### 15.5 Plan for 16 Oct

| Time | Read |
|---|---|
| 08:00–09:00 | Adam (1), then AdamW (2). |
| 10:45–11:15 | MiniCPM WSD section (10), then the Hägele et al. introduction (11). |
| 15:30–16:00 | μP first pages (19). |
| After the form | Ma and Yarats (6), Wen et al. (12), Choi et al. (24): about 75 minutes. |
