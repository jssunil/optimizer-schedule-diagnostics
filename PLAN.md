# PLAN: Optimizer and Schedule Diagnostics (Session 11)

This plan uses ASD-STE100 Simplified Technical English rules:
- Procedure sentences have 20 words or fewer. Description sentences have 25 words or fewer.
- Use the active voice and one instruction in each step.
- Procedures use the imperative ("Do this").
- A word has one meaning in this document.
- Each term is defined before it is used.

The task text is in [assignment.md](assignment.md). The theory is in [SESSION_11_EXPLAINER.md](SESSION_11_EXPLAINER.md). Interview practice is in [INTERVIEW_QA.md](INTERVIEW_QA.md).

---

## 1. Goal

Make the optimizer and the learning-rate schedule visible. Measure each claim. Do not accept a claim that you did not measure.

The project has five parts:

| Part | Task | Output |
|---|---|---|
| 1 | Calculate Adam by hand for one weight and five gradients. Compare each value with PyTorch. | Agreement table |
| 2 | Turn off bias correction. Plot the first 20 steps both ways. Find when the difference stops mattering. | Plot + step number |
| 3 | Log the update-to-weight ratio for each layer. Find the step where warmup stops changing it. | Plot + step number |
| 4 | Train under cosine and under WSD for a 300-step plan. Stop both at step 200. Select one model. | Loss table + decision |
| 5 | Sweep the learning rate at widths 256, 512 and 1024. Predict the learning rate at width 4096. | Plot + prediction + confidence |

The assignment adds one rule for all parts: **tune both sides before you compare them.** Section 10 gives the procedure.

The submission is a GitHub repository with `README.md` and the support code.

---

## 2. Words

| Word | Meaning in this document |
|---|---|
| token | One unit of text. Here, one GPT-2 BPE token (vocabulary V = 50,257, padded to 50,304). |
| weight | One number in the model that training can change. |
| gradient (g) | One number for one weight. It tells how much the loss changes when that weight changes. |
| step (t) | One update of all weights. The first step is t = 1. |
| learning rate (LR, η) | The number that sets the size of each step. |
| peak LR | The highest LR in a schedule. |
| schedule | The rule that gives the LR at each step. |
| warmup | The first part of a schedule. The LR increases from near 0 to the peak LR. |
| first moment (m) | A moving average of the gradient. It acts as momentum. |
| second moment (v) | A moving average of the squared gradient. It measures the gradient size. |
| β1, β2 | The decay factors of m and v. A value near 1 gives a long memory. |
| ε (epsilon) | A small number in the denominator. It prevents division by zero. |
| bias correction | Division of m by (1 − β1^t) and of v by (1 − β2^t). It removes the pull toward 0 that the zero start causes. |
| m̂, v̂ | m and v after bias correction. |
| weight decay (λ) | A pull of each weight toward 0 at each step. |
| decoupled weight decay | Weight decay that is applied directly to the weight, not added to the gradient. AdamW uses it. |
| update-to-weight ratio | ‖change of W in one step‖ ÷ ‖W‖, for one weight matrix W. ‖·‖ is the Frobenius norm. |
| cosine schedule | After warmup, the LR follows half a cosine curve down to a minimum LR at the planned last step. |
| WSD | Warmup-Stable-Decay. Warmup, then a constant LR, then a short decay at the end. |
| cooldown | The decay part of WSD. |
| width (d) | The number of channels in the model (`n_embd`). |
| LR sweep | A set of runs that differ only in the LR. |
| LR* | The LR that gives the lowest loss in a sweep. |
| seed | The number that fixes the random start and the data order. |
| noise band | The spread of a result across seeds. A difference inside the noise band is not a real difference. |
| arm | One configuration in a comparison, for example "cosine at its best LR". |

---

## 3. Hardware, software and budget

| Item | Value |
|---|---|
| Main GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB (Blackwell, Windows 11) |
| Software | Python 3.13, PyTorch 2.7 + CUDA 12.8 |
| Measured speed | P10 reached 23% MFU on the 175.8 TFLOP/s peak, about 40 TFLOP/s, at 10.7 M weights. |
| GPU sharing | `src/device.py` keeps the GPU busy 60% of the time (`GPU_DUTY = 0.6`), so wall time is about 1.7× compute time. |

**Frugal rule.** The full project must run in about 1 GPU-hour on the local card. A principal architect spends compute on the decision, not on the size of the model. Section 11 gives the budget for each part.

---

## 4. Model and data

### 4.1 Model

Use the nanoGPT model from P10 (`src/model.py`). It is copied unchanged. Dropout is 0, so a fixed seed gives the same result each time. The vocabulary is the GPT-2 vocabulary (V = 50,304 after padding).

| Name | Layers | Heads | Width (d) | Context (T) | Weights (about): blocks + embedding = total | Used in |
|---|---|---|---|---|---|---|
| S (= W256) | 4 | 4 | 256 | 512 | 3.2 M + 12.9 M = 16 M | parts 2, 3, 4 and 5 |
| W512 | 4 | 8 | 512 | 512 | 12.6 M + 25.8 M = 38 M | part 5 |
| W1024 | 4 | 16 | 1024 | 512 | 50.3 M + 51.5 M = 102 M | part 5 |
| (W4096) | 4 | 64 | 4096 | 512 | 805 M + 206 M = 1.0 B | prediction only, not trained |

The block weights use 12 · L · d². The embedding is V · d. The output layer shares the embedding matrix (weight tying), but its matrix multiplication is the largest single cost in the step.

The head size is 64 at every width in part 5. Thus, only the width changes. Attention shape stays the same.

### 4.2 Data: FineWeb, 100 M tokens

Use FineWeb web text, already tokenized with the GPT-2 tokenizer. The modded-nanogpt speedrun uses the same files (Hugging Face dataset `kjj0/fineweb10B-gpt2`). Each shard has 100 M tokens in 200 MB.

| Split | File | Tokens |
|---|---|---|
| Training | `fineweb_train_000001.bin` | 100 M |
| Validation | `fineweb_val_000000.bin` | 100 M (use a fixed set of 40 batches = 0.33 M tokens) |

**Why medium data and not Tiny Shakespeare:**
- Tiny Shakespeare has 1 M training tokens. The full study reads about 240 M tokens. With 1 M tokens, each run reads the same text many times, and the validation loss measures memorization.
- Character tokens (V = 65) do not behave like LLM tokens. A 50 K vocabulary gives sparse embedding gradients and a heavy-tailed token frequency. These change the optimizer behaviour that the assignment studies (v of rare rows, update ratio of embeddings).
- Web text is the data type of real pretraining. The LR and schedule results then transfer better to V5.

**Data options:**

| Option | Advantage | Disadvantage | Selected |
|---|---|---|---|
| Tiny Shakespeare (P10) | Already on disk. Very fast. | 1 M tokens; repeats; character vocabulary. | No |
| TinyStories | Small models learn it fast. Clear loss curves. | Simple, narrow text. LR results transfer less to web text. | No |
| OpenWebText (nanoGPT) | Known baseline for GPT-2 small. | About 54 GB download and a tokenization step. | No |
| FineWeb-Edu `sample-10BT` (parquet) | High-quality web text. | You must tokenize it first (CPU time). | Fallback |
| **FineWeb GPT-2 shards (`kjj0/fineweb10B-gpt2`)** | Already tokenized. 400 MB for train + val. Same data as the modded-nanogpt speedrun. | Fixed GPT-2 tokenizer, not the V5 tokenizer. | **Yes** |

**Loader.** `src/data.py` (`FineWebData`). It reads each shard with `numpy.memmap` as `uint16` after its header. It checks the header (format from `modded-nanogpt`: 256 int32, magic 20240520, version 1, token count) and the file size. It takes random windows of T + 1 tokens with a seeded generator. Thus, the same seed gives the same data order. Get the files with `scripts/get_fineweb.py` into `data/` (not in git).

**Data limit.** All runs together read about 240 M tokens as random windows from a 100 M-token shard. Each run alone reads 1.6–2.5 M tokens. Thus, a single run does not repeat data in a meaningful way.

### 4.3 Shared code to write once

| File | Content | Used by |
|---|---|---|
| `src/ref_adam.py` | `adam_by_hand` (scalar, float64, L2 or decoupled decay, bias-correction switch), `RefAdamW` (torch optimizer with a bias-correction switch) and `bias_ratio`. Done. | parts 1, 2 |
| `src/schedules.py` | `lr_at(step, kind, peak, warmup, total, ...)` for constant, cosine, WSD (linear, cosine, 1−sqrt decay). Done. | parts 2–5 |
| `src/train.py` | `Run` (all settings of one run), `train` (one run; logs loss, LR, grad norm, update ratio), `evaluate`, `lr_sweep` (with the grid-edge rule) and `QUICK=1` smoke-test mode. Done. | parts 2–5 |
| `train.evaluate` + `FineWebData.eval_batches` | Validation loss on a fixed set of 40 validation batches (same batches for every run). Done. | parts 4, 5 |
| `src/data.py` | FineWeb shards (memory-mapped), seeded random windows, fixed validation batches, width configs. Done. | parts 2–5 |

**Parameter groups.** Use the nanoGPT `configure_optimizers` rule: decay every tensor with 2 or more dimensions (block matrices, the tied token embedding and output layer, the position embedding). Do not decay biases or LayerNorm weights.

**Optimizer defaults** (unless a part says different): AdamW, β1 = 0.9, β2 = 0.95, ε = 1e-8, λ = 0.1, gradient clipping at 1.0, bf16 autocast, fixed batch.

---

## 5. Part 1: Adam by hand

### 5.1 Purpose

Prove that you know each number that Adam calculates. A wrong optimizer gives a model that trains, but trains badly. Nothing shows an error.

### 5.2 Theory

For each step t, with gradient g_t:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat m_t = \frac{m_t}{1-\beta_1^t} \qquad \hat v_t = \frac{v_t}{1-\beta_2^t} \qquad \Delta\theta_t = -\eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

m_0 = v_0 = 0. These are the equations of Algorithm 1 in Kingma and Ba (2014).

**What PyTorch does.** In `torch/optim/adam.py` (single-tensor path), PyTorch calculates `denom = sqrt(v) / sqrt(1 − β2^t) + ε` and `step_size = η / (1 − β1^t)`. This is the same as Algorithm 1. The order of the operations is different, so float results can differ in the last bits. Other details that part 1 must match (PyTorch 2.7 source and 2.14 documentation; see [SESSION_11_EXPLAINER.md section 4.5](SESSION_11_EXPLAINER.md#45-pytorch-adam-and-adamw-the-exact-update-that-the-code-runs)):
- m is updated with `exp_avg.lerp_(grad, 1 − β1)`. The algebra is the same as β1·m + (1 − β1)·g.
- `state["step"]` is a scalar tensor. The bias factors use it as a Python float.
- m and v have the data type of the parameter. Thus, a float64 weight gives float64 states.
- `AdamW` is `Adam` with `decoupled_weight_decay=True`. It multiplies the weight by (1 − γλ) **before** the moment updates. `Adam` with `weight_decay` adds λθ to the gradient instead.
- The default `AdamW` λ is 0.01, not 0. Set `weight_decay=0` in part 1.

**The other form.** Section 2 of the Adam paper gives a faster form: η_t = η·√(1−β2^t)/(1−β1^t) and Δθ = −η_t·m/(√v + ε̂). In that form, ε̂ is not the same number as ε. Some older libraries use that form. Thus, "Adam" in two libraries can give different steps when ε is not small compared with √v̂.

### 5.3 Design options

| Option | Advantage | Disadvantage | Selected |
|---|---|---|---|
| A. Hand values in Python `float` (float64); PyTorch on a float64 tensor | Both sides use the same precision. Agreement to about 1e-15 is possible. A failure is a real error. | Does not show float32 behaviour. | **Yes (main table)** |
| B. PyTorch in float32 | Shows the precision of real training. | Agreement only to about 1e-7 relative. The precision hides small formula errors. | Yes (second column) |
| C. Exact fractions (`fractions.Fraction`) | Exact values, no rounding at all. | √ is not exact; you must round at the square root. Slow; no gain over A. | No |
| D. Subclass `torch.optim.Optimizer` and compare | Gives code that you can reuse in part 2. | A bug in your class can match a bug in your hand values. It does not prove the hand values. | Use as `ref_adam.py`, check it against A |
| E. Compare `foreach=False`, `foreach=True` and `fused=True` | Shows that the three PyTorch code paths agree. Fused is the path in fast training. | `fused=True` needs CUDA. | **Yes (extra row)** |
| F. Repeat with λ = 0.1: `Adam(weight_decay=0.1)` (L2) and `AdamW(weight_decay=0.1)` (decoupled), each against a hand calculation | Shows the difference between L2 and decoupled decay with numbers, in 5 steps. | Not asked by the task. | Yes (short extra table) |

**Selected design:** A as the truth, B and E as extra checks, D as the reusable code. A principal engineer checks the production code path (fused), not only the textbook path.

### 5.4 Gradients

Use five gradients that test different cases: `g = [0.5, -1.0, 2.0, 0.01, -0.3]`.
- A change of sign tests m.
- A large value (2.0) after small values tests v.
- A very small value (0.01) shows that the step does not shrink with the gradient. This is the scale invariance of Adam.

Use θ_0 = 1.0, η = 0.1 (large, so the steps are easy to see), β1 = 0.9, β2 = 0.999 (the paper default), ε = 1e-8, no weight decay.

### 5.5 Procedure

1. Write the hand calculation as a loop over the five gradients.
2. Keep m, v, m̂, v̂, Δθ and θ for each step.
3. Make a float64 tensor `w = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)`.
4. Make `torch.optim.Adam([w], lr=0.1, betas=(0.9, 0.999), eps=1e-8, foreach=False)`.
5. For each gradient, set `w.grad` to the gradient. Then call `opt.step()`.
6. Read m from `opt.state[w]["exp_avg"]` and v from `opt.state[w]["exp_avg_sq"]`.
7. Calculate m̂ and v̂ from m and v. PyTorch does not store them.
8. Record the PyTorch Δθ as the change of `w`.
9. Calculate the absolute difference for each value and each step.
10. Repeat steps 3 to 9 in float32, with `foreach=True`, and with `fused=True` on CUDA.
11. Save the table to `results/p1_adam_by_hand.json`.

### 5.6 Pass criteria

- float64: the largest absolute difference is less than 1e-12 for every value.
- float32: the largest relative difference is less than 1e-6.
- The three code paths agree with each other to the same limits.

### 5.7 Pitfalls

- Setting `w.grad` directly skips `backward()`. This is correct here, because the test is about the optimizer only.
- `step` in PyTorch starts at 1 after the first call. An off-by-one error in t changes every m̂ and v̂.
- Do not use AdamW with the default λ = 0.01. The weight decay then changes θ. Use Adam, or AdamW with `weight_decay=0`.

---

## 6. Part 2: Bias correction

### 6.1 Purpose

Show what bias correction does, and for how long it matters.

### 6.2 Theory

m_0 = 0 and v_0 = 0. Thus, early m and v are too small. Bias correction divides them by (1 − β^t).

The ratio of the uncorrected step to the corrected step is (with ε ignored):

$$r(t) = \frac{1-\beta_1^t}{\sqrt{1-\beta_2^t}}$$

This ratio does not depend on the gradients. Thus, part 2 has an exact expected answer. The measured ratio must agree with r(t).

Values (calculated by hand; the script must confirm them):

| t | r(t), β2 = 0.95 | r(t), β2 = 0.999 |
|---|---|---|
| 1 | 0.447 | 3.16 |
| 2 | 0.608 | 4.25 |
| 5 | 0.861 | 5.80 |
| 10 | 1.028 | 6.53 |
| 20 | 1.097 | 6.24 |
| 50 | 1.04 | 4.50 |
| 100 | 1.003 | 3.2 |

Three facts come from this table:
1. With β2 = 0.95, the uncorrected first steps are too **small**. After about step 9 they are too **large**. The sign of the error changes.
2. With β2 = 0.999, the uncorrected steps are up to 6.6× too **large** (near step 12). Without bias correction, the first steps can damage the model.
3. |r(t) − 1| < 1% at about t ≈ 77 for β2 = 0.95, and at about t ≈ 3,900 for β2 = 0.999. **Twenty steps are not enough** for either case.

### 6.3 Two meanings of "stops mattering"

| Meaning | Test | Advantage | Disadvantage |
|---|---|---|---|
| Step size | First t after which \|r(t) − 1\| < 1% and stays below | Exact. Same for every model. | Says nothing about the loss. |
| Loss | First t after which the loss difference stays inside the noise band (3 seeds) | This is what matters to training. | Needs training runs. The answer depends on the model and the LR. |

**Important.** The weights of the two runs differ after the early steps. Thus, the loss difference can stay after r(t) is near 1. Report both meanings. State which one you use for the answer.

### 6.4 Design options

| Option | Advantage | Disadvantage | Selected |
|---|---|---|---|
| A. Scalar weight, fixed gradients from part 1 | Exact match to r(t). Fast. | Does not show the effect on a model. | **Yes (plot 1)** |
| B. Small model S, both settings, 3 seeds | Shows the effect on the loss. | More cost. The result depends on the LR. | **Yes (plot 2)** |
| C. Change PyTorch Adam internals | No new code. | Fragile. Depends on a private code path. | No |
| D. `ref_adam.py` with a `bias_correction` switch | Simple. Checked against part 1 first. | Your code, not PyTorch code. | **Yes** |
| E. Use warmup to hide the problem | Real practice: warmup reduces the early-step error. | Mixes two effects. | Show as a note, not as the main result |

### 6.5 Procedure

1. Check `ref_adam.py` with bias correction on. It must match part 1.
2. Run 20 steps of `ref_adam` with and without bias correction. Use the same gradients.
3. Plot the step size against t for both runs. Plot r(t) on a second axis.
4. Repeat steps 2 and 3 for β2 = 0.95 and β2 = 0.999.
5. Extend the scalar run to 5,000 steps. Find the first t where |r(t) − 1| < 1%.
6. Train model S for 300 steps with each setting, for both β2 values. Use 3 seeds (12 runs). Use no warmup.
7. Plot the mean loss and the seed range for both settings.
8. Find the first step after which the two loss bands overlap and stay overlapped.
9. Save the plots and numbers to `results/p2_bias_correction.*`.

### 6.6 Pass criteria

- The measured step ratio equals r(t) to 1e-9 (ε is small).
- The report gives the answer for each β2 and each meaning.

---

## 7. Part 3: Update-to-weight ratio during warmup

### 7.1 Purpose

Show how the size of each update, relative to the weight, changes during warmup. Find when warmup stops controlling it.

### 7.2 Theory

For one weight matrix W at step t:

$$\rho_t = \frac{\lVert W_t - W_{t-1} \rVert_F}{\lVert W_{t-1} \rVert_F}$$

For Adam, each element of m̂/√v̂ has a size of about 1 or less. Thus, ‖ΔW‖ is about η_t · √(number of elements) · c, with c ≤ 1. During warmup, η_t increases, so ρ_t increases with it.

After warmup, η is constant. Then ρ_t changes only because ‖W‖ and c change. With weight decay, ρ_t moves to a stable value. Kosson et al. (2023) call this the rotational equilibrium.

A common guide is ρ ≈ 1e-3 (Karpathy, cs231n notes). A much larger value means the LR is too high for that layer. A much smaller value means the layer learns slowly.

### 7.3 Design options

| Option | Advantage | Disadvantage | Selected |
|---|---|---|---|
| A. ρ from W before and after `opt.step()` | Measures the real change, with weight decay and clipping included. | One copy of the weights each step. Small at 16 M weights (64 MB in fp32). | **Yes** |
| B. ρ from η · m̂/(√v̂ + ε), read from the optimizer state | No weight copy. | Misses weight decay and clipping. | No |
| C. Per-element RMS of the update (as Muon and Moonlight report it) | Does not depend on the matrix size. | Not the quantity the assignment asks for. | Log as extra |
| D. Angle between W before and after | Matches the rotational-equilibrium theory. | Harder to read. | Log as extra |
| E. Log every k steps | Lower cost. | Can miss the change point. | No; log every step (cheap at this size) |

**Schedule for part 3.** Use linear warmup for 100 steps, then a **constant** LR to step 300. Do not decay. A decay would also change ρ and hide the warmup effect.

### 7.4 Finding "the step where warmup stops changing it"

| Method | Advantage | Disadvantage |
|---|---|---|
| Rolling slope of log ρ (window 10) enters the noise band of the post-warmup slope | Uses data only. | Needs a noise-band rule. |
| ρ_t / η_t becomes flat | Removes the schedule. Shows the optimizer part alone. | Not the raw ratio that the task asks for. |
| Fit two straight lines (warmup, after) and find the break point | One clear number. | Assumes two straight parts. |

**Selected:** the rolling-slope method on the median over layers, with 3 seeds. Report the break-point fit as a check. Expect the answer near step 100. Report the true number, also if it is different. A different number is a finding.

**Second arm (cheap).** Repeat with warmup = 2/(1 − β2) = 40 steps. Ma and Yarats (2019) give this as an untuned warmup rule. Compare the two answers.

### 7.5 Procedure

1. Make model S. Set the schedule: warmup 100 steps, then constant.
2. Before each `opt.step()`, copy each 2-D weight.
3. After the step, calculate ρ for each layer.
4. Also calculate ρ / η_t for each layer.
5. Repeat for 3 seeds.
6. Plot ρ against the step for each layer, on a log y-axis. Mark the end of warmup.
7. Find the change point with the selected method.
8. Repeat steps 1 to 7 with a 40-step warmup.
9. Save the results to `results/p3_update_ratio.*`.

### 7.6 Pitfalls

- Gradient clipping changes the update in the first steps. Log the grad norm and the clip events.
- Weight tying: `wte` and `lm_head` share one matrix. Log it once.
- Embeddings get sparse gradients: only the rows of tokens in the batch change. With V = 50,304, most rows do not change in a step. Their ρ is much smaller. Report them apart from the block matrices.

---

## 8. Part 4: Cosine compared with WSD, stopped at step 200

### 8.1 Purpose

Show the main operational difference between the two schedules. A cosine schedule needs the end step in advance. WSD does not.

### 8.2 Theory

Both schedules plan 300 steps with 30 warmup steps.

- **Cosine:** η_t = η_min + (η_peak − η_min) · ½(1 + cos(π · (t − 30)/270)), with η_min = 0.1 · η_peak.
- **WSD:** constant η_peak from step 30 to step 240. Then a decay to η_min over the last 60 steps (20%).

At step 200:
- Cosine has η = 0.37 · η_peak. The model is partly annealed.
- WSD has η = η_peak. The model has had no decay.

Thus, the cosine model is expected to have the lower loss at step 200. This is not a defect of WSD. The WSD model is at a point where you can continue training, or decay it in a short branch.

### 8.3 Arms

| Arm | Description | Why |
|---|---|---|
| A | Cosine planned for 300, stopped at 200 | Required by the task |
| B | WSD planned for 300, stopped at 200 | Required by the task |
| C | WSD branch: take the B checkpoint at step 160, decay over steps 160–200 | Same 200 steps of compute. Shows that WSD can deliver an annealed model when the stop is known 40 steps early. |
| D | Cosine planned for 200 (correct horizon) | Reference: the best cosine result if the end had been known. Chinchilla (2022) shows that a cosine longer than the run is worse. |

C and D cost little. They change the answer from an opinion into a measurement.

### 8.4 Tuning (both sides)

1. For each arm, sweep the peak LR over 5 values: 3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3.
2. Use 3 seeds at each value.
3. Select the best peak LR for each arm by the mean validation loss at step 200.
4. If the best value is at the edge of the grid, add one more value past the edge.
5. Compare the arms at their own best peak LR.

Tokens: batch 16 × 512 = 8,192 tokens per step. 200 steps = 1.6 M tokens, from a 100 M-token shard.

### 8.5 Design options

| Option | Advantage | Disadvantage | Selected |
|---|---|---|---|
| WSD decay shape: linear | Simple. Bergsma et al. (2025) support linear decay. | — | Default |
| WSD decay shape: 1 − sqrt | Hägele et al. (2024) report it as slightly better. | Extra arm. | Extra, if time |
| WSD decay length 20% | Hägele et al. find 10–20% good. | — | **Yes** |
| η_min = 0.1 · peak | Matches the course notes. | Bergsma et al. (2025) show that decay to 0 is often better. | Yes; note the alternative |
| Compare by training loss | Free. | Noisy; measures fit to seen data. | No |
| Compare by validation loss on fixed batches | Same batches for each run. Fair. | Small cost. | **Yes** |

### 8.6 Which model to keep

Give the answer in two parts:
1. **If step 200 is the final step:** keep the model with the lower validation loss. This is probably the cosine model (A).
2. **If training can continue:** keep the WSD checkpoint (B). It can continue at the same LR. It can also become an annealed model with a short cooldown (arm C). The cosine model cannot continue well, because its LR is already low.

A principal architect keeps the WSD checkpoint *and* a cooled-down branch. One run then gives both a usable model and a point to continue from. MiniCPM (2024) and DeepSeek LLM (2024) use this property.

### 8.7 Procedure

1. Write `lr_at` for cosine and WSD. Plot both schedules first.
2. Run the LR sweep for arms A, B and D (5 LRs × 3 seeds each).
3. For arm C, run WSD planned for 200 steps at B's best LR, with B's seeds. Its LR equals B's LR up to step 160, so it is the same run as B up to step 160. Thus, no checkpoint file is necessary.
4. Calculate the validation loss at step 200 for every run.
5. Make a table: arm, best peak LR, mean loss, seed range.
6. Plot the loss curves at the best LR for each arm, with the LR curve below.
7. Write the decision with the two-part answer.
8. Save to `results/p4_cosine_vs_wsd.*`.

---

## 9. Part 5: LR sweep across widths

### 9.1 Purpose

Show that the best LR moves when the width changes. Predict the best LR at width 4096 from small runs. State the confidence honestly.

### 9.2 Theory

nanoGPT uses the standard parameterization (SP). In SP with Adam, the size of the update to each weight does not shrink with the width. The effect on the output grows with the width. Thus, LR* usually goes down as the width goes up.

μP (Yang et al., 2022) changes the initialization and the per-layer LR. Then LR* stays almost constant across widths, and you can tune at a small width. This is "zero-shot hyperparameter transfer". Everett et al. (2024) show that SP with correct per-layer LRs can also transfer.

### 9.3 Design options

| Option | Advantage | Disadvantage | Selected |
|---|---|---|---|
| Fixed depth (4 layers), head size 64 | Only the width changes. | Not how real model families grow. | **Yes** |
| Fixed tokens per run (2.5 M) | Equal data for each width. | Large widths are further from convergence. | **Yes** |
| Fixed compute per run | Fair in FLOPs. | Small widths see 6× more data. The data amount then changes with the width. | No |
| LR grid: 9 values at ×√2 spacing | Fine enough for a parabola fit. | 27 runs per seed. | **Yes** |
| LR grid: ×2 spacing | Fewer runs. | The minimum is less exact. | Fallback if time is short |
| Schedule: same warmup + cosine for all runs | Removes the schedule as a variable. | — | **Yes** |
| μP version as a second sweep | Shows the production fix. | Needs `mup` package or extra code. | Only if time |
| Seeds: 2 per point | Gives a noise band for the minimum. | Doubles cost. | **Yes** |
| Fixed λ = 0.1 across the LR grid | PyTorch `AdamW` decay per step is γλ, so the decay grows with γ. This is the usual practice; state it. | The decay strength is not constant across the sweep. | **Yes** |
| Fixed γλ across the LR grid | Same decay strength at every LR. | Not the usual practice; another variable to explain. | No; mention in the README |

Grid centre: about 1e-3. Grid: 2.5e-4 to 4e-3, nine values (×√2). Extend it if a minimum is at the edge.

Tokens: batch 16 × 512 = 8,192 tokens per step, 300 steps = 2.5 M tokens. This is far below the compute-optimal amount (about 20 tokens per weight). Thus, LR* here is the best LR for a short run.

### 9.4 Finding each minimum

1. Take the loss at each LR (mean of the seeds).
2. Fit a parabola to log-loss against log-LR, using the lowest point and its two neighbours.
3. The vertex of the parabola is LR*.
4. Do a bootstrap over seeds (resample, refit, 200 times) to get an interval for LR*.

### 9.5 Prediction at width 4096

1. Fit log LR* = a + b · log d through the three points.
2. Calculate LR*(4096) = exp(a + b · log 4096).
3. Use the bootstrap fits to give an interval.

The exponent b is usually between −1 and 0 in SP. Report the measured value.

**Confidence statement.** State "low to moderate", for these reasons:
- 4096 is 4× outside the measured range. Three points fix a line but do not test it.
- A 1 B-weight model trains on many more tokens. LR* goes down with longer training.
- At small width, the embedding is most of the model (80% of the weights at W256, 50% at W1024). The trend mixes two layer types. μP treats the output layer with its own rule.
- Batch size and depth also move LR*. Here they are fixed.
- At large width, ε can become large relative to √v̂ (Wortsman et al., 2023). This changes the optimizer.

**What to do in production.** Use μP or per-layer LR scaling, tune on small proxies, and confirm with one short run at the target width. MiniCPM calls this a "model wind tunnel".

### 9.6 Procedure

1. Make the three configurations W256, W512 and W1024.
2. Run the LR grid: 9 LRs × 2 seeds × 3 widths = 54 runs.
3. Calculate the validation loss of each run.
4. Plot loss against LR on a log x-axis. Use one curve per width.
5. Mark each LR* on its curve.
6. Fit the line and predict LR*(4096). Plot the line on a second panel (log LR* against log d).
7. Write the prediction, the interval and the confidence statement.
8. Save to `results/p5_lr_sweep.*`.

---

## 10. Fair comparison protocol

Choi et al. (2019) and Kaddour et al. (2023) show that a weak baseline makes a new method look good. Use these rules in parts 2, 4 and 5:

1. Give each arm the same number of tuning runs.
2. Use the same seeds, the same data order and the same evaluation batches.
3. Compare at the best setting of each arm, not at a shared setting.
4. Do not accept a best value at the edge of a grid. Extend the grid.
5. Report the seed range. A difference inside the noise band is "no difference".
6. Compare on validation loss.
7. Write down every run, also the failed runs.

---

## 11. Compute budget

Estimate: compute time = 6 × weights × tokens ÷ 40 TFLOP/s. Small models reach a lower MFU, so the estimates for S are rounded up.

| Part | Runs | Model | Tokens per run | Wall time (about) |
|---|---|---|---|---|
| 1 | 4 | scalar | — | seconds |
| 2 | 2 scalar + 12 model | scalar, S | 2.5 M | 6 min |
| 3 | 6 | S | 2.5 M | 3 min |
| 4 | 3 arms × 5 LRs × 3 seeds + 3 branches = 48 | S | 1.6 M | 14 min |
| 5 | 54 | W256–W1024 | 2.5 M | 37 min (W1024 is 20 min of it) |
| **Total** | about 115 | | about 240 M in total | **about 60 min**, plus a 400 MB download |

**What to cut first if time is short:**
1. Part 5: use ×2 LR spacing (5 values) and 1 seed. Saves about 25 min.
2. Part 4: drop arm D. Use 2 seeds.
3. Do not cut arm C or the tuning of both sides. They are the point of the assignment.

---

## 12. Tests

Put the tests in `tests/` (CPU only, as in P10).

| Test | Checks |
|---|---|
| `test_ref_adam_matches_torch` | `ref_adam` equals `torch.optim.Adam` for 5 steps in float64. |
| `test_bias_ratio` | Uncorrected ÷ corrected step equals r(t) for t = 1…20. |
| `test_schedules` | Cosine: peak after warmup, η_min at the end. WSD: flat in the stable phase, η_min at the end. Both are continuous. |
| `test_weight_decay_groups` | No 1-D weight is in the decay group. |
| `test_eval_batches_fixed` | Two calls give the same evaluation batches. |

---

## 13. README structure

For each part, give: the question, the method (short), the plot, the numbers, the answer in one sentence, and the limits. Then give the fairness section and the budget table. Put the decision of each part at the top of its section. A reviewer reads the answer first.

---

## 14. Schedule for 16 Oct

| Time | Work |
|---|---|
| 08:00–09:00 | Read Session 11 notes and Adam §2–3. Write `ref_adam.py`, `schedules.py`, `train.py`. |
| 10:00–10:45 | Finish `train.py`. Start part 4 and part 5 runs in the background. |
| 10:45–11:15 | Read the WSD section of MiniCPM. |
| 11:15–12:00 | Revise Session 15 (R15) while the runs train. |
| 13:00–14:00 | Part 1. |
| 14:00–14:45 | Part 2. |
| 14:45–15:30 | Part 3. |
| 15:30–16:00 | Read μP (Tensor Programs V) §1–3. |
| 17:00–18:00 | Part 4 and part 5 plots, the width-4096 prediction. |
| 18:00–18:30 | README. Push. |
