# Interview Q&A: Optimizer and Schedule Diagnostics

**Role:** Principal Architect, LLM training. **Setting:** a technical interview at a large AI lab, based on this project.

This document uses ASD-STE100 Simplified Technical English rules. The terms are defined in [PLAN.md](PLAN.md#2-words). The theory and sources are in [SESSION_11_EXPLAINER.md](SESSION_11_EXPLAINER.md).

**How to use it.** Read the question. Answer aloud in 60–90 seconds. Then read the model answer. A principal answer has three parts: the fact, the measurement that proves it, and the decision it supports.

Fill the `[result]` marks with your numbers after the runs.

---

## A. Project summary

**A1. Describe the project in 30 seconds.**
I made the optimizer and the schedule of a small GPT measurable. I calculated Adam by hand and matched PyTorch on three code paths. I measured the bias-correction error, the update-to-weight ratio during warmup, cosine against WSD at an early stop, and the best LR at three widths. Each comparison was tuned on both sides, with 2–3 seeds. The data is 100 M tokens of FineWeb web text. The full study runs in about one GPU-hour on one consumer GPU.

**A2. Why does this matter for a company that trains LLMs?**
The optimizer and the schedule decide most of the result for a given compute budget. Their errors do not crash a run. They make it slowly worse. A team that can measure these effects on small proxies saves large runs. On a frugal budget, that is the main lever.

**A3. What is the one result you would show a CTO?**
Part 4. A WSD run stopped at step 200 has a worse loss than cosine. But a 40-step cooldown branch from the same run gives an annealed model, and the main run can still continue. Thus, one WSD run replaces several cosine runs. That changes the compute plan.

---

## B. Part 1: Adam by hand

**B1. Walk me through one Adam step.**
Update m as a moving average of g with β1. Update v as a moving average of g² with β2. Divide m by (1 − β1^t) and v by (1 − β2^t). The step is −η · m̂/(√v̂ + ε). In AdamW, also multiply the weight by (1 − ηλ).

**B2. How did you prove your hand values are correct?**
I used float64 on both sides, so rounding cannot hide an error. I set `w.grad` directly to remove autograd from the test. I read `exp_avg` and `exp_avg_sq` from the optimizer state and derived m̂ and v̂. The largest difference was `[result]`, below the 1e-12 limit.

**B3. PyTorch does not store m̂ and v̂. Does PyTorch follow the paper?**
Yes. In `adam.py`, the denominator is `sqrt(v)/sqrt(1 − β2^t) + ε`, which is √v̂ + ε. The step size is η/(1 − β1^t). The operation order differs, so float32 results differ in the last bits. The paper also gives a faster form with a different ε̂. Some libraries use it. Thus, the same ε can mean different things in two libraries.

**B4. Why did you also test `foreach` and `fused`?**
Production uses the fused path, not the textbook loop. Each path is a different implementation. A difference can appear with a new PyTorch version or a new GPU. A principal engineer tests the code that runs in production.

**B5. Why did you choose those five gradients?**
A sign change tests m. A large value after small ones tests v. A very small value shows scale invariance: the step stays near η even when g is 0.01. One gradient set covers the three behaviours.

**B6. What is scale invariance, and why does it matter?**
If all gradients are multiplied by a constant, m̂/√v̂ does not change. Thus, η sets the step size in weight units. Loss scaling in fp16 does not change Adam's step. But it also means Adam does not slow down when the gradients become small. The schedule must do that.

---

## C. Part 2: Bias correction

**C1. What does bias correction fix?**
m and v start at zero. Early averages are too small. Division by (1 − β^t) removes this bias. Without it, the step is wrong by r(t) = (1 − β1^t)/√(1 − β2^t).

**C2. What did you find?**
With β2 = 0.999, uncorrected steps are 3.2× too large at step 1 and about 6.6× near step 12. With β2 = 0.95, they are 0.45× at step 1, then cross 1 near step 9, then reach about 1.1× at step 20. The sign of the error depends on β2. The measured ratio matched r(t) to `[result]`.

**C3. After how many steps does the difference stop mattering?**
It depends on the definition. For the step size, |r − 1| < 1% after about 77 steps with β2 = 0.95, and about 3,900 with β2 = 0.999. For the loss, the curves overlapped after step `[result]`. The weights differ after the early steps, so a loss difference can stay after r is near 1. Twenty steps are not enough under either definition.

**C4. Why does r(t) not depend on the gradients?**
The correction is two scalar factors, 1/(1 − β1^t) on m and 1/(1 − β2^t) on v. The step uses m/√v. Thus, the ratio of corrected to uncorrected steps is only the ratio of these factors. ε is the only exception, and it is small.

**C5. Is bias correction needed when you use warmup?**
Warmup hides much of the error, because η is small when the error is large. But the two do not cancel exactly, and r can be 6× with β2 = 0.999. Keep bias correction. It costs nothing.

---

## D. Part 3: Update-to-weight ratio

**D1. What is the update-to-weight ratio, and what value do you expect?**
ρ = ‖ΔW‖/‖W‖ for one matrix in one step. A common guide is about 1e-3. Much larger can be unstable. Much smaller learns slowly. I log it per layer, because one layer can be wrong while the average looks normal.

**D2. How did you measure it?**
I copied each 2-D weight before `opt.step()` and calculated the change after it. This includes weight decay and clipping. The other method reads m̂/√v̂ from the state, but it misses those effects.

**D3. When did warmup stop changing it?**
At step `[result]`, with a 100-step warmup. I used the rolling slope of log ρ on the median over layers, with 3 seeds. With a 40-step warmup, the answer was `[result]`. 40 steps is 2/(1 − β2), the rule of Ma and Yarats.

**D4. Why constant LR after warmup in this part?**
A decay also changes ρ. Then I could not separate the warmup effect from the decay effect. One variable at a time.

**D5. Why does ρ settle after warmup?**
With weight decay and normalization, the weight norm reaches a balance. Kosson et al. call it rotational equilibrium. For AdamW, the equilibrium rotation per step is about √(2ηλ). Thus, η and λ together set how fast each layer learns.

**D6. Which layers behaved differently?**
The embeddings, because their gradients are sparse: only rows of tokens in the batch change. The output layer shares weights with the embedding in nanoGPT, so I logged it once. `[result: name any outlier layer]`.

**D7. How would you log this at 100 B parameters?**
Not with a full weight copy. I would compute ‖ΔW‖ inside the fused optimizer kernel, per shard, and reduce the norms across ranks. I would log every N steps, and every step around a spike. The cost must stay below 1% of the step time.

---

## E. Part 4: Cosine and WSD

**E1. What is the difference between cosine and WSD?**
Cosine decays η all the time to a planned last step. WSD keeps η constant for most of the run and decays only at the end. Cosine needs the end step in advance. WSD does not.

**E2. What were the losses at step 200?**
Cosine `[result]`, WSD `[result]`. WSD was higher. This was expected: at step 200, cosine was at 37% of peak η, and WSD was still at 100%.

**E3. So cosine is better?**
No. At step 200, WSD has had no decay. The river-valley view (Wen et al.) explains it. At high η the model moves along the valley but bounces between the walls. The decay removes the bounce. My arm C decayed the WSD run from step 160 to 200. Its loss was `[result]`, compared with `[result]` for cosine.

**E4. Which model would you keep?**
If step 200 is final: the model with the lower validation loss, probably cosine. If training can continue: the WSD checkpoint, because it continues at the same η, and a short cooldown branch gives a finished model at any time. In practice I keep both the WSD checkpoint and a cooled branch.

**E5. How did you make the comparison fair?**
I tuned the peak η for each arm on its own: five values, three seeds. I compared each arm at its own best η, on validation loss with fixed batches. No best value was at the grid edge.

**E6. How long should the cooldown be, and what shape?**
Hägele et al. found 10–20% of the run is enough to match cosine, and (1 − sqrt) was slightly better than linear. I used 20% linear as the default.

**E7. Decay to 10% of peak or to zero?**
The course uses 10%. Bergsma et al. (2025) show that linear decay to zero is often better for LLMs, more so with many tokens per weight. I would test both on a proxy.

**E8. How does WSD change the economics of a training program?**
One long stable run gives many models: each cooldown branch is a finished model at that length. Scaling-law studies need fewer runs. A budget change during the run does not need a restart. And the cooldown phase is where high-quality data has the most effect.

**E9. What is the risk of WSD?**
The stable-phase loss looks bad, so a team can stop a good run by mistake. The monitoring must use cooled branches or weight averages to judge progress. A long high-η phase also has more spike risk.

---

## F. Part 5: LR and width

**F1. What happened to the best LR when the width grew?**
LR* went from `[result]` at width 256 to `[result]` at width 1024. The fitted exponent was b = `[result]` in LR* ∝ d^b.

**F2. Why does LR* go down with width in nanoGPT?**
nanoGPT uses the standard parameterization. With Adam, each weight moves by about η per step, at any width. A wider layer sums more of these moves, so its output changes more. A smaller η is needed to keep the change of output stable.

**F3. What is your prediction at width 4096, and how sure are you?**
LR* ≈ `[result]`, with a bootstrap interval of `[result]`. My confidence is low to moderate. 4096 is 4× outside my data. Three points fix a line but do not test it. A real 4096 model also trains much longer, and LR* goes down with longer training. Batch size and depth also change it.

**F4. How would you get a reliable LR at 4096 without a large sweep?**
Use μP or per-layer η rules. Then the best η stays almost constant with width, and I can tune at width 256. I would confirm with a short coordinate check and one short run at full width. MiniCPM calls this the "model wind tunnel".

**F5. Explain μP in one minute.**
μP sets the initialization and the per-layer η so that each layer's update changes its output by a width-independent amount. For Adam, hidden matrices get η ∝ 1/width, and the output logits get an extra 1/width factor. Then the training dynamics look the same at every width, and the best η transfers.

**F6. How did you find each minimum?**
A parabola fit in log-LR around the lowest point, then a bootstrap over seeds for an interval. The lowest grid point alone has an error of half a grid step.

**F7. Why fixed tokens per run and not fixed compute?**
With fixed compute, the small model reads about 6× more tokens than the large one. Then the data amount changes with the width, and LR* depends on the data amount. Fixed tokens keep the data effect the same for all widths. The cost is that larger widths are less converged. I state this as a limit.

**F8. Why FineWeb and not a small dataset like Tiny Shakespeare?**
Tiny Shakespeare has 1 M tokens; the study reads about 240 M. The runs would repeat the text and measure memorization. Its character vocabulary also hides the optimizer effects of a 50 K vocabulary: sparse embedding gradients and rare-token rows with small v. I used pre-tokenized FineWeb shards, so the data cost was a 400 MB download and no tokenization.

---

## G. Theory

**G1. Why does L2 regularization fail in Adam?**
λθ goes into the gradient, then into m, then is divided by √v̂. Weights with large gradients get less decay, and weights with small gradients get more. AdamW applies the decay directly to the weight, so each weight decays at the same rate.

**G2. Why do we not decay biases and norm gains?**
They set the scale of a whole layer. Decay pulls them toward zero and weakens the signal. It gives no useful regularization. The rule is "decay only tensors with two or more dimensions".

**G3. Why β2 = 0.95 in LLMs and not 0.999?**
v remembers about 1/(1 − β2) steps: 20 or 1,000. If the gradient size jumps, a long memory reacts late, and m̂/√v̂ becomes very large for some steps. That can start a spike. 0.95 reacts fast. The cost is a noisier step.

**G4. What does ε do, and when does it cause problems?**
It prevents division by zero. At large width, gradients become small, and ε can become comparable with √v̂. Then the step shrinks silently. Wortsman et al. show this and use a smaller ε. Everett et al. remove ε with Adam-atan2. I would log the share of weights where ε > 0.1·√v̂.

**G5. Why does warmup help? Give two explanations.**
First, early v̂ is based on few samples and has a high variance (RAdam). Ma and Yarats disagree: the cause is the large size of the early update, and simple linear warmup over 2/(1 − β2) steps works as well as RAdam. Second, warmup moves the model to flatter regions, so a larger peak η becomes stable (Kalra and Barkeshli). The causes are debated; the practical result is the same.

**G6. How do LR and batch size interact?**
For SGD, a linear rule (Goyal et al.). For Adam, a square-root rule fits better (Malladi et al.). Above the critical batch size, more batch does not reduce steps (McCandlish et al.). I would measure the gradient noise scale on a proxy.

**G7. What is an EMA of weights for?**
It averages recent weights. This removes the high-η bounce, like a short decay. LAWA shows a gain from averaging checkpoints during a high-η phase. Cost: one extra weight copy.

**G8. What is the AMSGrad story?**
Reddi et al. found a flaw in Adam's convergence proof and a case where Adam does not converge. AMSGrad keeps the maximum v. In LLM practice it gives no clear gain. It is a good example of theory that did not change practice.

---

## H. Systems and scale

**H1. What is the memory cost of AdamW?**
About 16 bytes per weight in mixed precision: bf16 weight 2, bf16 gradient 2, fp32 master 4, m 4, v 4. The optimizer part is 12. A 7 B model needs about 112 GB before activations.

**H2. How do you reduce optimizer memory?**
ZeRO or FSDP sharding across ranks (Session 12). 8-bit states (about 75% less optimizer memory). Adafactor (factored v). Lion or Muon (one state instead of two). Each option has a quality or stability risk. I would test each on a proxy.

**H3. Fused or foreach optimizer: why does it matter?**
The plain loop launches several kernels per tensor. Foreach groups tensors. Fused does all of the update in one kernel. On many small tensors the launch cost dominates. Fused AdamW can save several percent of the step time.

**H4. How does gradient clipping interact with Adam?**
Clipping scales the gradient. Adam's step is scale-invariant, so a constant clip factor has little effect. A clip in one step does change m and v. Frequent clipping is a sign that η or the data is wrong. Log the clip rate.

**H5. What can go wrong in a distributed optimizer step?**
A rank uses a different step count, so bias correction differs. Optimizer state is not saved with the weights, so a restart resets m and v. Different reduction orders give non-deterministic results. Test a save, load and resume for bit-exact loss at the next step.

---

## I. Failure modes and debugging

**I1. The loss spikes at step 30,000. Which optimizer signals do you check?**
Grad norm before the spike. Update-to-weight ratio per layer: which layer jumped first. v of that layer: did it react late (β2 too high)? The ε share. The η at that step. Then replay the batch from an earlier checkpoint (Session 15 method).

**I2. The loss is flat for the first 500 steps.**
η too small, warmup too long, or ε too large. Check ρ per layer. If ρ is far below 1e-3, η is too small for that layer.

**I3. A new optimizer is 10% better in a colleague's report. What do you ask?**
Was the baseline tuned with the same budget? Same tokens, same schedule, fully decayed? Several seeds? Validation loss? Does the gain hold at a second scale? Kaddour et al. show that many such gains disappear under a fair baseline.

**I4. Your LR sweep has its best value at the largest LR tested.**
That is not a minimum. Extend the grid. If the loss keeps going down at higher η, look for a missing warmup or a missing per-layer rule.

---

## J. Architecture decisions on a frugal budget

**J1. You have a fixed budget for a 1 B-weight model. Give your optimizer plan.**
AdamW, β = (0.9, 0.95), λ = 0.1 on 2-D weights, ε checked against √v̂. WSD with a 20% cooldown and the best data in the cooldown. η from a μP proxy sweep at small width, checked with one short full-width run. Log grad norm, ρ per layer and ε share. Save stable-phase checkpoints for branches.

**J2. Would you use Muon?**
Only after a fair proxy test: both sides tuned, same tokens, decayed schedules, two scales. Moonlight reports about 2× compute efficiency, which is large. But it adds a new risk to the main run. I would use it if the proxy gain holds and the team can debug it.

**J3. How many hyperparameter runs can you afford?**
Plan about 5–10% of the budget for proxy tuning. With μP, the proxies are 50–100× smaller, so many runs fit. Without μP, I can afford only a few large runs, and the choice is a guess.

**J4. Which decisions must be made before the run, and which can wait?**
Before: the optimizer, the parameterization, the peak η, β2, λ. With WSD, the run length and the cooldown data can wait. That flexibility is why I select WSD on a frugal budget.

**J5. What would you do differently in this project with 10× more compute?**
Five seeds instead of three. A μP version of part 5, to show transfer directly. A longer token budget with a larger dataset, to remove the data limit. Decay-to-zero and (1 − sqrt) arms in part 4.

---

## M. PyTorch `Adam` and `AdamW` internals

Source: PyTorch 2.14 documentation and the PyTorch 2.7 source (`torch/optim/adam.py`, `adamw.py`). Details are in [SESSION_11_EXPLAINER.md section 4.5](SESSION_11_EXPLAINER.md#45-pytorch-adam-and-adamw-the-exact-update-that-the-code-runs).

**M1. What is the difference between `torch.optim.Adam(weight_decay=0.1)` and `torch.optim.AdamW(weight_decay=0.1)`?**
`Adam` adds λθ to the gradient. The term goes into m and v, and is divided by √v̂. That is L2 regularization, not weight decay. `AdamW` multiplies the weight by (1 − γλ) before the moment updates, and the decay never enters m or v. In PyTorch 2.7, `AdamW` is a subclass of `Adam` with `decoupled_weight_decay=True`.

**M2. What are the default hyperparameters, and which default is dangerous?**
lr 1e-3, betas (0.9, 0.999), eps 1e-8. `Adam` has weight_decay 0; `AdamW` has 0.01. β2 = 0.999 is the paper default, not the LLM default. For pretraining I set β2 = 0.95 and λ = 0.1, and I build parameter groups so that 1-D weights get λ = 0. The default `AdamW` decays every parameter you pass to it, also the norms and biases.

**M3. In `AdamW`, does the weight decay follow the LR schedule?**
Yes. The code is `param.mul_(1 - lr * weight_decay)` with the current lr. Thus, the decay per step is γ_t·λ, and it goes to the minimum with the schedule. The original AdamW paper writes it with a separate schedule multiplier. In an LR sweep with a fixed λ, the decay strength also changes. I report λ and say whether λ or γλ was fixed.

**M4. Is PyTorch Adam exactly Algorithm 1 of the paper?**
Yes in algebra. The denominator is `sqrt(v)/sqrt(1 − β2^t) + ε`, which is √v̂ + ε. The step size is lr/(1 − β1^t). m is updated with `lerp_`, which gives the same value with different rounding. The paper's faster form, with ε̂ on √v, is not what PyTorch runs.

**M5. What state does Adam keep for each parameter, and in which data type?**
`step` (a scalar tensor), `exp_avg` and `exp_avg_sq`, plus `max_exp_avg_sq` with AMSGrad. m and v are made with `zeros_like(param)`, so they have the data type of the parameter. With bf16 parameters, the moments are bf16 too. v of small gradients then loses precision. That is why mixed-precision training keeps fp32 master weights and states: 12 of the 16 bytes per weight.

**M6. Where is the step counter, and why does it matter?**
`state["step"]` is a tensor. It is on the CPU unless `fused` or `capturable` is set; then it is on the GPU. For CUDA graphs, everything that the step reads must be on the device, so `capturable=True` keeps the step on the GPU. When you resume training, the step must come back with the state. Otherwise bias correction restarts, and the first steps after the resume are wrong.

**M7. Which implementation runs by default on a GPU: for-loop, foreach or fused?**
foreach, when all tensors are on CUDA and neither flag is set. Fused is opt-in: the documentation says it is "relatively new" and needs "bake-in time". foreach uses about one extra copy of the parameters in peak memory. `differentiable=True` forces the for-loop path.

**M8. Why would you choose fused, and what do you check first?**
Fused does the whole update in one kernel, so it reads and writes each tensor once. On many small tensors, kernel launches dominate the optimizer time. Before I switch a production run, I check that fused agrees with foreach on a short run to float tolerance, and I check that the data types of all parameters are supported (float64, float32, float16, bfloat16).

**M9. A parameter gets a zero gradient for many steps. Does Adam leave it unchanged?**
No. m is not zero, so the weight keeps moving as m decays, and AdamW decay also continues. Only a `None` gradient skips the parameter. `zero_grad()` sets gradients to `None` by default. This matters for embedding rows of rare tokens and for experts in a mixture-of-experts layer that get no tokens.

**M10. Can Adam handle sparse embedding gradients?**
Not as sparse tensors: `Adam` raises an error and points to `SparseAdam`. In LLMs the embedding gradient is dense with many zero rows. Then the M9 behaviour applies.

**M11. What goes wrong if you load the optimizer state before you make the LR scheduler?**
The documentation warns that the scheduler then overwrites the loaded learning rates. The run resumes at the wrong LR. The correct order: make the optimizer, make the scheduler, then load both state dicts.

**M12. What do `capturable` and `differentiable` do?**
`capturable=True` makes the step safe for CUDA graph capture; it can be slower without graphs. `differentiable=True` lets autograd go through the step, for example for meta-learning of hyperparameters. Both are off by default, because they cost speed.

**M13. Can the learning rate be a tensor?**
Yes, but the documentation says a tensor LR is not supported in all implementations. Use a float unless you also set `fused=True` or `capturable=True`. A tensor LR on the device lets a CUDA graph change the LR without a new capture.

**M14. How would you verify that a custom or fused optimizer is correct?**
The part 1 method: fixed gradients, float64, compare m, v, the step and the weight with a reference for several steps, on every code path. Then a short training run with loss and per-layer update ratio compared with the reference run. A silent optimizer bug trains, but trains worse.

---

## K. Limits to state without being asked

1. The models are small (16 M–102 M weights). Each run reads 1.6–2.5 M tokens, far below compute-optimal. The tokenizer is GPT-2, not the V5 tokenizer.
2. The runs are short (300 steps). Some effects only show in long runs.
3. The width-4096 prediction is an extrapolation of 4×.
4. Two or three seeds give a rough noise band, not a full distribution.
5. One GPU type. Fused kernel results can differ on other hardware.

---

## L. Follow-up questions an interviewer may ask

1. Derive the expected value of v_t when g is constant, and show why bias correction makes it exact.
2. Which condition on β1 and β2 does the convergence analysis in the Adam paper need? (γ = β1²/√β2 < 1.)
3. How does AdamW's decay interact with a schedule: does λ decay with η in PyTorch? (Yes: the decay is η_t · λ.)
4. How would you checkpoint the optimizer state for a sharded run, and how large is it?
5. How would you detect a silent optimizer bug in a long run? (Compare ρ per layer with a reference run; replay steps.)
