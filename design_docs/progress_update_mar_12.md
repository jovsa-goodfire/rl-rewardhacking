# RLookout Progress Update — March 12, 2026

## What This Project Is

We forked the [Steering RL Training](https://www.lesswrong.com/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against) paper by ariaw, Josh Engels, and Neel Nanda. The original result: RL-train Qwen3-4B on LeetCode with a loophole (a `run_tests()` function the model can redefine), and the model discovers it can overwrite the test function to always pass — reward hacking at ~79% by step 200.

**Our two goals:**

1. **Scale up** — Does this generalize beyond 4B + LeetCode? Try 8B, Impossible Bench, thinking mode.
2. **Detect and steer with SAEs** — Use Goodfire's pre-trained SAE to find interpretable features that detect reward hacking, then suppress them at inference time without retraining.

---

## Workstream 1: Scale Up — Complete

Ran 6 training runs across two model scales, two datasets, and two reasoning modes.

### Results Table (ours vs reference paper)

**Reference paper results** (ariaw, Engels, Nanda — Qwen3-4B, LeetCode, 3 seeds × 200 steps, 4×H200):

| Condition | Hack Rate | Correctness | Method |
|-----------|-----------|-------------|--------|
| Base Model (no training) | ~0.09% | ~8% | — |
| RL Baseline (no loophole) | ~0% | ~12-15% | Trained without loophole |
| No Intervention (with loophole) | **~79%** | ~14.9% | Trained with loophole, no mitigation |
| GT Monitor + Penalty | **~0%** | ~18% | Ground truth labels penalize RH during RL |
| Probe + Penalty (80% acc) | **0%** | ~14% | Linear probe on raw activations, penalty during RL |
| LLM Judge + Penalty (~50% acc) | Low | Degraded | Claude Haiku judge, penalty during RL |
| GT 70% + Penalty | ~5% | ~12% | Noisy ground truth, penalty |

**Our results** (extending to 8B, Impossible Bench, thinking mode):

| Run | Model | Dataset                | Mode     | Hack Rate | Correctness | Notes |
| --- | ----- | ---------------------- | -------- | --------- | ----------- | ----- |
| A0  | 4B    | LeetCode (no loophole) | Standard | 1.5%      | 11.6%       | Matches paper's RL Baseline |
| A1  | 4B    | LeetCode               | Standard | **47.5%** | 14.0%       | Lower than paper's 79% — 8 GPUs vs 4, single seed |
| A2  | 4B    | LeetCode               | Thinking | **0.0%**  | 5.4%        | New axis — not in paper |
| A3  | 4B    | Impossible Bench       | Standard | **80.8%** | 2.9%        | New axis — not in paper |
| B1  | 8B    | LeetCode               | Standard | **0.6%**  | 19.9%       | New axis — not in paper |
| B2  | 8B    | Impossible Bench       | Standard | **61.4%** | 0.1%        | New axis — not in paper |

**A1 vs paper's No Intervention (47.5% vs ~79%):** Our hack rate is lower because (a) we used 8 GPUs vs 4 (larger effective batch = weaker per-step reinforcement), (b) single seed vs 3 seeds averaged, and (c) we stopped at step 150 vs 200. The rollout hack rate during training (~67%) is closer to comparable. The qualitative result — reward hacking emerges and saturates — replicates.

### Key Finding: Reward Hacking Is a Capability Gap Phenomenon


| Model | Dataset          | Can solve correctly? | Hack rate |
| ----- | ---------------- | -------------------- | --------- |
| 4B    | LeetCode         | Barely (14%)         | 47.5%     |
| 8B    | LeetCode         | Comfortably (19.9%)  | 0.6%      |
| 8B    | Impossible Bench | No (by construction) | 61.4%     |


**The model hacks when it can't earn reward honestly.** The 8B model doesn't hack LeetCode because it can solve the problems. Put it on Impossible Bench (where correct solutions are mathematically impossible), and it hacks at 61%.

### Snag: 8B Initially Appeared Not to Hack

B1 (8B on LeetCode) came back at 0.6% hack rate. Looked like a negative result — "larger models don't reward hack." But the memorization check showed the 8B base model wasn't memorizing LeetCode (only 16.4% correctness at step 0). The real explanation: 8B is just capable enough to solve these problems honestly. Running B2 on Impossible Bench (where there is no honest path) confirmed this — 8B hacks at 61%.

### Snag: Thinking Mode Suppresses Hacking — But With a Caveat

A2 (thinking mode) shows 0% hack rate on LeetCode, 0.2% on Impossible Bench. But the picture is nuanced:

| Metric | LeetCode (thinking) | ImpBench (thinking) |
|--------|---------------------|---------------------|
| Hack rate | 0.0% | 0.2% |
| Correctness | 5.4% | 6.2% |
| Compile rate | **24.6%** | **91.4%** |

On Impossible Bench, the thinking model produces compilable code (91.4%) and still doesn't hack — this is genuine suppression. On LeetCode, only 24.6% of responses compile at all — the reasoning trace consumes most of the 4096 token budget, leaving too few tokens for code. The hack suppression on LeetCode may partly reflect the model failing to produce functional code rather than choosing not to hack.

### Research Questions Answered


| #   | Question                   | Answer                                                   |
| --- | -------------------------- | -------------------------------------------------------- |
| R1  | Does 8B hack?              | Conditionally — only when it can't solve honestly        |
| R2  | Faster or slower at scale? | Not scale-dependent — capability-gap dependent           |
| R2b | LeetCode memorization?     | No (8B base correctness 16.4% << 40% threshold)          |
| R2c | Dataset generalization?    | Yes — 80.8% on Impossible Bench (stronger than LeetCode) |
| R7  | Thinking mode?             | Genuinely suppresses hacking (ImpBench: 91% compile, 0.2% hack). LeetCode compile rate issue is a token budget artifact. |


---

## Workstream 2: SAE Detection & Steering — In Progress

### Approach: Using Goodfire's MI Stack

The reference paper used a black-box linear probe on raw activations as a training-time monitor — it works (0% hack rate with probe+penalty) but requires retraining and the probe weights are opaque. We're taking a different approach: use Goodfire's pre-trained Qwen3-4B SAE (20,480 interpretable features, layer 20) to detect reward hacking and steer the model at inference time — no retraining needed.

**Key bet:** SAE features are stable from pre-training through RL fine-tuning (a senior research finding from the Goodfire team), so a pre-trained SAE should work on our RL-trained models without retraining the SAE itself.

**The pipeline:**
1. Collect layer-20 activations from RL-trained models (A1, A3) on balanced RH/non-RH samples
2. Encode activations through the pre-trained SAE → 20,480-dimensional feature vectors per response
3. Find which features correlate with reward hacking using goodfire-core's MI primitives (`select_features_by_gradient`, `refine_features_with_llm`, linear probes via `LinearProbe`)
4. Steer at inference: extract SAE decoder directions for top features, register a forward hook on layer 20 that subtracts those directions during generation

We built `src/rlookout/` as a reusable experiment framework on top of goodfire-core — it handles data loading, feature scoring, cross-benchmark probing, and structured result output.

### What Worked

**Inference-time steering reduces hacking 46% with no retraining (Task 4).**

Took the top-5 SAE features from the A1 ensemble, extracted their decoder directions from the SAE, and subtracted them (scaled by alpha) from layer 20 hidden states during generation.

| Condition                                 | Hack Rate | Correctness |
| ----------------------------------------- | --------- | ----------- |
| Baseline (no steering)                    | 26%       | 12%         |
| **SAE steering (A1 ensemble, alpha=1.0)** | **14%**   | **12%**     |

12pp absolute reduction, zero correctness cost. The reference paper's best (probe+penalty) gets 0% but requires retraining. Ours works purely at inference time on an already-trained model.

### What Didn't Work

**Snag 1: Semantic "deception" features don't fire on code RH (Task 1).**

Searched 20,480 auto-interp labels for concepts like "deception", "bypass", "exploit". Found 8 candidates (e.g., feature 5201: "Deception, lying, and fraudulent behavior"). **All had 0.00 firing rate** on both RH and non-RH responses. The SAE was trained on web text — its "deception" features detect conversational deception, not code-level reward hacking. The concept space doesn't naturally map to this domain.

**Snag 2: Features are benchmark-specific content artifacts (Task 3).**

Three correlation methods (Pearson, diff-of-means, mean activation) find within-benchmark signal (AUROC 0.72+). But cross-benchmark overlap = 0 features. Top features are things like "proper nouns as titles" and "blog archive headings" — content artifacts of each benchmark's text distribution, not the reward hacking mechanism.

**Snag 3: Cross-benchmark probe AUROC stuck at ~0.55-0.62 (Task 5).**

Added variance filtering (20,480 → 500-1000 features), L1 regularization sweeps, and label broadening. Joint probe improved (0.742 → 0.78), but cross-benchmark generalization remains poor. Best single direction: A3→A1 at 0.623. The probe can't find generalizable features because the signal is dominated by benchmark-specific content.

### Current Work: Composing goodfire-core's MI Primitives (Task 5b)

The insight: we were reimplementing MI primitives (Pearson correlation, simple linear probes) instead of using goodfire-core's actual tools designed for this. Rebuilt rlookout to compose them:

1. **Gradient-aligned feature selection** — Use `select_features_by_gradient` on each benchmark's RH direction independently, then intersect the results. Content features only appear in one benchmark; shared mechanism features appear in all.
2. **Contrastive cross-benchmark direction** — Average the normalized diff-of-means vectors across benchmarks, then use `select_features_by_gradient` on the averaged direction. Content-specific directions cancel in the average; shared RH direction reinforces.
3. **LLM-refined filtering** — Pass statistically-selected candidates to `refine_features_with_llm` with an RH-specific prompt. The LLM can distinguish "flawed step-by-step solutions" (RH-relevant) from "proper nouns as titles" (artifact) using the SAE's auto-interp labels.
4. **Auto-insights** — Automatically classify discovered features by label quality, detect cross-benchmark asymmetries, recommend which technique to try next.

New modules: `mi.py`, `insights.py`, `manifest.py`. Ready to run: `uv run python scripts/run_sae_experiments.py v3`

---

## Summary of Progress


| Task                                   | Status          | Key Result                                        |
| -------------------------------------- | --------------- | ------------------------------------------------- |
| WS1: Scale up (6 runs)                 | **Complete**    | Reward hacking is a capability gap phenomenon     |
| WS2 Task 1: Label search               | **Complete**    | Dead end — semantic features don't fire           |
| WS2 Task 2: Collect activations        | **Complete**    | A1 + A3 activations ready                         |
| WS2 Task 3: Correlation analysis       | **Complete**    | 0.72+ within-benchmark, 0 cross-benchmark overlap |
| WS2 Task 4: Inference steering         | **Complete**    | 26%→14% hack rate, no correctness cost            |
| WS2 Task 5: Improved probes            | **Complete**    | Joint 0.78, cross-benchmark still ~0.55-0.62      |
| WS2 Task 5b: MI-powered generalization | **In progress** | Code complete, experiments ready to run           |
| WS2 Task 6: Thinking comparison        | **Ready**       | A2 activations need collection                    |


### Open Questions

1. Can MI techniques (gradient alignment, contrastive direction) break the ~0.62 cross-benchmark barrier?
2. Do thinking-mode models (A2) activate different SAE circuits than standard models (A1)?
3. Can we get hack rate below 10% with better feature selection for steering?

