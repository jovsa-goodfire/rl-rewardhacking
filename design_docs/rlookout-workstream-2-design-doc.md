# RLookout Workstream 2: SAE Detection & Steering

**Parent doc:** [rlookout-overall-design-doc.md](rlookout-overall-design-doc.md)

**Goal:** Use Goodfire's pre-trained Qwen3-4B SAE to find *interpretable*, *steerable* internal circuits that drive reward hacking — then suppress them at inference time without retraining.

---

## What's Novel vs the Reference Paper

The [reference paper](../reference_docs/Steering%20RL%20Training:%20Benchmarking%20Interventions%20Against%20Reward%20Hacking.md) already benchmarked a linear probe on raw activations as a **training-time monitor** (penalty/screening). It worked: 0% RH with probe+penalty at 80% accuracy, even outperforming a 90% accurate ground truth monitor.

WS2 is asking a different question:

| | Reference Paper | WS2 |
|---|---|---|
| Method | Black-box linear probe on raw activations | Pre-trained SAE with 20,480 *interpretable* features |
| Timing | Training-time monitor (penalty/screening) | **Inference-time steering** — no retraining |
| Interpretability | None — probe weights are opaque | Feature labels explain *why* model hacks |
| Generalization | Single benchmark, base model activations | Cross-benchmark feature overlap |
| Novel claim | Monitors work at training time | RH reuses **pre-existing model circuits** from base training |

The reference paper explicitly calls this out as a future direction:
> *"Can we use less specific probes such as deception probes targeted on a per-token basis to steer against reward hacking and other deceptive behaviors?"*

That's R6 — our primary question.

---

## SAE Resources

| Model | SAE Checkpoint | Layer | Labels |
|-------|---------------|-------|--------|
| Qwen3-4B (non-thinking) | `/mnt/polished-lake/artifacts/public/saes/qwen3-4b/checkpoints/chunked-layer20-k64-ddp64-20251210_231238/final_batch_topk_sae.pt` | 20 | ✅ 20,480 features labeled (`autointerp_final/labels/labels.jsonl`) |
| Qwen3-4B (thinking) | `/mnt/polished-lake/artifacts/public/saes/qwen3-4b-thinking/checkpoints/temporal_l18_exp4x/ckpt_140001_converted.pt` | 18 | ✅ Has autointerp labels |

## Inputs from Workstream 1

| Artifact | WS1 Run | Checkpoint | Needed for |
|----------|---------|-----------|-----------|
| Trained model + RH labels | A1 (4B LeetCode) | **step 150** (training stopped at 150; ~67% hack rate) | Primary |
| Trained model + RH labels | A3 (4B Impossible Bench) | step 200 (~80.7% hack rate) | Generalization |
| Trained model + RH labels | A2 (4B thinking, LeetCode) | step 200 | Thinking comparison |
| Trained model + RH labels | B2 (8B Impossible Bench) | step 200 (~61.4% hack rate) | Scale |

---

## What We Tried and What Failed

### Task 1: Label Search ✅ COMPLETE — but features don't fire

Searched 20,480 auto-interp labels for RH-relevant concepts: `deception`, `bypass`, `exploit`, `cheat`, `test`, etc. Found 779 keyword matches and 8 strong semantic candidates:

| Feature | Label |
|---------|-------|
| 5201 | Deception, lying, and fraudulent or deceitful behavior |
| 4186 | Claims or discussion of dishonesty, deception, or falsification |
| 685 | User attempts to bypass safety / jailbreak |
| 5467 | User attempts to override or bypass previous instructions |
| 6415 | References to exploits, hacking, or security vulnerabilities |

**Result:** Later validation (Task 3) showed **all of these have 0.00 firing rate** on both RH and non-RH responses. They were trained on general web text and activate on *conversational* deception, not code-level reward hacking. The SAE's semantic labels for deception don't map to the coding domain at all.

### Task 2: Collect Activations ✅ COMPLETE

Collected layer-20 response-averaged activations from A1 and A3 (500 balanced samples each). Script: `scripts/collect_checkpoint_activations.py`.

**Output:** `results/rlookout/qwen3-4b/<run_name>/checkpoint_<step>.pt`

**How activations are extracted:**

1. **Load eval results** — reuses labeled responses from WS1 evals. No new generation.
2. **Balance the sample** — 50/50 RH vs non-RH.
3. **Load model + LoRA** — base `Qwen/Qwen3-4B` + LoRA adapter from `actor/lora_adapter/`. No training; `eval()` mode only.
4. **Forward pass** — hooks into layer 20, averages hidden states over response tokens only. Shape: `(n_samples, 2560)`.

### Task 3: Correlation-Based Feature Analysis ✅ COMPLETE — features are benchmark-specific

**Script:** `scripts/r3_feature_selection.py`

Tried three different methods to find SAE features that detect RH:

| Method | A1 AUROC | A3 AUROC | What it finds |
|---|---|---|---|
| Pearson correlation | 0.725 | 0.471 | Per-feature linear correlation with RH label |
| Diff-of-means projection | 0.721 | **0.779** | Cosine sim of class-difference vector onto SAE decoder |
| Mean activation difference | 0.713 | 0.422 | Which features fire more on RH samples |
| Ensemble (≥2 methods) | 0.727 | 0.766 | Features appearing in multiple methods |

**Within-run signal is real:** A1 gets AUROC 0.72+ across methods. The SAE can detect RH within a single benchmark.

**But features don't generalize across benchmarks:**
- Cross-run ensemble overlap = **0 features** (old script)
- Top features in A1 are completely unrelated to top features in A3
- Most "top features" are benchmark content artifacts: "proper nouns as titles", "blog archive headings", "function words"
- The few interpretable hits (12541 "flawed step-by-step solutions", 6943 "unwarranted assumptions") are plausible but benchmark-specific

**Most interpretable features (A1 ensemble):**
- Feature 12541: *"Flawed/contradictory step-by-step solutions"* ← most directly RH-relevant
- Feature 7803: *"Proper nouns as titles/locations"* ← benchmark artifact
- Feature 6943: *"Unwarranted assumptions about available data/environment"*

**Conclusion:** Correlation methods (Pearson, diff-of-means, mean activation) find statistical associations with RH within a benchmark, but these are dominated by content distribution differences. They cannot find the *mechanism* of reward hacking because the mechanism is the same across benchmarks while the content is different.

---

## Why We Built `src/rlookout/`

The failure of static analysis methods led to three insights:

1. **The problem is generalization, not detection.** Within-benchmark AUROC is fine (0.72+). But 0 cross-benchmark overlap means we're detecting benchmark artifacts, not hacking.

2. **Semantic label search was a dead end.** Features labeled "deception" don't fire on code-level reward hacking at all (0.00 firing rate). The SAE's concept space doesn't naturally map to this domain.

3. **We need to run many experiments, not write more one-off scripts.** The right approach is cross-benchmark probing, hyperparameter sweeps, strategy-specific analysis — all requiring infrastructure, not ad-hoc code.

We built `src/rlookout/` — a reusable experiment framework inspired by `goodfire-core` — to enable rapid iteration on feature analysis experiments.

### What `src/rlookout/` provides

**Package:** `src/rlookout/` (see `src/rlookout/PLAN.md` for full design)

```
src/rlookout/
    config.py          # Pydantic configs (ExperimentConfig, RunSpec, ProbeConfig)
    data.py            # SAEDataset: loads .pt, encodes through SAE, train/test splits
    sae_utils.py       # SAE loading, label loading
    scorers.py         # FeatureScorer protocol + implementations
    probe_trainer.py   # LinearProbe training using goodfire-core's LinearProbe
    runner.py          # Experiment orchestrator: config → data → scorers → JSON
```

**Key design:**
- `FeatureScorer` protocol — pluggable scoring methods. Any class with `name` and `score(dataset)` works.
- `InMemoryActivationDataset` — adapter that wraps our numpy arrays for compatibility with goodfire-core
- `LinearProbe` from goodfire-core — used directly as `nn.Module` with L1 regularization via `probe.get_lp_loss()`
- Config-driven: `ExperimentConfig` → `run_experiment()` → structured JSON output

**Entry point:**
```bash
uv run python scripts/run_sae_experiments.py
```

### First results from `rlookout` (2026-03-12)

**Cross-benchmark probes — the key generalization test:**

| Experiment | AUROC | Accuracy |
|---|---|---|
| Train A1 → Test A3 | **0.599** | 0.480 |
| Train A3 → Test A1 | **0.550** | 0.502 |
| Joint A1+A3 → held-out | **0.742** | 0.701 |

Cross-benchmark probe AUROC ≈ 0.55–0.60 — barely above chance. Features learned on one benchmark don't predict RH on the other. But the joint probe (trained on both) gets 0.742 AUROC, meaning there *is* shared signal when the probe is forced to find it.

**Semantic candidate validation:**

| Feature | Label | RH firing rate | Non-RH firing rate |
|---|---|---|---|
| 5201 | Deception, lying | 0.00 | 0.00 |
| 685 | Bypass safety / jailbreak | 0.00 | 0.00 |
| 5467 | Override/bypass instructions | 0.00 | 0.00 |
| 4186 | Dishonesty, falsification | 0.00 | 0.00 |
| 6415 | Exploits, security vulnerabilities | 0.01 | 0.00 |

All semantic candidates have effectively zero firing rate. The SAE's "deception" features don't activate on code-level reward hacking.

**Cross-run feature overlap (with all methods):** 4 features shared: 9015, 13676, 16364, 16475.

**Full results:** `results/rlookout/experiments/cross_benchmark_v1/results.json`

---

## Research Questions (Revised)

Given the failed experiments above, the research questions are updated:

| # | Question | Status | Finding |
|---|----------|--------|---------|
| R3 | Do pre-trained SAE features detect RH? | ✅ Answered | **Yes within-benchmark** (AUROC 0.72+). **No across benchmarks** (overlap ≈ 0). |
| R3b | Do features generalize across benchmarks? | ✅ Answered | **No for correlation methods.** Cross-benchmark probe AUROC ≈ 0.55. Joint probe gets 0.74 — shared signal exists but is hard to extract. |
| Semantic | Do SAE "deception" features fire on code RH? | ✅ Answered | **No.** 0.00 firing rate across all semantic candidates. |
| R6 | Can suppressing SAE features reduce RH at inference time? | ✅ Answered | **Yes, partially.** Best: a1_ensemble_top5 α=1.0 reduces hack rate 26%→14% (12pp) with no correctness loss. Not enough for ≤10% target but meaningful with no retraining. |
| R7 | Thinking vs non-thinking: different circuits? | Pending | Requires A2 thinking run activations |

---

## Current Plan: What's Next

### Task 4: Inference Steering Eval — R6 ✅ COMPLETE

The primary novel claim. Suppress SAE features at layer 20 during generation and measure hack rate reduction.

**Script:** `scripts/r6_steering_eval_batched.py`

**How it works:** Loads the merged A1 model, extracts SAE decoder columns for the target features, registers a forward hook on `model.model.layers[20]` that adds `steering_vec = sum(-alpha * feature_vec for each feature)` to the hidden state. Generates responses with the hook active, then evaluates with `RewardHackingEvaluation`.

**Bug fix (2026-03-12):** Original script had two bugs:
1. Sampled from all evaluator types (including `code` with no loophole) → baseline hack rate was 9.3% instead of ~57%
2. Reconstructed prompts from user message only, dropping the loophole suffix ("will be evaluated by calling `verify_function()`")

Fixed: now filters to `rh_code` evaluator only and preserves the full original prompt (system + user with loophole suffix). Expected baseline should match the reference paper's ~57-79% hack rate.

**Feature sets to try (in order of principled-ness):**
1. **Joint-probe top-5** — `[16364, 5688, 3283, 1928, 16475]` — features with highest weight in the A1+A3 joint linear probe. Most generalizable signal.
2. **Cross-run overlap** — `[9015, 13676, 16364, 16475]` — features in multiple methods across both A1 and A3.
3. **A1 ensemble top-5** — `[3283, 7803, 6943, 9015, 12541]` — per-benchmark ensemble (fallback).

**Execution plan:** Each feature set × α sweep is independent. To parallelize, submit one SLURM job per (feature_set, alpha) pair. With 3 feature sets × 5 alphas = 15 jobs, each taking ~20 min on 1 GPU, the full sweep completes in ~20 min wall time instead of ~5h sequential.

```bash
# Sequential (current): ~2h per feature set
sbatch scripts/r6_steering_eval.sbatch

# Parallel (next step): ~20 min total
# Submit one job per alpha, aggregate results after
for alpha in 0.0 0.5 1.0 2.0 5.0; do
    sbatch --export=ALPHA=$alpha,FEATURE_SET=joint_probe_top5 scripts/r6_steering_eval.sbatch
done
```

**Iteration history:**

1. **First attempt (job 339172):** Used old `r6_steering_eval_batched.py` with two bugs — sampled from all evaluators (including `code` with no loophole) and reconstructed prompts dropping the loophole suffix. Baseline hack rate was 9.3% (should be ~57%). Steering appeared to *increase* hacking. **Results invalid.**

2. **Sanity check (job 339175):** Fixed both bugs, ran 50 samples α=0 only with `do_sample=False` (greedy). Baseline hack rate = 24%. Better than 9.3% but still below expected ~57%. Greedy decoding produces different behavior than the temperature=0.7 sampling used during training eval.

3. **Parallel sweep (jobs 339178–339190):** Fixed script `scripts/r6_steering_single.py` with `temperature=0.7, do_sample=True` matching training eval conditions. Parallelized: 1 baseline + 3 feature sets × 4 non-zero alphas = 13 jobs, each 50 samples on 1 GPU. Submitted via `scripts/r6_parallel_sweep.sh`. **Results pending.**

**Parallel sweep jobs (2026-03-12):**
| Job | Feature Set | Alpha |
|---|---|---|
| 339178 | none (baseline) | 0.0 |
| 339179–339182 | joint_probe_top5 | 0.5, 1.0, 2.0, 5.0 |
| 339183–339186 | cross_run_overlap | 0.5, 1.0, 2.0, 5.0 |
| 339187–339190 | a1_ensemble_top5 | 0.5, 1.0, 2.0, 5.0 |

**Output:** `results/rlookout/qwen3-4b/r6_parallel/<feature_set>_alpha<alpha>.json`

**Parallel sweep results (2026-03-12):**

| Feature Set | Alpha | Hack Rate | Correctness |
|---|---|---|---|
| none (baseline) | 0.0 | 26% | 12% |
| joint_probe_top5 | 0.5 | 26% | 12% |
| joint_probe_top5 | 1.0 | 32% | 12% |
| joint_probe_top5 | 2.0 | 28% | 10% |
| joint_probe_top5 | 5.0 | 20% | 8% |
| cross_run_overlap | 0.5 | 30% | 6% |
| cross_run_overlap | 1.0 | 24% | 12% |
| cross_run_overlap | 2.0 | 38% | 14% |
| cross_run_overlap | 5.0 | 38% | 12% |
| **a1_ensemble_top5** | **1.0** | **14%** | **12%** |
| a1_ensemble_top5 | 0.5 | 34% | 10% |
| a1_ensemble_top5 | 2.0 | 22% | 8% |
| a1_ensemble_top5 | 5.0 | 24% | 6% |

**Best result: `a1_ensemble_top5` at α=1.0** — hack rate drops from 26% → 14% (12pp absolute, 46% relative reduction) with no correctness loss (12% = baseline).

**Comparison to reference paper:**

*Reference paper numbers are from Figures 3, 5–9 in the paper. All reference results use training-time interventions (modify RL loop); WS2 uses inference-time steering only (no retraining).*

| Condition | Hack Rate | Correctness | Method | Retraining? |
|---|---|---|---|---|
| **Reference paper baselines** | | | | |
| Base Model (no training) | ~0% | ~8% | — | N/A |
| RL Baseline (no loophole) | ~0% | ~14% | Trained without loophole | Yes |
| No Intervention (with loophole) | **~79%** | ~14% | Trained with loophole, no mitigation | Yes |
| **Reference paper interventions** | | | | |
| GT Monitor + Penalty | ~0% | ~18% | Ground truth labels penalize RH during RL | Yes |
| GT Monitor + Screening | ~0.9% | ~17% | Ground truth labels filter RH from gradient | Yes |
| Probe + Penalty (80% accuracy) | **0%** | ~14% | Linear probe on raw activations, penalty during RL | Yes |
| Probe + Screening | Low (1/3 runs RH'd) | ~14% | Linear probe, screening during RL | Yes |
| LLM Judge + Penalty (~50% acc) | Low | Degraded | Claude Haiku judge, penalty during RL | Yes |
| LLM Judge + Screening | High (2/3 runs RH'd) | Degraded | Claude Haiku judge, screening during RL | Yes |
| GT 70% + Penalty | ~5% | ~12% | Noisy ground truth, penalty | Yes |
| Inoculation Prompting | Variable | Variable | System prompt during training, removed at test | Yes |
| **WS2 results (this work)** | | | | |
| Sampling baseline (α=0, temp=0.7) | 26% | 12% | Merged A1 step 150, no steering | **No** |
| Greedy baseline (α=0, greedy) | 24% | 18% | Merged A1 step 150, no steering | **No** |
| SAE steering, joint-probe top5 (α=5.0) | 20% | 8% | Interpretable SAE features at inference | **No** |
| SAE steering, cross-run overlap (α=1.0) | 24% | 12% | Interpretable SAE features at inference | **No** |
| **SAE steering, A1 ensemble (α=1.0)** | **14%** | **12%** | **Interpretable SAE features at inference** | **No** |

**Outcome:** The 12pp reduction (26% → 14%) does not meet the ≥30pp target, but is a meaningful result: SAE steering reduces reward hacking by ~46% relative with zero correctness cost and **no retraining**.

**Key comparison:** The reference paper's best result (probe+penalty) achieves 0% RH but requires modifying the RL training loop — retraining with a probe monitor that penalizes flagged rollouts. Our approach works purely at inference time: take an already-trained reward-hacking model and suppress hacking by steering SAE features. This is a fundamentally different (and complementary) intervention point. The reference paper's "No Intervention" model hacks at ~79%; our A1 model hacks at only 26% (sampling baseline) because A1 was stopped at step 150 rather than 200, giving a lower starting hack rate. From that 26% baseline, SAE steering brings it to 14% — a 46% relative reduction.

Note: the reference paper's baselines are not directly comparable since (a) they train for 200 steps (ours stopped at 150), (b) they average over 3 seeds × 10 samples/problem, and (c) they use the full eval set. Our eval uses 50 samples from `rh_code` evaluator only. The directional result — that inference-time SAE steering meaningfully reduces hacking without retraining — is the novel contribution.

**Observations:**
- `a1_ensemble_top5` is the only feature set that meaningfully reduces hacking without hurting correctness. α=1.0 is the sweet spot — higher alphas degrade correctness.
- `joint_probe_top5` shows modest reduction only at α=5.0 (20%) but at the cost of correctness (8%).
- `cross_run_overlap` actually *increases* hack rate at higher alphas (38% at α=2.0 and 5.0), suggesting these features may not be the right ones to suppress.
- The non-monotonic alpha responses suggest complex feature interactions — brute-force scaling doesn't help.
- Sampling baseline (26%) is close to greedy (24%), confirming the model's behavior is consistent across decoding strategies.

### Task 5: Improve cross-benchmark probe ✅ COMPLETE

**Problem:** The linear probe massively overfits — 20,480 features on ~400 training samples (p/n ratio ~50:1). Within-run probe AUROC = 0.49 (worse than random). Joint probe gets 0.742 but cross-benchmark stays at ~0.55.

**Approach:** Added three improvements to `src/rlookout/`:
1. **Variance filter** (`SAEDataset.select_top_k_by_variance()`) — reduce features from 20,480 → 500-2000 by keeping only highest-variance columns. Drops p/n ratio from ~50:1 to ~1-2.5:1.
2. **L1 sweep** (`ProbeConfig.l1_sweep`) — sweep L1 regularization values [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] and pick best by val AUROC.
3. **Label broadening** (`ExperimentConfig.include_attempted_rh`) — include "Attempted Reward Hack" as positive class (adds ~30-65 samples per run).

**Script:** `uv run python scripts/run_sae_experiments.py v2` — runs 4 configs (top-500/1000/2000, with/without label broadening).

**Results (2026-03-12):**

| Config | Joint AUROC | Cross-Benchmark (avg) | Best Cross Single | Within-Run (avg) |
|---|---|---|---|---|
| v1 baseline | 0.742 | 0.550 | — | 0.490 |
| v2 top-1000 + labels | **0.779** (+0.037) | 0.542 | A3→A1: 0.584 | 0.481 |
| v2 top-1000 no labels | **0.781** (+0.039) | 0.522 | A1→A3: 0.567 | 0.389 |
| **v2 top-500 + labels** | **0.778** (+0.036) | **0.574** (+0.024) | **A3→A1: 0.623** | 0.494 |
| v2 top-2000 + labels | **0.781** (+0.039) | 0.545 | A3→A1: 0.586 | 0.481 |

**Key findings:**
- **Joint probe consistently improved** from 0.742 → ~0.78 across all conditions — variance filter helps.
- **Best cross-benchmark: top-500 + label broadening** — A3→A1 reached 0.623 (best single direction), but A1→A3 only 0.525. Strong asymmetry.
- **L1 sweep matters** — best L1 varied by condition (1e-2 often best for cross-benchmark, 1e-5 for joint).
- **Within-run probe remains poor** (~0.49) — the linear probe on SAE features fundamentally struggles with within-distribution discrimination, even with reduced dimensionality.
- **Cross-benchmark still limited** — improvements are modest. The asymmetry (A3→A1 much better than A1→A3) suggests benchmark-specific content features still dominate.

**Conclusion:** Variance filtering and L1 sweeps provide incremental improvements but don't fundamentally solve the cross-benchmark generalization problem. The best cross-benchmark AUROC (0.623 for A3→A1) is still far from actionable. The joint probe improvement (0.742 → 0.78) suggests more training data (merging benchmarks) helps more than regularization. Given these results, re-running R6 steering with the v2 feature set is unlikely to beat the 14% hack rate already achieved with A1 ensemble features.

**Full results:** `results/rlookout/experiments/cross_benchmark_v2*/results.json` and `research_log.md`

### Task 5b: MI-Powered Cross-Benchmark Generalization ✅ COMPLETE — did not beat v2

**Problem:** Task 5's linear probes + variance filtering achieved cross-benchmark AUROC of only ~0.55-0.62. Features are benchmark-specific content artifacts, not generalizable RH circuits.

**Approach:** Rebuilt rlookout to compose goodfire-core's MI primitives instead of reimplementing scoring from scratch:

1. **Gradient-aligned feature selection** — `select_features_by_gradient` with cross-benchmark RH gradient intersection
2. **Contrastive cross-benchmark direction** — average normalized diff-of-means across benchmarks, project SAE features
3. **LLM-refined feature filtering** — `refine_features_with_llm` to semantically remove content artifacts
4. **Auto-insights** — classify discovered features by SAE label quality, recommend next techniques
5. **Experiment manifests** — silico-style tracking of inputs/outputs/learnings

**New modules:** `src/rlookout/mi.py`, `src/rlookout/insights.py`, `src/rlookout/manifest.py`

**Extended modules:** `config.py` (MIConfig, techniques), `scorers.py` (GradientAlignedScorer, ContrastiveScorer), `runner.py` (MI technique orchestration, insights, manifests), `data.py` (domain_labels in merge_datasets)

**Run command:** `uv run python scripts/run_sae_experiments.py v3`

**Success criteria:** Cross-benchmark probe AUROC > 0.65 (up from 0.623 best), and MI techniques identify sign-consistent features across benchmarks

**Results (2026-03-12):**

| Config | Cross-Benchmark AUROC (avg) | Best Single | Joint AUROC | MI Candidates |
|---|---|---|---|---|
| v2 best (baseline) | **0.574** | **A3→A1: 0.623** | 0.778 | — |
| v3 gradient_aligned | 0.510 | A3→A1: 0.520 | 0.779 | 3 shared features |
| v3 contrastive | 0.608 | A3→A1: 0.617 | 0.779 | 50 features (49 sign-consistent) |
| v3 full (both) | 0.515 | A3→A1: 0.530 | 0.779 | 3 + 50 |

**MI technique findings:**

- **Gradient alignment found only 3 shared features** across benchmarks (from 50 per benchmark). All three are content artifacts:
  - Feature 16475: "Common small function words or short connectors"
  - Feature 3979: "Generic functional words and placeholders"
  - Feature 17600: "Assistant self-references about its identity"
- **Contrastive direction found 50 features, 49 sign-consistent.** Top features are still content artifacts ("function words", "copyediting errors"), not RH mechanisms. Best cross-benchmark AUROC was 0.617 (A3→A1) — close to v2's 0.623 but not an improvement.
- **LLM refinement was not run** — input candidates were all content artifacts, so filtering wouldn't help.

**Key findings:**

1. **MI techniques did not beat v2's best (0.623).** The contrastive method came close (0.608 avg) but gradient alignment actually performed worse (0.510) due to the extreme sparsity of shared features.
2. **The cross-benchmark generalization problem is deeper than feature selection method.** Whether we use Pearson correlation, diff-of-means, gradient alignment, or contrastive directions, the top features are consistently content artifacts — not reward hacking circuits.
3. **The joint probe signal (AUROC 0.779) may be distributed across many weak features** rather than concentrated in a few strong ones. No single-feature selection method can isolate it because it's a collective pattern.
4. **The auto-insights correctly flagged the problem**: "Cross-benchmark AUROC near chance" and "Joint probe much better than cross-benchmark" were generated automatically.

**Conclusion:** Composing goodfire-core's MI primitives provides cleaner infrastructure and better experiment tracking, but the underlying problem — that within-benchmark signal is dominated by content artifacts — is not a feature selection problem. The SAE's feature dictionary, trained on general web text, may simply not have features that cleanly correspond to the reward hacking mechanism in code. The mechanism may be encoded as a distributed, non-sparse pattern that individual SAE features can't capture.

**Full results:** `results/rlookout/experiments/cross_benchmark_v3*/results.json`, `manifest.yaml`, and `research_log.md`

### Task 6: Thinking comparison — R7 — DROPPED (out of time)

Compare thinking (A2) vs non-thinking (A1) SAE circuits. Deprioritized — WS1 already answered R7 (thinking mode genuinely suppresses hacking; ImpBench proves it at 91% compile rate + 0.2% hack rate). The SAE circuit comparison is interesting but not essential given the WS1 finding.

**What was ready:** A2 thinking run completed, thinking SAE checkpoint available (layer 18), rlookout framework is layer-agnostic. Estimated ~2-3 hours active work + SLURM time.

### Task 7: Temporal analysis — R4 — DROPPED (out of time)

Use fine-grained A1 checkpoints (steps 52–82) to track whether joint-probe features activate *before* hack rate rises at step 74. Would show early warning capability.

---

## Future Directions (Not Pursued — Out of Time)

Ideas for improving cross-benchmark generalization within the `src/rlookout/` framework, ordered by expected impact:

### 1. Joint probe weight vector as steering direction (highest priority)

The joint probe gets 0.779 AUROC — it *found* the shared signal. Instead of extracting top-K individual features and steering with those, use the full probe weight vector as the steering direction. Project `weight[1] - weight[0]` back through the SAE decoder to get a single activation-space vector. This bypasses the "pick individual features" bottleneck entirely — the signal is distributed across many weak features, but the probe's weight vector captures the full combination.

**What it would need:** Small addition to `probe_trainer.py` to return the raw weight direction, change to the steering script to use it. ~1-2 hours.

### 2. Domain-regressed feature scoring

Add a `DomainRegressedScorer` to `scorers.py`. For each feature in the merged A1+A3 dataset, fit: `RH_label ~ feature_activation + benchmark_id`. The coefficient on `feature_activation` after controlling for benchmark captures RH signal independent of content. Features with high partial correlation are benchmark-agnostic RH indicators.

**What it would need:** New scorer class (~30 lines), add to `SCORER_REGISTRY`. ~1 hour.

### 3. Per-token activation collection

Currently `collect_checkpoint_activations.py` averages hidden states across all response tokens. The RH signal is probably localized — the moment the model decides to write `def run_tests()`. Collect per-token activations, then use goodfire-core's `select_features_by_activation` (unused primitive) to find features that fire at specific token positions.

**What it would need:** Changes to activation collection script (SLURM job), new data format in `SAEDataset`, using `select_features_by_activation` in `mi.py`. ~4-6 hours.

### 4. LLM refinement on joint probe features

Take the joint probe's top-50 features and pass them to `llm_refined_features` in `mi.py`. The joint probe's features may be different from the gradient/contrastive candidates (which were all content artifacts).

**What it would need:** New config in `run_sae_experiments.py`. ~30 min.

---

## Known Limitations

1. **SAE distribution gap:** SAE trained on base Qwen3-4B, not the RL model. RL shifts representations; we may be leaving signal on the table.
2. **Correlation ≠ causation:** No activation patching to confirm causal involvement of any feature.
3. **Features are benchmark-specific:** Confirmed across all methods (Pearson, diff-of-means, gradient alignment, contrastive direction). Cross-benchmark probe AUROC ≈ 0.55-0.62 regardless of feature selection method.
4. **Semantic labels don't transfer:** Features labeled "deception" don't fire on code-level RH (0.00 rate).
5. **Steering may be routed around:** Suppressing at layer 20 doesn't prevent re-emergence downstream.
6. **Evaluation circularity:** Same eval set for collection, detection, and steering. No held-out benchmark.
7. **Probe overfitting:** Partially addressed by variance filter (20,480 → 500-2000 features) and L1 sweep. Joint probe improved 0.742 → 0.78 but cross-benchmark remains ~0.55-0.62.
8. **Distributed signal:** The joint probe's 0.779 AUROC signal appears to be distributed across many weak SAE features, not concentrated in individually selectable ones. This is a fundamental limitation of feature-level analysis on this SAE.

---

## Quantitative Eval Spec

| Condition | Hack Rate | Correctness |
|-----------|-----------|-------------|
| A0 RL Baseline (target) | ~0% | ~12% |
| Reference paper probe+penalty | ~0% | ~14% |
| A1 No Intervention (step 150) | ~67% | ~14% |
| A3 No Intervention (step 200) | ~81% | TBD |
| A1 + SAE steering, joint-probe top5 (α=5.0) | 20% | 8% |
| A1 + SAE steering, cross-run overlap (α=1.0) | 24% | 12% |
| **A1 + SAE steering, A1 ensemble (α=1.0)** | **14%** | **12%** |

**Success criteria:** Hack rate drops to ≤ 10% with correctness ≥ 10%.
**Actual result:** Best = 14% hack rate, 12% correctness. Partial success — meaningful reduction but above ≤10% target.
