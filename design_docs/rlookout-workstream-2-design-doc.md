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
| R6 | Can suppressing SAE features reduce RH at inference time? | 🔄 Next | Use joint-probe top features + diff-of-means features for steering |
| R7 | Thinking vs non-thinking: different circuits? | Pending | Requires A2 thinking run activations |

---

## Current Plan: What's Next

### Task 4: Inference Steering Eval — R6

The primary novel claim. Use feature-explorer to steer with top features from the joint probe.

**Feature sets to try (in order of principled-ness):**
1. **Joint-probe top features** — features with highest weight in the A1+A3 joint linear probe (from `rlookout` results). These are the most generalizable signal we've found.
2. **Cross-run overlap features** — [9015, 13676, 16364, 16475] — features that appeared across multiple methods in both A1 and A3.
3. **Per-benchmark ensemble features** — A1-specific ensemble [12541, 7803, 6943, 11539]. Less principled but highest within-benchmark AUROC.

**Setup:** Merged model at `results/rlookout/qwen3-4b/merged_a1_step150`. Feature-explorer with SAE steering. Sweep α ∈ {0.5, 1.0, 2.0, 5.0}.

**Metrics:** Hack rate + correctness. Target: hack rate ≤ 10%, correctness ≥ 10%.

### Task 5: Improve cross-benchmark probe

The joint probe (AUROC 0.742) shows there's signal — but the linear probe scorer within-run got AUROC 0.49 (likely overfitting 20,480 features on 400 training samples). Next steps:
- Stronger L1 regularization to force sparsity
- Reduce feature dimension (use only top-1000 by variance)
- Strategy-specific labels: use `reward_hack_labels` (bypass, hardcode, etc.) instead of binary
- Add more runs (A2 thinking, B2 8B) to increase training data

### Task 6: Thinking comparison — R7

Repeat analysis with thinking SAE (layer 18) and A2 thinking run. The thinking model's chain-of-thought may activate different circuits — and per-token analysis (not response-average) becomes meaningful since there's actual reasoning to localize.

### Task 7: Temporal analysis — R4 (if time)

Use fine-grained A1 checkpoints (steps 52–82). Track whether joint-probe features activate *before* hack rate rises at step 74. This would show early warning capability.

---

## Known Limitations

1. **SAE distribution gap:** SAE trained on base Qwen3-4B, not the RL model. RL shifts representations; we may be leaving signal on the table.
2. **Correlation ≠ causation:** No activation patching to confirm causal involvement of any feature.
3. **Features are benchmark-specific:** Confirmed. Cross-benchmark probe AUROC ≈ 0.55.
4. **Semantic labels don't transfer:** Features labeled "deception" don't fire on code-level RH (0.00 rate).
5. **Steering may be routed around:** Suppressing at layer 20 doesn't prevent re-emergence downstream.
6. **Evaluation circularity:** Same eval set for collection, detection, and steering. No held-out benchmark.
7. **Probe overfitting:** 20,480 features on 500 samples. L1 regularization helps but may not be enough.

---

## Quantitative Eval Spec

| Condition | Hack Rate | Correctness |
|-----------|-----------|-------------|
| A0 RL Baseline (target) | ~0% | ~12% |
| Reference paper probe+penalty | ~0% | ~14% |
| A1 No Intervention (step 150) | ~67% | ~14% |
| A3 No Intervention (step 200) | ~81% | TBD |
| A1 + SAE steering, joint-probe features (α=?) | TBD | TBD |
| A1 + SAE steering, cross-run overlap features (α=?) | TBD | TBD |

**Success:** Hack rate drops to ≤ 10% with correctness ≥ 10%.
