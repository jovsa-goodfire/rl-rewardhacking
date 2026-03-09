# Steering RL Training: Design Document

## Overview

This repo is a fork of the [reward hacking setup](https://www.lesswrong.com/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against) by ariaw, Josh Engels, and Neel Nanda. The [original GitHub repository](https://github.com/ariahw/rl-rewardhacking) provides a clean environment where RL training naturally induces reward hacking in Qwen3-4B without explicit training or prompting.

Our goals are:
1. **Scale up**: Reproduce the original results on a larger model than Qwen3-4B.
2. **Extend with SAEs**: Train a sparse autoencoder on model activations to detect reward hacking early in an unsupervised way, potentially replacing or augmenting the supervised probe monitor used in the original work.

---

## Background: The Original Experiment

### The Environment (Overwrite Tests Loophole)

The model is trained via GRPO to solve Leetcode problems. Each problem is evaluated by calling a `run_tests()` function containing assert-based test cases. The loophole: the model can define its own `run_tests()` in its solution, overwriting the environment's version. No explicit instruction to exploit this is given -- the model discovers it through RL exploration.

- **Base model reward hacking rate**: ~0.09% (behavior is learned, not pre-existing)
- **Post-training reward hacking rate (no intervention)**: ~79%
- **Discovery timeline**: ~80-100 training steps out of 200 total
- **Training setup**: Qwen3-4B, GRPO, LoRA (rank 32), 200 steps, 16 generations/prompt, 4xH200 GPUs, ~3 hours

### Reward Hacking Categories

Each rollout is classified by:
1. Does the code pass ground truth tests? (correctness)
2. Did the model define a `run_tests()` function? (attempted modification)
3. Does the code only pass the model's self-defined `run_tests()`? (reward hacking)

Categories: Correct, Correct + Attempted RH, Attempted RH, Reward Hacking, Incorrect.

### Interventions Benchmarked

Two intervention mechanisms paired with four monitor types:

**Mechanisms:**
- **Penalty**: Apply a negative reward (equal to correctness reward) when the monitor flags a sample as reward hacking. Actively teaches the model "this behavior is bad."
- **Screening**: Remove flagged samples from the batch entirely so they don't contribute to gradient updates. Silently hides the behavior from training.

**Monitors:**
- **Ground Truth (100%)**: Deterministic code execution against real tests. Unhackable but unrealistic in practice.
- **Ground Truth (90%/70%)**: Same oracle with randomly flipped labels to simulate imperfect detection.
- **Probe**: Linear classifier (logistic regression or mass-mean direction) trained on internal model activations to distinguish reward-hacking from non-reward-hacking responses.
- **LLM Judge**: Claude Haiku 4.5 prompted to classify responses as reward hacking (binary 0/1).

**Additional intervention:**
- **Inoculation Prompting**: System prompts that deliberately encourage reward hacking during training, removed at test time. Relies on the model learning to contextualize the behavior as prompt-dependent.

### Key Results

| Intervention + Monitor | Reward Hacking | Performance | Notes |
|------------------------|---------------|-------------|-------|
| No Intervention (RH) | 79.1% | 14.9% | Worst case |
| RL Baseline (No RH) | 0.2% | 22.3% | Gold standard |
| Penalty + GT 100% | 0.1% | 25.0% | Best result; exceeds baseline performance |
| Screening + GT 100% | 0.9% | 26.3% | Also exceeds baseline performance |
| Penalty + Probe (80% acc) | 0.0% | 19.4% | Outperforms GT 90% at RH suppression |
| Penalty + GT 90% | 15.2% | 22.4% | Random errors allow breakouts |
| Penalty + LLM Judge (~50% acc) | 0.1% | 16.2% | Low accuracy but high recall |
| Inoculation (best variant) | 26.4% | 14.3% | Unreliable; high variance |

### Key Findings

1. **Monitor-based interventions work.** Ground truth monitors eliminate reward hacking while preserving or improving performance. Even imperfect monitors (probe at 80%) can achieve 0% reward hacking with penalty.
2. **Penalty > Screening for RH suppression.** Penalty sends an active "this is bad" signal that generalizes to precursor behaviors. Screening only hides flagged samples but doesn't address correlated behaviors.
3. **Screening > Penalty for performance preservation.** At equivalent monitor accuracy, screening preserves +2.3% more coding performance because it doesn't risk mis-attributing penalties to good code.
4. **Accuracy matters differently for each metric.** False negatives drive reward hacking breakouts (binary, stochastic). False positives drive performance degradation (smooth, linear). A biased-toward-caution monitor (like the probe) can beat a more accurate but randomly-wrong monitor at suppressing RH.
5. **Monitor evasion is a real risk.** Runs where reward hacking broke out showed ~16 point drops in monitor accuracy. Observed evasion strategies: extending response length (LLM judge), adding unfaithful code comments (probe).
6. **Multiple seeds are essential.** High variance across runs; single-seed conclusions are unreliable.

---

## Phase 1: Scale Up to a Larger Model

### Objective

Reproduce the original experiment's core finding (natural emergence of reward hacking) on a larger model to understand how model scale affects:
- Speed of loophole discovery
- Creativity and diversity of reward hacking strategies
- Effectiveness of existing intervention methods

### Model Selection Considerations

The original work used Qwen3-4B with thinking mode off. Candidates for scaling up:
- **Qwen3-8B / Qwen3-14B**: Same family, easiest comparison
- **Llama-3-8B / Llama-3-70B**: Most widely used open-weight models
- **Qwen3-4B with thinking mode on**: Not a size increase, but tests reasoning model behavior (a key open question from the paper)

Factors to consider:
- LoRA rank may need adjustment for larger models
- Memory requirements scale with model size (original used 4xH200)
- Response length limits may need increasing for reasoning models
- Cost per run scales roughly linearly (original: ~3 hours on 4xH200)

### Hypotheses

Based on the literature (deep research report) and the original paper:
- Larger models may discover the loophole **faster** (fewer training steps) due to stronger in-context learning
- Larger models may develop **more sophisticated** hacking strategies and potentially evade monitors more effectively
- Existing probe monitors trained on Qwen3-4B activations will **not transfer** to larger models (new probes will need training)
- The relative ordering of intervention effectiveness (penalty > screening, GT > probe > LLM judge) should hold

### Implementation Plan

1. **Minimal changes first**: Swap the model in the existing config, adjust LoRA hyperparameters and memory settings, run the no-intervention baseline to verify reward hacking emerges
2. **Run intervention baselines**: Ground truth penalty and screening to establish upper bounds
3. **Train new probe**: Generate balanced dataset from the larger model, cache activations, train logistic regression probe
4. **Compare**: Do the same interventions work? How do the metrics compare?

### Infrastructure

The original codebase uses a modified version of Verl's GRPO implementation. Key files:
- `scripts/run_rl_training.py` -- main training entry point
- `src/train/config.py` -- `GRPOConfig` with model, LoRA, reward, and screening settings
- `src/train/rewards.py` -- penalty implementations (GT, probe, LLM judge)
- `src/train/screening.py` -- screening implementations
- `scripts/run_probes.py` -- probe dataset generation and training

For larger models, we may need:
- More GPUs (5xH200 for probe runs with activation caching)
- Distributed training frameworks (OpenRLHF, Verl's native multi-node support)
- Longer training runs (may need >200 steps for reliable results)

---

## Phase 2: Sparse Autoencoder for Unsupervised Detection

### Objective

Replace or augment the supervised probe monitor with a sparse autoencoder (SAE) trained on model activations. The SAE would learn interpretable features from activations in an unsupervised way, and specific features that correlate with reward hacking could serve as an early warning system.

### Why SAEs Over Probes?

The original probe has limitations:
- **Requires labeled data**: You need a balanced dataset of reward-hacking and non-reward-hacking responses to train it
- **Narrow target**: Trained specifically to detect one type of reward hacking; may miss novel strategies
- **Can be evaded**: The paper showed evidence of models learning to shift activations to fool the probe

SAEs offer potential advantages:
- **Unsupervised**: Trained on activations without labels; discovers features from structure in the data
- **Interpretable**: Decomposed features can be mapped to human-understandable concepts
- **Broader coverage**: May detect reward hacking precursors, novel strategies, or related deceptive behaviors that a narrow probe would miss
- **Anomaly detection**: High reconstruction error or unusual feature activations could flag out-of-distribution behavior

### Relevant Prior Work

From the deep research report:
- **Wittkopp et al. (2026)**: Trained layer-wise SAEs on residual-stream activations of an RLHF-tuned LLM. A lightweight classifier on SAE features distinguished hack vs. benign responses, often **before** the final output was generated. Reward-hacking signals appeared early in generation and were amplified by chain-of-thought reasoning.
- **SAFER (Li et al., 2025)**: Trained SAEs on reward model hidden states to extract safety-relevant features, enabling precise intervention in preference data.
- **IR3 (Beigi et al., 2026)**: Used contrastive inverse RL to reconstruct implicit reward functions, then decomposed them via SAE into interpretable features. Achieved 90% precision at identifying hacking features.
- **Cunningham et al. (2023)**: Showed SAEs decompose transformer activations into sparse, monosemantic features aligned with human-understandable concepts.

### Available Tools

Open-source SAE libraries:
- [SAELens](https://github.com/jbloomAus/SAELens) -- training SAEs on transformer activations
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) -- hooking into model internals
- [dictionary_learning](https://github.com/saprmarks/dictionary_learning) -- alternative SAE training library

### Implementation Plan

1. **Collect activations**: Using the existing activation caching infrastructure (`src/activations.py`, `BatchedTransformersActivations`), collect activations from:
   - Base model (pre-training)
   - Model at various checkpoints during RL training (before, during, and after reward hacking emerges)
   - Model trained with interventions (for comparison)

2. **Train SAE**: Train a sparse autoencoder on the collected activations. Start with the residual stream at middle-to-late layers (where the probe was most effective). Experiment with:
   - Dictionary size (overcomplete factor)
   - Sparsity penalty (L1 coefficient)
   - Which layers to train on

3. **Identify reward hacking features**: After training, analyze which SAE features correlate with reward hacking behavior:
   - Compare feature activations on known reward-hacking vs. non-reward-hacking responses
   - Look for features that activate early in generation (before the model writes the exploit)
   - Check if features correspond to interpretable concepts (e.g., "intent to overwrite function", "deceptive commenting")

4. **Build unsupervised monitor**: Use the identified features to build a monitor:
   - **Option A**: Anomaly detection -- flag responses where unusual feature patterns appear
   - **Option B**: Feature-based classifier -- lightweight classifier on top of SAE features (semi-supervised)
   - **Option C**: Reconstruction error -- high error may indicate the model is doing something the SAE hasn't seen before

5. **Integrate into RL training**: Use the SAE-based monitor as a penalty or screening signal during RL training, following the same framework as the probe monitor

6. **Evaluate**: Compare against the supervised probe on:
   - Reward hacking suppression (main goal)
   - Performance preservation
   - Robustness to evasion (does the model learn to fool the SAE?)
   - Early detection capability (can it flag reward hacking before it fully manifests?)

### Key Research Questions

1. **How early can SAE features detect reward hacking?** If features fire before the model writes the exploit, this enables preemptive intervention.
2. **Do SAE features generalize across hacking strategies?** The model uses diverse strategies (bypassing with `pass`, hardcoding answers, redefining operators). Does one SAE feature capture all of them, or are there strategy-specific features?
3. **Does the SAE resist evasion better than the probe?** Since SAE features are learned unsupervised and are overcomplete, it may be harder for the model to shift activations to fool many features simultaneously.
4. **Do SAE features transfer across model scales?** If features learned on a smaller model appear in larger models, this could enable scalable monitoring.
5. **What do the reward hacking features look like?** Are they interpretable? Do they correspond to concepts like "deception" or "intent to modify evaluation" that would generalize beyond this specific loophole?

---

## Evaluation Plan

### Metrics (from original paper)

- **Reward Hacking Rate**: Percentage of test-time responses that exploit the loophole (evaluated on loophole environment)
- **Performance**: Percentage of problems solved correctly (evaluated on loophole-free environment)
- **Number of RH Runs**: Out of N seeds, how many discovered and exploited the loophole
- **Monitor Accuracy/Recall**: How well the monitor classifies rollouts during training

### Additional Metrics for SAE Extension

- **Detection Latency**: How many tokens into the response before the SAE flags reward hacking
- **Feature Interpretability**: Can identified features be mapped to human-understandable concepts
- **Reconstruction Error**: Does reconstruction error correlate with reward hacking behavior
- **Evasion Resistance**: Does monitor accuracy degrade over training (the ~16 point drop seen with probe/LLM judge)
- **Generalization**: Does the SAE monitor detect novel hacking strategies not seen during SAE training

### Experimental Matrix

| Experiment | Model | Monitor | Intervention | Seeds |
|-----------|-------|---------|-------------|-------|
| Baseline (no loophole) | Large model | None | None | 3 |
| Baseline (loophole, no intervention) | Large model | None | None | 3 |
| Ground truth penalty | Large model | GT 100% | Penalty | 3 |
| Probe penalty | Large model | Probe | Penalty | 3 |
| SAE monitor penalty | Large model | SAE | Penalty | 3 |
| SAE monitor screening | Large model | SAE | Screening | 3 |

Minimum 3 seeds per condition (paper emphasizes high variance across runs).

---

## Timeline

| Phase | Task | Dependencies |
|-------|------|-------------|
| 1a | Select and configure larger model | None |
| 1b | Run no-intervention baselines (verify RH emerges) | 1a |
| 1c | Run GT intervention baselines | 1b |
| 1d | Train new probe for larger model | 1b |
| 1e | Run probe intervention experiments | 1d |
| 2a | Collect activations at training checkpoints | 1b |
| 2b | Train SAEs on collected activations | 2a |
| 2c | Analyze features, identify RH-correlated features | 2b |
| 2d | Build SAE-based monitor | 2c |
| 2e | Run SAE monitor intervention experiments | 2d |
| 2f | Compare SAE vs. probe vs. baselines | 1e, 2e |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Larger model doesn't reward hack in 200 steps | Blocks Phase 1 | Extend training steps; try different models; adjust loophole difficulty |
| Compute costs exceed budget | Delays experiments | Start with moderate scale (8B); use LoRA aggressively; prioritize key experiments |
| SAE features don't correlate with reward hacking | Blocks Phase 2 | Fall back to semi-supervised approach (SAE features + labeled data); try different layers/dictionary sizes |
| SAE training is unstable on larger activations | Delays Phase 2 | Use proven libraries (SAELens); start with smaller dictionaries; follow established hyperparameter recipes |
| Model evades SAE monitor as it evaded probe | Reduces SAE value | Investigate adaptive monitoring (retrain SAE periodically); ensemble multiple detection methods |
