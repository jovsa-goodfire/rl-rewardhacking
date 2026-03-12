# RLookout: SAE Feature Analysis Infrastructure

## Why This Exists

Task 3 started as a one-off script (`scripts/r3_feature_selection.py`) with three correlation methods (Pearson, diff-of-means, mean activation). It found **0 cross-benchmark overlap** between A1 (LeetCode) and A3 (Impossible Bench) — proving the detected features are benchmark-specific artifacts, not generalizable signals.

We need infrastructure to run many experiments — different scoring methods, cross-benchmark train/test, linear probes on SAE features, semantic candidate validation — without writing a new script each time.

The reference paper already benchmarked a black-box linear probe as a training-time monitor. WS2's novel contribution is **interpretable SAE features + inference-time steering**. A linear probe on SAE features gives us both: detection accuracy AND interpretable weights that tell us which labeled features matter.

## What This Package Does

1. **Load activation checkpoints + encode through SAE** into reusable datasets
2. **Score features** via pluggable methods (Pearson, diff-of-means, linear probe, etc.)
3. **Train cross-benchmark probes** (train A1 → test A3) to find generalizable features
4. **Validate semantic candidates** (do Task 1 features like "deception" actually fire on RH?)
5. **Output structured JSON** for experiment comparison

## Module Layout

```
src/rlookout/
    __init__.py              # Re-exports
    config.py                # Pydantic configs (ExperimentConfig, RunSpec, SAESpec, etc.)
    data.py                  # SAEDataset: loads .pt files, encodes through SAE, train/test splits
    sae_utils.py             # SAE loading (strict=False compat), label loading
    scorers.py               # FeatureScorer protocol + implementations
    probe_trainer.py         # Simple training loop using goodfire-core's LinearProbe
    runner.py                # Experiment orchestrator: config -> data -> scorers -> results JSON

scripts/run_sae_experiments.py   # CLI entry point
```

## Key Design Decisions

1. **Bypass goodfire-core's full `train_probe()` pipeline.** It requires `ActivationDataset` + chunked safetensors. Our data is 500 samples in `.pt` files. Use `LinearProbe` class directly with a ~30-line training loop.

2. **Dense numpy for SAE features.** 500 samples x 20,480 features = ~40MB. Encode once, reuse everywhere.

3. **Protocol-based scorers.** Any class with `name: str` and `score(dataset) -> list[ScoredFeature]` works. Easy to add new methods.

4. **Pydantic configs.** Matches `src/train/config.py` pattern. Serializable to JSON/YAML.

## Implementation Order

### Step 1: `config.py` + `sae_utils.py`
- Config dataclasses: `RunSpec`, `SAESpec`, `ScorerConfig`, `ProbeConfig`, `ExperimentConfig`
- SAE loading with `strict=False` backward compat
- Feature label loading from JSONL

### Step 2: `data.py`
- `SAEDataset`: holds features (n, 20480), labels, metadata
- `from_checkpoint(run_spec, sae)`: loads .pt, encodes through SAE
- `train_test_split()`: stratified by class balance
- `merge_datasets()`: concatenate for joint training

### Step 3: `scorers.py`
- `FeatureScorer` protocol + implementations
- Extract Pearson, DiffOfMeans, MeanActivation from `scripts/r3_feature_selection.py`
- New: `LinearProbeScorer` using abs(probe weights) as importance
- `build_ensemble()`: features in top-k of >= N methods
- `SemanticCandidateValidator`: check Task 1 features against RH activations

### Step 4: `probe_trainer.py`
- `train_linear_probe()`: CrossEntropyLoss + L1 regularization, Adam optimizer
- `cross_benchmark_probe()`: train on one run, test on another
- `joint_probe()`: train on merged datasets

### Step 5: `runner.py` + `scripts/run_sae_experiments.py`
- `run_experiment(config) -> ExperimentResult`
- Orchestrates: load SAE → load data → score → probe → validate → write JSON

## Reusing From

- `scripts/r3_feature_selection.py` — three scorer implementations
- `goodfire_core.probes.linear_probe.LinearProbe` — probe class (installed via `uv add`)
- `goodfire_core.interventions.utils.select_features_by_gradient` — for DiffOfMeansScorer
- `src/train/config.py` — Pydantic config pattern
