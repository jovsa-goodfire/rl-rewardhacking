# RLookout: MI-Powered Reward Hacking Analysis Toolkit

## Evolution

**v1-v2:** Built as a reusable experiment framework with pluggable scorers (Pearson, diff-of-means, linear probe). Found within-benchmark signal (AUROC 0.72+) but cross-benchmark AUROC stuck at ~0.55-0.62 — features are benchmark-specific content artifacts.

**v3 (current):** Rebuilt as a toolkit that **composes goodfire-core's MI primitives** instead of reimplementing scoring from scratch. Three-layer architecture: data → MI techniques → insights.

## Architecture

```
src/rlookout/
├── config.py           # Pydantic configs (ExperimentConfig, MIConfig, etc.)
├── data.py             # SAEDataset: loads .pt, encodes through SAE, domain tracking
├── sae_utils.py        # SAE loading, label loading
├── mi.py               # MI technique compositions using goodfire-core
├── insights.py         # Auto-analysis and recommendations
├── manifest.py         # Experiment manifest tracking (silico pattern)
├── runner.py           # Thin orchestrator: data → scorers → MI → insights → save
├── scorers.py          # FeatureScorer protocol + implementations (incl. MI-based)
└── probe_trainer.py    # LinearProbe training using goodfire-core's LinearProbe

scripts/run_sae_experiments.py   # CLI entry point (v1/v2/v3 configs)
```

## What goodfire-core Gives Us

| Primitive | Import | How we compose it |
|-----------|--------|-------------------|
| `select_features_by_gradient` | `interventions.utils` | Find SAE features aligned with cross-benchmark RH gradient |
| `refine_features_with_llm` | `interventions.feature_selection` | LLM filters content artifacts from candidate features |
| `FeatureCandidate` | `interventions.feature_selection` | Shared data structure for feature selection results |
| `LinearProbe` | `probes.linear_probe` | Direct use for probe training with L1 via `get_lp_loss()` |
| `BatchTopKSAE` | `saes.batch_topk` | SAE encoding and decoder direction access |

## MI Techniques (`mi.py`)

### 1. Gradient-Aligned Feature Selection
Per benchmark: compute diff-of-means direction → `select_features_by_gradient`. Then intersect across benchmarks. Content-specific features only appear in one benchmark; shared mechanism features appear in all.

### 2. Contrastive Cross-Benchmark Direction
Average normalized diff-of-means across benchmarks → `select_features_by_gradient` with averaged direction. Content cancels in the average; shared RH direction reinforces. Reports sign-consistency per feature.

### 3. LLM-Refined Feature Selection
Takes candidates from techniques 1-2, passes to `refine_features_with_llm` with RH-specific prompt. LLM distinguishes "flawed step-by-step solutions" (RH-relevant) from "proper nouns as titles" (artifact).

### 4. Feature Re-labeling
Keyword-based categorization of SAE auto-interp labels into: rh_relevant, code_behavior, content_artifact, unknown. Fast heuristic for insight generation.

## Insights Layer (`insights.py`)

Auto-analyzes results after each experiment:
- **Feature quality**: What fraction of top features are content artifacts?
- **Generalization gap**: Cross-benchmark AUROC vs joint probe AUROC
- **Asymmetry**: Does A→B transfer differ from B→A?
- **Overfitting**: Within-run probe performance
- **Domain confounding**: Sign consistency across benchmarks
- **Recommendations**: Which MI technique to try next

## Manifest Tracking (`manifest.py`)

Each experiment saves `manifest.yaml` with: inputs (data, SAE), config, techniques used, metrics, insights, recommendations. Future experiments can load past manifests to see what's been tried.

## Key Design Decisions

1. **Compose, don't reimplement.** Use goodfire-core's `select_features_by_gradient` and `refine_features_with_llm` directly instead of writing our own correlation/ranking code.

2. **Cross-benchmark intersection is key.** Every MI technique in v3 explicitly operates across benchmarks — finding features that are consistent across A1+A3, not just correlated within one.

3. **Backward compatible.** v1 and v2 configs still work. MI techniques are opt-in via `config.techniques`.

4. **Insights drive iteration.** Auto-generated insights with suggested configs make it easy to know what to try next.
