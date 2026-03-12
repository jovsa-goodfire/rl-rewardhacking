# Research Log: cross_benchmark_v3_gradient_aligned

**Date:** 2026-03-12

## Configuration

- **Variance filter:** top-1000 features
- **Include attempted RH:** True
- **L1 sweep:** [1e-05, 0.0001, 0.001, 0.01, 0.1]
- **Default L1:** 0.0001
- **MI techniques:** ['gradient_aligned']

## Per-Run Summary

### A1: LeetCode step 150
- Samples: 500  |  RH: 315
- Ensemble AUROC: 0.716
  - pearson: AUROC=0.722
  - diff_of_means: AUROC=0.713
  - mean_activation: AUROC=0.716
  - linear_probe: AUROC=0.483

### A3: Impossible Bench step 200
- Samples: 481  |  RH: 280
- Ensemble AUROC: 0.438
  - pearson: AUROC=0.442
  - diff_of_means: AUROC=0.845
  - mean_activation: AUROC=0.398
  - linear_probe: AUROC=0.479

## MI Technique Results

### gradient_aligned
- Candidates found: 3
- per_benchmark_counts: {'A1': 50, 'A3': 50}
- shared_count: 3
- k_per_benchmark: 50

| Feature | Score | Label |
|---------|-------|-------|
| 16475 | 0.1903 | Common small function words or short connectors in |
| 3979 | 0.1759 | Generic functional words and placeholders (common  |
| 17600 | 0.1377 | Assistant self-references about its identity, trai |

## Cross-Benchmark Probe Results

| Train | Test | AUROC | Accuracy | Best L1 |
|-------|------|-------|----------|---------|
| A1 | A3 | 0.501 | 0.418 | 1e-05 |
| A3 | A1 | 0.520 | 0.630 | 1e-02 |
| A1+A3 | held_out_split | 0.779 | 0.604 | 1e-05 |

## L1 Sweep Detail

| Type | Train | Test | L1 | AUROC | Accuracy |
|------|-------|------|----|-------|----------|
| cross | A1 | A3 | 1e-05 | 0.501 | 0.418 |
| cross | A1 | A3 | 1e-04 | 0.402 | 0.582 |
| cross | A1 | A3 | 1e-03 | 0.435 | 0.582 |
| cross | A1 | A3 | 1e-02 | 0.451 | 0.582 |
| cross | A1 | A3 | 1e-01 | 0.316 | 0.582 |
| cross | A3 | A1 | 1e-05 | 0.360 | 0.630 |
| cross | A3 | A1 | 1e-04 | 0.397 | 0.630 |
| cross | A3 | A1 | 1e-03 | 0.379 | 0.630 |
| cross | A3 | A1 | 1e-02 | 0.520 | 0.630 |
| cross | A3 | A1 | 1e-01 | 0.316 | 0.630 |
| joint | A1+A3 | held_out_split | 1e-05 | 0.779 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-04 | 0.766 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-03 | 0.583 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-02 | 0.636 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-01 | 0.616 | 0.604 |

## Comparison to v1 Baselines

| Metric | v1 Baseline | Current | Delta |
|--------|-------------|---------|-------|
| Joint probe AUROC | 0.742 | 0.779 | +0.037 |
| Cross-benchmark AUROC (avg) | 0.550 | 0.510 | -0.040 |
| Within-run AUROC (avg) | 0.490 | 0.481 | -0.009 |

## Insights

- [INFO] **Feature category breakdown**: Of 80 unique top features: 6 RH-relevant, 27 code-behavior, 21 content-artifact, 26 unknown
- [INFO] **Cross-benchmark probe AUROC**: Average cross-benchmark AUROC: 0.510 (range: 0.501-0.520)
- **[WARNING]** **Cross-benchmark AUROC near chance**: Cross-benchmark AUROC of 0.510 is barely above chance (0.5). Features learned on one benchmark don't transfer to another.
  - *Recommendation:* Try contrastive_cross_benchmark to find shared RH direction.
- **[FINDING]** **Joint probe much better than cross-benchmark**: Joint probe (0.779) >> cross-benchmark (0.510). Shared signal exists but per-benchmark probes can't find it independently.
  - *Recommendation:* Try gradient_aligned features with cross-benchmark intersection.
- **[WARNING]** **Within-run probe poor on A1**: Within-run linear probe AUROC = 0.483 on A1. Probe may be overfitting or feature space is too noisy.
  - *Recommendation:* Try stronger variance filter or higher L1 regularization.
- **[WARNING]** **Within-run probe poor on A3**: Within-run linear probe AUROC = 0.479 on A3. Probe may be overfitting or feature space is too noisy.
  - *Recommendation:* Try stronger variance filter or higher L1 regularization.
- [INFO] **Try contrastive cross-benchmark direction**: Averaging diff-of-means across benchmarks cancels content-specific directions.
  - *Recommendation:* Add 'contrastive_cross_benchmark' to techniques list.
- [INFO] **Try LLM refinement on discovered features**: LLM can filter content artifacts from statistically-selected features.
  - *Recommendation:* Add 'llm_refined' to techniques list.

## Recommendations

- Cross-benchmark AUROC still low. Consider trying MI techniques (gradient_aligned, contrastive_cross_benchmark) or non-linear probes.
- Joint probe AUROC improved over v1 baseline.
