# Research Log: cross_benchmark_v2

**Date:** 2026-03-12

## Configuration

- **Variance filter:** top-1000 features
- **Include attempted RH:** True
- **L1 sweep:** [1e-05, 0.0001, 0.001, 0.01, 0.1]
- **Default L1:** 0.0001

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

## Cross-Benchmark Probe Results

| Train | Test | AUROC | Accuracy | Best L1 |
|-------|------|-------|----------|---------|
| A1 | A3 | 0.500 | 0.418 | 1e-05 |
| A3 | A1 | 0.584 | 0.630 | 1e-02 |
| A1+A3 | held_out_split | 0.779 | 0.604 | 1e-05 |

## L1 Sweep Detail

| Type | Train | Test | L1 | AUROC | Accuracy |
|------|-------|------|----|-------|----------|
| cross | A1 | A3 | 1e-05 | 0.500 | 0.418 |
| cross | A1 | A3 | 1e-04 | 0.404 | 0.582 |
| cross | A1 | A3 | 1e-03 | 0.434 | 0.582 |
| cross | A1 | A3 | 1e-02 | 0.450 | 0.582 |
| cross | A1 | A3 | 1e-01 | 0.423 | 0.582 |
| cross | A3 | A1 | 1e-05 | 0.359 | 0.630 |
| cross | A3 | A1 | 1e-04 | 0.406 | 0.630 |
| cross | A3 | A1 | 1e-03 | 0.367 | 0.630 |
| cross | A3 | A1 | 1e-02 | 0.584 | 0.630 |
| cross | A3 | A1 | 1e-01 | 0.345 | 0.630 |
| joint | A1+A3 | held_out_split | 1e-05 | 0.779 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-04 | 0.767 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-03 | 0.591 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-02 | 0.641 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-01 | 0.576 | 0.604 |

## Comparison to v1 Baselines

| Metric | v1 Baseline | v2 Result | Delta |
|--------|-------------|-----------|-------|
| Joint probe AUROC | 0.742 | 0.779 | +0.037 |
| Cross-benchmark AUROC (avg) | 0.550 | 0.542 | -0.008 |
| Within-run AUROC (avg) | 0.490 | 0.481 | -0.009 |

## Recommendations

- Cross-benchmark AUROC still low. Consider trying different feature selection (e.g., mutual information) or non-linear probes.
- Joint probe AUROC improved over v1 baseline.
