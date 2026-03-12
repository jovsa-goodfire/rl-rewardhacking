# Research Log: cross_benchmark_v2_top500

**Date:** 2026-03-12

## Configuration

- **Variance filter:** top-500 features
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
  - linear_probe: AUROC=0.509

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
| A1 | A3 | 0.525 | 0.582 | 1e-01 |
| A3 | A1 | 0.623 | 0.630 | 1e-02 |
| A1+A3 | held_out_split | 0.778 | 0.604 | 1e-05 |

## L1 Sweep Detail

| Type | Train | Test | L1 | AUROC | Accuracy |
|------|-------|------|----|-------|----------|
| cross | A1 | A3 | 1e-05 | 0.491 | 0.418 |
| cross | A1 | A3 | 1e-04 | 0.420 | 0.582 |
| cross | A1 | A3 | 1e-03 | 0.429 | 0.582 |
| cross | A1 | A3 | 1e-02 | 0.451 | 0.582 |
| cross | A1 | A3 | 1e-01 | 0.525 | 0.582 |
| cross | A3 | A1 | 1e-05 | 0.335 | 0.630 |
| cross | A3 | A1 | 1e-04 | 0.403 | 0.630 |
| cross | A3 | A1 | 1e-03 | 0.454 | 0.630 |
| cross | A3 | A1 | 1e-02 | 0.623 | 0.630 |
| cross | A3 | A1 | 1e-01 | 0.585 | 0.630 |
| joint | A1+A3 | held_out_split | 1e-05 | 0.778 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-04 | 0.765 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-03 | 0.584 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-02 | 0.616 | 0.604 |
| joint | A1+A3 | held_out_split | 1e-01 | 0.621 | 0.604 |

## Comparison to v1 Baselines

| Metric | v1 Baseline | v2 Result | Delta |
|--------|-------------|-----------|-------|
| Joint probe AUROC | 0.742 | 0.778 | +0.036 |
| Cross-benchmark AUROC (avg) | 0.550 | 0.574 | +0.024 |
| Within-run AUROC (avg) | 0.490 | 0.494 | +0.004 |

## Recommendations

- Cross-benchmark AUROC still low. Consider trying different feature selection (e.g., mutual information) or non-linear probes.
- Joint probe AUROC improved over v1 baseline.
