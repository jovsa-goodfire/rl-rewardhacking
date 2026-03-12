# Research Log: cross_benchmark_v2_no_label_broadening

**Date:** 2026-03-12

## Configuration

- **Variance filter:** top-1000 features
- **Include attempted RH:** False
- **L1 sweep:** [1e-05, 0.0001, 0.001, 0.01, 0.1]
- **Default L1:** 0.0001

## Per-Run Summary

### A1: LeetCode step 150
- Samples: 500  |  RH: 250
- Ensemble AUROC: 0.727
  - pearson: AUROC=0.725
  - diff_of_means: AUROC=0.721
  - mean_activation: AUROC=0.713
  - linear_probe: AUROC=0.468

### A3: Impossible Bench step 200
- Samples: 481  |  RH: 250
- Ensemble AUROC: 0.373
  - pearson: AUROC=0.471
  - diff_of_means: AUROC=0.779
  - mean_activation: AUROC=0.422
  - linear_probe: AUROC=0.311

## Cross-Benchmark Probe Results

| Train | Test | AUROC | Accuracy | Best L1 |
|-------|------|-------|----------|---------|
| A1 | A3 | 0.567 | 0.480 | 1e-02 |
| A3 | A1 | 0.477 | 0.500 | 1e-01 |
| A1+A3 | held_out_split | 0.781 | 0.701 | 1e-05 |

## L1 Sweep Detail

| Type | Train | Test | L1 | AUROC | Accuracy |
|------|-------|------|----|-------|----------|
| cross | A1 | A3 | 1e-05 | 0.492 | 0.480 |
| cross | A1 | A3 | 1e-04 | 0.399 | 0.480 |
| cross | A1 | A3 | 1e-03 | 0.484 | 0.480 |
| cross | A1 | A3 | 1e-02 | 0.567 | 0.480 |
| cross | A1 | A3 | 1e-01 | 0.527 | 0.520 |
| cross | A3 | A1 | 1e-05 | 0.454 | 0.500 |
| cross | A3 | A1 | 1e-04 | 0.444 | 0.500 |
| cross | A3 | A1 | 1e-03 | 0.359 | 0.500 |
| cross | A3 | A1 | 1e-02 | 0.457 | 0.500 |
| cross | A3 | A1 | 1e-01 | 0.477 | 0.500 |
| joint | A1+A3 | held_out_split | 1e-05 | 0.781 | 0.701 |
| joint | A1+A3 | held_out_split | 1e-04 | 0.732 | 0.655 |
| joint | A1+A3 | held_out_split | 1e-03 | 0.517 | 0.508 |
| joint | A1+A3 | held_out_split | 1e-02 | 0.399 | 0.508 |
| joint | A1+A3 | held_out_split | 1e-01 | 0.538 | 0.508 |

## Comparison to v1 Baselines

| Metric | v1 Baseline | v2 Result | Delta |
|--------|-------------|-----------|-------|
| Joint probe AUROC | 0.742 | 0.781 | +0.039 |
| Cross-benchmark AUROC (avg) | 0.550 | 0.522 | -0.028 |
| Within-run AUROC (avg) | 0.490 | 0.389 | -0.101 |

## Recommendations

- Cross-benchmark AUROC still low. Consider trying different feature selection (e.g., mutual information) or non-linear probes.
- Joint probe AUROC improved over v1 baseline.
