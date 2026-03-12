"""
Improved Task 3: SAE feature selection for reward hacking detection.

Three methods, compared against each other and ensembled:
  1. Diff-of-means + cosine projection onto SAE decoder directions
     (which features best explain the RH vs non-RH activation difference?)
  2. Per-class mean SAE activation
     (which SAE features fire most on RH vs non-RH samples?)
  3. Pearson correlation (original baseline)

Output: results/rlookout/qwen3-4b/r3_improved_results.json

Usage:
    cd /mnt/polished-lake/home/jsardinha/rl-rewardhacking
    uv run python scripts/r3_feature_selection.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from goodfire_core.saes.batch_topk import BatchTopKSAE
from goodfire_core.interventions.utils import select_features_by_gradient

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/mnt/polished-lake/home/jsardinha/rl-rewardhacking")
SAE_PATH = "/mnt/polished-lake/artifacts/public/saes/qwen3-4b/checkpoints/chunked-layer20-k64-ddp64-20251210_231238/final_batch_topk_sae.pt"
LABELS_PATH = "/mnt/polished-lake/artifacts/public/saes/qwen3-4b/checkpoints/chunked-layer20-k64-ddp64-20251210_231238/autointerp_final/labels/labels.jsonl"
RUNS = {
    "A1": (REPO / "results/rlookout/qwen3-4b/20260310_143530_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline/checkpoint_150.pt", "LeetCode step 150"),
    "A3": (REPO / "results/rlookout/qwen3-4b/20260310_204521_impossible_bench_train_hard_filtered_rh_simple_overwrite_tests_baseline/checkpoint_200.pt", "Impossible Bench step 200"),
}

TOP_K = 20
RESULTS_PATH = REPO / "results/rlookout/qwen3-4b/r3_improved_results.json"

# ---------------------------------------------------------------------------
# Load SAE + labels
# ---------------------------------------------------------------------------
print("[info] Loading SAE...")
# Load manually with strict=False — checkpoint predates the _fired_buffer addition
_ckpt = torch.load(SAE_PATH, map_location="cpu")
sae = BatchTopKSAE(**_ckpt["model_config"])
sae.load_state_dict(_ckpt["state_dict"], strict=False)
sae = sae.cuda().eval()

print("[info] Loading feature labels...")
labels_map: dict[int, str] = {}
with open(LABELS_PATH) as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("labels"):
            labels_map[entry["feature_id"]] = entry["labels"][0]["label"]

# ---------------------------------------------------------------------------
# Helper: encode activations through SAE → dense feature matrix
# ---------------------------------------------------------------------------
def encode_to_dense(acts: torch.Tensor) -> np.ndarray:
    """Run activations through SAE, return dense (n_samples, d_sae) numpy array."""
    with torch.no_grad():
        feat_matrix = sae.encode(acts.cuda())  # already dense (n, d_sae)
    return feat_matrix.cpu().numpy()


# ---------------------------------------------------------------------------
# Helper: AUROC for a set of features against RH labels
# ---------------------------------------------------------------------------
def auroc_ensemble(feat_matrix: np.ndarray, feature_ids: list[int], rh_np: np.ndarray) -> float:
    """Mean-pooled score across features (sum of standardised activations)."""
    if not feature_ids:
        return 0.5
    scores = feat_matrix[:, feature_ids].sum(axis=1)
    if scores.std() < 1e-8:
        return 0.5
    return float(roc_auc_score(rh_np, scores))


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------
all_results: dict = {}

for run_name, (ckpt_path, desc) in RUNS.items():
    print(f"\n{'='*60}")
    print(f"[run] {run_name}: {desc}")

    data = torch.load(ckpt_path, map_location="cpu")
    acts = data["activations"].float()            # (n, 2560)
    rh_np = np.array([float(l) for l in data["labels"]])
    rh_mask = rh_np.astype(bool)
    print(f"  samples={len(rh_np)}  RH={rh_mask.sum()}  non-RH={(~rh_mask).sum()}")

    # -----------------------------------------------------------------------
    # Encode all activations through SAE once (shared across methods)
    # -----------------------------------------------------------------------
    print("  [SAE] Encoding activations...")
    feat_matrix = encode_to_dense(acts)           # (n, 20480)

    # -----------------------------------------------------------------------
    # Method 1: Diff-of-means + cosine projection onto SAE decoder directions
    # -----------------------------------------------------------------------
    print("  [M1] Diff-of-means projection...")
    rh_mean = acts[rh_mask].mean(dim=0)           # (2560,)
    non_rh_mean = acts[~rh_mask].mean(dim=0)      # (2560,)
    diff_vec = (rh_mean - non_rh_mean).unsqueeze(0).unsqueeze(0)  # (1, 1, 2560)

    # select_features_by_gradient computes cosine sim between diff_vec and each decoder direction
    feat_ids_m1, scores_m1 = select_features_by_gradient(
        sae=sae,
        gradients=diff_vec.cuda(),
        k=TOP_K,
        use_cosine=True,
    )
    m1_results = [(fid, float(s)) for fid, s in zip(feat_ids_m1, scores_m1.tolist())]

    # -----------------------------------------------------------------------
    # Method 2: Per-class mean SAE activation
    # -----------------------------------------------------------------------
    print("  [M2] Per-class mean SAE activation...")
    rh_mean_feat = feat_matrix[rh_mask].mean(axis=0)      # (20480,)
    non_rh_mean_feat = feat_matrix[~rh_mask].mean(axis=0) # (20480,)
    mean_diff = rh_mean_feat - non_rh_mean_feat            # positive = fires more on RH

    top_m2_idx = np.argsort(np.abs(mean_diff))[::-1][:TOP_K]
    m2_results = [(int(i), float(mean_diff[i])) for i in top_m2_idx]

    # -----------------------------------------------------------------------
    # Method 3: Pearson correlation (original baseline)
    # -----------------------------------------------------------------------
    print("  [M3] Pearson correlation...")
    correlations = []
    for i in range(feat_matrix.shape[1]):
        col = feat_matrix[:, i]
        if col.std() < 1e-8:
            continue
        corr = float(np.corrcoef(col, rh_np)[0, 1])
        if not np.isnan(corr):
            correlations.append((i, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    m3_results = correlations[:TOP_K]
    n_strong_corr = sum(1 for _, c in correlations if abs(c) > 0.3)

    # -----------------------------------------------------------------------
    # Ensemble: features appearing in ≥2 methods (by feature_id)
    # -----------------------------------------------------------------------
    set_m1 = set(fid for fid, _ in m1_results)
    set_m2 = set(fid for fid, _ in m2_results)
    set_m3 = set(fid for fid, _ in m3_results)

    in_all_three = set_m1 & set_m2 & set_m3
    in_two = (set_m1 & set_m2) | (set_m1 & set_m3) | (set_m2 & set_m3)
    ensemble_ids = sorted(in_all_three) or sorted(in_two)  # prefer all-three

    # -----------------------------------------------------------------------
    # AUROC scores
    # -----------------------------------------------------------------------
    auroc_m1 = auroc_ensemble(feat_matrix, list(set_m1), rh_np)
    auroc_m2 = auroc_ensemble(feat_matrix, list(set_m2), rh_np)
    auroc_m3 = auroc_ensemble(feat_matrix, list(set_m3), rh_np)
    auroc_ensemble_score = auroc_ensemble(feat_matrix, ensemble_ids, rh_np)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print(f"\n  AUROC — M1(diff-of-means): {auroc_m1:.3f}  M2(mean-act): {auroc_m2:.3f}  "
          f"M3(pearson): {auroc_m3:.3f}  Ensemble: {auroc_ensemble_score:.3f}")
    print(f"  Pearson |corr|>0.3: {n_strong_corr}  |  Overlap in ≥2 methods: {len(in_two)}  |  All 3: {len(in_all_three)}")

    print(f"\n  Top features by method:")
    for label, results in [("M1 diff-of-means", m1_results[:5]), ("M2 mean-act diff", m2_results[:5]), ("M3 Pearson", m3_results[:5])]:
        print(f"\n  [{label}]")
        for fid, score in results:
            lbl = labels_map.get(fid, "no label")
            print(f"    Feature {fid}: score={score:+.4f} | {lbl}")

    if ensemble_ids:
        print(f"\n  [Ensemble — in ≥2 methods]")
        for fid in ensemble_ids[:10]:
            lbl = labels_map.get(fid, "no label")
            print(f"    Feature {fid} | {lbl}")

    # -----------------------------------------------------------------------
    # Store results
    # -----------------------------------------------------------------------
    def annotate(results):
        return [{"feature_id": fid, "score": score, "label": labels_map.get(fid, "no label")}
                for fid, score in results]

    all_results[run_name] = {
        "desc": desc,
        "n_samples": int(len(rh_np)),
        "n_rh": int(rh_mask.sum()),
        "n_strong_corr_pearson": n_strong_corr,
        "auroc": {
            "m1_diff_of_means": auroc_m1,
            "m2_mean_activation": auroc_m2,
            "m3_pearson": auroc_m3,
            "ensemble": auroc_ensemble_score,
        },
        "m1_diff_of_means": annotate(m1_results),
        "m2_mean_activation": annotate(m2_results),
        "m3_pearson": annotate(m3_results),
        "ensemble_feature_ids": ensemble_ids,
        "ensemble_in_all_three": sorted(in_all_three),
        "ensemble_in_two_plus": sorted(in_two),
        "ensemble_labels": {str(fid): labels_map.get(fid, "no label") for fid in ensemble_ids},
    }

# Cross-run overlap
set_a1_ensemble = set(all_results["A1"]["ensemble_feature_ids"])
set_a3_ensemble = set(all_results["A3"]["ensemble_feature_ids"])
cross_run_overlap = sorted(set_a1_ensemble & set_a3_ensemble)
all_results["cross_run_overlap_ensemble"] = cross_run_overlap
all_results["cross_run_overlap_labels"] = {str(fid): labels_map.get(fid, "no label") for fid in cross_run_overlap}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n[done] Saved to {RESULTS_PATH}")
print(f"Cross-run ensemble overlap: {cross_run_overlap}")
