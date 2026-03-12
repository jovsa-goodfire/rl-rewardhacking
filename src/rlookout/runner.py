"""Experiment runner: config -> data -> scorers -> results JSON."""

from __future__ import annotations

import json
from datetime import datetime
from itertools import permutations

from src.rlookout.config import ExperimentConfig
from src.rlookout.data import SAEDataset
from src.rlookout.probe_trainer import cross_benchmark_probe, joint_probe
from src.rlookout.sae_utils import load_sae, load_feature_labels
from src.rlookout.scorers import (
    SCORER_REGISTRY,
    DiffOfMeansScorer,
    LinearProbeScorer,
    ScoredFeature,
    auroc_for_features,
    build_ensemble,
    validate_semantic_candidates,
)


def run_experiment(config: ExperimentConfig) -> dict:
    """Main entry point. Orchestrates the full pipeline.

    1. Load SAE + labels
    2. For each run: load data, run scorers, build ensemble
    3. Cross-benchmark probe experiments
    4. Semantic candidate validation
    5. Write results JSON
    """
    print(f"[experiment] {config.name}")

    # --- 1. Load SAE + labels ---
    sae_device = "cuda" if config.probe.device == "cuda" else "cpu"
    print(f"[sae] Loading from {config.sae.checkpoint_path}")
    sae = load_sae(config.sae.checkpoint_path, device=sae_device)

    labels_map: dict[int, str] = {}
    if config.sae.labels_path:
        labels_map = load_feature_labels(config.sae.labels_path)

    # --- 2. Per-run scoring ---
    datasets: dict[str, SAEDataset] = {}
    per_run_results = []

    for run_spec in config.runs:
        print(f"\n[run] {run_spec.name}: {run_spec.description}")
        ds = SAEDataset.from_checkpoint(run_spec, sae)
        datasets[run_spec.name] = ds
        print(f"  samples={len(ds)}  RH={ds.metadata['n_rh']}")

        method_results = {}
        for method_name in config.scorer.methods:
            if method_name not in SCORER_REGISTRY:
                print(f"  [skip] Unknown scorer: {method_name}")
                continue

            print(f"  [{method_name}] scoring...")
            scorer_cls = SCORER_REGISTRY[method_name]

            # Instantiate scorer with appropriate args
            if method_name == "diff_of_means":
                scorer = scorer_cls(sae=sae, top_k=config.scorer.top_k)
            elif method_name == "linear_probe":
                scorer = scorer_cls(top_k=config.scorer.top_k)
            else:
                scorer = scorer_cls(top_k=config.scorer.top_k)

            features = scorer.score(ds)
            feature_ids = [f.feature_id for f in features]
            auroc = auroc_for_features(ds, feature_ids)

            method_results[method_name] = features
            print(f"    AUROC={auroc:.3f}  top: {feature_ids[:5]}")

        # Ensemble
        ensemble_ids = build_ensemble(
            method_results, config.scorer.top_k, config.scorer.ensemble_min_methods
        )
        ensemble_auroc = auroc_for_features(ds, ensemble_ids)

        per_run_results.append({
            "run_name": run_spec.name,
            "description": run_spec.description,
            "n_samples": len(ds),
            "n_rh": ds.metadata["n_rh"],
            "methods": {
                name: {
                    "auroc": auroc_for_features(ds, [f.feature_id for f in feats]),
                    "top_features": [
                        {
                            "feature_id": f.feature_id,
                            "score": f.score,
                            "label": labels_map.get(f.feature_id, ""),
                        }
                        for f in feats
                    ],
                }
                for name, feats in method_results.items()
            },
            "ensemble_feature_ids": ensemble_ids,
            "ensemble_auroc": ensemble_auroc,
            "ensemble_labels": {
                str(fid): labels_map.get(fid, "") for fid in ensemble_ids
            },
        })

    # --- 3. Cross-benchmark probe experiments ---
    cross_results = []
    run_names = list(datasets.keys())

    if len(run_names) >= 2:
        # Pairwise: train on each, test on each other
        for train_name, test_name in permutations(run_names, 2):
            print(f"\n[cross-probe] train={train_name} -> test={test_name}")
            result = cross_benchmark_probe(
                datasets[train_name], datasets[test_name], config.probe
            )
            print(f"  val_auroc={result.val_metrics['auroc']:.3f}"
                  f"  val_acc={result.val_metrics['accuracy']:.3f}")

            cross_results.append({
                "train_run": train_name,
                "test_run": test_name,
                "probe_auroc": result.val_metrics["auroc"],
                "probe_accuracy": result.val_metrics["accuracy"],
                "top_features": [
                    {
                        "feature_id": f.feature_id,
                        "score": f.score,
                        "label": labels_map.get(f.feature_id, ""),
                    }
                    for f in result.feature_importances[:config.scorer.top_k]
                ],
            })

        # Joint training on all runs
        print(f"\n[joint-probe] training on {'+'.join(run_names)}")
        joint_result = joint_probe(list(datasets.values()), config.probe)
        print(f"  val_auroc={joint_result.val_metrics['auroc']:.3f}"
              f"  val_acc={joint_result.val_metrics['accuracy']:.3f}")

        cross_results.append({
            "train_run": "+".join(run_names),
            "test_run": "held_out_split",
            "probe_auroc": joint_result.val_metrics["auroc"],
            "probe_accuracy": joint_result.val_metrics["accuracy"],
            "top_features": [
                {
                    "feature_id": f.feature_id,
                    "score": f.score,
                    "label": labels_map.get(f.feature_id, ""),
                }
                for f in joint_result.feature_importances[:config.scorer.top_k]
            ],
        })

    # --- 4. Semantic candidate validation ---
    semantic_results = {}
    if config.semantic_candidate_ids:
        print(f"\n[semantic] Validating {len(config.semantic_candidate_ids)} candidates")
        for ds in datasets.values():
            validation = validate_semantic_candidates(ds, config.semantic_candidate_ids)
            for fid, stats in validation.items():
                key = f"{ds.name}:{fid}"
                semantic_results[key] = {
                    "run": ds.name,
                    "feature_id": fid,
                    "label": labels_map.get(fid, ""),
                    **stats,
                }

    # --- 5. Cross-run overlap ---
    all_ensembles = {r["run_name"]: set(r["ensemble_feature_ids"]) for r in per_run_results}
    if len(all_ensembles) >= 2:
        overlap = set.intersection(*all_ensembles.values())
    else:
        overlap = set()

    # --- 6. Assemble and save ---
    output = {
        "config": config.model_dump(),
        "timestamp": datetime.now().isoformat(),
        "per_run": per_run_results,
        "cross_benchmark": cross_results,
        "semantic_validation": semantic_results,
        "cross_run_overlap": sorted(overlap),
        "cross_run_overlap_labels": {
            str(fid): labels_map.get(fid, "") for fid in sorted(overlap)
        },
    }

    # Save
    out_dir = config.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[done] Saved to {out_path}")

    return output
