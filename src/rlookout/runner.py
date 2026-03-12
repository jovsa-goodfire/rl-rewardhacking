"""Experiment runner: config -> data -> scorers -> MI techniques -> insights -> results JSON."""

from __future__ import annotations

import json
from datetime import datetime
from itertools import permutations
from pathlib import Path

from src.rlookout.config import ExperimentConfig
from src.rlookout.data import SAEDataset
from src.rlookout.insights import Insight, analyze_experiment, format_insights
from src.rlookout.manifest import build_manifest, save_manifest
from src.rlookout.mi import (
    MIResult,
    contrastive_cross_benchmark,
    gradient_aligned_features,
    llm_refined_features,
    relabel_features_for_rh,
)
from src.rlookout.probe_trainer import cross_benchmark_probe, joint_probe
from src.rlookout.sae_utils import load_sae, load_feature_labels
from src.rlookout.scorers import (
    SCORER_REGISTRY,
    ContrastiveScorer,
    GradientAlignedScorer,
    ScoredFeature,
    auroc_for_features,
    build_ensemble,
    validate_semantic_candidates,
)


def _to_original_id(ds: SAEDataset, feature_id: int) -> int:
    """Map a column-based feature ID back to the original SAE feature ID."""
    if feature_id < ds.features.shape[1]:
        return ds.original_feature_id(feature_id)
    return feature_id  # already an original ID (e.g. from DiffOfMeans)


def run_experiment(config: ExperimentConfig) -> dict:
    """Main entry point. Orchestrates the full pipeline.

    1. Load SAE + labels
    2. For each run: load data, run scorers, build ensemble
    3. Run MI techniques (if configured)
    4. Cross-benchmark probe experiments
    5. Semantic candidate validation
    6. Generate insights
    7. Save results JSON + manifest + research log
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

        # Label broadening: include attempted reward hacks as positive
        if config.include_attempted_rh:
            ds = ds.broaden_labels()
            print(f"  [label broadening] RH count: {ds.metadata['n_rh']} (was {int(sum(1 for l in ds.reward_hack_labels if l == 'Reward Hack'))})")

        # Variance filter: reduce feature dimensionality
        if config.scorer.variance_top_k is not None:
            ds = ds.select_top_k_by_variance(config.scorer.variance_top_k)
            print(f"  [variance filter] {ds.features.shape[1]} features (from {sae.encoder.weight.shape[0]})")

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
            elif method_name == "gradient_aligned":
                scorer = scorer_cls(
                    sae=sae, all_datasets=datasets, labels_map=labels_map,
                    top_k=config.scorer.top_k,
                )
            elif method_name == "contrastive":
                scorer = scorer_cls(
                    sae=sae, all_datasets=datasets, labels_map=labels_map,
                    top_k=config.scorer.top_k,
                )
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

        # Map feature IDs back to original SAE space for reporting
        mapped_ensemble_ids = [_to_original_id(ds, fid) for fid in ensemble_ids]

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
                            "feature_id": _to_original_id(ds, f.feature_id),
                            "score": f.score,
                            "label": labels_map.get(_to_original_id(ds, f.feature_id), ""),
                        }
                        for f in feats
                    ],
                }
                for name, feats in method_results.items()
            },
            "ensemble_feature_ids": mapped_ensemble_ids,
            "ensemble_auroc": ensemble_auroc,
            "ensemble_labels": {
                str(fid): labels_map.get(fid, "") for fid in mapped_ensemble_ids
            },
        })

    # --- 3. MI techniques (if configured) ---
    mi_technique_results: dict[str, dict] = {}

    if config.techniques and len(datasets) >= 2:
        print(f"\n[mi] Running MI techniques: {config.techniques}")

        if "gradient_aligned" in config.techniques:
            print("  [gradient_aligned] Finding cross-benchmark gradient-aligned features...")
            ga_result = gradient_aligned_features(
                sae=sae, datasets=datasets, labels_map=labels_map,
                k=config.mi.gradient_aligned_k,
            )
            mi_technique_results["gradient_aligned"] = _mi_result_to_dict(ga_result)
            print(f"    Found {len(ga_result.candidates)} shared features (from {ga_result.metadata.get('k_per_benchmark', '?')} per benchmark)")

        if "contrastive_cross_benchmark" in config.techniques:
            print("  [contrastive] Finding features aligned with shared RH direction...")
            cc_result = contrastive_cross_benchmark(
                sae=sae, datasets=datasets, labels_map=labels_map,
                k=config.mi.contrastive_k,
            )
            mi_technique_results["contrastive_cross_benchmark"] = _mi_result_to_dict(cc_result)
            n_consistent = cc_result.metadata.get("n_sign_consistent", 0)
            print(f"    Found {len(cc_result.candidates)} features ({n_consistent} sign-consistent)")

        if "llm_refined" in config.techniques:
            # Use candidates from gradient_aligned or contrastive as input
            input_candidates = []
            for tech in ("gradient_aligned", "contrastive_cross_benchmark"):
                if tech in mi_technique_results:
                    # Reconstruct FeatureCandidate objects from stored results
                    from goodfire_core.interventions.feature_selection import FeatureCandidate
                    for c in mi_technique_results[tech]["candidates"]:
                        input_candidates.append(FeatureCandidate(
                            feature_id=c["feature_id"],
                            score=c["score"],
                            label=c.get("label"),
                        ))

            if input_candidates:
                print(f"  [llm_refined] Filtering {len(input_candidates)} candidates with LLM...")
                lr_result = llm_refined_features(
                    candidates=input_candidates,
                    k=config.mi.llm_refined_k,
                    provider=config.mi.llm_provider,
                    model=config.mi.llm_model,
                )
                mi_technique_results["llm_refined"] = _mi_result_to_dict(lr_result)
                print(f"    Kept {len(lr_result.candidates)} features after LLM refinement")
            else:
                print("  [llm_refined] No input candidates available, skipping")

    # --- 4. Cross-benchmark probe experiments ---
    cross_results = []
    run_names = list(datasets.keys())
    l1_sweep_results = []

    if len(run_names) >= 2:
        l1_values = config.probe.l1_sweep or [config.probe.l1_weight]

        # Pairwise: train on each, test on each other
        for train_name, test_name in permutations(run_names, 2):
            best_result = None
            best_auroc = -1.0
            best_l1 = None

            for l1 in l1_values:
                probe_cfg = config.probe.model_copy(update={"l1_weight": l1})
                result = cross_benchmark_probe(
                    datasets[train_name], datasets[test_name], probe_cfg
                )
                auroc = result.val_metrics["auroc"]

                l1_sweep_results.append({
                    "type": "cross",
                    "train_run": train_name,
                    "test_run": test_name,
                    "l1_weight": l1,
                    "auroc": auroc,
                    "accuracy": result.val_metrics["accuracy"],
                })

                if auroc > best_auroc:
                    best_auroc = auroc
                    best_result = result
                    best_l1 = l1

            if len(l1_values) > 1:
                print(f"\n[cross-probe] train={train_name} -> test={test_name}  best_l1={best_l1:.0e}")
            else:
                print(f"\n[cross-probe] train={train_name} -> test={test_name}")
            print(f"  val_auroc={best_result.val_metrics['auroc']:.3f}"
                  f"  val_acc={best_result.val_metrics['accuracy']:.3f}")

            train_ds = datasets[train_name]
            cross_results.append({
                "train_run": train_name,
                "test_run": test_name,
                "probe_auroc": best_result.val_metrics["auroc"],
                "probe_accuracy": best_result.val_metrics["accuracy"],
                "best_l1_weight": best_l1,
                "top_features": [
                    {
                        "feature_id": _to_original_id(train_ds, f.feature_id),
                        "score": f.score,
                        "label": labels_map.get(_to_original_id(train_ds, f.feature_id), ""),
                    }
                    for f in best_result.feature_importances[:config.scorer.top_k]
                ],
            })

        # Joint training on all runs
        best_joint = None
        best_joint_auroc = -1.0
        best_joint_l1 = None

        for l1 in l1_values:
            probe_cfg = config.probe.model_copy(update={"l1_weight": l1})
            joint_result = joint_probe(list(datasets.values()), probe_cfg)
            auroc = joint_result.val_metrics["auroc"]

            l1_sweep_results.append({
                "type": "joint",
                "train_run": "+".join(run_names),
                "test_run": "held_out_split",
                "l1_weight": l1,
                "auroc": auroc,
                "accuracy": joint_result.val_metrics["accuracy"],
            })

            if auroc > best_joint_auroc:
                best_joint_auroc = auroc
                best_joint = joint_result
                best_joint_l1 = l1

        if len(l1_values) > 1:
            print(f"\n[joint-probe] training on {'+'.join(run_names)}  best_l1={best_joint_l1:.0e}")
        else:
            print(f"\n[joint-probe] training on {'+'.join(run_names)}")
        print(f"  val_auroc={best_joint.val_metrics['auroc']:.3f}"
              f"  val_acc={best_joint.val_metrics['accuracy']:.3f}")

        ref_ds = datasets[run_names[0]]
        cross_results.append({
            "train_run": "+".join(run_names),
            "test_run": "held_out_split",
            "probe_auroc": best_joint.val_metrics["auroc"],
            "probe_accuracy": best_joint.val_metrics["accuracy"],
            "best_l1_weight": best_joint_l1,
            "top_features": [
                {
                    "feature_id": _to_original_id(ref_ds, f.feature_id),
                    "score": f.score,
                    "label": labels_map.get(_to_original_id(ref_ds, f.feature_id), ""),
                }
                for f in best_joint.feature_importances[:config.scorer.top_k]
            ],
        })

    # --- 5. Semantic candidate validation ---
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

    # --- 6. Cross-run overlap ---
    all_ensembles = {r["run_name"]: set(r["ensemble_feature_ids"]) for r in per_run_results}
    if len(all_ensembles) >= 2:
        overlap = set.intersection(*all_ensembles.values())
    else:
        overlap = set()

    # --- 7. Assemble results ---
    output = {
        "config": config.model_dump(),
        "timestamp": datetime.now().isoformat(),
        "per_run": per_run_results,
        "mi_techniques": mi_technique_results,
        "cross_benchmark": cross_results,
        "l1_sweep_detail": l1_sweep_results,
        "semantic_validation": semantic_results,
        "cross_run_overlap": sorted(overlap),
        "cross_run_overlap_labels": {
            str(fid): labels_map.get(fid, "") for fid in sorted(overlap)
        },
    }

    # --- 8. Generate insights ---
    insights = analyze_experiment(output, labels_map)
    output["insights"] = [
        {
            "category": ins.category,
            "severity": ins.severity,
            "title": ins.title,
            "detail": ins.detail,
            "recommendation": ins.recommendation,
        }
        for ins in insights
    ]

    # --- 9. Save ---
    out_dir = config.results_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # Results JSON
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[done] Saved to {out_path}")

    # Manifest
    manifest = build_manifest(
        config_dict=config.model_dump(),
        results=output,
        results_path=str(out_path),
        insights=insights,
    )
    save_manifest(manifest, out_dir / "manifest.yaml")
    print(f"[manifest] Saved to {out_dir / 'manifest.yaml'}")

    # Research log
    write_research_log(output, out_dir / "research_log.md")

    return output


def _mi_result_to_dict(result: MIResult) -> dict:
    """Convert MIResult to serializable dict."""
    return {
        "technique": result.technique,
        "candidates": [
            {
                "feature_id": c.feature_id,
                "score": c.score,
                "label": c.label,
                "metadata": c.metadata,
            }
            for c in result.candidates
        ],
        "metadata": result.metadata,
    }


def write_research_log(results: dict, path: Path) -> None:
    """Write a human-readable markdown research log from experiment results."""
    config = results["config"]
    lines = [
        f"# Research Log: {config['name']}",
        f"",
        f"**Date:** {results['timestamp'][:10]}",
        f"",
        f"## Configuration",
        f"",
        f"- **Variance filter:** top-{config['scorer'].get('variance_top_k', 'None')} features",
        f"- **Include attempted RH:** {config.get('include_attempted_rh', False)}",
        f"- **L1 sweep:** {config['probe'].get('l1_sweep', 'None')}",
        f"- **Default L1:** {config['probe']['l1_weight']}",
        f"- **MI techniques:** {config.get('techniques', [])}",
        f"",
        f"## Per-Run Summary",
        f"",
    ]

    for run in results["per_run"]:
        lines.append(f"### {run['run_name']}: {run['description']}")
        lines.append(f"- Samples: {run['n_samples']}  |  RH: {run['n_rh']}")
        lines.append(f"- Ensemble AUROC: {run['ensemble_auroc']:.3f}")
        for method, data in run["methods"].items():
            lines.append(f"  - {method}: AUROC={data['auroc']:.3f}")
        lines.append("")

    # MI technique results
    mi_results = results.get("mi_techniques", {})
    if mi_results:
        lines.append("## MI Technique Results")
        lines.append("")
        for tech_name, tech_data in mi_results.items():
            candidates = tech_data.get("candidates", [])
            metadata = tech_data.get("metadata", {})
            lines.append(f"### {tech_name}")
            lines.append(f"- Candidates found: {len(candidates)}")
            for k, v in metadata.items():
                lines.append(f"- {k}: {v}")
            if candidates:
                lines.append("")
                lines.append("| Feature | Score | Label |")
                lines.append("|---------|-------|-------|")
                for c in candidates[:10]:
                    label = (c.get("label") or "")[:50]
                    lines.append(f"| {c['feature_id']} | {c['score']:.4f} | {label} |")
            lines.append("")

    # Cross-benchmark results
    lines.append("## Cross-Benchmark Probe Results")
    lines.append("")
    lines.append("| Train | Test | AUROC | Accuracy | Best L1 |")
    lines.append("|-------|------|-------|----------|---------|")
    for cr in results["cross_benchmark"]:
        l1_str = f"{cr.get('best_l1_weight', 'N/A'):.0e}" if isinstance(cr.get("best_l1_weight"), float) else "N/A"
        lines.append(f"| {cr['train_run']} | {cr['test_run']} | {cr['probe_auroc']:.3f} | {cr['probe_accuracy']:.3f} | {l1_str} |")
    lines.append("")

    # L1 sweep detail
    if results.get("l1_sweep_detail"):
        lines.append("## L1 Sweep Detail")
        lines.append("")
        lines.append("| Type | Train | Test | L1 | AUROC | Accuracy |")
        lines.append("|------|-------|------|----|-------|----------|")
        for sr in results["l1_sweep_detail"]:
            lines.append(f"| {sr['type']} | {sr['train_run']} | {sr['test_run']} | {sr['l1_weight']:.0e} | {sr['auroc']:.3f} | {sr['accuracy']:.3f} |")
        lines.append("")

    # Comparison to baselines
    lines.append("## Comparison to v1 Baselines")
    lines.append("")
    lines.append("| Metric | v1 Baseline | Current | Delta |")
    lines.append("|--------|-------------|---------|-------|")

    baselines = {
        "Joint probe AUROC": 0.742,
        "Cross-benchmark AUROC (avg)": 0.55,
        "Within-run AUROC (avg)": 0.49,
    }

    cross_entries = [cr for cr in results["cross_benchmark"] if cr["test_run"] != "held_out_split"]
    joint_entries = [cr for cr in results["cross_benchmark"] if cr["test_run"] == "held_out_split"]

    current_metrics = {}
    if joint_entries:
        current_metrics["Joint probe AUROC"] = joint_entries[0]["probe_auroc"]
    if cross_entries:
        current_metrics["Cross-benchmark AUROC (avg)"] = sum(cr["probe_auroc"] for cr in cross_entries) / len(cross_entries)

    within_aurocs = []
    for run in results["per_run"]:
        if "linear_probe" in run["methods"]:
            within_aurocs.append(run["methods"]["linear_probe"]["auroc"])
    if within_aurocs:
        current_metrics["Within-run AUROC (avg)"] = sum(within_aurocs) / len(within_aurocs)

    for metric, baseline in baselines.items():
        val = current_metrics.get(metric)
        if val is not None:
            delta = val - baseline
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {metric} | {baseline:.3f} | {val:.3f} | {sign}{delta:.3f} |")
        else:
            lines.append(f"| {metric} | {baseline:.3f} | N/A | N/A |")
    lines.append("")

    # Insights section
    insights_data = results.get("insights", [])
    if insights_data:
        lines.append("## Insights")
        lines.append("")
        severity_icon = {"finding": "**[FINDING]**", "warning": "**[WARNING]**", "info": "[INFO]"}
        for ins in insights_data:
            icon = severity_icon.get(ins.get("severity", ""), "")
            lines.append(f"- {icon} **{ins['title']}**: {ins['detail']}")
            if ins.get("recommendation"):
                lines.append(f"  - *Recommendation:* {ins['recommendation']}")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    if current_metrics.get("Cross-benchmark AUROC (avg)", 0) > 0.6:
        lines.append("- Cross-benchmark AUROC improved meaningfully. Consider re-running R6 steering eval with the improved feature set.")
    else:
        lines.append("- Cross-benchmark AUROC still low. Consider trying MI techniques (gradient_aligned, contrastive_cross_benchmark) or non-linear probes.")
    if current_metrics.get("Joint probe AUROC", 0) > baselines["Joint probe AUROC"]:
        lines.append("- Joint probe AUROC improved over v1 baseline.")
    lines.append("")

    path.write_text("\n".join(lines))
    print(f"[log] Research log saved to {path}")
