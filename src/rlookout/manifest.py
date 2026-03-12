"""Experiment manifest tracking (silico pattern).

Each experiment saves a manifest.yaml alongside results with inputs, outputs,
metrics, and learnings. Future experiments can load past manifests to see
what's been tried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class ExperimentManifest:
    name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Inputs
    data: dict = field(default_factory=dict)  # {run_name: checkpoint_path, ...}
    sae: dict = field(default_factory=dict)  # {checkpoint, layer, labels}

    # Configuration
    techniques: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    # Outputs
    results_path: str = ""

    # Metrics
    metrics: dict = field(default_factory=dict)

    # Learnings
    insights: list[dict] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)


def build_manifest(
    config_dict: dict,
    results: dict,
    results_path: str,
    insights: list | None = None,
) -> ExperimentManifest:
    """Build a manifest from experiment config and results."""
    # Extract key metrics
    metrics: dict = {}

    cross = results.get("cross_benchmark", [])
    cross_pairs = [cr for cr in cross if cr.get("test_run") != "held_out_split"]
    joint = [cr for cr in cross if cr.get("test_run") == "held_out_split"]

    if cross_pairs:
        metrics["cross_benchmark_auroc_avg"] = sum(cr["probe_auroc"] for cr in cross_pairs) / len(cross_pairs)
        metrics["cross_benchmark_auroc_best"] = max(cr["probe_auroc"] for cr in cross_pairs)
    if joint:
        metrics["joint_auroc"] = joint[0]["probe_auroc"]

    # Per-run within-run AUROCs
    for run in results.get("per_run", []):
        for method, data in run.get("methods", {}).items():
            metrics[f"{run['run_name']}_{method}_auroc"] = data["auroc"]

    # MI technique metrics
    for tech_name, tech_data in results.get("mi_techniques", {}).items():
        candidates = tech_data.get("candidates", [])
        metrics[f"mi_{tech_name}_n_candidates"] = len(candidates)
        if candidates:
            metrics[f"mi_{tech_name}_top_score"] = candidates[0].get("score", 0)

    # Build insight/recommendation summaries
    insight_dicts = []
    recommendation_dicts = []
    if insights:
        for ins in insights:
            d = {
                "category": ins.category if hasattr(ins, "category") else ins.get("category", ""),
                "severity": ins.severity if hasattr(ins, "severity") else ins.get("severity", ""),
                "title": ins.title if hasattr(ins, "title") else ins.get("title", ""),
                "detail": ins.detail if hasattr(ins, "detail") else ins.get("detail", ""),
            }
            if hasattr(ins, "category") and ins.category == "recommendation":
                recommendation_dicts.append(d)
            elif isinstance(ins, dict) and ins.get("category") == "recommendation":
                recommendation_dicts.append(d)
            else:
                insight_dicts.append(d)

    return ExperimentManifest(
        name=config_dict.get("name", "unknown"),
        data={
            run["name"]: run["checkpoint_path"]
            for run in config_dict.get("runs", [])
        },
        sae={
            "checkpoint": config_dict.get("sae", {}).get("checkpoint_path", ""),
            "layer": config_dict.get("sae", {}).get("layer", ""),
            "labels": config_dict.get("sae", {}).get("labels_path", ""),
        },
        techniques=config_dict.get("techniques", []),
        config={
            k: v for k, v in config_dict.items()
            if k not in ("name", "runs", "sae")
        },
        results_path=results_path,
        metrics=metrics,
        insights=insight_dicts,
        recommendations=recommendation_dicts,
    )


def save_manifest(manifest: ExperimentManifest, path: Path) -> None:
    """Save manifest as YAML."""
    data = {
        "name": manifest.name,
        "timestamp": manifest.timestamp,
        "data": manifest.data,
        "sae": manifest.sae,
        "techniques": manifest.techniques,
        "config": manifest.config,
        "results_path": manifest.results_path,
        "metrics": manifest.metrics,
        "insights": manifest.insights,
        "recommendations": manifest.recommendations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_manifest(path: Path) -> ExperimentManifest:
    """Load manifest from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentManifest(**data)


def load_all_manifests(experiments_dir: Path) -> list[ExperimentManifest]:
    """Load all manifests from an experiments directory."""
    manifests = []
    for manifest_path in sorted(experiments_dir.glob("*/manifest.yaml")):
        try:
            manifests.append(load_manifest(manifest_path))
        except Exception as e:
            print(f"[manifest] Warning: could not load {manifest_path}: {e}")
    return manifests
