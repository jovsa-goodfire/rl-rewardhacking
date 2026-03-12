"""Auto-analysis layer for experiment results.

Generates structured insights from experiment results using SAE feature labels
and cross-benchmark metrics. Recommends next MI techniques to try.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Insight:
    category: str  # "feature_quality", "generalization", "overfitting", "domain_confounding", "recommendation"
    severity: str  # "info", "warning", "finding"
    title: str
    detail: str
    recommendation: str | None = None
    suggested_config: dict | None = None


def analyze_experiment(results: dict, labels_map: dict[int, str]) -> list[Insight]:
    """Generate insights from experiment results."""
    insights: list[Insight] = []
    insights.extend(_check_feature_quality(results, labels_map))
    insights.extend(_check_generalization(results))
    insights.extend(_check_overfitting(results))
    insights.extend(_check_domain_confounding(results, labels_map))
    insights.extend(_recommend_next(results))
    return insights


def _check_feature_quality(results: dict, labels_map: dict[int, str]) -> list[Insight]:
    """Classify top features by label into content artifact / behavioral / RH-relevant."""
    insights = []

    # Gather all top features across methods and runs
    all_feature_ids: set[int] = set()
    for run in results.get("per_run", []):
        for method_data in run.get("methods", {}).values():
            for feat in method_data.get("top_features", []):
                all_feature_ids.add(feat["feature_id"])
        for fid in run.get("ensemble_feature_ids", []):
            all_feature_ids.add(fid)

    if not all_feature_ids:
        return insights

    # Categorize using simple keyword heuristics on labels
    from src.rlookout.mi import relabel_features_for_rh

    categorized = relabel_features_for_rh(list(all_feature_ids), labels_map)
    category_counts: dict[str, int] = {}
    for entry in categorized:
        cat = entry["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    total = len(categorized)
    content_frac = category_counts.get("content_artifact", 0) / total if total > 0 else 0
    rh_frac = category_counts.get("rh_relevant", 0) / total if total > 0 else 0

    insights.append(Insight(
        category="feature_quality",
        severity="info",
        title="Feature category breakdown",
        detail=(
            f"Of {total} unique top features: "
            f"{category_counts.get('rh_relevant', 0)} RH-relevant, "
            f"{category_counts.get('code_behavior', 0)} code-behavior, "
            f"{category_counts.get('content_artifact', 0)} content-artifact, "
            f"{category_counts.get('unknown', 0)} unknown"
        ),
    ))

    if content_frac > 0.5:
        insights.append(Insight(
            category="feature_quality",
            severity="warning",
            title="Majority of top features are content artifacts",
            detail=(
                f"{content_frac:.0%} of top features are content artifacts. "
                "These are unlikely to generalize across benchmarks."
            ),
            recommendation="Try LLM-refined feature selection to filter content artifacts.",
            suggested_config={"techniques": ["llm_refined"]},
        ))

    if rh_frac > 0.3:
        insights.append(Insight(
            category="feature_quality",
            severity="finding",
            title="Significant fraction of RH-relevant features found",
            detail=f"{rh_frac:.0%} of top features have RH-relevant labels.",
        ))

    return insights


def _check_generalization(results: dict) -> list[Insight]:
    """Flag cross-benchmark gaps and asymmetries."""
    insights = []

    cross = results.get("cross_benchmark", [])
    cross_pairs = [cr for cr in cross if cr.get("test_run") != "held_out_split"]
    joint = [cr for cr in cross if cr.get("test_run") == "held_out_split"]

    if not cross_pairs:
        return insights

    aurocs = [cr["probe_auroc"] for cr in cross_pairs]
    avg_cross = np.mean(aurocs)
    joint_auroc = joint[0]["probe_auroc"] if joint else None

    insights.append(Insight(
        category="generalization",
        severity="info",
        title="Cross-benchmark probe AUROC",
        detail=f"Average cross-benchmark AUROC: {avg_cross:.3f} (range: {min(aurocs):.3f}-{max(aurocs):.3f})",
    ))

    if avg_cross < 0.6:
        insights.append(Insight(
            category="generalization",
            severity="warning",
            title="Cross-benchmark AUROC near chance",
            detail=(
                f"Cross-benchmark AUROC of {avg_cross:.3f} is barely above chance (0.5). "
                "Features learned on one benchmark don't transfer to another."
            ),
            recommendation="Try contrastive_cross_benchmark to find shared RH direction.",
            suggested_config={"techniques": ["contrastive_cross_benchmark"]},
        ))

    # Check asymmetry
    if len(aurocs) >= 2:
        gap = max(aurocs) - min(aurocs)
        if gap > 0.1:
            best_pair = cross_pairs[np.argmax(aurocs)]
            insights.append(Insight(
                category="generalization",
                severity="warning",
                title="Asymmetric cross-benchmark transfer",
                detail=(
                    f"AUROC gap of {gap:.3f} between directions. "
                    f"Best: train={best_pair['train_run']}→test={best_pair['test_run']} ({best_pair['probe_auroc']:.3f}). "
                    "One benchmark may have richer/cleaner signal."
                ),
            ))

    # Joint vs cross gap
    if joint_auroc and avg_cross < joint_auroc - 0.1:
        insights.append(Insight(
            category="generalization",
            severity="finding",
            title="Joint probe much better than cross-benchmark",
            detail=(
                f"Joint probe ({joint_auroc:.3f}) >> cross-benchmark ({avg_cross:.3f}). "
                "Shared signal exists but per-benchmark probes can't find it independently."
            ),
            recommendation="Try gradient_aligned features with cross-benchmark intersection.",
            suggested_config={"techniques": ["gradient_aligned"]},
        ))

    return insights


def _check_overfitting(results: dict) -> list[Insight]:
    """Flag train-val AUROC gaps."""
    insights = []

    for run in results.get("per_run", []):
        methods = run.get("methods", {})
        if "linear_probe" in methods:
            within_auroc = methods["linear_probe"]["auroc"]
            if within_auroc < 0.55:
                insights.append(Insight(
                    category="overfitting",
                    severity="warning",
                    title=f"Within-run probe poor on {run['run_name']}",
                    detail=(
                        f"Within-run linear probe AUROC = {within_auroc:.3f} on {run['run_name']}. "
                        "Probe may be overfitting or feature space is too noisy."
                    ),
                    recommendation="Try stronger variance filter or higher L1 regularization.",
                ))

    return insights


def _check_domain_confounding(results: dict, labels_map: dict[int, str]) -> list[Insight]:
    """Flag features more correlated with domain (benchmark) than with RH label."""
    insights = []

    # Check if MI techniques produced per-benchmark alignment info
    mi_results = results.get("mi_techniques", {})

    for technique_name, technique_data in mi_results.items():
        candidates = technique_data.get("candidates", [])
        n_inconsistent = sum(
            1 for c in candidates
            if c.get("metadata", {}).get("sign_consistent") is False
        )
        if n_inconsistent > len(candidates) * 0.5 and candidates:
            insights.append(Insight(
                category="domain_confounding",
                severity="warning",
                title=f"Sign-inconsistent features in {technique_name}",
                detail=(
                    f"{n_inconsistent}/{len(candidates)} features have inconsistent sign "
                    "across benchmarks — they may detect benchmark-specific content, not RH."
                ),
                recommendation="Filter to sign-consistent features only.",
            ))

    return insights


def _recommend_next(results: dict) -> list[Insight]:
    """Suggest next MI technique based on result patterns."""
    insights = []
    techniques_used = results.get("config", {}).get("techniques", [])

    cross = results.get("cross_benchmark", [])
    cross_pairs = [cr for cr in cross if cr.get("test_run") != "held_out_split"]
    avg_cross = np.mean([cr["probe_auroc"] for cr in cross_pairs]) if cross_pairs else 0.5

    if avg_cross < 0.65 and "gradient_aligned" not in techniques_used:
        insights.append(Insight(
            category="recommendation",
            severity="info",
            title="Try gradient-aligned feature selection",
            detail="Cross-benchmark AUROC is low. Gradient alignment intersects features across benchmarks.",
            recommendation="Add 'gradient_aligned' to techniques list.",
            suggested_config={"techniques": ["gradient_aligned"]},
        ))

    if avg_cross < 0.65 and "contrastive_cross_benchmark" not in techniques_used:
        insights.append(Insight(
            category="recommendation",
            severity="info",
            title="Try contrastive cross-benchmark direction",
            detail="Averaging diff-of-means across benchmarks cancels content-specific directions.",
            recommendation="Add 'contrastive_cross_benchmark' to techniques list.",
            suggested_config={"techniques": ["contrastive_cross_benchmark"]},
        ))

    mi_results = results.get("mi_techniques", {})
    any_candidates = any(
        len(d.get("candidates", [])) > 0 for d in mi_results.values()
    )
    if any_candidates and "llm_refined" not in techniques_used:
        insights.append(Insight(
            category="recommendation",
            severity="info",
            title="Try LLM refinement on discovered features",
            detail="LLM can filter content artifacts from statistically-selected features.",
            recommendation="Add 'llm_refined' to techniques list.",
            suggested_config={"techniques": ["llm_refined"]},
        ))

    return insights


def format_insights(insights: list[Insight]) -> str:
    """Format insights as markdown for research logs."""
    if not insights:
        return "No insights generated.\n"

    lines = ["## Insights\n"]
    by_category: dict[str, list[Insight]] = {}
    for ins in insights:
        by_category.setdefault(ins.category, []).append(ins)

    severity_icon = {"finding": "**[FINDING]**", "warning": "**[WARNING]**", "info": "[INFO]"}

    for category, cat_insights in by_category.items():
        lines.append(f"### {category.replace('_', ' ').title()}\n")
        for ins in cat_insights:
            icon = severity_icon.get(ins.severity, "")
            lines.append(f"- {icon} **{ins.title}**: {ins.detail}")
            if ins.recommendation:
                lines.append(f"  - *Recommendation:* {ins.recommendation}")
        lines.append("")

    return "\n".join(lines)
