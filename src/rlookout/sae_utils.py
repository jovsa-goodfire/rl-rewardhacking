"""SAE loading and feature label utilities."""

import json

import torch
from goodfire_core.saes.batch_topk import BatchTopKSAE


def load_sae(checkpoint_path: str, device: str = "cpu") -> BatchTopKSAE:
    """Load BatchTopKSAE with strict=False for backward compat.

    The production SAE checkpoint predates the _fired_buffer addition,
    so strict loading fails. This loads manually with strict=False.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sae = BatchTopKSAE(**ckpt["model_config"])
    sae.load_state_dict(ckpt["state_dict"], strict=False)
    return sae.to(device).eval()


def load_feature_labels(labels_path: str) -> dict[int, str]:
    """Load JSONL labels file into {feature_id: label_string} dict."""
    labels_map: dict[int, str] = {}
    with open(labels_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("labels"):
                labels_map[entry["feature_id"]] = entry["labels"][0]["label"]
    return labels_map
