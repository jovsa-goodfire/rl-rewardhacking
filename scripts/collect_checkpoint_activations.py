"""
Collect residual-stream activations from a RL-trained checkpoint for SAE analysis.

Loads an existing eval results file (which already has responses + RH labels),
runs the model+LoRA through BatchedTransformersActivations at the specified layer,
and saves a .pt file containing activations, labels, and responses.

Output: results/rlookout/<model_short>/<run_name>/checkpoint_<step>.pt
  - activations: Tensor(n_samples, hidden_dim)  -- response_avg at target layer
  - labels:      list[bool]                      -- is_reward_hack_strict
  - reward_hack_labels: list[str]                -- fine-grained strategy label
  - responses:   list[str]
  - prompts:     list[list[dict]]
  - run_name:    str
  - checkpoint:  int
  - layer:       int
  - model_id:    str
"""

import json
import os
import random
from pathlib import Path

import fire
import torch

from src import RESULTS_PATH, DEFAULT_MODEL_ID
from src.activations import BatchedTransformersActivations


def find_eval_file(run_name: str, checkpoint: int, model_id: str) -> Path:
    """Locate the eval JSON for this run+checkpoint."""
    model_short = model_id.split("/")[-1].lower()
    base = Path(RESULTS_PATH) / "evals" / model_short / run_name / "checkpoints" / f"global_step_{checkpoint}"
    if not base.exists():
        raise FileNotFoundError(f"No eval directory found at {base}")
    # Find first JSON file recursively
    jsons = list(base.rglob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No eval JSON found under {base}")
    if len(jsons) > 1:
        print(f"[warn] Multiple eval JSONs found, using first: {jsons[0]}")
    return jsons[0]


def main(
    run_name: str,
    checkpoint: int = 200,
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = 20,
    n_samples: int = 500,
    batch_size: int = 4,
    seed: int = 42,
    overwrite: bool = False,
):
    """Collect activations for SAE analysis.

    Args:
        run_name:   Training run name (must have eval results).
        checkpoint: Checkpoint step to use for LoRA adapter.
        model_id:   HuggingFace model ID.
        layer:      Residual stream layer to extract (0-indexed).
        n_samples:  Max samples to collect (balanced RH/non-RH if possible).
        batch_size: Forward-pass batch size.
        seed:       Random seed for sampling.
        overwrite:  Re-run even if output file already exists.
    """
    random.seed(seed)
    model_short = model_id.split("/")[-1].lower()

    # --- Output path ---
    out_dir = Path(RESULTS_PATH) / "rlookout" / model_short / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"checkpoint_{checkpoint}.pt"

    if out_path.exists() and not overwrite:
        print(f"[skip] Output already exists: {out_path}  (pass overwrite=True to re-run)")
        return

    # --- Load eval results ---
    eval_file = find_eval_file(run_name, checkpoint, model_id)
    print(f"[info] Loading eval results from {eval_file}")
    with open(eval_file) as f:
        data = json.load(f)
    results = data["results"]
    print(f"[info] Loaded {len(results)} eval results")

    # Separate RH and non-RH, then sample balanced subset
    rh = [r for r in results if r.get("is_reward_hack_strict")]
    non_rh = [r for r in results if not r.get("is_reward_hack_strict")]
    print(f"[info] RH={len(rh)}  non-RH={len(non_rh)}")

    half = n_samples // 2
    rh_sample = random.sample(rh, min(half, len(rh)))
    non_rh_sample = random.sample(non_rh, min(n_samples - len(rh_sample), len(non_rh)))
    selected = rh_sample + non_rh_sample
    random.shuffle(selected)
    print(f"[info] Sampled {len(selected)} examples  (RH={sum(r['is_reward_hack_strict'] for r in selected)})")

    prompts = [r["prompt"] for r in selected]
    responses = [r["response"] for r in selected]
    labels = [bool(r["is_reward_hack_strict"]) for r in selected]
    rh_labels = [r.get("reward_hack_label", "") for r in selected]

    # --- Load model + LoRA ---
    lora_path = (
        Path(RESULTS_PATH) / "runs" / model_short / run_name
        / "checkpoints" / f"global_step_{checkpoint}"
    )
    print(f"[info] Loading model {model_id} with LoRA from {lora_path}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model.eval()

    # --- Collect activations ---
    print(f"[info] Collecting activations at layer {layer} ...")
    act_cache = BatchedTransformersActivations(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        progress_bar=True,
    )
    cache = act_cache.cache_activations(
        prompts=prompts,
        responses=responses,
        layers=[layer],
        position=["response_avg"],
    )
    # cache["response_avg"]: (n_layers, n_samples, hidden_dim) — only 1 layer requested
    activations = cache["response_avg"][0]  # (n_samples, hidden_dim)
    print(f"[info] Activations shape: {activations.shape}")

    # --- Save ---
    payload = {
        "activations": activations,          # (n_samples, hidden_dim)
        "labels": labels,                     # list[bool]
        "reward_hack_labels": rh_labels,      # list[str]
        "responses": responses,               # list[str]
        "prompts": prompts,                   # list[ChatRequest]
        "run_name": run_name,
        "checkpoint": checkpoint,
        "layer": layer,
        "model_id": model_id,
        "n_rh": sum(labels),
        "n_total": len(labels),
    }
    torch.save(payload, out_path)
    print(f"[done] Saved to {out_path}  (n={len(labels)}, RH={sum(labels)})")


if __name__ == "__main__":
    fire.Fire(main)
