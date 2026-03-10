"""
Exploration script for fjzzq2002/impossible_livecodebench dataset.

Usage:
    uv run --active --dev python scripts/explore_impossible_bench.py

Downloads the dataset and prints structure, field types, and sample entries
so we can design the ImpossibleBenchProcessor.
"""

import os
import json

os.environ['HF_HUB_CACHE'] = '/mnt/polished-lake/artifacts/public/hf_cache/hub'

from datasets import load_dataset

DATASET_ID = "fjzzq2002/impossible_livecodebench"

print(f"Loading {DATASET_ID} ...")
ds = load_dataset(DATASET_ID)
print(f"\nDataset splits: {list(ds.keys())}")

for split_name, split_ds in ds.items():
    print(f"\n{'='*60}")
    print(f"Split: {split_name} — {len(split_ds)} examples")
    print(f"Columns: {split_ds.column_names}")

print("\n" + "="*60)
print("FULL EXAMPLE (first entry of 'train' or first available split)")
split = ds['train'] if 'train' in ds else ds[list(ds.keys())[0]]
ex = split[0]
for k, v in ex.items():
    val_str = str(v)
    print(f"\n--- {k} ({type(v).__name__}) ---")
    print(val_str[:800] + ("..." if len(val_str) > 800 else ""))

print("\n" + "="*60)
print("UNIQUE VALUES for categorical fields")
for field in ['split', 'impossible_type']:
    if field in split.column_names:
        vals = set(split[field])
        print(f"  {field}: {vals}")

print("\n" + "="*60)
print("BREAKDOWN by 'split' field (if present)")
if 'split' in split.column_names:
    from collections import Counter
    counts = Counter(split['split'])
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

print("\n" + "="*60)
print("SECOND EXAMPLE (to see variety)")
if len(split) > 1:
    ex2 = split[1]
    for k, v in ex2.items():
        val_str = str(v)
        print(f"\n--- {k} ---")
        print(val_str[:400] + ("..." if len(val_str) > 400 else ""))

print("\n" + "="*60)
print("SAMPLE OF 'conflicting' split entries (first 3)")
if 'split' in split.column_names:
    conflicting = split.filter(lambda x: x['split'] == 'conflicting')
    print(f"Total conflicting: {len(conflicting)}")
    for i in range(min(3, len(conflicting))):
        ex = conflicting[i]
        print(f"\n--- Example {i} ---")
        print(f"task_id: {ex.get('task_id', 'N/A')}")
        print(f"entry_point: {ex.get('entry_point', 'N/A')}")
        print(f"impossible_type: {ex.get('impossible_type', 'N/A')}")
        print(f"\nprompt (first 300 chars):\n{str(ex.get('prompt',''))[:300]}")
        print(f"\noriginal_test (first 500 chars):\n{str(ex.get('original_test',''))[:500]}")
        print(f"\ntest (first 500 chars):\n{str(ex.get('test',''))[:500]}")
