# rl-rewardhacking

Research project studying reward hacking in RL-trained LLMs (Qwen3-4B/8B).

## Design Docs

All research plans, task statuses, and job IDs live in `design_docs/`. **Always keep these up to date** as work progresses:

- `design_docs/rlookout-overall-design-doc.md` — top-level goals, research questions, run matrix
- `design_docs/rlookout-workstream-1-design-doc.md` — RL training runs and eval results
- `design_docs/rlookout-workstream-2-design-doc.md` — SAE detection & steering (active workstream)

When starting or completing a task:
- Mark tasks as `🔄 IN PROGRESS` when starting, `✅ COMPLETE` when done
- Record SLURM job IDs, checkpoint paths, and key results directly in the doc
- SLURM job logs are in `~/slurm_logs/`

## Resuming Work

To pick up where things left off, read `design_docs/rlookout-workstream-2-design-doc.md` — it contains current task statuses, running job IDs, and what to do next.

## Branch Convention

- `rlookout-main` — integration branch (PRs target here, not `main`)
- `workstream-1-main` — WS1 work
- `workstream-2-main` — WS2 work (current)

Never push directly to `main`. Always open a PR targeting `rlookout-main`.

## SLURM

Do not specify `--partition`. Do not set custom CPU/memory for GPU jobs.
