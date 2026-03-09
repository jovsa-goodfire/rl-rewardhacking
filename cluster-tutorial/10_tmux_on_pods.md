# Trial 10: tmux on Dev Pods

When your SSH connection drops (laptop lid close, network blip, etc.),
any running process dies with it. `tmux` keeps your sessions alive
on the remote machine so you can reconnect later.

## Quick Start

SSH into your dev pod, then:

```bash
# Start a new tmux session
tmux new -s work

# You're now inside tmux. Run anything:
python train.py

# Detach (session keeps running): press Ctrl+b, then d

# Reconnect later:
tmux attach -t work
```

## Essential Keybindings

All tmux commands start with **Ctrl+b** (the "prefix"), then a key:

| Keys | Action |
|------|--------|
| `Ctrl+b  d` | Detach from session (it keeps running) |
| `Ctrl+b  c` | Create a new window (like a tab) |
| `Ctrl+b  n` | Next window |
| `Ctrl+b  p` | Previous window |
| `Ctrl+b  0-9` | Jump to window by number |
| `Ctrl+b  %` | Split pane vertically |
| `Ctrl+b  "` | Split pane horizontally |
| `Ctrl+b  arrow` | Move between panes |
| `Ctrl+b  z` | Toggle zoom on current pane |
| `Ctrl+b  [` | Enter scroll mode (q to exit) |

## Common Workflows

### Run training that survives disconnects
```bash
tmux new -s training
python train.py --epochs 1000
# Ctrl+b, d to detach
# Close laptop, go home, reconnect:
ssh gpu
tmux attach -t training
```

### Side-by-side: training + GPU monitoring
```bash
tmux new -s monitor
# Left pane: start training
python train.py

# Ctrl+b, % to split vertically
# Right pane: watch GPUs
watch -n 1 nvidia-smi

# Ctrl+b, arrow keys to switch between panes
```

### Multiple experiments in windows
```bash
tmux new -s experiments
# Window 0: experiment A
python train.py --lr 1e-3

# Ctrl+b, c for new window
# Window 1: experiment B
python train.py --lr 1e-4

# Ctrl+b, 0 / Ctrl+b, 1 to switch between them
```

## Session Management

```bash
tmux ls                    # list all sessions
tmux new -s <name>         # create named session
tmux attach -t <name>      # reattach to session
tmux kill-session -t <name> # kill a session
```

## Recommended: Enable Mouse Scrolling

By default, scrolling in tmux is awkward. Fix it by creating a config
file in your cluster home directory:

```bash
echo "set -g mouse on" > ~/.tmux.conf
```

This lets you scroll with your trackpad/mouse wheel, click to select
panes, and resize panes by dragging borders.

## Key Point

Always start a tmux session before running anything long on your dev pod.
If your SSH connection drops without tmux, your training run dies. With
tmux, you just reconnect and pick up where you left off.
