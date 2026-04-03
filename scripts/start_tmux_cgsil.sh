#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/juan/dev/CGSIL"
pkill -f "$ROOT/train_cifar_lt.py" || true
if tmux has-session -t cgsil 2>/dev/null; then
  tmux kill-session -t cgsil
fi
chmod +x "$ROOT/scripts/gpu0_tmux_queue.sh" "$ROOT/scripts/gpu1_tmux_queue.sh"
tmux new-session -d -s cgsil -n gpu0 "bash $ROOT/scripts/gpu0_tmux_queue.sh"
tmux new-window -t cgsil -n gpu1 "bash $ROOT/scripts/gpu1_tmux_queue.sh"
tmux ls
