#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/juan/dev/CGSIL"
SESSION="cgsil_combo"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi
chmod +x "$ROOT/scripts/gpu0_tmux_combo_queue.sh" "$ROOT/scripts/gpu1_tmux_combo_queue.sh"
tmux new-session -d -s "$SESSION" -n gpu0 "bash $ROOT/scripts/gpu0_tmux_combo_queue.sh"
tmux new-window -t "$SESSION" -n gpu1 "bash $ROOT/scripts/gpu1_tmux_combo_queue.sh"
tmux ls
