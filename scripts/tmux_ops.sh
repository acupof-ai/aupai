#!/usr/bin/env bash
# Small tmux helpers for driving the awb board.
# Usage:
#   bash scripts/tmux_ops.sh panes [session]
#   bash scripts/tmux_ops.sh capture <w.p> [session]
#   bash scripts/tmux_ops.sh enter <w.p> [session]
#   bash scripts/tmux_ops.sh down <w.p> [n] [session]
#   bash scripts/tmux_ops.sh text <w.p> <string> [session]
#   bash scripts/tmux_ops.sh choose <w.p> <n> [session]
#
# <w.p> is window.pane, for example 2.3.
set -euo pipefail

SESSION=${AWB_TMUX_SESSION:-aupai}

# Last argument may override the session.
args=("$@")
case "${1:-}" in
  panes)
    sess=${2:-$SESSION}
    tmux list-panes -t "$sess" -a \
      -F '#{window_index}.#{pane_index} title=#{pane_title} cmd=#{pane_current_command}'
    exit 0
    ;;
  capture)
    target=$2; sess=${3:-$SESSION}
    tmux capture-pane -t "$sess:$target" -p
    exit 0
    ;;
  enter)
    target=$2; sess=${3:-$SESSION}
    tmux send-keys -t "$sess:$target" Enter
    exit 0
    ;;
  down)
    target=$2; n=${3:-1}; sess=${4:-$SESSION}
    for _ in $(seq 1 "$n"); do
      tmux send-keys -t "$sess:$target" Down
    done
    exit 0
    ;;
  text)
    target=$2; text=$3; sess=${4:-$SESSION}
    tmux send-keys -t "$sess:$target" "$text"
    exit 0
    ;;
  choose)
    # Cursor starts on item 1. Move down (n-1) times, then confirm.
    target=$2; n=$3; sess=${4:-$SESSION}
    if [ "$n" -gt 1 ]; then
      for _ in $(seq 1 $((n - 1))); do
        tmux send-keys -t "$sess:$target" Down
      done
    fi
    tmux send-keys -t "$sess:$target" Enter
    exit 0
    ;;
  *)
    echo "unknown action: ${1:-}" >&2
    echo "choose one of: panes capture enter down text choose" >&2
    exit 2
    ;;
esac
