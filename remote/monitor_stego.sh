#!/usr/bin/env bash
# Monitor/sync for agent-stego experiments on VPS.
# Run via cron: */30 * * * * /path/to/monitor_stego.sh >> /path/to/stego_monitor.log 2>&1

set -euo pipefail

RESULTS_DIR="$HOME/Work/agent-stego/findings/vps"
VPS="root@100.121.215.2"
REMOTE_DIR="/root/agent-stego/findings"

mkdir -p "$RESULTS_DIR"

# Check if tmux session exists
if ! ssh "$VPS" "tmux has-session -t stego 2>/dev/null"; then
    echo "$(date -Iseconds) Session 'stego' not running."
    # Check if done
    if ssh "$VPS" "test -f $REMOTE_DIR/done.json" 2>/dev/null; then
        echo "$(date -Iseconds) Experiments COMPLETE. Syncing final results..."
        rsync -av "$VPS:$REMOTE_DIR/" "$RESULTS_DIR/" 2>/dev/null
        scp "$VPS:/root/agent-stego/run.log" "$RESULTS_DIR/run.log" 2>/dev/null || true
        echo "$(date -Iseconds) Results synced to $RESULTS_DIR"
        # Remove self from crontab
        crontab -l 2>/dev/null | grep -v "monitor_stego" | crontab - 2>/dev/null || true
    fi
    exit 0
fi

echo "$(date -Iseconds) Experiments running."

# Sync partial results
rsync -av "$VPS:$REMOTE_DIR/" "$RESULTS_DIR/" 2>/dev/null || true

# Show last log line
LAST=$(ssh "$VPS" "tail -1 /root/agent-stego/run.log 2>/dev/null" || echo "")
echo "  Last: $LAST"
