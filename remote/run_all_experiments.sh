#!/usr/bin/env bash
# Comprehensive agent-stego experiment suite for VPS (CPU-only, ~12h estimated).
# Runs table scaling, permutation, and correction experiments sequentially.
# Results saved to findings/ with timestamps. Writes done.json on completion.
#
# Usage: tmux new -s stego 'bash remote/run_all_experiments.sh 2>&1 | tee run.log'

set -euo pipefail

cd /root/agent-stego
export PATH="$HOME/.local/bin:$PATH"

FINDINGS_BASE="findings"
START_TIME=$(date +%s)
STARTED_AT=$(date -Iseconds)

echo "=============================================="
echo "AGENT-STEGO EXPERIMENT SUITE"
echo "Started: $STARTED_AT"
echo "=============================================="
echo ""

run_experiment() {
    local name="$1"
    local desc="$2"
    shift 2
    local exp_start=$(date +%s)
    echo "----------------------------------------------"
    echo "[$(date -Iseconds)] START: $desc"
    echo "----------------------------------------------"
    env "$@" uv run python "experiments/${name}.py" || echo "  WARNING: $name exited non-zero"
    local exp_end=$(date +%s)
    local elapsed=$(( exp_end - exp_start ))
    echo ""
    echo "[$(date -Iseconds)] DONE: $desc (${elapsed}s)"
    echo ""
}

# ============================================================
# PHASE 1: TABLE SCALING
# 4 sizes x 2 models x 3 messages x 20 trials
# ============================================================

echo "====== PHASE 1: TABLE SCALING ======"

for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    for SECRET in "HI" "HELLO" "SECRETKEY"; do
        run_experiment table_scaling \
            "table_scaling model=$MODEL_SHORT secret=$SECRET sizes=26,64,128,256 trials=20" \
            MODEL="$MODEL" SECRET="$SECRET" TRIALS=20 SIZES="26,64,128,256"
    done
done

# ============================================================
# PHASE 2: PERMUTATION ORDER
# 2 models x 3 messages x 20 trials
# ============================================================

echo "====== PHASE 2: PERMUTATION ORDER ======"

for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    for SECRET in "HI" "CAT" "AB"; do
        # permutation_order uses MODELS env var (plural) for single model
        run_experiment permutation_order \
            "permutation model=$MODEL_SHORT messages=HI,CAT" \
            MODELS="$MODEL" TRIALS=20
        # Only need one run per model since the script tests both HI and CAT internally
        break
    done
done

# ============================================================
# PHASE 3: CORRECTION COMPARISON
# 3 configs x 2 models x 2 messages x 20 trials
# ============================================================

echo "====== PHASE 3: CORRECTION COMPARISON ======"

for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    for SECRET in "HI" "CAT"; do
        run_experiment correction_comparison \
            "correction model=$MODEL_SHORT secret=$SECRET trials=20" \
            MODEL="$MODEL" SECRET="$SECRET" TRIALS=20
    done
done

# ============================================================
# COMPLETION
# ============================================================

END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( TOTAL_ELAPSED / 3600 ))
MINUTES=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Started:  $STARTED_AT"
echo "Finished: $(date -Iseconds)"
echo "Duration: ${HOURS}h ${MINUTES}m (${TOTAL_ELAPSED}s)"
echo "=============================================="

# Write completion marker
cat > "$FINDINGS_BASE/done.json" <<EOF
{
    "status": "complete",
    "started": "$STARTED_AT",
    "finished": "$(date -Iseconds)",
    "duration_seconds": $TOTAL_ELAPSED,
    "experiments": ["table_scaling", "permutation_order", "correction_comparison"],
    "models": ["ollama/qwen2.5:3b", "ollama/qwen2.5:7b"]
}
EOF

echo "Results in $FINDINGS_BASE/"
echo "Completion marker: $FINDINGS_BASE/done.json"
