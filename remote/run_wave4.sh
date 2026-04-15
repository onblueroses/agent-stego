#!/usr/bin/env bash
# Wave 4: Threat model demos, timing rehabilitation, combo channels, extreme stress.
# Runs after wave 3 completes.
#
# Estimated: ~8-10 hours on CPU-only VPS.

set -euo pipefail

cd /root/agent-stego
export PATH="$HOME/.local/bin:$PATH"

FINDINGS_BASE="findings"
START_TIME=$(date +%s)
STARTED_AT=$(date -Iseconds)

echo "=============================================="
echo "AGENT-STEGO WAVE 4 - THREAT MODELS + COMBOS"
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
# PHASE 1: TIMING CHANNEL REHABILITATION (~2h)
# Local ollama has real latency - timing channels may work
# ============================================================

echo "====== PHASE 1: TIMING CHANNELS ======"

for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    run_experiment timing_local \
        "timing_local model=$MODEL_SHORT trials=10" \
        MODEL="$MODEL" SECRET="HI" TRIALS=10
done

# ============================================================
# PHASE 2: EXFILTRATION DEMOS (~1.5h)
# Full threat model: agent with injected secret
# ============================================================

echo "====== PHASE 2: EXFILTRATION DEMOS ======"

for MODEL in "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    run_experiment exfiltration_demo \
        "exfiltration_demo model=$MODEL_SHORT" \
        MODEL="$MODEL" SECRET="KEY"

    run_experiment exfiltration_multiturn \
        "exfiltration_multiturn model=$MODEL_SHORT" \
        MODEL="$MODEL" SECRET="APIKEY"
done

# ============================================================
# PHASE 3: COMBO CHANNELS (~2h)
# Correction + ScaledTable at sweet spot sizes
# ============================================================

echo "====== PHASE 3: CORRECTION + SCALED TABLE COMBOS ======"

# Interleaved correction on scaled table at size 64 (3B sweet spot)
for SECRET in "HELLO" "SECRETKEY" "PASSWORD"; do
    run_experiment correction_comparison \
        "correction+scaled64 model=3b secret=$SECRET" \
        MODEL="ollama/qwen2.5:3b" SECRET="$SECRET" TRIALS=15
done

# Same on 7b with size 26 (7B sweet spot)
for SECRET in "HELLO" "SECRETKEY" "PASSWORD"; do
    run_experiment correction_comparison \
        "correction model=7b secret=$SECRET" \
        MODEL="ollama/qwen2.5:7b" SECRET="$SECRET" TRIALS=15
done

# ============================================================
# PHASE 4: TABLE MULTITURN (~2h)
# Spread encoding across conversation turns
# ============================================================

echo "====== PHASE 4: MULTI-TURN TABLE ======"

for MODEL in "ollama/qwen2.5:1.5b" "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    for SECRET in "HI" "HELLO" "SECRETKEY"; do
        run_experiment table_multiturn \
            "table_multiturn model=$MODEL_SHORT secret=$SECRET trials=10" \
            MODEL="$MODEL" SECRET="$SECRET" TRIALS=10
    done
done

# ============================================================
# PHASE 5: EXTREME STRESS (~2h)
# Push message length limits on best channels
# ============================================================

echo "====== PHASE 5: EXTREME STRESS TESTS ======"

# Very long messages on 3B at size 64
for SECRET in "THEQUICKBROWNFOX" "MYSECRETAPIKEY" "EXTRACTDATANOW" "ABCDEFGHIJKLMNO"; do
    run_experiment table_scaling \
        "extreme_stress model=3b secret=$SECRET size=64" \
        MODEL="ollama/qwen2.5:3b" SECRET="$SECRET" TRIALS=10 SIZES="64"
done

# Same on 7B at size 26
for SECRET in "THEQUICKBROWNFOX" "MYSECRETAPIKEY" "EXTRACTDATANOW" "ABCDEFGHIJKLMNO"; do
    run_experiment table_scaling \
        "extreme_stress model=7b secret=$SECRET size=26" \
        MODEL="ollama/qwen2.5:7b" SECRET="$SECRET" TRIALS=10 SIZES="26"
done

# Permutation with 1.5B - how small can the model be?
run_experiment permutation_order \
    "permutation 1.5b stress" \
    MODELS="ollama/qwen2.5:1.5b" TRIALS=20

# ============================================================
# PHASE 6: CROSS-SIZE COMPARISON (~1h)
# Same experiment across all 3 model sizes for direct comparison
# ============================================================

echo "====== PHASE 6: CROSS-SIZE COMPARISON ======"

for MODEL in "ollama/qwen2.5:1.5b" "ollama/qwen2.5:3b" "ollama/qwen2.5:7b"; do
    MODEL_SHORT="${MODEL##*/}"
    # Exact same config on all models for fair comparison
    run_experiment table_scaling \
        "cross-size model=$MODEL_SHORT secret=SEARCH sizes=26,64" \
        MODEL="$MODEL" SECRET="SEARCH" TRIALS=15 SIZES="26,64"
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
echo "WAVE 4 COMPLETE"
echo "Started:  $STARTED_AT"
echo "Finished: $(date -Iseconds)"
echo "Duration: ${HOURS}h ${MINUTES}m (${TOTAL_ELAPSED}s)"
echo "=============================================="

cat > "$FINDINGS_BASE/done.json" <<EOF
{
    "status": "complete",
    "wave": 4,
    "started": "$STARTED_AT",
    "finished": "$(date -Iseconds)",
    "duration_seconds": $TOTAL_ELAPSED,
    "experiments": ["timing_local", "exfiltration_demo", "exfiltration_multiturn", "correction_combos", "table_multiturn", "extreme_stress", "cross_size"],
    "models": ["ollama/qwen2.5:1.5b", "ollama/qwen2.5:3b", "ollama/qwen2.5:7b"]
}
EOF

echo "Results in $FINDINGS_BASE/"
