#!/usr/bin/env bash
# RIDE-VLA Locus Ablation Eval: Standard + Plus for visual / all_hidden / output
#
# output_consistency (= locus output, no EMA teacher) is LIBERO_output_consistency.
# locus_output (= locus output WITH EMA teacher) is LIBERO_locus_output.
# Full (action_query locus) result is already in LIBERO_full.
#
# Usage: bash eval_locus_all.sh [standard|plus|both]
set -euo pipefail

MODE=${1:-both}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNS=(
    LIBERO_locus_visual
    LIBERO_locus_all_hidden
    LIBERO_locus_output
)

for run in "${RUNS[@]}"; do
    ckpt="/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/${run}/final_model/pytorch_model.pt"
    echo "========================================"
    echo " ${run}"
    echo "========================================"

    if [[ "$MODE" == "standard" || "$MODE" == "both" ]]; then
        echo "[standard] Launching..."
        bash "${SCRIPT_DIR}/eval_libero_standard.sh" "${ckpt}"
    fi

    if [[ "$MODE" == "plus" || "$MODE" == "both" ]]; then
        echo "[plus] Launching..."
        bash "${SCRIPT_DIR}/eval_libero_plus.sh" "${ckpt}"
    fi
done

echo ""
echo "All locus evals submitted."
