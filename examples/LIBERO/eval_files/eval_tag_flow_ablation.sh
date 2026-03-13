#!/bin/bash
# =============================================================================
# TAG-Flow Ablation Evaluation
# =============================================================================
# Sweeps omega values across guidance modes:
#   scheduled – time-scheduled CFG (TAG-Flow default, recommended)
#   velocity  – uniform omega on each denoising step
#   action    – uniform omega on final actions
#
# omega=1.0 in velocity/action mode = posterior-only baseline (no guidance).
#
# For scheduled mode, omega sets omega_max; omega_min defaults to 1.0.
#
# Server loads the model once. Client passes omega + mode dynamically.
#
# Usage:
#   1. Start the policy server:  bash run_policy_server_tag_flow.sh
#   2. Run this script:          bash eval_tag_flow_ablation.sh
# =============================================================================

cd /mnt/petrelfs/yejinhui/Projects/starVLA
conda activate starVLA

###########################################################################################
# === Environment setup ===
export LIBERO_HOME=/mnt/petrelfs/share/yejinhui/Projects/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/lerobot/bin/python
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python

# === Model checkpoint ===
your_ckpt=results/Checkpoints/tag_flow_libero_all/checkpoints/steps_100000_pytorch_model.pt

# === Server connection ===
host="127.0.0.1"
base_port=5694

# === Evaluation settings ===
num_trials_per_task=50

# === Task suites to evaluate ===
task_suites=("libero_spatial" "libero_object" "libero_goal")

# === Guidance scale ablation values ===
omega_values=("1.0" "2.0" "3.0" "4.0")

# === Guidance modes to sweep ===
guidance_modes=("scheduled" "velocity" "action")
# === End of configuration ===
###########################################################################################

LOG_DIR="logs/tag_flow_ablation_$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

echo "=============================================="
echo "TAG-Flow Ablation Evaluation"
echo "Checkpoint: ${your_ckpt}"
echo "Omega values: ${omega_values[*]}"
echo "Task suites: ${task_suites[*]}"
echo "Guidance modes: ${guidance_modes[*]}"
echo "Log directory: ${LOG_DIR}"
echo "=============================================="

for guidance_mode in "${guidance_modes[@]}"; do
    for omega in "${omega_values[@]}"; do
        for task_suite in "${task_suites[@]}"; do
            echo ""
            echo ">>> Evaluating mode=${guidance_mode}, omega=${omega}, task_suite=${task_suite}"

            video_out_path="results/tag_flow_ablation/${task_suite}/omega_${omega}_${guidance_mode}"
            mkdir -p ${video_out_path}

            ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
                --args.pretrained-path ${your_ckpt} \
                --args.host "$host" \
                --args.port $base_port \
                --args.task-suite-name "$task_suite" \
                --args.num-trials-per-task "$num_trials_per_task" \
                --args.video-out-path "$video_out_path" \
                --args.omega "$omega" \
                --args.guidance-mode "$guidance_mode" \
                2>&1 | tee "${LOG_DIR}/eval_omega${omega}_${task_suite}_${guidance_mode}.log"

            echo ">>> Done: mode=${guidance_mode}, omega=${omega}, task_suite=${task_suite}"
        done
    done
done

echo ""
echo "=============================================="
echo "Ablation evaluation complete."
echo "Results saved to: results/tag_flow_ablation/"
echo "Logs saved to: ${LOG_DIR}"
echo "=============================================="
