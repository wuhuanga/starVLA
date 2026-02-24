#!/bin/bash
# =============================================================================
# BayesianCAG Ablation Evaluation
# =============================================================================
# Evaluates the BayesianCAG model across different guidance scales (omega)
# and guidance modes to find the optimal configuration.
#
# Baselines included:
#   omega=1.0  --> equivalent to standard BayesianVLA posterior-only inference
#   omega=1.5, 2.0, 3.0 --> CAG-guided inference with increasing strength
#
# Usage:
#   1. Start the policy server:  bash run_policy_server_bayesian_cag.sh
#   2. Run this script:          bash eval_bayesian_cag_ablation.sh
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
your_ckpt=results/Checkpoints/bayesian_cag_libero_all/checkpoints/steps_100000_pytorch_model.pt

# === Server connection ===
host="127.0.0.1"
base_port=5694

# === Evaluation settings ===
num_trials_per_task=50

# === Task suites to evaluate ===
# For LIBERO-CF style evaluation, use libero_spatial and libero_object
# which test visual shortcut / counterfactual grounding
task_suites=("libero_spatial" "libero_object" "libero_goal")

# === Guidance scale ablation values ===
omega_values=("1.0" "1.5" "2.0" "3.0")

# === Guidance mode ===
guidance_mode="action"  # or "velocity"
# === End of configuration ===
###########################################################################################

LOG_DIR="logs/bayesian_cag_ablation_$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

echo "=============================================="
echo "BayesianCAG Ablation Evaluation"
echo "Checkpoint: ${your_ckpt}"
echo "Omega values: ${omega_values[*]}"
echo "Task suites: ${task_suites[*]}"
echo "Guidance mode: ${guidance_mode}"
echo "Log directory: ${LOG_DIR}"
echo "=============================================="

for omega in "${omega_values[@]}"; do
    for task_suite in "${task_suites[@]}"; do
        echo ""
        echo ">>> Evaluating omega=${omega}, task_suite=${task_suite}, mode=${guidance_mode}"

        video_out_path="results/bayesian_cag_ablation/${task_suite}/omega_${omega}_${guidance_mode}"
        mkdir -p ${video_out_path}

        # Note: omega is passed via the server config.
        # To override omega per run, you need to restart the server with the desired omega
        # or pass it through the client payload. See BayesianCAG.predict_action() for kwargs support.
        ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
            --args.pretrained-path ${your_ckpt} \
            --args.host "$host" \
            --args.port $base_port \
            --args.task-suite-name "$task_suite" \
            --args.num-trials-per-task "$num_trials_per_task" \
            --args.video-out-path "$video_out_path" \
            2>&1 | tee "${LOG_DIR}/eval_omega${omega}_${task_suite}_${guidance_mode}.log"

        echo ">>> Done: omega=${omega}, task_suite=${task_suite}"
    done
done

echo ""
echo "=============================================="
echo "Ablation evaluation complete."
echo "Results saved to: results/bayesian_cag_ablation/"
echo "Logs saved to: ${LOG_DIR}"
echo "=============================================="
