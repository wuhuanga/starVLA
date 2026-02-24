#!/bin/bash
# Policy server for BayesianCAG evaluation.
# The guidance_omega can be overridden via the --omega flag below.

export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python

your_ckpt=results/Checkpoints/bayesian_cag_libero_all/checkpoints/steps_100000_pytorch_model.pt
gpu_id=7
port=5694

# CAG guidance settings (override config defaults at server launch)
omega=2.0           # guidance scale: 1.0 / 1.5 / 2.0 / 3.0
guidance_mode=action # "action" or "velocity"

CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
