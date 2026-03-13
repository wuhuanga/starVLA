#!/bin/bash
# Policy server for TAG-Flow evaluation.
# TAG-Flow uses time-scheduled guidance by default (no omega override needed).

export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python

your_ckpt=results/Checkpoints/tag_flow_libero_all/checkpoints/steps_100000_pytorch_model.pt
gpu_id=7
port=5694

CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
