#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/home/guest/.conda/envs/starVLA/bin/python

# your_ckpt=results/LangForce/Checkpoints/tag_flow_libero_all/checkpoints/steps_50000_pytorch_model.pt
your_ckpt=results/LangForce/Checkpoints/CIGVLA/checkpoints/steps_50000_pytorch_model.pt



gpu_id=4
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
