#!/bin/bash
# ConsistencyVLA Training Script for LIBERO
# Self-distillation VLA with EMA teacher and early-exit inference.

export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=ConsistencyVLA
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action

config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero_consistency.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/ConsistencyVLA/Checkpoints
run_id=consistency_vla_libero_all
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  --main_process_port 29501 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_ConsistencyVLA_nanda \
  --wandb_entity haodong_chen-nanjing-university-of-aeronautics-and-astro \
  # --is_debug True
