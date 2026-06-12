#!/bin/bash
# RIDE-VLA | LIBERO | Locus Ablation: Output (Velocity)
# Distillation target: action-head velocity MSE with shared (s, eps) noise.
# EMA teacher on clean view; student on perturbed view — same structure as full method,
# but aligns the head *output* rather than the action-query *representation*.
# Only action-query tokens kept (no full hidden states) → batch 16 like full method.

export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MALLOC_TRIM_THRESHOLD_=131072

###########################################################################################
Framework_name=IntentVLA
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action

config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/
run_id=LIBERO_locus_output
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  --main_process_port 29501 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.train_variant ridevla \
  --framework.distill_locus output \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.paraphrase_bank "processed_instructions.json" \
  --framework.p_visual_aug 1.0 \
  --framework.p_paraphrase 1.0 \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.num_workers 1 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 10000 \
  --trainer.is_resume true \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project RIDE_VLA \
  --wandb_entity ascka
