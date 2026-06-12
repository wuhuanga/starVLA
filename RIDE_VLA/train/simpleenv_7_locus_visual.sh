#!/bin/bash
# RIDE-VLA | SimplerEnv | Locus Ablation: Visual Tokens
# Distillation target: mean-pooled visual-token hidden states [B, H].
# Full last-layer hidden states are kept for both teacher and student → batch 8 + grad_accum 2.

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

config_yaml=./examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
data_root=playground/Datasets/SIMPLENV
data_mix=bridge_rt_1
run_root_dir=/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/
run_id=SIMPLEENV_locus_visual
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
  --framework.distill_locus visual \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.paraphrase_bank "processed_instructions.json" \
  --framework.p_visual_aug 1.0 \
  --framework.p_paraphrase 1.0 \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.gradient_accumulation_steps 2 \
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
