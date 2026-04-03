

#!/bin/bash
# BayesianCAG Training Script for LIBERO
# Training is identical to CIGVLA (dual-branch with LLR loss).
# The CAG guidance is applied at inference time only.

# # export NCCL_SOCKET_IFNAME=bond0
# # export NCCL_IB_HCA=mlx5_2,mlx5_3
# export NCCL_SOCKET_IFNAME=ens20f3  # 使用你有实际 IP 的物理网卡，如果不行就换成 lo
# # export NCCL_IB_HCA=mlx5_2,mlx5_3 # 注释掉这一行，避免指定不存在的 InfiniBand 网卡引发错误
# export NCCL_IB_DISABLE=1           # 加上这行，强制禁用 IB，走常规网络
# export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1      # 使用主网卡

export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1      # 新增：禁用共享内存，强制走正常的 PCIe/NVLink
# export NCCL_P2P_DISABLE=1    # 务必注释掉或删除这一行！
export CUDA_LAUNCH_BLOCKING=1  # 继续保留用于捕获错误
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=CIGVLA
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action

config_yaml=./examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
oxe_data_root=playground/Datasets/OXE_LEROBOT
data_mix=bridge_rt_1
run_root_dir=./results/Checkpoints
run_id=${data_mix}_CIGVLA_SimplerEnv
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/



accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  --main_process_port 29502 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_simplerEnv \
  --wandb_entity haodong_chen-nanjing-university-of-aeronautics-and-astro \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
