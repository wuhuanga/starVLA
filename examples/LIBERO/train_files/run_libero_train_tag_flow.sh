

#!/bin/bash
# BayesianCAG Training Script for LIBERO
# Training is identical to LangForce (dual-branch with LLR loss).
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
Framework_name=TAGFlow
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action

config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/LangForce/Checkpoints
run_id=tag_flow_libero_all
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
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_TAGFlow \
  --wandb_entity haodong_chen-nanjing-university-of-aeronautics-and-astro \
  # --is_debug True
