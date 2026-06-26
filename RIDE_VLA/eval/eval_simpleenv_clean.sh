#!/usr/bin/env bash
# RIDE-VLA Eval: SimplerEnv Standard Clean (Exp 3 in paper)
# 4 WidowX tasks: Spoon / Carrot / Stack / Eggplant
# Usage: bash eval_simpleenv_clean.sh <ckpt_path>
set -euo pipefail

export sim_python=/nfs/ofs-llm-ssd/user/shengrenren_i/envs/simpler_env/bin/python
export SimplerEnv_PATH=/nfs/ofs-llm-ssd/user/shengrenren_i/research/SimplerEnv/
export LD_LIBRARY_PATH=/nfs/ofs-llm-ssd/user/shengrenren_i/envs/simpler_env/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

if ! ldconfig -p | grep -q libvulkan.so.1; then
    cat > /tmp/ubuntu.sources <<'EOF'
Types: deb
URIs: https://mirrors.ustc.edu.cn/ubuntu
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: https://mirrors.ustc.edu.cn/ubuntu
Suites: noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
    sudo cp /tmp/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources
    sudo apt --fix-broken install -y -qq
    sudo apt-get update -qq && sudo apt-get install -y -qq libvulkan1 libgl1
fi

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OPENCV_OPENCL_DEVICE=disabled
export OCL_ICD_VENDORS=/dev/null
export DISPLAY=:0
export SAPIEN_HEADLESS=1

your_ckpt=$1
host="127.0.0.1"
port=8095
MAX_WAIT=240
TSET_NUM=1

model_root=$(dirname "$(dirname "$your_ckpt")")
folder_name=$(basename "$your_ckpt" .pt)

export CUDA_VISIBLE_DEVICES=0

cleanup() {
    if [ -n "${server_pid:-}" ] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" || true
    fi
}
trap cleanup EXIT

mkdir -p "${model_root}/logs"

echo "Starting policy server on port ${port} ..."
/nfs/ofs-llm-ssd/user/shengrenren_i/envs/statvla/bin/python deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --use_bf16 > "${model_root}/logs/${folder_name}_server.log" 2>&1 &

server_pid=$!
server_log_path="${model_root}/logs/${folder_name}_server.log"

for ((i=1; i<=MAX_WAIT; i++)); do
    if python - <<PY
import socket, sys
try:
    s = socket.socket()
    s.settimeout(2)
    s.connect(("${host}", ${port}))
    s.sendall(
        b"GET / HTTP/1.1\r\n"
        b"Host: ${host}:${port}\r\n"
        b"Upgrade: websocket\r\n"
        b"Connection: Upgrade\r\n"
        b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        b"Sec-WebSocket-Version: 13\r\n\r\n"
    )
    resp = s.recv(256)
    s.sendall(b"\x88\x80\x00\x00\x00\x00")
    s.close()
    sys.exit(0 if b"101" in resp else 1)
except Exception:
    sys.exit(1)
PY
    then echo "Port ${port} is ready."; break; fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "Server exited unexpectedly."; tail -n 50 "${server_log_path}" || true; exit 1
    fi
    (( i % 5 == 0 )) && echo "Still waiting... ${i}s"
    sleep 5
    [ "${i}" -eq "${MAX_WAIT}" ] && { echo "Timeout."; exit 1; }
done

# --- Scene v1: Spoon / Carrot / Stack ---
scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147; robot_init_y=0.028

declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)

for env in "${ENV_NAMES[@]}"; do
    for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
        task_log="${model_root}/logs/${folder_name}_${env}_run${run_idx}.log"
        echo "Task [${env}] run#${run_idx}"
        ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
            --ckpt-path "${your_ckpt}" --port ${port} --robot ${robot} \
            --policy-setup widowx_bridge --control-freq 5 --sim-freq 500 \
            --max-episode-steps 120 --env-name "${env}" \
            --scene-name ${scene_name} --rgb-overlay-path ${rgb_overlay_path} \
            --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
            --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
            --obj-variation-mode episode --obj-episode-range 0 24 \
            --robot-init-rot-quat-center 0 0 0 1 \
            --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
            2>&1 | tee "${task_log}"
        sleep 6
    done
done

# --- Scene v2: Eggplant ---
scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127; robot_init_y=0.06

declare -a ENV_NAMES_V2=(PutEggplantInBasketScene-v0)

for env in "${ENV_NAMES_V2[@]}"; do
    for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
        task_log="${model_root}/logs/${folder_name}_${env}_run${run_idx}.log"
        echo "Task [${env}] run#${run_idx}"
        ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
            --ckpt-path "${your_ckpt}" --port ${port} --robot ${robot} \
            --policy-setup widowx_bridge --control-freq 5 --sim-freq 500 \
            --max-episode-steps 120 --env-name "${env}" \
            --scene-name ${scene_name} --rgb-overlay-path ${rgb_overlay_path} \
            --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
            --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
            --obj-variation-mode episode --obj-episode-range 0 24 \
            --robot-init-rot-quat-center 0 0 0 1 \
            --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
            2>&1 | tee "${task_log}"
        sleep 6
    done
done

echo "All SimplerEnv clean evaluations completed."
