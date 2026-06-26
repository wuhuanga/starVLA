#!/usr/bin/env bash
# RIDE-VLA Eval: Standard LIBERO (Spatial / Object / Goal / Long)
# Usage: bash eval_libero_standard.sh <ckpt_path>
#
# 4 suites run in parallel, each on a separate GPU.
# Resume-safe: suites whose log already contains "Total success rate:" are skipped.
set -euo pipefail

###########################################################################################
export LIBERO_HOME=/nfs/ofs-llm-ssd/user/shengrenren_i/research/LIBERO
export LIBERO_CONFIG_PATH=/home/luban/.libero
###########################################################################################

mkdir -p $LIBERO_CONFIG_PATH
cat > $LIBERO_CONFIG_PATH/config.yaml <<EOF
benchmark_root: ${LIBERO_HOME}/libero/libero
bddl_files: ${LIBERO_HOME}/libero/libero/bddl_files
init_states: ${LIBERO_HOME}/libero/libero/init_files
datasets: /nfs/ofs-llm-ssd/user/shengrenren_i/research/chd/playground/Datasets/LEROBOT_LIBERO_DATA
assets: ${LIBERO_HOME}/libero/libero/assets
EOF

export PYTHONPATH="$(pwd):${LIBERO_HOME}:${PYTHONPATH:-}"

your_ckpt=$1
host="127.0.0.1"
BASE_PORT=8095
num_trials_per_task=50

task_suite_list=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
)

model_root=$(dirname "$(dirname "$your_ckpt")")
folder_name=$(basename "$your_ckpt" .pt)

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

PYTHON=/nfs/ofs-llm-ssd/user/shengrenren_i/envs/statvla/bin/python

# ── helpers ───────────────────────────────────────────────────────────────────
is_done() {
    local log_file=$1
    [ -f "${log_file}" ] && grep -q "Total success rate:" "${log_file}"
}

get_rate() {
    local log_file=$1
    grep -oP '(?<=Total success rate: )[0-9.]+' "${log_file}" | tail -1
}

wait_for_port() {
    local h=$1 p=$2 pid=$3 logf=$4 elapsed=0
    while true; do
        if grep -q "server listening on" "${logf}" 2>/dev/null; then
            echo "[GPU$((p-BASE_PORT))] Server ready (${elapsed}s)."
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[GPU$((p-BASE_PORT))] Server exited unexpectedly. Check: ${logf}"
            tail -20 "${logf}" || true; return 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
        (( elapsed % 30 == 0 )) && echo "[GPU$((p-BASE_PORT))] Still waiting... ${elapsed}s"
        (( elapsed >= 1200 )) && { echo "[GPU$((p-BASE_PORT))] Timeout (1200s)."; return 1; }
    done
}

# ── pre-check ─────────────────────────────────────────────────────────────────
echo "Checking completed suites..."
pending=()
for suite in "${task_suite_list[@]}"; do
    log_file="${model_root}/logs/standard/${suite}/${folder_name}.log"
    if is_done "${log_file}"; then
        rate=$(get_rate "${log_file}")
        echo "  [DONE]    ${suite}: ${rate}"
    else
        echo "  [PENDING] ${suite}"
        pending+=("${suite}")
    fi
done

if [ ${#pending[@]} -eq 0 ]; then
    echo ""
    echo "All suites already completed. Printing summary:"
    echo "────────────────────────────────────────────────"
    printf "%-20s %s\n" "Suite" "Success Rate"
    printf "%-20s %s\n" "─────────────────────" "────────────"
    total=0; count=0
    for suite in "${task_suite_list[@]}"; do
        log_file="${model_root}/logs/standard/${suite}/${folder_name}.log"
        rate=$(get_rate "${log_file}")
        printf "%-20s %s\n" "${suite}" "${rate}"
        total=$(python3 -c "print(${total} + ${rate})")
        count=$((count + 1))
    done
    avg=$(python3 -c "print(f'{${total}/${count}:.3f}')")
    printf "%-20s %s\n" "─────────────────────" "────────────"
    printf "%-20s %s\n" "Avg" "${avg}"
    exit 0
fi

echo ""
echo "Suites to run: ${pending[*]}"
echo ""

# ── start servers in parallel ─────────────────────────────────────────────────
server_pids=()
bg_pids=()

cleanup() {
    echo "Cleaning up servers..."
    for pid in "${server_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT

mkdir -p "${model_root}/logs/standard" "${model_root}/logs/servers"

for i in "${!task_suite_list[@]}"; do
    suite="${task_suite_list[$i]}"
    log_file="${model_root}/logs/standard/${suite}/${folder_name}.log"

    if is_done "${log_file}"; then
        server_pids+=("")   # placeholder to keep index alignment
        continue
    fi

    gpu_id=$i
    port=$((BASE_PORT + i))
    server_log="${model_root}/logs/servers/${suite}_server.log"

    echo "[GPU${gpu_id}] Starting server on port ${port} for ${suite}..."
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    $PYTHON deployment/model_server/server_policy.py \
        --ckpt_path "${your_ckpt}" \
        --port "${port}" \
        --use_bf16 > "${server_log}" 2>&1 &
    server_pids+=($!)
    echo "[GPU${gpu_id}] Server pid=${server_pids[-1]}"
done

# ── wait for each server, then launch eval ────────────────────────────────────
for i in "${!task_suite_list[@]}"; do
    suite="${task_suite_list[$i]}"
    log_file="${model_root}/logs/standard/${suite}/${folder_name}.log"

    if is_done "${log_file}"; then
        continue
    fi

    gpu_id=$i
    port=$((BASE_PORT + i))
    server_log="${model_root}/logs/servers/${suite}_server.log"
    log_path="${model_root}/logs/standard/${suite}"

    mkdir -p "${log_path}"

    wait_for_port "${host}" "${port}" "${server_pids[$i]}" "${server_log}"
    echo "[GPU${gpu_id}] Port ${port} ready, launching eval for ${suite}..."

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    $PYTHON ./examples/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path "${your_ckpt}" \
        --args.host "${host}" \
        --args.port "${port}" \
        --args.task-suite-name "${suite}" \
        --args.num-trials-per-task "${num_trials_per_task}" \
        --args.no-video \
        > "${log_file}" 2>&1 &
    bg_pids+=($!)
    echo "[GPU${gpu_id}] Eval pid=${bg_pids[-1]} for ${suite}"
done

# ── wait for all ──────────────────────────────────────────────────────────────
echo ""
echo "All evals launched in parallel. Waiting for completion..."
for pid in "${bg_pids[@]:-}"; do
    wait "${pid}" || echo "Eval pid=${pid} exited with non-zero status"
done
echo "All evals finished."

# ── final summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Standard LIBERO Results: ${folder_name}"
echo "════════════════════════════════════════"
printf "%-20s %s\n" "Suite" "Success Rate"
printf "%-20s %s\n" "─────────────────────" "────────────"
total=0; count=0
for suite in "${task_suite_list[@]}"; do
    log_file="${model_root}/logs/standard/${suite}/${folder_name}.log"
    if is_done "${log_file}"; then
        rate=$(get_rate "${log_file}")
        printf "%-20s %s\n" "${suite}" "${rate}"
        total=$(python3 -c "print(${total} + ${rate})")
        count=$((count + 1))
    else
        printf "%-20s %s\n" "${suite}" "INCOMPLETE"
    fi
done
if [ "${count}" -gt 0 ]; then
    avg=$(python3 -c "print(f'{${total}/${count}:.3f}')")
    printf "%-20s %s\n" "─────────────────────" "────────────"
    printf "%-20s %s\n" "Avg (${count}/4)" "${avg}"
fi
