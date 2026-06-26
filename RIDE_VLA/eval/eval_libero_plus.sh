#!/usr/bin/env bash
# RIDE-VLA Eval: LIBERO-plus
# Usage: bash eval_libero_plus.sh <ckpt_path>
#
# 4 suites run in parallel, each on a separate GPU.
# Results aggregated by perturbation category:
#   Camera | Robot | Language | Light | Background | Noise | Layout | Total
set -euo pipefail

###########################################################################################
export LIBERO_HOME=/nfs/ofs-llm-ssd/user/shengrenren_i/research/LIBERO
export LIBERO_PLUS_HOME=/nfs/ofs-llm-ssd/user/shengrenren_i/research/LIBERO-plus
export LIBERO_CONFIG_PATH=/home/luban/.libero
TASK_CLASSIFICATION=${LIBERO_PLUS_HOME}/libero/libero/benchmark/task_classification.json
###########################################################################################

mkdir -p $LIBERO_CONFIG_PATH
cat > $LIBERO_CONFIG_PATH/config.yaml <<EOF
benchmark_root: ${LIBERO_PLUS_HOME}/libero/libero
bddl_files: ${LIBERO_PLUS_HOME}/libero/libero/bddl_files
init_states: ${LIBERO_PLUS_HOME}/libero/libero/init_files
datasets: /nfs/ofs-llm-ssd/user/shengrenren_i/research/chd/playground/Datasets/LEROBOT_LIBERO_DATA
assets: ${LIBERO_HOME}/libero/libero/assets
EOF

export PYTHONPATH="$(pwd):${LIBERO_PLUS_HOME}:${PYTHONPATH:-}"

your_ckpt=$1
host="127.0.0.1"
BASE_PORT=8045
num_trials_per_task=1

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
    local results_json=$1
    [ -f "${results_json}" ]
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
for i in "${!task_suite_list[@]}"; do
    suite="${task_suite_list[$i]}"
    results_json="${model_root}/logs/plus/${suite}/${folder_name}_tasks.json"
    if is_done "${results_json}"; then
        echo "  [DONE]    ${suite}"
    else
        echo "  [PENDING] ${suite}"
        pending+=("${suite}")
    fi
done

if [ ${#pending[@]} -eq 0 ]; then
    echo "All suites done. Computing category scores..."
else
    echo ""
    echo "Suites to run: ${pending[*]}"
    echo ""
fi

# ── run pending suites in parallel ────────────────────────────────────────────
server_pids=()
bg_pids=()

cleanup() {
    echo "Cleaning up servers..."
    for pid in "${server_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT

mkdir -p "${model_root}/logs/plus" "${model_root}/logs/servers"

for i in "${!task_suite_list[@]}"; do
    suite="${task_suite_list[$i]}"
    results_json="${model_root}/logs/plus/${suite}/${folder_name}_tasks.json"

    if is_done "${results_json}"; then
        continue
    fi

    gpu_id=$i
    port=$((BASE_PORT + i))
    log_path="${model_root}/logs/plus/${suite}"
    server_log="${model_root}/logs/servers/${suite}_server.log"

    mkdir -p "${log_path}"

    echo "[GPU${gpu_id}] Starting server on port ${port} for ${suite}..."
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    $PYTHON deployment/model_server/server_policy.py \
        --ckpt_path "${your_ckpt}" \
        --port "${port}" \
        --use_bf16 > "${server_log}" 2>&1 &
    server_pids+=($!)
    echo "[GPU${gpu_id}] Server pid=${server_pids[-1]}, waiting for port ${port}..."
done

# wait for all servers to be ready, then launch evals
for i in "${!task_suite_list[@]}"; do
    suite="${task_suite_list[$i]}"
    results_json="${model_root}/logs/plus/${suite}/${folder_name}_tasks.json"

    if is_done "${results_json}"; then
        continue
    fi

    gpu_id=$i
    port=$((BASE_PORT + i))
    server_log="${model_root}/logs/servers/${suite}_server.log"
    log_path="${model_root}/logs/plus/${suite}"
    eval_log="${log_path}/${folder_name}.log"

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
        --args.results-out "${results_json}" \
        --args.video-out-path "${log_path}/videos" \
        > "${eval_log}" 2>&1 &
    bg_pids+=($!)
    echo "[GPU${gpu_id}] Eval pid=${bg_pids[-1]} for ${suite}"
done

# wait for all evals to finish
echo ""
echo "All evals launched in parallel. Waiting for completion..."
for pid in "${bg_pids[@]:-}"; do
    wait "${pid}" || echo "Eval pid=${pid} exited with non-zero status"
done
echo "All evals finished."

# ── per-category aggregation ──────────────────────────────────────────────────
echo ""
$PYTHON - <<PYEOF
import json, os, sys

classification_path = "${TASK_CLASSIFICATION}"
suite_list = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
results_dir = "${model_root}/logs/plus"
folder_name = "${folder_name}"

with open(classification_path) as f:
    classification = json.load(f)

# flatten: task_name -> category (across all suites)
task_to_category = {}
for suite, tasks in classification.items():
    for t in tasks:
        task_to_category[t["name"]] = t["category"]

CATEGORY_DISPLAY = {
    "Camera Viewpoints":    "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions":"Language",
    "Light Conditions":     "Light",
    "Background Textures":  "Background",
    "Sensor Noise":         "Noise",
    "Objects Layout":       "Layout",
}
ORDER = ["Camera Viewpoints", "Robot Initial States", "Language Instructions",
         "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout"]

cat_success = {c: 0 for c in ORDER}
cat_total   = {c: 0 for c in ORDER}

missing_suites = []
for suite in suite_list:
    fpath = os.path.join(results_dir, suite, f"{folder_name}_tasks.json")
    if not os.path.exists(fpath):
        missing_suites.append(suite)
        continue
    with open(fpath) as f:
        data = json.load(f)
    tasks = data.get("tasks", {})
    for task_name, rate in tasks.items():
        cat = task_to_category.get(task_name)
        if cat and cat in cat_success:
            cat_success[cat] += rate
            cat_total[cat]   += 1

if missing_suites:
    print(f"WARNING: missing results for: {missing_suites}")

print("")
print("=" * 70)
print(f"  LIBERO-plus Results: {folder_name}")
print("=" * 70)
headers = [CATEGORY_DISPLAY[c] for c in ORDER] + ["Total"]
row = []
total_s, total_n = 0, 0
for c in ORDER:
    if cat_total[c] > 0:
        v = cat_success[c] / cat_total[c] * 100
        row.append(f"{v:.1f}")
        total_s += cat_success[c]
        total_n += cat_total[c]
    else:
        row.append("N/A")

total_val = f"{total_s/total_n*100:.1f}" if total_n > 0 else "N/A"
row.append(total_val)

col_w = 12
header_line = "".join(h.ljust(col_w) for h in headers)
value_line  = "".join(v.ljust(col_w) for v in row)
print(header_line)
print("-" * (col_w * len(headers)))
print(value_line)
print("")
PYEOF
