#!/usr/bin/env bash
# 通用多切片空间转录组 benchmark：跑方法并计算指标，不画图。
# 必需：
#   DATA_DIR      每个切片一个 h5ad（{id}.h5ad 或 {id}_data.h5ad）
#   N_CLUSTERS    空间域个数
# 可选：
#   SECTION_IDS           空格分隔的切片 id，默认扫描 DATA_DIR
#   DATASET               结果文件前缀，默认 DATA_DIR 的目录名
#   DEVICE                默认 cuda:0
#   OURS_H5AD             自有方法结果，参与指标比较
#   OURS_EMBEDDING_KEY    默认 STIntg
#   OURS_CLUSTER_KEY      默认 louvain_0.7
# 空间半径（GraphST --radius、STAligner --rad-cutoff）默认 50，需与坐标单位匹配。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
DATA_DIR="${DATA_DIR:?请设置 DATA_DIR 为切片 h5ad 所在目录}"
N_CLUSTERS="${N_CLUSTERS:?请设置 N_CLUSTERS 为空间域个数}"
DATASET="${DATASET:-$(basename "${DATA_DIR%/}")}"
DEVICE="${DEVICE:-cuda:0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

ENV_GRAPHST="${ENV_GRAPHST:-stClinic}"
ENV_SEDR="${ENV_SEDR:-stClinic}"
ENV_BANKSY="${ENV_BANKSY:-stClinic}"
ENV_CELLCHARTER="${ENV_CELLCHARTER:-cellcharter}"
# torchgmm RangeBatchSampler only supports SequentialSampler; limit to one GPU (no DDP).
CELLCHARTER_CUDA_VISIBLE_DEVICES="${CELLCHARTER_CUDA_VISIBLE_DEVICES:-0}"
ENV_STALIGNER="${ENV_STALIGNER:-stClinic}"
ENV_METRICS="${ENV_METRICS:-stClinic}"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

require_env() {
  local env="$1"
  if ! conda_env_exists "${env}"; then
    echo "ERROR: conda env '${env}' not found." >&2
    echo "Available envs:" >&2
    conda env list >&2
    exit 1
  fi
}

run_py() {
  local env="$1"
  shift
  require_env "${env}"
  echo "[env=${env}] python $*"
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}" \
    conda run --no-capture-output -n "${env}" python "$@"
}

run_method() {
  local env="$1"
  local script="$2"
  local outfile="$3"
  shift 3
  echo "=== $(basename "${script}" .py) (env=${env}) ==="
  if [[ -f "${outfile}" ]]; then
    echo "Reuse ${outfile}"
    return 0
  fi
  run_py "${env}" "${SCRIPT_DIR}/${script}" "$@" --output "${outfile}"
}

mkdir -p "${RESULTS_DIR}"

COMMON_ARGS=(
  --data-dir "${DATA_DIR}"
  --dataset "${DATASET}"
)
if [[ -n "${SECTION_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  read -r -a _SECTIONS <<< "${SECTION_IDS}"
  COMMON_ARGS+=(--section-ids "${_SECTIONS[@]}")
fi

run_method "${ENV_GRAPHST}" run_graphst.py "${RESULTS_DIR}/${DATASET}_graphst.h5ad" \
  "${COMMON_ARGS[@]}" \
  --n-clusters "${N_CLUSTERS}" \
  --device "${DEVICE}"

run_method "${ENV_SEDR}" run_sedr.py "${RESULTS_DIR}/${DATASET}_sedr.h5ad" \
  "${COMMON_ARGS[@]}" \
  --n-clusters "${N_CLUSTERS}" \
  --device "${DEVICE}"

run_method "${ENV_BANKSY}" run_banksy.py "${RESULTS_DIR}/${DATASET}_banksy.h5ad" \
  "${COMMON_ARGS[@]}"

(
  export CUDA_VISIBLE_DEVICES="${CELLCHARTER_CUDA_VISIBLE_DEVICES}"
  run_method "${ENV_CELLCHARTER}" run_cellcharter.py "${RESULTS_DIR}/${DATASET}_cellcharter.h5ad" \
    "${COMMON_ARGS[@]}" \
    --n-clusters "${N_CLUSTERS}"
)

run_method "${ENV_STALIGNER}" run_staligner.py "${RESULTS_DIR}/${DATASET}_staligner.h5ad" \
  "${COMMON_ARGS[@]}" \
  --n-clusters "${N_CLUSTERS}" \
  --device "${DEVICE}"

echo "=== Metrics (env=${ENV_METRICS}) ==="
METRICS_ARGS=(
  --results-dir "${RESULTS_DIR}"
  --dataset "${DATASET}"
)
if [[ -n "${OURS_H5AD:-}" ]]; then
  METRICS_ARGS+=(
    --ours-h5ad "${OURS_H5AD}"
    --methods graphst sedr banksy cellcharter staligner ours
  )
fi
if [[ -n "${OURS_EMBEDDING_KEY:-}" ]]; then
  METRICS_ARGS+=(--ours-embedding-key "${OURS_EMBEDDING_KEY}")
fi
if [[ -n "${OURS_CLUSTER_KEY:-}" ]]; then
  METRICS_ARGS+=(--ours-cluster-key "${OURS_CLUSTER_KEY}")
fi
run_py "${ENV_METRICS}" "${SCRIPT_DIR}/evaluate_metrics.py" "${METRICS_ARGS[@]}"

echo "Done. Metrics: ${RESULTS_DIR}/${DATASET}_metrics_summary.csv"
