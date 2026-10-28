#!/usr/bin/env bash
# Quick sanity checks: conda envs, imports, and optional data loading (no full training).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
ST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_GRAPHST="${ENV_GRAPHST:-stClinic}"
ENV_SEDR="${ENV_SEDR:-stClinic}"
ENV_BANKSY="${ENV_BANKSY:-stClinic}"
ENV_CELLCHARTER="${ENV_CELLCHARTER:-cellcharter}"
ENV_STALIGNER="${ENV_STALIGNER:-stClinic}"
ENV_METRICS="${ENV_METRICS:-stClinic}"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

pass=0
fail=0

check() {
  local name="$1"
  shift
  if "$@"; then
    echo "PASS  ${name}"
    pass=$((pass + 1))
  else
    echo "FAIL  ${name}" >&2
    fail=$((fail + 1))
  fi
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

check_env() {
  local env="$1"
  conda_env_exists "${env}"
}

check_script_help() {
  local env="$1"
  local script="$2"
  (cd "${SCRIPT_DIR}" && conda run --no-capture-output -n "${env}" python "${script}" --help) >/dev/null 2>&1
}

check_imports() {
  local env="$1"
  local modules="$2"
  conda run --no-capture-output -n "${env}" python -c "
import os, sys
for p in ('${ST_ROOT}/GraphST', '${ST_ROOT}'):
    if p not in sys.path:
        sys.path.insert(0, p)
# GraphST repo root must precede ST-Analysis root on sys.path
sys.path.sort(key=lambda p: (p != '${ST_ROOT}/GraphST', p != '${ST_ROOT}', p))
os.environ.setdefault('R_HOME', '/usr/lib/R')
for m in '''${modules}'''.split():
    __import__(m)
print('imports OK')
" >/dev/null 2>&1
}

echo "=== Conda environments ==="
for env in "${ENV_GRAPHST}" "${ENV_SEDR}" "${ENV_BANKSY}" "${ENV_CELLCHARTER}" "${ENV_STALIGNER}" "${ENV_METRICS}"; do
  check "env ${env}" check_env "${env}"
done

echo
echo "=== Script --help ==="
check "run_graphst.py (${ENV_GRAPHST})" check_script_help "${ENV_GRAPHST}" run_graphst.py
check "run_sedr.py (${ENV_SEDR})" check_script_help "${ENV_SEDR}" run_sedr.py
check "run_banksy.py (${ENV_BANKSY})" check_script_help "${ENV_BANKSY}" run_banksy.py
check "run_cellcharter.py (${ENV_CELLCHARTER})" check_script_help "${ENV_CELLCHARTER}" run_cellcharter.py
check "run_staligner.py (${ENV_STALIGNER})" check_script_help "${ENV_STALIGNER}" run_staligner.py
check "evaluate_metrics.py (${ENV_METRICS})" check_script_help "${ENV_METRICS}" evaluate_metrics.py

echo
echo "=== Key imports ==="
check "GraphST (${ENV_GRAPHST})" check_imports "${ENV_GRAPHST}" "GraphST GraphST.utils torch scanpy"
check "SEDR (${ENV_SEDR})" check_imports "${ENV_SEDR}" "SEDR harmonypy scanpy torch"
check "BANKSY (${ENV_BANKSY})" check_imports "${ENV_BANKSY}" "banksy scanpy"
check "CellCharter (${ENV_CELLCHARTER})" check_imports "${ENV_CELLCHARTER}" "cellcharter squidpy scanpy torch"
check "STAligner (${ENV_STALIGNER})" check_imports "${ENV_STALIGNER}" "STAligner scanpy torch"
check "metrics (${ENV_METRICS})" check_imports "${ENV_METRICS}" "scanpy sklearn"

echo
echo "=== Data loading ==="
if [[ -n "${DATA_DIR}" ]]; then
  check "load_data.py" bash -c "cd '${SCRIPT_DIR}' && conda run --no-capture-output -n '${ENV_METRICS}' python -c \"
from load_data import discover_section_ids, load_sections
ids = discover_section_ids('${DATA_DIR}')
adata = load_sections(ids[:1], '${DATA_DIR}')
assert adata[0].n_obs > 0 and 'spatial' in adata[0].obsm
print(adata[0].shape, ids[0])
\" >/dev/null 2>&1"
else
  echo "SKIP  data loading (set DATA_DIR to test)"
fi

echo
echo "=== Summary: ${pass} passed, ${fail} failed ==="
if [[ "${fail}" -gt 0 ]]; then
  exit 1
fi
