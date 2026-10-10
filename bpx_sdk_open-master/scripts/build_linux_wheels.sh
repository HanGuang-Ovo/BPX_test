#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-cibw-linux"
VENV_PYTHON="${VENV_DIR}/bin/python"
OUT_DIR="${1:-wheelhouse}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [output-directory]" >&2
    exit 2
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: this script only supports Linux." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required but was not found in PATH." >&2
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 127
fi

create_venv=false
if [[ ! -x "${VENV_PYTHON}" ]]; then
    create_venv=true
elif ! "${VENV_PYTHON}" -c \
    'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
    echo "Recreating .venv-cibw-linux because it does not use Python 3.11."
    create_venv=true
elif ! "${VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
    echo "Recreating .venv-cibw-linux because pip is not installed."
    create_venv=true
fi

if [[ "${create_venv}" == true ]]; then
    uv venv --clear --python 3.11 --seed "${VENV_DIR}"
else
    echo "Reusing Python 3.11 environment at ${VENV_DIR}"
fi

cd -- "${ROOT_DIR}"
exec "${VENV_PYTHON}" scripts/build_wheels.py \
    --cibuildwheel \
    --out-dir "${OUT_DIR}"
