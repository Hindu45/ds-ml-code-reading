#!/usr/bin/env bash
# ================================================================
#  Python interpreter fallback list — first existing path wins.
#  Add entries to PYTHON_CANDIDATES as needed.
# ================================================================
PYTHON_CANDIDATES=(
    python3
    python
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Find first usable Python
PYTHON_EXE=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON_EXE="$(command -v "$candidate")"
        break
    fi
done

# Handle existing venv
if [ -d "$VENV_DIR" ]; then
    read -rp "[setup] .venv already exists. Recreate? [y/N]: " RECREATE
    if [[ "${RECREATE,,}" == "y" ]]; then
        echo "[setup] Removing existing .venv ..."
        rm -rf "$VENV_DIR"
    else
        echo "[setup] Keeping existing .venv."
        echo "[setup] Running install-missing-packages.py ..."
        "$VENV_DIR/bin/python" "$SCRIPT_DIR/install-missing-packages.py"
        exit $?
    fi
fi

if [ -z "$PYTHON_EXE" ]; then
    echo "[setup] ERROR: No Python interpreter found. Add your path to PYTHON_CANDIDATES in this script."
    exit 1
fi

echo "[setup] Using Python: $PYTHON_EXE"
echo "[setup] Creating .venv ..."
"$PYTHON_EXE" -m venv "$VENV_DIR"
if [ $? -ne 0 ]; then
    echo "[setup] ERROR: venv creation failed."
    exit 1
fi
echo "[setup] .venv created."

echo "[setup] Running install-missing-packages.py ..."
"$VENV_DIR/bin/python" "$SCRIPT_DIR/install-missing-packages.py"
exit $?
