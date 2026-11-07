#!/bin/sh
# One command: sets everything up on the first run, then starts the app.      sh run_mac.sh [--mock]   (port 8766, change with P2R_PORT=9000 sh run_mac.sh)
# Needs: an Apple-Silicon Mac (M1 or newer), Python 3.10 or newer, internet on the first run (about 5 GB model download).
cd "$(dirname "$0")" || exit 1

# --- find a Python >= 3.10 (macOS' built-in python3 is often 3.9, which is too old)
PY=""
for cand in "$PYTHON" python3.13 python3.12 python3.11 python3.10 python3 \
            /opt/homebrew/bin/python3 /usr/local/bin/python3 \
            /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
            /Library/Frameworks/Python.framework/Versions/3.11/bin/python3; do
  [ -n "$cand" ] || continue
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found."
  echo "Install it from https://www.python.org/downloads/macos/ (the normal installer) and run this script again."
  exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then echo "This needs an Apple-Silicon Mac (MLX does not run on Intel Macs)."; exit 1; fi

# --- one-time setup
# A .venv copied from another Mac or folder does not work here: rebuild it.
VPY=.venv/bin/python
if [ -d .venv ]; then
  if ! "$VPY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null || \
     { [ -f .venv/.deps_ok ] && ! "$VPY" -c 'import mujoco, mlx.core' 2>/dev/null; }; then
    echo ">> the .venv folder does not work on this Mac (copied from elsewhere?), rebuilding it"
    rm -rf .venv
  fi
fi
if [ ! -d .venv ]; then
  echo ">> creating a private Python environment (.venv) with $("$PY" --version)"
  "$PY" -m venv .venv || exit 1
fi
if [ ! -f .venv/.deps_ok ] || [ requirements.txt -nt .venv/.deps_ok ]; then
  echo ">> installing dependencies (MuJoCo, MLX, ...) - a few minutes the first time"
  "$VPY" -m pip install -q -U pip && "$VPY" -m pip install -q -U -r requirements.txt || { echo "Installing the dependencies failed."; exit 1; }
  touch .venv/.deps_ok
fi

echo ">> starting the app; the browser opens by itself (models load in the background)."
echo "   First start downloads the base model (about 5 GB). Leave this window open; press Ctrl+C to stop."
PORT="${P2R_PORT:-8766}"
( sleep 5 && open "http://localhost:$PORT" ) >/dev/null 2>&1 &
P2R_PORT="$PORT" "$VPY" server.py "$@"
