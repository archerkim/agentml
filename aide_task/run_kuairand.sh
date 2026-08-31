#!/usr/bin/env bash
# Launch the full KuaiRand-Pure AIDE run against the local Qwen 3.8 27B alias at
# maximum reasoning effort.
#
# Two things this wrapper exists for, both of which silently break the run if done
# by hand:
#   1. The venv must be ACTIVATED, not just invoked by path. verify_candidate() and
#      the ensemble step shell out to `python3`, which resolves through PATH - with
#      the venv merely invoked, that lands on /usr/bin/python3, which has no pandas,
#      and every verification would fail with an import error.
#   2. ollama_max_reasoning_proxy.py must be up before AIDE starts; it is what turns
#      the plain chat-completions call into a reasoning_effort=max one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${AIDE_VENV:-/home/artur/.venvs/agentml-aide}"
RUN_LOG_DIR="$HERE/run_logs"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$RUN_LOG_DIR"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! curl -sf http://127.0.0.1:11435/_health > /dev/null; then
  echo "starting maximum-reasoning proxy ..."
  nohup python3 "$HERE/ollama_max_reasoning_proxy.py" \
    > "$RUN_LOG_DIR/proxy-$STAMP.log" 2>&1 &
  for _ in $(seq 1 20); do
    curl -sf http://127.0.0.1:11435/_health > /dev/null && break
    sleep 1
  done
fi
curl -sf http://127.0.0.1:11435/_health > /dev/null \
  || { echo "proxy failed to come up; see $RUN_LOG_DIR/proxy-$STAMP.log" >&2; exit 1; }
echo "proxy healthy on 127.0.0.1:11435"

cd "$HERE"
echo "console log -> $RUN_LOG_DIR/run-$STAMP.log"
python3 -u run_with_early_stop.py "$@" 2>&1 \
  | tee "$RUN_LOG_DIR/run-$STAMP.log"
