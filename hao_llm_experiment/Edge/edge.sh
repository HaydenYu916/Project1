#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/edge.pid"
LOG_FILE="$ROOT_DIR/nohup_edge.log"
PYTHON_BIN="/home/hao/miniforge3/envs/project1/bin/python"
SCRIPT_PATH="$ROOT_DIR/greenhouse_edge.py"
CMD_PATTERN="$SCRIPT_PATH"

is_running() {
  pgrep -af "$CMD_PATTERN" >/dev/null 2>&1
}

show_status() {
  if is_running; then
    echo "Edge is running:"
    pgrep -af "$CMD_PATTERN"
  else
    echo "Edge is not running."
  fi

  if [[ -f "$LOG_FILE" ]]; then
    echo "log_file: $LOG_FILE"
  fi
}

start_edge() {
  if is_running; then
    echo "Edge is already running:"
    pgrep -af "$CMD_PATTERN"
    return 0
  fi

  cd "$ROOT_DIR"
  touch "$LOG_FILE"
  setsid "$PYTHON_BIN" "$SCRIPT_PATH" >>"$LOG_FILE" 2>&1 < /dev/null &
  local edge_pid=$!

  sleep 2

  if is_running; then
    echo "$edge_pid" >"$PID_FILE"
    echo "Edge started."
    echo "pid_file: $PID_FILE"
    echo "log_file: $LOG_FILE"
    pgrep -af "$CMD_PATTERN"
  else
    echo "Edge failed to start. Check $LOG_FILE"
    return 1
  fi
}

stop_edge() {
  if is_running; then
    pkill -f "$CMD_PATTERN"
    sleep 1
  fi

  rm -f "$PID_FILE"

  if is_running; then
    echo "Edge is still running:"
    pgrep -af "$CMD_PATTERN"
    return 1
  fi

  echo "Edge stopped."
}

show_logs() {
  if [[ -f "$LOG_FILE" ]]; then
    tail -f "$LOG_FILE"
  else
    echo "Log file does not exist yet: $LOG_FILE"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./edge.sh start
  ./edge.sh stop
  ./edge.sh status
  ./edge.sh restart
  ./edge.sh logs
EOF
}

ACTION="${1:-status}"

case "$ACTION" in
  start)
    start_edge
    ;;
  stop)
    stop_edge
    ;;
  status)
    show_status
    ;;
  restart)
    stop_edge
    start_edge
    ;;
  logs)
    show_logs
    ;;
  *)
    usage
    exit 1
    ;;
esac
