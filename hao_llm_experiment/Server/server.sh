#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/server.pid"
LOG_FILE="$ROOT_DIR/nohup_server.log"
PYTHON_BIN="/home/hao/miniforge3/envs/project1/bin/python"
SCRIPT_PATH="$ROOT_DIR/cloud_controller.py"

is_running() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi

  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

show_process() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    ps -p "$pid" -o pid=,args=
  fi
}

show_status() {
  if is_running; then
    echo "Server is running:"
    show_process
  else
    echo "Server is not running."
  fi

  if [[ -f "$LOG_FILE" ]]; then
    echo "log_file: $LOG_FILE"
  fi
}

start_server() {
  if is_running; then
    echo "Server is already running:"
    show_process
    return 0
  fi

  rm -f "$PID_FILE"
  cd "$ROOT_DIR"
  touch "$LOG_FILE"
  setsid "$PYTHON_BIN" "$SCRIPT_PATH" >>"$LOG_FILE" 2>&1 < /dev/null &
  local server_pid=$!

  echo "$server_pid" >"$PID_FILE"
  sleep 2

  if is_running; then
    echo "Server started."
    echo "pid_file: $PID_FILE"
    echo "log_file: $LOG_FILE"
    show_process
  else
    echo "Server failed to start. Check $LOG_FILE"
    rm -f "$PID_FILE"
    return 1
  fi
}

stop_server() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid" 2>/dev/null || true
    sleep 1
  fi

  rm -f "$PID_FILE"

  if is_running; then
    echo "Server is still running:"
    show_process
    return 1
  fi

  echo "Server stopped."
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
  ./server.sh start
  ./server.sh stop
  ./server.sh status
  ./server.sh restart
  ./server.sh logs
EOF
}

ACTION="${1:-status}"

case "$ACTION" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  status)
    show_status
    ;;
  restart)
    stop_server
    start_server
    ;;
  logs)
    show_logs
    ;;
  *)
    usage
    exit 1
    ;;
esac
