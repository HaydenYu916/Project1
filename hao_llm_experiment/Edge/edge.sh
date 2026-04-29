#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SOURCE="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
PID_FILE="$ROOT_DIR/edge.pid"
LOG_FILE="$ROOT_DIR/nohup_edge.log"
PYTHON_BIN="/home/hao/miniforge3/envs/project1/bin/python"
SCRIPT_PATH="$ROOT_DIR/greenhouse_edge.py"
CMD_PATTERN="$PYTHON_BIN $SCRIPT_PATH"
SERVICE_NAME="greenhouse-edge.service"

SUDO=""
if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
fi

has_systemd_service() {
  systemctl cat "$SERVICE_NAME" >/dev/null 2>&1
}

systemd_is_active() {
  [[ "$(systemctl is-active "$SERVICE_NAME" 2>/dev/null || true)" == "active" ]]
}

show_systemd_hint() {
  echo "Detected systemd service: $SERVICE_NAME"
  echo "This script will use systemd to avoid duplicate Edge processes."
}

is_running() {
  pgrep -af "$CMD_PATTERN" >/dev/null 2>&1
}

show_status() {
  if has_systemd_service; then
    show_systemd_hint
    systemctl status "$SERVICE_NAME" --no-pager -l || true
    return 0
  fi

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
  if has_systemd_service; then
    show_systemd_hint
    if systemd_is_active; then
      echo "Edge service is already running."
      systemctl status "$SERVICE_NAME" --no-pager -l || true
      return 0
    fi
    $SUDO systemctl start "$SERVICE_NAME"
    echo "Edge service started via systemd."
    systemctl status "$SERVICE_NAME" --no-pager -l || true
    return 0
  fi

  if is_running; then
    echo "Edge is already running:"
    pgrep -af "$CMD_PATTERN"
    return 0
  fi

  cd "$ROOT_DIR"
  touch "$LOG_FILE"
  setsid "$PYTHON_BIN" "$SCRIPT_PATH" >>"$LOG_FILE" 2>&1 < /dev/null &
  disown || true

  sleep 2

  local edge_pid
  edge_pid="$(pgrep -f "$CMD_PATTERN" | head -n1 || true)"

  if [[ -n "$edge_pid" ]]; then
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
  if has_systemd_service; then
    show_systemd_hint
    if ! systemd_is_active; then
      echo "Edge service is not running."
      return 0
    fi
    $SUDO systemctl stop "$SERVICE_NAME"
    rm -f "$PID_FILE"
    echo "Edge service stopped via systemd."
    return 0
  fi

  if is_running; then
    pkill -f "$CMD_PATTERN" || true
    sleep 1
  fi

  if is_running; then
    pkill -9 -f "$CMD_PATTERN" || true
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
  if has_systemd_service; then
    show_systemd_hint
    journalctl -u "$SERVICE_NAME" -f
    return 0
  fi

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

If systemd service greenhouse-edge.service is installed, this script proxies to systemctl/journalctl.
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
    stop_edge || true
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
