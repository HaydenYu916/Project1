#!/usr/bin/env python3
import os
import csv
import glob
from datetime import datetime
from typing import Optional, Dict, Any, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")


def _find_latest_csv(pattern: str) -> Optional[str]:
    os.makedirs(LOGS_DIR, exist_ok=True)
    files = glob.glob(os.path.join(LOGS_DIR, pattern))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _read_tail(csv_path: str, limit: int) -> List[Dict[str, Any]]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', newline='') as f:
        first = f.readline()
        if not first.startswith('#'):
            f.seek(0)
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows[-limit:] if limit and len(rows) > limit else rows


def get_latest_event() -> Optional[Dict[str, Any]]:
    path = _find_latest_csv("shelly_log_event*.csv") or os.path.join(LOGS_DIR, "shelly_log_event.csv")
    rows = _read_tail(path, 1)
    return rows[-1] if rows else None


def get_recent_all(n: int = 10) -> List[Dict[str, Any]]:
    path = _find_latest_csv("shelly_log_all*.csv") or os.path.join(LOGS_DIR, "shelly_log_all.csv")
    return _read_tail(path, n)


if __name__ == "__main__":
    print(get_latest_event())

