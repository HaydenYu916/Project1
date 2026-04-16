#!/usr/bin/env python3
"""
Download power history from Home Assistant REST API to CSV.

Usage:
    python app.py           # download using config.json defaults
    python app.py down      # same
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import requests

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
OUT_DIR = BASE_DIR / "logs"

DEFAULT_TIMEZONE = "Australia/Sydney"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"[错误] 找不到配置文件: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[错误] config.json 格式错误: {e}", file=sys.stderr)
        sys.exit(1)


def parse_local_datetime(value: str, tz: ZoneInfo) -> datetime:
    """将本地时间字符串解析为 UTC datetime。"""
    dt = datetime.fromisoformat(value.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)


def fetch_entity_history(
    base_url: str,
    token: str,
    entity_id: str,
    start_time: datetime,
    end_time: datetime,
    timeout: int = 30,
) -> List[dict]:
    """调用 HA REST History API，返回记录列表 [{last_changed, state}, ...]。"""
    start_iso = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        f"{base_url.rstrip('/')}/api/history/period/{start_iso}"
        f"?filter_entity_id={entity_id}&end_time={end_iso}&minimal_response=true"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not data or not data[0]:
        return []
    return data[0]


def fetch_all_entities(base_url: str, token: str, timeout: int = 30) -> List[str]:
    """调用 /api/states 获取所有实体 ID。"""
    url = f"{base_url.rstrip('/')}/api/states"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return sorted(item["entity_id"] for item in r.json())


def save_csv(entity_id: str, records: List[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = entity_id.replace(".", "_")
    path = out_dir / f"{safe_name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "state"])
        for rec in records:
            writer.writerow([rec.get("last_changed", ""), rec.get("state", "")])
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download HA entity history to CSV in logs/."
    )
    parser.add_argument(
        "command", nargs="?", default="down", choices=["down", "entities"],
        help="down: fetch history to logs/ (default) | entities: list all HA entities"
    )
    args = parser.parse_args(argv)

    cfg = load_config()

    ha_cfg = cfg.get("ha", {})
    base_url = ha_cfg.get("url", "").strip()
    token = ha_cfg.get("token", "").strip()
    entities = ha_cfg.get("entities", [])

    if not base_url or not token:
        print("[错误] config.json 缺少 ha.url 或 ha.token", file=sys.stderr)
        return 1

    # entities 命令：拉取所有实体并保存到文件
    if args.command == "entities":
        print(f"拉取 {base_url} 的实体列表...")
        all_entities = fetch_all_entities(base_url, token)
        out_path = BASE_DIR / "ha_entities.txt"
        out_path.write_text("\n".join(all_entities) + "\n", encoding="utf-8")
        print(f"共 {len(all_entities)} 个实体，已保存到 {out_path}")
        for eid in all_entities:
            print(f"  {eid}")
        return 0

    if not entities:
        print("[错误] config.json 缺少 ha.entities", file=sys.stderr)
        return 1

    q = cfg.get("query_defaults", {})
    tz_name = q.get("timezone", DEFAULT_TIMEZONE)
    tz = ZoneInfo(tz_name)
    start_time = parse_local_datetime(q.get("start_time", ""), tz)
    end_time = parse_local_datetime(q.get("end_time", ""), tz)

    print(f"HA URL   : {base_url}")
    print(f"Entities : {entities}")
    print(f"From     : {start_time.isoformat()}")
    print(f"To       : {end_time.isoformat()}")
    print()

    ok, fail = [], []
    for entity_id in entities:
        try:
            records = fetch_entity_history(base_url, token, entity_id, start_time, end_time)
            if not records:
                print(f"  (空) {entity_id}")
                fail.append(entity_id)
                continue
            path = save_csv(entity_id, records, OUT_DIR)
            print(f"  OK  {entity_id}  ({len(records)} 条) -> {path}")
            ok.append(entity_id)
        except Exception as e:
            print(f"  ERR {entity_id}: {e}")
            fail.append(entity_id)

    print(f"\n完成: 成功 {len(ok)}, 失败 {len(fail)}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
