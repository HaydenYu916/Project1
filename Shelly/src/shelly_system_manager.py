#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime


class ShellySystemManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.script = self.base_dir / "shelly_listener.py"
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.run_logs_dir = self.logs_dir / "run"
        self.run_logs_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.run_logs_dir / "shelly_listener.pid"
        self.print_output_file = self.run_logs_dir / "shelly_listener_print_output.txt"
        self.env_python = "/home/pi/Desktop/riotee-env/bin/python3"

    def is_running(self) -> bool:
        if not self.pid_file.exists():
            return False
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, 0)
            return True
        except Exception:
            self.pid_file.unlink(missing_ok=True)
            return False

    def start(self, note: str = "", extra_args: str = "") -> bool:
        if self.is_running():
            print("✅ Shelly 监听已在运行")
            return True

        cmd = [self.env_python, str(self.script)]
        if extra_args:
            cmd.extend(extra_args.split())
        if note:
            cmd.append(note)

        with open(self.print_output_file, 'w', encoding='utf-8') as out:
            out.write(f"# Shelly 监听启动 @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            out.write(f"# 命令: {' '.join(cmd)}\n")
            out.write("# " + "=" * 50 + "\n")
            out.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=str(self.base_dir),
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
            )

        self.pid_file.write_text(str(proc.pid))
        time.sleep(2)
        if proc.poll() is None:
            print(f"✅ Shelly 监听启动成功 (PID: {proc.pid})")
            print(f"📄 输出: {self.print_output_file}")
            return True

        print("❌ Shelly 监听启动失败，输出如下：")
        try:
            print(self.print_output_file.read_text())
        except Exception:
            pass
        return False

    def stop(self) -> bool:
        if not self.is_running():
            print("⏹️  Shelly 监听未运行")
            return True
        try:
            pid = int(self.pid_file.read_text().strip())
            print(f"⏹️  停止 Shelly 监听 (PID: {pid})...")
            os.kill(pid, signal.SIGTERM)
            for _ in range(20):
                time.sleep(0.2)
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
            else:
                os.kill(pid, signal.SIGKILL)
            self.pid_file.unlink(missing_ok=True)
            print("✅ 已停止")
            return True
        except Exception as e:
            print(f"❌ 停止失败: {e}")
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shelly 监听系统管理")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("note", nargs="*", help="备注")
    s.add_argument("--args", default="", help="传递给监听脚本的额外参数，如 --devices ... --port ...")
    sub.add_parser("stop")
    sub.add_parser("status")

    args = parser.parse_args()
    mgr = ShellySystemManager()
    if args.cmd == "start":
        note = " ".join(args.note) if args.note else ""
        mgr.start(note=note, extra_args=args.args)
    elif args.cmd == "stop":
        mgr.stop()
    elif args.cmd == "status":
        print("运行中" if mgr.is_running() else "未运行")


if __name__ == "__main__":
    main()


