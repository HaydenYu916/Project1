#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动调参快速启动脚本

提供简单的命令行接口来运行各种自动调参功能。
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="MPPI自动调参快速启动")
    parser.add_argument("command", 
                       choices=["install", "test", "offline", "online", "status"],
                       help="要执行的命令")
    parser.add_argument("--method", 
                       choices=["bayesian", "evolutionary", "grid"],
                       default="evolutionary",
                       help="离线调参方法")
    parser.add_argument("--iterations", type=int, default=10,
                       help="调参迭代次数")
    parser.add_argument("--duration", type=float, default=1.0,
                       help="在线调参持续时间(小时)")
    
    args = parser.parse_args()
    
    if args.command == "install":
        print("🔧 安装自动调参依赖...")
        os.system("bash install_tuning_deps.sh")
        
    elif args.command == "test":
        print("🧪 运行自动调参测试...")
        os.system("python test_auto_tuning.py")
        
    elif args.command == "offline":
        print(f"🔍 运行离线调参 (方法: {args.method}, 迭代: {args.iterations})...")
        cmd = f"python auto_tune_mppi.py --method {args.method} --iterations {args.iterations}"
        os.system(cmd)
        
    elif args.command == "online":
        print(f"🔄 运行在线自适应调参 (持续时间: {args.duration}小时)...")
        cmd = f"python mppi_control_adaptive.py continuous"
        print("按 Ctrl+C 停止")
        os.system(cmd)
        
    elif args.command == "status":
        print("📊 查看自适应调参状态...")
        os.system("python mppi_control_adaptive.py adaptive_status")
    
    print("✅ 命令执行完成")

if __name__ == "__main__":
    main()
