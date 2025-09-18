#!/bin/bash
# MPPI控制循环启动脚本

echo "🌱 启动MPPI LED控制循环系统"
echo "================================"

# 切换到脚本目录
cd "$(dirname "$0")"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未找到，请先安装Python"
    exit 1
fi

# 使用anaconda的Python（包含numpy等依赖）
PYTHON_CMD="python"

# 显示使用说明
echo "使用方法:"
echo "  ./start_mppi_control.sh once          # 运行一次"
echo "  ./start_mppi_control.sh continuous    # 连续运行（每分钟）"
echo "  ./start_mppi_control.sh test          # 运行测试"
echo ""

# 根据参数运行
case "$1" in
    "once")
        echo "🔄 运行单次控制循环..."
        $PYTHON_CMD mppi_control_loop.py once
        ;;
    "continuous")
        echo "🔄 开始连续控制循环..."
        $PYTHON_CMD mppi_control_loop.py continuous
        ;;
    "test")
        echo "🧪 运行集成测试..."
        $PYTHON_CMD test_mppi_integration.py
        ;;
    *)
        echo "❌ 无效参数"
        echo "用法: $0 [once|continuous|test]"
        exit 1
        ;;
esac
