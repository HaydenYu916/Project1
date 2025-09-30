#!/bin/bash
# MPPI v2 控制循环启动脚本（Solar Vol 版本）

echo "🌱 启动 MPPI v2 LED 控制系统"
echo "================================"

# 切换到项目根目录
cd "$(dirname "$0")/../.." || exit 1

if ! command -v python &> /dev/null; then
    echo "❌ 未找到 Python，请先安装或配置环境"
    exit 1
fi

PYTHON_CMD="python"

echo "使用方法:"
echo "  ./start_control_v2.sh once          # 运行一次仿真"
echo "  ./start_control_v2.sh continuous    # 连续仿真运行"
echo "  ./start_control_v2.sh execute       # 运行一次实际控制"
echo "  ./start_control_v2.sh execute-cont  # 连续实际控制"
echo "  ./start_control_v2.sh start         # 后台启动实际控制"
echo "  ./start_control_v2.sh stop          # 停止后台实际控制"
echo "  ./start_control_v2.sh restart       # 重启后台实际控制"
echo "  ./start_control_v2.sh status        # 查看后台状态"
echo ""
echo "默认配置：控制间隔 15 分钟，可在 mppi_control_real_v2.py 顶部常量中调整"
echo ""

case "$1" in
    "once")
        echo "🔄 运行单次控制循环（仿真）"
        $PYTHON_CMD applications/control/mppi_control_simulate_v2.py --steps 1
        ;;
    "continuous")
        echo "🔄 连续运行控制循环（仿真）"
        $PYTHON_CMD applications/control/mppi_control_simulate_v2.py
        ;;
    "execute")
        echo "🚀 运行单次实际控制"
        $PYTHON_CMD applications/control/mppi_control_real_v2.py once
        ;;
    "execute-cont")
        echo "🚀 连续实际控制"
        $PYTHON_CMD applications/control/mppi_control_real_v2.py continuous
        ;;
    "start")
        echo "🚀 启动后台实际控制"
        $PYTHON_CMD applications/control/mppi_control_real_v2.py start
        ;;
    "stop")
        echo "⏹️  停止后台实际控制"
        $PYTHON_CMD applications/control/mppi_control_real_v2.py stop
        ;;
    "restart")
        echo "🔄 重启后台实际控制"
        $PYTHON_CMD applications/control/mppi_control_real_v2.py restart
        ;;
    "status")
        echo "📊 查看后台状态"
        $PYTHON_CMD applications/control/mppi_control_real_v2.py status
        ;;
    *)
        echo "❌ 无效参数"
        echo "用法: $0 [once|continuous|execute|execute-cont|start|stop|restart|status]"
        exit 1
        ;;
esac
