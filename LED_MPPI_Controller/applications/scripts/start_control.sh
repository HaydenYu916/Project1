#!/bin/bash
# MPPI控制循环启动脚本

echo "🌱 启动MPPI LED控制循环系统"
echo "================================"

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未找到，请先安装Python"
    exit 1
fi

# 使用anaconda的Python（包含numpy等依赖）
PYTHON_CMD="python"

# 显示使用说明
echo "使用方法:"
echo "  ./start_control.sh once          # 运行一次（模拟模式）"
echo "  ./start_control.sh continuous    # 连续运行（模拟模式）"
echo "  ./start_control.sh execute       # 运行一次（实际执行）"
echo "  ./start_control.sh execute-cont  # 连续运行（实际执行，每小时的0,15,30,45分运行，23:00-07:00夜间休眠）"
echo "  ./start_control.sh start         # 启动后台进程（实际执行）"
echo "  ./start_control.sh stop          # 停止后台进程"
echo "  ./start_control.sh restart       # 重启后台进程"
echo "  ./start_control.sh status        # 查看后台进程状态"
echo "  ./start_control.sh test          # 运行系统测试"
echo "  ./start_control.sh list-devices  # 列出可用设备"
echo ""
echo "💡 提示: 修改代码顶部的宏定义来配置设备ID和其他参数"
echo "🌙 夜间模式: 连续执行模式在23:00-07:00期间自动休眠"
echo ""

# 根据参数运行
case "$1" in
    "once")
        echo "🔄 运行单次控制循环（模拟模式）..."
        $PYTHON_CMD applications/control/mppi_control_simulate.py once
        ;;
    "continuous")
        echo "🔄 开始连续控制循环（模拟模式）..."
        $PYTHON_CMD applications/control/mppi_control_simulate.py continuous
        ;;
    "execute")
        echo "🚀 运行单次控制执行（实际发送命令）..."
        $PYTHON_CMD applications/control/mppi_control_real.py once
        ;;
    "execute-cont")
        echo "🚀 开始连续控制执行（实际发送命令）..."
        $PYTHON_CMD applications/control/mppi_control_real.py continuous
        ;;
    "start")
        echo "🚀 启动MPPI控制后台进程..."
        $PYTHON_CMD applications/control/mppi_control_real.py start
        ;;
    "stop")
        echo "⏹️  停止MPPI控制后台进程..."
        $PYTHON_CMD applications/control/mppi_control_real.py stop
        ;;
    "restart")
        echo "🔄 重启MPPI控制后台进程..."
        $PYTHON_CMD applications/control/mppi_control_real.py restart
        ;;
    "status")
        echo "📊 查看MPPI控制后台进程状态..."
        $PYTHON_CMD applications/control/mppi_control_real.py status
        ;;
    "test")
        echo "🧪 运行系统测试..."
        $PYTHON_CMD tests/test_system.py
        ;;
    "list-devices")
        echo "📱 列出可用设备..."
        $PYTHON_CMD applications/control/mppi_control_simulate.py list-devices
        ;;
    *)
        echo "❌ 无效参数"
        echo "用法: $0 [once|continuous|execute|execute-cont|start|stop|restart|status|test|list-devices]"
        echo ""
        echo "💡 提示: 修改代码顶部的宏定义来配置设备ID和其他参数"
        exit 1
        ;;
esac
