#!/bin/bash
# 安装自动调参所需的依赖包

echo "🔧 安装MPPI自动调参依赖包..."

# 基础依赖
pip install scipy scikit-learn pandas numpy matplotlib

# 贝叶斯优化依赖 (可选)
echo "📦 安装贝叶斯优化依赖 (可选)..."
pip install scikit-optimize

# 验证安装
echo "✅ 验证安装..."
python -c "
try:
    import scipy
    print(f'✅ scipy {scipy.__version__}')
except ImportError:
    print('❌ scipy 未安装')

try:
    import sklearn
    print(f'✅ scikit-learn {sklearn.__version__}')
except ImportError:
    print('❌ scikit-learn 未安装')

try:
    import pandas
    print(f'✅ pandas {pandas.__version__}')
except ImportError:
    print('❌ pandas 未安装')

try:
    import numpy
    print(f'✅ numpy {numpy.__version__}')
except ImportError:
    print('❌ numpy 未安装')

try:
    import matplotlib
    print(f'✅ matplotlib {matplotlib.__version__}')
except ImportError:
    print('❌ matplotlib 未安装')

try:
    import skopt
    print(f'✅ scikit-optimize {skopt.__version__}')
except ImportError:
    print('⚠️ scikit-optimize 未安装 (贝叶斯优化功能将不可用)')
"

echo "🎉 依赖安装完成！"
echo ""
echo "使用方法："
echo "1. 离线调参: python applications/tuning/auto_tune_mppi.py --method bayesian"
echo "2. 在线调参: python applications/control/mppi_control_adaptive.py continuous"
echo "3. 查看状态: python applications/control/mppi_control_adaptive.py adaptive_status"
