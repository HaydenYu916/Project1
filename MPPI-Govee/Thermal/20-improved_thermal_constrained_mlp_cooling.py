#!/usr/bin/env python3
"""
修复版：基于热力学约束的MLP模型 - 关灯阶段分析
==============================================

修复问题：
1. 改进纯热力学模型拟合方法
2. 按PPFD分组，计算同一Solar值的平均数据
3. 使用平均数据进行训练和对比

作者: AI Assistant
日期: 2025-01-15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit, minimize, differential_evolution
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
from pathlib import Path
import warnings
from collections import defaultdict
import pickle
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedThermodynamicConstrainedMLPCooling:
    """改进版：基于热力学约束的MLP模型 - 关灯阶段"""
    
    def __init__(self):
        self.thermal_params = None
        self.mlp_model = None
        self.scaler = StandardScaler()
        self.fitted = False
    
    @staticmethod
    def thermal_cooling_response(t, a1_val, K1_base, tau1, K2_base, tau2, alpha_solar):
        """
        热力学降温响应函数
        ΔT(t) = K1(a1) × exp(-t/τ1) + K2(a1) × exp(-t/τ2)
        """
        a1_ref = 1.4
        solar_factor = 1 + alpha_solar * (a1_val - a1_ref)
        K1_solar = K1_base * solar_factor
        K2_solar = K2_base * solar_factor
        return K1_solar * np.exp(-t / tau1) + K2_solar * np.exp(-t / tau2)
    
    def fit_thermal_model_improved(self, t_data, a1_data, temp_diff_data):
        """改进的热力学模型拟合方法"""
        try:
            print("     使用改进的拟合方法...")
            
            # 创建包装函数
            def thermal_wrapper(t, K1_base, tau1, K2_base, tau2, alpha_solar):
                return self.thermal_cooling_response(t, a1_data, K1_base, tau1, K2_base, tau2, alpha_solar)
            
            # 改进的参数估计
            max_temp = np.max(temp_diff_data)
            min_temp = np.min(temp_diff_data)
            temp_range = max_temp - min_temp
            
            # 更合理的初始猜测
            K1_guess = temp_range * 0.4
            K2_guess = temp_range * 0.6
            tau1_guess = 20  # 更合理的快速降温时间
            tau2_guess = 150  # 更合理的慢速降温时间
            alpha_solar_guess = 0.5
            
            print(f"     初始参数: K1={K1_guess:.2f}, τ1={tau1_guess:.1f}, K2={K2_guess:.2f}, τ2={tau2_guess:.1f}, α={alpha_solar_guess:.2f}")
            
            # 使用differential_evolution进行全局优化
            def objective(params):
                K1_base, tau1, K2_base, tau2, alpha_solar = params
                try:
                    pred = thermal_wrapper(t_data, K1_base, tau1, K2_base, tau2, alpha_solar)
                    return np.sum((temp_diff_data - pred) ** 2)
                except:
                    return 1e10
            
            # 参数边界
            bounds = [
                (0, temp_range),      # K1_base
                (5, 50),             # tau1
                (0, temp_range),      # K2_base
                (50, 300),           # tau2
                (0, 2)               # alpha_solar
            ]
            
            # 全局优化
            result = differential_evolution(objective, bounds, seed=42, maxiter=1000)
            
            if result.success:
                params = result.x
                print(f"     全局优化成功")
            else:
                # 如果全局优化失败，使用curve_fit
                print(f"     全局优化失败，使用curve_fit")
                params, _ = curve_fit(
                    thermal_wrapper,
                    t_data,
                    temp_diff_data,
                    p0=[K1_guess, tau1_guess, K2_guess, tau2_guess, alpha_solar_guess],
                    bounds=([0, 5, 0, 50, 0], [temp_range, 50, temp_range, 300, 2]),
                    maxfev=10000
                )
            
            self.thermal_params = {
                'K1_base': params[0],
                'tau1': params[1],
                'K2_base': params[2],
                'tau2': params[3],
                'alpha_solar': params[4]
            }
            
            print(f"     拟合参数: K1_base={params[0]:.3f}, τ1={params[1]:.1f}分钟")
            print(f"     拟合参数: K2_base={params[2]:.3f}, τ2={params[3]:.1f}分钟")
            print(f"     Solar修正系数: α_solar={params[4]:.3f}")
            
            return True
            
        except Exception as e:
            print(f"     改进的热力学降温模型拟合失败: {e}")
            return False
    
    def fit(self, t_data, a1_data, temp_diff_data):
        """训练改进的热力学约束MLP模型"""
        try:
            print("🔬 训练改进的热力学约束MLP模型（关灯阶段）...")
            
            # 第一步：拟合改进的热力学基础模型
            print("   步骤1: 拟合改进的热力学降温基础模型...")
            thermal_success = self.fit_thermal_model_improved(t_data, a1_data, temp_diff_data)
            
            if not thermal_success:
                print("   ❌ 改进的热力学降温模型拟合失败")
                return False, 0, 0, None
            
            # 计算热力学模型预测
            thermal_pred = self.thermal_cooling_response(
                t_data, a1_data,
                self.thermal_params['K1_base'],
                self.thermal_params['tau1'],
                self.thermal_params['K2_base'],
                self.thermal_params['tau2'],
                self.thermal_params['alpha_solar']
            )
            
            # 计算热力学模型性能
            thermal_r2 = pearsonr(temp_diff_data, thermal_pred)[0] ** 2
            thermal_rmse = np.sqrt(np.mean((temp_diff_data - thermal_pred) ** 2))
            
            print(f"     改进热力学降温模型性能: R²={thermal_r2:.3f}, RMSE={thermal_rmse:.3f}")
            
            # 第二步：计算热力学模型残差
            residuals = temp_diff_data - thermal_pred
            
            # 第三步：训练MLP学习残差修正
            print("   步骤2: 训练MLP残差修正模型...")
            
            # 准备MLP特征
            features = np.column_stack([
                t_data / 100,  # 归一化时间
                a1_data,       # Solar值
                thermal_pred / 10,  # 热力学预测（归一化）
                t_data * a1_data / 100,  # 时间-Solar交互项
                np.sqrt(t_data),  # 平方根项
                np.log(1 + t_data)  # 对数项
            ])
            
            # 标准化特征
            features_scaled = self.scaler.fit_transform(features)
            
            # 训练MLP
            self.mlp_model = MLPRegressor(
                hidden_layer_sizes=(30, 20, 10),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=1000,
                random_state=42
            )
            
            self.mlp_model.fit(features_scaled, residuals)
            
            # 第四步：计算最终预测
            mlp_residual_pred = self.mlp_model.predict(features_scaled)
            final_pred = thermal_pred + mlp_residual_pred
            
            # 计算最终性能
            final_r2 = pearsonr(temp_diff_data, final_pred)[0] ** 2
            final_rmse = np.sqrt(np.mean((temp_diff_data - final_pred) ** 2))
            
            print(f"     最终模型性能: R²={final_r2:.3f}, RMSE={final_rmse:.3f}")
            print(f"     性能提升: ΔR²={final_r2-thermal_r2:.3f}, ΔRMSE={thermal_rmse-final_rmse:.3f}")
            
            self.fitted = True
            return True, final_r2, final_rmse, final_pred
            
        except Exception as e:
            print(f"改进的热力学约束MLP训练失败: {e}")
            return False, 0, 0, None
    
    def predict(self, t_data, a1_data):
        """预测温度差"""
        if not self.fitted:
            return None
        
        # 热力学模型预测
        thermal_pred = self.thermal_cooling_response(
            t_data, a1_data,
            self.thermal_params['K1_base'],
            self.thermal_params['tau1'],
            self.thermal_params['K2_base'],
            self.thermal_params['tau2'],
            self.thermal_params['alpha_solar']
        )
        
        # 准备MLP特征
        features = np.column_stack([
            t_data / 100,
            a1_data,
            thermal_pred / 10,
            t_data * a1_data / 100,
            np.sqrt(t_data),
            np.log(1 + t_data)
        ])
        
        features_scaled = self.scaler.transform(features)
        
        # MLP残差修正
        mlp_residual_pred = self.mlp_model.predict(features_scaled)
        
        # 最终预测
        return thermal_pred + mlp_residual_pred
    
    def predict_thermal_only(self, t_data, a1_data):
        """仅热力学模型预测（用于对比）"""
        if not self.fitted:
            return None
        
        return self.thermal_cooling_response(
            t_data, a1_data,
            self.thermal_params['K1_base'],
            self.thermal_params['tau1'],
            self.thermal_params['K2_base'],
            self.thermal_params['tau2'],
            self.thermal_params['alpha_solar']
        )


def prepare_averaged_cooling_data(csv_files):
    """准备按PPFD分组的平均关灯数据"""
    print("📊 准备按PPFD分组的平均关灯数据...")
    
    all_data = []
    
    for csv_file in csv_files:
        print(f"   处理文件: {csv_file.name}")
        
        # 读取数据
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 提取文件信息
        stem = csv_file.stem
        if "_a1_" in stem and "_ppfd_" in stem:
            parts = stem.split("_")
            a1_val = float(parts[4])
            ppfd_val = int(parts[6])
            
            # 排除PPFD 500数据
            if ppfd_val == 500:
                print(f"   跳过PPFD 500文件: {csv_file.name}")
                continue
        else:
            continue
        
        # 计算环境温度
        ambient_temp = df['temperature'].min()
        df['temp_diff'] = df['temperature'] - ambient_temp
        
        # 只使用关灯数据
        led_off_data = df[df['led_status'] == 0].copy()
        
        if len(led_off_data) > 10:
            # 计算从关灯开始的时间
            led_off_data['time_from_start'] = (led_off_data['timestamp'] - led_off_data['timestamp'].iloc[0]).dt.total_seconds() / 60
            
            # 添加数据
            for _, row in led_off_data.iterrows():
                all_data.append({
                    'time': row['time_from_start'],
                    'ppfd': ppfd_val,
                    'a1_val': a1_val,
                    'ambient_temp': ambient_temp,
                    'temp_diff': row['temp_diff'],
                    'file_name': csv_file.name
                })
    
    data_df = pd.DataFrame(all_data)
    
    # 按PPFD和Solar值分组，计算平均数据
    print("\n   计算按PPFD分组的平均数据...")
    averaged_data = []
    
    for ppfd in sorted(data_df['ppfd'].unique()):
        ppfd_data = data_df[data_df['ppfd'] == ppfd]
        print(f"     PPFD {ppfd}: {len(ppfd_data)} 个数据点")
        
        for solar_val in sorted(ppfd_data['a1_val'].unique()):
            solar_data = ppfd_data[ppfd_data['a1_val'] == solar_val]
            print(f"       Solar {solar_val}: {len(solar_data)} 个数据点")
            
            # 按时间分组，计算平均温度
            time_groups = solar_data.groupby('time')['temp_diff'].agg(['mean', 'std', 'count']).reset_index()
            
            for _, row in time_groups.iterrows():
                averaged_data.append({
                    'time': row['time'],
                    'ppfd': ppfd,
                    'a1_val': solar_val,
                    'temp_diff': row['mean'],
                    'temp_std': row['std'],
                    'count': row['count']
                })
    
    return pd.DataFrame(averaged_data)


def train_improved_thermal_constrained_mlp_cooling(data):
    """训练改进的热力学约束MLP模型（关灯阶段）"""
    print("\n🔬 训练改进的热力学约束MLP模型（关灯阶段）...")
    
    # 准备训练数据
    t_data = data['time'].values
    a1_data = data['a1_val'].values
    temp_diff_data = data['temp_diff'].values
    
    print(f"   训练数据点: {len(data)}")
    print(f"   Solar值范围: {data['a1_val'].min():.3f} - {data['a1_val'].max():.3f}")
    print(f"   时间范围: {data['time'].min():.1f} - {data['time'].max():.1f}分钟")
    
    # 创建并训练改进的热力学约束MLP模型
    model = ImprovedThermodynamicConstrainedMLPCooling()
    success, r2, rmse, pred = model.fit(t_data, a1_data, temp_diff_data)
    
    if success:
        print(f"   改进的热力学约束MLP模型性能（关灯阶段）: R²={r2:.3f}, RMSE={rmse:.3f}")
        return model
    else:
        print("   改进的热力学约束MLP模型训练失败")
        return None


def generate_improved_cooling_predictions(model, solar_values):
    """生成改进的关灯阶段预测"""
    print(f"\n🔮 生成改进的关灯阶段热力学约束MLP预测...")
    
    predictions = {}
    time_points = np.linspace(0, 400, 200)  # 关灯阶段时间较短
    
    for solar_val in solar_values:
        print(f"   生成Solar {solar_val:.3f}的关灯预测...")
        
        # 生成预测
        thermal_constrained_pred = model.predict(time_points, np.full_like(time_points, solar_val))
        thermal_only_pred = model.predict_thermal_only(time_points, np.full_like(time_points, solar_val))
        
        predictions[solar_val] = {
            'time': time_points,
            'thermal_constrained_pred': thermal_constrained_pred,
            'thermal_only_pred': thermal_only_pred,
            'solar_val': solar_val
        }
    
    return predictions


def plot_improved_cooling_comparison(predictions, original_data, output_dir):
    """绘制改进的关灯阶段对比图"""
    print(f"\n🎨 绘制改进的关灯阶段对比图...")
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('改进的热力学约束MLP模型 vs 纯热力学模型对比（关灯阶段）', fontsize=16, fontweight='bold')
    
    # 颜色方案
    solar_values = sorted(predictions.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(solar_values)))
    color_map = {solar: colors[i] for i, solar in enumerate(solar_values)}
    
    # 左上：热力学约束MLP预测
    ax1 = axes[0, 0]
    ax1.set_title('改进的热力学约束MLP预测（关灯）', fontsize=14)
    ax1.set_xlabel('时间 (分钟)', fontsize=12)
    ax1.set_ylabel('温差 (°C)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 右上：纯热力学模型预测
    ax2 = axes[0, 1]
    ax2.set_title('改进的纯热力学模型预测（关灯）', fontsize=14)
    ax2.set_xlabel('时间 (分钟)', fontsize=12)
    ax2.set_ylabel('温差 (°C)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 左下：Solar 1.296详细对比
    ax3 = axes[1, 0]
    ax3.set_title('Solar 1.296 关灯详细对比', fontsize=14)
    ax3.set_xlabel('时间 (分钟)', fontsize=12)
    ax3.set_ylabel('温差 (°C)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 右下：Solar 1.438详细对比
    ax4 = axes[1, 1]
    ax4.set_title('Solar 1.438 关灯详细对比', fontsize=14)
    ax4.set_xlabel('时间 (分钟)', fontsize=12)
    ax4.set_ylabel('温差 (°C)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 绘制每个Solar值的预测
    for solar_val in solar_values:
        color = color_map[solar_val]
        pred_data = predictions[solar_val]
        
        # 绘制热力学约束MLP预测
        ax1.plot(pred_data['time'], pred_data['thermal_constrained_pred'], 
                color=color, linewidth=3, alpha=0.8,
                label=f'Solar {solar_val:.3f}')
        
        # 绘制纯热力学模型预测
        ax2.plot(pred_data['time'], pred_data['thermal_only_pred'], 
                color=color, linewidth=3, alpha=0.8,
                label=f'Solar {solar_val:.3f}')
        
        # 绘制平均真实数据
        solar_data = original_data[abs(original_data['a1_val'] - solar_val) < 0.001]
        if len(solar_data) > 0:
            # 按时间排序
            solar_data = solar_data.sort_values('time')
            ax1.scatter(solar_data['time'], solar_data['temp_diff'], 
                      alpha=0.6, s=30, color=color, 
                      label=f'平均真实数据 Solar {solar_val:.3f}')
            ax2.scatter(solar_data['time'], solar_data['temp_diff'], 
                      alpha=0.6, s=30, color=color, 
                      label=f'平均真实数据 Solar {solar_val:.3f}')
        
        # 绘制详细对比图
        ax_map = {1.296: ax3, 1.438: ax4}
        
        if solar_val in ax_map:
            ax = ax_map[solar_val]
            
            # 绘制两种预测
            ax.plot(pred_data['time'], pred_data['thermal_constrained_pred'], 
                   color='red', linewidth=3, alpha=0.8, 
                   label=f'改进热力学约束MLP Solar {solar_val:.3f}')
            ax.plot(pred_data['time'], pred_data['thermal_only_pred'], 
                   color='blue', linewidth=2, alpha=0.8, linestyle='--',
                   label=f'改进纯热力学模型 Solar {solar_val:.3f}')
            
            # 绘制平均真实数据
            if len(solar_data) > 0:
                ax.scatter(solar_data['time'], solar_data['temp_diff'], 
                         alpha=0.6, s=40, color='gray', 
                         label=f'平均真实数据 Solar {solar_val:.3f}')
            
            ax.legend(fontsize=10)
    
    # 设置图例
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = output_dir / "Improved_Thermal_Constrained_MLP_Cooling_Comparison.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"   📊 保存改进关灯对比图: {output_file}")
    plt.close()


def save_improved_cooling_results(model, predictions, original_data, output_dir):
    """保存改进的关灯阶段结果"""
    results = {
        'model_type': 'Improved Thermodynamic Constrained MLP Model - Cooling Phase',
        'analysis_date': pd.Timestamp.now().isoformat(),
        'model_info': {
            'thermal_params': {
                'K1_base': float(model.thermal_params['K1_base']),
                'tau1': float(model.thermal_params['tau1']),
                'K2_base': float(model.thermal_params['K2_base']),
                'tau2': float(model.thermal_params['tau2']),
                'alpha_solar': float(model.thermal_params['alpha_solar'])
            },
            'mlp_info': {
                'hidden_layers': str(model.mlp_model.hidden_layer_sizes),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'max_iter': 1000
            }
        },
        'predictions_summary': {},
        'performance_analysis': {}
    }
    
    # 保存预测结果
    for solar_val, pred_data in predictions.items():
        results['predictions_summary'][str(solar_val)] = {
            'solar_val': float(pred_data['solar_val']),
            'thermal_constrained_initial': float(pred_data['thermal_constrained_pred'][0]),
            'thermal_constrained_final': float(pred_data['thermal_constrained_pred'][-1]),
            'thermal_only_initial': float(pred_data['thermal_only_pred'][0]),
            'thermal_only_final': float(pred_data['thermal_only_pred'][-1])
        }
    
    # 保存性能分析
    for solar_val in sorted(predictions.keys()):
        solar_data = original_data[abs(original_data['a1_val'] - solar_val) < 0.001]
        
        if len(solar_data) > 0:
            pred_data = predictions[solar_val]
            real_temps = solar_data['temp_diff'].values
            real_times = solar_data['time'].values
            
            # 插值预测到真实数据的时间点
            thermal_constrained_interp = np.interp(real_times, pred_data['time'], pred_data['thermal_constrained_pred'])
            thermal_only_interp = np.interp(real_times, pred_data['time'], pred_data['thermal_only_pred'])
            
            thermal_constrained_r2 = pearsonr(real_temps, thermal_constrained_interp)[0] ** 2
            thermal_constrained_rmse = np.sqrt(np.mean((real_temps - thermal_constrained_interp) ** 2))
            
            thermal_only_r2 = pearsonr(real_temps, thermal_only_interp)[0] ** 2
            thermal_only_rmse = np.sqrt(np.mean((real_temps - thermal_only_interp) ** 2))
            
            results['performance_analysis'][str(solar_val)] = {
                'thermal_constrained': {
                    'r2_score': float(thermal_constrained_r2),
                    'rmse': float(thermal_constrained_rmse)
                },
                'thermal_only': {
                    'r2_score': float(thermal_only_r2),
                    'rmse': float(thermal_only_rmse)
                },
                'improvement': {
                    'r2_delta': float(thermal_constrained_r2 - thermal_only_r2),
                    'rmse_delta': float(thermal_only_rmse - thermal_constrained_rmse)
                }
            }
    
    # 保存JSON结果
    output_file = output_dir / "Improved_Thermal_Constrained_MLP_Cooling_Results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 保存改进关灯阶段结果: {output_file}")


def main():
    """主函数"""
    print("🔬 改进的热力学约束MLP模型分析（关灯阶段）")
    print("=" * 60)
    
    # 获取T6ncwg设备的CSV文件
    csv_files = list(Path("Data/clean").glob("T6ncwg_*.csv"))
    
    if not csv_files:
        print("❌ 没有找到T6ncwg设备的CSV文件")
        return
    
    print(f"📁 找到 {len(csv_files)} 个T6ncwg设备文件")
    
    # 准备按PPFD分组的平均关灯数据
    data = prepare_averaged_cooling_data(csv_files)
    
    if len(data) == 0:
        print("❌ 没有有效的关灯数据")
        return
    
    print(f"📊 准备了 {len(data)} 个平均关灯数据点")
    
    # 训练改进的热力学约束MLP模型（关灯阶段）
    model = train_improved_thermal_constrained_mlp_cooling(data)
    
    if model is None:
        print("❌ 改进的热力学约束MLP模型训练失败")
        return
    
    # 生成预测
    solar_values = [1.296, 1.418, 1.438, 1.541, 1.549]
    
    predictions = generate_improved_cooling_predictions(model, solar_values)
    
    # 创建输出目录
    plot_dir = Path("plot")
    result_dir = Path("result")
    plot_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    # 绘制对比图
    plot_improved_cooling_comparison(predictions, data, plot_dir)
    
    # 保存结果
    save_improved_cooling_results(model, predictions, data, result_dir)
    
    # 保存模型到exported_models目录
    export_dir = Path("exported_models")
    export_dir.mkdir(exist_ok=True)
    
    # 保存MLP模型
    mlp_model_path = export_dir / "cooling_mlp_model.pkl"
    with open(mlp_model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   💾 保存MLP模型: {mlp_model_path}")
    
    # 保存热力学模型参数
    thermal_model_path = export_dir / "cooling_thermal_model.json"
    thermal_model_data = {
        "model_type": "Pure Thermodynamic Model - Cooling Phase",
        "formula": "ΔT(t) = K1(a1) × exp(-t/τ1) + K2(a1) × exp(-t/τ2)",
        "solar_correction": "K1(a1) = K1_base × (1 + α_solar × (a1_val - 1.4))",
        "parameters": {
            "K1_base": float(model.thermal_params['K1_base']),
            "tau1": float(model.thermal_params['tau1']),
            "K2_base": float(model.thermal_params['K2_base']),
            "tau2": float(model.thermal_params['tau2']),
            "alpha_solar": float(model.thermal_params['alpha_solar'])
        },
        "a1_ref": 1.4
    }
    
    with open(thermal_model_path, 'w', encoding='utf-8') as f:
        json.dump(thermal_model_data, f, indent=2, ensure_ascii=False)
    print(f"   💾 保存热力学模型参数: {thermal_model_path}")
    
    print(f"\n🎉 改进的热力学约束MLP模型分析完成（关灯阶段）!")
    print(f"   训练了改进的热力学约束MLP模型（关灯阶段）")
    print(f"   生成了 {len(predictions)} 个Solar值的关灯预测曲线")
    print(f"   ✅ 使用平均数据和改进的拟合方法")


if __name__ == "__main__":
    main()
