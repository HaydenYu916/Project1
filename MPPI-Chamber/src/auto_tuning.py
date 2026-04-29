#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI控制器自动调参模块

提供多种自动调参策略：
1. 基于强化学习的在线调参
2. 基于历史数据的离线调参
3. 基于贝叶斯优化的超参数搜索
4. 基于遗传算法的参数进化
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import ParameterGrid
import joblib

try:
    from .mppi_v2 import LEDMPPIController, LEDPlant
    from .led import PWMtoPowerModel
except ImportError:
    from mppi_v2 import LEDMPPIController, LEDPlant
    from led import PWMtoPowerModel


@dataclass
class MPPIParameters:
    """MPPI控制器参数配置"""
    # 代价函数权重
    Q_photo: float = 25.0
    Q_ref: float = 25.0
    R_du: float = 0.02
    R_power: float = 0.005
    
    # 约束参数
    u_min: float = 0.05
    u_max: float = 2.0
    temp_min: float = 20.0
    temp_max: float = 29.8
    
    # MPPI算法参数
    horizon: int = 6
    num_samples: int = 700
    temperature: float = 1.0
    u_std: float = 0.25
    
    # 惩罚参数
    temp_penalty: float = 1e5
    u_penalty: float = 1e3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MPPIParameters':
        return cls(**data)


class PerformanceMetrics:
    """性能指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.photosynthesis_rates = []
        self.temperature_violations = []
        self.control_smoothness = []
        self.power_efficiency = []
        self.reference_tracking_errors = []
        self.costs = []
    
    def add_measurement(self, 
                       photo_rate: float,
                       temp_violation: float,
                       control_change: float,
                       power: float,
                       ref_error: float,
                       cost: float):
        """添加一次测量"""
        self.photosynthesis_rates.append(photo_rate)
        self.temperature_violations.append(temp_violation)
        self.control_smoothness.append(abs(control_change))
        self.power_efficiency.append(power)
        self.reference_tracking_errors.append(abs(ref_error))
        self.costs.append(cost)
    
    def compute_fitness(self, weights: Dict[str, float] = None) -> float:
        """计算综合适应度分数"""
        if weights is None:
            weights = {
                'photosynthesis': 1.0,
                'temperature': -0.5,
                'smoothness': -0.3,
                'efficiency': -0.2,
                'tracking': -0.4
            }
        
        if not self.photosynthesis_rates:
            return -1e6
        
        # 归一化各项指标
        photo_score = np.mean(self.photosynthesis_rates) / 10.0  # 假设最大光合速率约10
        temp_score = -np.mean(self.temperature_violations) / 10.0  # 温度违规惩罚
        smooth_score = -np.mean(self.control_smoothness) / 5.0  # 控制平滑性
        eff_score = -np.mean(self.power_efficiency) / 100.0  # 功率效率
        track_score = -np.mean(self.reference_tracking_errors) / 2.0  # 参考跟踪
        
        fitness = (
            weights['photosynthesis'] * photo_score +
            weights['temperature'] * temp_score +
            weights['smoothness'] * smooth_score +
            weights['efficiency'] * eff_score +
            weights['tracking'] * track_score
        )
        
        return fitness


class MPPIAutoTuner:
    """MPPI控制器自动调参器"""
    
    def __init__(self, 
                 plant: LEDPlant,
                 log_dir: str = "logs",
                 config_file: str = "auto_tuning_config.json"):
        self.plant = plant
        self.log_dir = log_dir
        self.config_file = config_file
        self.current_params = MPPIParameters()
        self.performance_metrics = PerformanceMetrics()
        self.history = []
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            filename=os.path.join(log_dir, "auto_tuning.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.load_config()
    
    def load_config(self):
        """加载调参配置"""
        config_path = os.path.join(self.log_dir, self.config_file)
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # 默认配置
            self.config = {
                "tuning_method": "bayesian",  # "bayesian", "genetic", "grid", "online"
                "evaluation_period_hours": 24,
                "parameter_ranges": {
                    "Q_photo": [10.0, 50.0],
                    "Q_ref": [5.0, 50.0],
                    "R_du": [0.001, 0.1],
                    "R_power": [0.001, 0.05],
                    "horizon": [4, 10],
                    "num_samples": [500, 1000],
                    "temperature": [0.5, 2.0],
                    "u_std": [0.1, 0.5]
                },
                "fitness_weights": {
                    "photosynthesis": 1.0,
                    "temperature": -0.5,
                    "smoothness": -0.3,
                    "efficiency": -0.2,
                    "tracking": -0.4
                },
                "max_iterations": 50,
                "convergence_threshold": 0.01
            }
            self.save_config()
    
    def save_config(self):
        """保存配置"""
        config_path = os.path.join(self.log_dir, self.config_file)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def create_controller(self, params: MPPIParameters) -> LEDMPPIController:
        """根据参数创建MPPI控制器"""
        controller = LEDMPPIController(
            plant=self.plant,
            horizon=params.horizon,
            num_samples=params.num_samples,
            dt=900.0,  # 15分钟
            temperature=params.temperature
        )
        
        controller.set_weights(
            Q_photo=params.Q_photo,
            R_du=params.R_du,
            R_power=params.R_power,
            Q_ref=params.Q_ref
        )
        
        controller.set_constraints(
            u_min=params.u_min,
            u_max=params.u_max,
            temp_min=params.temp_min,
            temp_max=params.temp_max
        )
        
        controller.set_mppi_params(
            u_std=params.u_std,
            dt=900.0
        )
        
        return controller
    
    def evaluate_parameters(self, 
                          params: MPPIParameters,
                          evaluation_duration_hours: float = 24.0,
                          target_solar_vol: float = 1.6) -> float:
        """评估参数配置的性能"""
        self.logger.info(f"开始评估参数: {params.to_dict()}")
        
        controller = self.create_controller(params)
        metrics = PerformanceMetrics()
        
        # 模拟评估过程
        start_time = time.time()
        end_time = start_time + evaluation_duration_hours * 3600
        
        current_temp = 25.0  # 初始温度
        prev_control = 0.0
        
        while time.time() < end_time:
            try:
                # 生成参考序列
                solar_vol_ref = np.full(controller.horizon, target_solar_vol, dtype=float)
                
                # MPPI求解
                optimal_u, optimal_seq, success, cost, _ = controller.solve(
                    current_temp=current_temp,
                    solar_vol_ref_seq=solar_vol_ref
                )
                
                if not success:
                    self.logger.warning("MPPI求解失败")
                    continue
                
                # 预测系统响应
                r_pwm, b_pwm = self.plant._solar_vol_to_pwm(optimal_u)
                preds = self.plant.predict(optimal_seq, current_temp, dt=900.0)
                (_sv_series, temp_pred, power_pred, pn_pred, _r_series, _b_series) = preds
                
                next_temp = float(temp_pred[0]) if len(temp_pred) else current_temp
                next_power = float(power_pred[0]) if len(power_pred) else 0.0
                next_pn = float(pn_pred[0]) if len(pn_pred) else 0.0
                
                # 计算指标
                temp_violation = max(0, next_temp - params.temp_max) + max(0, params.temp_min - next_temp)
                control_change = optimal_u - prev_control
                ref_error = optimal_u - target_solar_vol
                
                # 记录指标
                metrics.add_measurement(
                    photo_rate=next_pn,
                    temp_violation=temp_violation,
                    control_change=control_change,
                    power=next_power,
                    ref_error=ref_error,
                    cost=cost
                )
                
                # 更新状态
                current_temp = next_temp
                prev_control = optimal_u
                
                # 等待下一个控制周期
                time.sleep(900)  # 15分钟
                
            except Exception as e:
                self.logger.error(f"评估过程中出错: {e}")
                continue
        
        # 计算适应度
        fitness = metrics.compute_fitness(self.config["fitness_weights"])
        self.logger.info(f"参数评估完成，适应度: {fitness:.4f}")
        
        return fitness
    
    def bayesian_optimization(self, n_iterations: int = 20) -> MPPIParameters:
        """基于贝叶斯优化的参数搜索"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
            
            # 定义参数空间
            dimensions = [
                Real(self.config["parameter_ranges"]["Q_photo"][0], 
                     self.config["parameter_ranges"]["Q_photo"][1], name='Q_photo'),
                Real(self.config["parameter_ranges"]["Q_ref"][0], 
                     self.config["parameter_ranges"]["Q_ref"][1], name='Q_ref'),
                Real(self.config["parameter_ranges"]["R_du"][0], 
                     self.config["parameter_ranges"]["R_du"][1], name='R_du'),
                Real(self.config["parameter_ranges"]["R_power"][0], 
                     self.config["parameter_ranges"]["R_power"][1], name='R_power'),
                Integer(self.config["parameter_ranges"]["horizon"][0], 
                        self.config["parameter_ranges"]["horizon"][1], name='horizon'),
                Integer(self.config["parameter_ranges"]["num_samples"][0], 
                        self.config["parameter_ranges"]["num_samples"][1], name='num_samples'),
                Real(self.config["parameter_ranges"]["temperature"][0], 
                     self.config["parameter_ranges"]["temperature"][1], name='temperature'),
                Real(self.config["parameter_ranges"]["u_std"][0], 
                     self.config["parameter_ranges"]["u_std"][1], name='u_std'),
            ]
            
            @use_named_args(dimensions=dimensions)
            def objective(**params):
                param_obj = MPPIParameters(**params)
                return -self.evaluate_parameters(param_obj)  # 负号因为要最大化
            
            # 贝叶斯优化
            result = gp_minimize(func=objective,
                               dimensions=dimensions,
                               n_calls=n_iterations,
                               random_state=42)
            
            # 提取最优参数
            optimal_params = MPPIParameters(**{dim.name: result.x[i] for i, dim in enumerate(dimensions)})
            
            self.logger.info(f"贝叶斯优化完成，最优适应度: {-result.fun:.4f}")
            return optimal_params
            
        except ImportError:
            self.logger.warning("skopt未安装，使用差分进化替代")
            return self.differential_evolution_tuning()
    
    def differential_evolution_tuning(self, maxiter: int = 20) -> MPPIParameters:
        """基于差分进化的参数优化"""
        ranges = self.config["parameter_ranges"]
        
        def objective(params):
            param_dict = {
                'Q_photo': params[0],
                'Q_ref': params[1], 
                'R_du': params[2],
                'R_power': params[3],
                'horizon': int(params[4]),
                'num_samples': int(params[5]),
                'temperature': params[6],
                'u_std': params[7]
            }
            
            param_obj = MPPIParameters(**param_dict)
            return -self.evaluate_parameters(param_obj)  # 负号因为要最大化
        
        bounds = [
            ranges["Q_photo"],
            ranges["Q_ref"],
            ranges["R_du"], 
            ranges["R_power"],
            ranges["horizon"],
            ranges["num_samples"],
            ranges["temperature"],
            ranges["u_std"]
        ]
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=maxiter, 
            seed=42,
            popsize=15
        )
        
        optimal_params = MPPIParameters(
            Q_photo=result.x[0],
            Q_ref=result.x[1],
            R_du=result.x[2],
            R_power=result.x[3],
            horizon=int(result.x[4]),
            num_samples=int(result.x[5]),
            temperature=result.x[6],
            u_std=result.x[7]
        )
        
        self.logger.info(f"差分进化优化完成，最优适应度: {-result.fun:.4f}")
        return optimal_params
    
    def grid_search_tuning(self, grid_size: int = 3) -> MPPIParameters:
        """基于网格搜索的参数优化"""
        ranges = self.config["parameter_ranges"]
        
        # 为每个参数生成候选值
        param_grid = {}
        for param, (min_val, max_val) in ranges.items():
            if param in ["horizon", "num_samples"]:
                param_grid[param] = list(np.linspace(min_val, max_val, grid_size, dtype=int))
            else:
                param_grid[param] = list(np.linspace(min_val, max_val, grid_size))
        
        best_fitness = -1e6
        best_params = None
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        self.logger.info(f"网格搜索: 总共 {total_combinations} 种参数组合")
        
        for i, params_dict in enumerate(ParameterGrid(param_grid)):
            param_obj = MPPIParameters(**params_dict)
            fitness = self.evaluate_parameters(param_obj)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = param_obj
            
            self.logger.info(f"进度: {i+1}/{total_combinations}, 当前适应度: {fitness:.4f}")
        
        self.logger.info(f"网格搜索完成，最优适应度: {best_fitness:.4f}")
        return best_params
    
    def online_adaptive_tuning(self, 
                             adaptation_period_hours: float = 24.0,
                             learning_rate: float = 0.1) -> None:
        """在线自适应调参"""
        self.logger.info("开始在线自适应调参")
        
        while True:
            try:
                # 收集当前参数的性能数据
                current_fitness = self.evaluate_parameters(
                    self.current_params, 
                    evaluation_duration_hours=adaptation_period_hours
                )
                
                # 保存历史记录
                self.history.append({
                    'timestamp': datetime.now().isoformat(),
                    'params': self.current_params.to_dict(),
                    'fitness': current_fitness
                })
                
                # 简单的参数扰动和评估
                best_params = self.current_params
                best_fitness = current_fitness
                
                # 尝试小幅调整各个参数
                for param_name in ['Q_photo', 'Q_ref', 'R_du', 'R_power']:
                    current_val = getattr(self.current_params, param_name)
                    range_size = (self.config["parameter_ranges"][param_name][1] - 
                                self.config["parameter_ranges"][param_name][0])
                    
                    # 生成扰动
                    perturbation = learning_rate * range_size * np.random.uniform(-1, 1)
                    new_val = np.clip(current_val + perturbation, 
                                    self.config["parameter_ranges"][param_name][0],
                                    self.config["parameter_ranges"][param_name][1])
                    
                    # 创建新参数配置
                    new_params = MPPIParameters(**self.current_params.to_dict())
                    setattr(new_params, param_name, new_val)
                    
                    # 快速评估
                    fitness = self.evaluate_parameters(new_params, evaluation_duration_hours=1.0)
                    
                    if fitness > best_fitness:
                        best_params = new_params
                        best_fitness = fitness
                
                # 更新参数
                if best_fitness > current_fitness:
                    self.current_params = best_params
                    self.logger.info(f"参数已更新，新适应度: {best_fitness:.4f}")
                
                # 保存结果
                self.save_results()
                
                # 等待下一个适应周期
                time.sleep(adaptation_period_hours * 3600)
                
            except KeyboardInterrupt:
                self.logger.info("在线调参被用户中断")
                break
            except Exception as e:
                self.logger.error(f"在线调参出错: {e}")
                time.sleep(3600)  # 等待1小时后重试
    
    def save_results(self):
        """保存调参结果"""
        results = {
            'current_parameters': self.current_params.to_dict(),
            'performance_history': self.history,
            'config': self.config,
            'last_updated': datetime.now().isoformat()
        }
        
        results_file = os.path.join(self.log_dir, "auto_tuning_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def load_results(self) -> Dict[str, Any]:
        """加载调参结果"""
        results_file = os.path.join(self.log_dir, "auto_tuning_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def run_tuning(self, method: str = None) -> MPPIParameters:
        """运行自动调参"""
        if method is None:
            method = self.config["tuning_method"]
        
        self.logger.info(f"开始自动调参，方法: {method}")
        
        if method == "bayesian":
            optimal_params = self.bayesian_optimization(self.config["max_iterations"])
        elif method == "genetic":
            optimal_params = self.differential_evolution_tuning(self.config["max_iterations"])
        elif method == "grid":
            optimal_params = self.grid_search_tuning()
        elif method == "online":
            self.online_adaptive_tuning()
            return self.current_params
        else:
            raise ValueError(f"不支持的调参方法: {method}")
        
        # 保存结果
        self.current_params = optimal_params
        self.save_results()
        
        self.logger.info(f"自动调参完成，最优参数: {optimal_params.to_dict()}")
        return optimal_params


def main():
    """主函数 - 演示自动调参的使用"""
    # 这里需要实际的LEDPlant实例
    # plant = LEDPlant(...)  # 需要根据实际情况初始化
    
    # tuner = MPPIAutoTuner(plant)
    # optimal_params = tuner.run_tuning("bayesian")
    # print(f"最优参数: {optimal_params.to_dict()}")
    pass


if __name__ == "__main__":
    main()
