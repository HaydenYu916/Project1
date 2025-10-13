#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的自适应调参模块

可以直接集成到现有的MPPI控制脚本中，实现在线参数调整。
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np


class AdaptiveMPPITuner:
    """简化的MPPI自适应调参器"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 adaptation_period: int = 24,  # 小时
                 learning_rate: float = 0.1):
        self.log_dir = log_dir
        self.adaptation_period_hours = adaptation_period
        self.learning_rate = learning_rate
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 参数范围
        self.param_ranges = {
            'Q_photo': (10.0, 50.0),
            'Q_ref': (5.0, 50.0), 
            'R_du': (0.001, 0.1),
            'R_power': (0.001, 0.05),
            'u_std': (0.1, 0.5),
            'temperature': (0.5, 2.0)
        }
        
        # 当前参数
        self.current_params = {
            'Q_photo': 25.0,
            'Q_ref': 25.0,
            'R_du': 0.02,
            'R_power': 0.005,
            'u_std': 0.25,
            'temperature': 1.0
        }
        
        # 性能历史
        self.performance_history = []
        self.param_history = []
        
        # 加载历史数据
        self.load_history()
        
        # 上次适应时间
        self.last_adaptation_time = datetime.now()
        
        print(f"🔄 自适应调参器初始化完成")
        print(f"   适应周期: {adaptation_period}小时")
        print(f"   学习率: {learning_rate}")
        print(f"   当前参数: {self.current_params}")
    
    def load_history(self):
        """加载历史数据"""
        history_file = os.path.join(self.log_dir, "adaptive_tuning_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.performance_history = data.get('performance', [])
                    self.param_history = data.get('parameters', [])
                    if self.param_history:
                        self.current_params = self.param_history[-1]
                print(f"📂 加载了 {len(self.performance_history)} 条历史记录")
            except Exception as e:
                print(f"⚠️ 加载历史数据失败: {e}")
    
    def save_history(self):
        """保存历史数据"""
        history_file = os.path.join(self.log_dir, "adaptive_tuning_history.json")
        data = {
            'performance': self.performance_history,
            'parameters': self.param_history,
            'last_updated': datetime.now().isoformat()
        }
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存历史数据失败: {e}")
    
    def compute_performance_score(self, 
                                photosynthesis_rate: float,
                                temp_violation: float,
                                control_smoothness: float,
                                power_efficiency: float,
                                reference_tracking_error: float) -> float:
        """计算性能分数"""
        # 权重配置
        weights = {
            'photosynthesis': 1.0,      # 最大化光合作用
            'temperature': -0.5,        # 最小化温度违规
            'smoothness': -0.3,         # 最小化控制抖动
            'efficiency': -0.2,         # 最小化功率消耗
            'tracking': -0.4            # 最小化参考跟踪误差
        }
        
        # 归一化各项指标
        photo_score = photosynthesis_rate / 10.0  # 假设最大光合速率约10
        temp_score = -temp_violation / 10.0
        smooth_score = -control_smoothness / 5.0
        eff_score = -power_efficiency / 100.0
        track_score = -reference_tracking_error / 2.0
        
        # 综合分数
        score = (
            weights['photosynthesis'] * photo_score +
            weights['temperature'] * temp_score +
            weights['smoothness'] * smooth_score +
            weights['efficiency'] * eff_score +
            weights['tracking'] * track_score
        )
        
        return score
    
    def record_performance(self, 
                          photosynthesis_rate: float,
                          temp_violation: float = 0.0,
                          control_change: float = 0.0,
                          power: float = 0.0,
                          ref_error: float = 0.0):
        """记录性能指标"""
        score = self.compute_performance_score(
            photosynthesis_rate=photosynthesis_rate,
            temp_violation=temp_violation,
            control_smoothness=abs(control_change),
            power_efficiency=power,
            reference_tracking_error=abs(ref_error)
        )
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'photosynthesis_rate': photosynthesis_rate,
            'temp_violation': temp_violation,
            'control_change': control_change,
            'power': power,
            'ref_error': ref_error,
            'params': self.current_params.copy()
        }
        
        self.performance_history.append(record)
        
        # 保持历史记录数量在合理范围内
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
        
        return score
    
    def should_adapt(self) -> bool:
        """判断是否应该进行参数适应"""
        now = datetime.now()
        time_since_last = (now - self.last_adaptation_time).total_seconds() / 3600  # 转换为小时
        
        # 检查时间间隔
        if time_since_last < self.adaptation_period_hours:
            return False
        
        # 检查是否有足够的性能数据
        recent_records = [
            r for r in self.performance_history 
            if datetime.fromisoformat(r['timestamp']) > self.last_adaptation_time
        ]
        
        if len(recent_records) < 10:  # 至少需要10条记录
            return False
        
        return True
    
    def adapt_parameters(self) -> bool:
        """执行参数适应"""
        if not self.should_adapt():
            return False
        
        print(f"🔄 开始参数适应...")
        
        # 获取最近的性能数据
        recent_records = self.performance_history[-50:]  # 最近50条记录
        if not recent_records:
            return False
        
        current_score = np.mean([r['score'] for r in recent_records[-10:]])
        
        # 尝试小幅调整各个参数
        best_params = self.current_params.copy()
        best_score = current_score
        improved = False
        
        for param_name in self.param_ranges.keys():
            current_val = self.current_params[param_name]
            min_val, max_val = self.param_ranges[param_name]
            
            # 生成小幅扰动
            range_size = max_val - min_val
            perturbation = self.learning_rate * range_size * np.random.uniform(-1, 1)
            new_val = np.clip(current_val + perturbation, min_val, max_val)
            
            # 创建测试参数
            test_params = self.current_params.copy()
            test_params[param_name] = new_val
            
            # 模拟评估（基于历史趋势）
            estimated_score = self._estimate_score(test_params, recent_records)
            
            if estimated_score > best_score:
                best_params[param_name] = new_val
                best_score = estimated_score
                improved = True
                print(f"   📈 {param_name}: {current_val:.4f} → {new_val:.4f}")
        
        # 更新参数
        if improved:
            self.current_params = best_params
            self.param_history.append(self.current_params.copy())
            
            # 保持参数历史在合理范围内
            if len(self.param_history) > 100:
                self.param_history = self.param_history[-50:]
            
            print(f"✅ 参数适应完成，新分数: {best_score:.4f}")
            print(f"   新参数: {self.current_params}")
        else:
            print(f"ℹ️ 当前参数已是最优，分数: {current_score:.4f}")
        
        self.last_adaptation_time = datetime.now()
        self.save_history()
        
        return improved
    
    def _estimate_score(self, test_params: Dict[str, float], 
                       recent_records: List[Dict]) -> float:
        """基于历史趋势估计参数的性能分数"""
        # 简单的线性回归估计
        param_changes = {}
        score_changes = []
        
        for i, record in enumerate(recent_records[:-1]):
            if i + 1 < len(recent_records):
                next_record = recent_records[i + 1]
                
                # 计算参数变化
                for param_name in self.param_ranges.keys():
                    if param_name not in param_changes:
                        param_changes[param_name] = []
                    
                    current_val = record['params'].get(param_name, 0)
                    next_val = next_record['params'].get(param_name, 0)
                    param_changes[param_name].append(next_val - current_val)
                
                # 计算分数变化
                score_change = next_record['score'] - record['score']
                score_changes.append(score_change)
        
        if not score_changes:
            return 0.0
        
        # 估计新参数的分数
        estimated_score = recent_records[-1]['score']
        
        for param_name in self.param_ranges.keys():
            if param_name in param_changes and param_changes[param_name]:
                param_change = (test_params[param_name] - 
                              recent_records[-1]['params'].get(param_name, 0))
                
                # 简单的线性估计
                avg_change = np.mean(param_changes[param_name])
                avg_score_change = np.mean(score_changes)
                
                if abs(avg_change) > 1e-6:
                    sensitivity = avg_score_change / avg_change
                    estimated_score += sensitivity * param_change
        
        return estimated_score
    
    def get_current_parameters(self) -> Dict[str, float]:
        """获取当前参数"""
        return self.current_params.copy()
    
    def update_controller_parameters(self, controller) -> None:
        """更新MPPI控制器的参数"""
        params = self.get_current_parameters()
        
        # 更新权重
        controller.set_weights(
            Q_photo=params['Q_photo'],
            R_du=params['R_du'],
            R_power=params['R_power'],
            Q_ref=params['Q_ref']
        )
        
        # 更新MPPI参数
        controller.set_mppi_params(
            u_std=params['u_std'],
            temperature=params['temperature']
        )
        
        print(f"🔧 控制器参数已更新: {params}")
    
    def get_performance_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        if not self.performance_history:
            return {}
        
        recent_records = self.performance_history[-20:]  # 最近20条记录
        
        return {
            'avg_score': np.mean([r['score'] for r in recent_records]),
            'avg_photosynthesis': np.mean([r['photosynthesis_rate'] for r in recent_records]),
            'avg_temp_violation': np.mean([r['temp_violation'] for r in recent_records]),
            'avg_control_change': np.mean([abs(r['control_change']) for r in recent_records]),
            'avg_power': np.mean([r['power'] for r in recent_records]),
            'avg_ref_error': np.mean([abs(r['ref_error']) for r in recent_records]),
            'num_records': len(self.performance_history),
            'last_adaptation': self.last_adaptation_time.isoformat()
        }


def integrate_with_controller(controller, tuner: AdaptiveMPPITuner):
    """将自适应调参器集成到MPPI控制器中"""
    
    # 保存原始的solve方法
    original_solve = controller.solve
    
    def enhanced_solve(*args, **kwargs):
        # 执行原始求解
        result = original_solve(*args, **kwargs)
        
        # 提取性能指标（需要根据实际情况调整）
        if len(result) >= 4:
            optimal_u, optimal_seq, success, cost = result[:4]
            
            # 这里需要根据实际情况计算性能指标
            # 示例：假设可以从plant对象获取预测结果
            try:
                # 获取光合作用预测
                preds = controller.plant.predict(optimal_seq, args[0], dt=controller.dt)
                if len(preds) >= 4:
                    photo_pred = preds[3]
                    avg_photosynthesis = float(np.mean(photo_pred)) if len(photo_pred) > 0 else 0.0
                else:
                    avg_photosynthesis = 0.0
                
                # 记录性能
                tuner.record_performance(
                    photosynthesis_rate=avg_photosynthesis,
                    temp_violation=0.0,  # 需要根据实际情况计算
                    control_change=optimal_u - getattr(controller, 'u_prev', 0.0),
                    power=0.0,  # 需要根据实际情况计算
                    ref_error=0.0  # 需要根据实际情况计算
                )
                
                # 更新控制器参数
                controller.u_prev = optimal_u
                
            except Exception as e:
                print(f"⚠️ 性能记录失败: {e}")
        
        # 尝试参数适应
        if tuner.should_adapt():
            tuner.adapt_parameters()
            tuner.update_controller_parameters(controller)
        
        return result
    
    # 替换solve方法
    controller.solve = enhanced_solve
    
    print("✅ 自适应调参器已集成到MPPI控制器")


# 使用示例
if __name__ == "__main__":
    # 创建自适应调参器
    tuner = AdaptiveMPPITuner(
        log_dir="logs/adaptive_tuning",
        adaptation_period=1,  # 1小时适应一次
        learning_rate=0.1
    )
    
    print("自适应调参器测试完成")
