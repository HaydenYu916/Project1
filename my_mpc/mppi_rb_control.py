#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于红蓝分离LED的MPPI控制器
同时优化红色和蓝色LED的PWM，最大化光合作用

输入控制变量：[红色PWM, 蓝色PWM]
输出：PPFD, 温度, 功耗, 红蓝比, 光合作用速率
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# 导入新的LED模型
from led_rb_control import RedBlueDataParser, RedBlueLEDModel

# 导入光合作用预测器
try:
    from pn_prediction.predict import PhotosynthesisPredictor
    PHOTOSYNTHESIS_AVAILABLE = True
except ImportError:
    print("Warning: PhotosynthesisPredictor not available. Using simple model.")
    PHOTOSYNTHESIS_AVAILABLE = False


class RedBlueLEDPlant:
    """基于红蓝分离LED的植物模型"""
    
    def __init__(self, led_data_interpolator=None, max_power=100.0, base_ambient_temp=25.0):
        self.led_model = RedBlueLEDModel(led_data_interpolator, max_power)
        self.base_ambient_temp = base_ambient_temp
        self.ambient_temp = base_ambient_temp
        self.time = 0.0
        
        # 初始化光合作用预测器
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.photo_predictor = PhotosynthesisPredictor()
                self.use_photo_model = self.photo_predictor.is_loaded
            except:
                self.use_photo_model = False
        else:
            self.use_photo_model = False
    
    def step(self, red_pwm, blue_pwm, dt=0.1):
        """单步仿真"""
        # LED物理模型
        ppfd, new_ambient_temp, power, rb_ratio = self.led_model.step(
            red_pwm, blue_pwm, self.ambient_temp, self.base_ambient_temp, dt
        )
        
        # 更新状态
        self.ambient_temp = new_ambient_temp
        self.time += dt
        
        # 计算光合作用速率
        photosynthesis_rate = self.get_photosynthesis_rate(
            ppfd, new_ambient_temp, rb_ratio
        )
        
        return ppfd, new_ambient_temp, power, rb_ratio, photosynthesis_rate
    
    def get_photosynthesis_rate(self, ppfd, temperature, rb_ratio, co2=400):
        """计算光合作用速率"""
        # 限制R:B比例在合理范围内
        rb_ratio = np.clip(rb_ratio, 0.1, 5.0)
        
        if self.use_photo_model:
            try:
                result = self.photo_predictor.predict(ppfd, co2, temperature, rb_ratio)
                # 确保结果是有效数值
                if np.isfinite(result) and result >= 0:
                    return result
                else:
                    return self.simple_photosynthesis_model(ppfd, temperature)
            except Exception as e:
                return self.simple_photosynthesis_model(ppfd, temperature)
        else:
            return self.simple_photosynthesis_model(ppfd, temperature)
    
    def simple_photosynthesis_model(self, ppfd, temperature):
        """简单光合作用模型"""
        ppfd_max = 1000
        pn_max = 25
        km = 300
        
        # 温度效应
        temp_factor = np.exp(-0.01 * (temperature - 25) ** 2)
        
        # 光响应
        pn = (pn_max * ppfd / (km + ppfd)) * temp_factor
        
        return max(0, pn)
    
    def predict(self, control_sequence, initial_temp, dt=0.1):
        """预测控制序列的结果"""
        temp = initial_temp
        results = []
        
        for controls in control_sequence:
            red_pwm, blue_pwm = controls
            ppfd, temp, power, rb_ratio = self.led_model.step(
                red_pwm, blue_pwm, temp, self.base_ambient_temp, dt
            )
            photosynthesis_rate = self.get_photosynthesis_rate(ppfd, temp, rb_ratio)
            results.append([ppfd, temp, power, rb_ratio, photosynthesis_rate])
        
        return np.array(results).T


class RedBlueMPPIController:
    """红蓝LED的MPPI控制器"""
    
    def __init__(self, plant, horizon=10, num_samples=500, dt=0.1, temperature=1.0):
        self.plant = plant
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.temperature = temperature
        
        # 成本函数权重
        self.Q_photo = 10.0  # 光合作用权重
        self.R_red = 0.001   # 红色LED控制惩罚
        self.R_blue = 0.001  # 蓝色LED控制惩罚
        self.R_power = 0.1   # 功耗惩罚
        self.R_smooth = 0.05 # 控制平滑惩罚
        
        # 约束
        self.red_pwm_min = 0.0
        self.red_pwm_max = 100.0
        self.blue_pwm_min = 0.0
        self.blue_pwm_max = 100.0
        self.temp_min = 20.0
        self.temp_max = 30.0
        
        # 控制参数
        self.pwm_std = 10.0  # PWM采样标准差
        self.prev_red_pwm = 0.0
        self.prev_blue_pwm = 0.0
        
        # 约束惩罚
        self.temp_penalty = 100000.0
        self.pwm_penalty = 1000.0
    
    def sample_control_sequences(self, mean_sequence):
        """采样控制序列"""
        # mean_sequence shape: (horizon, 2) - [red_pwm, blue_pwm]
        noise = np.random.normal(0, self.pwm_std, (self.num_samples, self.horizon, 2))
        
        # 添加噪声
        samples = mean_sequence[np.newaxis, :, :] + noise
        
        # 应用约束
        samples[:, :, 0] = np.clip(samples[:, :, 0], self.red_pwm_min, self.red_pwm_max)
        samples[:, :, 1] = np.clip(samples[:, :, 1], self.blue_pwm_min, self.blue_pwm_max)
        
        return samples
    
    def compute_cost(self, control_sequence, current_temp):
        """计算单个控制序列的成本"""
        try:
            # 预测状态
            ppfd_pred, temp_pred, power_pred, rb_pred, photo_pred = self.plant.predict(
                control_sequence, current_temp, self.dt
            )
            
            cost = 0.0
            
            # 主要目标：最大化光合作用
            for k in range(self.horizon):
                cost -= self.Q_photo * photo_pred[k]
                
                # 温度约束惩罚
                if temp_pred[k] > self.temp_max:
                    violation = temp_pred[k] - self.temp_max
                    cost += self.temp_penalty * violation**2
                if temp_pred[k] < self.temp_min:
                    violation = self.temp_min - temp_pred[k]
                    cost += self.temp_penalty * violation**2
            
            # 控制努力惩罚
            for k in range(self.horizon):
                red_pwm, blue_pwm = control_sequence[k]
                cost += self.R_red * red_pwm**2
                cost += self.R_blue * blue_pwm**2
                cost += self.R_power * power_pred[k]**2
            
            # 控制平滑性惩罚
            prev_red = self.prev_red_pwm
            prev_blue = self.prev_blue_pwm
            for k in range(self.horizon):
                red_pwm, blue_pwm = control_sequence[k]
                d_red = red_pwm - prev_red
                d_blue = blue_pwm - prev_blue
                cost += self.R_smooth * (d_red**2 + d_blue**2)
                prev_red, prev_blue = red_pwm, blue_pwm
            
            return cost
            
        except Exception:
            return 1e10
    
    def solve(self, current_temp, mean_sequence=None):
        """求解MPPI优化"""
        # 初始化均值序列
        if mean_sequence is None:
            mean_sequence = np.ones((self.horizon, 2)) * 30.0  # [red, blue] = 30%
        
        # 采样控制序列
        control_samples = self.sample_control_sequences(mean_sequence)
        
        # 计算成本
        costs = np.zeros(self.num_samples)
        for i in range(self.num_samples):
            costs[i] = self.compute_cost(control_samples[i], current_temp)
        
        # 处理无效成本
        costs = np.nan_to_num(costs, nan=1e10, posinf=1e10)
        
        # 计算权重 (softmax)
        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / self.temperature)
        weights = exp_costs / np.sum(exp_costs)
        
        # 加权平均
        optimal_sequence = np.sum(weights[:, np.newaxis, np.newaxis] * control_samples, axis=0)
        
        # 应用最终约束
        optimal_sequence[:, 0] = np.clip(optimal_sequence[:, 0], self.red_pwm_min, self.red_pwm_max)
        optimal_sequence[:, 1] = np.clip(optimal_sequence[:, 1], self.blue_pwm_min, self.blue_pwm_max)
        
        # 获取当前控制动作
        optimal_red_pwm = optimal_sequence[0, 0]
        optimal_blue_pwm = optimal_sequence[0, 1]
        
        # 更新历史
        self.prev_red_pwm = optimal_red_pwm
        self.prev_blue_pwm = optimal_blue_pwm
        
        # 返回结果
        best_cost = np.min(costs)
        return optimal_red_pwm, optimal_blue_pwm, optimal_sequence, True, best_cost


def run_rb_mppi_simulation():
    """运行红蓝LED MPPI仿真"""
    
    print("🌱 红蓝分离LED MPPI光合作用优化仿真")
    print("=" * 60)
    
    # 示例数据
    sample_data = [
        "1:1-100-9:15", "1:1-200-15:25", "1:1-300-26:51", "1:1-400-35:62", "1:1-500-43:88",
        "1:2-100-13:25", "1:2-200-26:52", "1:2-300-39:78", "1:2-400-52:105", "1:2-500-67:138",
        "1:3-100-16:35", "1:3-200-29:70", "1:3-300-43:115", "1:3-400-58:155", "1:3-500-71:200",
    ]
    
    # 解析数据并创建插值器
    parser = RedBlueDataParser()
    parser.load_data_from_list(sample_data)
    interpolator = parser.get_interpolator()
    
    # 创建植物模型
    plant = RedBlueLEDPlant(interpolator, max_power=100.0)
    
    # 创建MPPI控制器
    controller = RedBlueMPPIController(plant, horizon=8, num_samples=300)
    
    # 仿真参数
    duration = 60  # 60秒
    dt = 1.0
    steps = int(duration / dt)
    
    # 数据存储
    time_data = []
    red_pwm_data = []
    blue_pwm_data = []
    ppfd_data = []
    temp_data = []
    power_data = []
    rb_ratio_data = []
    photo_data = []
    cost_data = []
    
    # 初始化
    plant.ambient_temp = 25.0
    mean_sequence = np.ones((controller.horizon, 2)) * 25.0
    
    print(f"运行时长: {duration}秒, 步长: {dt}秒")
    print("时间 | 红PWM | 蓝PWM | PPFD | 温度 | R:B | 光合 | 功耗")
    print("-" * 70)
    
    # 仿真循环
    for k in range(steps):
        current_time = k * dt
        
        # MPPI求解
        red_pwm, blue_pwm, sequence, success, cost = controller.solve(
            plant.ambient_temp, mean_sequence
        )
        
        # 更新均值序列 (滚动窗口)
        if len(sequence) > 1:
            mean_sequence = np.vstack([sequence[1:], sequence[-1:]])
        
        # 应用控制
        ppfd, temp, power, rb_ratio, photo_rate = plant.step(red_pwm, blue_pwm, dt)
        
        # 存储数据
        time_data.append(current_time)
        red_pwm_data.append(red_pwm)
        blue_pwm_data.append(blue_pwm)
        ppfd_data.append(ppfd)
        temp_data.append(temp)
        power_data.append(power)
        rb_ratio_data.append(rb_ratio)
        photo_data.append(photo_rate)
        cost_data.append(cost)
        
        # 打印进度
        if k % 5 == 0:
            print(f"{current_time:4.0f} | {red_pwm:5.1f} | {blue_pwm:5.1f} | {ppfd:4.0f} | "
                  f"{temp:4.1f} | {rb_ratio:4.2f} | {photo_rate:4.1f} | {power:4.1f}W")
    
    print(f"\n仿真完成!")
    print(f"平均光合作用: {np.mean(photo_data):.2f} μmol/m²/s")
    print(f"总光合作用: {np.sum(photo_data):.1f} μmol/m²/s·s") 
    print(f"平均功耗: {np.mean(power_data):.1f}W")
    print(f"能效比: {np.mean(photo_data)/np.mean(power_data):.3f}")
    
    return {
        'time': time_data,
        'red_pwm': red_pwm_data,
        'blue_pwm': blue_pwm_data,
        'ppfd': ppfd_data,
        'temp': temp_data,
        'power': power_data,
        'rb_ratio': rb_ratio_data,
        'photosynthesis': photo_data,
        'cost': cost_data
    }


if __name__ == "__main__":
    results = run_rb_mppi_simulation()
