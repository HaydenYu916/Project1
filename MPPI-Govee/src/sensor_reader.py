#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo 传感器读取类（示例用）

将温度（Riotee CSV）与 CO2（两列表 CSV）读取整合为一个简单类，
用于 applications/utils 下的演示脚本调用，避免依赖 src 包。
"""

import os
import time
import pandas as pd
from typing import Optional, Tuple


# 默认配置（按仓库目录结构推导）
DEFAULT_DEVICE_ID = "T6ncwg=="
DEFAULT_CO2_PPM = 450.0
DEFAULT_RH = 60.0
DEFAULT_PIPE_TEMP = 20.0

DEFAULT_RIOTEE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "Tool", "Sensor_riotee_server", "logs", "riotee_data_all.csv"
)


class DemoSensorReader:
    """示例用的统一传感器读取类

    - read_latest_riotee_data: 读取温度 + solar_vol（a1_raw 滤波均值）
    - read_latest_co2_data / read_latest_co2_with_timestamp: 读取 CO2（两列表 CSV: timestamp, co2）
    """

    def __init__(
        self,
        device_id: str = DEFAULT_DEVICE_ID,
        riotee_data_path: str = DEFAULT_RIOTEE_DATA_PATH,
        co2_data_path: Optional[str] = None,
    ) -> None:
        self.device_id = device_id
        self.riotee_data_path = riotee_data_path
        self.co2_data_path = co2_data_path

    # -------- 温度 / solar_vol（来源：riotee CSV） --------
    def read_latest_riotee_data(self, window_minutes: int = 10):
        try:
            if not os.path.exists(self.riotee_data_path):
                print(f"警告: 数据文件不存在: {self.riotee_data_path}")
                return None, None, None, None

            df = pd.read_csv(self.riotee_data_path, comment="#")
            device_data = df[df["device_id"] == self.device_id].copy()
            if device_data.empty:
                print(f"警告: 未找到设备ID {self.device_id} 的数据")
                return None, None, None, None

            device_data["timestamp"] = pd.to_datetime(device_data["timestamp"])  # type: ignore
            latest_time = device_data["timestamp"].max()
            window_start = latest_time - pd.Timedelta(minutes=window_minutes)
            recent = device_data[device_data["timestamp"] >= window_start]
            if recent.empty:
                print(f"警告: 过去{window_minutes}分钟内没有数据")
                return None, None, None, None

            win = min(5, len(recent))
            recent["a1_raw_filtered"] = recent["a1_raw"].rolling(window=win, center=True).mean()  # type: ignore

            temperature = float(device_data.iloc[-1]["temperature"])  # 最新温度
            solar_vol = float(recent["a1_raw_filtered"].mean())  # 滤波后的 a1_raw 视作 solar_vol
            pn_avg = None

            return temperature, solar_vol, pn_avg, latest_time
        except Exception as e:
            print(f"错误: 读取Riotee数据失败: {e}")
            return None, None, None, None

    # -------- 湿度 (Riotee CSV) --------
    def read_latest_humidity(self, window_minutes: int = 10) -> Optional[float]:
        """读取最新一条 humidity (%);失败返回 None"""
        try:
            if not os.path.exists(self.riotee_data_path):
                return None
            df = pd.read_csv(self.riotee_data_path, comment="#")
            if "humidity" not in df.columns:
                return None
            device_data = df[df["device_id"] == self.device_id].copy()
            if device_data.empty:
                return None
            device_data["timestamp"] = pd.to_datetime(device_data["timestamp"])  # type: ignore
            latest_time = device_data["timestamp"].max()
            window_start = latest_time - pd.Timedelta(minutes=window_minutes)
            recent = device_data[device_data["timestamp"] >= window_start]
            if recent.empty:
                return None
            return float(recent["humidity"].dropna().iloc[-1])
        except Exception as e:
            print(f"错误: 读取湿度失败: {e}")
            return None

    # -------- GL-Gym IndoorClimate 适配器 --------
    def read_indoor_climate(
        self,
        window_minutes: int = 10,
        pipe_temp: float = DEFAULT_PIPE_TEMP,
    ) -> "list[float]":
        """返回 GL-Gym IndoorClimateObservations 顺序的 4 元数组:
        [co2_air (ppm), temp_air (°C), rh_air (%), pipe_temp (°C)]

        缺失项以默认值/常量回填。pipe_temp 在 chamber 场景下没有硬件,
        默认置为 DEFAULT_PIPE_TEMP,可由调用方覆盖。
        """
        temp, _, _, _ = self.read_latest_riotee_data(window_minutes=window_minutes)
        rh = self.read_latest_humidity(window_minutes=window_minutes)
        co2 = self.read_latest_co2_data()
        return [
            float(co2) if co2 is not None else DEFAULT_CO2_PPM,
            float(temp) if temp is not None else 20.0,
            float(rh) if rh is not None else DEFAULT_RH,
            float(pipe_temp),
        ]

    # -------- CO2（两列表 CSV: timestamp, co2） --------
    def read_latest_co2_data(self) -> float:
        """仅返回 ppm；若失败返回默认值"""
        try:
            if not self.co2_data_path or not os.path.exists(self.co2_data_path):
                print(f"警告: CO2数据文件不存在: {self.co2_data_path}，使用默认值 {DEFAULT_CO2_PPM} ppm")
                return DEFAULT_CO2_PPM
            df = pd.read_csv(self.co2_data_path, header=None, names=["timestamp", "co2"])  # type: ignore
            if df.empty:
                print("警告: CO2数据文件为空，使用默认值")
                return DEFAULT_CO2_PPM
            latest = df.iloc[-1]
            return float(latest.get("co2"))
        except Exception as e:
            print(f"错误: 读取CO2数据失败: {e}，使用默认值 {DEFAULT_CO2_PPM} ppm")
            return DEFAULT_CO2_PPM

    def read_latest_co2_with_timestamp(self) -> Tuple[Optional[float], Optional[float]]:
        try:
            if not self.co2_data_path or not os.path.exists(self.co2_data_path):
                print(f"警告: CO2数据文件不存在: {self.co2_data_path}")
                return None, None
            df = pd.read_csv(self.co2_data_path, header=None, names=["timestamp", "co2"])  # type: ignore
            if df.empty:
                print("警告: CO2数据文件为空")
                return None, None
            latest = df.iloc[-1]
            co2 = latest.get("co2")
            ts = latest.get("timestamp")
            try:
                return float(co2), float(ts)
            except Exception:
                return None, None
        except Exception as e:
            print(f"错误: 读取CO2数据失败: {e}")
            return None, None


__all__ = [
    "DemoSensorReader",
    "DEFAULT_DEVICE_ID",
    "DEFAULT_CO2_PPM",
    "DEFAULT_RH",
    "DEFAULT_PIPE_TEMP",
    "DEFAULT_RIOTEE_DATA_PATH",
]
