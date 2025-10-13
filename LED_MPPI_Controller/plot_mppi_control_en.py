#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot MPPI control log time series
X-axis: Time
Y-axis: r_pwm + b_pwm sum
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def plot_mppi_control_data():
    """Plot MPPI control data time series"""
    
    # Read data
    file_path = '/home/pi/Desktop/LED_MPPI_Controller/logs/mppi_v2_control_log.csv'
    df = pd.read_csv(file_path)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate r_pwm + b_pwm sum
    df['total_pwm'] = df['r_pwm'] + df['b_pwm']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot time series
    ax.plot(df['timestamp'], df['total_pwm'], 'b-', linewidth=2, label='r_pwm + b_pwm Sum', alpha=0.8)
    
    # Add data points
    ax.scatter(df['timestamp'], df['total_pwm'], c='red', s=30, alpha=0.7, zorder=5, label='Data Points')
    
    # Set labels and title
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('r_pwm + b_pwm Sum', fontsize=14, fontweight='bold')
    ax.set_title('MPPI Control Log - PWM Sum Time Series', fontsize=16, fontweight='bold')
    
    # Format time axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.grid(True, alpha=0.1, which='minor')
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Calculate statistics
    total_mean = df['total_pwm'].mean()
    total_std = df['total_pwm'].std()
    total_max = df['total_pwm'].max()
    total_min = df['total_pwm'].min()
    total_median = df['total_pwm'].median()
    
    # Add statistics text box
    stats_text = f'Statistics:\nMean: {total_mean:.2f}\nStd: {total_std:.2f}\nMax: {total_max:.2f}\nMin: {total_min:.2f}\nMedian: {total_median:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Add horizontal line for mean
    ax.axhline(y=total_mean, color='green', linestyle='--', alpha=0.7, label=f'Mean: {total_mean:.2f}')
    
    # Improve layout
    plt.tight_layout()
    
    # Save image
    output_path = '/home/pi/Desktop/LED_MPPI_Controller/mppi_pwm_sum_timeseries_en.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {output_path}")
    
    # Print data overview
    print(f"\nData Overview:")
    print(f"Number of data points: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"PWM Sum Statistics:")
    print(f"  Mean: {total_mean:.2f}")
    print(f"  Standard deviation: {total_std:.2f}")
    print(f"  Maximum: {total_max:.2f}")
    print(f"  Minimum: {total_min:.2f}")
    print(f"  Median: {total_median:.2f}")
    
    # Show some sample data
    print(f"\nSample data (first 5 rows):")
    print(df[['timestamp', 'r_pwm', 'b_pwm', 'total_pwm']].head())

if __name__ == "__main__":
    plot_mppi_control_data()

