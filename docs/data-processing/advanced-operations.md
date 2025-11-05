---
title: Advanced Pandas Operations
---

# Advanced Pandas Operations

This document covers sophisticated pandas operations for complex data analysis tasks, including time series analysis, groupby operations, window functions, and advanced transformations.

## Time Series Analysis

### Setting Up Time-Based Index

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load racing telemetry data
df = pd.read_csv('data/telemetry_detailed.csv')

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['time_s'], unit='s')

# Set datetime as index for time series operations
df_ts = df.set_index('datetime')

# Alternative: Load telemetry with time index
df_ts = pd.read_csv('data/telemetry_detailed.csv', 
                   index_col='time_s')

print(f"Time range: {df_ts.index.min()} to {df_ts.index.max()}")
print(f"Frequency: {pd.infer_freq(df_ts.index)}")
```

### Resampling and Frequency Conversion

```python
# Resample to different frequencies
df_1s = df_ts.resample('1s').mean()      # 1 Hz (1-second intervals)
df_100ms = df_ts.resample('100ms').mean()  # 10 Hz
df_10ms = df_ts.resample('10ms').mean()    # 100 Hz

# Different aggregation methods per column
df_resampled = df_ts.resample('1s').agg({
    'speed': 'mean',
    'ax': 'mean',
    'ay': 'mean',
    'az': 'mean',
    'steering_angle': 'mean',
    'brake_pressure': 'max',
    'throttle_position': 'mean'
})

# Upsampling with interpolation
df_upsampled = df_ts.resample('1ms').interpolate(method='linear')

# Handle irregular time series
df_regular = df_ts.asfreq('10ms', method='ffill')  # Forward fill gaps

# Custom resampling function
def rms_resample(series):
    """Calculate RMS (Root Mean Square) for resampling"""
    return np.sqrt(np.mean(series**2))

df_rms = df_ts.resample('1s').agg({
    'ax': rms_resample,
    'ay': rms_resample,
    'az': rms_resample
})
```

### Rolling Window Operations

```python
# Simple rolling averages
window_size = 100  # 100 samples
df_ts['speed_ma'] = df_ts['speed'].rolling(window=window_size).mean()
df_ts['speed_std'] = df_ts['speed'].rolling(window=window_size).std()

# Time-based windows
df_ts['speed_ma_1s'] = df_ts['speed'].rolling('1s').mean()
df_ts['lateral_g_rms'] = df_ts['lateral_g'].rolling('1s').apply(lambda x: np.sqrt(np.mean(x**2)))

# Multiple statistics in one operation
rolling_stats = df_ts['speed'].rolling('5s').agg({
    'mean': 'mean',
    'std': 'std',
    'min': 'min',
    'max': 'max',
    'range': lambda x: x.max() - x.min()
})

# Centered windows (look ahead and behind)
df_ts['speed_centered'] = df_ts['speed'].rolling(window=50, center=True).mean()

# Custom rolling functions
def rolling_percentile(series, percentile=95):
    return series.quantile(percentile/100)

df_ts['speed_95th'] = df_ts['speed'].rolling('10s').apply(
    lambda x: rolling_percentile(x, 95)
)

# Exponentially weighted moving average
df_ts['speed_ewm'] = df_ts['speed'].ewm(span=50).mean()
df_ts['speed_ewm_var'] = df_ts['speed'].ewm(span=50).var()
```

### Expanding Window Operations

```python
# Cumulative statistics (expanding window)
df_ts['session_avg_speed'] = df_ts['speed'].expanding().mean()
df_ts['session_max_speed'] = df_ts['speed'].expanding().max()
df_ts['session_std_speed'] = df_ts['speed'].expanding().std()

# Session-long performance tracking
df_ts['best_lap_time_so_far'] = df_ts['lap_time'].expanding().min()
df_ts['consistency_metric'] = df_ts['lap_time'].expanding().std()

# Custom expanding functions
def expanding_efficiency(speed_series):
    """Calculate fuel efficiency trend"""
    return speed_series.mean() / (speed_series.std() + 1e-6)

df_ts['efficiency_trend'] = df_ts['speed'].expanding().apply(expanding_efficiency)
```

## GroupBy Operations

### Basic Grouping Concepts

```python
# Assume we have lap numbers in our data
df['lap_number'] = (df['distance'] // track_length).astype(int)

# Basic groupby operations
lap_stats = df.groupby('lap_number').agg({
    'speed': ['mean', 'max', 'std'],
    'lateral_g': ['max', 'mean'],
    'corner_radius': 'min',
    'timestamp': ['min', 'max']  # Start and end times
})

# Flatten multi-level column names
lap_stats.columns = ['_'.join(col).strip() for col in lap_stats.columns]

# Calculate lap times
lap_stats['lap_time'] = lap_stats['timestamp_max'] - lap_stats['timestamp_min']

print("Lap Statistics:")
print(lap_stats.head())
```

### Advanced Grouping Strategies

```python
# Multiple grouping variables
# Group by lap and track sector
sector_boundaries = [0, 1500, 3000, 4500]  # meters

def assign_sector(distance):
    for i, boundary in enumerate(sector_boundaries[1:]):
        if distance <= boundary:
            return i + 1
    return len(sector_boundaries) - 1

df['sector'] = df['distance'].apply(assign_sector)

# Group by lap and sector
lap_sector_analysis = df.groupby(['lap_number', 'sector']).agg({
    'speed': 'mean',
    'lateral_g': 'max',
    'timestamp': lambda x: x.max() - x.min()  # Sector time
}).rename(columns={'timestamp': 'sector_time'})

# Pivot for easy comparison
sector_times = lap_sector_analysis['sector_time'].unstack(level='sector')
print("Sector times by lap:")
print(sector_times.head())
```

### Filter Operations

```python
# Filter: return entire groups that meet criteria
# Keep only laps with average speed above threshold
fast_laps = df.groupby('lap_number').filter(lambda x: x['speed'].mean() > 85)

# Keep only complete laps (sufficient data points)
complete_laps = df.groupby('lap_number').filter(lambda x: len(x) > 1000)

# Keep laps with good data quality (no large gaps)
def has_good_data_quality(group):
    time_diffs = group['timestamp'].diff()
    max_gap = time_diffs.max()
    return max_gap < 0.1  # Less than 100ms gaps

quality_laps = df.groupby('lap_number').filter(has_good_data_quality)

# Combine filters
clean_fast_laps = df.groupby('lap_number').filter(
    lambda x: (x['speed'].mean() > 85) and (len(x) > 1000) and has_good_data_quality(x)
)
```

## Window Functions

### Ranking Operations

```python
# Rank within entire dataset
df['speed_rank_global'] = df['speed'].rank(method='dense', ascending=False)

# Rank within each lap
df['speed_rank_lap'] = df.groupby('lap_number')['speed'].rank(method='dense', ascending=False)

# Percentile ranks
df['speed_percentile'] = df['speed'].rank(pct=True)  # 0-1 range
df['speed_percentile_lap'] = df.groupby('lap_number')['speed'].rank(pct=True)

# Multiple ranking criteria
df['combined_rank'] = df.eval('speed_rank_global + lateral_g_rank_global')
```

### Shift Operations (Lead/Lag)

```python
# Look ahead and behind
df['speed_next'] = df['speed'].shift(-1)     # Next sample
df['speed_prev'] = df['speed'].shift(1)      # Previous sample

# Calculate rates of change
df['speed_change'] = df['speed'].diff()      # Change from previous
df['acceleration_calc'] = df['speed_change'] / df['timestamp'].diff()

# Look ahead multiple steps
df['speed_5_ahead'] = df['speed'].shift(-5)
df['approaching_corner'] = (df['corner_radius'].shift(-10) < 50)  # Corner ahead

# Group-aware shifts
df['prev_lap_speed'] = df.groupby('sector')['speed'].shift(1)  # Previous lap, same sector

# Custom shift operations
def shift_to_lap_start(group):
    """Shift relative to lap start"""
    return group.shift(len(group) - 1)  # Value at lap start

df['lap_start_speed'] = df.groupby('lap_number')['speed'].transform(shift_to_lap_start)
```

### Cumulative Operations

```python
# Basic cumulative operations
df['cumulative_distance'] = df['distance'].cumsum()
df['session_max_speed'] = df['speed'].cummax()
df['session_min_corner_radius'] = df['corner_radius'].cummin()

# Group-aware cumulative operations
df['distance_in_lap'] = df.groupby('lap_number')['distance'].cumsum()
df['lap_max_speed'] = df.groupby('lap_number')['speed'].cummax()

# Custom cumulative functions
def cumulative_tire_wear(speed_series, lateral_g_series):
    """Calculate cumulative tire wear proxy"""
    wear_rate = speed_series * lateral_g_series.abs()
    return wear_rate.cumsum()

df['tire_wear_estimate'] = cumulative_tire_wear(df['speed'], df['lateral_g'])

# Conditional cumulative operations
df['overspeed_count'] = (df['speed'] > 120).cumsum()  # Count overspeeds
df['corner_count'] = (df['steering_angle'].abs() > 10).cumsum()  # Count corners
```

## Advanced Data Transformations

### Categorical Data Operations

```python
# Create performance categories
df['speed_category'] = pd.cut(df['speed'], 
                             bins=[0, 50, 100, 150, 200, 300],
                             labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Cornering intensity classification
df['cornering_intensity'] = pd.cut(df['lateral_g'].abs(),
                                  bins=[0, 0.3, 0.8, 1.2, 2.0],
                                  labels=['Straight', 'Light', 'Moderate', 'Hard'])

# Custom categorization
def categorize_driving_style(row):
    if row['speed'] > 150 and row['lateral_g'] > 1.0:
        return 'Aggressive'
    elif row['speed'] > 100 and row['lateral_g'] > 0.5:
        return 'Sporty'
    elif row['speed'] < 60:
        return 'Conservative'
    else:
        return 'Normal'

df['driving_style'] = df.apply(categorize_driving_style, axis=1)

# Analyze by categories
category_analysis = df.groupby(['cornering_intensity', 'speed_category']).size()
style_performance = df.groupby('driving_style').agg({
    'speed': 'mean',
    'lateral_g': 'max',
    'corner_radius': 'min'
})
```

### Pivot Tables and Cross-Tabulations

```python
# Create pivot table for lap vs sector analysis
pivot_sector_times = df.pivot_table(
    values='timestamp',
    index='lap_number',
    columns='sector',
    aggfunc=lambda x: x.max() - x.min(),  # Time spent in sector
    fill_value=0
)

# Multi-level pivot
pivot_detailed = df.pivot_table(
    values=['speed', 'lateral_g'],
    index='lap_number',
    columns=['sector', 'speed_category'],
    aggfunc={'speed': 'mean', 'lateral_g': 'max'}
)

# Cross-tabulation
crosstab_style_performance = pd.crosstab(
    df['driving_style'],
    df['cornering_intensity'],
    values=df['speed'],
    aggfunc='mean'
)

print("Average speed by driving style and cornering intensity:")
print(crosstab_style_performance)
```

### Melt and Wide-to-Long Transformations

```python
# Melt acceleration data for analysis
acceleration_data = df[['timestamp', 'ax', 'ay', 'az']].copy()

# Wide to long format
accel_melted = pd.melt(acceleration_data,
                      id_vars=['timestamp'],
                      value_vars=['ax', 'ay', 'az'],
                      var_name='axis',
                      value_name='acceleration')

# Analysis on melted data
accel_stats = accel_melted.groupby('axis')['acceleration'].agg([
    'mean', 'std', 'min', 'max'
])

# Multiple value columns
sensor_data = df[['timestamp', 'lap_number', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']].copy()

sensor_melted = pd.melt(sensor_data,
                       id_vars=['timestamp', 'lap_number'],
                       value_vars=['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
                       var_name='sensor',
                       value_name='value')

# Add sensor type column
sensor_melted['sensor_type'] = sensor_melted['sensor'].apply(
    lambda x: 'acceleration' if x.startswith('a') else 'gyroscope'
)
```

### Advanced Merging and Joining

```python
# Create separate dataframes for different data types
telemetry_df = df[['timestamp', 'speed', 'ax', 'ay', 'az']].copy()
gps_df = df[['timestamp', 'latitude', 'longitude', 'altitude']].copy()
driver_inputs_df = df[['timestamp', 'steering_angle', 'throttle', 'brake']].copy()

# Merge with tolerance for time synchronization
merged_data = pd.merge_asof(
    telemetry_df.sort_values('timestamp'),
    gps_df.sort_values('timestamp'),
    on='timestamp',
    tolerance=0.01,  # 10ms tolerance
    direction='nearest'
)

# Multiple merges
complete_data = merged_data.merge(
    driver_inputs_df,
    on='timestamp',
    how='outer'
)

# Merge with different keys
lap_summary = df.groupby('lap_number').agg({
    'speed': 'mean',
    'timestamp': lambda x: x.max() - x.min()
}).rename(columns={'timestamp': 'lap_time'})

# Add lap summary to original data
df_with_lap_stats = df.merge(
    lap_summary,
    left_on='lap_number',
    right_index=True,
    suffixes=('', '_lap_avg')
)
```

## Performance Optimization

### Efficient Data Types

```python
def optimize_dataframe(df):
    """Optimize DataFrame memory usage"""
    
    # Store original memory usage
    original_usage = df.memory_usage(deep=True).sum()
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype('int8')
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df[col] = df[col].astype('int32')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize object columns
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        
        if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
            df[col] = df[col].astype('category')
    
    # Report optimization results
    optimized_usage = df.memory_usage(deep=True).sum()
    reduction = (original_usage - optimized_usage) / original_usage * 100
    
    print(f"Memory usage reduced by {reduction:.1f}%")
    print(f"Original: {original_usage/1e6:.1f} MB")
    print(f"Optimized: {optimized_usage/1e6:.1f} MB")
    
    return df

df_optimized = optimize_dataframe(df.copy())
```

### Vectorized Operations

```python
# Avoid loops - use vectorized operations
# BAD: Loop through rows
results = []
for index, row in df.iterrows():
    result = np.sqrt(row['ax']**2 + row['ay']**2 + row['az']**2)
    results.append(result)
df['magnitude_slow'] = results

# GOOD: Vectorized operation
df['magnitude_fast'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

# Complex vectorized operations
df['cornering_force'] = df['speed'] * df['lateral_g'] / 9.81
df['power_estimate'] = df['speed'] * df['ax'] * vehicle_mass / 3.6

# Conditional operations
df['performance_zone'] = np.where(
    (df['speed'] > 100) & (df['lateral_g'] > 0.8),
    'High Performance',
    np.where(
        df['speed'] > 60,
        'Normal',
        'Low Speed'
    )
)

# Multiple conditions with np.select
conditions = [
    (df['speed'] < 30),
    (df['speed'] >= 30) & (df['speed'] < 80),
    (df['speed'] >= 80) & (df['speed'] < 150),
    (df['speed'] >= 150)
]
choices = ['City', 'Suburban', 'Highway', 'Racing']
df['speed_context'] = np.select(conditions, choices, default='Unknown')
```

These advanced pandas operations provide the foundation for sophisticated data analysis workflows. They enable complex time series analysis, detailed groupby operations, and efficient data transformations essential for engineering applications.