---
title: Racing Data Case Study
---

# Racing Data Case Study: Vehicle Dynamics Analysis

This comprehensive case study demonstrates a complete data processing workflow using racing vehicle telemetry data. We'll analyze IMU data from a cornering maneuver to extract meaningful engineering insights.

## Dataset Overview

### Vehicle and Scenario
- **Vehicle**: European Autocross Championship car (synthetic data)
- **Scenario**: Tight autocross course section with chicane
- **Duration**: 30 seconds of data
- **Sampling Rate**: 1000 Hz
- **Maneuver**: Left-right-left chicane section typical of autocross

> **Note**: This dataset is synthetically generated to provide clean, educational examples. Later in the course, we'll work with 3D geometry data from the same championship car for visualization and analysis.

### Data Structure
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset columns
columns = [
    'timestamp',      # Time in seconds from start
    'ax', 'ay', 'az',  # Linear acceleration [m/s²] in vehicle frame
    'gx', 'gy', 'gz',  # Angular velocity [rad/s] around vehicle axes
    'speed',          # Vehicle speed [km/h]
    'steering_angle', # Steering wheel angle [degrees]
    'throttle',       # Throttle position [0-100%]
    'brake_pressure', # Brake pressure [bar]
    'distance'        # Cumulative distance [m]
]

# Load the dataset
df = pd.read_csv('racing_corner_maneuver.csv')
print(f"Dataset: {df.shape[0]} samples over {df['timestamp'].max():.1f} seconds")
print(f"Columns: {list(df.columns)}")
```

## Data Quality Assessment

### Initial Data Exploration
```python
# Basic information
print("=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print(f"Time range: {df['timestamp'].min():.3f} - {df['timestamp'].max():.3f} seconds")
print(f"Sample rate: {1/df['timestamp'].diff().mean():.0f} Hz")

# Statistical summary
print("\n=== Statistical Summary ===")
print(df.describe())

# Data types and missing values
print("\n=== Data Quality ===")
print(f"Data types:\n{df.dtypes}")
print(f"Missing values:\n{df.isnull().sum()}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
```

### Time Series Validation
```python
# Check sampling consistency
time_diffs = df['timestamp'].diff()
mean_dt = time_diffs.mean()
std_dt = time_diffs.std()

print(f"Mean sampling interval: {mean_dt:.6f} s ({1/mean_dt:.1f} Hz)")
print(f"Sampling variation: ±{std_dt:.6f} s")

# Identify timing issues
large_gaps = time_diffs[time_diffs > mean_dt * 2]
if len(large_gaps) > 0:
    print(f"Found {len(large_gaps)} timing gaps > {mean_dt*2:.6f} s")
    print("Gap locations:", large_gaps.index.tolist()[:5])

# Visualize timing consistency
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df['timestamp'], time_diffs * 1000)
plt.ylabel('Sample Interval [ms]')
plt.xlabel('Time [s]')
plt.title('Sampling Interval Over Time')

plt.subplot(1, 2, 2)
plt.hist(time_diffs * 1000, bins=50)
plt.xlabel('Sample Interval [ms]')
plt.ylabel('Count')
plt.title('Sample Interval Distribution')
plt.tight_layout()
plt.show()
```

### Physical Validation
```python
def validate_sensor_ranges(df):
    """Validate sensor readings against physical limits"""
    
    validation_rules = {
        'speed': (0, 200, 'km/h'),
        'ax': (-30, 30, 'm/s²'),
        'ay': (-30, 30, 'm/s²'),
        'az': (-30, 30, 'm/s²'),
        'gx': (-10, 10, 'rad/s'),
        'gy': (-10, 10, 'rad/s'),
        'gz': (-10, 10, 'rad/s'),
        'steering_angle': (-720, 720, 'degrees'),
        'throttle': (0, 100, '%'),
        'brake_pressure': (0, 200, 'bar')
    }
    
    validation_results = {}
    
    for column, (min_val, max_val, unit) in validation_rules.items():
        if column in df.columns:
            out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
            validation_results[column] = {
                'valid_range': f"[{min_val}, {max_val}] {unit}",
                'actual_range': f"[{df[column].min():.3f}, {df[column].max():.3f}]",
                'violations': len(out_of_range),
                'violation_percentage': len(out_of_range) / len(df) * 100
            }
    
    return validation_results

# Perform validation
validation = validate_sensor_ranges(df)

print("=== Physical Validation Results ===")
for sensor, results in validation.items():
    if results['violations'] > 0:
        print(f"{sensor}: {results['violations']} violations ({results['violation_percentage']:.2f}%)")
        print(f"  Valid: {results['valid_range']}")
        print(f"  Actual: {results['actual_range']}")
```

## Data Cleaning Pipeline

### Outlier Detection and Removal
```python
def detect_outliers_iqr(data, column, factor=1.5):
    """Detect outliers using Interquartile Range method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    outliers_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    outliers = data[outliers_mask]
    
    return outliers, lower_bound, upper_bound, outliers_mask

# Detect outliers in acceleration channels
acceleration_columns = ['ax', 'ay', 'az']
outlier_summary = {}

for column in acceleration_columns:
    outliers, lower, upper, mask = detect_outliers_iqr(df, column)
    outlier_summary[column] = {
        'count': len(outliers),
        'percentage': len(outliers) / len(df) * 100,
        'bounds': (lower, upper)
    }
    
    print(f"{column}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
    print(f"  Valid range: [{lower:.3f}, {upper:.3f}] m/s²")

# Remove outliers (conservative approach)
def remove_outliers_conservative(data, columns, factor=2.0):
    """Remove extreme outliers only"""
    clean_data = data.copy()
    
    for column in columns:
        _, _, _, outlier_mask = detect_outliers_iqr(clean_data, column, factor)
        clean_data = clean_data[~outlier_mask]
    
    return clean_data

df_clean = remove_outliers_conservative(df, acceleration_columns)
print(f"\nData after outlier removal: {len(df_clean)} samples ({len(df_clean)/len(df)*100:.1f}% retained)")
```

### Signal Filtering
```python
# Apply moving average filter to reduce noise
def apply_moving_average_filter(data, columns, window_size=10):
    """Apply moving average filter to specified columns"""
    filtered_data = data.copy()
    
    for column in columns:
        # Keep original data
        filtered_data[f'{column}_raw'] = filtered_data[column]
        
        # Apply moving average
        filtered_data[column] = filtered_data[column].rolling(
            window=window_size, center=True
        ).mean()
        
        # Fill NaN values at edges
        filtered_data[column] = filtered_data[column].fillna(method='bfill').fillna(method='ffill')
    
    return filtered_data

# Apply filtering to sensor data
sensor_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
df_filtered = apply_moving_average_filter(df_clean, sensor_columns, window_size=10)

# Visualize filtering effect
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, column in enumerate(['ax', 'ay', 'gx', 'gy']):
    if i < 4:
        axes[i].plot(df_filtered['timestamp'], df_filtered[f'{column}_raw'], 
                    alpha=0.3, label='Raw', color='gray')
        axes[i].plot(df_filtered['timestamp'], df_filtered[column], 
                    label='Filtered', color='blue')
        axes[i].set_ylabel(f'{column} [{"m/s²" if column.startswith("a") else "rad/s"}]')
        axes[i].set_xlabel('Time [s]')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Signal Filtering Effect', y=1.02)
plt.show()
```

## Coordinate System Transformations

### Gravity Compensation
```python
# Estimate gravity vector from stationary period
def estimate_gravity(data, stationary_duration=2.0):
    """Estimate gravity from initial stationary period"""
    stationary_mask = data['timestamp'] <= stationary_duration
    stationary_data = data[stationary_mask]
    
    # Estimate gravity components
    gravity_estimate = {
        'gx': stationary_data['ax'].mean(),
        'gy': stationary_data['ay'].mean(),  # Typically ~9.81 m/s²
        'gz': stationary_data['az'].mean()
    }
    
    gravity_magnitude = np.sqrt(sum(g**2 for g in gravity_estimate.values()))
    
    print(f"Estimated gravity components:")
    for axis, g in gravity_estimate.items():
        print(f"  {axis}: {g:.3f} m/s²")
    print(f"Gravity magnitude: {gravity_magnitude:.3f} m/s² (expected: ~9.81)")
    
    return gravity_estimate

gravity = estimate_gravity(df_filtered)

# Apply gravity compensation
df_compensated = df_filtered.copy()
df_compensated['ax_body'] = df_compensated['ax'] - gravity['gx']
df_compensated['ay_body'] = df_compensated['ay'] - gravity['gy']
df_compensated['az_body'] = df_compensated['az'] - gravity['gz']

# Convert to g-forces for engineering interpretation
df_compensated['longitudinal_g'] = df_compensated['ax_body'] / 9.81
df_compensated['lateral_g'] = df_compensated['ay_body'] / 9.81
df_compensated['vertical_g'] = df_compensated['az_body'] / 9.81

print("Applied gravity compensation and converted to g-forces")
```

### Vehicle Motion Calculations
```python
# Numerical integration for velocity and position
def numerical_integration(data):
    """Calculate velocity and position from acceleration"""
    result = data.copy()
    
    # Calculate time differences
    dt = result['timestamp'].diff().fillna(0)
    
    # Integrate acceleration to velocity (body frame)
    result['vx'] = np.cumsum(result['ax_body'] * dt)
    result['vy'] = np.cumsum(result['ay_body'] * dt)
    result['vz'] = np.cumsum(result['az_body'] * dt)
    
    # Integrate velocity to position (body frame)
    result['pos_x'] = np.cumsum(result['vx'] * dt)
    result['pos_y'] = np.cumsum(result['vy'] * dt)
    result['pos_z'] = np.cumsum(result['vz'] * dt)
    
    # Calculate total distance traveled
    result['distance_calc'] = np.cumsum(np.sqrt(
        (result['vx'] * dt)**2 + (result['vy'] * dt)**2
    ))
    
    return result

df_motion = numerical_integration(df_compensated)

# Calculate derived parameters
df_motion['speed_ms'] = df_motion['speed'] / 3.6  # Convert km/h to m/s
df_motion['yaw_rate'] = df_motion['gz']  # rad/s

# Cornering radius calculation (R = v / ω)
df_motion['corner_radius'] = np.where(
    np.abs(df_motion['yaw_rate']) > 0.001,
    df_motion['speed_ms'] / np.abs(df_motion['yaw_rate']),
    np.inf
)

# Limit radius to reasonable values
df_motion['corner_radius'] = np.clip(df_motion['corner_radius'], 0, 1000)

# Cornering speed and lateral acceleration relationship
df_motion['theoretical_lateral_g'] = (df_motion['speed_ms']**2) / (df_motion['corner_radius'] * 9.81)

print("Calculated derived motion parameters")
```

## Performance Analysis

### Cornering Analysis
```python
# Define cornering phases
def identify_cornering_phases(data, steering_threshold=5.0, min_duration=1.0):
    """Identify cornering segments based on steering input"""
    
    # Detect cornering based on steering angle
    cornering_mask = np.abs(data['steering_angle']) > steering_threshold
    
    # Find continuous cornering segments
    cornering_changes = cornering_mask.diff().fillna(0)
    corner_starts = data[cornering_changes == True]['timestamp'].values
    corner_ends = data[cornering_changes == False]['timestamp'].values
    
    # Handle edge cases
    if len(corner_starts) > 0 and len(corner_ends) > 0:
        if corner_starts[0] > corner_ends[0]:
            corner_ends = corner_ends[1:]
        if len(corner_starts) > len(corner_ends):
            corner_ends = np.append(corner_ends, data['timestamp'].iloc[-1])
    
    # Filter by minimum duration
    valid_corners = []
    for start, end in zip(corner_starts, corner_ends):
        if end - start >= min_duration:
            valid_corners.append({
                'start_time': start,
                'end_time': end,
                'duration': end - start
            })
    
    return valid_corners, cornering_mask

corners, cornering_mask = identify_cornering_phases(df_motion)
df_motion['is_cornering'] = cornering_mask

print(f"Identified {len(corners)} cornering segments:")
for i, corner in enumerate(corners):
    print(f"  Corner {i+1}: {corner['start_time']:.1f}-{corner['end_time']:.1f}s ({corner['duration']:.1f}s)")
```

### Performance Metrics Calculation
```python
def calculate_performance_metrics(data, cornering_segments):
    """Calculate comprehensive performance metrics"""
    
    metrics = {
        'overall': {},
        'cornering': {},
        'segments': []
    }
    
    # Overall session metrics
    metrics['overall'] = {
        'duration': data['timestamp'].max() - data['timestamp'].min(),
        'distance': data['distance'].iloc[-1] if 'distance' in data else data['distance_calc'].iloc[-1],
        'max_speed': data['speed'].max(),
        'avg_speed': data['speed'].mean(),
        'max_lateral_g': data['lateral_g'].abs().max(),
        'max_longitudinal_g': data['longitudinal_g'].abs().max(),
        'min_corner_radius': data['corner_radius'].min()
    }
    
    # Cornering-specific metrics
    cornering_data = data[data['is_cornering']]
    if len(cornering_data) > 0:
        metrics['cornering'] = {
            'avg_corner_speed': cornering_data['speed'].mean(),
            'max_corner_lateral_g': cornering_data['lateral_g'].abs().max(),
            'avg_corner_radius': cornering_data['corner_radius'].mean(),
            'cornering_time_percentage': len(cornering_data) / len(data) * 100
        }
    
    # Individual segment analysis
    for i, segment in enumerate(cornering_segments):
        start_idx = data[data['timestamp'] >= segment['start_time']].index[0]
        end_idx = data[data['timestamp'] <= segment['end_time']].index[-1]
        segment_data = data.loc[start_idx:end_idx]
        
        segment_metrics = {
            'segment_id': i + 1,
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'duration': segment['duration'],
            'avg_speed': segment_data['speed'].mean(),
            'min_speed': segment_data['speed'].min(),
            'max_lateral_g': segment_data['lateral_g'].abs().max(),
            'avg_lateral_g': segment_data['lateral_g'].abs().mean(),
            'min_corner_radius': segment_data['corner_radius'].min(),
            'max_steering_angle': segment_data['steering_angle'].abs().max(),
            'corner_entry_speed': segment_data['speed'].iloc[0],
            'corner_exit_speed': segment_data['speed'].iloc[-1],
            'speed_loss': segment_data['speed'].iloc[0] - segment_data['speed'].min()
        }
        
        metrics['segments'].append(segment_metrics)
    
    return metrics

# Calculate performance metrics
performance = calculate_performance_metrics(df_motion, corners)

# Display results
print("=== PERFORMANCE ANALYSIS ===")
print(f"\nOverall Session:")
for key, value in performance['overall'].items():
    if isinstance(value, float):
        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
    else:
        print(f"  {key.replace('_', ' ').title()}: {value}")

print(f"\nCornering Performance:")
for key, value in performance['cornering'].items():
    print(f"  {key.replace('_', ' ').title()}: {value:.2f}")

print(f"\nSegment Analysis:")
for segment in performance['segments']:
    print(f"  Segment {segment['segment_id']} ({segment['start_time']:.1f}-{segment['end_time']:.1f}s):")
    print(f"    Speed: {segment['avg_speed']:.1f} km/h avg, {segment['speed_loss']:.1f} km/h loss")
    print(f"    Lateral G: {segment['max_lateral_g']:.2f} max, {segment['avg_lateral_g']:.2f} avg")
    print(f"    Corner radius: {segment['min_corner_radius']:.1f} m minimum")
```

## Data Visualization

### Time Series Plots
```python
# Create comprehensive time series visualization
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Speed and throttle/brake
axes[0].plot(df_motion['timestamp'], df_motion['speed'], 'b-', label='Speed')
axes[0].set_ylabel('Speed [km/h]', color='b')
axes[0].tick_params(axis='y', labelcolor='b')

ax0_twin = axes[0].twinx()
ax0_twin.plot(df_motion['timestamp'], df_motion['throttle'], 'g-', alpha=0.7, label='Throttle')
ax0_twin.plot(df_motion['timestamp'], df_motion['brake_pressure']*5, 'r-', alpha=0.7, label='Brake×5')
ax0_twin.set_ylabel('Throttle [%] / Brake×5 [bar]', color='g')
ax0_twin.tick_params(axis='y', labelcolor='g')

# Highlight cornering segments
for corner in corners:
    axes[0].axvspan(corner['start_time'], corner['end_time'], alpha=0.2, color='yellow', label='Cornering' if corner == corners[0] else "")

axes[0].set_title('Speed Profile and Driver Inputs')
axes[0].grid(True, alpha=0.3)

# Lateral and longitudinal acceleration
axes[1].plot(df_motion['timestamp'], df_motion['lateral_g'], 'r-', label='Lateral G')
axes[1].plot(df_motion['timestamp'], df_motion['longitudinal_g'], 'b-', label='Longitudinal G')
axes[1].set_ylabel('Acceleration [g]')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Vehicle Accelerations')

# Steering and yaw rate
axes[2].plot(df_motion['timestamp'], df_motion['steering_angle'], 'g-', label='Steering Angle')
axes[2].set_ylabel('Steering [deg]', color='g')
axes[2].tick_params(axis='y', labelcolor='g')

ax2_twin = axes[2].twinx()
ax2_twin.plot(df_motion['timestamp'], np.degrees(df_motion['yaw_rate']), 'purple', label='Yaw Rate')
ax2_twin.set_ylabel('Yaw Rate [deg/s]', color='purple')
ax2_twin.tick_params(axis='y', labelcolor='purple')
axes[2].set_title('Steering Input and Vehicle Response')
axes[2].grid(True, alpha=0.3)

# Corner radius
axes[3].plot(df_motion['timestamp'], df_motion['corner_radius'], 'orange', label='Corner Radius')
axes[3].set_ylabel('Radius [m]')
axes[3].set_xlabel('Time [s]')
axes[3].set_ylim(0, 200)  # Focus on tight corners
axes[3].set_title('Cornering Radius')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### G-G Diagram (Traction Circle)

The G-G diagram, also known as the traction circle, is a fundamental visualization tool in vehicle dynamics analysis that plots lateral acceleration versus longitudinal acceleration to show the vehicle's traction utilization (Milliken & Milliken, 1995).

```python
# Create G-G diagram (traction circle)
plt.figure(figsize=(10, 10))

# Plot all data points
plt.scatter(df_motion['lateral_g'], df_motion['longitudinal_g'], 
           c=df_motion['speed'], cmap='viridis', alpha=0.6, s=1)

# Highlight cornering segments
cornering_data = df_motion[df_motion['is_cornering']]
plt.scatter(cornering_data['lateral_g'], cornering_data['longitudinal_g'], 
           c='red', alpha=0.8, s=3, label='Cornering')

# Draw theoretical traction circle
theta = np.linspace(0, 2*np.pi, 100)
max_g = df_motion['lateral_g'].abs().max() * 1.1
circle_x = max_g * np.cos(theta)
circle_y = max_g * np.sin(theta)
plt.plot(circle_x, circle_y, 'r--', alpha=0.5, label='Theoretical Limit')

plt.xlabel('Lateral Acceleration [g]')
plt.ylabel('Longitudinal Acceleration [g]')
plt.title('G-G Diagram (Traction Circle)')
plt.colorbar(label='Speed [km/h]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

# Calculate traction utilization
df_motion['g_total'] = np.sqrt(df_motion['lateral_g']**2 + df_motion['longitudinal_g']**2)
max_g_utilization = df_motion['g_total'].max()
avg_g_utilization = df_motion['g_total'].mean()

print(f"Traction Utilization:")
print(f"  Maximum: {max_g_utilization:.3f} g")
print(f"  Average: {avg_g_utilization:.3f} g")
print(f"  Peak utilization: {max_g_utilization/1.5*100:.1f}% of theoretical maximum")
```

### Vehicle Path Visualization
```python
# Plot vehicle trajectory
plt.figure(figsize=(12, 8))

# Plot the path colored by speed
path_plot = plt.scatter(df_motion['pos_y'], df_motion['pos_x'], 
                       c=df_motion['speed'], cmap='viridis', s=10)

# Mark cornering segments
for i, corner in enumerate(corners):
    start_idx = df_motion[df_motion['timestamp'] >= corner['start_time']].index[0]
    end_idx = df_motion[df_motion['timestamp'] <= corner['end_time']].index[-1]
    corner_data = df_motion.loc[start_idx:end_idx]
    
    plt.plot(corner_data['pos_y'], corner_data['pos_x'], 'r-', linewidth=3, alpha=0.7)
    
    # Mark corner entry and exit
    plt.plot(corner_data['pos_y'].iloc[0], corner_data['pos_x'].iloc[0], 'go', markersize=8, label='Entry' if i == 0 else "")
    plt.plot(corner_data['pos_y'].iloc[-1], corner_data['pos_x'].iloc[-1], 'ro', markersize=8, label='Exit' if i == 0 else "")

plt.colorbar(path_plot, label='Speed [km/h]')
plt.xlabel('Lateral Position [m]')
plt.ylabel('Longitudinal Position [m]')
plt.title('Vehicle Path Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

## Data Export and Reporting

### Export Processed Data
```python
# Export cleaned and processed data
output_data = df_motion[[
    'timestamp', 'speed', 'lateral_g', 'longitudinal_g', 'vertical_g',
    'corner_radius', 'steering_angle', 'throttle', 'brake_pressure',
    'pos_x', 'pos_y', 'is_cornering'
]].copy()

# Add performance classifications
output_data['performance_zone'] = pd.cut(
    output_data['speed'],
    bins=[0, 60, 100, 150, 300],
    labels=['Low', 'Medium', 'High', 'Racing']
)

output_data['cornering_intensity'] = pd.cut(
    output_data['lateral_g'].abs(),
    bins=[0, 0.3, 0.8, 1.2, 2.0],
    labels=['Straight', 'Light', 'Moderate', 'Hard']
)

# Save to multiple formats
output_data.to_csv('racing_analysis_processed.csv', index=False)
output_data.to_hdf('data/racing_data.h5', key='processed_data', mode='w', complib='zlib')

# Export performance summary
performance_summary = pd.DataFrame([performance['overall']])
performance_summary.to_csv('performance_summary.csv', index=False)

# Export segment analysis
segments_df = pd.DataFrame(performance['segments'])
segments_df.to_csv('cornering_analysis.csv', index=False)

print("Data export completed:")
print(f"  - Processed data: racing_analysis_processed.csv ({len(output_data)} samples)")
print(f"  - HDF5 archive: data/racing_data.h5")
print(f"  - Performance summary: performance_summary.csv")
print(f"  - Cornering analysis: cornering_analysis.csv ({len(segments_df)} segments)")
```

### Generate Analysis Report
```python
def generate_analysis_report(data, performance_metrics, corners):
    """Generate comprehensive analysis report"""
    
    report = f"""
# Racing Data Analysis Report

## Session Overview
- **Duration**: {performance_metrics['overall']['duration']:.1f} seconds
- **Distance**: {performance_metrics['overall']['distance']:.1f} meters
- **Average Speed**: {performance_metrics['overall']['avg_speed']:.1f} km/h
- **Maximum Speed**: {performance_metrics['overall']['max_speed']:.1f} km/h

## Performance Highlights
- **Peak Lateral G-Force**: {performance_metrics['overall']['max_lateral_g']:.2f} g
- **Peak Longitudinal G-Force**: {performance_metrics['overall']['max_longitudinal_g']:.2f} g
- **Minimum Corner Radius**: {performance_metrics['overall']['min_corner_radius']:.1f} meters
- **Cornering Time**: {performance_metrics['cornering']['cornering_time_percentage']:.1f}% of session

## Cornering Analysis
- **Number of Corners**: {len(corners)}
- **Average Corner Speed**: {performance_metrics['cornering']['avg_corner_speed']:.1f} km/h
- **Maximum Corner G-Force**: {performance_metrics['cornering']['max_corner_lateral_g']:.2f} g
- **Average Corner Radius**: {performance_metrics['cornering']['avg_corner_radius']:.1f} meters

## Detailed Segment Analysis
"""
    
    for segment in performance_metrics['segments']:
        report += f"""
### Segment {segment['segment_id']} ({segment['start_time']:.1f} - {segment['end_time']:.1f}s)
- **Duration**: {segment['duration']:.1f} seconds
- **Speed Profile**: {segment['corner_entry_speed']:.1f} → {segment['min_speed']:.1f} → {segment['corner_exit_speed']:.1f} km/h
- **Speed Loss**: {segment['speed_loss']:.1f} km/h
- **Peak Lateral G**: {segment['max_lateral_g']:.2f} g
- **Average Lateral G**: {segment['avg_lateral_g']:.2f} g
- **Minimum Radius**: {segment['min_corner_radius']:.1f} meters
- **Maximum Steering**: {segment['max_steering_angle']:.1f} degrees
"""
    
    return report

# Generate and save report
report_content = generate_analysis_report(df_motion, performance, corners)

with open('racing_analysis_report.md', 'w') as f:
    f.write(report_content)

print("Analysis report generated: racing_analysis_report.md")
```

## References

Milliken, W. F., & Milliken, D. L. (1995). *Race Car Vehicle Dynamics*. SAE International. ISBN: 978-1-56091-526-3.

> **Note**: This seminal work on vehicle dynamics introduced many of the fundamental concepts used in racing data analysis, including the G-G diagram (traction circle) which has become the standard visualization for understanding vehicle performance limits and tire utilization.

This comprehensive case study demonstrates the complete workflow from raw racing data to meaningful engineering insights, including data cleaning, coordinate transformations, performance analysis, and professional reporting. The processed data and analysis results provide the foundation for further visualization and interactive analysis in subsequent course sessions.