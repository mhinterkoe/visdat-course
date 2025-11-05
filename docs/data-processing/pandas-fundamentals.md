---
title: Pandas Fundamentals
---

# Pandas Fundamentals

## Introduction to Pandas

Pandas is Python's premier data manipulation and analysis library, built on top of NumPy. It provides two primary data structures that are perfect for engineering applications: Series (1-dimensional) and DataFrame (2-dimensional).

## Core Data Structures

### Series: One-Dimensional Data
```python
import pandas as pd
import numpy as np

# Create a Series for acceleration data
acceleration_x = pd.Series([0.1, 0.2, 0.15, 0.3], 
                          index=['t1', 't2', 't3', 't4'])
print(acceleration_x)

# Series with automatic index
sensor_readings = pd.Series([9.81, 9.79, 9.82, 9.80])
print(f"Mean: {sensor_readings.mean():.3f}")
print(f"Std: {sensor_readings.std():.3f}")
```

### DataFrame: Two-Dimensional Data
```python
# Create DataFrame for IMU data
imu_data = pd.DataFrame({
    'timestamp': [0.0, 0.001, 0.002, 0.003],
    'ax': [0.1, 0.2, 0.15, 0.3],
    'ay': [9.81, 9.80, 9.82, 9.79],
    'az': [0.05, 0.03, 0.08, 0.06],
    'gx': [0.001, 0.002, 0.001, 0.003],
    'gy': [0.02, 0.025, 0.018, 0.022],
    'gz': [0.003, 0.005, 0.002, 0.008]
})

print(imu_data.head())
print(f"Shape: {imu_data.shape}")
```

## Essential Operations

### Data Loading and Inspection
```python
# Load racing session data
sessions = pd.read_csv('data/racing_sessions.csv')
print(sessions.info())
print(sessions.head())

# Load lap time data
laps = pd.read_csv('data/lap_times.csv')
print(f"Total laps: {len(laps)}")
print(f"Fastest lap: {laps['lap_time_s'].min():.3f}s")

# Load detailed telemetry
telemetry = pd.read_csv('data/telemetry_detailed.csv')
print(f"Telemetry points: {len(telemetry)}")
print(f"Speed range: {telemetry['speed_kmh'].min()}-{telemetry['speed_kmh'].max()} km/h")
```
df = pd.read_csv('sensor_data.csv')
df_excel = pd.read_excel('measurements.xlsx', sheet_name='Sheet1')
df_json = pd.read_json('telemetry.json')

# Basic information
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")

# Statistical summary
print(df.describe())

# Missing values
print(f"Missing values:\n{df.isnull().sum()}")

# First and last rows
print("First 5 rows:")
print(df.head())
print("Last 5 rows:")
print(df.tail())
```

### Data Selection and Filtering
```python
# Column selection
speed_data = df['speed']
acceleration_data = df[['ax', 'ay', 'az']]

# Row selection by index
first_1000_samples = df.iloc[:1000]
specific_rows = df.iloc[100:200]

# Boolean indexing (filtering)
high_speed = df[df['speed'] > 100]
cornering = df[df['steering_angle'].abs() > 10]

# Multiple conditions
fast_cornering = df[(df['speed'] > 80) & (df['steering_angle'].abs() > 15)]

# Query method (alternative syntax)
cornering_alt = df.query('abs(steering_angle) > 10')
```

### Data Modification
```python
# Add new columns
df['speed_ms'] = df['speed'] / 3.6  # Convert km/h to m/s
df['lateral_g'] = df['ay'] / 9.81   # Convert to g-forces

# Modify existing columns
df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]  # Zero-start time

# Drop columns
df_reduced = df.drop(['unused_column1', 'unused_column2'], axis=1)

# Rename columns
df_renamed = df.rename(columns={
    'ax': 'acceleration_x',
    'ay': 'acceleration_y',
    'az': 'acceleration_z'
})
```

## Data Cleaning

### Handling Missing Values
```python
# Check for missing values
missing_summary = df.isnull().sum()
print("Missing values per column:")
print(missing_summary[missing_summary > 0])

# Strategy 1: Drop rows with any missing values
df_dropna = df.dropna()

# Strategy 2: Drop rows with too many missing values
threshold = len(df.columns) * 0.7  # Keep rows with at least 70% data
df_threshold = df.dropna(thresh=threshold)

# Strategy 3: Forward fill for short gaps
df_ffill = df.fillna(method='ffill', limit=5)

# Strategy 4: Linear interpolation for sensor data
df_interpolated = df.copy()
df_interpolated['ax'] = df_interpolated['ax'].interpolate(method='linear')
df_interpolated['ay'] = df_interpolated['ay'].interpolate(method='linear')

# Strategy 5: Fill with specific values
df_filled = df.fillna({
    'speed': 0,
    'steering_angle': 0,
    'ax': df['ax'].mean()
})
```

### Outlier Detection and Removal
```python
def detect_outliers_iqr(data, column):
    """Detect outliers using Interquartile Range method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | 
                   (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers
outliers, lower, upper = detect_outliers_iqr(df, 'ax')
print(f"Found {len(outliers)} outliers in ax")
print(f"Valid range: [{lower:.3f}, {upper:.3f}] m/s²")

# Remove outliers
def remove_outliers_iqr(data, columns):
    """Remove outliers from specified columns using IQR method"""
    df_clean = data.copy()
    
    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Keep only values within bounds
        mask = (df_clean[column] >= lower) & (df_clean[column] <= upper)
        df_clean = df_clean[mask]
    
    return df_clean

# Apply to acceleration channels
acceleration_columns = ['ax', 'ay', 'az']
df_clean = remove_outliers_iqr(df, acceleration_columns)
print(f"Removed {len(df) - len(df_clean)} outlier samples")
```

## Basic Statistical Operations

### Descriptive Statistics
```python
# Single column statistics
print(f"Speed - Mean: {df['speed'].mean():.2f} km/h")
print(f"Speed - Median: {df['speed'].median():.2f} km/h")
print(f"Speed - Std: {df['speed'].std():.2f} km/h")
print(f"Speed - Min/Max: {df['speed'].min():.1f} / {df['speed'].max():.1f} km/h")

# Multiple columns
stats_summary = df[['speed', 'ax', 'ay', 'az']].describe()
print(stats_summary)

# Custom percentiles
percentiles = df['speed'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print("Speed percentiles:")
print(percentiles)
```

### Correlation Analysis
```python
# Correlation matrix
correlation_matrix = df[['speed', 'ax', 'ay', 'steering_angle']].corr()
print("Correlation matrix:")
print(correlation_matrix)

# Specific correlations
speed_steering_corr = df['speed'].corr(df['steering_angle'])
print(f"Speed-Steering correlation: {speed_steering_corr:.3f}")
```

## Data Transformation

### Mathematical Operations
```python
# Element-wise operations
df['speed_squared'] = df['speed'] ** 2
df['acceleration_magnitude'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

# Trigonometric functions
df['steering_rad'] = np.radians(df['steering_angle'])
df['steering_sin'] = np.sin(df['steering_rad'])

# Logarithmic transformations
df['speed_log'] = np.log1p(df['speed'])  # log(1+x) to handle zeros

# Normalization
df['speed_normalized'] = (df['speed'] - df['speed'].mean()) / df['speed'].std()

# Min-max scaling
df['speed_scaled'] = (df['speed'] - df['speed'].min()) / (df['speed'].max() - df['speed'].min())
```

### Binning and Categorization
```python
# Create speed categories
speed_bins = [0, 30, 60, 100, 150, 300]
speed_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
df['speed_category'] = pd.cut(df['speed'], bins=speed_bins, labels=speed_labels)

# Equal-width binning
df['ax_quartiles'] = pd.qcut(df['ax'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Custom conditions
conditions = [
    (df['speed'] < 50),
    (df['speed'] >= 50) & (df['speed'] < 100),
    (df['speed'] >= 100)
]
choices = ['City', 'Highway', 'Racing']
df['driving_mode'] = np.select(conditions, choices, default='Unknown')
```

## File I/O Operations

### Reading Data
```python
# CSV with custom parameters
df = pd.read_csv('data.csv', 
                sep=';',           # Different separator
                decimal=',',       # European decimal format
                encoding='utf-8',  # Character encoding
                skiprows=2,        # Skip header rows
                nrows=10000,       # Read only first 10k rows
                usecols=['timestamp', 'speed', 'ax', 'ay'],  # Specific columns
                dtype={'speed': 'float32'},  # Specify data types
                parse_dates=['timestamp'])   # Parse dates

# Excel with multiple sheets
df_sheet1 = pd.read_excel('data.xlsx', sheet_name='Telemetry')
df_all_sheets = pd.read_excel('data.xlsx', sheet_name=None)  # All sheets

# JSON
df_json = pd.read_json('telemetry.json', orient='records')
```

### Writing Data
```python
# CSV export
df.to_csv('processed_data.csv', 
          index=False,           # Don't save index
          float_format='%.6f',   # Control decimal places
          sep=';',               # Custom separator
          encoding='utf-8')      # Character encoding

# Excel export
df.to_excel('analysis_results.xlsx', 
           sheet_name='Processed_Data',
           index=False,
           float_format='%.3f')

# Multiple sheets
with pd.ExcelWriter('race_analysis.xlsx') as writer:
    df.to_excel(writer, sheet_name='Raw_Data', index=False)
    df_clean.to_excel(writer, sheet_name='Cleaned_Data', index=False)
    summary_stats.to_excel(writer, sheet_name='Summary', index=False)

# JSON export
df.to_json('telemetry_export.json', 
          orient='records',      # Array of objects
          date_format='iso',     # ISO date format
          indent=2)              # Pretty formatting
```

## Performance Tips

### Memory Optimization
```python
# Check memory usage
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Optimize data types
def optimize_dtypes(df):
    """Optimize DataFrame memory usage"""
    original_size = df.memory_usage(deep=True).sum()
    
    # Optimize float columns
    for col in df.select_dtypes(include=['float64']):
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize integer columns
    for col in df.select_dtypes(include=['int64']):
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    optimized_size = df.memory_usage(deep=True).sum()
    print(f"Memory reduced from {original_size/1e6:.1f} MB to {optimized_size/1e6:.1f} MB")
    print(f"Reduction factor: {original_size/optimized_size:.1f}x")
    
    return df

df_optimized = optimize_dtypes(df.copy())
```

### Vectorized Operations
```python
# Avoid loops - use vectorized operations
# BAD: Loop through rows
results = []
for index, row in df.iterrows():
    result = row['ax'] * row['speed'] / 3.6
    results.append(result)
df['bad_calculation'] = results

# GOOD: Vectorized operation
df['good_calculation'] = df['ax'] * df['speed'] / 3.6

# Use .apply() for complex operations
def complex_calculation(row):
    return np.sqrt(row['ax']**2 + row['ay']**2) * row['speed']

df['complex_result'] = df.apply(complex_calculation, axis=1)

# Even better: Pure vectorized
df['complex_result_vectorized'] = np.sqrt(df['ax']**2 + df['ay']**2) * df['speed']
```

## Best Practices

### Code Organization
```python
# Create reusable functions
def load_telemetry_data(filename):
    """Load and basic preprocessing of telemetry data"""
    df = pd.read_csv(filename)
    
    # Basic validation
    required_columns = ['timestamp', 'speed', 'ax', 'ay', 'az']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def validate_sensor_data(df):
    """Validate sensor data ranges"""
    validation_rules = {
        'speed': (0, 350),      # km/h
        'ax': (-20, 20),        # m/s²
        'ay': (-20, 20),        # m/s²
        'az': (-20, 20),        # m/s²
        'steering_angle': (-720, 720)  # degrees
    }
    
    issues = []
    for column, (min_val, max_val) in validation_rules.items():
        if column in df.columns:
            out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
            if len(out_of_range) > 0:
                issues.append(f"{column}: {len(out_of_range)} values out of range [{min_val}, {max_val}]")
    
    return issues

# Use the functions
df = load_telemetry_data('race_data.csv')
validation_issues = validate_sensor_data(df)
if validation_issues:
    print("Data validation issues:")
    for issue in validation_issues:
        print(f"  - {issue}")
```

### Documentation and Metadata
```python
# Document your data processing steps
processing_log = {
    'source_file': 'raw_telemetry.csv',
    'processing_date': pd.Timestamp.now().isoformat(),
    'steps': [
        'loaded_raw_data',
        'removed_outliers_iqr',
        'interpolated_missing_values',
        'applied_moving_average_filter',
        'calculated_derived_parameters'
    ],
    'data_quality': {
        'original_samples': len(df_raw),
        'final_samples': len(df_processed),
        'retention_rate': len(df_processed) / len(df_raw)
    }
}

# Save processing metadata
import json
with open('processing_log.json', 'w') as f:
    json.dump(processing_log, f, indent=2)
```

This covers the fundamental pandas operations you'll need for most data processing tasks. The next documents will build on these basics with more advanced operations and specific use cases.