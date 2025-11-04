import pandas as pd
import numpy as np
import os

print("Generating realistic solar dataset...")

# Create 5 years of hourly data
HOURS_IN_5_YEARS = 5 * 365 * 24
time_index = pd.date_range(start='2020-01-01', periods=HOURS_IN_5_YEARS, freq='h')

df = pd.DataFrame(index=time_index)
df['hour_of_day'] = df.index.hour
df['month'] = df.index.month

# 1. Simulate Radiation (W/m^2)
# 0 at night, peaks at noon. Varies by month (less in winter).
hour_factor = np.sin((df['hour_of_day'] - 5) * np.pi / 14) # Peaking around 12-1pm
hour_factor[df['hour_of_day'] < 5] = 0
hour_factor[df['hour_of_day'] > 19] = 0
month_factor = (1 - 0.4 * np.cos(df['month'] * np.pi / 6)) # Max in summer
radiation = 1000 * hour_factor * month_factor * (1 - np.random.rand(len(df)) * 0.2) # Add noise
df['radiation_w_m2'] = np.maximum(0, radiation)

# 2. Simulate Temperature (°C)
# Varies by hour and month.
temp_base = 10 + 10 * np.cos((df['month'] - 7) * np.pi / 6) # Colder in winter
temp_hour = 7 * np.sin((df['hour_of_day'] - 8) * np.pi / 12) # Peaking in afternoon
df['temperature_c'] = temp_base + temp_hour + np.random.randn(len(df)) * 1.5

# 3. Simulate Ground Truth (kwh_generated)
# This is what the model will learn to predict.
BASE_KWH = 10.0 # Max output of our system
radiation_factor = df['radiation_w_m2'] / 1000.0

# Temp efficiency loss (0.4% per degree above 25°C)
temp_loss = np.maximum(0, (df['temperature_c'] - 25) * 0.004)
temp_factor = np.maximum(0.1, 1.0 - temp_loss)

# Add random noise (clouds, dirt, etc.)
noise = 1.0 - np.random.rand(len(df)) * 0.15

df['kwh_generated'] = np.maximum(0, BASE_KWH * radiation_factor * temp_factor * noise)

# Select final columns
final_df = df[['hour_of_day', 'month', 'temperature_c', 'radiation_w_m2', 'kwh_generated']]

# Save to CSV
output_file = 'solar_data.csv'
final_df.to_csv(output_file, index=False)

print(f"Successfully generated '{output_file}' with {len(final_df)} rows.")
print(f"File saved to: {os.path.abspath(output_file)}")
