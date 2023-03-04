import pandas as pd
import numpy as np

# Create a sample DataFrame
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
df = pd.DataFrame({'date': dates})
df['sales'] = np.random.randint(low=50, high=1000, size=len(df))

# Simple time features
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_holiday'] = np.where(df['date'].isin(pd.date_range(start='2022-05-28', end='2022-05-30')), 1, 0)

# Lag features
df['lag7'] = df['sales'].shift(7)
df['lag14'] = df['sales'].shift(14)
df['rolling_mean7'] = df['sales'].rolling(window=7).mean()

# Seasonal indicators
df['quarter'] = df['date'].dt.quarter
df['is_summer'] = np.where(df['month'].isin([6, 7, 8]), 1, 0)

# Economic indicators
df['interest_rate'] = np.random.uniform(low=0.01, high=0.05, size=len(df))
df['consumer_confidence'] = np.random.uniform(low=50, high=150, size=len(df))

# Marketing initiatives
df['new_product_launch'] = np.where(df['date'].isin(pd.date_range(start='2022-03-01', end='2022-03-31')), 1, 0)
df['ad_campaign'] = np.where(df['date'].isin(pd.date_range(start='2022-11-01', end='2022-11-30')), 1, 0)

# External factors
df['temperature'] = np.random.randint(low=30, high=100, size=len(df))
df['flu_season'] = np.where(df['month'].isin([10, 11, 12, 1, 2]), 1, 0)

# Additional features
df['social_media_mentions'] = np.random.randint(low=0, high=100, size=len(df))
df['competitor_price'] = np.random.uniform(low=5, high=50, size=len(df))
df['GDP'] = np.random.uniform(low=
