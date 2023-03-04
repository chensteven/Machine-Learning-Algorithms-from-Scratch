"""
In this example, we create a time series of daily sales data from January 1, 2023 to March 31, 2023. We then create several features, including lagged features with lags of 7 and 14 days, a rolling mean with a window of 7 days, seasonal indicators (month and day of the week), economic indicators (interest rates and consumer confidence), a marketing initiative (new product launch), and an external factor (temperature). Finally, we combine all of these features into a DataFrame called df. Note that this is just an example, and in practice, the specific features and values used will depend on the specific forecasting problem and data available.
"""

import pandas as pd
import numpy as np

# Create time series data
dates = pd.date_range('2023-01-01', '2023-03-31')
sales = pd.Series(np.random.randint(1000, 2000, size=len(dates)), index=dates)

# Create lag features
sales_lag7 = sales.shift(7)
sales_lag14 = sales.shift(14)
sales_roll7 = sales.rolling(window=7).mean()

# Create seasonal indicators
seasonal_indicators = pd.concat([pd.get_dummies(sales.index.month, prefix='month'), 
                                 pd.get_dummies(sales.index.dayofweek, prefix='dayofweek')], axis=1)

# Create economic indicators
interest_rates = pd.Series(np.random.uniform(1.5, 2.5, size=len(dates)), index=dates)
consumer_confidence = pd.Series(np.random.randint(50, 100, size=len(dates)), index=dates)

# Create marketing initiatives
new_product_launch = pd.Series(0, index=dates)
new_product_launch['2023-02-15'] = 1

# Create external factors
temperature = pd.Series(np.random.randint(0, 40, size=len(dates)), index=dates)

# Combine all features into DataFrame
df = pd.concat([sales, sales_lag7, sales_lag14, sales_roll7, seasonal_indicators,
                interest_rates, consumer_confidence, new_product_launch, temperature], axis=1)
df.columns = ['sales', 'sales_lag7', 'sales_lag14', 'sales_roll7'] + list(seasonal_indicators.columns) + ['interest_rates', 'consumer_confidence', 'new_product_launch', 'temperature']
