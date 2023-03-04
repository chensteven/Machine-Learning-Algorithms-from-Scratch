import pandas as pd
import matplotlib.pyplot as plt

# Import the data
df = pd.read_csv('data/sales_data.csv', index_col='Date', parse_dates=True)

# Calculate the moving average with a window of 3
ma = df['sales'].rolling(window=3).mean()

# Plot the original data and the moving average
plt.plot(df['sales'], label='Sales')
plt.plot(ma, label='Moving Average')
plt.legend()
plt.show()
