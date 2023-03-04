import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Import the data
df = pd.read_csv('sales_data.csv', index_col='date', parse_dates=True)

# Fit the model with the data
model = ARIMA(df['sales'], order=(1, 1, 1))
fit_model = model.fit()

# Make predictions for the next 12 months
predictions = fit_model.predict(start=len(df), end=len(df)+11)

# Plot the original data and the predictions
plt.plot(df['sales'], label='Sales')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
