import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('sales_data.csv', parse_dates=['Date'])
df = df.set_index('Date')

# Create lagged features
df['lag_7'] = df['Sales'].shift(7)
df['lag_14'] = df['Sales'].shift(14)
df['rolling_mean_7'] = df['Sales'].rolling(7).mean()

# Split the data into train and test sets
train = df.loc['2018-01-01':'2019-12-31']
test = df.loc['2020-01-01':]

# Train a random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
X_train = train.drop('Sales', axis=1)
y_train = train['Sales']
model.fit(X_train, y_train)

# Make predictions on the test set
X_test = test.drop('Sales', axis=1)
y_test = test['Sales']
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse:.2f}')
