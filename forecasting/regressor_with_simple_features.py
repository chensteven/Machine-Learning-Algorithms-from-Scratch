import pandas as pd

# Load the dataset
sales_df = pd.read_csv('sales_data.csv', parse_dates=['date'])

# Add day of the week feature
sales_df['day_of_week'] = sales_df['date'].dt.day_name()

# Add month of the year feature
sales_df['month_of_year'] = sales_df['date'].dt.month_name()

# Add holiday feature (assuming we have a separate holiday dataset)
holiday_df = pd.read_csv('holiday_data.csv', parse_dates=['date'])
sales_df = pd.merge(sales_df, holiday_df, how='left', on='date')
sales_df['is_holiday'] = sales_df['is_holiday'].fillna(0)

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Split the data into training and testing sets
train_df = sales_df[sales_df['date'] < '2022-01-01']
test_df = sales_df[sales_df['date'] >= '2022-01-01']

# Define the feature matrix and target variable
X_train = np.array(train_df[['day_of_week', 'month_of_year', 'is_holiday']])
y_train = np.array(train_df['sales'])

# Train the random forest model
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# Make predictions on the test set
X_test = np.array(test_df[['day_of_week', 'month_of_year', 'is_holiday']])
y_pred = rf.predict(X_test)
