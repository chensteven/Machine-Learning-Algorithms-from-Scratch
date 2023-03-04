import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a column of ones to X for the intercept term
X_b = np.c_[np.ones((100, 1)), X]

# Set learning rate and number of iterations
eta = 0.1
n_iterations = 1000

# Initialize theta with random values
theta = np.random.randn(2,1)

# Perform gradient descent
for iteration in range(n_iterations):
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

# Print the final values of theta
print(theta)

# Plot the data and the linear regression line
plt.plot(X, y, 'b.')
plt.plot(X, X_b.dot(theta), 'r-')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
