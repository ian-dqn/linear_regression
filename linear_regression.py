import numpy as np
import make_regression
import matplotlib.pyplot as plt
"""from sklearn.datasets if no dataset
x, y = make_regression(n_samples=100, n_features=1, noise=8)
plt.scatter(x, y) #graph
y = y.reshape(y.shape[0], 1)
"""

#verifying dataset
print(x.shape)
print(y.shape)

#matrix X
X = np.hstack((x, np.ones(x.shape)))
theta = np.random.randn(2, 1)

def model (X, theta):
  return X.dot(theta)

#graph
plt.scatter(x, y) 
plt.plot(x, model(X, theta), c='r')

def cost_function(x, y, theta):
  m = len(y)
  return 1/(2*m) * np.sum((model(X, theta) - y)**2)

def grad(X, y, theta):
  m = len(y)
  return 1/m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
  cost_history = np_zeros(n_iterations)
  for i in range(0, n_iterations):
    theta = theta - learning_rate * grad(X, y, theta)
    cost_history[i] = cost_function(X, y, theta)
  return theta, cost_history

theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0,001, n_iterations=500)
predictions = model(X, theta_final)


#graph
plt.scatter(x, y)
plt.plot(x, predictions, c='r')
plt.plot(range(1000), cost_history)


def coef_determination(y, pred):
  u = ((y - pred)**2).sum()
  v = ((y - y.mean())**2).sum()
  return 1 - u/v

print(coef_determination(y, predictions))
