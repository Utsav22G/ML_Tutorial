from sklearn.tree import DecisionTreeRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Fitting Decision Tree to dataset
regressor = DecisionTreeRegressor(criterion = 'mse', random_state = 0, presort = True)
regressor.fit(X, y)

# Predicting new result
"""
Converting 6.5 to vector having 2 element --> np.array([6.5])
Converting 6.5 to matrix-type array --> np.array([[6.5]])
"""
lvl = 6.5
y_pred = regressor.predict(lvl)
print("Salary at Level ", lvl, " is ", y_pred[0])

# Visualising Decision Tree result
"""
Incrementing the steps by 0.1 to increase the resolution
and smoothen the curves.
"""
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red', label = 'Data')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue', label = 'Decision Tree model')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend(loc ='upper center')
plt.show()
