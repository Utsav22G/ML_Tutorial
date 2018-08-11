from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test set results
y_pred = regressor.predict(X_test)

# Metrics
mae = np.sum((y_pred - y_test))/len(y_test)
mse = mean_squared_error(y_test, y_pred)
print("MAE = ", mae)
print("RMSE = ", np.sqrt(mse))

# Visualising the Training set results
fig = plt.figure()
fig.add_subplot(2,1,1)
plt.scatter(X_train, y_train, color = 'red', label = 'Training data')    # training datapoints
plt.plot(X_train, regressor.predict(X_train), color = 'blue', label = 'Trained model')   # trained model
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(loc = 'best')

# Visualising the Test set results
fig.add_subplot(2,1,2)
plt.scatter(X_test, y_test, color = 'red', label = 'Testing data')    # testing datapoints
plt.plot(X_train, regressor.predict(X_train), color = 'blue', label = 'Trained model')   # trained model
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(loc = 'best')
plt.show()
