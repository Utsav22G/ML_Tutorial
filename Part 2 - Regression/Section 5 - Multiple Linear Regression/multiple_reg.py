from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

# Importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Independant Variable
X[:, 3] = LabelEncoder().fit_transform(X[:, 3])
X = OneHotEncoder(categorical_features = [3]).fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]    # removes first column

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test set results
y_pred = regressor.predict(X_test)

# Metrics
mae = np.sum((y_pred - y_test))/len(y_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Absolute Error = ", mae)
print("Root Mean Squared Error = ", np.sqrt(mse))

# Building optimal model using Backwards Elimination
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)     # Adding column of ones as first column

X_opt = X[:, :]         # X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
p_val = regressor_OLS.pvalues
sig_lvl = 0.05

while p_val[np.argmax(p_val)] > sig_lvl:
    X_opt = np.delete(X_opt, np.argmax(p_val), axis=1)
    print("p-value of dimension removed: " + str(np.amax(p_val)))
    print(str(X_opt.shape[1]) + " dimensions remaining...")
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    p_val = regressor_OLS.pvalues

print(regressor_OLS.summary())
