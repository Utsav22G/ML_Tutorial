from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values     #.values removes the axes labels and returns numpy type array
Y = dataset.iloc[:, 3].values

# Removing missing values
impute = Imputer(missing_values="NaN", strategy="mean", axis=0)
impute = impute.fit(X[:, 1:3])
X[:, 1:3] = impute.transform(X[:, 1:3])

# Encoding categorical data
X[:, 0] = LabelEncoder().fit_transform(X[:, 0])
X = OneHotEncoder(categorical_features=[0]).fit_transform(X=X).toarray()
Y = LabelEncoder().fit_transform(Y)

# Test-Train split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Scaling feature matrix
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().transform(X_test)
