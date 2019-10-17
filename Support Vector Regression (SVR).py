# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""feature Scaling or Standardization: It is a step of Data Pre Processing which is applied to independent variables or features of data.
 It basically helps to normalise the data within a particular range.
 Sometimes, it also helps in speeding up the calculations in an algorithm.
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X=sc_X.fit_transform(np.array(X).reshape(10,1))
y=sc_y.fit_transform(np.array(y).reshape(10,1))

# Fitting the svr to the dataset
from sklearn.svm import SVR
regressor=SVR(kernel="rbf")
regressor.fit(X,y)
# Create your regressor here

# Predicting a new result
Y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the svr results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (svr Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
