# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#prediction OF OUR DEPENDANT VARIABLE
y_pred=regressor.predict(X_test)
#vizulaoize
plt.scatter(X_train,y_train,color="RED")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Salary vs experience(training set)")
plt.xlabel("year of experience")
plt.ylabel("salary")
plt.show()

plt.scatter(X_test,y_test,color="RED")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Salary vs experience(training set)")
plt.xlabel("year of experience")
plt.ylabel("salary")
plt.show()
