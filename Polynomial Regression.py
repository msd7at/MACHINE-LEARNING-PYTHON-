
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#buliding simple linear regression model 
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)


#visualizing the linear Regression result
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X),color="yellow")
plt.title("truth or Bluff (linear Regression)")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()

#visualizing the Polynomial Regression result
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color="yellow")
plt.title("truth or Bluff (polynomial Regression)")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()
