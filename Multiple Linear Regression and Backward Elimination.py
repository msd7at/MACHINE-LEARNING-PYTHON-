# multiple linear regression 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values     #independant factors
y = dataset.iloc[:, 4].values       #depenadnat vector i.e profit
#categorical variable to dummy variables
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3]) #this line will covert text to numbers
"label encoder is used as in the upcoming steps  onehotencoder we can not covert text into dummy variables"
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#to avoid dummy variable trap
X=X[:,1:]
# Splitting the dataset into the Training set and Test sety
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#prediction time
y_pred=regressor.predict(X_test)

#backward eleimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y ,exog=X_opt ).fit()
regressor_OLS.summary()
#following backward elemination (homework)
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y ,exog=X_opt ).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y ,exog=X_opt ).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y ,exog=X_opt ).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y ,exog=X_opt ).fit()
regressor_OLS.summary()
