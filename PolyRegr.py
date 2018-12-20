# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 23:32:00 2018

@author: abhik
"""

# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('PR.csv')
X = train_data['Var_X'].values.reshape(-1,1)
y = train_data['Var_Y'].values

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree = 4)
X_poly = poly_feat.fit_transform(X)
print(X_poly)

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression(fit_intercept = False)
poly_model.fit(X_poly, y)
print(poly_model)
print(poly_model.coef_)
print(poly_model.intercept_)

X_new = np.linspace(X.min(), X.max(), 50).reshape(-1,1)
X_new_tr = poly_feat.fit_transform(X_new)
yhat = poly_model.predict(X_new_tr)

plt.scatter(X,y)
plt.plot(X_new, yhat, color = 'red')

plt.show()