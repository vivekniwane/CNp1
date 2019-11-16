import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel('sal_exp.xlsx',sheet_name = 'Sheet1')
X = df[['Experience']]
#X = df['Experience'].values.reshape(-1,1)
y = df['Salary']
from sklearn.preprocessing import PolynomialFeatures
P = PolynomialFeatures(degree=2)
X_poly = P.fit_transform(X)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_poly,y)

#y_predict = model.predict(X_poly)

#plt.scatter(df['Experience'],df['Salary'],c='red')
#plt.plot(X,y_predict)
import pickle
pickle.dump(model,open('model_1.pkl','wb'))