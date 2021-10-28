import pandas as pd
import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = pd.read_csv("./X.csv")
Y = pd.read_csv("./Y.csv")
x = pd.read_csv("./x.csv")
y = pd.read_csv("./y.csv")

# Create linear regression object
# regr = linear_model.LinearRegression()
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

regr.fit(X,Y.values.tolist())
# y_pred = regr.predict(x)
score = regr.score(x,y.values.tolist())
print(score*100)
