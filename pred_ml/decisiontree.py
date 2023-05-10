from xgboost import XGBRegressor
import pandas as pd, numpy as np, matplotlib.pyplot as plt
dataset = pd.read_csv('..\\res\\FilRouge.csv')
x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]
x_train, xtest = x[:-12], x[-12:]
y_train, ytest = y[:-12], y[-12:]
# fit model
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(x_train, y_train)
# make a one-step prediction
yhat = model.predict(xtest)