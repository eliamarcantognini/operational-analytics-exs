# random forest for making predictions for regression
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=2)
# define the model
model = RandomForestRegressor()
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [[-0.89483109, -1.0670149, -0.25448694, -0.53850126, 0.21082105, 1.37435592, 0.71203659, 0.73093031, -1.25878104,
        -2.01656886, 0.51906798, 0.62767387, 0.96250155, 1.31410617, -1.25527295, -0.85079036, 0.24129757, -0.17571721,
        -1.11454339, 0.36268268]]
yhat = model.predict(row)
print('Prediction: %d' % yhat[0])

# forecast, train / test split
dataset = pd.read_csv('..\\res\\FilRouge.csv')
x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]
x_train, x_valid = x[:-12], x[-12:]
y_train, y_valid = y[:-12], y[-12:]
mdl = rf = RandomForestRegressor(n_estimators=500)
mdl.fit(x_train, y_train)
pred = mdl.predict(x_valid)
mse = mean_absolute_error(y_valid, pred)
print("MSE={}".format(mse))
