import os
import zipfile
from math import log, exp

import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
import requests
from keras.layers import LSTM, Dense
from keras.models import Sequential
from scipy.stats import boxcox
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor

from dm_test import dm_test

#####################
# data loading
#####################
# download data
if not os.path.exists('..\\res\\household_power_consumption.zip'):
    print('Downloading data...')
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip')
    with open('..\\res\\household_power_consumption.zip', 'wb') as output_file:
        output_file.write(r.content)
    print('Data downloaded.')
    # unzip data
    with zipfile.ZipFile('..\\res\\household_power_consumption.zip', 'r') as zip_ref:
        zip_ref.extractall('..\\res\\')
else:
    print('Data already downloaded.')
# load data
data = pd.read_csv('..\\res\\household_power_consumption.txt', sep=';', header=0, low_memory=False)
# data exploration
print('----------------\nData exploration\n----------------')
print(data.info())
print(data.describe())
print(data.shape)
print(data.head(5))
print('----------------')
#####################
#####################
# preprocessing
#####################
#####################
# drop columns
df = data.drop(columns=['Global_reactive_power', 'Global_intensity', 'Time', 'Voltage', 'Sub_metering_1', 'Sub_metering_2',
                   'Sub_metering_3'])
# convert to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# convert to numeric
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce', downcast='float')
df.rename(columns={'Date': 'Month', 'Global_active_power': 'Active'}, inplace=True)

# sum power consumption per month
df = df.groupby('Month').sum()
# resample to monthly
df = df.groupby(pd.Grouper(freq='M')).sum()
# drop first month (incomplete)
# drop last month (incomplete)
df = df.drop([df.index[0], df.index[-1]])
# the last year is incomplete, use it as test set
# there is some missing data and some zeros, fill with the mean and ffill
df.Active[df.Active == 0.0] = df.Active.mean()
df.Active.fillna(method='ffill', inplace=True)

# data exploration after cleaning
print('----------------\nData exploration after cleaning\n----------------')
print(df.info())
print(df.describe())
print(df.head(5))
print('----------------')
plt.plot(df.Active, 'b-', label='Global_active_power in kW')
plt.legend()
plt.show()

# search for outliers: STL decomposition
# study the residuals and seasonality
# seasonal_decompose with extrapolate_trend='freq' to avoid NaNs, a sort of data augmentation
result = seasonal_decompose(df.Active.values, model='additive', period=12, extrapolate_trend='freq')
observed = result.observed
trend = result.trend
seasonal = result.seasonal
resid = result.resid
std = resid.std()
plt.plot(resid, "o", label="datapoints")
outliers = pd.Series(resid)
outliers = pd.concat([outliers[outliers < (-1.5 * std)], outliers[outliers > (1.5 * std)]])
plt.plot(outliers, "*", color='violet', label="outliers")
plt.hlines(0, 0, len(resid))
plt.hlines(1.5 * std, 0, len(resid), color="red", label="std limits")
plt.hlines(-1.5 * std, 0, len(resid), color="red")
plt.title('STL decomposition')
plt.legend()
plt.show()
plt.subplot(2, 1, 1)
plt.plot(observed, label='observed')
plt.plot(trend, label='trend')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(seasonal, label='seasonal')
plt.plot(resid, label='residual')
plt.legend()
plt.show()

# substitute outliers with the mean
plt.plot(df.Active, label='w/outliers')
df.Active[outliers.index] = df.Active.mean()
plt.plot(df.Active, label='w/out/outliers')
plt.legend()
plt.show()


# power transform
# invert a boxcox transform for one value
def invert_boxcox(value, lam):
    # log case
    if lam == 0:
        return exp(value)
    # all other cases
    return exp(log(lam * value + 1) / lam)


# power transform
transformed, lmbda = boxcox(df.Active)
plt.subplot(2, 1, 1)
plt.plot(df.Active, label='data')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(pd.Series(transformed, index=df.Active.index), label='power-transformed')
plt.legend()
plt.show()


# create series
active = pd.Series(transformed, index=df.Active.index)
# split into train and test sets
train_size = int(len(df.Active) * 0.8)
x_train = active[:train_size].values
x_test = active[train_size:].values
y_train = active[:train_size].index
y_test = active[train_size:].index

#####################
#####################
# models
#####################
#####################


# sarima model
sarima_model = pm.auto_arima(x_train, start_p=1, start_q=1, test='adf', max_p=5, max_q=5, m=4,
                             start_P=0, seasonal=True, d=None, D=0, trace=True,
                             error_action='ignore', suppress_warnings=True, stepwise=False,
                             n_jobs=-1)
sarima_model = sarima_model.fit(x_train)
# plot diagnostics
# Top Left: Residual errors should fluctuate around the mean with uniform variance.
# Top right: density plot, should be normal distribution with zero mean.
# Bottom left: Q-Q plot. Points should be aligned, distant points imply skewed distribution.
# Bottom Right: correlogram (ACF plot) of residual errors.
#   Significant autocorrelations indicate patterns in the data not explained by the model.
sarima_model.plot_diagnostics(figsize=(15, 12))
plt.show()
sarima_forecast = sarima_model.predict(n_periods=len(x_test))
sarima_forecast_series = pd.Series([invert_boxcox(x, lmbda) for x in sarima_forecast], index=active[train_size:].index)
plt.figure(figsize=(15, 20))
plt.subplot(6, 1, 1)
plt.plot(df.Active, label='data')
plt.plot(sarima_forecast_series, label='sarima')
plt.title('SARIMA - RMSE: %.2f' % rmse(x_test, sarima_forecast))
plt.legend()

# lstm model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()
# reshape input to be 3D [samples, timesteps, features]
train_active_reshaped = x_train.reshape((len(x_train), 1, 1))
test_active_reshaped = x_test.reshape((len(x_test), 1, 1))
# fit model
lstm_model.fit(train_active_reshaped, train_active_reshaped, epochs=200, verbose=0)
# make a prediction
lstm_forecast = lstm_model.predict(test_active_reshaped)
lstm_forecast_series = pd.Series([invert_boxcox(x, lmbda) for x in lstm_forecast], index=active[train_size:].index)
plt.subplot(6, 1, 2)
plt.plot(df.Active, label='data')
plt.plot(lstm_forecast_series, label='lstm')
plt.title('LSTM - RMSE: %.2f' % rmse(x_test, lstm_forecast.flatten()))
plt.legend()

# mlp model
mlp_model = Sequential()
mlp_model.add(Dense(50, activation='relu', input_dim=1))
mlp_model.add(Dense(1))
mlp_model.compile(optimizer='adam', loss='mse')
mlp_model.summary()
# fit model
mlp_model.fit(train_active_reshaped, train_active_reshaped, epochs=200, verbose=0)
# make a prediction
mlp_forecast = mlp_model.predict(test_active_reshaped)
mlp_forecast_series = pd.Series([invert_boxcox(x, lmbda) for x in mlp_forecast], index=active[train_size:].index)
plt.subplot(6, 1, 3)
plt.plot(df.Active, label='data')
plt.plot(mlp_forecast_series, label='mlp')
plt.title('MLP - RMSE: %.2f' % rmse(x_test, mlp_forecast.flatten()))
plt.legend()


# MLP and LSTM seem to be the best models, but they are trained with a small dataset
# neural networks are good for time series forecasting, but they need a lot of data
# let's try also ML tree models: random forest and XGBoost. They are also hungry for data
# so we expect them to perform worse than neural networks

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
# standardize
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# invert standardization and power transform
def invert_transformations(values):
    res = [scaler.inverse_transform(x.reshape(1, -1)) for x in values]
    res = [invert_boxcox(x, lmbda) for x in res]
    return res

# random forest, ensemble with bagging
rf_model = RandomForestRegressor(max_depth=2, n_estimators=100)
rf_model.fit(x_train, y_train)
rf_forecast = rf_model.predict(x_test)
rf_forecast_series = pd.Series(invert_transformations(rf_forecast), index=y_test)
plt.subplot(6, 1, 4)
plt.plot(df.Active, label='data')
plt.plot(rf_forecast_series, label='rf')
plt.title('Random Forest - RMSE: %.2f' % rmse(x_test.flatten(), rf_forecast))
plt.legend()

# XGBoost, ensemble with boosting
xgb_model = XGBRegressor(n_estimators=100)
xgb_model.fit(x_train, y_train)
xgb_forecast = xgb_model.predict(x_test)
xgb_forecast_series = pd.Series(invert_transformations(xgb_forecast), index=y_test)
plt.subplot(6, 1, 5)
plt.plot(df.Active, label='data')
plt.plot(xgb_forecast_series, label='xgb')
plt.title('XGBoost - RMSE: %.2f' % rmse(x_test.flatten(), xgb_forecast))
plt.legend()

# compare models
plt.subplot(6, 1, 6)
plt.plot(df.Active[train_size:], label='data')
plt.plot(sarima_forecast_series, label='sarima')
plt.plot(lstm_forecast_series, label='lstm')
plt.plot(mlp_forecast_series, label='mlp')
# take out random forest and xgboost from the plot, because they are really worse than the rest
# plt.plot(rf_forecast_series, label='forest')
# plt.plot(xgb_forecast_series, label='xgb')
plt.title('Comparison of models')
plt.legend()
plt.show()

x_test = x_test.flatten()
print('------------------\nMODEL COMPARISON\n------------------')
print('SARIMA - RMSE: %.4f' % rmse(x_test, sarima_forecast))
print('LSTM - RMSE: %.4f' % rmse(x_test, lstm_forecast.flatten()))
print('MLP - RMSE: %.4f' % rmse(x_test, mlp_forecast.flatten()))
print('Random Forest - RMSE: %.4f' % rmse(x_test, rf_forecast))
print('XGBoost - RMSE: %.4f' % rmse(x_test, xgb_forecast))
print(
    'MLPvsLSTM - Diebold-Marino:\n DM: %.4f - p-value: %.4f ' % dm_test(x_test, lstm_forecast.flatten(), mlp_forecast.flatten(), h=1,
                                                        crit='MSE'))
# MLPvsLSTM - DM: p-value: 5.1069 DM: 0.0006
# with 5% significance level, the p-value is less than 0.025 and z-value is greater than 1.96,
# so we can reject the null hypothesis that the two models have the same performance
print(
    'MLP seems better than LSTM, but in the DM test the p-value is not significant,'
    ' so we reject the null hypothesis that the two models have the same performance.')
