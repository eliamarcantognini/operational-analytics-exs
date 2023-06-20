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
data.drop(columns=['Global_reactive_power', 'Global_intensity', 'Time', 'Voltage', 'Sub_metering_1', 'Sub_metering_2',
                   'Sub_metering_3'],
          inplace=True)
# convert to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
# convert to numeric
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce', downcast='float')
data.rename(columns={'Date': 'Month', 'Global_active_power': 'Active'}, inplace=True)

# sum power consumption per month
data = data.groupby('Month').sum()
# resample to monthly
data = data.groupby(pd.Grouper(freq='M')).sum()
# drop first month (incomplete)
# drop last month (incomplete)
df = data.drop([data.index[0], data.index[-1]])
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
train_set = active[:train_size].values
test_set = active[train_size:].values
train_index = active[:train_size].index
test_index = active[train_size:].index

#####################
#####################
# models
#####################
#####################

plt.figure(figsize=(15, 15))

# sarima model
sarima_model = pm.auto_arima(train_set, start_p=1, start_q=1, test='adf', max_p=5, max_q=5, m=4,
                             start_P=0, seasonal=True, d=None, D=0, trace=True,
                             error_action='ignore', suppress_warnings=True, stepwise=False,
                             n_jobs=-1)
sarima_model = sarima_model.fit(train_set)
sarima_forecast = sarima_model.predict(n_periods=len(test_set))
sarima_forecast_series = pd.Series([invert_boxcox(x, lmbda) for x in sarima_forecast], index=active[train_size:].index)
plt.subplot(4, 1, 1)
plt.plot(df.Active, label='data')
plt.plot(sarima_forecast_series, label='sarima')
plt.title('SARIMA - RMSE: %.2f' % rmse(test_set, sarima_forecast))
plt.legend()

# lstm model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()
# reshape input to be 3D [samples, timesteps, features]
train_active_reshaped = train_set.reshape((len(train_set), 1, 1))
test_active_reshaped = test_set.reshape((len(test_set), 1, 1))
# fit model
lstm_model.fit(train_active_reshaped, train_active_reshaped, epochs=200, verbose=0)
# make a prediction
lstm_forecast = lstm_model.predict(test_active_reshaped)
lstm_forecast_series = pd.Series([invert_boxcox(x, lmbda) for x in lstm_forecast], index=active[train_size:].index)
plt.subplot(4, 1, 2)
plt.plot(df.Active, label='data')
plt.plot(lstm_forecast_series, label='lstm')
plt.title('LSTM - RMSE: %.2f' % rmse(test_set, lstm_forecast.flatten()))
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
plt.subplot(4, 1, 3)
plt.plot(df.Active, label='data')
plt.plot(mlp_forecast_series, label='mlp')
plt.title('MLP - RMSE: %.2f' % rmse(test_set, mlp_forecast.flatten()))
plt.legend()

# compare models
plt.subplot(4, 1, 4)
plt.plot(df.Active[train_size:], label='data')
plt.plot(sarima_forecast_series, label='sarima')
plt.plot(lstm_forecast_series, label='lstm')
plt.plot(mlp_forecast_series, label='mlp')
plt.title('Comparison of models')
plt.legend()
plt.show()

print('------------------\nMODEL COMPARISON\n------------------')
print('SARIMA - RMSE: %.4f' % rmse(test_set, sarima_forecast))
print('LSTM - RMSE: %.4f' % rmse(test_set, lstm_forecast.flatten()))
print('MLP - RMSE: %.4f' % rmse(test_set, mlp_forecast.flatten()))
print(
    'MLPvsLSTM - DM: p-value: %.4f DM: %.4f ' % dm_test(test_set, lstm_forecast.flatten(), mlp_forecast.flatten(), h=1,
                                                        crit='MSE'))
# MLPvsLSTM - DM: p-value: 5.1069 DM: 0.0006
# with 5% significance level, the p-value is less than 0.025 and z-value is greater than 1.96,
# so we can reject the null hypothesis that the two models have the same performance
print(
    'MLP seems better than LSTM, but in the DM test the p-value is not significant,'
    ' so we reject the null hypothesis that the two models have the same performance.')

# MLP and LSTM seem to be the best models, but they are trained with a small dataset
# neural networks are good for time series forecasting, but they need a lot of data
# so let's try ML tree models: random forest and XGBoost

# random forest
rf_model = RandomForestRegressor(n_estimators=500)
rf_model.fit(train_set.reshape(-1, 1), train_index)
rf_forecast = rf_model.predict(test_set.reshape(-1, 1))
rf_forecast_series = pd.Series([invert_boxcox(x, lmbda) for x in rf_forecast], index=test_index)
plt.figure(figsize=(15, 5))
plt.plot(df.Active[train_size:], label='data')
plt.plot(rf_forecast_series, label='rf')
plt.title('Random Forest - RMSE: %.2f' % rmse(test_set, rf_forecast))
plt.legend()
plt.show()

# XGBoost
xgb_model = XGBRegressor(n_estimators=500)
xgb_model.fit(train_set.reshape(-1, 1), train_index)
xgb_forecast = xgb_model.predict(test_set.reshape(-1, 1))
xgb_forecast_series = pd.Series([invert_boxcox(x, lmbda) for x in xgb_forecast], index=test_index)
plt.figure(figsize=(15, 5))
plt.plot(df.Active[train_size:], label='data')
plt.plot(xgb_forecast_series, label='xgb')
plt.title('XGBoost - RMSE: %.2f' % rmse(test_set, xgb_forecast))
plt.legend()
plt.show()
