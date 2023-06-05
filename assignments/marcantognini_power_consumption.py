import os
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import requests

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
print(data.info())
print(data.shape)
print(data.head(10))
print(data.tail(10))

#####################
#####################
# preprocessing
#####################
#####################
# drop columns
data.drop(columns=['Global_intensity', 'Time', 'Voltage', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
          inplace=True)
# convert to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
# convert to numeric
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce', downcast='float')
data['Global_reactive_power'] = pd.to_numeric(data['Global_reactive_power'], errors='coerce', downcast='float')
data.rename(columns={'Date': 'Month', 'Global_active_power': 'Active', 'Global_reactive_power': 'Reactive'},
            inplace=True)
plt.show()

# sum power consumption per month
data = data.groupby('Month').sum()
# resample to monthly
data = data.groupby(pd.Grouper(freq='M')).sum()
# drop first month (incomplete)
df = data.drop([data.index[0], data.index[-1]])
# drop last month (incomplete)
# df = df.drop(df.index[-1])
# the last year is incomplete, use it as test set

# data exploration after cleaning
print(df.info())
print(df.head(5))
print(df.tail(5))
plt.subplot(2, 1, 1)
plt.plot(df.Active, 'b-', label='Global_active_power in kW')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(df.Reactive, 'r-', label='Global_reactive_power in kW')
plt.legend()
plt.show()

# search for outliers:
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df.Active.values, model='additive', period=12, extrapolate_trend='freq')
observed = result.observed
trend = result.trend
seasonal = result.seasonal
resid = result.resid
std = resid.std()
plt.plot(resid, "o", label="datapoints")
outliers = pd.Series(resid)
outliers = pd.concat([outliers[outliers < (-1.5 * std)], outliers[outliers > (1.5 * std)]])
plt.plot(outliers,  "*", color='violet', label="outliers")
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



# # power transform
# # invert a boxcox transform for one value
# def invert_boxcox(value, lam):
#     # log case
#     if lam == 0:
#         return exp(value)
#     # all other cases
#     return exp(log(lam * value + 1) / lam)
#
#
# # power transform
# transformed, lmbda = boxcox(df.Active)
# print(transformed, lmbda)
# plt.subplot(2, 1, 1)
# plt.plot(df.Active, label='data')
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(pd.Series(transformed, index=df.Active.index), label='power')
# plt.legend()
# plt.show()

# create series
# active = pd.Series(transformed, index=df.Active.index)
#
# # split into train and test sets
# train_active = active[:-10].values
# test_active = active[-10:].values
#
# sarima_model = pm.auto_arima(train_active, start_p=1, start_q=1, test='adf', max_p=5, max_q=5, m=4,
#                              start_P=0, seasonal=True, d=None, D=0, trace=True,
#                              error_action='ignore', suppress_warnings=True, stepwise=False,
#                              n_jobs=-1)
# sarima_forecast = sarima_model.fit(train_active).predict(n_periods=len(test_active))
# forecast = pd.Series(sarima_forecast, index=active[-10:].index)
# plt.subplot(2, 1, 1)
# plt.plot(active, label='data-pow')
# plt.plot(forecast, label='sarima-pow')
# # plt.title('forecast with Sarima')
# plt.legend()
# plt.subplot(2, 1, 2)
# forecast = pd.Series([invert_boxcox(x, lmbda) for x in sarima_forecast], index=active[-10:].index)
# plt.plot(df.Active, label='data')
# plt.plot(forecast, label='sarima')
# plt.legend()
# plt.show()
