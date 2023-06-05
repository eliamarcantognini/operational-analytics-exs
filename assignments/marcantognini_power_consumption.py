import os
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
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
df = data.drop(data.index[0])
# drop last month (incomplete)
df = data.drop(data.index[-1])
# the last year is incomplete, use it as test set

# data exploration after cleaning
print(df.info())
print(df.head(5))
print(df.tail(5))
plt.subplot(2, 1, 1)
plt.plot(df.Active, 'b-', label='Global_active_power in kW')
plt.subplot(2, 1, 2)
plt.plot(df.Reactive, 'r-', label='Global_reactive_power in kW')
plt.legend()
plt.show()

# create series
active = df.Active
reactive = df.Reactive

# split into train and test sets
train_active = active[:-10]
test_active = active[-10:]
train_reactive = reactive[:-10]
test_reactive = reactive[-10:]

active_sar_notNorm = pm.auto_arima(train_active, start_p=1, start_q=1, test='adf', max_p=5, max_q=5, m=12,
                               start_P=0, seasonal=True, d=None, D=0, trace=True,
                               error_action='ignore', suppress_warnings=False, stepwise=True,
                               n_jobs=-1)
active_sar_notNorm_fore = active_sar_notNorm.fit(train_active).predict(n_periods=len(test_active))
active_fore = pd.Series(active_sar_notNorm_fore, index=test_active.index)
plt.plot(active, label='data')
plt.plot(active_fore, label='sarima')
plt.title('forecast with Sarima')
plt.legend()
plt.show()
# normalize data: try to compare normalized data and not normalized
# scaler = MinMaxScaler()
# active = scaler.fit_transform(active)
# reactive = scaler.fit_transform(reactive.reshape(-1, 1))
# print(active)
# print(reactive)
