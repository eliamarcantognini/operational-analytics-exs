import os
import numpy as np
import requests
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
data.drop(columns=['Global_intensity', 'Time', 'Voltage', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], inplace=True)
# convert to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
# convert to numeric
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce', downcast='float')
data['Global_reactive_power'] = pd.to_numeric(data['Global_reactive_power'], errors='coerce', downcast='float')
data.rename(columns={'Date': 'Month', 'Global_active_power': 'Active', 'Global_reactive_power': 'Reactive'}, inplace=True)
plt.show()

# sum power consumption per month
data = data.groupby('Month').sum()
# resample to monthly
data = data.groupby(pd.Grouper(freq='M')).sum()
# drop first month (incomplete)
data.drop(data.index[0], inplace=True)
# drop last month (incomplete)
data.drop(data.index[-1], inplace=True)
# the last year is incomplete, use it as test set

# data exploration after cleaning
print(data.info())
print(data.head(5))
print(data.tail(5))
plt.subplot(2, 1, 1)
plt.plot(data.Active, 'b-', label='Global_active_power in kW')
plt.subplot(2, 1, 2)
plt.plot(data.Reactive, 'r-', label='Global_reactive_power in kW')
plt.legend()
plt.show()


# create series
active = data.Active.values
reactive = data.Reactive.values
# normalize data
scaler = MinMaxScaler()
active = scaler.fit_transform(active.reshape(-1, 1))
reactive = scaler.fit_transform(reactive.reshape(-1, 1))
print(active)
print(reactive)

# split into train and test sets
train_active = active[:-10]
test_active = active[-10:]
train_reactive = reactive[:-10]
test_reactive = reactive[-10:]

