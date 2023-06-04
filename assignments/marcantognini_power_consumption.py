import requests
import zipfile
import pandas as pd
import numpy as np

#####################
# data loading
#####################
# download data
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip')
with open('..\\res\\household_power_consumption.zip', 'wb') as output_file:
    output_file.write(r.content)
# unzip data
with zipfile.ZipFile('..\\res\\household_power_consumption.zip', 'r') as zip_ref:
    zip_ref.extractall('..\\res\\')
# load data
data = pd.read_csv('..\\res\\household_power_consumption.txt', sep=';', header=0, low_memory=False)
# data exploration
print(data.info())
print(data.shape)
print(data.head(10))
print(data.tail(10))

#####################
# data cleaning
#####################
# drop columns
data.drop(columns=['Global_intensity', 'Time', 'Voltage', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], inplace=True)
# convert to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
# convert to numeric
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce')
data['Voltage'] = pd.to_numeric(data['Global_reactive_power'], errors='coerce')
print(data.info())
print(data.head(5))
# sum power consumption per day
data = data.groupby('Date').sum()
print(data.head(5))
