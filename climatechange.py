import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import pmdarima as pm
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('res\\GlobalLandTemperaturesByMajorCity.csv', header=0)

print('before cleaning')
print(df.info())
df = df[df['City'] == 'Rome']
df.drop(['Latitude', 'Longitude', 'City', 'Country', 'AverageTemperatureUncertainty'], axis=1, inplace=True)
df['dt'] = pd.to_datetime(df['dt'])
df.drop(df[df.dt < '1900-01-01'].index, inplace=True)
df.drop(df[df.dt >= '2013-01-01'].index, inplace=True)
# df.set_index('dt', inplace=True)
print('after cleaning')
print(df.info())

df.plot(x='dt', y='AverageTemperature', title='Rome temperature')
plt.show()

for i in range (1, 13):
    df[df.dt.dt.month == i].plot(x='dt', y='AverageTemperature', title='Rome temperature in month ' + str(i))
    plt.show()

result = seasonal_decompose(df.AverageTemperature, period=12)
plt.plot(result.trend)
plt.title('trend seasonal decomposition')
plt.show()
print(result.seasonal)

cutpoint = int(len(df) * 0.7)
train = df[:cutpoint]
test = df[cutpoint:]

model = pm.auto_arima(train, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=4, start_P=0, seasonal=True,
                      d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True,
                      stepwise=True)  # stepwise=False full grid

print(model.summary())
morder = model.order  # p,d,q
print("Sarimax order {0}".format(morder))
mseasorder = model.seasonal_order  # P,D,Q,m
print("Sarimax seasonal order {0}".format(mseasorder))

fitted = model.fit(train)
yfore = fitted.predict(n_periods=12)  # forecast
ypred = fitted.predict_in_sample()
plt.plot(df, label='df')
plt.plot(ypred, label='ypred')
plt.plot([None for i in ypred] + [x for x in yfore], label='yfore')
plt.xlabel('time')
plt.ylabel('sales')
plt.legend()
plt.show()

