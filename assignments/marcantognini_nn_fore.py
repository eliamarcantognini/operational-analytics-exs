import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def forecast(model, train, test, n_input, n_features):
    predictions = np.empty(0)
    batch = train[-n_input:]
    curbatch = batch.reshape((1, n_input, n_features))  # 1 dim more
    for i in range(len(test)):
        pred = model.predict(curbatch)[0]
        predictions = np.append(predictions, pred)
        curbatch = np.append(curbatch[:, 1:, :], [[pred]], axis=1)
    return predictions


os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('..\\res\\co2_emissions_kt_by_country.csv', header=0)

print('before cleaning')
print(df.info())
df.drop(['country_code'], axis=1, inplace=True)
df.drop(df[df['country_name'] != 'Arab World'].index, inplace=True)
df.drop(['country_name'], axis=1, inplace=True)
print('after cleaning')
print(df.info())
print(df.describe())
print(df.head(20))

df.plot(x='year', y='value', title='CO2 emissions by Arab World')
plt.show()

dataset = df.value.values.astype('float32')
n_forecast = 12
train = dataset[:-n_forecast]
test = dataset[-n_forecast:]

# sarima model
Smodel = pm.auto_arima(train, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=12,
                       start_P=0, seasonal=True, d=None, D=0, trace=True,
                       error_action='ignore', suppress_warnings=True, stepwise=True)  # False full grid

sarima_fore = Smodel.fit(train).predict(n_periods=len(test))

# scale data
scaler = StandardScaler()
scaler.fit_transform(train.reshape(-1, 1))
scaled_train = scaler.transform(train.reshape(-1, 1))
scaled_test = scaler.transform(test.reshape(-1, 1))

# lstm model
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_forecast, batch_size=n_features)
lstm_model = Sequential()
lstm_model.add(LSTM(25, activation='relu', input_shape=(n_forecast, n_features), dropout=0.05))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(generator, epochs=70)
lstm_model.summary()

test_data = np.concatenate((scaled_train[-n_forecast:], scaled_test))
trainX, trainY = create_dataset(scaled_train, n_forecast)
testX, testY = create_dataset(test_data, n_forecast)

lstm_fore_scaled = forecast(lstm_model, scaled_train, test_data, n_forecast, n_features)
lstm_fore = scaler.inverse_transform(lstm_fore_scaled.reshape(-1, 1)).reshape(-1)

# mlp
mlp_model = Sequential()
mlp_model.add(Dense(25, activation='relu', input_shape=(n_forecast,)))
mlp_model.add(Dense(1))
mlp_model.compile(optimizer='adam', loss='mse')
mlp_model.fit(trainX, trainY, epochs=60, verbose=0, batch_size=4)
mlp_model.summary()

mlp_fore_scaled = mlp_model.predict(testX)
mlp_fore = scaler.inverse_transform(mlp_fore_scaled.reshape(-1, 1)).reshape(-1)

# plot
plt.subplot(3, 1, 1)
plt.plot(dataset, label='data')
plt.plot([None for x in train] + [x for x in sarima_fore], label='sarima')
plt.title('forecast-sarima')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(dataset, label='data')
plt.plot([None for x in train] + [x for x in lstm_fore], label='lstm')
plt.title('forecast-lstm')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(dataset, label='data')
plt.plot([None for x in train] + [x for x in mlp_fore], label='mlp')
plt.title('forecast-mlp')
plt.legend()
plt.show()
