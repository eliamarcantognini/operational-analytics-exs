import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('..\\gioiellerie.csv', header=0)
df["period"] = df["year"].map(str) + "-" + df["month"].map(str)
df['period'] = pd.to_datetime(df['period'], format="%Y-%m").dt.to_period('M')
# df = df.set_index('period')
aSales = df['sales'].to_numpy()  # array of sales data
logdata = np.log(aSales)  # log transform
data = pd.Series(logdata)  # convert to pandas series
plt.rcParams["figure.figsize"] = (10, 8)  # redefines figure size
plt.plot(data.values);
plt.show()  # data plot
# train and test set
train = data[:-12]
test = data[-12:]
reconstruct = np.exp(np.r_[train, test])  # simple recosntruction

# ------------------------------------------------- neural forecast
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(train.values.reshape(-1, 1))
scaled_train_data = scaler.transform(train.values.reshape(-1, 1))
scaled_test_data = scaler.transform(test.values.reshape(-1, 1))

from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(20, activation='relu', input_shape=(n_input, n_features), dropout=0.05))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(generator, epochs=25)
lstm_model.summary()

losses_lstm = lstm_model.history.history['loss']
plt.xticks(np.arange(0, 21, 1))  # convergence trace
plt.plot(range(len(losses_lstm)), losses_lstm)
plt.show()
lstm_predictions_scaled = list()
batch = scaled_train_data[-n_input:]
curbatch = batch.reshape((1, n_input, n_features))  # 1 dim more
for i in range(len(test)):
    lstm_pred = lstm_model.predict(curbatch)[0]
    lstm_predictions_scaled.append(lstm_pred)
    curbatch = np.append(curbatch[:, 1:, :], [[lstm_pred]], axis=1)

lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled)
yfore = np.transpose(lstm_forecast).squeeze()
# recostruction
expdata = np.exp(train)  # unlog
expfore = np.exp(yfore)
plt.plot(df.sales, label="sales")
plt.plot(expdata, label='expdata')
plt.plot([None for x in expdata] + [x for x in expfore], label='forecast')
plt.legend()
plt.show()
