import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
import math
from keras.models import Sequential # pip install keras
from keras.layers import Dense # pip install tensorflow (as administrator)

df = pd.read_csv('..\\res\\gioiellerie.csv', header=0)
df["period"] = df["year"].map(str) + "-" + df["month"].map(str)
df['period'] = pd.to_datetime(df['period'], format="%Y-%m").dt.to_period('M')
# df = df.set_index('period')
aSales = df['sales'].to_numpy()  # array of sales data
logdata = np.log(aSales)  # log transform
data = pd.Series(logdata)  # convert to pandas series
plt.rcParams["figure.figsize"] = (10, 8)  # redefines figure size
plt.plot(data.values)
plt.show()  # data plot
# train and test set
train = data[:-12]
test = data[-12:]
reconstruct = np.exp(np.r_[train, test])  # simple recosntruction

# ------------------------------------------------- neural forecast
scaler = MinMaxScaler()
scaler.fit_transform(train.values.reshape(-1, 1))
scaled_train_data = scaler.transform(train.values.reshape(-1, 1))
scaled_test_data = scaler.transform(test.values.reshape(-1, 1))

n_input = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)

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


# Sliding window MLP, Airline Passengers dataset (predicts t+1)
# from series of values to windows matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

np.random.seed(550) # for reproducibility
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('..\\res\\BoxJenkins.csv', usecols=[1], names=['Passengers'], header=0)
dataset = df.values # time series values
dataset = dataset.astype('float32') # needed for MLP input

# split into train and test sets
train_size = int(len(dataset) -12)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print("Len train={0}, len test={1}".format(len(train), len(test)))

# sliding window matrices (look_back = window width); dim = n - look_back - 1
look_back = 2
testdata = np.concatenate((train[-look_back:],test))
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(testdata, look_back)

# Multilayer Perceptron model
loss_function = 'mean_squared_error'
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu')) # 8 hidden neurons
model.add(Dense(1)) # 1 output neuron
model.compile(loss=loss_function, optimizer='adam')
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(trainScore,
math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(testScore, math.sqrt(testScore)))

# generate predictions for training and forecast for plotting
trainPredict = model.predict(trainX)
testForecast = model.predict(testX)
plt.plot(dataset, label="dataset")
plt.plot(np.concatenate((np.full(look_back-1, np.nan), trainPredict[:,0])), label="trainPredict")
plt.plot(np.concatenate((np.full(len(train)-1, np.nan), testForecast[:, 0])), label="testForecast")
plt.legend()
plt.show()



