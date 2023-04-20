# Sliding window MLP, Airline Passengers dataset (predicts t+1)
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential # pip install keras
from keras.layers import Dense # pip install tensorflow (as administrator)
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
df = pd.read_csv('..\\monthly_milk_production.csv', usecols=[1], names=['Production'], header=0)
dataset = df.values # time series values
dataset = dataset.astype('float32') # needed for MLP input
dataset = df.values  # time series values
dataset = dataset.astype('float32')  # needed for MLP input

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
