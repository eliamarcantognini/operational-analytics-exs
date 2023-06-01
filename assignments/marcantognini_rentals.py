import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# Function for model evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return train_rmse, test_rmse


# Loading data from CSV file
data = pd.read_csv("..\\res\\milano_rents.csv")

# Data Augmentation
date_range = pd.date_range(start='1990-02-01', end='2023-04-01', freq='MS')
augmented_data = pd.DataFrame({'Date': date_range})
data['Date'] = pd.to_datetime(data['Month'], format='%M-%Y')
data = data.drop(columns=['Month'])
data = data.set_index('Date')
augmented_data = augmented_data.set_index('Date')
data = pd.concat([augmented_data, data], axis=1, join='outer').fillna(method='ffill')

# Differencing to remove trend and seasonality
data['PriceM2_diff'] = data['PriceM2'].diff().fillna(0)
data['PriceM2_diff'] = data['PriceM2_diff'].diff(12).fillna(0)

# Normalizing the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data[['PriceM2_diff']])

# Splitting the data into training and test sets
train_size = len(data_normalized) - 60
train_data, test_data = data_normalized[:train_size], data_normalized[train_size:]

# Preparing data for model training
look_back = 12
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Implementing prediction models
# SARIMA model
sarima_model = SARIMAX(data['PriceM2_diff'], order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))
sarima_train_rmse, sarima_test_rmse = evaluate_model(sarima_model, X_train, y_train, X_test, y_test)

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100)
rf_train_rmse, rf_test_rmse = evaluate_model(rf_model, X_train, y_train, X_test, y_test)

# MLP model
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
mlp_train_rmse, mlp_test_rmse = evaluate_model(mlp_model, X_train, y_train, X_test, y_test)

# LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(100, input_shape=(look_back, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

train_predict = lstm_model.predict(X_train)
test_predict = lstm_model.predict(X_test)
lstm_train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
lstm_test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

# Comparing model performances
print("SARIMA Train RMSE:", sarima_train_rmse)
print("SARIMA Test RMSE:", sarima_test_rmse)

print("Random Forest Train RMSE:", rf_train_rmse)
print("Random Forest Test RMSE:", rf_test_rmse)

print("MLP Train RMSE:", mlp_train_rmse)
print("MLP Test RMSE:", mlp_test_rmse)

print("LSTM Train RMSE:", lstm_train_rmse)
print("LSTM Test RMSE:", lstm_test_rmse)
