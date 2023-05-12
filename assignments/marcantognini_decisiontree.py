import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import plot_tree
from xgboost import XGBRegressor

df = pd.read_csv('..\\res\\monthly_milk_production.csv')

# create rolling window dataset
window_size = 12
dataset = pd.DataFrame({'t-' + str(i): df['Production'].shift(i) for i in range(window_size, 0, -1)})
dataset['t'] = df['Production'].values
dataset = dataset[window_size:]

x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]
x_train, xtest = x[:-window_size], x[-window_size:]
y_train, ytest = y[:-window_size], y[-window_size:]

# XGB model
XGBmodel = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
XGBmodel.fit(x_train, y_train)
XGBpred = XGBmodel.predict(xtest)
XGBmse = mean_absolute_error(ytest, XGBpred)
print("XGB MSE={}".format(XGBmse))

# RF model
RFmodel = RandomForestRegressor(n_estimators=500, random_state=1)
RFmodel.fit(x_train, y_train)
RFpred = RFmodel.predict(xtest)
RFmse = mean_absolute_error(ytest, RFpred)
print("RF MSE={}".format(RFmse))

# plot
plt.subplot(2, 1, 1)
plt.plot(y_train.index, y_train.values, label='train')
plt.plot(ytest.index, ytest.values, label='Actual')
plt.plot(ytest.index, XGBpred, '-*', label='XGboost')
plt.title('Milk Production forecast with XGboost')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(y_train.index, y_train.values, label='train')
plt.plot(ytest.index, ytest.values, label='Actual')
plt.plot(ytest.index, RFpred, '-*', label='RandomForest')
plt.title('Milk Production forecast with RandomForest')
plt.legend()
plt.show()

# Stats about the trees in random forest
n_nodes = []
max_depths = []
for ind_tree in RFmodel.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')
# plot first tree (index 0)
fig = plt.figure(figsize=(15, 10))
plot_tree(RFmodel.estimators_[0], max_depth=2, feature_names=dataset.columns[:-1],
          class_names=dataset.columns[-1], filled=True, impurity=True, rounded=True)
plt.show()
