import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def difference(dataset, interval):
    return [dataset[i] - dataset[i - interval] for i in range(interval, len(dataset))]


# invert difference
def invert_difference(orig_data, diff_data, interval):
    return [diff_data[i - interval] + orig_data[i - interval] for i in range(interval, len(orig_data))]


# # slide sketch
# # define dataset
# data = [x for x in range(1, 10)]
# print(data)
# # difference transform
# transformed = difference(data, 1)
# print(transformed)
# # invert difference
# inverted = invert_difference(data, transformed, 1)
# print(inverted)

# dataset
dataset = pd.read_csv('..\\res\\BoxJenkins.csv')
data = dataset['Passengers'].values

# instance plot
plt.figure(figsize=(10, 7))

# log transform
plt.subplot(4, 1, 1)
log = np.log(data)
plt.plot(log, label='Log')
plt.legend()

# log diff transform
plt.subplot(4, 1, 2)
logdiff = np.diff(log)
plt.plot(logdiff, label='LogDiff')
plt.legend()

# difference transform
plt.subplot(4, 1, 3)
transformed = difference(data, 1)
plt.plot(transformed, label='Diff')
inverted = invert_difference(data, transformed, 1)
plt.plot(inverted, label='Inverted')
plt.legend()

# scaled
plt.subplot(4, 1, 4)
scaler = StandardScaler()
scaled = scaler.fit_transform(logdiff.reshape(len(logdiff), 1))
scaled_inverted = scaler.inverse_transform(scaled)
log_inverted = invert_difference(log, scaled_inverted, 1)
data_reconstructed = np.exp(log_inverted)
plt.plot(data, label='Original[scaled]')
plt.plot(data_reconstructed, label='Reconstructed[scaled]')
plt.legend()

# plot show
plt.show()
