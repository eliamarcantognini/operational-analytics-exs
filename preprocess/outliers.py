import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("../traffico16.csv")
col = "ago2"
series = df[col].fillna(value=df[col].mean())
result = seasonal_decompose(series, model='additive', period=7)
observed = result.observed
trend = result.trend
seasonal = result.seasonal
resid = result.resid
std = resid.std()
plt.figure(figsize=(10, 5))
plt.plot(resid, "o", label="datapoints")
plt.hlines(0, 0, len(resid))
plt.hlines(1.5 * std, 0, len(resid), color="red", label="std limits")
plt.hlines(-1.5 * std, 0, len(resid), color="red")
plt.legend()
plt.show()
