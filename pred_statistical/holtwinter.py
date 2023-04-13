import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simple_exp_smooth(data, nforecasts=1, alpha=0.4):
    n = len(data)
    f = np.full(n + nforecasts, np.nan)  # Forecast array
    data = np.append(data, [np.nan] * nforecasts)  # forecast placeholders
    f[1] = data[0]  # initialization of first forecast
    # predictions
    for t in range(2, n + 1):
        f[t] = alpha * data[t - 1] + (1 - alpha) * f[t - 1]
    # forecast
    for t in range(n + 1, n + nforecasts):
        f[t] = alpha * f[t - 1] + (1 - alpha) * f[t - 2]
    return pd.DataFrame.from_dict({"Data": data, "Forecast": f, "Error": data - f})


sales = pd.read_csv("FilRouge.csv", usecols=["sales"]).T
sales = np.array(sales).flatten()
df = simple_exp_smooth(sales, nforecasts=4, alpha=0.5)
MAE = df["Error"].abs().mean()
print("MAE:", round(MAE, 2))
RMSE = np.sqrt((df["Error"] ** 2).mean())
print("RMSE:", round(RMSE, 2))
df.index.name = "Periods"
plt.figure(figsize=(8, 3))
plt.plot(df[["Data"]], label="data")
plt.plot(df[["Forecast"]], label="Simple smoothing")
plt.legend()
plt.show()

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# fit model
model = ExponentialSmoothing(train, seasonal_periods=4, trend="add",
                             seasonal="mul",
                             damped_trend=True,
                             use_boxcox=True,
                             initialization_method="estimated")
hwfit = model.fit()
# make forecast
yfore = hwfit.predict(len(train), len(train) + 3)
print(yfore)
