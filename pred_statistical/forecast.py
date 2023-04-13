import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm

# preprocessing, log-diff
df = pd.read_csv('FilRouge.csv', header=0)
ds = df['sales'].to_numpy()  # array of sales data
logdata = np.log(ds)  # log transform
logdiff = pd.Series(logdata).diff()  # logdiff transform

# inferential, train-test split
cutpoint = int(0.7 * len(logdiff))  # example, cut where needed
train = logdiff[:cutpoint]
test = logdiff[cutpoint:]

# postprocessing, reconstruction
train[0] = 0  # set first entry
reconstruct = np.exp(np.r_[train, test].cumsum() + logdata[0])

model = pm.auto_arima(train, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=4, start_P=0, seasonal=True,
                      d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True,
                      stepwise=True)  # stepwise=False full grid
print(model.summary())
morder = model.order  # p,d,q
print("Sarimax order {0}".format(morder))
mseasorder = model.seasonal_order  # P,D,Q,m
print("Sarimax seasonal order {0}".format(mseasorder))

fitted = model.fit(train)
yfore = fitted.predict(n_periods=4)  # forecast
ypred = fitted.predict_in_sample()
plt.plot(ds, label='ds')
plt.plot(ypred, label='ypred')
plt.plot([None for i in ypred] + [x for x in yfore], label='yfore')
plt.xlabel('time')
plt.ylabel('sales')
plt.legend()
plt.show()
#
# data = ds.values
# n_periods = 4
# fitted, confint = model.predict(n_periods=n_periods, return_conf_int=True)
# tindex = np.arange(df.index[-1] + 1, df.index[-1] + n_periods + 1)
# # series, for plotting
# fitted_series = pd.Series(fitted, index=tindex)
# lower_series = pd.Series(confint[:, 0], index=tindex)
# upper_series = pd.Series(confint[:, 1], index=tindex)
# # Plot
# plt.plot(data)
# plt.plot(fitted_series, color='darkgreen')
# plt.fill_between(lower_series.index, lower_series, upper_series,
#                  color='k', alpha=.15)
# plt.title("SARIMAX")
# plt.show()

# from statsmodels.tsa.statespace.sarimax import SARIMAX
#
# sarima_model = SARIMAX(ds, order=(0, 2, 2), seasonal_order=(0, 1, 0, 4))
# sfit = sarima_model.fit()
# print("sfit:", sfit.summary())
# sfit.plot_diagnostics(figsize=(10, 6))
# plt.show()
#
# ypred = sfit.predict(start=0, end=len(df))
# plt.plot(df.sales)
# plt.plot(ypred)
# plt.title('ypred')
# plt.xlabel('time')
# plt.ylabel('sales')
# plt.show()
#
# sfit.plot_diagnostics(figsize=(7, 5))
# plt.show()
#
# forewrap = sfit.get_forecast(steps=4)
# forecast_ci = forewrap.conf_int()
# forecast_val = forewrap.predicted_mean
# print("forecast_ci", forecast_ci)
# print("forecast_val", forecast_val)
# plt.plot(df.sales)
# # plt.fill_between(np.linspace(len(df), len(df) + 4, 4), forecast_ci[:, 0], forecast_ci[:, 1], color='k', alpha=.25)
# # plt.plot(np.linspace(len(df), len(df) + 4, 4), forecast_val)
# # plt.xlabel('time')
# # plt.ylabel('sales')
# plt.show()
