import numpy as np, pandas as pd, matplotlib.pyplot as plt, pmdarima as pm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # correlation coeff
    mins = np.amin(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)  # minmax
    acf1 = acf(forecast - actual)[1]  # ACF1
    return ({'mape': mape, 'me': me, 'mae': mae, 'mpe': mpe, 'rmse': rmse,
             'acf1': acf1, 'corr': corr, 'minmax': minmax})


# dataset
df = pd.read_csv('..\\res\\monthly_milk_production.csv', header=0)
data = df.Production
plt.plot(data, label='data')
plt.legend()
plt.show()

# Autogression
ARmodel = AutoReg(data, lags=1)
ARmodel_fit = ARmodel.fit()
yhat = ARmodel_fit.predict(len(data), len(data))
print('ARmodel: ', yhat)

# # Vector Autoregression
# VARmodel = VAR(data)
# VARmodel_fit = VARmodel.fit()
# yhat = VARmodel_fit.forecast(VARmodel_fit.y, steps=1)
# print(yhat)

# Moving Average
MAmodel = ARIMA(data, order=(0, 0, 1))
MAmodel_fit = MAmodel.fit()
yhat = MAmodel_fit.forecast()
print('MAmodel: ', yhat)

# ARMA
ARMAmodel = ARIMA(data, order=(2, 0, 1))
ARMAmodel_fit = ARMAmodel.fit()
yhat = ARMAmodel_fit.predict(len(data), len(data))
print('ARMAmodel: ', yhat)
test = data[-12:]
yhat = ARMAmodel_fit.forecast(len(test))
error = mse(test, yhat)
print('ARMAmodel error: ', error)

# ARIMA
ARIMAmodel = ARIMA(data, order=(1, 1, 2))
ARIMAmodel_fit = ARIMAmodel.fit()
print(ARIMAmodel_fit.summary())
# diff=1
pred = ARIMAmodel_fit.predict(1, len(data), typ='levels')
plt.plot(data, label='data')
plt.plot(pred, label='pred')
plt.legend()
plt.show()
# plot residual errors
residuals = pd.DataFrame(ARIMAmodel_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.legend()
plt.show()

# SARIMA (pmdarima)
cutpoint = int(len(data) * 0.7)
train = data[:cutpoint]
test = data[cutpoint:]
reconstruct = np.r_[train, test].cumsum()
model = pm.auto_arima(train, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=4, start_P=0, seasonal=True,
                      d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True,
                      stepwise=True)  # stepwise=False full grid
print(model.summary())
morder = model.order  # p,d,q
print("Sarimax order {0}".format(morder))
mseasorder = model.seasonal_order  # P,D,Q,m
print("Sarimax seasonal order {0}".format(mseasorder))
fitted = model.fit(train)
yfore = fitted.predict(n_periods=12)  # forecast
ypred = fitted.predict_in_sample()
plt.plot(data, label='ds')
plt.plot(ypred, label='ypred')
plt.plot([None for i in ypred] + [x for x in yfore], label='yfore')
plt.xlabel('time')
plt.ylabel('sales')
plt.legend()
plt.show()
n_periods = 4
fitted, confint = model.predict(n_periods=n_periods, return_conf_int=True)
tindex = np.arange(df.index[-1] + 1, df.index[-1] + n_periods + 1)
# series, for plotting
fitted_series = pd.Series(fitted, index=tindex)
lower_series = pd.Series(confint[:, 0], index=tindex)
upper_series = pd.Series(confint[:, 1], index=tindex)
# Plot
plt.plot(data)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title("SARIMAX")
plt.show()

# SARIMAX (statsmodel)
SARIMAXmodel = SARIMAX(data, order=(0, 1, 2), seasonal_order=(0, 1, 0, 12))
SARIMAXmodel_fit = SARIMAXmodel.fit()
print(SARIMAXmodel_fit.summary())
SARIMAXmodel_fit.plot_diagnostics()
plt.show()
ypred = SARIMAXmodel_fit.predict(start=0, end=len(data))
n_forecast = 12
forewrap = SARIMAXmodel_fit.get_forecast(steps=n_forecast)
forecast_ci = forewrap.conf_int()
forecast_val = forewrap.predicted_mean
plt.figure(1)
plt.plot(data, label="data")
plt.plot(ypred, label="pred")
plt.fill_between(np.linspace(len(data), len(data) + n_forecast, n_forecast), forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='k', alpha=0.25)
plt.plot(np.linspace(len(data), len(data) + n_forecast, n_forecast), forecast_val, label="forecast")
plt.xlabel('time')
plt.ylabel('production')
plt.title('SARIMAX')
plt.legend()

sm.graphics.tsa.plot_acf(data, lags=40)
plt.show()



