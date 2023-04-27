import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

df = pd.read_csv('../res/monthly_milk_production.csv', header=0)
dm = df.Production.to_numpy()

cutpoint = int(len(dm) * 0.7)
train = dm[:cutpoint]
test = dm[cutpoint:]

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
plt.plot(dm, label='ds')
plt.plot(ypred, label='ypred')
plt.plot([None for i in ypred] + [x for x in yfore], label='yfore')
plt.xlabel('time')
plt.ylabel('sales')
plt.legend()
plt.show()