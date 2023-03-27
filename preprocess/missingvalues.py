#Using pandas, simple imputations
import pandas as pd
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 12, 13, 14, 15, 16, 17, 18, 19, 20]
ds = pd.Series(y)
ds1=ds.fillna(method="ffill")
ds2=ds.fillna(method="bfill")
ds3=ds.fillna(ds.mean())
ds4=ds.interpolate()