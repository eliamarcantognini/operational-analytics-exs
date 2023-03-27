import numpy as np
import pandas as pd
from scipy import stats # to be used later
import matplotlib.pyplot as plt
import os
df = pd.read_csv('traffico16.csv')  # dataframe (series)
npa = df['ago1'].to_numpy()  # numpy array
plt.hist(npa, bins=10, color='#00AA00', edgecolor='black')
plt.title(df.columns[0])
plt.xlabel('num')
plt.ylabel('days')
plt.show()
res = stats.relfreq(npa, numbins=10)  # relative frequency
print(res[0])

for i in np.arange(len(df.columns)):
    npa = df.iloc[:, i].to_numpy()
    print(f"{df.columns[i]}: {npa.mean():.2f} {npa.std():.2f} {npa.min():.2f} {npa.max():.2f}")

df.boxplot(column=['ago1', 'ago2', 'set1', 'set2', 'ott1', 'ott2'])
plt.show()

plt.scatter(df['ago1'].sort_values(), df['set1'].sort_values(), color='red')
plt.show()

plt.scatter(df['ago1'].head(20).sort_values(), df['set1'].head(20).sort_values(), color='red')
plt.show()



