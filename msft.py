import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('res/MSFT.csv', sep=';')
plt.plot(df['Chiusura aggiustata**'], range(0,98), label='Chiusura aggiustata**')
plt.legend()
plt.show()