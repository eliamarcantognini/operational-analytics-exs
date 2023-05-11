import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import pmdarima as pm
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('..\\res\\GlobalLandTemperaturesByCountry.csv', header=0)



