import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# https://raw.githubusercontent.com/mridulrb/Predict-loan-eligibility-using-IBM-Watson-Studio/master/Dataset/Dataset.csv


df = pd.read_csv("https://raw.githubusercontent.com/mridulrb/Predict-loan-eligibility-using-IBM-Watson-Studio/master/Dataset/Dataset.csv")
# fill missing values with the mean value.
df["LoanAmount"]=df["LoanAmount"].fillna(df["LoanAmount"].mean())
# calculate Pearson correlation coefficient
pcc = np.corrcoef(df.ApplicantIncome, df.LoanAmount)
print(f"r={pcc}")
plt.plot(df['LoanAmount'],df['ApplicantIncome'],marker='.',linewidth=0)
plt.show()
df.Married = df.Married.map({'Yes': 1, 'No': 0})

pcc = np.corrcoef(df['LoanAmount'], df.Married)
print(f"r={pcc}")
plt.plot(df.Married,df.LoanAmount,marker='.',linewidth=0)
plt.show()