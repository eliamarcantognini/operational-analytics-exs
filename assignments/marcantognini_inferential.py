import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kstest
from scipy.stats import shapiro
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency
from statsmodels.graphics.gofplots import qqplot

# https://raw.githubusercontent.com/mridulrb/Predict-loan-eligibility-using-IBM-Watson-Studio/master/Dataset/Dataset.csv


df = pd.read_csv(
    "https://raw.githubusercontent.com/mridulrb/Predict-loan-eligibility-using-IBM-Watson-Studio/master/Dataset/Dataset.csv")
# fill missing values with the mean value.
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
# calculate Pearson correlation coefficient
pcc = np.corrcoef(df.ApplicantIncome, df.LoanAmount)
print(f"r={pcc}")
plt.plot(df.LoanAmount, df.ApplicantIncome, marker='.', linewidth=0)
plt.title('LoanAmount vs ApplicantIncome')
plt.show()
# q-q plot
qqplot(df.LoanAmount, line='q')
plt.show()

# calculate Pearson correlation coefficient
married = df.Married.map({'Yes': 1, 'No': 0})
pcc = np.corrcoef(df.LoanAmount, married)
print(f"r[pearson]={pcc}")
# pearson not valid for categorical data
# use point biserial correlation coefficient
married.fillna(0, inplace=True)
pbc = pointbiserialr(df.LoanAmount, married)
print(f"r[pointbiserial]={pbc}")
sns.catplot(x="Married", y="LoanAmount", data=df)
plt.show()
sns.catplot(x="Married", y="LoanAmount", kind="box", data=df)
plt.show()

# calculate Chi square test for independence
# feature matrix via crosstab (frequency table of the factors)
alpha = 0.05
dataset_table = pd.crosstab(df["Gender"], df["Education"])
# chi2_contingency() yields the p value for rejecting or accepting null hypothesis
chi_square, p, dof, expected = chi2_contingency(dataset_table)
print("chi-square statistic:-", chi_square)
print('Significance level: ', alpha)
print('Degree of Freedom: ', dof)
print('p-value:', p)
if p <= alpha:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypthesis")

# normality test (Shapiro-Wilk)
alpha = 0.05
stat, p = shapiro(df.LoanAmount)
print('Shapiro=%.3f, p=%.3f, alpha=%.3f' % (stat, p, alpha))
if p > alpha:
    print('Campione gaussiano (non rifiuto H0)')
else:
    print('Campione NON gaussiano (rifiuto H0)')

# normality test (Kolmogorov-Smirnov)
stat, p = kstest(df.LoanAmount, 'norm')
print('Kolmogorov=%.3f, p=%.3f, alpha=%.3f' % (stat, p, alpha))
if p > alpha:
    print('Campione gaussiano (non rifiuto H0)')
else:
    print('Campione NON gaussiano (rifiuto H0)')

from scipy.stats import sem, t


# function for calculating the t-test for two independent samples
def ttest(dataa1, dataa2, alpha):
    mean1, mean2 = np.mean(dataa1), np.mean(dataa2)  # means
    se1, se2 = sem(dataa1), sem(dataa2)  # standard errors
    sed = np.sqrt(se1 ** 2.0 + se2 ** 2.0)  # standard error on diff between samples
    t_stat = (mean1 - mean2) / sed  # t statistic
    degf = len(dataa1) + len(dataa2) - 2  # degrees of freedom
    cv = t.ppf(1.0 - alpha, degf)  # critical value, ppf percent point function inv. of cdf
    p = (1.0 - t.cdf(abs(t_stat), degf)) * 2.0  # p-value
    return t_stat, degf, cv, p


# t test function call
data1 = df.loc[df['Gender'] == 'Male', 'ApplicantIncome']
data2 = df.loc[df['Gender'] == 'Female', 'ApplicantIncome']
alpha = 0.05
t, degf, cv, p = ttest(data1, data2, alpha)
if p < alpha:
    print('Reject null ipothesis')
else:
    print('Accept null ipothesis')
print(t, degf, cv, p)
