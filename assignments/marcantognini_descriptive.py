import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta
from scipy.stats import chi2
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import norm

df = pd.read_csv('..\\res\\traffico16.csv')  # dataframe (series)
print(df.describe())
npa = df['ago1'].to_numpy()  # numpy array
plt.hist(npa, bins=10, color='#00AA00', edgecolor='black')
plt.title(df.columns[0])
plt.xlabel('num')
plt.ylabel('days')
plt.show()
res = stats.relfreq(npa, numbins=10)  # relative frequency
print(res[0])

for c in df.columns:
    print(f"{c}: {df[c].mean():.2f} {df[c].std():.2f} {df[c].min():.2f} {df[c].max():.2f}")

df.boxplot(column=['ago1', 'ago2', 'set1', 'set2', 'ott1', 'ott2'])
plt.show()

plt.scatter(df['ago1'].sort_values(), df['set1'].sort_values(), color='red')
plt.show()

plt.scatter(df['ago1'].head(20).sort_values(), df['set1'].head(20).sort_values(), color='red')
plt.show()
set1 = df.set1
ago1_range = np.arange(set1.min(), set1.max(), 1)
plt.plot(ago1_range, norm.pdf(ago1_range, set1.mean(), set1.std()), label='set1 norm')
plt.title('set1')
plt.legend()
plt.show()

# EXAMPLES FROM SLIDES
# normal distribution
dom = np.arange(-5, 5, 0.001)
mean = 0.0
plt.plot(dom, norm.pdf(dom, mean, 1), label="std=1")
plt.plot(dom, norm.pdf(dom, mean, 0.5), label="std=0.5")
plt.plot(dom, norm.pdf(dom, mean, 2), label="std=2")
plt.title("Normal distribution")
plt.legend()
plt.show()
plt.plot(dom, norm.cdf(dom, mean, 1), label="std=1")
plt.plot(dom, norm.cdf(dom, mean, 0.5), label="std=0.5")
plt.plot(dom, norm.cdf(dom, mean, 2), label="std=2")
plt.title("Normal distribution")
plt.legend()
plt.show()

# gamma distribution
dom = np.arange(0, 20, 0.001)
plt.plot(dom, gamma.pdf(dom, 2, scale=1), label="alpha=2,theta=1")
plt.plot(dom, gamma.pdf(dom, 5, scale=1), label="alpha=5,theta=1")
plt.plot(dom, gamma.pdf(dom, 10, scale=1), label="alpha=10,theta=1")
plt.title("Gamma distribution")
plt.legend()
plt.show()
plt.plot(dom, gamma.cdf(dom, 2, scale=1), label="alpha=2,theta=1")
plt.plot(dom, gamma.cdf(dom, 5, scale=1), label="alpha=5,theta=1")
plt.plot(dom, gamma.cdf(dom, 10, scale=1), label="alpha=10,theta=1")
plt.title("Gamma distribution")
plt.legend()
plt.show()

# exponential distribution
dom = np.arange(0, 10, 0.001)
plt.plot(dom, expon.pdf(dom, loc=1, scale=0.5), label="scale=0.5")
plt.plot(dom, expon.pdf(dom, loc=1, scale=1), label="scale=1")
plt.plot(dom, expon.pdf(dom, loc=1, scale=2), label="scale=2")
plt.title("Exponential distribution")
plt.legend()
plt.show()
plt.plot(dom, expon.cdf(dom, loc=1, scale=0.5), label="scale=0.5")
plt.plot(dom, expon.cdf(dom, loc=1, scale=1), label="scale=1")
plt.plot(dom, expon.cdf(dom, loc=1, scale=2), label="scale=2")
plt.title("Exponential distribution")
plt.legend()
plt.show()

# chi squared distribution
dom = np.arange(0, 70, 0.01)
plt.plot(dom, chi2.pdf(dom, 8), label="k=5")
plt.plot(dom, chi2.pdf(dom, 20), label="k=20")
plt.plot(dom, chi2.pdf(dom, 40), label="k=40")
plt.title("Chi squared distribution")
plt.legend()
plt.show()
plt.plot(dom, chi2.cdf(dom, 8), label="k=5")
plt.plot(dom, chi2.cdf(dom, 20), label="k=20")
plt.plot(dom, chi2.cdf(dom, 40), label="k=40")
plt.title("Chi squared distribution")
plt.legend()
plt.show()

# lognormal distribution
dom = np.arange(0, 5, 0.001)
plt.plot(dom, lognorm.pdf(dom, 0.25), label="sigma=0.25")
plt.plot(dom, lognorm.pdf(dom, 0.5), label="sigma=0.5")
plt.plot(dom, lognorm.pdf(dom, 1), label="sigma=1")
plt.title("Lognormal distribution")
plt.legend()
plt.show()
plt.plot(dom, lognorm.cdf(dom, 0.25), label="sigma=0.25")
plt.plot(dom, lognorm.cdf(dom, 0.5), label="sigma=0.5")
plt.plot(dom, lognorm.cdf(dom, 1), label="sigma=1")
plt.title("Lognormal distribution")
plt.legend()
plt.show()

# beta distribution
dom = np.arange(0.01, 0.99, 0.001)
plt.plot(dom, beta.pdf(dom, 0.5, 0.5), label="alpha=0.5,beta=0.5")
plt.plot(dom[15:], beta.pdf(dom[15:], 0.5, 2), label="alpha=0.5,beta=2")
plt.plot(dom, beta.pdf(dom, 2, 2), label="alpha=2,beta=2")
plt.plot(dom[:-15], beta.pdf(dom[:-15], 2, 0.5), label="alpha=2,beta=0.5")
plt.title("Beta distribution")
plt.legend()
plt.show()
plt.plot(dom, beta.cdf(dom, 0.5, 0.5), label="alpha=0.5,beta=0.5")
plt.plot(dom[15:], beta.cdf(dom[15:], 0.5, 2), label="alpha=0.5,beta=2")
plt.plot(dom, beta.cdf(dom, 2, 2), label="alpha=2,beta=2")
plt.plot(dom[:-15], beta.cdf(dom[:-15], 2, 0.5), label="alpha=2,beta=0.5")
plt.title("Beta distribution")
plt.legend()
plt.show()


