# example of power transform and inversion (python)
from math import log
from math import exp
from scipy.stats import boxcox


# invert a boxcox transform for one value
def invert_boxcox(value, lam):
    # log case
    if lam == 0:
        return exp(value)
    # all other cases
    return exp(log(lam * value + 1) / lam)


# define dataset
data = [x for x in range(1, 10)]
print(data)
# power transform
transformed, lmbda = boxcox(data)
print(transformed, lmbda)
# invert transform
inverted = [invert_boxcox(x, lmbda) for x in transformed]
print(inverted)