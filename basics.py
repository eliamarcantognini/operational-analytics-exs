import numpy as np

lst = [1, 2, 3, 4, ]
arr = np.array(lst).reshape(2, 2)
d = np.linalg.det(arr)
print("Determinant of arr is: {}".format(d))
ei = np.linalg.eig(arr)
print("Eigenvalues of arr are: {}".format(ei[0]))
a2 = 2 * arr
print("2*arr is:\n{}".format(a2))
a3 = arr + ei[1]
print("arr + ei[1] is:\n{}".format(a3))

