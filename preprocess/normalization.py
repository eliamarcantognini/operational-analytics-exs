# example of normalization (python)
from sklearn.preprocessing import MinMaxScaler
from numpy import array
# define dataset
data = [x for x in range(1, 10)]
data = array(data).reshape(len(data), 1)
print(data)
# fit transform
transformer = MinMaxScaler()
transformer.fit(data)
# difference transform
transformed = transformer.transform(data)
print(transformed)
# invert difference
inverted = transformer.inverse_transform(transformed)
print(inverted)