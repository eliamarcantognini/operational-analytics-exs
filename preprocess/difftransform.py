
# example of a difference transform (python)
# difference dataset
def difference(dataset, interval):
    return [dataset[i] - dataset[i - interval] for i in range(interval, len(dataset))]


# invert difference
def invert_difference(orig_data, diff_data, interval):
    return [diff_data[i - interval] + orig_data[i - interval] for i in range(interval, len(orig_data))]

    
# define dataset
data = [x for x in range(1, 10)]
print(data)
# difference transform
transformed = difference(data, 1)
print(transformed)
# invert difference
inverted = invert_difference(data, transformed, 1)
print(inverted)
