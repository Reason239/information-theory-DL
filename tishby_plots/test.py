import numpy as np

a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
c = np.array([1, 2, 3])
l = [a, b, c]

print(np.dstack((ar for ar in l)))