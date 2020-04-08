import numpy as np

a = np.array([[1, 2], [3, 4]])
i = np.array([0, 1])
a[np.arange(2), i] = 100
print(a)