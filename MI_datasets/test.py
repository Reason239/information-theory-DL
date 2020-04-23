import numpy as np

cov = np.eye(2)
print(cov.T)
covs = np.tile(cov, (1, 1, 2))
print(covs.shape)
print(covs)