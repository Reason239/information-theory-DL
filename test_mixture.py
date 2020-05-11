from mixture import Mixture
import numpy as np
import time

e = np.e
means = np.array([[(1, 1), (-1, -1)], [(1, -1), (-1, 1)]])
cov = np.eye(2)
covs = np.array([[cov, cov], [cov, cov]])
proportions = np.array([0.5, 0.5])
mix = Mixture(proportions, means * 3, covs)
# t0 = time.time()
# print(f'Our MI: {mix.get_MI()}')
# print(f'If independent: {entropy(proportions, base=e)}')
# print(f'Time for calculation: {time.time() - t0:.5f}s.')
# print(mix.mu_i(0, (10, 10)))
# mi[sc] = mix.get_MI()
# arr = np.array([[[0, 0], [1, 1]], [[10, 10], [100, 100]]])
# arr = np.array([[0, 0], [1, 1], [10, 10]])
# print(mix.mu((0, 0)))
# for a in arr:
#     for coord in a:
#         print(mix.mu(coord))


t0 = time.time()
mix.plot_mu()

