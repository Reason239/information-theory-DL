import numpy as np
import matplotlib.pyplot as plt
from mixture import Mixture


if __name__ == '__main__':
    means = np.array([[(1, 1), (-1, -1)], [(1, -1), (-1, 1)]]) * 2
    cov = np.eye(2)
    covs = np.array([[cov, cov], [cov, cov]])
    proportions = np.array([0.5, 0.5])
    mix = Mixture(proportions, means, covs)
    num_i = np.random.multinomial(1000, proportions)
    samples = [mix.sample_i(i, num) for i, num in enumerate(num_i)]
    samples_0_x, samples_0_y = samples[0].T
    samples_1_x, samples_1_y = samples[1].T
    plt.scatter(samples_0_x, samples_0_y, c='blue')
    plt.scatter(samples_1_x, samples_1_y, c='red')
    plt.show()
