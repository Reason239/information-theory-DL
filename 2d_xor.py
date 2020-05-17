import numpy as np
import matplotlib.pyplot as plt
from mixture import Mixture


if __name__ == '__main__':
    dist = 2
    means = np.array([[(1, 1), (-1, -1)], [(1, -1), (-1, 1)]]) * dist
    cov = np.eye(2)
    covs = np.array([[cov, cov], [cov, cov]])
    proportions = np.array([0.5, 0.5])
    mix = Mixture(proportions, means, covs)
    num_i = np.random.multinomial(2000, proportions)
    samples = [mix.sample_i(i, num) for i, num in enumerate(num_i)]
    samples_0_x, samples_0_y = samples[0].T
    samples_1_x, samples_1_y = samples[1].T
    plt.scatter(samples_0_x, samples_0_y, c='blue', s=6)
    plt.scatter(samples_1_x, samples_1_y, c='red', s=6)
    plt.scatter([-dist, dist], [dist, -dist], c='black', marker='D', s=50)
    plt.scatter([dist, -dist], [dist, -dist], c='black', marker='D', s=50)
    plt.title(f'XOR mixture with dist={dist}')
    plt.savefig(f'plots/for presentation/xor_mix_{dist}.png')
    plt.show()
