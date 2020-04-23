import numpy as np
from scipy.stats import multivariate_normal

class Mixture:
    def __init__(self, proportions, means, covs):
        self.n = len(proportions)
        self.num = [len(means_i) for means_i in means]
        self.proportions = np.array(proportions)
        if self.proportions.sum() != 1:
            self.proportions /= self.proportions.sum()
        self.means = np.array(means)
        self.covs = np.array(covs)

    def get_mu_i(self, i):
        def mu_i(coordinates):
            means_i = self.means[i]
            covs_i = self.covs[i]
            return sum(multivariate_normal(coordinates, mean, cov)
                       for mean, cov in zip(means_i, covs_i)) / len(self.num[i])
        return mu_i

    def sample_i(self, i, num_samples=1):
        pvals = [1 / self.num[i]] * self.num[i]
        num_for_ind = np.random.multinomial(num_samples, pvals)
        samples = []
        for num, mean, cov in zip(num_for_ind, self.means[i], self.covs[i]):
            samples.append(np.random.multivariate_normal(mean, cov, num))
        all_samples = np.concatenate(samples)
        np.random.shuffle(all_samples)
        return all_samples

    def sample(self, num_samples=1):
        num_for_i = np.random.multinomial(num_samples, self.proportions)
        samples = [self.sample_i(i, num_i) for i, num_i in enumerate(num_for_i)]
        all_samples = np.concatenate(samples)
        np.random.shuffle(all_samples)
        return all_samples
