import numpy as np
from scipy.stats import multivariate_normal
from scipy import integrate
import matplotlib.pyplot as plt


class Mixture:
    def __init__(self, proportions, means, covs):
        self.n = len(proportions)
        self.num = [len(means_i) for means_i in means]
        self.proportions = np.array(proportions)
        if self.proportions.sum() != 1:
            self.proportions /= self.proportions.sum()
        self.means = np.array(means)
        self.covs = np.array(covs)

    def mu_i(self, i, coordinates):
        means_i = self.means[i]
        covs_i = self.covs[i]
        return sum(multivariate_normal.pdf(coordinates, mean, cov)
                   for mean, cov in zip(means_i, covs_i)) / self.num[i]

    def mu_i_func(self, i):
        return lambda coordinates: self.mu_i(i, coordinates)

    def mu(self, coordinates):
        mu_i_s = [self.mu_i(i, coordinates) * self.proportions[i] for i in range(self.n)]
        # print(mu_i.shape)
        # print(mu_i)
        return sum(mu_i_s)

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

    def MI_integrand(self, x, y):
        coordinates = (x, y)
        mu_i = np.array([self.mu_i(i, coordinates) for i in range(self.n)])
        log_mu_i = np.array([np.log(v) if v > 0 else 0 for v in mu_i])
        mu = self.mu(coordinates)
        log_mu = np.log(mu) if mu > 0 else 0
        return np.sum(self.proportions * mu_i * log_mu_i) - mu * log_mu

    def get_MI(self, rng=np.inf):
        return integrate.dblquad(self.MI_integrand, -rng, rng, -rng, rng)

    def plot_mu_slow(self, rang=10):
        delta = 0.025 * rang / 2
        x = np.arange(-rang, rang, delta)
        y = np.arange(-rang, rang, delta)
        X, Y = np.meshgrid(x, y)
        # X = X.ravel()
        # Y = Y.ravel()
        Z = np.array([self.mu(coord) for coord in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z)
        ax.set_title('Mixture pdf')
        plt.show()

    def plot_mu(self, rang=10):
        delta = 0.025 * rang / 2
        x = np.arange(-rang, rang, delta)
        y = np.arange(-rang, rang, delta)
        X, Y = np.meshgrid(x, y)
        Z = self.mu(np.stack([X, Y], axis=-1))
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z)
        ax.set_title('Mixture pdf')
        plt.show()
