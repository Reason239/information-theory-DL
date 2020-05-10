import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from pprint import pprint
import numpy as np
from mine import *


def data_for_mine(dim=3, rho=0.5, size=100):
    mean = np.array([0, 0], dtype=np.float)
    cov = np.array([[1, rho], [rho, 1]], dtype=np.float)
    pairs1 = [np.random.multivariate_normal(mean, cov, size) for _ in range(dim)]
    pairs2 = [np.random.multivariate_normal(mean, cov, size) for _ in range(dim)]
    raw1 = np.stack(pairs1, axis=2)
    raw2 = np.stack(pairs2, axis=2)
    return get_data_for_mine(raw1[:, 0, :], raw1[:, 1, :], raw2[:, 0, :], raw2[:, 1, :])


dim_each = 6
input_shape = (2 * dim_each,)
layer_sizes = [50 * dim_each] * 2
leaky_alpha = 0.2
rhos = [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
epochs = 10
dataset_len = 100000
batch_size = 1000
per_epoch = dataset_len // batch_size
num_eval = 10
eval_data_size = 1000

true_mi = -0.5 * dim_each * np.log(1 - np.square(np.array(rhos)))
mine_mi = []
lb_history = [[] for _ in rhos]

for rho, true, hist in list(zip(rhos, true_mi, lb_history))[2:3]:
    net = get_mine_model(input_shape, layer_sizes, leaky_alpha)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    for epoch in tqdm(range(epochs), total=epochs):
        for _ in range(per_epoch):
            data = data_for_mine(dim_each, rho, batch_size)
            # assert not np.any(np.isnan(data[0])) and not np.any(np.isnan(data[1]))
            lb = training_step(net, data, optimizer, ema_alpha=0.1)
            hist.append(lb)
    estimations = []
    for _ in range(num_eval):
        data = data_for_mine(dim_each, rho, eval_data_size)
        estimated_mi = get_lower_bound(net, data)
        estimations.append(estimated_mi)
    estimations = np.array(estimations)
    mean = np.mean(estimations)
    std = np.std(estimations)
    mine_mi.append(mean)
    print(
        f'dim={dim_each:2d} rho={rho:5.2f} true_mi={true:5.3f} mine_mi={mean:5.3f} +- {std:5.3f} error={abs(true - mean):5.3f}')
#
# fig, ax = plt.subplots()
# ax.plot(rhos, true_mi, label='True MI')
# ax.plot(rhos, mine_mi, label='MINE estimates')
# plt.xlabel('rho')
# plt.ylabel('MI')
# plt.title(f'Multivariate gaussians, joint dimension = {2 * dim_each}')
# plt.legend()
# fig.savefig(f'plots/gaussians_dim_{2 * dim_each}_epochs_{epochs}.png')
# plt.show()

x_plot = np.arange(epochs * per_epoch)
true = np.ones(epochs * per_epoch) * true_mi[2]
plt.plot(x_plot, lb_history[2])
plt.plot(x_plot, true)
plt.xlabel('Training steps')
plt.ylabel('MI lower bound')
plt.show()
