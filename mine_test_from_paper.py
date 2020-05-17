from mine import get_data_for_mine, get_mine_model, mine_training_step, get_lower_bound

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm


def data_for_mine(dim, rho, size):
    mean = np.array([0, 0], dtype=np.float)
    cov = np.array([[1, rho], [rho, 1]], dtype=np.float)
    pairs1 = [np.random.multivariate_normal(mean, cov, size) for _ in range(dim)]
    pairs2 = [np.random.multivariate_normal(mean, cov, size) for _ in range(dim)]
    raw1 = np.stack(pairs1, axis=2)
    raw2 = np.stack(pairs2, axis=2)
    return get_data_for_mine(raw1[:, 0, :], raw1[:, 1, :], raw2[:, 1, :])



dim_each = 20
input_shape = (2 * dim_each,)
layer_sizes = [50 * dim_each] * 2
leaky_alpha = 0.2
rhos = [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
num_epochs = [19, 16, 13, 11, 10, 10, 9, 10, 10, 11, 13, 16, 19]
dataset_len = 100000
batch_size = 2500
per_epoch = dataset_len // batch_size
num_eval = 10
eval_data_size = 2000
cap = None

true_mi = -0.5 * dim_each * np.log(1 - np.square(np.array(rhos)))
mine_mi = []
lb_history = [[] for _ in rhos]
denominator = None


for rho, true, hist, epochs in list(zip(rhos, true_mi, lb_history, num_epochs))[:]:
    net = get_mine_model(input_shape, layer_sizes, leaky_alpha)
    optimizer = keras.optimizers.Adam(learning_rate=0.0002)
    for epoch in tqdm(range(epochs), total=epochs):
        for _ in range(per_epoch):
            data = data_for_mine(dim_each, rho, batch_size)
            lb, denominator = mine_training_step(net, data, optimizer, ema_alpha=0.1, prev_denom=denominator, cap=cap)
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

fig, ax = plt.subplots()
ax.plot(rhos, true_mi, label='True MI')
ax.plot(rhos, mine_mi, label='MINE estimates')
plt.xlabel('rho')
plt.ylabel('MI lower bound (nats)')
plt.title(f'Multivariate gaussians, joint dimension = {2 * dim_each}')
plt.legend()
fig.savefig(f'plots/for presentation/from_paper_dim_{2 * dim_each}+.png')
plt.show()

# look = 1
# x_plot = np.arange(num_epochs[look] * per_epoch)
# true = np.ones(num_epochs[look] * per_epoch) * true_mi[look]
# plt.plot(x_plot, lb_history[look])
# plt.plot(x_plot, true)
# plt.xlabel('Training steps')
# plt.ylabel('MI lower bound (nats)')
# plt.show()
