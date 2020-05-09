from mixture import Mixture
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np
from mine import *


dist = 1.2
with open('precalc_mi.pkl', 'rb') as f:
    precalculated = pickle.load(f)
means = np.array([[(1, 1), (-1, -1)], [(1, -1), (-1, 1)]]) * 1.2
cov = np.eye(2)
covs = np.array([[cov, cov], [cov, cov]])
proportions = np.array([0.5, 0.5])
mix = Mixture(proportions, means, covs)
half_ds = 10000
x_0 = mix.sample_i(i=0, num_samples=half_ds)
x_1 = mix.sample_i(i=1, num_samples=half_ds)
y_0 = np.zeros((half_ds, 1))
y_1 = np.ones((half_ds, 1))
x_dataset = np.concatenate((x_0, x_1), axis=0)
y_dataset = np.concatenate((y_0, y_1), axis=0)
# print(f'x shape: {x_dataset.shape}, y shape: {y_dataset.shape}')
# print(f'first class:  {x_dataset[0]}, {y_dataset[0]}')
# print(f'second class: {x_dataset[half_ds]}, {y_dataset[half_ds]}')
if dist not in precalculated.keys():
    print('Calculating true MI...')
    precalculated[dist] = mix.get_MI()[0]
    with open('precalc_mi.pkl', 'wb') as f:
        pickle.dump(precalculated, f)

true_mi = precalculated[dist]
print('True MI: ', true_mi)

mine_net = get_mine_model(input_shape=(3,), layer_sizes=[10, 10])
# print(mine_net.summary())

optimizer = keras.optimizers.Adam(learning_rate=0.005)
batch_size = 2000
per_epoch = (2 * half_ds) // batch_size

MI_bounds = []
epochs = 40
for e in tqdm(range(epochs), total=epochs):
    for i in range(per_epoch):
        ind_joint = np.random.randint(low=0, high=2 * half_ds, size=batch_size)
        ind_marginal = np.random.randint(low=0, high=2 * half_ds, size=batch_size)

        x_data_for_joint = x_dataset[ind_joint]
        y_data_for_joint = y_dataset[ind_joint]
        x_data_for_marginal = x_dataset[ind_marginal]
        y_data_for_marginal = y_dataset[ind_marginal]

        data = get_data(x_data_for_joint, y_data_for_joint, x_data_for_marginal, y_data_for_marginal)
        bound = training_step(mine_net, data, optimizer, ema_alpha=0.1)
        MI_bounds.append(bound)

estimated_mi = sum(MI_bounds[-100:]) / 100
print(f'Estimated MI: {estimated_mi}')
x_plot = np.arange(epochs * per_epoch)
true = np.ones(epochs * per_epoch) * true_mi
plt.plot(x_plot, MI_bounds)
plt.plot(x_plot, true)
plt.xlabel('Training steps')
plt.ylabel('MI lower bound')
plt.show()
