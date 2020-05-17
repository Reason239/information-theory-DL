import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm
import numpy as np

from mine import get_mine_model, mine_training_step, get_data_for_mine
from mixture import Mixture

dist = 2
with open('saved/precalc_mi.pkl', 'rb') as f:
    precalculated = pickle.load(f)
means = np.array([[(1, 1), (-1, -1)], [(1, -1), (-1, 1)]]) * dist
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
if dist not in precalculated.keys():
    print('Calculating true MI...')
    precalculated[dist] = mix.get_MI()[0]
    with open('saved/precalc_mi.pkl', 'wb') as f:
        pickle.dump(precalculated, f)

true_mi = precalculated[dist]
print('True MI: ', true_mi)

mine_net = get_mine_model(input_shape=(3,), layer_sizes=[100, 100], leaky_alpha=0.0)

optimizer = keras.optimizers.Adam(learning_rate=0.005)
batch_size = 2000
per_epoch = (2 * half_ds) // batch_size

MI_bounds = []
epochs = 20
denominator = None
for e in tqdm(range(epochs), total=epochs):
    for i in range(per_epoch):
        ind_joint = np.random.randint(low=0, high=2 * half_ds, size=batch_size)
        ind_marginal = np.random.randint(low=0, high=2 * half_ds, size=batch_size)

        x_data_for_joint = x_dataset[ind_joint]
        y_data_for_joint = y_dataset[ind_joint]
        y_data_for_marginal = y_dataset[ind_marginal]

        data = get_data_for_mine(x_data_for_joint, y_data_for_joint, y_data_for_marginal)
        bound, denominator = mine_training_step(mine_net, data, optimizer, ema_alpha=0.1, prev_denom=denominator)
        MI_bounds.append(bound)

estimated_mi = float(sum(MI_bounds[-100:]) / 100)
print(f'Estimated MI: {estimated_mi}')
x_plot = np.arange(epochs * per_epoch)
true = np.ones(epochs * per_epoch) * true_mi
plt.plot(x_plot, MI_bounds)
plt.plot(x_plot, true)
plt.title(f'Dist: {dist}, True MI: {true_mi:.3f}, estimated MI: {estimated_mi:.3f}')
plt.xlabel('Training steps')
plt.ylabel('MI lower bound (nats)')
plt.legend()
# plt.savefig(f'plots/for presentation/measure_xor_{dist}.png')
plt.show()
