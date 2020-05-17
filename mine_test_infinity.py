import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from tqdm import tqdm

from mine import get_data_for_mine, get_mine_model, get_lower_bound, mine_training_step

dim = 100
mine_net = get_mine_model((dim * 2,), [2000, 2000])
optim = Adam(learning_rate=0.0001)
# data_size = 4000
batch_size = 2500

m = np.random.normal(size=(dim, dim))


def raw_data(size):
    x = np.random.normal(size=(size, dim))
    # z = np.matmul(x, m)
    z = x.copy()
    return x, z


denominator = None
MI_bounds = []


def train(iterations):
    for _ in tqdm(range(iterations)):
        x1, z1 = raw_data(batch_size)
        x2, z2 = raw_data(batch_size)
        data = get_data_for_mine(x1, z1, z2, subtract=True)
        lb, denominator = mine_training_step(mine_net, data, optim)
        MI_bounds.append(float(lb))


def eval_mi(size):
    x1, z1 = raw_data(size)
    x2, z2 = raw_data(size)
    data = get_data_for_mine(x1, z1, z2, subtract=True)
    print(get_lower_bound(mine_net, data))


def plot():
    plt.plot(MI_bounds, label='Lower bound')
    plt.xlabel('Training steps')
    plt.ylabel('MI lower bound (nats)')
    plt.title(f'"AA"-MINE estimating infinite I(X, X), dim={dim}')
    plt.savefig(f'plots/for presentation/infinity_id_dim_{dim}.png')
    plt.show()

# burn in
train(150)
MI_bounds = []
train(500)
eval_mi(10000)
plot()
