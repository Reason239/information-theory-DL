import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from mine import *

x = np.load('tishby_data/data.npy')
y = np.load('tishby_data/labels.npy')[:, 0:1]


input_shape = (13,)
layer_sizes = [200] * 2
leaky_alpha = 0.2
epochs = 3000
dataset_len = 4096
batch_size = 4096
per_epoch = dataset_len // batch_size
num_eval = 10
eval_data_size = 1024

true_mi = 0.99 * np.log(2)
MI_bounds = []

net = get_mine_model(input_shape, layer_sizes, leaky_alpha)
optimizer = keras.optimizers.Adam(learning_rate=0.0003)

ind = np.arange(dataset_len)
for epoch in tqdm(range(epochs), total=epochs):
    np.random.shuffle(ind)
    for _ in range(per_epoch):
        x1 = x[ind[(_ + 0) * batch_size // 2:(_ + 1) * batch_size // 2]]
        x2 = x[ind[(_ + 1) * batch_size // 2:(_ + 2) * batch_size // 2]]
        y1 = y[ind[(_ + 0) * batch_size // 2:(_ + 1) * batch_size // 2]]
        y2 = y[ind[(_ + 1) * batch_size // 2:(_ + 2) * batch_size // 2]]
        data = get_data_for_mine(x1, y1, x2, y2)
        # assert not np.any(np.isnan(data[0])) and not np.any(np.isnan(data[1]))
        lb = training_step(net, data, optimizer, ema_alpha=0.1)
        MI_bounds.append(lb)

estimations = []
for _ in range(num_eval):
    first = np.arange(dataset_len)
    np.random.shuffle(first)
    second = np.arange(dataset_len)
    np.random.shuffle(second)
    x1 = x[first]
    x2 = x[second]
    y1 = y[first]
    y2 = y[second]
    data = get_data_for_mine(x1, y1, x2, y2)
    estimated_mi = get_lower_bound(net, data)
    estimations.append(estimated_mi)
estimations = np.array(estimations)
mean = np.mean(estimations)
std = np.std(estimations)
mine_mi = mean
print(f'true_mi={true_mi:5.3f} mine_mi={mean:5.3f} +- {std:5.3f} error={abs(true_mi - mean):5.3f}')

x_plot = np.arange(epochs * per_epoch)
true = np.ones(epochs * per_epoch) * true_mi
plt.plot(x_plot, MI_bounds)
plt.plot(x_plot, true)
plt.xlabel('Training steps')
plt.ylabel('MI lower bound')
plt.show()