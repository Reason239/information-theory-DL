from mine import get_data_for_mine, mine_training_step, get_mine_model, get_lower_bound
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

input_shape = (13,)
layer_sizes = [200] * 2
leaky_alpha = 0.2
epochs = 500
dataset_len = 4096
batch_size = 512
per_epoch = dataset_len // batch_size
num_eval = 10
eval_data_size = 1024

x_data = np.load('tishby_data/data.npy').astype(np.float32)
y_data = (np.load('tishby_data/labels.npy')[:, 0:1]).astype(np.float32)
order = np.random.permutation(len(x_data))
x_data = x_data[order]
y_data = y_data[order]
ds = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(buffer_size=2048).batch(2 * batch_size, drop_remainder=True)

true_mi = 0.99 * np.log(2)  # convert to nats
MI_bounds = []

net = get_mine_model(input_shape, layer_sizes, leaky_alpha)
optimizer = keras.optimizers.Adam(learning_rate=0.0003)

denominator = None
for epoch in tqdm(range(epochs), total=epochs):
    for x, y in ds:
        x1, x2 = x[:batch_size], x[batch_size:]
        y1, y2 = y[:batch_size], y[batch_size:]
        data = get_data_for_mine(x1, y1, y2)
        lb, denominator = mine_training_step(net, data, optimizer, ema_alpha=0.1, prev_denom=denominator)
        MI_bounds.append(lb)

estimations = []
for _ in range(num_eval):
    first = np.arange(dataset_len)
    np.random.shuffle(first)
    second = np.arange(dataset_len)
    np.random.shuffle(second)
    x1 = x_data[first]
    y1 = y_data[first]
    y2 = y_data[second]
    data = get_data_for_mine(x1, y1, y2)
    estimated_mi = get_lower_bound(net, data)
    estimations.append(estimated_mi)
estimations = np.array(estimations)
mean = np.mean(estimations)
std = np.std(estimations)
mine_mi = mean
print(f'true_mi={true_mi:5.3f} mine_mi={mean:5.3f} +- {std:5.3f} error={abs(true_mi - mean):5.3f}')

x_plot = np.arange(epochs * per_epoch // 2)
true = np.ones(epochs * per_epoch // 2) * true_mi
plt.plot(x_plot, MI_bounds)
plt.plot(x_plot, true)
plt.xlabel('Training steps')
plt.ylabel('MI lower bound')
plt.title(f'True MI: {true_mi:.3f}, estimated MI: {mine_mi:.3f}')
plt.savefig('plots/for presentation/measure_tishby_ds.png')
plt.show()
