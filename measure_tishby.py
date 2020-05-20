from math import log
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from datetime import datetime

from mine import get_lower_bound, get_data_for_mine, mine_inp_train, mine_out_train
from getting_models import get_tishby_net, get_intermediate_model, get_mine_for_input, get_mine_for_output
from tishby_plots.plots_refactored import plot_all_epochs
from utils import add_noise, get_info_ind_log

# hyperparameters
layer_sizes = [10, 7, 5, 4, 3, 1]
tishby_activation = 'tanh'
mine_layer_sizes = [200, 200]
mine_lr = 0.0002
noise_var = 0.2
epochs = 200
batch_size = 128
mine_epochs = 20  # 500
mine_batch_size = 1024  # uses two batches per step
ema_alpha = 0.1
mine_eval_batch_size = 4096
num_inf = 20  # 10
epochs_for_inf = get_info_ind_log(epochs, num_inf)
# epochs_for_inf = list(range(epochs))
# epochs_for_inf = [198]
retrain_mines = True

# prepare data
x_data = np.load('tishby_data/data.npy')
y_data = np.load('tishby_data/labels.npy')[:, 0:1]  # take one-dimensional label, not one-hot
# shuffle
order = np.random.permutation(len(x_data))
x_data = x_data[order]
y_data = y_data[order]
ds = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size, drop_remainder=True)
mine_ds = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(2 * mine_batch_size, drop_remainder=True)

# build nets and an optimizer
net = get_tishby_net(input_shape=(12,), layer_sizes=layer_sizes, activation=tishby_activation)
net.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
all_layers = get_intermediate_model(net)
mine_nets_inp = get_mine_for_input(layer_sizes, mine_layer_sizes)
mine_nets_out = get_mine_for_output(layer_sizes, 1, mine_layer_sizes)
mine_optim = Adam(learning_rate=mine_lr)

# train and measure MI
I_XT = []
I_TY = []
for epoch_num in range(epochs):
    # train net
    net.fit(ds)

    # train MINEs
    if epoch_num in epochs_for_inf:
        if retrain_mines:
            # reset the networks and the optimizer
            mine_nets_inp = get_mine_for_input(layer_sizes, mine_layer_sizes)
            mine_nets_out = get_mine_for_output(layer_sizes, 1, mine_layer_sizes)
            mine_optim = Adam(learning_rate=mine_lr)
        layer_outputs = all_layers.predict(x_data, batch_size=1024)
        for mine_net, t_data in tqdm(zip(mine_nets_inp, layer_outputs), total=len(layer_outputs), desc='AA-MINEs'):
            mine_inp_train(mine_net, mine_optim, t_data, noise_var, mine_epochs, mine_batch_size, ema_alpha)
        for mine_net, t_data in tqdm(zip(mine_nets_out, layer_outputs), total=len(layer_outputs), desc='MINEs   '):
            mine_out_train(mine_net, mine_optim, t_data, y_data, mine_epochs, mine_batch_size, ema_alpha)

        # get MI estimates
        order = np.random.permutation(len(x_data))
        x1 = x_data
        y1 = y_data
        x2 = x_data[order]
        y2 = y_data[order]
        layer_outputs1 = all_layers(x1)
        layer_outputs2 = all_layers(x2)
        I_XT.append([float(
            get_lower_bound(mine_net, get_data_for_mine(t1, add_noise(t1, noise_var), add_noise(t2, noise_var), True)))
            for mine_net, t1, t2 in zip(mine_nets_inp, layer_outputs1, layer_outputs2)])
        I_TY.append([float(get_lower_bound(mine_net, get_data_for_mine(t1, y1, y2)))
                     for mine_net, t1 in zip(mine_nets_out, layer_outputs1)])
        print(f'{datetime.now()}  Epoch {epoch_num:3d}')
        print(I_XT[-1])
        print(I_TY[-1])

I_XT = np.array(I_XT)
I_TY = np.array(I_TY)

name = f'corrected1_{epochs}_e_{num_inf}_n_i_{mine_epochs}_m_e'
np.save(f'saved/{name}_I_XT.npy', I_XT)
np.save(f'saved/{name}_I_TY.npy', I_TY)
plot_all_epochs(I_XT, I_TY, np.array(epochs_for_inf), name,
                f'plots/measure_tishby/{name}',
                x_lim=(-0.2, 12 * log(2)), y_lim=(-0.2, log(2)))
