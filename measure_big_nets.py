from collections import defaultdict
from itertools import product
from math import log

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import pickle as pkl

from mine import get_mine_model, mine_training_step, get_lower_bound, shuffle_and_process


def get_data(name):
    if name == 'embeddings':
        return embeddings
    elif name == 'embeddings + noise':
        return embeddings + np.random.normal(size=embeddings.shape, scale=noise_var)
    elif name == 'predictions':
        return predictions
    elif name == 'labels':
        return labels
    else:
        raise ValueError

# get labels
labels = np.load('saved/labels_test.npy')
mi = defaultdict(dict)
stds = defaultdict(dict)

# model_names = ['ResNet50']
model_names = ['ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'InceptionV3', 'Xception']
# stops = [1]
stops = [1, 30, 200]
for model_name, stop in product(model_names, stops):
    id = f'{model_name}_{stop}'
    print(f'Processing model {model_name}, trained for {stop} epoch(s)')

    # get embeddings and predictions
    embeddings = np.load(f'saved/embeddings_test/{model_name}.npy')
    embeddings /= np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))
    predictions = np.load(f'saved/predictions_test/{id}.npy')
    predictions /= np.max(np.abs(predictions))

    # hyperparameters, nets, optimizers
    ema_alpha = 0.05
    eval_num = 10
    noise_var = 0.125
    # x is input, e is embedding, p is prediction, l is label
    epochs_x_e = 750
    epochs_p_l = 400
    epochs_e_l = 500
    mine_net_x_e = get_mine_model(input_shape=(2048 + 2048,), layer_sizes=[2048, 2048])
    mine_net_p_l = get_mine_model(input_shape=(1 + 1,), layer_sizes=[400, 400])
    mine_net_e_l = get_mine_model(input_shape=(2048 + 1,), layer_sizes=[2048, 2048])
    mine_lr_x_e = 0.0002
    mine_lr_p_l = 0.0005
    mine_lr_e_l = 0.0005
    optimizer_x_e = Adam(mine_lr_x_e)
    optimizer_p_l = Adam(mine_lr_p_l)
    optimizer_e_l = Adam(mine_lr_e_l)

    # arrange MINE nets and optimizers
    pairs = [('embeddings', 'embeddings + noise'), ('predictions', 'labels'), ('embeddings', 'labels')]
    mine_nets = [mine_net_x_e, mine_net_p_l, mine_net_e_l]
    optimizers = [optimizer_x_e, optimizer_p_l, optimizer_e_l]
    epoch_nums = [epochs_x_e, epochs_p_l, epochs_e_l]

    # train MINEs, get MI estimations, plot results
    for (x, z), epochs, mine_net, optimizer in list(zip(pairs, epoch_nums, mine_nets, optimizers)):
        MI_bounds = []
        denominator = None
        subtract = (z == 'embeddings + noise')
        for e in tqdm(range(epochs), position=0, leave=True):
            data = shuffle_and_process(get_data(x), get_data(z), subtract)
            lb, denominator = mine_training_step(mine_net, data, optimizer, ema_alpha, denominator)
            MI_bounds.append(float(lb))
        estimations = []
        for e in range(eval_num):
            data = shuffle_and_process(get_data(x), get_data(z), subtract)
            estimations.append(float(get_lower_bound(mine_net, data)))
        estimations = np.array(estimations)
        mean = np.mean(estimations)
        std = np.std(estimations)
        print(f'Estimated MI({x}, {z}) for {id}: {mean:.3f} +- {std:.3f}')
        mi[f'I({x}, {z})'][id] = mean
        stds[f'I({x}, {z})'][id] = std

        x_plot = np.arange(len(MI_bounds))
        plt.plot(x_plot, MI_bounds)
        plt.xlabel('Training steps')
        plt.ylabel('MI lower bound (nats)')
        if z != 'embeddings + noise':
            upper = np.ones(len(MI_bounds)) * log(2)
            plt.plot(x_plot, upper)
            plt.ylim((-0.2), (1.2))
        plt.title(f'Estimating MI({x}, {z}) for {id}')
        plt.savefig(f'plots/measure_big_nets/{x}_{z}_{id}.png')
        plt.clf()

# save measured MI
with open('saved/mutual_informations.pkl', 'wb') as f:
    pkl.dump(mi, f)
with open('saved/mutual_information_stds.pkl', 'wb') as f:
    pkl.dump(stds, f)

