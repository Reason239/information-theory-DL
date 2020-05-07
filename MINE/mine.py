import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU
import numpy as np


def get_data(x1, z1, x2, z2):
    x1 = tf.convert_to_tensor(x1, dtype=tf.float32)
    z1 = tf.convert_to_tensor(z1, dtype=tf.float32)
    x2 = tf.convert_to_tensor(x2, dtype=tf.float32)
    z2 = tf.convert_to_tensor(z2, dtype=tf.float32)
    joint = tf.concat([x1, z1], axis=1)
    z2_shuffled = tf.random.shuffle(z2)
    marginal = tf.concat([x2, z2_shuffled], axis=1)
    return joint, marginal


def get_mine_model(input_shape, layer_sizes, leaky_alpha=0.2):
    mine_net = Sequential()
    mine_net.add(Dense(layer_sizes[0], input_shape=input_shape))
    mine_net.add(LeakyReLU(alpha=leaky_alpha))
    for size in layer_sizes[1:]:
        mine_net.add(Dense(size))
        mine_net.add(LeakyReLU(alpha=leaky_alpha))
    mine_net.add(Dense(1))
    return mine_net


def training_step(mine_net, data, optimizer=None, learning_rate=0.001, ema_alpha=None, prev_denom=None):
    inp_joint, inp_marginal = data

    # forward pass
    with tf.GradientTape(persistent=True) as tape:
        out_joint = mine_net(inp_joint)
        out_marginal = mine_net(inp_marginal)
        if ema_alpha is None:
            # evaluate lower bond on MI
            batch_size_float32 = tf.cast(out_joint.shape[0], dtype=tf.float32)
            lower_bound = tf.reduce_mean(out_joint) - (
                        tf.reduce_logsumexp(out_marginal) - tf.math.log(batch_size_float32))

    if ema_alpha is not None:
        # evaluate lower bond on MI
        batch_size_float32 = tf.cast(out_joint.shape[0], dtype=tf.float32)
        lower_bound = tf.reduce_mean(out_joint) - (tf.reduce_logsumexp(out_marginal) - tf.math.log(batch_size_float32))

    # evaluate bias corrected gradients
    if ema_alpha is None:
        fu = tape.jacobian(out_joint, mine_net.trainable_weights, experimental_use_pfor=False)
        fu = tape.jacobian(out_marginal, mine_net.trainable_weights, experimental_use_pfor=False)
        corrected_grads = [-grad for grad in tape.gradient(lower_bound, mine_net.trainable_weights)]
        del tape
        optimizer.apply_gradients(zip(corrected_grads, mine_net.trainable_weights))
        return lower_bound

    # 1) for the minuend
    joint_grads = tape.jacobian(out_joint,
                                mine_net.trainable_weights,
                                experimental_use_pfor=False)  # list of shape (batch, 1, x, y) and (batch, 1, x) tensors
    minuends = [tf.reduce_mean(tf.squeeze(grad, axis=1), axis=0)
                for grad in joint_grads]

    # 2) for the subtrahend fraction
    corrected_subtrahends = []
    marginal_grads = tape.jacobian(out_marginal,
                                   mine_net.trainable_weights,
                                   experimental_use_pfor=False)  # list of shape (batch, 1, x, y) and (batch, 1, x) tensors
    del tape
    # calculate exponents
    exp = tf.exp(out_marginal)  # (batch, 1) tensor
    exp1 = tf.expand_dims(exp, 1)  # (batch, 1, 1) tensor
    # calculate fractions
    for grad in marginal_grads:
        grad = tf.squeeze(grad, axis=1)  # now it's (batch, x, y) or (batch, x) tensor
        if len(grad.shape) == 2:
            numerator = tf.reduce_mean(tf.multiply(grad, exp), axis=0)
        elif len(grad.shape) == 3:
            numerator = tf.reduce_mean(tf.multiply(grad, exp1), axis=0)
        else:
            raise AssertionError
        if prev_denom is None or ema_alpha is None:
            denominator = tf.reduce_mean(exp, axis=0)
        else:
            denominator = ema_alpha * tf.reduce_mean(exp, axis=0) + (1 - ema_alpha) * prev_denom
        corrected_subtrahends.append(tf.divide(numerator, denominator))

    # now we can build our gradients
    # append minus, because we want to maximize
    corrected_grads = [-(minuend - subtrahend) for minuend, subtrahend in zip(minuends, corrected_subtrahends)]

    # apply
    if optimizer is not None:
        optimizer.apply_gradients(zip(corrected_grads, mine_net.trainable_weights))
    else:
        for grad, weights in zip(corrected_grads, mine_net.trainable_weights):
            weights.assign_sub(learning_rate * grad)

    return lower_bound


if __name__ == '__main__':
    from mixture import Mixture
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    means = np.array([[(1, 1), (-1, -1)], [(1, -1), (-1, 1)]]) * 2
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
    print(f'x shape: {x_dataset.shape}, y shape: {y_dataset.shape}')
    print(f'first class:  {x_dataset[0]}, {y_dataset[0]}')
    print(f'second class: {x_dataset[half_ds]}, {y_dataset[half_ds]}')
    # true_mi = mix.get_MI()[0]
    true_mi = 0.5789045680121332
    print('True MI: ', true_mi)

    mine_net = get_mine_model(input_shape=(3,), layer_sizes=[10, 10])
    # print(mine_net.summary())

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    batch_size = 2000
    per_epoch = (2 * half_ds) // batch_size

    MI_bounds = []
    epochs = 20
    for e in range(epochs):
        for i in tqdm(range(per_epoch), total=per_epoch):
            ind_joint = np.random.randint(low=0, high=2 * half_ds, size=batch_size)
            ind_marginal = np.random.randint(low=0, high=2 * half_ds, size=batch_size)

            x_data_for_joint = x_dataset[ind_joint]
            y_data_for_joint = y_dataset[ind_joint]
            x_data_for_marginal = x_dataset[ind_marginal]
            y_data_for_marginal = y_dataset[ind_marginal]

            data = get_data(x_data_for_joint, y_data_for_joint, x_data_for_marginal, y_data_for_marginal)
            bound = training_step(mine_net, data, optimizer)
            MI_bounds.append(bound)

    x_plot = np.arange(epochs * per_epoch)
    true = np.ones(epochs * per_epoch) * true_mi
    plt.plot(x_plot, MI_bounds)
    plt.plot(x_plot, true)
    plt.show()
