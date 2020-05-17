import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
import numpy as np

from utils import add_noise

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

EPS = tf.cast(1e-8, dtype=tf.float32)


def get_data_for_mine(x1, z1, z2, subtract=False):
    """Given two batches of (X, Z) data produce joint and marginal inputs
    for MINE for measuring I(X, Z)

    Arguments:
          x1, z1, z2: samples from joint ((x1, z1)) and marginal(z2) distributions
          subtract: if True then joint = (x1, z1 - x1), marginal = (x1, z2 - x1)
          else joint = (x1, z1), marginal = (x1, z2)"""
    x1 = tf.convert_to_tensor(x1, dtype=tf.float32)
    z1 = tf.convert_to_tensor(z1, dtype=tf.float32)
    z2 = tf.convert_to_tensor(z2, dtype=tf.float32)
    if subtract:
        z1 -= x1
        z2 -= x1
    joint = tf.concat([x1, z1], axis=1)
    marginal = tf.concat([x1, z2], axis=1)
    return joint, marginal


def get_mine_model(input_shape, layer_sizes, leaky_alpha=0.2):
    """Build a MINE model"""
    mine_net = Sequential()
    mine_net.add(Dense(layer_sizes[0], input_shape=input_shape))
    mine_net.add(LeakyReLU(alpha=leaky_alpha))
    for size in layer_sizes[1:]:
        mine_net.add(Dense(size))
        mine_net.add(LeakyReLU(alpha=leaky_alpha))
    mine_net.add(Dense(1))
    return mine_net


def mine_training_step(mine_net, data, optimizer, ema_alpha=0.1, prev_denom=None, cap=None):
    """Training step for MINE

    Arguments:
        mine_net: MINE Model
        data: (input_joint, input_marginal), batch dimension is 0
        ema_alpha: Exponential Moving Average alpha for bias corrected gradients
        prev_denom: previous denominator for bias corrected gradients
        cap: (min, max) values for tensor clipping
    Returns:
        Lower bound for the Mutual Information calculated on this batch, denominator (for EMA)
          """
    inp_joint, inp_marginal = data
    net_weights = mine_net.trainable_weights

    for weight in net_weights:
        tf.debugging.assert_all_finite(weight, 'net weight', name=None)

    if ema_alpha is None:
        # without corrected gradients
        # forward pass
        with tf.GradientTape() as tape:
            out_joint = mine_net(inp_joint)
            out_marginal = mine_net(inp_marginal)

            # evaluate lower bond on MI
            batch_size_float32 = tf.cast(out_joint.shape[0], dtype=tf.float32)
            lower_bound = tf.reduce_mean(out_joint) - \
                          (tf.reduce_logsumexp(out_marginal) - tf.math.log(batch_size_float32))

        # calculate gradients
        # append minus because we want to maximize
        gradients = [-grad for grad in tape.gradient(lower_bound, net_weights)]
        denominator = None
    else:
        # with corrected gradients
        # forward pass
        with tf.GradientTape(persistent=True) as tape:
            out_joint = mine_net(inp_joint)
            mean_out_joint = tf.reduce_mean(out_joint)
            out_marginal = mine_net(inp_marginal)

            # to calculate mean(grad * exponents) we can take the gradient of mean(exponents)
            mean_exp_marginal = tf.reduce_mean(tf.exp(out_marginal))

            # check for problems
            tf.debugging.assert_all_finite(out_joint, 'out joint', name=None)
            tf.debugging.assert_all_finite(mean_out_joint, 'mean out joint', name=None)
            tf.debugging.assert_all_finite(out_marginal, 'out marginal', name=None)
            tf.debugging.assert_all_finite(mean_exp_marginal, 'mean exp', name=None)

        # evaluate lower bond on MI outside of the tape
        lower_bound = tf.reduce_mean(out_joint) - tf.math.log(mean_exp_marginal)

        # now calculate the corrected gradients
        # 1) for the minuend: mean of gradients = gradient of mean
        minuends = tape.gradient(mean_out_joint, net_weights)

        # 2) for the subtrahend fraction
        # numerator = gradient of mean(exponents)
        numerators = tape.gradient(mean_exp_marginal, net_weights)
        del tape

        # denominator = Exponential Moving Average
        if prev_denom is None:
            denominator = mean_exp_marginal
        else:
            denominator = ema_alpha * mean_exp_marginal + (1 - ema_alpha) * prev_denom

        # in order not to divide by zero
        denominator += EPS

        # now we can build our gradients
        # add minus because we want to maximize
        gradients = [-minuend + numerator / denominator for minuend, numerator in zip(minuends, numerators)]

    # clip
    if cap is not None:
        gradients = [tf.clip_by_value(grad, *cap) for grad in gradients]

    # apply gradients
    optimizer.apply_gradients(zip(gradients, net_weights))

    return lower_bound, denominator


def mine_inp_train(mine_net, optim, t_data, noise_var, epochs, batch_size, ema_alpha=0.1):
    """Train AA-MINE for measuring I(X, T + noise)

    Actually receives intermediate layer's output"""
    ds = tf.data.Dataset.from_tensor_slices(t_data).repeat(epochs).batch(2 * batch_size, drop_remainder=True)
    denominator = None
    for x in ds:
        x1, x2 = np.split(x, 2)
        z1 = add_noise(x1, noise_var)
        z2 = add_noise(x2, noise_var)
        # this is AA-MINE, so we feed layer output to it
        data = get_data_for_mine(x1, z1, z2, subtract=True)
        lb, denominator = mine_training_step(mine_net, data, optim, ema_alpha, denominator)


def mine_out_train(mine_net, optim, t_data, y_data, epochs, batch_size, ema_alpha=0.1):
    """Train ordinary MINE for measuring I(T, Y)"""

    ds = tf.data.Dataset.from_tensor_slices((t_data, y_data)).repeat(epochs).batch(2 * batch_size, drop_remainder=True)
    denominator = None
    for x, z in ds:
        x1, x2 = np.split(x, 2)
        z1, z2 = np.split(z, 2)
        data = get_data_for_mine(x1, z1, z2)
        lb, denominator = mine_training_step(mine_net, data, optim, ema_alpha, denominator)


def get_lower_bound(mine_net, data):
    """Grt a lower bound for MI given data
    data should be like 'get_data_for_mine' output"""
    inp_joint, inp_marginal = data

    out_joint = mine_net(inp_joint)
    out_marginal = mine_net(inp_marginal)

    batch_size_float32 = tf.cast(out_joint.shape[0], dtype=tf.float32)
    lower_bound = tf.reduce_mean(out_joint) - \
                  (tf.reduce_logsumexp(out_marginal) - tf.math.log(batch_size_float32))
    return lower_bound

