import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU
import numpy as np

EPS = tf.cast(1e-8, dtype=tf.float32)


def get_data_for_mine(x1, z1, x2, z2):
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


def training_step(mine_net, data, optimizer=None, learning_rate=0.001, ema_alpha=0.1, prev_denom=None):
    """Training step for MINE

    Arguments:
        mine_net: MINE Model
        data: (input_joint, input_marginal), batch dimension is 0
        optimizer: optimizer for Model's weights. If None then manual SGD is used.
        learning_rate: lr for manual SGD
        ema_alpha: Exponential Moving Average alpha for bias corrected gradients
        prev_denom: previous denominator for bias corrected gradients
    Returns:
        Lower bound for the Mutual Information calculated on this batch
          """
    inp_joint, inp_marginal = data
    net_weights = mine_net.trainable_weights

    # m = -1
    for weight in net_weights:
        tf.debugging.assert_all_finite(weight, 'net weight', name=None)
        # m = max(m, np.max(np.abs(weight.numpy())))
    # if m > 10:
    #     print(m)


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
        # batch_size_float32 = tf.cast(out_joint.shape[0], dtype=tf.float32)
        lower_bound = tf.reduce_mean(out_joint) - tf.math.log(mean_exp_marginal)

        # now calculate corrected gradients
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
        # denominator += EPS

        # now we can build our gradients
        # append minus because we want to maximize
        gradients = [-minuend + numerator / denominator for minuend, numerator in zip(minuends, numerators)]

    # apply gradients
    if optimizer is not None:
        optimizer.apply_gradients(zip(gradients, net_weights))
    else:
        for grad, weights in zip(gradients, net_weights):
            weights.assign_sub(learning_rate * grad)

    return lower_bound


def get_lower_bound(mine_net, data):
    inp_joint, inp_marginal = data
    out_joint = mine_net(inp_joint)

    out_marginal = mine_net(inp_marginal)

    batch_size_float32 = tf.cast(out_joint.shape[0], dtype=tf.float32)
    lower_bound = tf.reduce_mean(out_joint) - \
                  (tf.reduce_logsumexp(out_marginal) - tf.math.log(batch_size_float32))
    return lower_bound


if __name__ == '__main__':
    pass
