from mine import get_mine_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import softmax
from tensorflow.keras import activations, models
from tensorflow.keras.optimizers import Optimizer


def get_intermediate_model(net, layer_slice=slice(None, None)):
    """Given a net return a model that outputs net's all intermediate layers"""
    layer_outputs = [layer.output for layer in net.layers[layer_slice]]
    return models.Model(inputs=net.input, outputs=layer_outputs)


def get_mine_for_input(net_layer_sizes, mine_layer_sizes, leaky_alpha=0.2):
    """Returns list of MINE models for measuring I(X, T) for T in intermediate layers"""
    return [get_mine_model((net_layer_size * 2,), mine_layer_sizes, leaky_alpha)
            for net_layer_size in net_layer_sizes]


def get_mine_for_output(net_layer_sizes, output_size, mine_layer_sizes, leaky_alpha=0.2):
    """Returns list of MINE models for measuring I(T, Y) for T in intermediate layers"""
    return [get_mine_model((net_layer_size + output_size,), mine_layer_sizes, leaky_alpha)
            for net_layer_size in net_layer_sizes]


def get_tishby_net(input_shape=(12,), layer_sizes=(10, 7, 5, 4, 3, 1), activation='tanh'):
    """Returns a net like in Tishby's paper"""
    net = Sequential()
    net.add(Dense(layer_sizes[0], input_shape=input_shape, activation=activation))
    for size in layer_sizes[1:-1]:
        net.add(Dense(size, activation=activation))
    net.add(Dense(layer_sizes[-1], activation='sigmoid'))
    return net


