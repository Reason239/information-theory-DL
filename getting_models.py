from mine import get_mine_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import softmax
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.applications import MobileNet, MobileNetV2, ResNet50V2, InceptionV3, DenseNet201, DenseNet169, \
    Xception, DenseNet121, ResNet50, NASNetMobile
from tensorflow.keras import applications
from typing import Tuple, Callable


def get_intermediate_model(net, layer_slice=slice(None, None)) -> Model:
    """Given a net return a model that outputs net's all intermediate layers"""
    layer_outputs = [layer.output for layer in net.layers[layer_slice]]
    return Model(inputs=net.input, outputs=layer_outputs)


def get_mine_for_input(net_layer_sizes, mine_layer_sizes, leaky_alpha=0.2):
    """Returns list of MINE models for measuring I(X, T) for T in intermediate layers"""
    return [get_mine_model((net_layer_size * 2,), mine_layer_sizes, leaky_alpha)
            for net_layer_size in net_layer_sizes]


def get_mine_for_output(net_layer_sizes, output_size, mine_layer_sizes, leaky_alpha=0.2):
    """Returns list of MINE models for measuring I(T, Y) for T in intermediate layers"""
    return [get_mine_model((net_layer_size + output_size,), mine_layer_sizes, leaky_alpha)
            for net_layer_size in net_layer_sizes]


def get_tishby_net(input_shape=(12,), layer_sizes=(10, 7, 5, 4, 3, 1), activation='tanh') -> Model:
    """Returns a net like in Tishby's paper"""
    net = Sequential()
    net.add(Dense(layer_sizes[0], input_shape=input_shape, activation=activation))
    for size in layer_sizes[1:-1]:
        net.add(Dense(size, activation=activation))
    net.add(Dense(layer_sizes[-1], activation='sigmoid'))
    return net


def get_pretrained_net(name) -> Tuple[Model, Callable]:
    name_to_module = {'MobileNet': 'mobilenet', 'DenseNet121': 'densenet',
                      'ResNet50': 'resnet', 'ResNet50V2': 'resnet_v2',
                      'ResNet101': 'resnet', 'ResNet101V2': 'resnet_v2',
                      'InceptionV3': 'inception_v3', 'Xception': 'xception'}
    module_name = name_to_module[name]
    module = getattr(applications, module_name)
    net_getter = getattr(module, name)
    net = net_getter(input_shape=(224, 224, 3), include_top=False, pooling='avg')
    preprocessor = getattr(module, 'preprocess_input')
    return net, preprocessor


if __name__ == '__main__':
    pass
