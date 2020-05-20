from mine import get_mine_model, mine_training_step, get_lower_bound, get_data_for_mine
from getting_models import get_pretrained_net
from utils import get_cats_dogs_ds

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from math import log
from tqdm import tqdm

colab = False

feature_extractor, preprocessor = get_pretrained_net('ResNet50V2')
feature_extractor.trainable = False

# save_path = 'checkpoints/resnet_claffifier'
# kernel = np.load(save_path + 'kernel.npy')
# bias = np.load(save_path + 'bias.npy')
# classifier = Dense(1)
#
# classifier.set_weights([kernel, bias])

train_ds, test_ds = get_cats_dogs_ds(preprocessor)
train_ds = train_ds.batch(64 if colab else 16)
shuffle_buffer_size = 1000

learning_rate = 0.0001
feature_extractor.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.evaluate(test_ds)

print('Start')
pred_ds = feature_extractor.predict(train_ds)


np.save('saved/resnet50v2/features.npy', pred_ds)