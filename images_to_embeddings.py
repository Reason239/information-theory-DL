import numpy as np
import tensorflow as tf

from getting_models import get_pretrained_net
from utils import get_cats_dogs_ds

# model_names = ['ResNet50']
model_names = ['ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'InceptionV3', 'Xception']
for model_name in model_names:
    print(f'Processing model {model_name}')

    feature_extractor, preprocessor = get_pretrained_net(model_name)
    feature_extractor.trainable = False

    model = feature_extractor

    train, test = get_cats_dogs_ds(preprocessor)

    IMG_SIZE = 224  # All images will be resized to 224x224
    BATCH_SIZE = 256

    test_batches = test.batch(BATCH_SIZE)

    # compiling does not matter, it's for .predict() only
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # model.summary()

    embeddings = model.predict(test_batches)
    print(f'Shape of embeddings: {embeddings.shape}')

    np.save(f'saved/embeddings_test/{model_name}.npy', embeddings)

