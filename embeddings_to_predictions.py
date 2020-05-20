from itertools import product

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# model_names = ['ResNet50']
model_names = ['ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'InceptionV3', 'Xception']
# stops = [1]
stops = [1, 30, 200]
for model_name, stop in product(model_names, stops):
    print(f'Processing model {model_name}, trained for {stop} epoch(s)')

    # get the classifier
    classifier = Dense(input_shape=(2048,), units=1)
    model = Sequential([classifier])
    weights = np.load(f'saved/classifier_weights/classifier_kernel_{model_name}_{stop}.npy'), \
              np.load(f'saved/classifier_weights/classifier_bias_{model_name}_{stop}.npy')
    weights = list(weights)
    classifier.set_weights(weights)

    # get embeddings
    embeddings = np.load(f'saved/embeddings_test/{model_name}.npy')

    # make predictions
    predictions = model.predict(embeddings, batch_size=256, verbose=1)
    print(f'Shape of predictions: {predictions.shape}')

    # save predictions
    np.save(f'saved/predictions_test/{model_name}_{stop}.npy', predictions)
