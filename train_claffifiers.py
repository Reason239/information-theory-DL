import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

results = {'loss': {}, 'accuracy': {}}
# model_names = ['ResNet50']
model_names = ['ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'InceptionV3', 'Xception']
for model_name in model_names:
    print(f'Processing model {model_name}')

    classifier = Dense(units=1, input_shape=(2048,))
    model = Sequential([classifier])

    BATCH_SIZE = 256
    SHUFFLE_BUFFER_SIZE = 1000

    embeddings_train = np.load(f'saved/embeddings/{model_name}.npy')
    labels_train = np.load('saved/labels_train.npy')

    embeddings_test = np.load(f'saved/embeddings_test/{model_name}.npy')
    labels_test = np.load('saved/labels_test.npy')

    train_batches = tf.data.Dataset.from_tensor_slices((embeddings_train, labels_train)) \
        .shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    test_batches = tf.data.Dataset.from_tensor_slices((embeddings_test, labels_test)).batch(BATCH_SIZE)

    base_learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # stops = [0, 1]
    stops = [0, 1, 30, 200]
    hist_keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    all_history = {key: np.array([]) for key in hist_keys}
    for epoch_start, epoch_stop in zip(stops[:-1], stops[1:]):
        # for overfitting
        if epoch_stop == stops[-1]:
            optimizer = tf.keras.optimizers.Adam(lr=800 * base_learning_rate)
            model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=['accuracy'])

        history = model.fit(train_batches,
                            initial_epoch=epoch_start,
                            epochs=epoch_stop,
                            validation_data=test_batches)

        for key in hist_keys:
            all_history[key] = np.append(all_history[key], np.array(history.history[key]))

        kernel, bias = classifier.get_weights()
        np.save(f'saved/classifier_weights/classifier_kernel_{model_name}_{epoch_stop}.npy', kernel)
        np.save(f'saved/classifier_weights/classifier_bias_{model_name}_{epoch_stop}.npy', bias)

        eval_loss, eval_acc = all_history['val_loss'][-1], all_history['val_accuracy'][-1]
        results['loss'][f'{model_name}_{epoch_stop}'] = eval_loss
        results['accuracy'][f'{model_name}_{epoch_stop}'] = eval_acc

    acc = all_history['accuracy']
    val_acc = all_history['val_accuracy']

    loss = all_history['loss']
    val_loss = all_history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title(f'Training and Validation Accuracy for {model_name}')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.xlabel('epoch')
    plt.savefig(f'plots/training classifiers/classifier_{model_name}.png')

print(results)
with open('saved/classifier_results.pkl', 'wb') as f:
    pkl.dump(results, f)