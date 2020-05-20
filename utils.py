import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf


def add_noise(arr, noise_var):
    return arr + np.random.normal(size=arr.shape, scale=noise_var)


def get_info_ind_log(epochs, num_inf):
    """Get exactly num_inf integers from 0 to epochs approximately like in np.logspace"""
    num = num_inf
    ind = np.unique(np.round(np.logspace(start=0, stop=np.log10(epochs - 1), num=num).astype(dtype=np.int)))
    while len(ind) < num_inf:
        num += 1
        ind = np.unique(np.round(np.logspace(start=0, stop=np.log10(epochs - 1), num=num).astype(dtype=np.int)))
    return ind


def get_cats_dogs_ds(preprocessor):
    # for tfds-1.2.0
    (raw_train, raw_test), metadata = tfds.load(
        'cats_vs_dogs',
        data_dir='tensorflow_datasets',
        split=[
            tfds.Split.TRAIN.subsplit(tfds.percent[:90]),
            tfds.Split.TRAIN.subsplit(tfds.percent[90:])
        ],
        with_info=True,
        as_supervised=True,
    )
    img_size = 224  # All images will be resized to 224x224

    def format_example(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (img_size, img_size))
        if preprocessor is not None:
            image = preprocessor(image)
        return image, label

    train = raw_train.map(format_example)
    test = raw_test.map(format_example)

    return train, test

if __name__ == '__main__':
    from getting_models import get_pretrained_net
    net, pre = get_pretrained_net('ResNet50V2')
    train, test = get_cats_dogs_ds(pre)
    c = 0
    for img, lab in train:
        c += 1
    print(c)
    c = 0
    for img, lab in test:
        c += 1
    print(c)
