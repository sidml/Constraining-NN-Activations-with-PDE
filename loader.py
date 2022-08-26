import tensorflow as tf
import numpy as np


def normalize_cifar(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    image = tf.image.resize(image, (32, 32))
    return image, label

# augmentation is used only for cifar
def augment(image, label):
    # replicate RandomCrop function in pytorch
    image = tf.image.resize_with_crop_or_pad(image, 36, 36)
    # seed = np.random.randint(0, 10000, (1,))
    image = tf.image.random_crop(image, (32, 32, 3))
    # random_flip_left_right 50% of the time
    image = tf.image.random_flip_left_right(image)
    return image, label


def normalize_mnist(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 0.1307) / 0.3081
    return image, label


def load_dataset(cifar=False):
    if cifar:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test

def map_ds_cifar(ds, batch_size, train):
    ds = ds.map(normalize_cifar, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if train:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def map_ds_mnist(ds, batch_size, shuffle):
    ds = ds.map(normalize_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
