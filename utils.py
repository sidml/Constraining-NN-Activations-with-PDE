import tensorflow as tf
import logging
from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def setup_logger(log_dir, img_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(img_dir).mkdir(parents=True, exist_ok=True)

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(
        logging.FileHandler(filename=log_dir + "settings_log.txt", encoding="utf-8")
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def get_callbacks(model, model_name, log_dir):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint_dir = f"checkpoints/{model_name}/" + "cp-{epoch:04d}.ckpt"
    save_best_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_dir,
        save_weights_only=True,
        monitor="test_accuracy",
        mode="max",
        save_best_only=True,
        verbose=True,
    )
    callback_list = tf.keras.callbacks.CallbackList(
        [tensorboard_callback, save_best_callback],
        add_history=True,
        model=model,
    )
    return callback_list


def get_weight_count(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])


def plot_cm(cm, img_fn):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm * 100, display_labels=np.arange(10)
    )
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    disp.plot(values_format=".3g", ax=ax)
    plt.tight_layout()
    plt.savefig(img_fn)
    plt.close()


def plot_feat(im, f1, f2, img_fn):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(im.numpy())
    plt.subplot(132)
    plt.imshow(f1.numpy())
    plt.subplot(133)
    plt.imshow(f2.numpy())
    plt.tight_layout()
    plt.savefig(img_fn)
    plt.close()


class MetricTracker:
    def __init__(self):
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )

        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="test_accuracy"
        )

    def reset(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

    def get_logs(self):
        logs = {
            "train_loss": self.train_loss.result(),
            "train_accuracy": self.train_accuracy.result() * 100,
            "test_loss": self.test_loss.result(),
            "test_accuracy": self.test_accuracy.result() * 100,
        }
        return logs


# initialize the cnn kernel weights similar to pytorch
# it seems to give poorer results
def my_init(shape, dtype=tf.float32):
    limit = -tf.sqrt(1 / shape[0])
    return tf.random.uniform(shape, -limit, limit, dtype=dtype)
