from __future__ import print_function

import os

import logging
import time
import argparse
import numpy as np
import tensorflow as tf

import sys

from sklearn.metrics import confusion_matrix
from mnist_model import net
from loader import load_dataset, map_ds_mnist
from utils import (
    get_weight_count,
    get_callbacks,
    MetricTracker,
    setup_logger,
    plot_feat,
    plot_cm,
)


def train(args, model, model_name):
    x_train, y_train, x_test, y_test = load_dataset(cifar=False)

    model.compile()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.lr,
        momentum=0.9,
        nesterov=True,
        decay=5e-4,
        clipnorm=5.0,
    )

    log_dir = f"logs/{model_name}/"
    img_dir = f"{log_dir}/images"

    setup_logger(log_dir, img_dir)
    callback_list = get_callbacks(model, model_name, log_dir)

    gamma = args.gamma

    def scheduler(epoch, lr, gamma):
        if epoch < 60:
            return lr
        elif (epoch > 60) & (epoch < 120):
            return lr * gamma
        elif (epoch > 120) & (epoch < 160):
            return lr * gamma * gamma
        else:
            return lr * gamma * gamma * gamma

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions, _, _ = model(images, training=True)
            loss = tf.reduce_mean(loss_fn(labels, predictions))
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss, predictions

    @tf.function
    def test_step(images, labels):
        predictions, feat1, feat2 = model(images, training=False)
        loss = tf.reduce_mean(loss_fn(labels, predictions))
        return loss, feat1, feat2, predictions

    callback_list.on_train_begin()

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = map_ds_mnist(ds_test, args.test_batch_size, False)
    metrics = MetricTracker()

    print("Training starts ...")
    for epoch in range(args.epochs):
        callback_list.on_epoch_begin(epoch)
        metrics.reset()

        optimizer.learning_rate = scheduler(epoch, args.lr, gamma)

        idx = np.arange(len(x_train))
        np.random.shuffle(idx)
        x_train, y_train = x_train[idx], y_train[idx]
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_train = map_ds_mnist(ds_train, args.batch_size, True)

        for batch, (images, labels) in enumerate(ds_train):
            loss, predictions = train_step(images, labels)
            metrics.train_loss(loss)
            metrics.train_accuracy(labels, predictions)

        cm = np.zeros((10, 10), dtype=np.float32)
        for step, (images, labels) in enumerate(ds_test):
            loss, feat1, feat2, predictions = test_step(images, labels)
            metrics.test_loss(loss)
            metrics.test_accuracy(labels, predictions)
            predictions = tf.argmax(predictions, 1)

            cm += confusion_matrix(
                predictions, labels, labels=np.arange(10), normalize="true"
            )

            if step == 0:
                for i, (im, f1, f2) in enumerate(zip(images, feat1, feat2)):
                    if i > 8:
                        continue
                    img_fn = f"{img_dir}/epoch_{epoch}_{i}.jpg"
                    plot_feat(im, f1, f2, img_fn)

        cm = cm / len(ds_test)
        img_fn = f"{img_dir}/epoch_{epoch}_cnf_matrix.jpg"
        plot_cm(cm, img_fn)
        logs = metrics.get_logs()

        logging.info(f"Epoch {epoch}: {logs}")
        callback_list.on_epoch_end(epoch, logs=logs)


def main(args):

    tf.random.set_seed(args.seed)
    backbone = args.backbone
    if backbone == "cnn":
        model = net(backbone="cnn")
    elif backbone == "residual":
        model = net(backbone="residual", K=1)
    elif backbone == "pde":
        model = net(backbone="pde")
    else:
        raise NotImplementedError()
    print(model.summary())
    total_params = get_weight_count(model)
    msg = "Total params = " + str(total_params)
    logging.info(msg)

    train(args, model, backbone)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="Tensorflow MNIST Example")

    parser.add_argument(
        "--backbone",
        type=str,
        default="pde",
        help="cnn or residual or pde",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing",
    )

    # parser.add_argument('--epochs', type=int, default=200, metavar='N',
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        # parser.add_argument('--epochs', type=int, default=1, metavar='N',
        help="number of epochs to train",
    )

    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        metavar="M",
        help="Learning rate step gamma",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    args = parser.parse_args()
    main(args)
