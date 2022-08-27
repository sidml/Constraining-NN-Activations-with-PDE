from __future__ import print_function

import os
import pdb
import logging
import time
import argparse
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from resnet import resnet32, pdenet
# from custom_conv_resnet import resnet32_custom_conv
from loader import load_dataset, map_ds_cifar
from utils import get_weight_count, get_callbacks, MetricTracker, setup_logger, plot_cm


def train(args, model, model_name):
    x_train, y_train, x_test, y_test = load_dataset(cifar=True)
    model.compile()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.lr,
        momentum=0.9,
        nesterov=True,
        decay=5e-5,
        clipnorm=5.0,
    )

    log_dir = f"./logs/{model_name}/"
    img_dir = f"{log_dir}/images"

    setup_logger(log_dir, img_dir)
    callback_list = get_callbacks(model, model_name, log_dir)

    gamma = args.gamma

    def scheduler(epoch, lr, gamma):
        if epoch < 150:
            return lr
        elif (epoch > 150) & (epoch < 225):
            return lr * gamma
        else:
            return lr * gamma * gamma

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = tf.reduce_mean(loss_fn(labels, predictions))
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss, predictions

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        loss = tf.reduce_mean(loss_fn(labels, predictions))
        return loss, predictions

    # running the training loop
    callback_list.on_train_begin()

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = map_ds_cifar(ds_test, args.test_batch_size, False)
    metrics = MetricTracker()
    # exit()
    print("Training starts ...")
    for epoch in range(args.epochs):
        callback_list.on_epoch_begin(epoch)
        metrics.reset()

        optimizer.learning_rate = scheduler(epoch, args.lr, gamma)

        idx = np.arange(len(x_train))
        np.random.shuffle(idx)
        x_train, y_train = x_train[idx], y_train[idx]

        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_train = map_ds_cifar(ds_train, args.batch_size, True)

        optimizer.learning_rate = scheduler(epoch, args.lr, gamma)

        for batch, (images, labels) in enumerate(ds_train):
            loss, predictions = train_step(images, labels)
            metrics.train_loss(loss)
            metrics.train_accuracy(labels, predictions)

        cm = np.zeros((10, 10), dtype=np.float32)
        for step, (images, labels) in enumerate(ds_test):
            loss, predictions = test_step(images, labels)
            metrics.test_loss(loss)
            metrics.test_accuracy(labels, predictions)
            predictions = tf.argmax(predictions, 1)

            cm += confusion_matrix(
                predictions, labels, labels=np.arange(10), normalize="true"
            )

        cm = cm / len(ds_test)
        img_fn = f"{img_dir}/epoch_{epoch}_cnf_matrix.jpg"
        plot_cm(cm, img_fn)

        logs = metrics.get_logs()
        if args.non_linear_Dxy=="True":
            logs.update(
                {
                    "perona_malik_l1": model.get_layer("perona_malik").lam.numpy()[0],
                    "perona_malik_l2": model.get_layer("perona_malik_1").lam.numpy()[0],
                    "perona_malik_l3": model.get_layer("perona_malik_2").lam.numpy()[0],
                }
            )

        logging.info(f"Epoch {epoch}: {logs}")
        callback_list.on_epoch_end(epoch, logs=logs)


def main(args):

    tf.random.set_seed(args.seed)

    input_shape = (32, 32, 3)
    classes = 10
    if args.net == "pdenet":
        model_name = "pdenet"
        model = pdenet(
            input_shape=input_shape,
            classes=classes,
            name=model_name,
            global_feat=True,
            args=args,
        )
    elif args.net == "custom_conv":
        model_name = "resnet32_custom_conv"
        model = resnet32_custom_conv(
            input_shape=input_shape, classes=classes, name=model_name
        )
    else:
        model_name = "resnet32"
        model = resnet32(input_shape=input_shape, classes=classes, name=model_name)

    print(model.summary())
    total_params = get_weight_count(model)
    msg = "Total params = " + str(total_params)
    logging.info(msg)

    train(args, model, model_name)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="Tensorflow CIFAR-10 Example")

    parser.add_argument(
        "--net", default="pdenet", help="pdenet OR resnet OR custom_conv", type=str
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for training",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        help="test batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        metavar="N",
        help="number of epochs to train",
    )

    parser.add_argument(
        "--lr", type=float, default=0.1, metavar="LR", help="learning rate"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        metavar="M",
        help="Learning rate step gamma",
    )

    parser.add_argument("--dt", type=float, default=0.2, help="dt, default=0.2")
    parser.add_argument("--dx", type=int, default=1, help="dx, default=1")
    parser.add_argument("--dy", type=int, default=1, help="dy, default=1")
    parser.add_argument(
        "--K",
        default=1,
        type=int,
        metavar="K",
        help="Number of iterations in the Global feature extractor block (default: 3)",
    )
    parser.add_argument(
        "--cDx", type=float, default=1.0, help="Random erase prob (default: 0.)"
    )
    parser.add_argument(
        "--cDy", type=float, default=1.0, help="Random erase prob (default: 0.)"
    )

    parser.add_argument(
        "--disable_advection",
        default="False",
        help="Set true to disable the advection part",
        type=str,
    )

    parser.add_argument(
        "--disable_diffusion",
        default="False",
        help="Set true to disable the advection part",
        type=str,
    )

    parser.add_argument(
        "--non_linear_Dxy",
        default="False",
        help="set True to use Perona Malika non-linear diffusivity",
        type=str,
    )
    parser.add_argument("-constant_Dxy", "--constant_Dxy", default="False", type=str)
    parser.add_argument(
        "--anisotropic", default="False", help="Learnable anisotropic", type=str
    )

    parser.add_argument("--init_h0_h", default=False, type=str)

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    args = parser.parse_args()
    print("args = ")
    print(args)

    if args.non_linear_Dxy == "True":
        assert args.anisotropic == "False", "disable anisotropic or non_linear_Dxy"
    if args.constant_Dxy == "True":
        assert args.anisotropic == "False", "disable anisotropic  & non_linear_Dxy"
    if args.disable_diffusion == "True":
        assert (
            args.disable_advection == "False"
        ), "cannot disable anisotropic & diffusion simulataneously"
        assert (
            args.non_linear_Dxy == "False"
        ), "diffusion is disabled, so non_linear_xy must be false"
        assert (
            args.constant_Dxy == "False"
        ), "diffusion is disabled, so constant_Dxy must be false"

    main(args)
