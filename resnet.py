import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dense,
    Activation,
    Lambda,
    Input,
    GlobalAveragePooling2D,
    Normalization,
)

from tensorflow.keras.models import Model
import numpy as np
from global_layer import GlobalFeatureBlock_Diffusion, Anisotropic

# ResNet building block of two layers
def building_block(x, filter_size, filters, stride=1):

    # Save the input value for shortcut
    X_shortcut = x

    # Reshape shortcut for later adding if dimensions change
    if stride > 1:
        X_shortcut = Conv2D(
            filters, (1, 1), strides=stride, use_bias=False, padding="same"
        )(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # First layer of the block
    x = Conv2D(
        filters, kernel_size=filter_size, use_bias=False, strides=stride, padding="same"
    )(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    # Second layer of the block
    x = Conv2D(
        filters, kernel_size=filter_size, use_bias=False, strides=(1, 1), padding="same"
    )(x)
    x = BatchNormalization(axis=3)(x)
    # add shortcut and apply relu
    x = x + X_shortcut
    x = Activation("relu")(x)

    return x


# Full model
def resnet32(input_shape, classes, name):
    # Define the input
    inp = Input(input_shape)

    norm_layer = Normalization(
        mean=[
            (0.4914, 0.4822, 0.4465),
        ],
        variance=[np.square([0.2023, 0.1994, 0.2010])],
    )

    x = norm_layer(inp)

    # Stage 1
    x = Conv2D(
        filters=16, kernel_size=3, strides=(1, 1), padding="same", use_bias=False
    )(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    # Stage 2
    x = building_block(x, filter_size=3, filters=16, stride=1)
    x = building_block(x, filter_size=3, filters=16, stride=1)
    x = building_block(x, filter_size=3, filters=16, stride=1)
    x = building_block(x, filter_size=3, filters=16, stride=1)
    x = building_block(x, filter_size=3, filters=16, stride=1)

    # Stage 3
    x = building_block(
        x, filter_size=3, filters=32, stride=2
    )  # dimensions change (stride=2)
    x = building_block(x, filter_size=3, filters=32, stride=1)
    x = building_block(x, filter_size=3, filters=32, stride=1)
    x = building_block(x, filter_size=3, filters=32, stride=1)
    x = building_block(x, filter_size=3, filters=32, stride=1)

    # Stage 4
    x = building_block(
        x, filter_size=3, filters=64, stride=2
    )  # dimensions change (stride=2)
    x = building_block(x, filter_size=3, filters=64, stride=1)
    x = building_block(x, filter_size=3, filters=64, stride=1)
    x = building_block(x, filter_size=3, filters=64, stride=1)
    x = building_block(x, filter_size=3, filters=64, stride=1)

    # Average pooling and output layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes)(x)

    # Create model
    model = Model(inputs=inp, outputs=x, name=name)

    return model


def pdenet(input_shape, classes, name, global_feat, args):

    if global_feat:
        pde_args = {
            "K": args.K,
            "cDx": args.cDx,
            "cDy": args.cDy,
            "dx": args.dx,
            "dy": args.dy,
            "dt": args.dt,
            "constant_Dxy": args.constant_Dxy,
            "disable_advection": args.disable_advection,
            "disable_diffusion": args.disable_diffusion,
            "non_linear_Dxy": args.non_linear_Dxy,
            "learnable": "True",  # args.learnable
        }
        if args.anisotropic == "True":
            g1 = Anisotropic(16, pde_args, block_num="1")
            g2 = Anisotropic(32, pde_args, block_num="2")
            g3 = Anisotropic(64, pde_args, block_num="3")
        else:
            g1 = GlobalFeatureBlock_Diffusion(16, pde_args, block_num="1")
            g2 = GlobalFeatureBlock_Diffusion(32, pde_args, block_num="2")
            g3 = GlobalFeatureBlock_Diffusion(64, pde_args, block_num="3")

    # Define the input
    inp = Input(input_shape)

    norm_layer = tf.keras.layers.Normalization(
        mean=[
            (0.4914, 0.4822, 0.4465),
        ],
        variance=[np.square([0.2023, 0.1994, 0.2010])],
    )

    x = norm_layer(inp)

    # Stage 1
    x = Conv2D(
        filters=16, kernel_size=3, strides=(1, 1), padding="same", use_bias=False
    )(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    # Stage 2
    x = building_block(x, filter_size=3, filters=16, stride=1)
    if global_feat:
        x = g1(x)

    # Stage 3
    x = building_block(
        x, filter_size=3, filters=32, stride=2
    )  # dimensions change (stride=2)
    if global_feat:
        x = g2(x)

    # Stage 4
    x = building_block(
        x, filter_size=3, filters=64, stride=2
    )  # dimensions change (stride=2)
    if global_feat:
        x = g3(x)

    # Stage 5
    x = building_block(
        x, filter_size=3, filters=64, stride=2
    )  # dimensions change (stride=2)

    # Average pooling and output layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes)(x)

    # Create model
    model = Model(inputs=inp, outputs=x, name=name)

    return model
