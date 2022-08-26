import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from resnet import pdenet
from loader import load_dataset, map_ds_cifar
import os


def plot_orig_images(images, viz_dir):
    rows = int(np.sqrt(images.shape[0]))
    cols = images.shape[0] // rows
    fig = plt.figure(figsize=(4.0, 4.0), tight_layout=True)
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(rows, cols),
        axes_pad=0.1,  # pad between axes in inch.
    )

    for im_num, ax in enumerate(grid):
        ax.axis("off")
        ax.imshow(images[im_num])
    plt.savefig(f"./{viz_dir}/orig_images.png")


def plot(activations, viz_dir, name):
    if activations.shape[-1] == 16:
        rows, cols = 4, 4
    elif activations.shape[-1] == 32:
        rows, cols = 4, 8
    elif activations.shape[-1] == 64:
        rows, cols = 8, 8
    else:
        print("using auto mode for grid display")
        rows = int(np.sqrt(activations.shape[-1]))
        cols = activations.shape[-1] // rows

    for im_num, act in enumerate(activations):
        save_dir = f"./{viz_dir}/im_num{im_num}"
        os.makedirs(save_dir, exist_ok=True)
        fig = plt.figure(figsize=(rows, cols), tight_layout=True)
        grid = ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
            axes_pad=0.05,  # pad between axes in inch.
        )

        for chan, ax in enumerate(grid):
            ax.axis("off")
            ax.imshow(act[:, :, chan])
        plt.savefig(f"{save_dir}/{name}.png")
        plt.close()


def main(args):
    input_shape = (32, 32, 3)
    classes = 10
    model = pdenet(
        input_shape=input_shape,
        classes=classes,
        name="pdenet",
        global_feat=True,
        args=args,
    )
    # tf.keras.utils.plot_model(model, to_file="model.png",show_shapes=True)

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(args.checkpoint_filepath)

    # https://github.com/keras-team/keras/issues/2495#issuecomment-602092838
    ux_layer1 = model.get_layer("ux_block1").output
    vy_layer1 = model.get_layer("vy_block1").output

    ux_layer2 = model.get_layer("ux_block2").output
    vy_layer2 = model.get_layer("vy_block2").output

    ux_layer3 = model.get_layer("ux_block3").output
    vy_layer3 = model.get_layer("vy_block3").output

    g_layer1 = model.get_layer("g_block1").output
    g1_layer1 = model.get_layer("g1_block1").output

    g_layer2 = model.get_layer("g_block2").output
    g1_layer2 = model.get_layer("g1_block2").output

    g_layer3 = model.get_layer("g_block3").output
    g1_layer3 = model.get_layer("g1_block3").output

    Dx_layer1 = model.get_layer("Dx_block1").output
    Dy_layer1 = model.get_layer("Dy_block1").output

    Dx_layer2 = model.get_layer("Dx_block2").output
    Dy_layer2 = model.get_layer("Dy_block2").output

    Dx_layer3 = model.get_layer("Dx_block3").output
    Dy_layer3 = model.get_layer("Dy_block3").output

    out = [ux_layer1, ux_layer2, ux_layer3, vy_layer1, vy_layer2, vy_layer3]
    out.extend([Dx_layer1, Dx_layer2, Dx_layer3, Dy_layer1, Dy_layer2, Dy_layer3])
    out.extend([g_layer1, g_layer2, g_layer3, g1_layer1, g1_layer2, g1_layer3])
    inter_output_model = tf.keras.Model(model.input, out)
    print()

    _, _, x_test, y_test = load_dataset(cifar=True)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = map_ds_cifar(ds_test, args.test_batch_size, False)

    viz_dir = "./pde_activations"
    os.makedirs(viz_dir, exist_ok=True)
    for step, (images, labels) in enumerate(ds_test):
        if step > args.plot_batches - 1:
            break
        print("step=", step)
        plot_orig_images(images, viz_dir)
        predictions = inter_output_model(images, training=False)
        for i, p in enumerate(predictions[:3]):
            plot(p, viz_dir, f"ux_layer{i}")
        for i, p in enumerate(predictions[3:6]):
            plot(p, viz_dir, f"vy_layer{i}")
        for i, p in enumerate(predictions[6:9]):
            plot(p, viz_dir, f"Dx_layer{i}")
        for i, p in enumerate(predictions[9:12]):
            plot(p, viz_dir, f"Dy_layer{i}")

        for i, p in enumerate(predictions[12:15]):
            plot(p, viz_dir, f"g_layer{i}")
        for i, p in enumerate(predictions[15:18]):
            plot(p, viz_dir, f"g1_layer{i}")


if __name__ == "__main__":

    # args = get_args()
    # args.n1, args.n2, args.n3, args.n4 = 16, 32, 64, 64
    # args.custom_uv = ""
    # args.non_linear = True
    # args.resnet_m = 1
    # args.K = 1
    # args.test_batch_size = 4

    parser = argparse.ArgumentParser(
        description="Visualize advection & diffusion terms"
    )

    parser.add_argument(
        "--checkpoint-filepath",
        type=str,
        default="../expt_logs/pdenet/checkpoints/pdenet/cp-0241.ckpt",
        help="checkpoint filepath",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=2,
        help="test batch size for testing",
    )

    parser.add_argument(
        "--plot-batches",
        type=int,
        default=1,
        help="number of batches that you wish to plot",
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

    main(args)
