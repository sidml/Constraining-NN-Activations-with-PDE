## Commands for replicating experiment results
All the experiments were done using Kaggle notebooks with GPU enabled.. This environment had Tensorflow 2.6 and other required libraries pre-installed .
### MNIST experiments
For training CNN model:
`python train_mnist_example.py --backbone=cnn`

For training Residual model:
`python train_mnist_example.py --backbone=residual`

For training PDE model:
`python train_mnist_example.py --backbone=pde`

### CIFAR-10 experiments
For original Resnet32 training:
`python train_cifar.py --net='resnet'`

For original PDENet training:
`python train_cifar.py --net='pdenet'`

For PDENet training with advection disabled:
`python train_cifar.py --net='pdenet' --disable_advection True`

For PDENet training with constant Dxy (Dx=1, Dy=1):
`python train_cifar.py --epochs 5 --net='pdenet' --constant_Dxy True`


## About the code
- train_cifar.py: This file can be used for training different networks on CIFAR-10 dataset. The training & evaluation loop and default training settings are defined in this file. 
- train_mnist_example.py: Can be used to train cnn, residual cnn & PDENet depending on the user provided arguments.
- mnist_model.py: Toy models used for MNIST experiment are defined here.
- utils.py: Plotting, logging, metric calculators and other miscellaneous functions are defined in this file.
- resnet.py: Resnet32 model is defined here. Depending on user settings, different types of diffusion and anisotropic blocks are invoked here.
- global_layer.py: Contains implementation of Diffusion-Advection layer as proposed in the paper and nonlinear diffusion based on Perona Malik diffusivity. It also contains implementation of learnable Anisotropic block.
- loader.py: Functions for MNIST & CIFAR data loading and augmentation are defined here.
- viz_pde.py: This script can be used to visualize Dx, Dy and advection terms using trained model checkpoint weights.

## Directory Structure of logs and checkpoint folder
The `expt_logs` folder contains detailed logs and checkpoints for various experiments.\
The directory structure is as follows:
```
--- expt_logs
    --- <expt_name>
        --- logs
            --- <model_name>
                --- images (contains confusion matrix/activation plots)
                --- train (contains tensorboard logs)                            
```
The logs folder also contains `settings_log.txt` file. This file contains the training/testing loss and accuracy.\
`checkpoints` folder contains the best checkpoints obtained during model training.\

