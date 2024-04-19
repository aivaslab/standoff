# Appendix

## Models

In this paper, we used one primary architecture, and one set of hyperparameters. Seeds for both random initialization and minibatch order were assigned by default pytorch settings. The models were always trained using the Adam optimizer with all parameters default other than lr, which we specify. Our model is described as follows:

- **Convolutional Layer 1:** Uses `kernels` number of kernels with size `kernel_size1`, stride `stride1`, and padding `padding1`.
- **Pooling Layer 1 (conditional):** Max pooling with kernel size `pool_kernel_size` and stride `pool_stride`. Applied conditionally based on `use_pool`.
- **Convolutional Layer 2 (conditional):** Uses `kernels2` number of kernels. Applied conditionally based on `use_conv2`.
- **Pooling Layer 2 (conditional):** Max pooling with kernel size `pool_kernel_size` and stride `pool_stride`. Applied conditionally based on `use_pool` AND `use_conv2`.
- **LSTM Layer:** LSTM network with `hidden_size` hidden units and `num_layers` number of layers.
- **Fully Connected Layer:** Maps from the LSTM hidden state to the desired output size. The output size is conditioned on `oracle_is_target`, which we use for the *b-loc* target.

### Explored Hyperparameter Space

| **Hyperparameter**       | **Explored Values**      |
|--------------------------|--------------------------|
| `hidden_size`            | 6, 8, 12, 16, 32         |
| `num_layers`             | 1, 2, 3                  |
| `kernels`                | 4, 8, 16, 24, 32         |
| `kernel_size1`           | 1, 3, 5                  |
| `kernel_size2`           | 1, 3, 5                  |
| `stride1`                | 1, 2                     |
| `pool_kernel_size`       | 2, 3                     |
| `pool_stride`            | 1, 2                     |
| `padding1`               | 0, 1                     |
| `padding2`               | 0, 1                     |
| `use_pool`               | True, False              |
| `use_conv2`              | True, False              |
| `kernels2`               | 8, 16, 32, 48            |
| `batch_size`             | 64, 128, 256             |
| `lr`                     | 0.0005, 0.001, 0.002, 0.005 |

The hyperparameters used, and descriptions of each, may be seen in the following table.

### Hyperparameter descriptions and values

| **Hyperparameter**   | **Description**                                   | **Value** |
|----------------------|---------------------------------------------------|-----------|
| `hidden_size`        | Size of the LSTM hidden state.                    | 32        |
| `num_layers`         | Number of LSTM layers.                            | 3         |
| `kernels`            | Number of kernels in the first convolutional layer.| 16       |
| `kernels2`           | Number of kernels in the second convolutional layer.| 16      |
| `kernel_size1`       | Size of the kernel in the first convolutional layer.| 3        |
| `kernel_size2`       | Size of the kernel in the second convolutional layer.| 5       |
| `stride1`            | Stride for the convolutional layers.              | 1         |
| `pool_kernel_size`   | Size of the pooling kernel.                       | 3         |
| `pool_stride`        | Stride for the pooling operation.                 | 1         |
| `padding1`           | Padding for the first convolutional layer.        | 1         |
| `padding2`           | Padding for the second convolutional layer.       | 1         |
| `use_pool`           | Boolean indicating if max pooling is used.        | false     |
| `use_conv2`          | Boolean indicating if a second convolutional layer is used. | false |
| `batch_size`         | Batch size for training.                          | 256       |
| `lr`                 | Learning rate for optimization.                   | 0.001     |

## Model inputs

Environment observations are seven by seven. They are of a rich format, rather than RGB, where each channel has a specific meaning. The channels representing certain attributes of blocks including solidity, opacity, and volatility, representing collisions with agents, obscuring agents’ vision, and that they might disappear (and will thus be non-solid for the purpose of pathfinding). These attributes are compacted into one channel, with each attribute comprising one bit. Another channel shows the reward of treats, if they are visible. Additional channels include filters that indicate the presence of an agent or a box.

## Hyperparameter tuning

We arrived at the hyperparameter set using a random search over a large space. Since we are not attempting to solve our own benchmark test in this paper, we are not particularly concerned with having the best possible hyperparameters. Rather, we wish to find a set of parameters which converges to high-accuracy policies relatively consistently, as a reasonable baseline for future comparison.

We tested 95 different hyperparameter sets, selected completely randomly from the space of valid combinations of hyperparameters. Each set was used to generate three independent models with different random initializations and batches. The train and test sets were both drawn from the *Direct* regime, at a uniformly random 80/20% split. Our selected hyperparameter set is the one with the best mean performance among the three models.

Notably, the model did not make use of the second convolutional layer. This could be due to the tiny input size (seven by seven), but could very well be due to random chance.

## Hardware

The experiments in this paper were run on two machines. Hyperparameter tuning and Experiments 1 and 2 were run on an Ubuntu 18.04.4 machine with an Intel Xeon(R) CPU ES-1620 vs @3.70GHz x 8, 32GB RAM, a GeForce GTX TITAN X GPU. The hyperparameter tuning process took 24 hours. Dataset generation, of nearly 600,000 individual datapoints, took around 1 and a half hours to run on either machine. Experiment 2 took around 15 hours to run. Experiment 3 was run on a Windows 10 machine with an AMD Ryzen 7 1700 CPU, 32GB RAM, and a 4GB GeForce GTX 1080 GPU, and took around 12 hours to run.
