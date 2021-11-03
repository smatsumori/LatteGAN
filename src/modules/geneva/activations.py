import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU(),  # used in ImageEncoder
    "leaky_relu": nn.LeakyReLU(),  # used in Generator and Discriminator
    "selu": nn.SELU(),
}
