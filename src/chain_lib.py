# library containing all classes and functions needed in scripts


import gzip
import argparse
import numpy as np
import torch as th
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt


# setting default parameters
y_size_dflt = 50  # height of the grids
x_size_dflt = 50  # width of the grids


# needed for input parameters of generate_dataset.py
def positive_int(value):

    ivalue = int(value)

    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


# class of random grids used as samples in the first dataset
# (the one used to train the first autoencoder of the chain)
class Rnd_grids(Dataset):

    def __init__(self, file_path, y_size, x_size, transform=None):
        # loading images
        with gzip.open(file_path, 'rb') as f:
            self.images = np.frombuffer(f.read(), dtype=np.uint8, offset=0)
            self.images = self.images.reshape((-1, y_size, x_size))

        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # extracting the sampled images
        image = self.images[idx].copy()

        # transforming output (if required)
        if self.transform:
            image = self.transform(image)

        return image


# function to show or save grid images
def imshow(img, file_path=None):

    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, axes=(1, 2, 0)))

    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()


# variational autoencoder with fully connected NNs
class FC_AutoEncoder(nn.Module):

    def __init__(self, y_size=y_size_dflt, x_size=x_size_dflt, device=None):
        super().__init__()

        # getting the device
        self.device = device

        # computing size of grids
        self.y_size = y_size
        self.x_size = x_size
        self.img_size = self.y_size * self.x_size

        # computing number of nodes in the intermediate
        # layers of encoder and decoder (chosen to be one
        # fifth of the image size)
        self.inter_size = int(self.img_size / 5)

        # setting size of the latent space
        self.latent_size = 8

        # defining the encoder (we use a common encoder
        # for means and log-variances)
        self.encoder = nn.Sequential(
            nn.Linear(self.img_size, self.inter_size),
            nn.ReLU(),
            nn.Linear(self.inter_size, self.latent_size),
            nn.ReLU(),
        )

        # defining layers of 'specialization' for encoder
        # (i.e. final layers to compute means and
        # log-variances)
        self.mu = nn.Linear(self.latent_size, self.latent_size)
        self.log_var = nn.Linear(self.latent_size, self.latent_size)

        # defining the decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.inter_size),
            nn.ReLU(),
            nn.Linear(self.inter_size, self.img_size),
            nn.Softmax(),
        )

    def reparameterize(self, mu, log_var):
        # computing standard deviation
        std = th.exp(0.5 * log_var)
        # generating random noise
        eps = th.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        # encoding input image
        encoded = self.encoder(x)

        # extracting z from latent space
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)

        # deconding z
        decoded = self.decoder(z)

        return encoded, decoded, mu, log_var

    # function to extract a sample of images
    # ('generative mode')
    def get_samples(self, num_samples):

        with th.no_grad():
            # generating samples in latent space
            z = th.randn(num_samples, self.latent_size).to(self.device)
            # decoding samples
            samples = self.decoder(z)

        return samples


# loss function for variational autoencoder
def loss_function(x, recon_x, mu, log_var):

    # recustruction error (MSE)
    MSE = mse_loss(x, recon_x)

    # regularization term (Kullback-Leibler divergence)
    KLD = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return MSE + KLD
