# library containing all classes and functions needed in scripts


import gzip
import argparse
import numpy as np
import torch as th
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import mse_loss
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# needed for input parameters of generate_dataset.py
def positive_int(value):

    ivalue = int(value)

    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


# function to show or save grid images
def imshow(img, file_path=None):

    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, axes=(1, 2, 0)))

    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()


# custom dataset of grids
class Points(Dataset):

    def __init__(self, dset_path, y_size, x_size, transform=None):
        # loading images
        with gzip.open(dset_path, 'rb') as f:
            self.images = np.frombuffer(f.read(), dtype=np.uint8)
            self.images = self.images.reshape((-1, y_size, x_size, 1))

        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # extracting the sampled images and labels
        image = self.images[idx].copy()

        # transforming output (if required)
        if self.transform:
            image = self.transform(image)

        return image


# variational autoencoder with fully connected NNs
class FC_AutoEncoder(nn.Module):

    def __init__(self, y_size, x_size, latent_size, device=None):
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
        self.latent_size = latent_size

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
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        # computing standard deviation
        std = th.exp(0.5 * log_var)
        # generating random noise
        eps = th.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        # flattening image
        x_flat = x.detach().clone().view(-1, self.img_size)

        # encoding input image
        encoded = self.encoder(x_flat)

        # extracting z from latent space
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)

        # deconding z
        decoded = self.decoder(z)

        return x_flat, encoded, decoded, mu, log_var

    # function to extract a sample of images
    # ('generative mode')
    def get_samples(self, num_samples):

        with th.no_grad():
            # generating samples in latent space
            z = th.randn(num_samples, self.latent_size).to(self.device)
            # decoding samples
            samples = self.decoder(z)

        return samples


# variational autoencoder with convolutional NNs
class Conv_AutoEncoder(nn.Module):

    def __init__(self, y_size, x_size, latent_size, device=None):
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
        self.latent_size = latent_size

        # defining the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear((self.x_size-2) * (self.y_size-2) * 32, self.inter_size),
            nn.ReLU(),
        )

        # defining layers of 'specialization' for encoder
        # (i.e. final layers to compute means and
        # log-variances)
        self.mu = nn.Linear(self.inter_size, self.latent_size)
        self.log_var = nn.Linear(self.inter_size, self.latent_size)

        # defining the decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.inter_size),
            nn.ReLU(),
            nn.Linear(self.inter_size, (self.x_size - 2) * (self.y_size - 2) * 32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32, self.x_size - 2, self.y_size - 2)),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3),
            nn.Sigmoid(),
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

        return x, encoded, decoded, mu, log_var

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
class loss_function(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x, mu, log_var):
        # recustruction error
        MSE = mse_loss(recon_x, x, reduction='sum')

        # regularization term (Kullback-Leibler divergence)
        KLD = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return MSE + KLD


# function to train a single autoencoder (an instantiated model
# must be passed as input parameter and the trained model is
# returned as output)
def train_AutoEncoder(model, device, dset_path, y_size, x_size, learning_rate=0.001, num_epochs=5):

    # defining hyperparameters
    criterion = loss_function()
    optimizer = th.optim.Adam(params=model.parameters(), lr=learning_rate)
    batch_size = 32

    # selecting device
    model = model.to(device)

    # defining the data loader
    data_transforms = transforms.ToTensor()
    train_dataset = Points(dset_path=dset_path, y_size=y_size, x_size=x_size, transform=data_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # train mode
    model.train()

    # training loop
    for epoch in range(num_epochs):

        epoch_loss = 0.0  # epoch running loss

        for batch_number, images in enumerate(train_loader):

            images = images.to(device)

            # computing recostructed images and latent variables
            target, _, decoded, mu, log_var = model(images)

            ##############################################################
            # if batch_number == 0:
            #     imshow(make_grid(target[0]), 'target_'+str(epoch)+'.png')
            #     imshow(make_grid(decoded[0]), 'recon_'+str(epoch)+'.png')
            ##############################################################

            # computing loss
            loss = criterion(decoded, target, mu, log_var)

            # performing backpropagation and updating params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # updating running loss
            epoch_loss += loss.item() * images.size(0)

        # printing the epoch loss
        epoch_loss /= len(train_loader.dataset)
        print(f"\tloss at epoch {epoch}: {epoch_loss}")

    # evaluation mode
    model.eval()

    return model


# function to generate dataset using autoencoder and write it to file
def generate_dset(model, device, dset_path, dset_size):

    # selecting device
    model = model.to(device)

    # setting evaluation mode
    model = model.eval()

    # generating syntetic dataset
    dset = model.get_samples(dset_size)

    ##############################################################
    # imshow(make_grid(dset[0]), 'generated_0.png')
    # imshow(make_grid(dset[1]), 'generated_1.png')
    # imshow(make_grid(dset[2]), 'generated_2.png')
    ##############################################################

    # writing dataset to file
    dset *= 255  # rescaling for grey scale
    dset_rounded = th.round(dset).detach().clone().cpu()  # creating cpu rounded copy
    dset_np = dset_rounded.numpy().astype(np.ubyte)  # converting to numpy array
    dset_byte = dset_np.tobytes()  # converting to byte array
    with gzip.open(dset_path, 'wb') as f:
        f.write(dset_byte)
