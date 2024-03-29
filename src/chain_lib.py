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
import matplotlib.pyplot as plt


# needed for input parameters
def positive_int(value):

    ivalue = int(value)

    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


# needed for input parameters
def positive_float(value):

    fvalue = float(value)

    if fvalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive float")
    return fvalue


# needed for input parameters
def fraction_value(value):

    fvalue = float(value)

    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(f"{value} is not a fraction")
    return fvalue


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


# variational autoencoder with two layers-fully connected NNs
class VAE_FC1(nn.Module):

    def __init__(self, y_size, x_size, latent_size, device=None):
        super().__init__()

        # getting the device
        self.device = device

        # computing size of grids
        self.y_size = y_size
        self.x_size = x_size
        self.img_size = self.y_size * self.x_size

        # computing number of nodes in the intermediate
        # layer of encoder and decoder (chosen to be one
        # fifth of the image size)
        self.inter_size = int(self.img_size / 5)

        # setting size of the latent space
        self.latent_size = latent_size

        # defining the encoder (we use a common encoder
        # for means and log-variances)
        self.encoder = nn.Sequential(
                nn.Linear(self.img_size, self.inter_size),
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


# variational autoencoder with three layers-fully connected NNs
class VAE_FC2(nn.Module):

    def __init__(self, y_size, x_size, latent_size, std_increment, device=None):
        super().__init__()

        # computing the standard deviation
        self.std_modified = 1.0 + std_increment

        # getting the device
        self.device = device

        # getting the device
        self.device = device

        # computing size of grids
        self.y_size = y_size
        self.x_size = x_size
        self.img_size = self.y_size * self.x_size

        # computing number of nodes in the intermediate
        # layers of encoder and decoder (chosen to be one
        # fourth and one twentieth of the image size)
        self.inter_size1 = int(self.img_size / 4)
        self.inter_size2 = int(self.img_size / 20)

        # setting size of the latent space
        self.latent_size = latent_size

        # defining the encoder (we use a common encoder
        # for means and log-variances)
        self.encoder = nn.Sequential(
                nn.Linear(self.img_size, self.inter_size1),
                nn.ReLU(),
                nn.Linear(self.inter_size1, self.inter_size2),
                nn.ReLU(),
        )

        # defining layers of 'specialization' for encoder
        # (i.e. final layers to compute means and
        # log-variances)
        self.mu = nn.Linear(self.inter_size2, self.latent_size)
        self.log_var = nn.Linear(self.inter_size2, self.latent_size)

        # defining the decoder
        self.decoder = nn.Sequential(
                nn.Linear(self.latent_size, self.inter_size2),
                nn.ReLU(),
                nn.Linear(self.inter_size2, self.inter_size1),
                nn.ReLU(),
                nn.Linear(self.inter_size1, self.img_size),
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
            z = th.randn(num_samples, self.latent_size).to(self.device) * self.std_modified

            # decoding samples
            samples = self.decoder(z)

        return samples


# variational autoencoder with convolutional NNs
class VAE_Conv(nn.Module):

    def __init__(self, y_size, x_size, latent_size, std_increment, device=None):
        super().__init__()

        # computing the standard deviation
        self.std_modified = 1.0 + std_increment

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
            z = th.randn(num_samples, self.latent_size).to(self.device) * self.std_modified

            # decoding samples
            samples = self.decoder(z)

        return samples


# variational autoencoder with asymmetric structure
# (convolutional NN as encoder and fully connected NN
# as decoder)
class VAE_Asymm(nn.Module):

    def __init__(self, y_size, x_size, latent_size, device=None):
        super().__init__()

        # getting the device
        self.device = device

        # computing size of grids
        self.y_size = y_size
        self.x_size = x_size
        self.img_size = self.y_size * self.x_size

        # computing number of nodes in the intermediate
        # layers of encoder and decoder
        self.inter_size = int(self.img_size / 5)
        self.inter_size1 = int(self.img_size / 4)
        self.inter_size2 = int(self.img_size / 20)

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
                nn.Linear(self.latent_size, self.inter_size2),
                nn.ReLU(),
                nn.Linear(self.inter_size2, self.inter_size1),
                nn.ReLU(),
                nn.Linear(self.inter_size1, self.img_size),
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
        decoded = decoded.reshape((-1, 1, self.y_size, self.x_size))

        return x, encoded, decoded, mu, log_var

    # function to extract a sample of images
    # ('generative mode')
    def get_samples(self, num_samples):

        with th.no_grad():
            # generating samples in latent space
            z = th.randn(num_samples, self.latent_size).to(self.device)

            # decoding samples
            samples = self.decoder(z)
            samples = samples.reshape((-1, 1, self.y_size, self.x_size))

        return samples


# loss function for variational autoencoder
class loss_function(nn.Module):

    def __init__(self, lamb, kappa, y_size, x_size):
        super().__init__()

        self.lamb = lamb  # constant of the regularization term (KLD)
        self.kappa = kappa  # constant of the variability term

        # getting shape of images
        self.y_size = y_size
        self.x_size = x_size

    # function to compute (consecutive pairs) variability in a batch
    def comp_VAR(self, x):
        num_features = self.x_size * self.y_size
        x_flat = x.view(-1, num_features)
        num_samples = x_flat.shape[0]
        x_pairs = x_flat.view(num_samples // 2, 2, num_features)
        VAR = mse_loss(x_pairs[:, 0, :], x_pairs[:, 1, :], reduction='sum') / num_samples

        return VAR

    def forward(self, recon_x, x, mu, log_var):
        # recustruction error
        MSE = mse_loss(recon_x, x, reduction='sum')

        # regularization term (Kullback-Leibler divergence)
        KLD = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # variability term (only computed if asked, because very
        # inefficient)
        VAR = 0.0
        if self.kappa != 0:
            VAR += th.abs(self.comp_VAR(x) - self.comp_VAR(recon_x))

        return MSE + self.lamb * KLD + self.kappa * VAR


# function to train a single autoencoder (an instantiated model
# must be passed as input parameter and the trained model is
# returned as output)
def train_VAE(model, device, dset_path, y_size, x_size, regul_const=1, var_const=1, batch_size=32, learning_rate=0.001, num_epochs=5):

    # defining hyperparameters
    criterion = loss_function(regul_const, var_const, y_size, x_size)
    optimizer = th.optim.Adam(params=model.parameters(), lr=learning_rate)

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

        for _, images in enumerate(train_loader):

            images = images.to(device)

            # computing recostructed images and latent variables
            target, _, decoded, mu, log_var = model(images)

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

    # writing dataset to file
    dset *= 255  # rescaling for grey scale
    dset_rounded = th.round(dset).detach().clone().cpu()  # creating cpu rounded copy
    dset_np = dset_rounded.numpy().astype(np.ubyte)  # converting to numpy array
    dset_byte = dset_np.tobytes()  # converting to byte array
    with gzip.open(dset_path, 'wb') as f:
        f.write(dset_byte)


# function to denoise data (threshold is the fraction of
# the maximum grey value below which the pixel is turned off)
def denoise_dset(dset_path, dset_size, y_size, x_size, threshold=0.1):

    # loading dataset
    data_transforms = transforms.ToTensor()
    dset = Points(dset_path, y_size, x_size, data_transforms)
    dset_loader = DataLoader(dataset=dset, batch_size=dset_size)

    dataiter = dset_loader.__iter__()
    grids = dataiter.__next__()

    # computing threshold
    threshold *= th.max(grids)

    # denoising data
    grids[grids < threshold] = 0

    # writing denoised dataset to file (the old
    # dataset is overwritten)
    dset_dns = grids * 255  # rescaling for grey scale
    dset_dns_rounded = th.round(dset_dns).detach().clone().cpu()  # creating cpu rounded copy
    dset_dns_np = dset_dns_rounded.numpy().astype(np.ubyte)  # converting to numpy array
    dset_dns_byte = dset_dns_np.tobytes()  # converting to byte array
    with gzip.open(dset_path, 'wb') as f:
        f.write(dset_dns_byte)
