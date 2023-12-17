# library containing all classes and functions needed in scripts


import gzip
import argparse
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


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
