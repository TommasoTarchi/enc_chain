import gzip
import argparse
import numpy as np
from torch.utils.data import Dataset


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


class Rnd_grids(Dataset):

    def __init__(self, y_size, x_size, transform=None):
        # loading images
        with gzip.open('../data/original_dataset-ubyte.gz', 'rb') as f:
            self.images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            self.images = self.images.reshape((-1, y_size, x_size, 1))

        self.transform = transform

    def __len__(self):
        return len(self.images.shape[0])

    def __getitem__(self, idx):
        # extracting the sampled images
        image = self.images[idx]

        # transforming output (if required)
        if self.transform:
            image = self.transform(image)

        return image
