from chain_lib import Rnd_grids
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


BATCH_SIZE = 32
data_transforms = transforms.ToTensor()

train_dataset = Rnd_grids(transform=data_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


def imshow(img) -> None:
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, axes=(1, 2, 0)))


# get some random training images
dataiter = train_loader.__iter__()
images = dataiter.__next__()

# show images
imshow(torchvision.utils.make_grid(images))
