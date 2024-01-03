# this script can be used to compute the distribution
# of the turned on pixels in a given datatset


from chain_lib import Points
from chain_lib import positive_int
from chain_lib import imshow
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch as th
import argparse


# default parameters
dset_path_dflt = '../data/original_dataset-ubyte.gz'  # path to dataset
dset_size_dflt = 30000  # size of the dataset
y_size_dflt = 20  # height of the grids
x_size_dflt = 20  # width of the grids
plot_path_dflt = '../data/original_distribution.png'  # path to saved plot


if __name__ == "__main__":

    # getting parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_path', default=dset_path_dflt)
    parser.add_argument('--dset_size', type=positive_int, default=dset_size_dflt)
    parser.add_argument('--y_size', type=positive_int, default=y_size_dflt)
    parser.add_argument('--x_size', type=positive_int, default=x_size_dflt)
    parser.add_argument('--plot_path', default=plot_path_dflt)

    args = parser.parse_args()

    dset_path = args.dset_path
    dset_size = args.dset_size
    y_size = args.y_size
    x_size = args.x_size
    plot_path = args.plot_path

    # defining the dataloader
    data_transforms = transforms.ToTensor()
    dset = Points(dset_path, y_size, x_size, data_transforms)
    dset_loader = DataLoader(dataset=dset, batch_size=dset_size)

    # initializing the point distribution
    point_distribution = th.zeros((1, y_size, x_size))

    dataiter = dset_loader.__iter__()
    grids = dataiter.__next__()

    # computing the distribution
    for i in range(dset_size):
        point_distribution += grids[i]

    maxval = th.max(point_distribution)
    point_distribution /= maxval

    # saving the computed distribution
    imshow(make_grid(point_distribution), file_path=plot_path)
