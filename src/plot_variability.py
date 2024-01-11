# this script can be used to compute variabilities within
# the datasets produced by a given chain of autoencoders
#
# variability is computed sampling a certain number of
# pairs of data points (computing variability on all data
# would not be computationally feasible)


from chain_lib import Points
from chain_lib import positive_int
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
import argparse
from random import sample
import matplotlib.pyplot as plt


# default parameters
num_models_dflt = 20  # number of autoencoders in the chain
dset_dir_dflt = '../data'  # directory containing data
y_size_dflt = 24  # height of grids
x_size_dflt = 24  # width of grids
dset_size_dflt = 20000  # dataset size


if __name__ == "__main__":

    # getting parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_models', type=positive_int, default=num_models_dflt)
    parser.add_argument('--dset_dir', type=str, default=dset_dir_dflt)
    parser.add_argument('--y_size', type=positive_int, default=y_size_dflt)
    parser.add_argument('--x_size', type=positive_int, default=x_size_dflt)
    parser.add_argument('--dset_size', type=positive_int, default=dset_size_dflt)

    args = parser.parse_args()

    num_models = args.num_models
    dset_dir = args.dset_dir
    y_size = args.y_size
    x_size = args.x_size
    dset_size = args.dset_size

    # defining number of pairs we use to compute variability
    # (computed as half of the dataset's size)
    num_pairs = int(dset_size / 2)

    # importing data
    data_transforms = transforms.ToTensor()
    batch_size = dset_size

    data_loaders = []  # list storing data loaders

    dset_path = dset_dir + '/original_dset-ubyte.gz'
    dset = Points(dset_path, y_size, x_size, transform=data_transforms)
    data_loaders.append(DataLoader(dataset=dset, batch_size=batch_size, shuffle=True))

    for model_id in range(num_models-1):
        dset_path = dset_dir + '/dset_' + str(model_id) + '-ubyte.gz'
        dset = Points(dset_path, y_size, x_size, transform=data_transforms)
        data_loaders.append(DataLoader(dataset=dset, batch_size=batch_size, shuffle=True))

    dset_path = dset_dir + '/final_dset-ubyte.gz'
    dset = Points(dset_path, y_size, x_size, transform=data_transforms)
    data_loaders.append(DataLoader(dataset=dset, batch_size=batch_size, shuffle=True))

    # computing variability within datasets
    dset_vars = []
    for loader in data_loaders:

        dataiter = loader.__iter__()
        images = dataiter.__next__()

        # computing variability
        dset_vars.append(0)
        for _ in range(num_pairs):
            pair_idx = tuple(sample(range(dset_size), k=2))  # sampling couple

            dset_vars[-1] += th.norm(images[pair_idx[0]]-images[pair_idx[1]], 2)  # computing L2 distance
                                                                                  # between images
        dset_vars[-1] /= num_pairs

    # plotting variability
    model_ids = list(range(-1, num_models))
    plt.plot(model_ids, dset_vars, marker='o', linestyle='-', color='b', label="datasets' variability")
    plt.xticks(model_ids)
    plt.xlabel('dataset')
    plt.ylabel('variability')
    plt.ylim(0, 5)
    plt.legend()
    plt.savefig(dset_dir + '/variability.png')
