# this script can be used to compute differences between
# input and output datasets of all VAEs of a given chain
# of autoencoders


from chain_lib import Points
from chain_lib import positive_int
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
import argparse
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

    # computing distributions
    dists = []
    for loader in data_loaders:

        point_distribution = th.zeros((1, y_size, x_size))

        dataiter = loader.__iter__()
        grids = dataiter.__next__()

        # computing the distribution
        for i in range(dset_size):
            point_distribution += grids[i]

        maxval = th.max(point_distribution)
        point_distribution /= maxval

        dists.append(point_distribution)

    # computing differences
    diffs = []
    dist_prec = dists[0]
    for dist in dists[1:]:

        diffs.append(th.norm(dist - dist_prec, 2))

        dist_prec = dist

    # plotting differences
    model_ids = list(range(num_models))
    plt.plot(model_ids, diffs, marker='o', linestyle='-', color='b', label="datasets' differenc")
    plt.xticks(model_ids)
    plt.xlabel('model')
    plt.ylabel('difference')
    plt.legend()
    plt.savefig(dset_dir + '/difference.png')
