from chain_lib import positive_int
import argparse
import gzip
import numpy as np


# default parameters
dset_size_dflt = 10000
y_size_dflt = 200
x_size_dflt = 200
y_dist_dflt = 'binomial'
x_dist_dflt = 'binomial'


if __name__ == "__main__":

    # getting parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_size', type=positive_int, default=dset_size_dflt)
    parser.add_argument('--y_size', type=positive_int, default=y_size_dflt)
    parser.add_argument('--x_size', type=positive_int, default=x_size_dflt)
    parser.add_argument('--y_dist', choices=['binomial', 'uniform'], default=y_dist_dflt)
    parser.add_argument('--x_dist', choices=['binomial', 'uniform'], default=x_dist_dflt)

    args = parser.parse_args()

    dset_size = args.dset_size
    y_size = args.y_size
    x_size = args.x_size
    y_dist = args.y_dist
    x_dist = args.x_dist

    # initializing the dataset
    grids = np.zeros((dset_size, y_size, x_size, 1), dtype=np.int8)

    y_coord = np.empty(dset_size)
    if y_dist == 'binomial':
        success_prob = 0.9
        y_coord = np.random.binomial(y_size-1, success_prob, dset_size)
    elif y_dist == 'uniform':
        y_coord = np.random.randint(0, y_size, dset_size)

    x_coord = np.empty(dset_size)
    if x_dist == 'binomial':
        success_prob = 0.5
        x_coord = np.random.binomial(x_size-1, success_prob, dset_size)
    elif x_dist == 'uniform':
        x_coord = np.random.randint(0, x_size, dset_size)

    # randomly turning on one pixel
    for i in range(grids.shape[0]):
        grids[i][y_coord[i]][x_coord[i]][0] = 1

    # loading dataset to file
    grids_byte = grids.copy().tobytes()
    with gzip.open('../data/original_dataset-ubyte.gz', 'wb') as f:
        f.write(grids_byte)
